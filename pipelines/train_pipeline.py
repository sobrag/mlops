from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

import wandb
from sklearn.model_selection import train_test_split

from src.config import Config
from src.data.load_data import load_csv
from src.data.preprocess import preprocess_pipeline
from src.features.vectorize import build_vectorizer
from src.models.calibrate import CalibrateConfig, calibrate_model
from src.models.evaluate import EvalConfig, evaluate_predictions
from src.models.predict import predict_proba_real
from src.models.train import TrainConfig, train_model
from src.utils.io import ensure_dir, save_joblib, save_json


def get_git_sha_short() -> str:
    """Get short Git SHA of current commit, or 'unknown' if not in a git repo."""
    try:
        import subprocess
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return sha
    except Exception:
        return "unknown"


def _resolve_dataset_path(cfg, wb_run=None) -> Path:
    p = Path(cfg.raw_data_path)

    if p.exists():
        return p

    use_wandb = bool(getattr(cfg, "use_wandb", False))
    dataset_artifact = getattr(cfg, "dataset_artifact", None)

    if use_wandb and wb_run is not None and dataset_artifact:
        art = wb_run.use_artifact(dataset_artifact)
        d = Path(art.download())
        csvs = list(d.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(
                f"Dataset artifact '{dataset_artifact}' downloaded to '{d}', but no CSV found."
            )
        return csvs[0]

    raise FileNotFoundError(
        f"Dataset not found at '{p}'. "
        f"Either provide the file locally or set cfg.dataset_artifact and enable W&B."
    )


def _wandb_init_if_enabled(cfg, run_dir: Path):
    if not bool(getattr(cfg, "use_wandb", False)):
        return None

    project = getattr(cfg, "wandb_project", None) or os.getenv("WANDB_PROJECT") or "mlops"
    entity = getattr(cfg, "wandb_entity", None) or os.getenv("WANDB_ENTITY") or None
    # Environment variable takes precedence over config
    mode = os.getenv("WANDB_MODE") or getattr(cfg, "wandb_mode", None) or "online"

    split_cfg = getattr(cfg, "split", None)
    split_tag = None
    if split_cfg and all(k in split_cfg for k in ("train_frac", "val_frac", "tail_frac")):
        split_tag = f"split_{split_cfg['train_frac']}_{split_cfg['val_frac']}_{split_cfg['tail_frac']}"
    dataset_tag = None
    if getattr(cfg, "dataset_artifact", None):
        dataset_tag = f"dataset:{getattr(cfg, 'dataset_artifact').split(':')[0]}"
    else:
        dataset_tag = f"dataset:{Path(cfg.raw_data_path).stem}"

    tags = ["training"]
    if split_tag:
        tags.append(split_tag)
    if dataset_tag:
        tags.append(dataset_tag)

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_dir.name,
        job_type="training",
        tags=tags,
        mode=mode,
    )

    try:
        cfg_dict = asdict(cfg)
    except Exception:
        cfg_dict = {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}

    safe_cfg = {}
    for k, v in cfg_dict.items():
        if isinstance(v, Path):
            safe_cfg[k] = str(v)
        else:
            safe_cfg[k] = v

    run.config.update(safe_cfg, allow_val_change=True)
    run.summary["run_id"] = run_dir.name
    run.summary["run_dir"] = str(run_dir)

    return run


def run_training(cfg: Config, run_dir: Path, *, max_rows: int | None = None) -> dict:
    """
    Train + calibrate + evaluate the model and persist a full, versioned artifact bundle into run_dir.
    W&B logging is optional and controlled by cfg.use_wandb.
    """
    ensure_dir(run_dir)

    wb_run = None
    try:
        wb_run = _wandb_init_if_enabled(cfg, run_dir)

        dataset_path = _resolve_dataset_path(cfg, wb_run=wb_run)
        df_raw = load_csv(dataset_path)
        if max_rows is not None:
            df_raw = df_raw.head(max_rows).copy()

        # Split config (train/val/tail)
        split_cfg = getattr(cfg, "split", None)

        # Split: train/val/tail (tail reserved for drift checks, not used here)
        split_cfg = getattr(cfg, "split", None)
        if split_cfg and all(k in split_cfg for k in ("train_frac", "val_frac", "tail_frac")):
            n = len(df_raw)
            train_end = int(n * split_cfg["train_frac"])
            val_end = int(n * (split_cfg["train_frac"] + split_cfg["val_frac"]))
            df_train_raw = df_raw.iloc[:train_end]
            df_val_raw = df_raw.iloc[train_end:val_end]
            df_tail_raw = df_raw.iloc[val_end:]
        else:
            df_train_raw = df_raw
            df_val_raw = None
            df_tail_raw = None

        if wb_run is not None:
            wb_run.summary["dataset_path"] = str(dataset_path)
            if getattr(cfg, "dataset_artifact", None):
                wb_run.summary["dataset_artifact"] = getattr(cfg, "dataset_artifact")
            wb_run.summary["split"] = {
                "train_frac": split_cfg.get("train_frac") if split_cfg else None,
                "val_frac": split_cfg.get("val_frac") if split_cfg else None,
                "tail_frac": split_cfg.get("tail_frac") if split_cfg else None,
                "train_rows": len(df_train_raw),
                "val_rows": len(df_val_raw) if df_val_raw is not None else 0,
                "tail_rows": len(df_tail_raw) if df_tail_raw is not None else 0,
            }

        df_train, metadata_train = preprocess_pipeline(df_train_raw)
        if df_val_raw is not None and len(df_val_raw) > 0:
            df_val, metadata_val = preprocess_pipeline(df_val_raw)
        else:
            df_val, metadata_val = None, None

        # For reproducibility store train metadata only
        metadata = metadata_train
        preprocessing_meta_path = run_dir / "preprocessing_metadata.json"
        save_json(metadata, preprocessing_meta_path)

        # Force numpy arrays to avoid pandas Arrow dtypes with pyarrow backend
        X_train = df_train[cfg.clean_col].astype(str).to_numpy()
        y_train = df_train[cfg.label_col].astype(int).to_numpy()

        if df_val is not None:
            X_val = df_val[cfg.clean_col].astype(str).to_numpy()
            y_val = df_val[cfg.label_col].astype(int).to_numpy()
        else:
            # fallback to random split like before
            X_all = df_train[cfg.clean_col].astype(str).to_numpy()
            y_all = df_train[cfg.label_col].astype(int).to_numpy()
            X_train, X_val, y_train, y_val = train_test_split(
                X_all,
                y_all,
                test_size=cfg.test_size,
                random_state=cfg.seed,
                stratify=y_all,
            )

        vectorizer = build_vectorizer(
            max_features=cfg.tfidf_max_features,
            ngram_range=cfg.tfidf_ngram_range,
            stop_words=cfg.tfidf_stop_words,
        )
        Xtr = vectorizer.fit_transform(X_train)
        Xte = vectorizer.transform(X_val)

        train_cfg = TrainConfig(
            lr_solver=cfg.lr_solver,
            lr_max_iter=cfg.lr_max_iter,
            random_state=cfg.seed,
        )
        base_model = train_model(cfg.model_name, Xtr, y_train, cfg=train_cfg)

        calib_cfg = CalibrateConfig(method="isotonic", cv=5)
        model = calibrate_model(base_model, Xtr, y_train, cfg=calib_cfg)

        p_real = predict_proba_real(model, Xte)
        eval_cfg = EvalConfig(threshold=0.5)
        metrics = evaluate_predictions(y_val, p_real, cfg=eval_cfg)

        if wb_run is not None:
            wandb.log({f"metrics/{k}": v for k, v in metrics.items()})

        vectorizer_path = run_dir / "vectorizer.joblib"
        model_path = run_dir / "model.joblib"
        metrics_path = run_dir / "metrics.json"
        run_config_path = run_dir / "run_config.json"

        # Persist vectorizer with its native save (includes config payload)
        vectorizer.save(vectorizer_path)
        save_joblib(model, model_path)
        save_json(metrics, metrics_path)

        run_cfg = {
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "dataset_path": str(dataset_path),
            "dataset_artifact": getattr(cfg, "dataset_artifact", None),
            "model_name": cfg.model_name,
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
            "train_config": asdict(train_cfg),
            "calibrate_config": asdict(calib_cfg),
            "eval_config": asdict(eval_cfg),
            "artifacts": {
                "vectorizer": str(vectorizer_path),
                "model": str(model_path),
                "metrics": str(metrics_path),
                "run_config": str(run_config_path),
                "preprocessing_metadata": str(preprocessing_meta_path),
            },
        }
        save_json(run_cfg, run_config_path)

        if wb_run is not None:
            art = wandb.Artifact(
                name="credibility_model_bundle",
                type="model",
                metadata={
                    "run_id": run_dir.name,
                    "git_sha": get_git_sha_short(),
                    "dataset_artifact": getattr(cfg, "dataset_artifact", None),
                    "metrics": metrics,
                },
            )
            art.add_file(str(model_path))
            art.add_file(str(vectorizer_path))
            art.add_file(str(metrics_path))
            art.add_file(str(run_config_path))
            art.add_file(str(preprocessing_meta_path))

            wb_run.log_artifact(art, aliases=["latest", run_dir.name])

        return metrics

    finally:
        if wb_run is not None:
            wb_run.finish()



if __name__ == "__main__":
    import argparse
    import logging
    from datetime import datetime
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train fake news detection model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yml",
        help="Path to config YAML file (default: configs/train.yml)"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit dataset to N rows for quick testing"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom run name (default: auto-generated timestamp)"
    )
    args = parser.parse_args()
    
    # Load config
    logger.info(f"Config: {args.config}")
    cfg = Config.from_yaml(args.config)
    
    # Create run directory
    if args.run_name:
        run_name = args.run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
    
    run_dir = cfg.artifacts_dir / run_name
    logger.info(f"Run dir: {run_dir}")
    split_cfg = getattr(cfg, "split", None)
    split_msg = (
        f"split train/val/tail={split_cfg['train_frac']}/{split_cfg['val_frac']}/{split_cfg['tail_frac']}"
        if split_cfg
        else f"random split test_size={cfg.test_size}"
    )
    logger.info(
        f"Dataset={cfg.raw_data_path}, artifact={cfg.dataset_artifact}, max_rows={args.max_rows or 'all'}, "
        f"W&B={cfg.use_wandb} ({cfg.wandb_mode}), {split_msg}"
    )
    
    # Run training
    try:
        logger.info("Training start")
        metrics = run_training(cfg, run_dir, max_rows=args.max_rows)
        metric_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info(f"Training done | {metric_str} | artifacts={run_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
