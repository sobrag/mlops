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

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_dir.name,
        job_type="training",
        tags=["mlops", "training"],
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

        if wb_run is not None:
            wb_run.summary["dataset_path"] = str(dataset_path)
            if getattr(cfg, "dataset_artifact", None):
                wb_run.summary["dataset_artifact"] = getattr(cfg, "dataset_artifact")

        df, metadata = preprocess_pipeline(df_raw)
        preprocessing_meta_path = run_dir / "preprocessing_metadata.json"
        save_json(metadata, preprocessing_meta_path)

        X = df[cfg.clean_col].values
        y = df[cfg.label_col].values

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=cfg.test_size,
            random_state=cfg.seed,
            stratify=y,
        )

        vectorizer = build_vectorizer(
            max_features=cfg.tfidf_max_features,
            ngram_range=cfg.tfidf_ngram_range,
            stop_words=cfg.tfidf_stop_words,
        )
        Xtr = vectorizer.fit_transform(X_train)
        Xte = vectorizer.transform(X_test)

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
        metrics = evaluate_predictions(y_test, p_real, cfg=eval_cfg)

        if wb_run is not None:
            wandb.log({f"metrics/{k}": v for k, v in metrics.items()})

        vectorizer_path = run_dir / "vectorizer.joblib"
        model_path = run_dir / "model.joblib"
        metrics_path = run_dir / "metrics.json"
        run_config_path = run_dir / "run_config.json"

        # Fixed: use save_joblib instead of vectorizer.save()
        save_joblib(vectorizer, vectorizer_path)
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
    logger.info(f"Loading config from: {args.config}")
    cfg = Config.from_yaml(args.config)
    
    # Create run directory
    if args.run_name:
        run_name = args.run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
    
    run_dir = cfg.artifacts_dir / run_name
    logger.info(f"Run directory: {run_dir}")
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Dataset path: {cfg.raw_data_path}")
    logger.info(f"Dataset artifact: {cfg.dataset_artifact}")
    logger.info(f"Use W&B: {cfg.use_wandb}")
    logger.info(f"W&B mode: {cfg.wandb_mode}")
    logger.info(f"Max rows: {args.max_rows or 'all'}")
    logger.info(f"Seed: {cfg.seed}")
    logger.info(f"Test size: {cfg.test_size}")
    logger.info("=" * 60)
    
    # Run training
    try:
        logger.info("Starting training pipeline...")
        metrics = run_training(cfg, run_dir, max_rows=args.max_rows)
        
        # Print results
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("Final metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        logger.info(f"\nArtifacts saved to: {run_dir}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error("TRAINING FAILED!")
        logger.error("=" * 60)
        logger.error(f"Error: {e}", exc_info=True)
        raise