from __future__ import annotations
from dataclasses import asdict
from sklearn.model_selection import train_test_split
from datetime import datetime
import subprocess
from pathlib import Path

from src.config import Config
from src.utils.io import ensure_dir, save_joblib, save_json
from src.data.load_data import load_csv
from src.data.preprocess import preprocess_pipeline
from src.features.vectorize import build_vectorizer
from src.models.train import train_model, TrainConfig
from src.models.calibrate import calibrate_model, CalibrateConfig
from src.models.predict import predict_proba_real
from src.models.evaluate import evaluate_predictions, EvalConfig


def run_training(cfg: Config, run_dir: Path, *, max_rows: int | None = None) -> dict:
    """
    Train + calibrate + evaluate the model and persist a full, versioned artifact bundle into run_dir.
    Neptune logging is optional and controlled by cfg.use_neptune.
    """
    ensure_dir(run_dir)

    nep_run = None
    try:
        # -------------------------
        # Optional: Neptune tracking
        # -------------------------
        if cfg.use_neptune:
            import os
            import neptune

            project = cfg.neptune_project or os.getenv("NEPTUNE_PROJECT")

            nep_run = neptune.init_run(
                project=project,  # can be None if NEPTUNE_PROJECT is set
                name=run_dir.name,
                mode=getattr(cfg, "neptune_mode", "async"),
                tags=["mlops", "training"],
            )

            nep_run["run/id"] = run_dir.name
            nep_run["run/dir"] = str(run_dir)
            nep_run["parameters"] = {
                "seed": cfg.seed,
                "test_size": cfg.test_size,
                "tfidf_max_features": cfg.tfidf_max_features,
                "tfidf_ngram_range": f"{cfg.tfidf_ngram_range[0]}-{cfg.tfidf_ngram_range[1]}",
                "tfidf_stop_words": cfg.tfidf_stop_words,
                "lr_solver": cfg.lr_solver,
                "lr_max_iter": cfg.lr_max_iter,
            }

        # -------------------------
        # 1) Load raw data
        # -------------------------
        df_raw = load_csv(cfg.raw_data_path)
        if max_rows is not None:
            df_raw = df_raw.head(max_rows).copy()

        # -------------------------
        # 2) Preprocess
        # -------------------------
        df, metadata = preprocess_pipeline(df_raw)
        save_json(metadata, run_dir / "preprocessing_metadata.json")

        X = df[cfg.clean_col].values
        y = df[cfg.label_col].values

        # -------------------------
        # 3) Split
        # -------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=cfg.test_size,
            random_state=cfg.seed,
            stratify=y,
        )

        # -------------------------
        # 4) Vectorize
        # -------------------------
        vectorizer = build_vectorizer(
            max_features=cfg.tfidf_max_features,
            ngram_range=cfg.tfidf_ngram_range,
            stop_words=cfg.tfidf_stop_words,
        )
        Xtr = vectorizer.fit_transform(X_train)
        Xte = vectorizer.transform(X_test)

        # -------------------------
        # 5) Train
        # -------------------------
        train_cfg = TrainConfig(
            lr_solver=cfg.lr_solver,
            lr_max_iter=cfg.lr_max_iter,
            random_state=cfg.seed,
        )
        base_model = train_model("logreg", Xtr, y_train, cfg=train_cfg)

        # -------------------------
        # 6) Calibrate
        # -------------------------
        calib_cfg = CalibrateConfig(method="isotonic", cv=5)
        model = calibrate_model(base_model, Xtr, y_train, cfg=calib_cfg)

        # -------------------------
        # 7) Evaluate
        # -------------------------
        p_real = predict_proba_real(model, Xte)
        eval_cfg = EvalConfig(threshold=0.5)
        metrics = evaluate_predictions(y_test, p_real, cfg=eval_cfg)

        if nep_run is not None:
            nep_run["metrics"] = metrics

        # -------------------------
        # 8) Save artifacts in run_dir
        # -------------------------
        vectorizer_path = run_dir / "vectorizer.joblib"
        model_path = run_dir / "model.joblib"
        metrics_path = run_dir / "metrics.json"
        run_config_path = run_dir / "run_config.json"

        vectorizer.save(vectorizer_path)
        save_joblib(model, model_path)
        save_json(metrics, metrics_path)

        run_cfg = {
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "config": {k: str(v) for k, v in asdict(cfg).items()},
            "train_config": asdict(train_cfg),
            "calibrate_config": asdict(calib_cfg),
            "eval_config": asdict(eval_cfg),
            "artifacts": {
                "vectorizer": str(vectorizer_path),
                "model": str(model_path),
                "metrics": str(metrics_path),
                "run_config": str(run_config_path),
                "preprocessing_metadata": str(run_dir / "preprocessing_metadata.json"),
            },
        }
        save_json(run_cfg, run_config_path)

        if nep_run is not None:
            nep_run["artifacts/model"].track_files(str(model_path))
            nep_run["artifacts/vectorizer"].track_files(str(vectorizer_path))
            nep_run["artifacts/metrics"].track_files(str(metrics_path))
            nep_run["artifacts/run_config"].track_files(str(run_config_path))
            nep_run["artifacts/preprocessing_metadata"].track_files(
                str(run_dir / "preprocessing_metadata.json")
            )

        return metrics

    finally:
        if nep_run is not None:
            nep_run.stop()



def get_git_sha_short() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        return out
    except Exception:
        return "nogit"


def make_run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{get_git_sha_short()}"


def main() -> None:
    cfg = Config.from_yaml("configs/train.yml")

    # Ensure base dirs
    ensure_dir(cfg.results_dir)
    ensure_dir(cfg.artifacts_dir)

    # Create versioned run directory
    run_id = make_run_id()
    run_dir = cfg.artifacts_dir / run_id

    metrics = run_training(cfg, run_dir)

    print("Training complete.")
    print("Run ID:", run_id)
    print("Artifacts saved in:", run_dir)
    print("Metrics:", metrics)



if __name__ == "__main__":
    main()
