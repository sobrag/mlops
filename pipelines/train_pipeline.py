from __future__ import annotations

from dataclasses import asdict
from sklearn.model_selection import train_test_split

from src.config import Config
from src.utils.io import ensure_dir, save_joblib, save_json
from src.data.load_data import load_csv
from src.data.preprocess import preprocess_pipeline
from src.features.vectorize import build_vectorizer
from src.models.train import train_model, TrainConfig
from src.models.calibrate import calibrate_model, CalibrateConfig
from src.models.predict import predict_proba_real
from src.models.evaluate import evaluate_predictions, EvalConfig


def main() -> None:
    cfg = Config()
    ensure_dir(cfg.results_dir)

    # 1) Load raw data
    df_raw = load_csv(cfg.raw_data_path)

    # 2) Preprocess data
    # Returns: cleaned dataframe + metadata (for reproducibility/debugging)
    df, metadata = preprocess_pipeline(df_raw)
    save_json(metadata, cfg.results_dir / "preprocessing_metadata.json")

    X = df[cfg.clean_col].values
    y = df[cfg.label_col].values

    # 3) Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=y,
    )

    # 4) Vectorization
    vectorizer = build_vectorizer(
        max_features=cfg.tfidf_max_features,
        ngram_range=cfg.tfidf_ngram_range,
        stop_words=cfg.tfidf_stop_words,
    )
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    # 5) Model training (baseline: Logistic Regression)
    train_cfg = TrainConfig(
        lr_solver=cfg.lr_solver,
        lr_max_iter=cfg.lr_max_iter,
        random_state=cfg.seed,
    )
    base_model = train_model("logreg", Xtr, y_train, cfg=train_cfg)

    # 6) Probability calibration (credibility score)
    calib_cfg = CalibrateConfig(method="isotonic", cv=5)
    model = calibrate_model(base_model, Xtr, y_train, cfg=calib_cfg)

    # 7) Evaluation
    p_real = predict_proba_real(model, Xte)
    eval_cfg = EvalConfig(threshold=0.5)
    metrics = evaluate_predictions(y_test, p_real, cfg=eval_cfg)

    # 8) Save artifacts
    vectorizer.save(cfg.vectorizer_path)
    save_joblib(model, cfg.model_path)
    save_json(metrics, cfg.metrics_path)

    # Save full run configuration for reproducibility
    run_cfg = {
        "config": {k: str(v) for k, v in asdict(cfg).items()},
        "train_config": asdict(train_cfg),
        "calibrate_config": asdict(calib_cfg),
        "eval_config": asdict(eval_cfg),
    }
    save_json(run_cfg, cfg.run_config_path)

    print("Training complete.")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
