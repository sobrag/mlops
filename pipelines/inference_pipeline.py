"""Inference pipeline for making predictions on new data."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import Config
from src.data.load_data import load_csv
from src.data.preprocess import clean_text, preprocess_pipeline
from src.features.vectorize import TextVectorizer
from src.models.evaluate import EvalConfig, evaluate_predictions
from src.models.predict import predict_all
from src.utils.io import ensure_dir, load_joblib, save_json


logger = logging.getLogger(__name__)


def _latest_run_dir(artifacts_dir: Path) -> Path:
    """Pick the most recent artifacts run folder."""
    runs = sorted([d for d in artifacts_dir.iterdir() if d.is_dir()], reverse=True)
    if not runs:
        raise FileNotFoundError(f"No artifact runs found in {artifacts_dir}")
    return runs[0]


def _load_artifacts(run_dir: Path):
    """Load vectorizer and model from a run directory."""
    vec_path = run_dir / "vectorizer.joblib"
    model_path = run_dir / "model.joblib"

    if not vec_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"Artifacts not found in {run_dir}. "
            f"Expected {vec_path.name} and {model_path.name}"
        )

    vectorizer = TextVectorizer.load(vec_path)
    model = load_joblib(model_path)
    return vectorizer, model


def _preprocess_for_inference(df_raw: pd.DataFrame, cfg: Config):
    """
    Reuse the training preprocessing when labels are present; otherwise apply a light clean.
    """
    has_labels = cfg.label_col in df_raw.columns
    has_required = all(col in df_raw.columns for col in [cfg.title_col, cfg.text_col])

    if has_labels and has_required:
        df_prepared, metadata = preprocess_pipeline(df_raw)
        metadata = dict(metadata)
        metadata["has_labels"] = True
        return df_prepared, metadata

    if not has_required:
        missing = [c for c in [cfg.title_col, cfg.text_col] if c not in df_raw.columns]
        raise ValueError(f"Missing required columns for inference: {missing}")

    # Copy only needed columns
    df = df_raw[[cfg.title_col, cfg.text_col]].copy()
    
    # Clean NaN and strip
    df[cfg.title_col] = df[cfg.title_col].fillna("").astype(str).str.strip()
    df[cfg.text_col] = df[cfg.text_col].fillna("").astype(str).str.strip()
    
    # Create combined column
    df[cfg.combined_col] = df[cfg.title_col] + ". " + df[cfg.text_col]
    
    # Apply cleaning
    df[cfg.clean_col] = df[cfg.combined_col].apply(clean_text)

    metadata = {
        "total_rows": len(df),
        "columns": df.columns.tolist(),
        "has_labels": False,
    }
    return df, metadata


def run_inference(
    cfg: Config,
    input_path: Path,
    output_dir: Path,
    run_dir: Optional[Path] = None,
    *,
    max_rows: Optional[int] = None,
    threshold: float = 0.5,
    score_scale: float = 100.0,
) -> dict:
    """Load artifacts, preprocess data, run predictions, and persist outputs/metrics."""
    ensure_dir(output_dir)

    # Load artifacts
    run_dir = run_dir or _latest_run_dir(cfg.artifacts_dir)
    logger.info(f"Using artifacts from: {run_dir}")  
    vectorizer, model = _load_artifacts(run_dir)
    logger.info("Model and vectorizer loaded successfully")

    # Load input
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            f"Please provide a valid CSV file for inference."
        )
    
    logger.info(f"Loading dataset from: {input_path}")
    df_raw = load_csv(input_path)
    logger.info(f"Successfully loaded {len(df_raw)} rows from {input_path.name}")

    if max_rows is not None:
        df_raw = df_raw.head(max_rows).copy()
        logger.info(f"Limited to {max_rows} rows for inference")

    # Preprocess
    logger.info("Starting preprocessing...")
    df, metadata = _preprocess_for_inference(df_raw, cfg)
    logger.info(f"Preprocessing complete: {len(df)} rows ready for inference")
    
    if not metadata.get("has_labels", False):
        logger.info("No labels detected - running prediction-only mode")
    
    # Vectorize
    logger.info("Vectorizing text...")
    X_vec = vectorizer.transform(df[cfg.clean_col].values)
    logger.info(f"Vectorization complete: {X_vec.shape} feature matrix")

    # Predict
    logger.info("Running predictions...")
    proba, scores, labels = predict_all(
        model, X_vec, threshold=threshold, scale=score_scale
    )
    
    # Validate
    expected_rows = len(df)
    if len(proba) != expected_rows:
        raise ValueError(
            f"Prediction mismatch: {len(proba)} predictions for {expected_rows} rows"
        )
    
    logger.info(f"Predictions complete: {len(labels)} samples processed")

    # Create output
    df_out = df.copy()
    df_out["probability_real"] = proba
    df_out["credibility_score"] = scores
    df_out["label_pred"] = labels

    # Save
    preds_path_csv = output_dir / "predictions.csv"
    preds_path_parquet = output_dir / "predictions.parquet"
    df_out.to_csv(preds_path_csv, index=False)
    try:
        df_out.to_parquet(preds_path_parquet, index=False)
        parquet_saved = True
    except ImportError:
        parquet_saved = False
        logger.warning(
            "Parquet export skipped: install `pyarrow` or `fastparquet` to enable it."
        )
    
    logger.info("Predictions saved:")
    logger.info(f"   CSV: {preds_path_csv}")
    if parquet_saved:
        logger.info(f"   Parquet: {preds_path_parquet}")
    else:
        logger.info("   Parquet: skipped (missing pyarrow/fastparquet)")

    # Summary
    results = {
        "run_dir": str(run_dir),
        "input": str(input_path),
        "output_dir": str(output_dir),
        "metadata": metadata,
        "threshold": threshold,
        "score_scale": score_scale,
        "total_predictions": len(df_out),
        "predicted_real": int((df_out["label_pred"] == 1).sum()),
        "predicted_fake": int((df_out["label_pred"] == 0).sum()),
        "avg_credibility": float(df_out["credibility_score"].mean()),
        "low_credibility_count": int((df_out["credibility_score"] < 30).sum()),
        "parquet_saved": parquet_saved,
    }

    # Evaluate if labels present
    if cfg.label_col in df_out.columns:
        logger.info("Evaluating predictions against labels...")
        eval_cfg = EvalConfig(threshold=threshold)
        metrics = evaluate_predictions(df_out[cfg.label_col], proba, cfg=eval_cfg)
        metrics_path = output_dir / "metrics.json"
        save_json(metrics, metrics_path)
        results["metrics"] = metrics
        results["metrics_path"] = str(metrics_path)
        logger.info(f"Evaluation metrics saved to {metrics_path}")

    # Save summary
    summary_path = output_dir / "inference_summary.json"
    save_json(results, summary_path)
    logger.info(f"Summary saved to {summary_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    from datetime import datetime
    import time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Run inference with saved artifacts")
    parser.add_argument("--config", type=str, default="configs/train.yml")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file for inference")
    parser.add_argument("--run-dir", type=str, default=None, help="Artifacts run directory (default: latest)")
    parser.add_argument("--output-dir", type=str, default=None, help="Where to write predictions")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit rows for testing")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--score-scale", type=float, default=100.0, help="Credibility score scale")
    args = parser.parse_args()

    # Load config
    logger.info(f"Loading config from: {args.config}")
    cfg = Config.from_yaml(args.config)

    # Determine run directories
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = _latest_run_dir(cfg.artifacts_dir)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = cfg.results_dir / "inference" / f"run_{ts}"

    logger.info("=" * 60)
    logger.info("INFERENCE CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Model directory: {run_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Max rows: {args.max_rows or 'all'}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Score scale: {args.score_scale}")
    logger.info("=" * 60)

    
    start_time = time.time()
    
    try:
        logger.info("Starting inference pipeline...")
        
        results = run_inference(
            cfg=cfg,
            input_path=Path(args.input),
            output_dir=output_dir,
            run_dir=run_dir,
            max_rows=args.max_rows,
            threshold=args.threshold,
            score_scale=args.score_scale,
        )
        
        elapsed = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("INFERENCE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Elapsed time: {elapsed:.2f}s")
        logger.info(f"Total predictions: {results['total_predictions']}")
        logger.info(f"Predicted REAL: {results['predicted_real']} ({results['predicted_real']/results['total_predictions']:.1%})")
        logger.info(f"Predicted FAKE: {results['predicted_fake']} ({results['predicted_fake']/results['total_predictions']:.1%})")
        logger.info(f"Avg credibility: {results['avg_credibility']:.1f}/{args.score_scale}")
        
        if "metrics" in results:
            logger.info("\nEvaluation Metrics:")
            for key, value in results["metrics"].items():
                logger.info(f"  {key}: {value:.4f}")
        
        logger.info(f"\nPredictions saved to: {output_dir}")
        logger.info("=" * 60)
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("=" * 60)
        logger.error("INFERENCE FAILED")
        logger.error("=" * 60)
        logger.error(f"Elapsed time: {elapsed:.2f}s")
        logger.error(f"Error: {e}", exc_info=True)
        logger.error("=" * 60)
        raise
