from __future__ import annotations

import pandas as pd
import json
import joblib
import wandb
from sklearn.model_selection import train_test_split
from pathlib import Path
from src.config import Config
from src.data.preprocess import preprocess_pipeline

def log_error_analysis(run_id: str, run_dir_path: str, config_path: str = "configs/train.yml", max_examples: int = 20):
    """
    Log FP/FN examples to an existing W&B run.

    Args:
        run_id: W&B run id of the existing training run
        run_dir_path: path to the artifacts directory of that run
        config_path: path to the YAML config file
        max_examples: max number of FP/FN examples to log
    """
    # Load config
    cfg = Config.from_yaml(config_path)
    run_dir = Path(run_dir_path)  # ensure Path-like

    # Load trained model and vectorizer
    model = joblib.load(f"{run_dir}/model.joblib")
    vectorizer = joblib.load(f"{run_dir}/vectorizer.joblib")

    # Load dataset
    df_raw = pd.read_csv(cfg.raw_data_path)    
    df, _ = preprocess_pipeline(df_raw)            
    X = df[cfg.clean_col]                           
    y = df[cfg.label_col]

    # Train/test split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=y
    )

    # Transform and predict
    X_test_vect = vectorizer.transform(X_test)
    y_pred_prob = model.predict_proba(X_test_vect)[:,1]
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Build FP/FN tables
    df_test = pd.DataFrame({
        "text": X_test,
        "true_label": y_test,
        "pred_label": y_pred,
        "pred_prob": y_pred_prob
    })
    fp = df_test[(df_test.true_label==0) & (df_test.pred_label==1)]
    fn = df_test[(df_test.true_label==1) & (df_test.pred_label==0)]

    # Resume W&B run
    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        id=run_id,
        resume="allow"
    )

    # Log FP/FN counts + sample examples
    run.log({
        "num_fp": len(fp),
        "num_fn": len(fn),
        "fp_examples": wandb.Table(dataframe=fp.head(max_examples)),
        "fn_examples": wandb.Table(dataframe=fn.head(max_examples))
    })
    
    # Update existing METRICS.JSON
    metrics_path = run_dir / "metrics.json"

    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            existing_metrics = json.load(f)
        
        existing_metrics["num_fp"] = len(fp)
        existing_metrics["num_fn"] = len(fn)
        
        with open(metrics_path, "w") as f:
            json.dump(existing_metrics, f, indent=4)
        print(f"metrics.json aggiornato in locale!")
    else:
        print(f"Attenzione: metrics.json non trovato in {run_dir}")

    run.finish()
    print(f"Logged FP/FN to W&B run {run_id}. Num FP={len(fp)}, Num FN={len(fn)}")


# Example usage:
# python -m src.models.error_analysis_wandb --run_id baseline_lr_20260131_153000 --run_dir artifacts/run_baseline_lr
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Log FP/FN examples to an existing W&B run")
    parser.add_argument("--run_id", type=str, required=True, help="Existing W&B run ID")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to artifacts directory of the run")
    parser.add_argument("--config", type=str, default="configs/train.yml", help="Path to config YAML")
    parser.add_argument("--max_examples", type=int, default=20, help="Max FP/FN examples to log")
    args = parser.parse_args()

    log_error_analysis(
        run_id=args.run_id,
        run_dir_path=args.run_dir,
        config_path=args.config,
        max_examples=args.max_examples
    )
