from pathlib import Path

from src.config import Config
from pipelines.train_pipeline import run_training
from src.utils.io import load_joblib, load_json


def test_train_save_load_predict_smoke(tmp_path: Path):
    """
    End-to-end smoke test: train, save artifacts, reload them, verify metrics.
    Uses the committed sample dataset for CI compatibility.
    """
    # Load config pointing to sample dataset
    cfg = Config.from_yaml("configs/test.yml")
    
    # Override output paths to use tmp_path (so tests don't pollute repo)
    from dataclasses import replace
    cfg = replace(
        cfg,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
    )

    run_dir = tmp_path / "artifacts" / "test_run"
    
    # Run training with limited rows for speed
    metrics = run_training(cfg, run_dir, max_rows=100)

    # 1) Check all expected artifacts exist
    assert run_dir.exists(), f"Run directory not created: {run_dir}"
    assert (run_dir / "vectorizer.joblib").exists(), "Vectorizer not saved"
    assert (run_dir / "model.joblib").exists(), "Model not saved"
    assert (run_dir / "metrics.json").exists(), "Metrics not saved"
    assert (run_dir / "run_config.json").exists(), "Run config not saved"
    assert (run_dir / "preprocessing_metadata.json").exists(), "Preprocessing metadata not saved"

    # 2) Verify artifacts can be reloaded
    vectorizer = load_joblib(run_dir / "vectorizer.joblib")
    model = load_joblib(run_dir / "model.joblib")
    saved_metrics = load_json(run_dir / "metrics.json")
    
    assert vectorizer is not None, "Vectorizer failed to load"
    assert model is not None, "Model failed to load"
    assert isinstance(saved_metrics, dict), "Metrics should be a dict"

    # 3) Validate metrics structure and values
    assert isinstance(metrics, dict), "Returned metrics should be a dict"
    assert len(metrics) > 0, "Metrics dict should not be empty"
    
    # Check specific metric keys (adjust based on your actual metrics)
    assert "accuracy" in metrics, "Accuracy metric missing"
    assert 0 <= metrics["accuracy"] <= 1, f"Invalid accuracy: {metrics['accuracy']}"
    
    # Optional: check other common metrics
    expected_keys = ["accuracy", "precision", "recall", "f1"]
    for key in expected_keys:
        if key in metrics:
            assert 0 <= metrics[key] <= 1, f"Invalid {key}: {metrics[key]}"

    print(f"Test passed! Metrics: {metrics}")