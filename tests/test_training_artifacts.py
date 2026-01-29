from pathlib import Path

from src.config import Config
from pipelines.train_pipeline import run_training
from src.utils.io import load_joblib, load_json


def test_train_save_load_predict_smoke(tmp_path: Path):
    # Load real config (YAML), but write artifacts to tmp_path
    cfg = Config.from_yaml("configs/train.yml")

    run_dir = tmp_path / "run"
    metrics = run_training(cfg, run_dir, max_rows=300)

    # 1) Artifacts exist
    assert (run_dir / "vectorizer.joblib").exists()
    assert (run_dir / "model.joblib").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "run_config.json").exists()
    assert (run_dir / "preprocessing_metadata.json").exists()

    # 2) Artifacts reload
    load_joblib(run_dir / "vectorizer.joblib")
    load_joblib(run_dir / "model.joblib")
    m = load_json(run_dir / "metrics.json")

    assert isinstance(m, dict)
    assert isinstance(metrics, dict)

    # 3) Sanity check
    assert len(m) > 0
