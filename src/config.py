from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class Config:
    """
    Central configuration for the project.
    Defines paths and key hyperparameters for reproducibility.
    """

    # -------------------------
    # Reproducibility
    # -------------------------
    seed: int = 42
    test_size: float = 0.2

    # -------------------------
    # Dataset columns
    # -------------------------
    title_col: str = "title"
    text_col: str = "text"
    label_col: str = "label"

    # Derived columns created by preprocessing
    combined_col: str = "full_text"
    clean_col: str = "clean_text"

    # -------------------------
    # TF-IDF configuration
    # -------------------------
    tfidf_max_features: int = 5000
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    tfidf_stop_words: str = "english"

    # -------------------------
    # Logistic Regression baseline
    # -------------------------
    lr_solver: str = "liblinear"
    lr_max_iter: int = 100

    # -------------------------
    # Paths
    # -------------------------
    # project_root points to the root of the repository
    project_root: Path = Path(__file__).resolve().parents[1]

    # Input data (NOT versioned)
    data_dir: Path = project_root / "data"
    raw_data_path: Path = data_dir / "WELFake_Dataset.csv"

    # Outputs / artifacts
    results_dir: Path = project_root / "results"
    vectorizer_path: Path = results_dir / "tfidf_vectorizer.joblib"
    model_path: Path = results_dir / "credibility_model_calibrated.joblib"
    metrics_path: Path = results_dir / "metrics.json"
    run_config_path: Path = results_dir / "run_config.json"
    preprocess_metadata_path: Path = results_dir / "preprocessing_metadata.json"