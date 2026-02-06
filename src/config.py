from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Tuple, Optional, Dict

import yaml


def _as_path(project_root: Path, p: str | Path) -> Path:
    """Resolve relative paths w.r.t. project root."""
    p = Path(p)
    return p if p.is_absolute() else (project_root / p).resolve()


@dataclass(frozen=True)
class Config:
    """
    Central configuration for the project.
    Defaults live here; YAML overrides them.
    """

    # Model selection
    model_name: str = "logreg"

    # Reproducibility
    seed: int = 42
    test_size: float = 0.2

    # Dataset columns
    title_col: str = "title"
    text_col: str = "text"
    label_col: str = "label"

    # Derived columns created by preprocessing
    combined_col: str = "full_text"
    clean_col: str = "clean_text"

    # TF-IDF configuration
    tfidf_max_features: int = 5000
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    tfidf_stop_words: str = "english"

    # Logistic Regression baseline
    lr_solver: str = "liblinear"
    lr_max_iter: int = 100
    lr_C: float = 1.0

    # Logging / tracking
    use_wandb: bool = False  
    wandb_project: Optional[str] = None  
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"

    # W&B Dataset artifact (optional)
    dataset_artifact: Optional[str] = None

    # Paths
    project_root: Path = Path(__file__).resolve().parents[1]

    # Input data
    raw_data_path: Path = project_root / "data" / "sample_welfake.csv"

    # Outputs
    results_dir: Path = project_root / "results"
    artifacts_dir: Path = project_root / "artifacts"

    preprocess_metadata_path: Path = results_dir / "preprocessing_metadata.json"

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "Config":
        """
        Load config from YAML. YAML overrides dataclass defaults.
        """
        yaml_path = Path(yaml_path)
        base = cls()

        with open(yaml_path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}

        # Flatten nested sections
        paths = data.pop("paths", {}) or {}
        logging_cfg = data.pop("logging", {}) or {}
        data_cfg = data.pop("data", {}) or {}
        data_cfg = data.pop("data", {}) or {}

        cfg = base

        # Handle tfidf_ngram_range
        if "tfidf_ngram_range" in data:
            ngr = data["tfidf_ngram_range"]
            if isinstance(ngr, (list, tuple)) and len(ngr) == 2:
                data["tfidf_ngram_range"] = (int(ngr[0]), int(ngr[1]))

        # Apply top-level overrides
        for k, v in data.items():
            if hasattr(cfg, k):
                cfg = replace(cfg, **{k: v})

        # Apply logging overrides
        for k, v in logging_cfg.items():
            if hasattr(cfg, k):
                cfg = replace(cfg, **{k: v})

        # Apply data overrides (dataset_artifact)
        # Apply data overrides (dataset_artifact)
        for k, v in data_cfg.items():
            if hasattr(cfg, k):
                cfg = replace(cfg, **{k: v})

        # Apply path overrides
        if "raw_data_path" in paths:
            cfg = replace(cfg, raw_data_path=_as_path(cfg.project_root, paths["raw_data_path"]))
        if "results_dir" in paths:
            cfg = replace(cfg, results_dir=_as_path(cfg.project_root, paths["results_dir"]))
        if "artifacts_dir" in paths:
            cfg = replace(cfg, artifacts_dir=_as_path(cfg.project_root, paths["artifacts_dir"]))

        # Recompute dependent paths
        cfg = replace(cfg, preprocess_metadata_path=cfg.results_dir / "preprocessing_metadata.json")

        return cfg