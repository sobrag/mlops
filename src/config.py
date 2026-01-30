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
    # Logging / tracking
    # -------------------------
    use_neptune: bool = False
    neptune_project: Optional[str] = None
    neptune_api_token_env: str = "NEPTUNE_API_TOKEN"  # token read from env
    neptune_mode: str = "async"


    # -------------------------
    # Paths
    # -------------------------
    project_root: Path = Path(__file__).resolve().parents[1]

    # Input data
    raw_data_path: Path = project_root / "data" / "sample_welfake.csv"

    # Outputs
    results_dir: Path = project_root / "results"
    artifacts_dir: Path = project_root / "artifacts"

    # (Optional legacy single-file outputs; you can remove later once artifacts are in place)
    preprocess_metadata_path: Path = results_dir / "preprocessing_metadata.json"

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "Config":
        """
        Load config from YAML. YAML overrides dataclass defaults.
        Supports optional nested sections:
          paths: { raw_data_path: "...", results_dir: "...", artifacts_dir: "..." }
          logging: { use_neptune: true, neptune_project: "...", neptune_api_token_env: "..." }
        """
        yaml_path = Path(yaml_path)
        base = cls()  # defaults

        with open(yaml_path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}

        # Flatten optional nested keys (so your YAML can be nice and clean)
        paths = data.pop("paths", {}) or {}
        logging = data.pop("logging", {}) or {}

        # Start from defaults and override with YAML
        cfg = base

        # Top-level overrides (seed, test_size, tfidf_*, lr_*, columns...)
        # Handle tfidf_ngram_range if YAML uses list [1,2]
        if "tfidf_ngram_range" in data:
            ngr = data["tfidf_ngram_range"]
            if isinstance(ngr, (list, tuple)) and len(ngr) == 2:
                data["tfidf_ngram_range"] = (int(ngr[0]), int(ngr[1]))

        # Apply simple overrides that match dataclass fields
        # (ignore unknown keys to avoid crashing on extra YAML keys)
        for k, v in data.items():
            if hasattr(cfg, k):
                cfg = replace(cfg, **{k: v})

        # Apply logging overrides
        for k, v in logging.items():
            if hasattr(cfg, k):
                cfg = replace(cfg, **{k: v})

        # Apply path overrides (resolve relative to project root)
        # Note: project_root stays the repo root inferred from this file.
        if "raw_data_path" in paths:
            cfg = replace(cfg, raw_data_path=_as_path(cfg.project_root, paths["raw_data_path"]))
        if "results_dir" in paths:
            cfg = replace(cfg, results_dir=_as_path(cfg.project_root, paths["results_dir"]))
        if "artifacts_dir" in paths:
            cfg = replace(cfg, artifacts_dir=_as_path(cfg.project_root, paths["artifacts_dir"]))

        # Recompute dependent paths if results_dir changed
        cfg = replace(cfg, preprocess_metadata_path=cfg.results_dir / "preprocessing_metadata.json")

        return cfg
