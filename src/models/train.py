from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal

from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


ModelName = Literal["logreg", "rf", "svc"]


@dataclass(frozen=True)
class TrainConfig:
    """
    Training hyperparameters (keep these aligned with your notebook defaults).
    """
    # Logistic Regression
    lr_solver: str = "liblinear"
    lr_max_iter: int = 100
    lr_C: float = 1.0

    # Random Forest
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = None
    rf_min_samples_split: int = 2
    rf_min_samples_leaf: int = 1

    # SVC
    svc_C: float = 1.0
    svc_kernel: str = "rbf"
    svc_max_iter: int = 5000

    # Shared
    random_state: int = 0
    n_jobs: int = -1


def build_model(name: ModelName, cfg: TrainConfig) -> BaseEstimator:
    """
    Build an unfitted model instance.
    """
    if name == "logreg":
        return LogisticRegression(
            solver=cfg.lr_solver,
            max_iter=cfg.lr_max_iter,
            C=cfg.lr_C,
            # Optional but safe for reproducibility
            random_state=cfg.random_state,
        )

    if name == "rf":
        return RandomForestClassifier(
            n_estimators=cfg.rf_n_estimators,
            max_depth=cfg.rf_max_depth,
            min_samples_split=cfg.rf_min_samples_split,
            min_samples_leaf=cfg.rf_min_samples_leaf,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
        )

    if name == "svc":
        # probability=True is required for predict_proba -> credibility score
        return SVC(
            C=cfg.svc_C,
            kernel=cfg.svc_kernel,
            probability=True,
            max_iter=cfg.svc_max_iter,
            random_state=cfg.random_state,
        )

    raise ValueError(f"Unknown model name: {name}")


def train_model(
    name: ModelName,
    X_train: csr_matrix,
    y_train,
    cfg: Optional[TrainConfig] = None,
    override_params: Optional[Dict[str, Any]] = None,
) -> BaseEstimator:
    """
    Train a model on vectorized text.

    Args:
        name: one of {"logreg","rf","svc"}
        X_train: sparse TF-IDF matrix
        y_train: labels (0/1)
        cfg: training configuration
        override_params: optional dict to override estimator params (e.g. {"C": 0.5})

    Returns:
        fitted sklearn estimator
    """
    cfg = cfg or TrainConfig()
    model = build_model(name, cfg)

    if override_params:
        model.set_params(**override_params)

    model.fit(X_train, y_train)
    return model
