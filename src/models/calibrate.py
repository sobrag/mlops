from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV


CalibMethod = Literal["sigmoid", "isotonic"]


@dataclass(frozen=True)
class CalibrateConfig:
    """
    Configuration for probability calibration.

    Notes:
    - "sigmoid" (Platt scaling) is usually more stable on smaller datasets.
    - "isotonic" is more flexible but can overfit if data is limited.
    """
    method: CalibMethod = "isotonic"
    cv: int = 5


def calibrate_model(
    base_model: BaseEstimator,
    X_cal: csr_matrix,
    y_cal,
    cfg: Optional[CalibrateConfig] = None,
) -> BaseEstimator:
    """
    Wrap a fitted classifier with probability calibration.

    Args:
        base_model: fitted sklearn classifier (must support decision_function or predict_proba)
        X_cal: calibration features (typically X_train; calibration is done via CV)
        y_cal: calibration labels
        cfg: calibration configuration

    Returns:
        CalibratedClassifierCV model (fitted)
    """
    cfg = cfg or CalibrateConfig()

    # CalibratedClassifierCV will refit internally across CV folds.
    cal = CalibratedClassifierCV(
        estimator=base_model,  # sklearn >= 1.2 uses 'estimator' instead of 'base_estimator'
        method=cfg.method,
        cv=cfg.cv,
    )
    cal.fit(X_cal, y_cal)
    return cal
