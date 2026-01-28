from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    log_loss,
)


# CHECK THIS
# Optional: only used if you want a quick calibration diagnostic number
# (won't break if you don't use it in your pipeline)
try:
    from sklearn.calibration import calibration_curve
except Exception:  # pragma: no cover
    calibration_curve = None


@dataclass(frozen=True)
class EvalConfig:
    """
    Evaluation configuration for binary + probability outputs.
    """
    threshold: float = 0.5
    eps: float = 1e-15  # for clipping probabilities in log-loss


def evaluate_predictions(
    y_true,
    p_real: np.ndarray,
    cfg: Optional[EvalConfig] = None,
) -> Dict[str, float]:
    """
    Evaluate a calibrated probability output (P(real|x)).

    This is the right evaluation set for a "credibility score" system:
    - ranking quality: AUC
    - probability quality: Brier, LogLoss
    - optional classification metrics at a chosen threshold: accuracy/precision/recall/F1

    Args:
        y_true: true labels (0/1)
        p_real: predicted probabilities for class 1 ("real")
        cfg: evaluation config (threshold, eps)

    Returns:
        dict of metrics
    """
    cfg = cfg or EvalConfig()

    y_true = np.asarray(y_true).astype(int)
    p_real = np.asarray(p_real).astype(float)

    # Defensive: avoid log(0) in log loss
    p_clip = np.clip(p_real, cfg.eps, 1.0 - cfg.eps)

    # Hard predictions for threshold-based metrics
    y_pred = (p_real >= cfg.threshold).astype(int)

    metrics: Dict[str, float] = {
        "threshold": float(cfg.threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, p_real)),
        "brier": float(brier_score_loss(y_true, p_real)),
        "log_loss": float(log_loss(y_true, p_clip)),
    }

    return metrics


def expected_calibration_error(
    y_true,
    p_real: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Simple Expected Calibration Error (ECE) for quick diagnostics.

    ECE is NOT required, but it's a nice MLOps metric for a credibility score.

    Returns:
        ece in [0, 1] (lower is better)
    """
    y_true = np.asarray(y_true).astype(int)
    p_real = np.asarray(p_real).astype(float)

    # Bin by predicted probability
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p_real, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        conf = float(np.mean(p_real[mask]))        # average predicted prob in bin
        acc = float(np.mean(y_true[mask]))         # empirical frequency in bin
        weight = float(np.mean(mask))              # fraction of samples in bin
        ece += weight * abs(acc - conf)

    return float(ece)


def calibration_curve_points(
    y_true,
    p_real: np.ndarray,
    n_bins: int = 10,
):
    """
    Return points for a reliability diagram (fraction_of_positives vs mean_predicted_value).

    Use this in notebooks/plots, not necessarily in production scripts.
    """
    if calibration_curve is None:
        raise ImportError("sklearn.calibration.calibration_curve not available in this environment.")
    frac_pos, mean_pred = calibration_curve(y_true, p_real, n_bins=n_bins, strategy="uniform")
    return frac_pos, mean_pred
