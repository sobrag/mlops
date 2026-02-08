from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator


def predict_proba_real(
    model: BaseEstimator,
    X: csr_matrix,
    positive_label_index: int = 0,
) -> np.ndarray:
    """
    Predict probability of the positive (\"real\") class.

    Args:
        model: fitted sklearn classifier with predict_proba
        X: vectorized input texts
        positive_label_index: index of the \"real\" class in predict_proba output
                              (WELFake dataset: 0=real, 1=fake)

    Returns:
        Array of probabilities P(y = real | x)
    """
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Model must implement predict_proba().")

    proba = model.predict_proba(X)
    return proba[:, positive_label_index]


def predict_credibility_score(
    model: BaseEstimator,
    X: csr_matrix,
    scale: float = 100.0,
) -> np.ndarray:
    """
    Predict credibility score from calibrated probabilities.

    The score is defined as:
        credibility_score = P(real | x) * scale

    Note: WELFake dataset uses label 0 = 'real', label 1 = 'fake'.
    
    Args:
        model: fitted (ideally calibrated) classifier
        X: vectorized input texts
        scale: scaling factor (default: 100 for [0,100] score)

    Returns:
        Array of credibility scores
    """
    # check modello
    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "Model must implement predict_proba(), ideally calibrated for credibility score."
        )
    
    p_real = predict_proba_real(model, X)
    return p_real * scale


def predict_label(
    model: BaseEstimator,
    X: csr_matrix,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Predict binary labels from probabilities using a fixed threshold.

    Args:
        model: fitted classifier
        X: vectorized input texts
        threshold: decision threshold on P(real)

    Returns:
        Array of predicted labels (0/1)
    """
    p_real = predict_proba_real(model, X)
    return (p_real >= threshold).astype(int)


def predict_all(
    model: BaseEstimator,
    X: csr_matrix,
    threshold: float = 0.5,
    scale: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience helper returning all prediction outputs at once.

    Returns:
        (probabilities, credibility_scores, labels)
    """
    p_real = predict_proba_real(model, X)
    scores = p_real * scale
    labels = (p_real >= threshold).astype(int)
    return p_real, scores, labels
