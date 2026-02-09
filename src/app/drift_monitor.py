"""
Real-time drift monitoring for the prediction API.

Tracks prediction statistics in a rolling window and calculates drift metrics
compared to reference statistics computed during training.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class PredictionRecord:
    """Single prediction record for drift tracking."""
    timestamp: float
    probability: float
    text_length: int  # word count
    label: str


@dataclass
class DriftMetrics:
    """Computed drift metrics."""
    prediction_mean_shift: float  # difference in mean probability
    prediction_std_shift: float   # difference in std of probabilities
    text_length_shift: float      # difference in mean text length
    sample_count: int
    is_drifted: bool
    drift_details: Dict[str, Any] = field(default_factory=dict)


class DriftMonitor:
    """
    Monitors prediction drift in real-time using a rolling window.
    
    Compares current prediction statistics against reference statistics
    to detect potential data or concept drift.
    """
    
    # Thresholds for drift detection
    PROB_MEAN_THRESHOLD = 0.15      # 15% shift in mean probability
    PROB_STD_THRESHOLD = 0.10       # 10% shift in std
    TEXT_LENGTH_THRESHOLD = 0.30    # 30% shift in mean text length
    MIN_SAMPLES = 50                # Minimum samples before computing drift
    
    def __init__(
        self,
        window_size: int = 1000,
        reference_stats_path: Optional[Path] = None,
    ):
        """
        Initialize drift monitor.
        
        Args:
            window_size: Number of predictions to keep in rolling window
            reference_stats_path: Path to JSON with reference statistics
        """
        self.window_size = window_size
        self._records: deque[PredictionRecord] = deque(maxlen=window_size)
        self._lock = threading.Lock()
        
        # Reference statistics (from training data)
        self._ref_stats: Optional[Dict[str, Any]] = None
        if reference_stats_path and reference_stats_path.exists():
            self._load_reference_stats(reference_stats_path)
        else:
            # Default reference stats (calibrated model should have ~0.5 mean on balanced data)
            self._ref_stats = {
                "prob_mean": 0.5,
                "prob_std": 0.25,
                "text_length_mean": 300,
                "text_length_std": 200,
            }
        
        # Metrics history for trend analysis
        self._metrics_history: deque[DriftMetrics] = deque(maxlen=100)
        self._last_computed: float = 0
    
    def _load_reference_stats(self, path: Path) -> None:
        """Load reference statistics from JSON file."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            # Extract relevant stats
            self._ref_stats = {
                "prob_mean": np.mean(data.get("prob_dist", [0.5])),
                "prob_std": np.std(data.get("prob_dist", [0.25])),
                "text_length_mean": data.get("text_length_mean", 300),
                "text_length_std": data.get("text_length_std", 200),
            }
        except Exception:
            # Fallback to defaults
            self._ref_stats = {
                "prob_mean": 0.5,
                "prob_std": 0.25,
                "text_length_mean": 300,
                "text_length_std": 200,
            }
    
    def record_prediction(
        self,
        probability: float,
        text: str,
        label: str,
    ) -> None:
        """
        Record a prediction for drift monitoring.
        
        Args:
            probability: Predicted probability (0-1)
            text: Input text
            label: Predicted label ('real' or 'fake')
        """
        text_length = len(text.split())
        record = PredictionRecord(
            timestamp=time.time(),
            probability=probability,
            text_length=text_length,
            label=label,
        )
        
        with self._lock:
            self._records.append(record)
    
    def compute_drift(self) -> DriftMetrics:
        """
        Compute current drift metrics.
        
        Returns:
            DriftMetrics with current drift status
        """
        with self._lock:
            records = list(self._records)
        
        sample_count = len(records)
        
        if sample_count < self.MIN_SAMPLES:
            return DriftMetrics(
                prediction_mean_shift=0.0,
                prediction_std_shift=0.0,
                text_length_shift=0.0,
                sample_count=sample_count,
                is_drifted=False,
                drift_details={"status": "insufficient_samples", "min_required": self.MIN_SAMPLES},
            )
        
        # Compute current statistics
        probs = np.array([r.probability for r in records])
        lengths = np.array([r.text_length for r in records])
        
        current_prob_mean = float(np.mean(probs))
        current_prob_std = float(np.std(probs))
        current_length_mean = float(np.mean(lengths))
        
        # Compute shifts (normalized)
        ref_prob_mean = self._ref_stats["prob_mean"]
        ref_prob_std = self._ref_stats["prob_std"]
        ref_length_mean = self._ref_stats["text_length_mean"]
        
        prob_mean_shift = abs(current_prob_mean - ref_prob_mean)
        prob_std_shift = abs(current_prob_std - ref_prob_std)
        length_shift = abs(current_length_mean - ref_length_mean) / max(ref_length_mean, 1)
        
        # Check if drifted
        is_drifted = (
            prob_mean_shift > self.PROB_MEAN_THRESHOLD or
            prob_std_shift > self.PROB_STD_THRESHOLD or
            length_shift > self.TEXT_LENGTH_THRESHOLD
        )
        
        # Label distribution
        label_counts = {}
        for r in records:
            label_counts[r.label] = label_counts.get(r.label, 0) + 1
        
        metrics = DriftMetrics(
            prediction_mean_shift=prob_mean_shift,
            prediction_std_shift=prob_std_shift,
            text_length_shift=length_shift,
            sample_count=sample_count,
            is_drifted=is_drifted,
            drift_details={
                "current_prob_mean": current_prob_mean,
                "current_prob_std": current_prob_std,
                "current_length_mean": current_length_mean,
                "reference_prob_mean": ref_prob_mean,
                "reference_prob_std": ref_prob_std,
                "reference_length_mean": ref_length_mean,
                "label_distribution": label_counts,
                "thresholds": {
                    "prob_mean": self.PROB_MEAN_THRESHOLD,
                    "prob_std": self.PROB_STD_THRESHOLD,
                    "text_length": self.TEXT_LENGTH_THRESHOLD,
                },
            },
        )
        
        self._metrics_history.append(metrics)
        self._last_computed = time.time()
        
        return metrics
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current drift monitoring status.
        
        Returns:
            Dictionary with drift status and metrics
        """
        metrics = self.compute_drift()
        
        return {
            "is_drifted": metrics.is_drifted,
            "sample_count": metrics.sample_count,
            "window_size": self.window_size,
            "metrics": {
                "prediction_mean_shift": round(metrics.prediction_mean_shift, 4),
                "prediction_std_shift": round(metrics.prediction_std_shift, 4),
                "text_length_shift": round(metrics.text_length_shift, 4),
            },
            "details": metrics.drift_details,
            "last_computed": self._last_computed,
        }
    
    def reset(self) -> None:
        """Clear all recorded predictions."""
        with self._lock:
            self._records.clear()
            self._metrics_history.clear()


# Global drift monitor instance
_drift_monitor: Optional[DriftMonitor] = None


def get_drift_monitor() -> DriftMonitor:
    """Get or create the global drift monitor instance."""
    global _drift_monitor
    if _drift_monitor is None:
        _drift_monitor = DriftMonitor()
    return _drift_monitor


def set_drift_monitor(monitor: DriftMonitor) -> None:
    """Set the global drift monitor instance."""
    global _drift_monitor
    _drift_monitor = monitor
