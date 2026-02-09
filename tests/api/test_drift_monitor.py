"""
Unit tests for the DriftMonitor class.
"""
import time

from src.app.drift_monitor import DriftMonitor, PredictionRecord, DriftMetrics


class TestDriftMonitorInit:
    """Tests for DriftMonitor initialization."""

    def test_default_window_size(self):
        """Default window size should be 1000."""
        monitor = DriftMonitor()
        assert monitor.window_size == 1000

    def test_custom_window_size(self):
        """Custom window size should be respected."""
        monitor = DriftMonitor(window_size=500)
        assert monitor.window_size == 500

    def test_default_reference_stats(self):
        """Default reference stats should be set."""
        monitor = DriftMonitor()
        assert monitor._ref_stats is not None
        assert "prob_mean" in monitor._ref_stats
        assert "prob_std" in monitor._ref_stats
        assert "text_length_mean" in monitor._ref_stats

    def test_starts_empty(self):
        """Monitor should start with no records."""
        monitor = DriftMonitor()
        assert len(monitor._records) == 0


class TestRecordPrediction:
    """Tests for record_prediction method."""

    def test_records_prediction(self):
        """Should record a prediction."""
        monitor = DriftMonitor()
        monitor.record_prediction(
            probability=0.8,
            text="This is a test article",
            label="real"
        )
        assert len(monitor._records) == 1

    def test_records_multiple_predictions(self):
        """Should record multiple predictions."""
        monitor = DriftMonitor()
        for i in range(10):
            monitor.record_prediction(
                probability=0.5 + i * 0.05,
                text=f"Test article number {i}",
                label="real" if i % 2 == 0 else "fake"
            )
        assert len(monitor._records) == 10

    def test_window_size_limit(self):
        """Records should not exceed window size."""
        monitor = DriftMonitor(window_size=5)
        for i in range(10):
            monitor.record_prediction(
                probability=0.5,
                text=f"Article {i}",
                label="real"
            )
        assert len(monitor._records) == 5

    def test_record_stores_probability(self):
        """Recorded prediction should have correct probability."""
        monitor = DriftMonitor()
        monitor.record_prediction(probability=0.75, text="Test text", label="real")
        assert monitor._records[0].probability == 0.75

    def test_record_stores_text_length(self):
        """Recorded prediction should have correct text length (word count)."""
        monitor = DriftMonitor()
        monitor.record_prediction(probability=0.5, text="one two three four", label="real")
        assert monitor._records[0].text_length == 4

    def test_record_stores_label(self):
        """Recorded prediction should have correct label."""
        monitor = DriftMonitor()
        monitor.record_prediction(probability=0.3, text="Test", label="fake")
        assert monitor._records[0].label == "fake"

    def test_record_stores_timestamp(self):
        """Recorded prediction should have timestamp."""
        before = time.time()
        monitor = DriftMonitor()
        monitor.record_prediction(probability=0.5, text="Test", label="real")
        after = time.time()
        
        assert before <= monitor._records[0].timestamp <= after


class TestComputeDrift:
    """Tests for compute_drift method."""

    def test_insufficient_samples(self):
        """Should return insufficient_samples when below MIN_SAMPLES."""
        monitor = DriftMonitor()
        for i in range(10):  # Less than MIN_SAMPLES (50)
            monitor.record_prediction(probability=0.5, text="Test", label="real")
        
        metrics = monitor.compute_drift()
        assert metrics.is_drifted is False
        assert metrics.drift_details["status"] == "insufficient_samples"

    def test_returns_drift_metrics_type(self):
        """Should return DriftMetrics instance."""
        monitor = DriftMonitor()
        metrics = monitor.compute_drift()
        assert isinstance(metrics, DriftMetrics)

    def test_no_drift_with_stable_predictions(self):
        """Should not detect drift with stable predictions near reference."""
        import random
        random.seed(42)
        
        monitor = DriftMonitor()
        # Add predictions close to reference (prob_mean=0.5, length_mean=300, std~0.25)
        for i in range(60):
            # Generate probabilities with mean ~0.5 and std ~0.25
            prob = max(0.0, min(1.0, random.gauss(0.5, 0.25)))
            monitor.record_prediction(
                probability=prob,
                text=" ".join(["word"] * 300),  # 300 words
                label="real" if prob > 0.5 else "fake"
            )
        
        metrics = monitor.compute_drift()
        assert metrics.is_drifted is False
        assert metrics.sample_count == 60

    def test_drift_detected_high_prob_shift(self):
        """Should detect drift when probability mean shifts significantly."""
        monitor = DriftMonitor()
        # Add predictions with high probability (far from reference 0.5)
        for i in range(60):
            monitor.record_prediction(
                probability=0.9,  # Much higher than reference 0.5
                text=" ".join(["word"] * 300),
                label="real"
            )
        
        metrics = monitor.compute_drift()
        assert metrics.is_drifted is True
        assert metrics.prediction_mean_shift > monitor.PROB_MEAN_THRESHOLD

    def test_drift_detected_text_length_shift(self):
        """Should detect drift when text length shifts significantly."""
        monitor = DriftMonitor()
        # Add predictions with very short text (far from reference 300)
        for i in range(60):
            monitor.record_prediction(
                probability=0.5,
                text="short",  # Only 1 word vs reference 300
                label="real"
            )
        
        metrics = monitor.compute_drift()
        assert metrics.is_drifted is True
        assert metrics.text_length_shift > monitor.TEXT_LENGTH_THRESHOLD

    def test_computes_label_distribution(self):
        """Should compute label distribution in drift details."""
        monitor = DriftMonitor()
        for i in range(60):
            monitor.record_prediction(
                probability=0.5,
                text=" ".join(["word"] * 300),
                label="real" if i < 40 else "fake"
            )
        
        metrics = monitor.compute_drift()
        assert "label_distribution" in metrics.drift_details
        assert metrics.drift_details["label_distribution"]["real"] == 40
        assert metrics.drift_details["label_distribution"]["fake"] == 20


class TestGetStatus:
    """Tests for get_status method."""

    def test_status_has_required_keys(self):
        """Status should have all required keys."""
        monitor = DriftMonitor()
        status = monitor.get_status()
        
        assert "is_drifted" in status
        assert "sample_count" in status
        assert "window_size" in status
        assert "metrics" in status
        assert "details" in status
        assert "last_computed" in status

    def test_status_metrics_rounded(self):
        """Status metrics should be rounded to 4 decimal places."""
        monitor = DriftMonitor()
        for i in range(60):
            monitor.record_prediction(probability=0.5, text=" ".join(["w"] * 300), label="real")
        
        status = monitor.get_status()
        metrics = status["metrics"]
        
        # Check they are properly formatted (max 4 decimal places)
        for key in ["prediction_mean_shift", "prediction_std_shift", "text_length_shift"]:
            value = metrics[key]
            # Verify it's rounded (string representation check)
            assert len(str(value).split(".")[-1]) <= 4 if "." in str(value) else True


class TestReset:
    """Tests for reset method."""

    def test_reset_clears_records(self):
        """Reset should clear all recorded predictions."""
        monitor = DriftMonitor()
        for i in range(10):
            monitor.record_prediction(probability=0.5, text="Test", label="real")
        
        assert len(monitor._records) == 10
        
        monitor.reset()
        
        assert len(monitor._records) == 0

    def test_reset_clears_metrics_history(self):
        """Reset should clear metrics history."""
        monitor = DriftMonitor()
        for i in range(60):
            monitor.record_prediction(probability=0.5, text=" ".join(["w"] * 300), label="real")
        
        monitor.compute_drift()  # This adds to metrics history
        assert len(monitor._metrics_history) > 0
        
        monitor.reset()
        
        assert len(monitor._metrics_history) == 0

    def test_status_after_reset(self):
        """Status after reset should show 0 samples."""
        monitor = DriftMonitor()
        for i in range(60):
            monitor.record_prediction(probability=0.5, text="Test", label="real")
        
        monitor.reset()
        status = monitor.get_status()
        
        assert status["sample_count"] == 0
        assert status["is_drifted"] is False


class TestPredictionRecord:
    """Tests for PredictionRecord dataclass."""

    def test_create_record(self):
        """Should create prediction record with all fields."""
        record = PredictionRecord(
            timestamp=1234567890.0,
            probability=0.75,
            text_length=100,
            label="real"
        )
        
        assert record.timestamp == 1234567890.0
        assert record.probability == 0.75
        assert record.text_length == 100
        assert record.label == "real"


class TestDriftMetrics:
    """Tests for DriftMetrics dataclass."""

    def test_create_metrics(self):
        """Should create drift metrics with all fields."""
        metrics = DriftMetrics(
            prediction_mean_shift=0.1,
            prediction_std_shift=0.05,
            text_length_shift=0.2,
            sample_count=100,
            is_drifted=False
        )
        
        assert metrics.prediction_mean_shift == 0.1
        assert metrics.prediction_std_shift == 0.05
        assert metrics.text_length_shift == 0.2
        assert metrics.sample_count == 100
        assert metrics.is_drifted is False

    def test_default_drift_details(self):
        """Drift details should default to empty dict."""
        metrics = DriftMetrics(
            prediction_mean_shift=0.1,
            prediction_std_shift=0.05,
            text_length_shift=0.2,
            sample_count=100,
            is_drifted=False
        )
        
        assert metrics.drift_details == {}


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_recording(self):
        """Should handle concurrent recordings safely."""
        import threading
        
        monitor = DriftMonitor(window_size=1000)
        
        def record_many(n):
            for i in range(n):
                monitor.record_prediction(
                    probability=0.5,
                    text=f"Article {i}",
                    label="real"
                )
        
        threads = [threading.Thread(target=record_many, args=(100,)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All 500 records should be present (or window_size, whichever is smaller)
        assert len(monitor._records) == 500
