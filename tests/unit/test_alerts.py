"""Unit tests for proximity alerts (TDD).

Tests:
- Distance < threshold triggers alert (True)
- Distance >= threshold does not trigger alert (False)
- Missing distance (None) results in alert = None (never false positive)
- Zero or negative distance edge cases
- Pure function behavior
"""

from __future__ import annotations

import pytest

from vision_monitor.alerts import evaluate_alert, evaluate_detection_alerts
from vision_monitor.config import AlertSettings
from vision_monitor.schemas import BoundingBox, Detection


class TestEvaluateAlert:
    """Tests for single distance alert evaluation."""

    def test_object_closer_than_threshold_triggers_alert(self):
        # Threshold: 2.0m, object distance: 1.2m -> True
        assert evaluate_alert(distance_m=1.2, threshold_m=2.0) is True

    def test_object_further_than_threshold_no_alert(self):
        # Threshold: 2.0m, object distance: 2.5m -> False
        assert evaluate_alert(distance_m=2.5, threshold_m=2.0) is False

    def test_object_at_exact_threshold_no_alert(self):
        # At boundary (2.0m) -> False (strictly closer is < threshold)
        assert evaluate_alert(distance_m=2.0, threshold_m=2.0) is False

    def test_missing_distance_returns_none(self):
        # Depth unavailable -> None, NEVER False or True
        assert evaluate_alert(distance_m=None, threshold_m=2.0) is None

    def test_nan_or_negative_distance_returns_none(self):
        assert evaluate_alert(distance_m=float("nan"), threshold_m=2.0) is None
        assert evaluate_alert(distance_m=-1.0, threshold_m=2.0) is None
        assert evaluate_alert(distance_m=float("inf"), threshold_m=2.0) is None

    def test_negative_or_zero_threshold_raises_value_error(self):
        with pytest.raises(ValueError, match="Threshold must be positive"):
            evaluate_alert(distance_m=1.0, threshold_m=0.0)


class TestEvaluateDetectionAlerts:
    """Tests for updating Detection schema objects with alert flags."""

    def test_updates_detections_in_place_and_returns(self):
        settings = AlertSettings(distance_threshold_m=2.0)
        dets = [
            Detection(
                class_name="person",
                confidence=0.9,
                bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
                distance_m=1.5,
            ),
            Detection(
                class_name="chair",
                confidence=0.8,
                bbox=BoundingBox(x1=20, y1=20, x2=30, y2=30),
                distance_m=3.0,
            ),
            Detection(
                class_name="bottle",
                confidence=0.7,
                bbox=BoundingBox(x1=40, y1=40, x2=50, y2=50),
                distance_m=None,
            ),
        ]

        evaluated = evaluate_detection_alerts(dets, settings.distance_threshold_m)

        assert evaluated[0].alert is True
        assert evaluated[1].alert is False
        assert evaluated[2].alert is None
