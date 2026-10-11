"""Unit tests for schemas — the shared data contracts.

These tests are FAST (no model loading, no camera, no I/O).
They verify the data layer that every other module depends on.

TDD cycle demonstrated:
1. RED   — We write tests for BoundingBox properties, Detection defaults, etc.
2. GREEN — The schemas module already implements these (we wrote it first for
           Sprint 1, but in strict TDD you'd write tests before schemas).
3. REFACTOR — If a test reveals awkward API, we improve the schema.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vision_monitor.schemas import BoundingBox, Detection, DetectionResult

# ─── BoundingBox ─────────────────────────────────────────────────────────


class TestBoundingBox:
    """Tests for the BoundingBox data model."""

    def test_basic_creation(self):
        bbox = BoundingBox(x1=10, y1=20, x2=110, y2=120)
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 110
        assert bbox.y2 == 120

    def test_width_and_height(self):
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        assert bbox.width == 100.0
        assert bbox.height == 50.0

    def test_center(self):
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        cx, cy = bbox.center
        assert cx == 50.0
        assert cy == 50.0

    def test_area(self):
        bbox = BoundingBox(x1=10, y1=20, x2=60, y2=70)
        assert bbox.area == 50.0 * 50.0

    def test_zero_area_box(self):
        """A box with zero width or height has zero area."""
        bbox = BoundingBox(x1=50, y1=50, x2=50, y2=50)
        assert bbox.area == 0.0

    def test_float_coordinates(self):
        """YOLO outputs float coordinates — we must handle them."""
        bbox = BoundingBox(x1=10.5, y1=20.3, x2=110.7, y2=120.9)
        assert bbox.width == pytest.approx(100.2)
        assert bbox.height == pytest.approx(100.6)


# ─── Detection ───────────────────────────────────────────────────────────


class TestDetection:
    """Tests for the Detection data model."""

    def test_basic_detection(self):
        det = Detection(
            class_name="person",
            confidence=0.85,
            bbox=BoundingBox(x1=10, y1=20, x2=100, y2=200),
        )
        assert det.class_name == "person"
        assert det.confidence == 0.85
        assert det.distance_m is None  # Not set yet
        assert det.alert is None  # Not evaluated yet

    def test_confidence_must_be_in_range(self):
        """Confidence outside [0, 1] must be rejected."""
        with pytest.raises(ValidationError):
            Detection(
                class_name="car",
                confidence=1.5,
                bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
            )

    def test_negative_confidence_rejected(self):
        with pytest.raises(ValidationError):
            Detection(
                class_name="car",
                confidence=-0.1,
                bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
            )

    def test_optional_distance(self):
        """distance_m should accept None, 0, and positive values."""
        det_none = Detection(
            class_name="cat",
            confidence=0.9,
            bbox=BoundingBox(x1=0, y1=0, x2=50, y2=50),
            distance_m=None,
        )
        assert det_none.distance_m is None

        det_val = Detection(
            class_name="cat",
            confidence=0.9,
            bbox=BoundingBox(x1=0, y1=0, x2=50, y2=50),
            distance_m=3.5,
        )
        assert det_val.distance_m == 3.5

    def test_optional_alert(self):
        det = Detection(
            class_name="dog",
            confidence=0.7,
            bbox=BoundingBox(x1=0, y1=0, x2=50, y2=50),
            alert=True,
        )
        assert det.alert is True


# ─── DetectionResult ─────────────────────────────────────────────────────


class TestDetectionResult:
    """Tests for the per-frame detection result container."""

    def test_empty_detections(self):
        result = DetectionResult(
            detections=[],
            inference_time_ms=15.3,
            image_width=640,
            image_height=480,
            model_name="yolo26n.pt",
        )
        assert len(result.detections) == 0
        assert result.inference_time_ms == 15.3
        assert result.model_name == "yolo26n.pt"

    def test_with_detections(self):
        dets = [
            Detection(
                class_name="person",
                confidence=0.9,
                bbox=BoundingBox(x1=10, y1=10, x2=100, y2=200),
            ),
            Detection(
                class_name="car",
                confidence=0.7,
                bbox=BoundingBox(x1=200, y1=100, x2=400, y2=300),
            ),
        ]
        result = DetectionResult(
            detections=dets,
            inference_time_ms=42.5,
            image_width=1920,
            image_height=1080,
            model_name="yolo26n.pt",
        )
        assert len(result.detections) == 2
        assert result.detections[0].class_name == "person"
        assert result.detections[1].class_name == "car"

    def test_negative_inference_time_rejected(self):
        with pytest.raises(ValidationError):
            DetectionResult(
                detections=[],
                inference_time_ms=-1.0,
                image_width=640,
                image_height=480,
                model_name="yolo26n.pt",
            )

    def test_invalid_dimensions_rejected(self):
        with pytest.raises(ValidationError):
            DetectionResult(
                detections=[],
                inference_time_ms=10.0,
                image_width=0,
                image_height=480,
                model_name="yolo26n.pt",
            )
