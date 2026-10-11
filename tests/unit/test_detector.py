"""Smoke test for the YOLO detector.

This test actually loads the YOLO26n model and runs inference on a
synthetic image. It's marked 'slow' because model loading takes 1-3 seconds.

What this test verifies:
- Model loads without error
- Inference returns a valid DetectionResult
- DetectionResult has correct image dimensions
- Inference time is positive and finite
- Model name is recorded correctly

What this test does NOT verify:
- Detection quality (synthetic shapes won't produce real detections)
- GPU execution (CPU only in CI)
- Camera integration (no hardware needed)
"""

from __future__ import annotations

import numpy as np
import pytest

from vision_monitor.config import DetectorSettings
from vision_monitor.detector import Detector
from vision_monitor.exceptions import ModelLoadError
from vision_monitor.schemas import DetectionResult


@pytest.fixture(scope="module")
def detector() -> Detector:
    """Load the YOLO model once for all tests in this module.

    scope='module' means the model is loaded once and shared across all
    test functions — this avoids paying the 1-3 second load cost per test.
    """
    settings = DetectorSettings(model_name="yolo26n.pt", device="cpu")
    det = Detector(settings)
    det.load()
    return det


@pytest.fixture()
def synthetic_image() -> np.ndarray:
    """A 640x480 BGR image with colored rectangles."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[100:300, 100:300] = (0, 255, 0)  # Green rectangle
    img[200:400, 300:500] = (255, 0, 0)  # Blue rectangle
    return img


class TestDetectorLoad:
    """Tests for model loading behavior."""

    def test_load_invalid_model_raises(self):
        settings = DetectorSettings(model_name="nonexistent_model_xyz.pt")
        det = Detector(settings)
        with pytest.raises(ModelLoadError, match="Failed to load"):
            det.load()

    def test_is_loaded_before_load(self):
        settings = DetectorSettings()
        det = Detector(settings)
        assert det.is_loaded is False

    @pytest.mark.slow
    def test_is_loaded_after_load(self, detector: Detector):
        assert detector.is_loaded is True


@pytest.mark.slow
class TestDetectorInference:
    """Tests for inference on loaded model."""

    def test_detect_returns_detection_result(self, detector: Detector, synthetic_image: np.ndarray):
        result = detector.detect(synthetic_image)
        assert isinstance(result, DetectionResult)

    def test_result_has_correct_dimensions(self, detector: Detector, synthetic_image: np.ndarray):
        result = detector.detect(synthetic_image)
        assert result.image_width == 640
        assert result.image_height == 480

    def test_result_has_positive_inference_time(
        self, detector: Detector, synthetic_image: np.ndarray
    ):
        result = detector.detect(synthetic_image)
        assert result.inference_time_ms > 0
        assert result.inference_time_ms < 60_000  # sanity: under 60 seconds

    def test_result_records_model_name(self, detector: Detector, synthetic_image: np.ndarray):
        result = detector.detect(synthetic_image)
        assert result.model_name == "yolo26n.pt"

    def test_detections_are_list(self, detector: Detector, synthetic_image: np.ndarray):
        result = detector.detect(synthetic_image)
        assert isinstance(result.detections, list)

    def test_detect_before_load_raises(self, synthetic_image: np.ndarray):
        settings = DetectorSettings()
        det = Detector(settings)
        with pytest.raises(ModelLoadError, match="not loaded"):
            det.detect(synthetic_image)

    def test_invalid_image_raises(self, detector: Detector):
        """Grayscale (2D) image should be rejected."""
        bad_img = np.zeros((480, 640), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected BGR"):
            detector.detect(bad_img)

    def test_single_channel_image_raises(self, detector: Detector):
        """Single-channel 3D image should be rejected."""
        bad_img = np.zeros((480, 640, 1), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected BGR"):
            detector.detect(bad_img)
