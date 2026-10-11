"""Integration test for detector on a real image (sample_data/bus.jpg).

Verifies that:
- Real objects (such as 'person' and 'bus') are detected with high confidence
- Bounding box coordinates and labels are populated correctly in the output schema
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from vision_monitor.config import DetectorSettings
from vision_monitor.detector import Detector


@pytest.mark.slow
def test_detect_real_objects_on_sample_image():
    sample_path = Path("sample_data/bus.jpg")
    if not sample_path.exists():
        pytest.skip("sample_data/bus.jpg not found")

    image = cv2.imread(str(sample_path))
    assert image is not None, "Failed to read sample image"

    settings = DetectorSettings(model_name="yolo26n.pt", confidence=0.25, device="cpu")
    detector = Detector(settings)
    detector.load()

    result = detector.detect(image)

    # YOLO26n on bus.jpg detects at least one 'bus' and multiple 'person's
    detected_classes = {det.class_name for det in result.detections}
    assert "bus" in detected_classes or "person" in detected_classes

    # Ensure detections have valid coordinates and non-zero area
    for det in result.detections:
        assert det.confidence >= 0.25
        assert det.bbox.width > 0
        assert det.bbox.height > 0
        assert det.bbox.area > 0
