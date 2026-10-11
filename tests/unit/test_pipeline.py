"""Unit tests for the end-to-end VisionPipeline.

Verifies:
- Frame processing coordinates detector, depth provider, and alert engine
- Frame timings are recorded accurately
- Detections receive estimated distance and alert states
"""

from __future__ import annotations

import numpy as np
import pytest

from vision_monitor.config import AlertSettings
from vision_monitor.depth import SimulatedDepthProvider
from vision_monitor.pipeline import VisionPipeline


@pytest.fixture()
def dummy_detector():
    """Mock-like detector that returns deterministic detections without loading YOLO weights."""

    class MockDetector:
        is_loaded = True

        def detect(self, image: np.ndarray):
            from vision_monitor.schemas import BoundingBox, Detection, DetectionResult

            return DetectionResult(
                detections=[
                    Detection(
                        class_name="person",
                        confidence=0.92,
                        bbox=BoundingBox(x1=20, y1=20, x2=80, y2=80),
                    )
                ],
                inference_time_ms=18.5,
                image_width=image.shape[1],
                image_height=image.shape[0],
                model_name="mock-model",
            )

    return MockDetector()


class TestVisionPipeline:
    def test_pipeline_execution(self, dummy_detector):
        depth_provider = SimulatedDepthProvider(default_distance_m=1.2)
        alert_settings = AlertSettings(distance_threshold_m=2.0)

        pipeline = VisionPipeline(
            detector=dummy_detector,
            depth_provider=depth_provider,
            alert_settings=alert_settings,
        )

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        pipeline_output = pipeline.process_frame(frame)

        # Verify detections were enriched
        assert len(pipeline_output.result.detections) == 1
        det = pipeline_output.result.detections[0]
        assert det.distance_m == pytest.approx(1.2, rel=1e-2)
        assert det.alert is True  # 1.2m < 2.0m threshold -> alert!

        # Verify timings are recorded
        assert pipeline_output.total_time_ms > 0
        assert pipeline_output.is_depth_simulated is True

    def test_pipeline_properties(self, dummy_detector):
        depth_provider = SimulatedDepthProvider(default_distance_m=1.5)
        alert_settings = AlertSettings(distance_threshold_m=2.5)

        pipeline = VisionPipeline(
            detector=dummy_detector,
            depth_provider=depth_provider,
            alert_settings=alert_settings,
        )

        assert pipeline.detector is dummy_detector
        assert pipeline.depth_provider is depth_provider
        assert pipeline.alert_settings is alert_settings
