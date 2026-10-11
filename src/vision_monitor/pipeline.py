"""Vision pipeline orchestrator.

Coordinates:
1. Input frame validation & preprocessing
2. Object detection (via Detector)
3. Depth map retrieval & distance estimation per bounding box (via BaseDepthProvider)
4. Proximity alert evaluation (via alerts.py)
5. Per-stage latency metrics tracking
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from vision_monitor.alerts import evaluate_detection_alerts

if TYPE_CHECKING:
    from vision_monitor.config import AlertSettings
    from vision_monitor.depth import BaseDepthProvider
    from vision_monitor.detector import Detector
    from vision_monitor.schemas import DetectionResult


@dataclass(frozen=True)
class PipelineOutput:
    """Consolidated output from a single frame run through the vision pipeline."""

    result: DetectionResult
    preprocess_time_ms: float
    inference_time_ms: float
    depth_time_ms: float
    total_time_ms: float
    is_depth_simulated: bool


class VisionPipeline:
    """Orchestrates image detection, depth estimation, and proximity alerting."""

    def __init__(
        self,
        detector: Detector,
        depth_provider: BaseDepthProvider,
        alert_settings: AlertSettings,
    ) -> None:
        self._detector = detector
        self._depth_provider = depth_provider
        self._alert_settings = alert_settings

    @property
    def detector(self) -> Detector:
        """Underlying object detector."""
        return self._detector

    @property
    def depth_provider(self) -> BaseDepthProvider:
        """Underlying depth estimation provider."""
        return self._depth_provider

    @property
    def alert_settings(self) -> AlertSettings:
        """Underlying proximity alert settings."""
        return self._alert_settings

    def process_frame(self, frame: np.ndarray) -> PipelineOutput:
        """Process a single video or camera frame through all pipeline stages."""
        t_start = time.perf_counter()

        # Stage 1: Preprocessing & validation
        t_pre_start = time.perf_counter()
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("Input frame must be a 3D NumPy array")
        t_pre_end = time.perf_counter()
        preprocess_time_ms = (t_pre_end - t_pre_start) * 1000.0

        # Stage 2: Inference
        result = self._detector.detect(frame)
        inference_time_ms = result.inference_time_ms

        # Stage 3: Depth Acquisition & Distance Estimation
        t_depth_start = time.perf_counter()
        depth_map = self._depth_provider.get_depth_map(frame)

        if depth_map is not None:
            for det in result.detections:
                det.distance_m = self._depth_provider.estimate_distance(det.bbox, depth_map)

        # Stage 4: Alert Evaluation
        evaluate_detection_alerts(result.detections, self._alert_settings.distance_threshold_m)

        t_depth_end = time.perf_counter()
        depth_time_ms = (t_depth_end - t_depth_start) * 1000.0

        t_end = time.perf_counter()
        total_time_ms = (t_end - t_start) * 1000.0

        return PipelineOutput(
            result=result,
            preprocess_time_ms=round(preprocess_time_ms, 2),
            inference_time_ms=round(inference_time_ms, 2),
            depth_time_ms=round(depth_time_ms, 2),
            total_time_ms=round(total_time_ms, 2),
            is_depth_simulated=self._depth_provider.is_simulated,
        )
