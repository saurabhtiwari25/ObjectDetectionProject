"""YOLO object detector using Ultralytics.

Key concepts:
- The YOLO model is loaded ONCE and reused for every frame. Model loading is
  expensive (~1-3 seconds); inference is cheap (~20-80ms per frame on CPU).
- Ultralytics downloads pretrained weights on first use to ~/.cache/ultralytics.
- YOLO26n (Nano) is the smallest variant: ~6MB, ~80 COCO classes, optimized
  for speed over accuracy. Perfect for CPU-only real-time detection.
- The model internally uses PyTorch tensors. Input images are converted from
  NumPy (BGR, HWC) to PyTorch tensors (RGB, CHW, normalized) by Ultralytics.
- We return structured Detection objects, keeping this module independent
  of visualization, cameras, and depth estimation.

How YOLO works (simplified):
1. Input image is resized to a fixed grid (e.g., 640×640).
2. A CNN backbone extracts feature maps at multiple scales.
3. Detection heads predict bounding boxes + class probabilities per grid cell.
4. Non-Maximum Suppression (NMS) removes duplicate/overlapping boxes.
5. Final output: list of (class, confidence, x1, y1, x2, y2).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from ultralytics import YOLO

from vision_monitor.exceptions import ModelLoadError
from vision_monitor.schemas import BoundingBox, Detection, DetectionResult

if TYPE_CHECKING:
    from vision_monitor.config import DetectorSettings

logger = logging.getLogger(__name__)


class Detector:
    """Wraps an Ultralytics YOLO model for single-image inference.

    Usage:
        detector = Detector(settings)
        result = detector.detect(frame)
        for det in result.detections:
            print(det.class_name, det.confidence)
    """

    def __init__(self, settings: DetectorSettings) -> None:
        self._settings = settings
        self._model: YOLO | None = None
        self._model_name = settings.model_name

    def load(self) -> None:
        """Load model weights. Call once before detect().

        Separated from __init__ so callers can control when the expensive
        weight-loading happens (important for tests and API startup).
        """
        try:
            logger.info(
                "Loading YOLO model: %s (device=%s)",
                self._model_name,
                self._settings.device,
            )
            self._model = YOLO(self._model_name)
            # Warm-up: first inference is always slower due to graph compilation.
            # A single dummy pass ensures steady-state latency from the second call.
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self._model.predict(
                dummy,
                imgsz=64,
                conf=0.99,
                verbose=False,
                device=self._settings.device,
            )
            logger.info("Model loaded and warmed up successfully.")
        except Exception as exc:
            raise ModelLoadError(f"Failed to load model '{self._model_name}': {exc}") from exc

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Run inference on a single BGR image (NumPy array).

        Args:
            image: OpenCV-format image, shape (H, W, 3), dtype uint8, BGR color.

        Returns:
            DetectionResult with all detections above the confidence threshold.

        Raises:
            ModelLoadError: If the model hasn't been loaded yet.
            ValueError: If the input image is invalid.
        """
        if self._model is None:
            raise ModelLoadError("Model not loaded. Call detector.load() first.")

        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected BGR image with shape (H, W, 3), got: "
                f"type={type(image).__name__}, shape={getattr(image, 'shape', 'N/A')}"
            )

        h, w = image.shape[:2]

        # time.perf_counter() uses the highest-resolution monotonic clock.
        # On CPU this is sufficient. For GPU we'd need torch.cuda.synchronize().
        t_start = time.perf_counter()
        results = self._model.predict(
            image,
            imgsz=self._settings.image_size,
            conf=self._settings.confidence,
            verbose=False,
            device=self._settings.device,
        )
        t_end = time.perf_counter()

        inference_ms = (t_end - t_start) * 1000.0

        detections = self._parse_results(results)

        return DetectionResult(
            detections=detections,
            inference_time_ms=round(inference_ms, 2),
            image_width=w,
            image_height=h,
            model_name=self._model_name,
        )

    def _parse_results(self, results) -> list[Detection]:
        """Convert Ultralytics Results objects to our Detection schema.

        Ultralytics returns a list of Results objects (one per image in a batch).
        Each result.boxes contains: xyxy coordinates, confidence, and class id.
        Class names are stored in result.names (dict: id → name).
        """
        detections: list[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            names = result.names  # {0: 'person', 1: 'bicycle', ...}

            for i in range(len(boxes)):
                # .xyxy gives (x1, y1, x2, y2) in pixel coords, shape (1, 4)
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                cls_name = names.get(cls_id, f"class_{cls_id}")

                detections.append(
                    Detection(
                        class_name=cls_name,
                        confidence=round(conf, 4),
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    )
                )

        return detections
