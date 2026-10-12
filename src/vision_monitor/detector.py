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

    def __init__(self, settings: DetectorSettings) -> None:
        self._settings = settings
        self._model: YOLO | None = None
        self._model_name = settings.model_name

    def load(self) -> None:
        try:
            logger.info(
                "Loading YOLO model: %s (device=%s)",
                self._model_name,
                self._settings.device,
            )
            self._model = YOLO(self._model_name)
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
        if self._model is None:
            raise ModelLoadError("Model not loaded. Call detector.load() first.")

        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected BGR image with shape (H, W, 3), got: "
                f"type={type(image).__name__}, shape={getattr(image, 'shape', 'N/A')}"
            )

        h, w = image.shape[:2]

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
        detections: list[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            names = result.names

            for i in range(len(boxes)):
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
