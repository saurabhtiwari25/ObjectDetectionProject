from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from vision_monitor.exceptions import CameraError

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv"}


class CameraSource:

    def __init__(
        self,
        source: int | str | Path = 0,
        width: int = 640,
        height: int = 480,
    ) -> None:
        self._source = source
        self._width = width
        self._height = height
        self._cap: cv2.VideoCapture | None = None
        self._static_frame: np.ndarray | None = None  # For single-image mode
        self._is_image = False

    def open(self) -> None:
        source = self._source

        if isinstance(source, (str, Path)):
            ext = Path(source).suffix.lower()
            if ext in _IMAGE_EXTENSIONS:
                self._open_image(str(source))
                return

        cap = cv2.VideoCapture(int(source) if isinstance(source, int) else str(source))

        if not cap.isOpened():
            raise CameraError(f"Cannot open video source: {source}")

        if isinstance(source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

        self._cap = cap
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("Opened source=%s, resolution=%dx%d", source, actual_w, actual_h)

    def _open_image(self, path: str) -> None:
        img = cv2.imread(path)
        if img is None:
            raise CameraError(f"Cannot read image file: {path}")
        self._static_frame = img
        self._is_image = True
        h, w = img.shape[:2]
        logger.info("Opened image source=%s, resolution=%dx%d", path, w, h)

    def read(self) -> np.ndarray:
        if self._is_image and self._static_frame is not None:
            return self._static_frame.copy()

        if self._cap is None or not self._cap.isOpened():
            raise CameraError("Camera is not open.")

        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise CameraError("Failed to read frame (camera disconnected or video ended).")

        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera released.")
        self._static_frame = None
        self._is_image = False

    @property
    def is_open(self) -> bool:
        if self._is_image:
            return self._static_frame is not None
        return self._cap is not None and self._cap.isOpened()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        try:
            return self.read()
        except CameraError:
            raise StopIteration from None
