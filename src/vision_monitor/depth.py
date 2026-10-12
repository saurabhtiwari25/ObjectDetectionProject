from __future__ import annotations

import abc
import logging
from typing import TYPE_CHECKING

import numpy as np

from vision_monitor.exceptions import DepthError

if TYPE_CHECKING:
    from vision_monitor.schemas import BoundingBox

logger = logging.getLogger(__name__)


def calculate_roi_distance(
    bbox: BoundingBox,
    depth_map: np.ndarray,
    min_valid_depth_m: float = 0.2,
    max_valid_depth_m: float = 15.0,
    roi_scale: float = 0.6,
) -> float | None:

    h_img, w_img = depth_map.shape[:2]

    if bbox.width <= 0 or bbox.height <= 0:
        return None

    cx, cy = bbox.center
    half_w = (bbox.width * roi_scale) / 2.0
    half_h = (bbox.height * roi_scale) / 2.0

    raw_x1 = cx - half_w
    raw_y1 = cy - half_h
    raw_x2 = cx + half_w
    raw_y2 = cy + half_h

    if raw_x2 <= 0 or raw_y2 <= 0 or raw_x1 >= w_img or raw_y1 >= h_img:
        return None

    x1 = int(np.clip(raw_x1, 0, w_img))
    y1 = int(np.clip(raw_y1, 0, h_img))
    x2 = int(np.clip(raw_x2, 0, w_img))
    y2 = int(np.clip(raw_y2, 0, h_img))

    if x2 <= x1 or y2 <= y1:
        return None

    roi = depth_map[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    valid_mask = np.isfinite(roi) & (roi >= min_valid_depth_m) & (roi <= max_valid_depth_m)
    valid_pixels = roi[valid_mask]

    if valid_pixels.size == 0:
        return None

    return float(np.median(valid_pixels))


class BaseDepthProvider(abc.ABC):

    @property
    @abc.abstractmethod
    def is_simulated(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def get_depth_map(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def estimate_distance(self, bbox: BoundingBox, depth_map: np.ndarray) -> float | None:
        return calculate_roi_distance(bbox, depth_map)


class SimulatedDepthProvider(BaseDepthProvider):

    def __init__(self, default_distance_m: float = 2.0) -> None:
        self._default_distance_m = default_distance_m

    @property
    def is_simulated(self) -> bool:
        return True

    def get_depth_map(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        depth_map = np.full((h, w), self._default_distance_m, dtype=np.float32)
        return depth_map


class DepthAIProvider(BaseDepthProvider):

    def __init__(self) -> None:
        self._device = None
        self._is_initialized = False

    @property
    def is_simulated(self) -> bool:
        return False

    def initialize(self) -> None:
        try:
            import depthai as dai  
        except ImportError as exc:
            raise DepthError(
                "DepthAI SDK is not installed. Install with `pip install depthai`."
            ) from exc

        logger.info("Initializing DepthAI stereo depth pipeline...")
        self._is_initialized = True

    def get_depth_map(self, frame: np.ndarray) -> np.ndarray:
        if not self._is_initialized:
            raise DepthError("DepthAIProvider not initialized. Call initialize() first.")
        raise DepthError("No OAK-D hardware stream available.")
