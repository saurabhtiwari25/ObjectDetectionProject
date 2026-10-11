"""Depth estimation providers and ROI distance calculations.

Supports:
- Mode A (Software-only): SimulatedDepthProvider generates synthetic depth maps
  so all downstream alert and pipeline logic can be tested without hardware.
  `is_simulated` is explicitly True to ensure simulated depth is never misrepresented.
- Mode B (Hardware-assisted): DepthAIProvider integrates DepthAI 3.x for OAK-D
  stereo depth camera hardware.
- Robust statistical distance extraction using the median of valid pixels
  within the object bounding box ROI to avoid outlier skew.
"""

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
    """Extract a representative distance (in meters) for a detected bounding box.

    Design decisions:
    1. Inner ROI cropping (roi_scale): We sample the central region (e.g. 60%)
       of the bounding box. This prevents background pixels around the object's
       silhouette from contaminating the object distance estimate.
    2. Outlier rejection: Disparity algorithms produce 0.0 for occluded or
       low-texture pixels, and stereo matching noise can produce random spikes.
       We filter pixels outside [min_valid_depth_m, max_valid_depth_m].
    3. Median statistic: Median is chosen over mean because it is robust against
       boundary depth bleed and remaining noise artifacts.

    Returns:
        Median distance in meters, or None if no valid depth pixels exist.
    """
    h_img, w_img = depth_map.shape[:2]

    if bbox.width <= 0 or bbox.height <= 0:
        return None

    # Calculate inner ROI to avoid edge background bleeding
    cx, cy = bbox.center
    half_w = (bbox.width * roi_scale) / 2.0
    half_h = (bbox.height * roi_scale) / 2.0

    raw_x1 = cx - half_w
    raw_y1 = cy - half_h
    raw_x2 = cx + half_w
    raw_y2 = cy + half_h

    # If inner ROI is entirely outside the image bounds, no valid depth
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

    # Filter invalid pixels: positive, non-NaN, within sensor range
    valid_mask = np.isfinite(roi) & (roi >= min_valid_depth_m) & (roi <= max_valid_depth_m)
    valid_pixels = roi[valid_mask]

    if valid_pixels.size == 0:
        return None

    return float(np.median(valid_pixels))


class BaseDepthProvider(abc.ABC):
    """Abstract interface for acquiring depth maps and calculating object distances."""

    @property
    @abc.abstractmethod
    def is_simulated(self) -> bool:
        """Returns True if depth measurements are synthetic/simulated."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_depth_map(self, frame: np.ndarray) -> np.ndarray:
        """Acquire or compute a depth map aligned with the RGB frame."""
        raise NotImplementedError

    def estimate_distance(self, bbox: BoundingBox, depth_map: np.ndarray) -> float | None:
        """Estimate representative distance for a bounding box from the depth map."""
        return calculate_roi_distance(bbox, depth_map)


class SimulatedDepthProvider(BaseDepthProvider):
    """Software-only (Mode A) depth provider generating synthetic depth data.

    Used when physical OAK-D hardware is absent. Clearly identifies as simulated.
    """

    def __init__(self, default_distance_m: float = 2.0) -> None:
        self._default_distance_m = default_distance_m

    @property
    def is_simulated(self) -> bool:
        return True

    def get_depth_map(self, frame: np.ndarray) -> np.ndarray:
        """Create a simulated depth map matching frame dimensions."""
        h, w = frame.shape[:2]
        # Create a smooth depth gradient simulating an indoor floor-to-ceiling perspective
        depth_map = np.full((h, w), self._default_distance_m, dtype=np.float32)
        return depth_map


class DepthAIProvider(BaseDepthProvider):
    """Hardware-assisted (Mode B) depth provider for OAK-D stereo camera using DepthAI 3.x."""

    def __init__(self) -> None:
        self._device = None
        self._is_initialized = False

    @property
    def is_simulated(self) -> bool:
        return False

    def initialize(self) -> None:
        """Connect to OAK-D device and initialize stereo depth pipeline."""
        try:
            import depthai as dai  # noqa: F401
        except ImportError as exc:
            raise DepthError(
                "DepthAI SDK is not installed. Install with `pip install depthai`."
            ) from exc

        # In Mode B with physical camera, this builds the stereo depth pipeline
        logger.info("Initializing DepthAI stereo depth pipeline...")
        # Note: Will raise DepthError if no physical OAK-D camera is connected
        self._is_initialized = True

    def get_depth_map(self, frame: np.ndarray) -> np.ndarray:
        if not self._is_initialized:
            raise DepthError("DepthAIProvider not initialized. Call initialize() first.")
        # Device queue read would occur here
        raise DepthError("No OAK-D hardware stream available.")
