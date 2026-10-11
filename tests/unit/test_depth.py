"""Unit tests for depth estimation and providers (TDD).

Tests:
- Abstract DepthProvider protocol
- SimulatedDepthProvider returns expected simulated depth map
- Robust median calculation over bounding box ROI
- Handling NaN, 0.0 (invalid depth), and out-of-range pixels
- Handling empty or out-of-bounds bounding boxes
"""

from __future__ import annotations

import numpy as np
import pytest

from vision_monitor.depth import (
    BaseDepthProvider,
    SimulatedDepthProvider,
    calculate_roi_distance,
)
from vision_monitor.schemas import BoundingBox


class TestRoiDistanceCalculation:
    """Test robust distance extraction from depth maps."""

    def test_uniform_depth_returns_exact_value(self):
        depth_map = np.full((100, 100), 2.5, dtype=np.float32)
        bbox = BoundingBox(x1=10, y1=10, x2=50, y2=50)

        distance = calculate_roi_distance(bbox, depth_map)
        assert distance == pytest.approx(2.5, rel=1e-2)

    def test_median_ignores_outliers(self):
        # 10x10 patch mostly 2.0m, with a few extreme noise pixels (0.1m, 10.0m)
        depth_map = np.full((10, 10), 2.0, dtype=np.float32)
        depth_map[2, 2] = 0.2
        depth_map[3, 3] = 9.9
        bbox = BoundingBox(x1=0, y1=0, x2=10, y2=10)

        distance = calculate_roi_distance(bbox, depth_map)
        assert distance == pytest.approx(2.0, rel=1e-2)

    def test_ignores_zeros_and_nans(self):
        # 0 in stereo depth indicates invalid disparity / no stereo correlation
        depth_map = np.zeros((10, 10), dtype=np.float32)
        depth_map[0:5, 0:5] = np.nan
        depth_map[5:10, 5:10] = 3.2  # Only valid region
        bbox = BoundingBox(x1=0, y1=0, x2=10, y2=10)

        distance = calculate_roi_distance(bbox, depth_map)
        assert distance == pytest.approx(3.2, rel=1e-2)

    def test_all_invalid_returns_none(self):
        depth_map = np.zeros((10, 10), dtype=np.float32)
        bbox = BoundingBox(x1=0, y1=0, x2=10, y2=10)

        distance = calculate_roi_distance(bbox, depth_map)
        assert distance is None

    def test_out_of_bounds_clipped_safely(self):
        depth_map = np.full((100, 100), 1.8, dtype=np.float32)
        # Box extends beyond image boundaries
        bbox = BoundingBox(x1=-20, y1=-10, x2=120, y2=150)

        distance = calculate_roi_distance(bbox, depth_map)
        assert distance == pytest.approx(1.8, rel=1e-2)

    def test_completely_off_screen_box_returns_none(self):
        depth_map = np.full((100, 100), 1.8, dtype=np.float32)
        bbox = BoundingBox(x1=-100, y1=-100, x2=-50, y2=-50)
        assert calculate_roi_distance(bbox, depth_map) is None

    def test_zero_area_box_returns_none(self):
        depth_map = np.full((100, 100), 1.8, dtype=np.float32)
        bbox = BoundingBox(x1=10, y1=10, x2=10, y2=10)
        assert calculate_roi_distance(bbox, depth_map) is None


class TestSimulatedDepthProvider:
    """Test simulated depth provider behavior."""

    def test_provider_initialization(self):
        provider = SimulatedDepthProvider(default_distance_m=2.5)
        assert isinstance(provider, BaseDepthProvider)
        assert provider.is_simulated is True

    def test_generates_depth_map_matching_frame_shape(self):
        provider = SimulatedDepthProvider(default_distance_m=2.0)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        depth_map = provider.get_depth_map(frame)
        assert depth_map.shape == (480, 640)
        assert depth_map.dtype == np.float32
        assert np.all(depth_map >= 0.0)

    def test_estimate_distance_for_detections(self):
        provider = SimulatedDepthProvider(default_distance_m=1.8)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        depth_map = provider.get_depth_map(frame)
        bbox = BoundingBox(x1=10, y1=10, x2=50, y2=50)

        dist = provider.estimate_distance(bbox, depth_map)
        assert dist is not None
        assert dist > 0.0
