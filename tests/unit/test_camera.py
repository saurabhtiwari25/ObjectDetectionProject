"""Unit tests for camera input abstraction.

Tests the CameraSource with images and error handling.
No physical webcam required — we test with image files and mocks.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_monitor.camera import CameraSource
from vision_monitor.exceptions import CameraError


@pytest.fixture()
def sample_image_path(tmp_path: Path) -> Path:
    """Create a temporary test image on disk."""
    img = np.zeros((100, 150, 3), dtype=np.uint8)
    img[:, :] = (128, 64, 200)
    path = tmp_path / "test_img.jpg"
    cv2.imwrite(str(path), img)
    return path


class TestCameraSourceImage:
    """Test CameraSource with image files (no webcam needed)."""

    def test_open_image_file(self, sample_image_path: Path):
        with CameraSource(source=str(sample_image_path)) as cam:
            assert cam.is_open

    def test_read_image_returns_ndarray(self, sample_image_path: Path):
        with CameraSource(source=str(sample_image_path)) as cam:
            frame = cam.read()
            assert isinstance(frame, np.ndarray)
            assert frame.ndim == 3
            assert frame.shape[2] == 3

    def test_read_image_multiple_times(self, sample_image_path: Path):
        """Image source should loop — return the same frame every read."""
        with CameraSource(source=str(sample_image_path)) as cam:
            f1 = cam.read()
            f2 = cam.read()
            assert np.array_equal(f1, f2)

    def test_read_image_returns_copy(self, sample_image_path: Path):
        """Successive reads should return independent copies."""
        with CameraSource(source=str(sample_image_path)) as cam:
            f1 = cam.read()
            f2 = cam.read()
            f1[:] = 0  # Mutate first frame
            assert not np.array_equal(f1, f2)  # Second should be unaffected

    def test_release_closes_source(self, sample_image_path: Path):
        cam = CameraSource(source=str(sample_image_path))
        cam.open()
        assert cam.is_open
        cam.release()
        assert not cam.is_open


class TestCameraSourceErrors:
    """Test error handling."""

    def test_open_nonexistent_image_raises(self):
        with (
            pytest.raises(CameraError, match="Cannot read image"),
            CameraSource(source="nonexistent_image.jpg"),
        ):
            pass

    def test_read_before_open_raises(self):
        cam = CameraSource(source=0)
        with pytest.raises(CameraError, match="not open"):
            cam.read()

    def test_context_manager_releases_on_exception(self, sample_image_path: Path):
        """Ensure resources are freed even when body raises."""
        cam = CameraSource(source=str(sample_image_path))
        try:
            with cam:
                raise RuntimeError("Simulated crash")
        except RuntimeError:
            pass
        assert not cam.is_open


class TestCameraSourceIterator:
    """Test the iterator protocol."""

    def test_iterate_image_source(self, sample_image_path: Path):
        """Should be able to use `for frame in cam:` on image source."""
        with CameraSource(source=str(sample_image_path)) as cam:
            count = 0
            for frame in cam:
                assert isinstance(frame, np.ndarray)
                count += 1
                if count >= 3:
                    break
            assert count == 3
