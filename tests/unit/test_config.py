"""Unit tests for configuration validation.

Tests that settings enforce constraints (ranges, types) and that
defaults produce a working Mode A configuration.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vision_monitor.config import (
    AlertSettings,
    AppSettings,
    CameraSettings,
    DetectorSettings,
    load_settings,
)


class TestCameraSettings:
    def test_defaults(self):
        s = CameraSettings()
        assert s.index == 0
        assert s.width == 640
        assert s.height == 480

    def test_custom_resolution(self):
        s = CameraSettings(width=1280, height=720)
        assert s.width == 1280
        assert s.height == 720

    def test_invalid_width_rejected(self):
        with pytest.raises(ValidationError):
            CameraSettings(width=10)  # Below minimum 160

    def test_invalid_height_rejected(self):
        with pytest.raises(ValidationError):
            CameraSettings(height=5000)  # Above maximum 2160


class TestDetectorSettings:
    def test_defaults(self):
        s = DetectorSettings()
        assert s.model_name == "yolo26n.pt"
        assert s.confidence == 0.25
        assert s.image_size == 640
        assert s.device == "cpu"

    def test_confidence_boundaries(self):
        DetectorSettings(confidence=0.01)  # min
        DetectorSettings(confidence=1.0)  # max
        with pytest.raises(ValidationError):
            DetectorSettings(confidence=0.0)  # below min
        with pytest.raises(ValidationError):
            DetectorSettings(confidence=1.1)

    def test_image_size_boundaries(self):
        DetectorSettings(image_size=32)  # min
        DetectorSettings(image_size=1280)  # max
        with pytest.raises(ValidationError):
            DetectorSettings(image_size=16)
        with pytest.raises(ValidationError):
            DetectorSettings(image_size=2048)


class TestAlertSettings:
    def test_defaults(self):
        s = AlertSettings()
        assert s.distance_threshold_m == 2.0

    def test_zero_distance_rejected(self):
        with pytest.raises(ValidationError):
            AlertSettings(distance_threshold_m=0.0)

    def test_negative_distance_rejected(self):
        with pytest.raises(ValidationError):
            AlertSettings(distance_threshold_m=-1.0)


class TestAppSettings:
    def test_defaults_produce_valid_config(self):
        s = AppSettings()
        assert s.camera.index == 0
        assert s.detector.model_name == "yolo26n.pt"
        assert s.alerts.distance_threshold_m == 2.0

    def test_frozen(self):
        """Settings must be immutable after creation."""
        s = AppSettings()
        with pytest.raises(ValidationError):
            s.api_port = 9999


class TestLoadSettings:
    def test_load_defaults(self):
        s = load_settings()
        assert isinstance(s, AppSettings)

    def test_load_with_overrides(self):
        s = load_settings(
            detector={"model_name": "yolo26s.pt", "confidence": 0.5},
            api_port=9000,
        )
        assert s.detector.model_name == "yolo26s.pt"
        assert s.detector.confidence == 0.5
        assert s.api_port == 9000
