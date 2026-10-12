from __future__ import annotations


class VisionMonitorError(Exception):
    """Base exception for all Smart Vision Monitor errors."""


class ModelLoadError(VisionMonitorError):
    """Failed to load or initialize a detection model."""


class CameraError(VisionMonitorError):
    """Failed to open, read from, or release a camera device."""


class DepthError(VisionMonitorError):
    """Failed to acquire or process depth data."""


class ConfigError(VisionMonitorError):
    """Invalid or missing configuration."""
