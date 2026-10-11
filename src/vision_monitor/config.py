"""Application configuration using Pydantic settings.

Design decisions:
- Pydantic validates types at load time, so typos in .env files fail fast.
- All settings have safe defaults for Mode A (software-only, CPU, webcam).
- Settings are frozen after creation to prevent accidental mutation during runtime.
- The config object is created once and passed explicitly (no hidden global state).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CameraSettings(BaseModel):
    """Webcam / input source configuration."""

    index: int = Field(default=0, description="Webcam device index (0 = default)")
    width: int = Field(default=640, ge=160, le=3840)
    height: int = Field(default=480, ge=120, le=2160)


class DetectorSettings(BaseModel):
    """YOLO model configuration."""

    model_name: str = Field(
        default="yolo26n.pt",
        description="Ultralytics model identifier — downloaded on first use",
    )
    confidence: float = Field(
        default=0.25,
        ge=0.01,
        le=1.0,
        description="Minimum confidence threshold for detections",
    )
    image_size: int = Field(
        default=640,
        ge=32,
        le=1280,
        description="Input image size for YOLO (must be multiple of 32)",
    )
    device: str = Field(
        default="cpu",
        description="Inference device: 'cpu', 'cuda', 'cuda:0', 'mps', etc.",
    )


class AlertSettings(BaseModel):
    """Proximity alert configuration."""

    distance_threshold_m: float = Field(
        default=2.0,
        gt=0.0,
        description="Alert when object is closer than this distance (meters)",
    )


class AppSettings(BaseModel):
    """Top-level application settings — immutable after creation."""

    model_config = {"frozen": True}

    camera: CameraSettings = Field(default_factory=CameraSettings)
    detector: DetectorSettings = Field(default_factory=DetectorSettings)
    alerts: AlertSettings = Field(default_factory=AlertSettings)
    api_host: str = "0.0.0.0"
    api_port: int = 8000


def load_settings(**overrides) -> AppSettings:
    """Create settings, optionally overriding defaults.

    In later sprints this will read from .env files and environment variables.
    For now, explicit keyword overrides keep tests deterministic.
    """
    return AppSettings(**overrides)
