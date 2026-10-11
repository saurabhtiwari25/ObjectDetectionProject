"""Shared data models (schemas) for detection results.

Design decisions:
- These are plain Pydantic models — no business logic, just data shape.
- Separating schemas from logic means detector.py, alerts.py, and api.py
  all share the same contract without importing each other.
- Optional fields (distance_m, alert) start as None. Later sprints
  will populate them from depth estimation and alert logic.
- BoundingBox uses (x1, y1, x2, y2) format — top-left and bottom-right
  corners in pixel coordinates. This matches YOLO's output format and
  OpenCV's rectangle drawing convention.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates.

    (x1, y1) = top-left corner
    (x2, y2) = bottom-right corner
    Origin is top-left of the image, x increases right, y increases down.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


class Detection(BaseModel):
    """A single detected object — the core data unit flowing through the pipeline."""

    class_name: str = Field(description="Human-readable class label, e.g. 'person'")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence score")
    bbox: BoundingBox = Field(description="Bounding box in pixel coordinates")
    distance_m: float | None = Field(
        default=None,
        description="Estimated distance in meters. None = depth unavailable.",
    )
    alert: bool | None = Field(
        default=None,
        description="True if object is within alert threshold. None = not evaluated.",
    )


class DetectionResult(BaseModel):
    """Complete result from one frame of inference."""

    detections: list[Detection] = Field(default_factory=list)
    inference_time_ms: float = Field(ge=0.0, description="Model inference time in milliseconds")
    image_width: int = Field(ge=1)
    image_height: int = Field(ge=1)
    model_name: str = Field(description="Which model produced these detections")
