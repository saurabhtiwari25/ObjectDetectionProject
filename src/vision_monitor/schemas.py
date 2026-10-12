from __future__ import annotations

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):

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

    detections: list[Detection] = Field(default_factory=list)
    inference_time_ms: float = Field(ge=0.0, description="Model inference time in milliseconds")
    image_width: int = Field(ge=1)
    image_height: int = Field(ge=1)
    model_name: str = Field(description="Which model produced these detections")
