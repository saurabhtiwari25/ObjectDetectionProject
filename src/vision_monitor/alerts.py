from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vision_monitor.schemas import Detection


def evaluate_alert(distance_m: float | None, threshold_m: float) -> bool | None:

    if threshold_m <= 0:
        raise ValueError(f"Threshold must be positive, got {threshold_m}")

    if distance_m is None or not math.isfinite(distance_m) or distance_m < 0:
        return None

    return distance_m < threshold_m


def evaluate_detection_alerts(detections: list[Detection], threshold_m: float) -> list[Detection]:

    for det in detections:
        det.alert = evaluate_alert(det.distance_m, threshold_m)
    return detections
