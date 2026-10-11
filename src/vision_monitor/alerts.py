"""Proximity alert evaluation component.

Design decisions:
- Functions in this module are pure functions: deterministic, no side effects,
  no dependencies on camera hardware or deep learning models.
- Never confuse missing depth (None) with a distant object (False). If distance
  is None, alert is None.
- A proximity alert is triggered (True) strictly when measured distance_m < threshold_m.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vision_monitor.schemas import Detection


def evaluate_alert(distance_m: float | None, threshold_m: float) -> bool | None:
    """Evaluate whether a single distance measurement triggers a proximity alert.

    Args:
        distance_m: Estimated distance in meters, or None if unavailable/invalid.
        threshold_m: Distance threshold in meters. Must be strictly positive.

    Returns:
        True: Object is closer than threshold_m.
        False: Object is at or further than threshold_m.
        None: Distance data is missing/invalid (cannot determine alert state).

    Raises:
        ValueError: If threshold_m <= 0.
    """
    if threshold_m <= 0:
        raise ValueError(f"Threshold must be positive, got {threshold_m}")

    if distance_m is None or not math.isfinite(distance_m) or distance_m < 0:
        return None

    return distance_m < threshold_m


def evaluate_detection_alerts(detections: list[Detection], threshold_m: float) -> list[Detection]:
    """Evaluate proximity alerts across a list of detections.

    Updates each detection's `alert` field in-place and returns the list.
    """
    for det in detections:
        det.alert = evaluate_alert(det.distance_m, threshold_m)
    return detections
