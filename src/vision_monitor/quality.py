"""Detection quality evaluation metrics.

Computes standard object detection quality metrics:
- IoU (Intersection over Union) between predicted and ground-truth boxes
- Precision, Recall, F1-score at a given IoU threshold
- AP (Average Precision) per class via precision-recall curve
- mAP (mean Average Precision) across all classes

Ground-truth format (JSON):
    {
        "image": "bus.jpg",
        "annotations": [
            {"class_name": "bus", "bbox": [x1, y1, x2, y2]},
            {"class_name": "person", "bbox": [x1, y1, x2, y2]}
        ]
    }
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vision_monitor.schemas import BoundingBox, Detection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroundTruth:
    """A single ground-truth annotation."""

    class_name: str
    bbox: BoundingBox


@dataclass(frozen=True)
class MatchResult:
    """Result of matching predictions against ground truth."""

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    iou_threshold: float


@dataclass
class ClassAP:
    """Average Precision for a single class."""

    class_name: str
    ap: float
    precision_values: list[float] = field(default_factory=list)
    recall_values: list[float] = field(default_factory=list)
    num_predictions: int = 0
    num_ground_truths: int = 0


@dataclass(frozen=True)
class QualityReport:
    """Complete quality evaluation report."""

    match: MatchResult
    per_class_ap: list[ClassAP]
    mean_ap: float
    num_images: int


def compute_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Compute Intersection over Union between two bounding boxes.

    IoU = Area(Intersection) / Area(Union)
    Range: 0.0 (no overlap) to 1.0 (perfect overlap).
    """
    x1 = max(box_a.x1, box_b.x1)
    y1 = max(box_a.y1, box_b.y1)
    x2 = min(box_a.x2, box_b.x2)
    y2 = min(box_a.y2, box_b.y2)

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    union = box_a.area + box_b.area - intersection
    if union <= 0:
        return 0.0

    return min(1.0, max(0.0, float(intersection / union)))


def match_detections(
    predictions: list[Detection],
    ground_truths: list[GroundTruth],
    iou_threshold: float = 0.5,
) -> MatchResult:
    """Match predicted detections to ground-truth annotations.

    Uses greedy matching: predictions sorted by confidence (highest first).
    Each ground truth can only be matched once.
    A match requires: same class AND IoU >= threshold.
    """
    # Sort predictions by confidence descending
    sorted_preds = sorted(predictions, key=lambda d: d.confidence, reverse=True)
    matched_gt: set[int] = set()

    tp = 0
    fp = 0

    for pred in sorted_preds:
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue
            if pred.class_name != gt.class_name:
                continue

            iou = compute_iou(pred.bbox, gt.bbox)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(ground_truths) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return MatchResult(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        iou_threshold=iou_threshold,
    )


def compute_ap(
    predictions: list[Detection],
    ground_truths: list[GroundTruth],
    class_name: str,
    iou_threshold: float = 0.5,
) -> ClassAP:
    """Compute Average Precision for a single class.

    Uses the 11-point interpolation method (PASCAL VOC style).
    """
    class_preds = sorted(
        [p for p in predictions if p.class_name == class_name],
        key=lambda d: d.confidence,
        reverse=True,
    )
    class_gts = [g for g in ground_truths if g.class_name == class_name]

    if not class_gts:
        return ClassAP(
            class_name=class_name,
            ap=0.0,
            num_predictions=len(class_preds),
            num_ground_truths=0,
        )

    if not class_preds:
        return ClassAP(
            class_name=class_name,
            ap=0.0,
            precision_values=[],
            recall_values=[],
            num_predictions=0,
            num_ground_truths=len(class_gts),
        )

    matched_gt: set[int] = set()
    tp_list: list[int] = []
    fp_list: list[int] = []

    for pred in class_preds:
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(class_gts):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou(pred.bbox, gt.bbox)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp_list.append(1)
            fp_list.append(0)
            matched_gt.add(best_gt_idx)
        else:
            tp_list.append(0)
            fp_list.append(1)

    # Cumulative sums
    tp_cumsum = np.cumsum(tp_list).astype(float)
    fp_cumsum = np.cumsum(fp_list).astype(float)

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / len(class_gts)

    # 11-point interpolation (with small epsilon to avoid IEEE-754 precision drops)
    ap = 0.0
    recall_levels = np.linspace(0, 1, 11)
    for r_level in recall_levels:
        prec_at_recall = precisions[recalls >= (r_level - 1e-6)]
        if len(prec_at_recall) > 0:
            ap += float(np.max(prec_at_recall))
    ap /= 11.0

    return ClassAP(
        class_name=class_name,
        ap=round(ap, 4),
        precision_values=precisions.tolist(),
        recall_values=recalls.tolist(),
        num_predictions=len(class_preds),
        num_ground_truths=len(class_gts),
    )


def compute_map(
    predictions: list[Detection],
    ground_truths: list[GroundTruth],
    iou_threshold: float = 0.5,
) -> tuple[float, list[ClassAP]]:
    """Compute mean Average Precision across all classes.

    Returns (mAP, list of per-class AP results).
    """
    all_classes = set()
    for gt in ground_truths:
        all_classes.add(gt.class_name)
    for pred in predictions:
        all_classes.add(pred.class_name)

    per_class: list[ClassAP] = []
    for cls in sorted(all_classes):
        ap = compute_ap(predictions, ground_truths, cls, iou_threshold)
        per_class.append(ap)

    # mAP is computed only over classes that have ground truths
    gt_classes = [c for c in per_class if c.num_ground_truths > 0]
    mean_ap = (
        sum(c.ap for c in gt_classes) / len(gt_classes) if gt_classes else 0.0
    )

    return round(mean_ap, 4), per_class


def load_ground_truth(annotation_path: Path | str) -> list[GroundTruth]:
    """Load ground-truth annotations from a JSON file."""
    path = Path(annotation_path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    annotations = []
    for ann in data.get("annotations", []):
        bbox_coords = ann["bbox"]  # [x1, y1, x2, y2]
        annotations.append(
            GroundTruth(
                class_name=ann["class_name"],
                bbox=BoundingBox(
                    x1=bbox_coords[0],
                    y1=bbox_coords[1],
                    x2=bbox_coords[2],
                    y2=bbox_coords[3],
                ),
            )
        )
    return annotations


def evaluate_quality(
    predictions: list[Detection],
    ground_truths: list[GroundTruth],
    iou_threshold: float = 0.5,
    num_images: int = 1,
) -> QualityReport:
    """Run full quality evaluation: matching + mAP."""
    match = match_detections(predictions, ground_truths, iou_threshold)
    mean_ap, per_class_ap = compute_map(predictions, ground_truths, iou_threshold)

    return QualityReport(
        match=match,
        per_class_ap=per_class_ap,
        mean_ap=mean_ap,
        num_images=num_images,
    )
