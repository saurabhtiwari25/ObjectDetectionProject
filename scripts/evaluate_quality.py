from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2

from vision_monitor.config import DetectorSettings
from vision_monitor.detector import Detector
from vision_monitor.quality import evaluate_quality, load_ground_truth

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run detection quality evaluation against ground-truth annotations."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("sample_data/bus.jpg"),
        help="Path to evaluation image (default: sample_data/bus.jpg)",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("sample_data/annotations/bus.json"),
        help="Path to ground-truth JSON annotations (default: sample_data/annotations/bus.json)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU matching threshold (default: 0.5)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Confidence cutoff for detector (default: 0.25)",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/quality_report.json"),
        help="Output path for evaluation JSON report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.image.is_file():
        logger.error("Image file not found: %s", args.image)
        sys.exit(1)

    if not args.annotations.is_file():
        logger.error("Annotations file not found: %s", args.annotations)
        sys.exit(1)

    print("=" * 65)
    print("  Smart Vision Monitor — Detection Quality Evaluation")
    print("=" * 65)
    print(f"Image       : {args.image}")
    print(f"Annotations : {args.annotations}")
    print(f"IoU Thresh  : {args.iou_threshold}")
    print(f"Conf Thresh : {args.conf_threshold}")
    print("-" * 65)

    frame = cv2.imread(str(args.image))
    if frame is None:
        logger.error("Failed to read image %s", args.image)
        sys.exit(1)

    ground_truths = load_ground_truth(args.annotations)
    print(f"Loaded {len(ground_truths)} ground-truth annotation(s):")
    for idx, gt in enumerate(ground_truths, 1):
        box_str = f"[{gt.bbox.x1:.1f}, {gt.bbox.y1:.1f}, {gt.bbox.x2:.1f}, {gt.bbox.y2:.1f}]"
        print(f"  {idx}. {gt.class_name:<10} {box_str}")

    detector_settings = DetectorSettings(confidence=args.conf_threshold)
    detector = Detector(detector_settings)
    detector.load()

    result = detector.detect(frame)
    print(f"\nModel detected {len(result.detections)} object(s):")
    for idx, det in enumerate(result.detections, 1):
        b = det.bbox
        box_str = f"[{b.x1:.1f}, {b.y1:.1f}, {b.x2:.1f}, {b.y2:.1f}]"
        print(f"  {idx}. {det.class_name:<10} (conf: {det.confidence:.2f}) {box_str}")

    report = evaluate_quality(
        predictions=result.detections,
        ground_truths=ground_truths,
        iou_threshold=args.iou_threshold,
        num_images=1,
    )

    print("\n" + "=" * 65)
    print("  Evaluation Results")
    print("=" * 65)
    m = report.match
    print(f"True Positives  (TP) : {m.true_positives}")
    print(f"False Positives (FP) : {m.false_positives}")
    print(f"False Negatives (FN) : {m.false_negatives}")
    print(f"Precision            : {m.precision * 100:.2f}%")
    print(f"Recall               : {m.recall * 100:.2f}%")
    print(f"F1-Score             : {m.f1:.4f}")
    print(f"mAP@{int(args.iou_threshold * 100)}              : {report.mean_ap:.4f}")
    print("-" * 65)
    print("Per-Class AP:")
    for cap in report.per_class_ap:
        if cap.num_ground_truths > 0:
            stats = f"(GTs: {cap.num_ground_truths}, Preds: {cap.num_predictions})"
            print(f"  {cap.class_name:<12} : AP = {cap.ap:.4f} {stats}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    report_dict = {
        "image": str(args.image),
        "iou_threshold": args.iou_threshold,
        "true_positives": m.true_positives,
        "false_positives": m.false_positives,
        "false_negatives": m.false_negatives,
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "mAP": report.mean_ap,
        "per_class": [
            {
                "class_name": c.class_name,
                "ap": c.ap,
                "num_predictions": c.num_predictions,
                "num_ground_truths": c.num_ground_truths,
            }
            for c in report.per_class_ap
            if c.num_ground_truths > 0
        ],
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    print(f"\nQuality report saved to: {args.out_json}")
    print("=" * 65)


if __name__ == "__main__":
    main()
