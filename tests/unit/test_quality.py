"""Unit tests for detection quality evaluation (IoU, precision, recall, mAP)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vision_monitor.quality import (
    GroundTruth,
    compute_ap,
    compute_iou,
    compute_map,
    evaluate_quality,
    load_ground_truth,
    match_detections,
)
from vision_monitor.schemas import BoundingBox, Detection


class TestComputeIoU:
    def test_identical_boxes_iou_is_one(self):
        box = BoundingBox(x1=10, y1=10, x2=50, y2=50)
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_disjoint_boxes_iou_is_zero(self):
        box_a = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        box_b = BoundingBox(x1=20, y1=20, x2=30, y2=30)
        assert compute_iou(box_a, box_b) == 0.0

    def test_partial_overlap(self):
        # box_a: [0, 0, 10, 10] area = 100
        # box_b: [5, 0, 15, 10] area = 100
        # intersection: [5, 0, 10, 10] width=5, height=10 -> area = 50
        # union: 100 + 100 - 50 = 150
        # IoU = 50 / 150 = 1/3 ~ 0.3333
        box_a = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        box_b = BoundingBox(x1=5, y1=0, x2=15, y2=10)
        assert compute_iou(box_a, box_b) == pytest.approx(1 / 3, abs=1e-4)

    def test_touching_boxes_zero_iou(self):
        box_a = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        box_b = BoundingBox(x1=10, y1=0, x2=20, y2=10)
        assert compute_iou(box_a, box_b) == 0.0


class TestMatchDetections:
    def test_perfect_matches(self):
        preds = [
            Detection(
                class_name="person",
                confidence=0.9,
                bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50),
            )
        ]
        gts = [
            GroundTruth(
                class_name="person",
                bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50),
            )
        ]
        res = match_detections(preds, gts, iou_threshold=0.5)
        assert res.true_positives == 1
        assert res.false_positives == 0
        assert res.false_negatives == 0
        assert res.precision == 1.0
        assert res.recall == 1.0
        assert res.f1 == 1.0

    def test_class_mismatch_counted_as_false_positive(self):
        preds = [
            Detection(
                class_name="car",
                confidence=0.9,
                bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50),
            )
        ]
        gts = [
            GroundTruth(
                class_name="person",
                bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50),
            )
        ]
        res = match_detections(preds, gts, iou_threshold=0.5)
        assert res.true_positives == 0
        assert res.false_positives == 1
        assert res.false_negatives == 1
        assert res.precision == 0.0
        assert res.recall == 0.0

    def test_low_iou_below_threshold(self):
        preds = [
            Detection(
                class_name="person",
                confidence=0.9,
                bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
            )
        ]
        gts = [
            GroundTruth(
                class_name="person",
                bbox=BoundingBox(x1=5, y1=0, x2=15, y2=10),  # IoU is ~0.33
            )
        ]
        res = match_detections(preds, gts, iou_threshold=0.5)
        assert res.true_positives == 0
        assert res.false_positives == 1
        assert res.false_negatives == 1


class TestComputeAPAndMAP:
    def test_empty_predictions(self):
        gts = [
            GroundTruth(
                class_name="person",
                bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
            )
        ]
        ap = compute_ap([], gts, "person")
        assert ap.ap == 0.0
        assert ap.num_predictions == 0
        assert ap.num_ground_truths == 1

    def test_map_calculation(self):
        preds = [
            Detection(
                class_name="person",
                confidence=0.95,
                bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50),
            ),
            Detection(
                class_name="car",
                confidence=0.90,
                bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200),
            ),
        ]
        gts = [
            GroundTruth(
                class_name="person",
                bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50),
            ),
            GroundTruth(
                class_name="car",
                bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200),
            ),
        ]
        mean_ap, per_class = compute_map(preds, gts, iou_threshold=0.5)
        assert mean_ap == pytest.approx(1.0)
        assert len(per_class) == 2

    def test_11_point_interpolation_fractional_recall(self):
        # 10 ground truths, 3 detected perfectly -> recall is 0.3
        # Recall levels 0.0, 0.1, 0.2, 0.3 should each have precision 1.0 (4/11 ~ 0.3636)
        gts = [
            GroundTruth(
                class_name="person",
                bbox=BoundingBox(x1=i * 20, y1=0, x2=i * 20 + 10, y2=10),
            )
            for i in range(10)
        ]
        preds = [
            Detection(
                class_name="person",
                confidence=0.9 - i * 0.1,
                bbox=BoundingBox(x1=i * 20, y1=0, x2=i * 20 + 10, y2=10),
            )
            for i in range(3)
        ]
        ap_res = compute_ap(preds, gts, "person")
        assert ap_res.ap == pytest.approx(4.0 / 11.0, abs=1e-3)


class TestLoadGroundTruth:
    def test_load_valid_json(self, tmp_path: Path):
        ann_file = tmp_path / "test_ann.json"
        data = {
            "image": "test.jpg",
            "annotations": [
                {"class_name": "cat", "bbox": [10.0, 20.0, 30.0, 40.0]},
            ],
        }
        ann_file.write_text(json.dumps(data), encoding="utf-8")

        gts = load_ground_truth(ann_file)
        assert len(gts) == 1
        assert gts[0].class_name == "cat"
        assert gts[0].bbox.x1 == 10.0
        assert gts[0].bbox.y1 == 20.0
        assert gts[0].bbox.x2 == 30.0
        assert gts[0].bbox.y2 == 40.0


class TestEvaluateQuality:
    def test_complete_report(self):
        preds = [
            Detection(
                class_name="bottle",
                confidence=0.88,
                bbox=BoundingBox(x1=5, y1=5, x2=25, y2=25),
            )
        ]
        gts = [
            GroundTruth(
                class_name="bottle",
                bbox=BoundingBox(x1=5, y1=5, x2=25, y2=25),
            )
        ]
        report = evaluate_quality(preds, gts, iou_threshold=0.5, num_images=1)
        assert report.match.true_positives == 1
        assert report.mean_ap == pytest.approx(1.0)
        assert report.num_images == 1
