"""Unit tests for spatial geometry and evaluation metrics: IoU, NMS, and evaluation runner."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.models.schemas import BoundingBox  # noqa: E402
from src.vision.metrics import calculate_iou, non_max_suppression, evaluate_detections  # noqa: E402


class TestIoUCalculation:
    def test_identical_boxes_give_iou_one(self):
        b1 = BoundingBox.from_xyxy(10, 10, 50, 50, 100, 100)
        b2 = BoundingBox.from_xyxy(10, 10, 50, 50, 100, 100)
        iou = calculate_iou(b1, b2)
        assert iou == pytest.approx(1.0)

    def test_disjoint_boxes_give_iou_zero(self):
        b1 = BoundingBox.from_xyxy(0, 0, 10, 10, 100, 100)
        b2 = BoundingBox.from_xyxy(50, 50, 80, 80, 100, 100)
        iou = calculate_iou(b1, b2)
        assert iou == 0.0

    def test_partial_overlap_calculation(self):
        # 20x20 boxes overlapping by 10x20 = 200 overlap, union = 400 + 400 - 200 = 600 -> IoU = 1/3
        b1 = BoundingBox.from_xyxy(0, 0, 20, 20, 100, 100)
        b2 = BoundingBox.from_xyxy(10, 0, 30, 20, 100, 100)
        iou = calculate_iou(b1, b2)
        assert iou == pytest.approx(200.0 / 600.0, abs=1e-4)


class TestNonMaxSuppression:
    def test_empty_input_returns_empty(self):
        assert non_max_suppression([], [], []) == []

    def test_suppresses_redundant_overlapping_box(self):
        # Two high-overlap boxes with same class -> keep highest score only
        b1 = BoundingBox.from_xyxy(10, 10, 50, 50, 100, 100)
        b2 = BoundingBox.from_xyxy(12, 12, 52, 52, 100, 100)
        boxes = [b1, b2]
        scores = [0.95, 0.70]
        classes = [0, 0]

        keep = non_max_suppression(boxes, scores, classes, iou_threshold=0.45)
        assert keep == [0]

    def test_preserves_non_overlapping_boxes(self):
        b1 = BoundingBox.from_xyxy(0, 0, 20, 20, 100, 100)
        b2 = BoundingBox.from_xyxy(60, 60, 80, 80, 100, 100)
        boxes = [b1, b2]
        scores = [0.80, 0.85]
        classes = [1, 2]

        keep = non_max_suppression(boxes, scores, classes, iou_threshold=0.45)
        assert len(keep) == 2


class TestEvaluateDetections:
    def test_perfect_detection_scores_one(self):
        b = BoundingBox.from_xyxy(10, 10, 30, 30, 100, 100)
        gt = [[{"class_name": "short", "bbox": b}]]
        pred = [[{"class_name": "short", "confidence": 0.90, "bbox": b}]]

        mean_iou, precision, recall, f1, breakdown = evaluate_detections(gt, pred, iou_threshold=0.5)
        assert precision == pytest.approx(1.0)
        assert recall == pytest.approx(1.0)
        assert f1 == pytest.approx(1.0)
        assert mean_iou == pytest.approx(1.0)
        assert breakdown["short"].true_positives == 1

    def test_false_positive_penalizes_precision(self):
        b1 = BoundingBox.from_xyxy(10, 10, 30, 30, 100, 100)
        b2 = BoundingBox.from_xyxy(60, 60, 80, 80, 100, 100)
        gt = [[{"class_name": "short", "bbox": b1}]]
        # 1 TP + 1 FP
        pred = [
            [
                {"class_name": "short", "confidence": 0.95, "bbox": b1},
                {"class_name": "short", "confidence": 0.80, "bbox": b2},
            ]
        ]
        _, precision, recall, _, _ = evaluate_detections(gt, pred, iou_threshold=0.5)
        assert precision == pytest.approx(0.5)
        assert recall == pytest.approx(1.0)
