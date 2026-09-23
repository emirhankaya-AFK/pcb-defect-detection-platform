"""Evaluation metrics and spatial geometry utilities: IoU, NMS, and Precision/Recall/F1 calculation."""
from __future__ import annotations

from typing import List, Tuple
from src.models.schemas import BoundingBox, ClassMetrics, DEFECT_CLASS_MAP


def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(box1.x_min, box2.x_min)
    y1 = max(box1.y_min, box2.y_min)
    x2 = min(box1.x_max, box2.x_max)
    y2 = min(box1.y_max, box2.y_max)

    intersection_w = max(0, x2 - x1)
    intersection_h = max(0, y2 - y1)
    intersection_area = intersection_w * intersection_h

    area1 = (box1.x_max - box1.x_min) * (box1.y_max - box1.y_min)
    area2 = (box2.x_max - box2.x_min) * (box2.y_max - box2.y_min)
    union_area = area1 + area2 - intersection_area

    if union_area <= 0:
        return 0.0

    return round(float(intersection_area / union_area), 6)


def non_max_suppression(
    boxes: List[BoundingBox],
    scores: List[float],
    class_ids: List[int],
    iou_threshold: float = 0.45,
) -> List[int]:
    """Applies Non-Maximum Suppression (NMS) to eliminate duplicate overlapping bounding boxes.

    Returns:
        List of surviving indices sorted by confidence score.
    """
    if not boxes:
        return []

    # Sort indices by descending score
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep: List[int] = []

    while order:
        current = order.pop(0)
        keep.append(current)

        remaining: List[int] = []
        for other in order:
            # Only suppress if same class or high general overlap
            if class_ids[current] == class_ids[other]:
                iou = calculate_iou(boxes[current], boxes[other])
                if iou < iou_threshold:
                    remaining.append(other)
            else:
                remaining.append(other)
        order = remaining

    return keep


def evaluate_detections(
    ground_truths: List[List[dict]],
    predictions: List[List[dict]],
    iou_threshold: float = 0.50,
) -> Tuple[float, float, float, float, dict[str, ClassMetrics]]:
    """Evaluates predicted defect boxes against ground-truth annotations across images.

    Returns:
        (mean_iou, precision, recall, f1, per_class_breakdown).
    """
    per_class: dict[str, dict] = {
        cls_enum.value: {"tp": 0, "fp": 0, "fn": 0, "ious": []}
        for cls_enum in DEFECT_CLASS_MAP.values()
    }

    all_matched_ious: list[float] = []

    for gt_list, pred_list in zip(ground_truths, predictions):
        matched_gt: set[int] = set()

        # Sort predictions by confidence descending
        sorted_preds = sorted(pred_list, key=lambda p: p.get("confidence", 0.0), reverse=True)

        for pred in sorted_preds:
            pred_class = pred["class_name"]
            pred_box = pred["bbox"]

            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gt_list):
                if gt_idx in matched_gt:
                    continue
                if gt["class_name"] != pred_class:
                    continue

                iou = calculate_iou(pred_box, gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                per_class[pred_class]["tp"] += 1
                per_class[pred_class]["ious"].append(best_iou)
                all_matched_ious.append(best_iou)
            else:
                per_class[pred_class]["fp"] += 1

        # Any un-matched ground truth is a False Negative
        for gt_idx, gt in enumerate(gt_list):
            if gt_idx not in matched_gt:
                per_class[gt["class_name"]]["fn"] += 1

    # Aggregate overall metrics
    total_tp = sum(data["tp"] for data in per_class.values())
    total_fp = sum(data["fp"] for data in per_class.values())
    total_fn = sum(data["fn"] for data in per_class.values())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = sum(all_matched_ious) / len(all_matched_ious) if all_matched_ious else 0.0

    breakdown: dict[str, ClassMetrics] = {}
    for name, data in per_class.items():
        tp, fp, fn = data["tp"], data["fp"], data["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f_score = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        breakdown[name] = ClassMetrics(
            class_name=name,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=round(p, 4),
            recall=round(r, 4),
            f1_score=round(f_score, 4),
            support=tp + fn,
        )

    return (
        round(mean_iou, 4),
        round(precision, 4),
        round(recall, 4),
        round(f1, 4),
        breakdown,
    )
