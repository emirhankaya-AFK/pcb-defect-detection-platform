"""Benchmark evaluation script for PCB Defect Detection & Quality Control Platform.
Measures Precision, Recall, F1 per class, Matched True-Positive Mean IoU, and inference latency (FPS).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.models.schemas import BoundingBox  # noqa: E402
from src.pipeline.inspector import PCBInspector  # noqa: E402
from src.vision.metrics import evaluate_detections  # noqa: E402
from dataset.generate_dataset import generate_benchmark_dataset  # noqa: E402


def run_benchmark():
    dataset_dir = ROOT / "dataset"
    manifest_path = dataset_dir / "ground_truth.json"

    # Ensure dataset is generated
    sample_1 = dataset_dir / "images" / "pcb_sample_001.png"
    if not manifest_path.exists() or not sample_1.exists():
        generate_benchmark_dataset(output_dir=dataset_dir, num_samples=18, seed=42)

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest_data.get("samples", [])

    inspector = PCBInspector(conf_threshold=0.45, iou_threshold=0.40)

    gt_all: list[list[dict]] = []
    pred_all: list[list[dict]] = []
    latencies: list[float] = []

    passed_correct = 0
    total_samples = len(samples)

    print("\n" + "=" * 78)
    print("  SYNTHETIC PCB AOI PROTOTYPE — BENCHMARK ACCURACY & EVALUATION REPORT")
    print("=" * 78)

    for s in samples:
        filename = s["filename"]
        img_path = dataset_dir / "images" / filename
        is_defective_gt = s["is_defective"]

        with Image.open(img_path) as img:
            t0 = time.perf_counter()
            res = inspector.inspect_image(img, filename=filename)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        # Ground truth boxes
        current_gt = []
        for d in s.get("defects", []):
            b_dict = d["bbox"]
            current_gt.append({
                "class_id": d["class_id"],
                "class_name": d["class_name"],
                "bbox": BoundingBox(**b_dict),
            })
        gt_all.append(current_gt)

        # Predicted boxes
        current_pred = []
        for det in res.defects:
            current_pred.append({
                "class_id": det.class_id,
                "class_name": det.class_name,
                "confidence": det.confidence,
                "bbox": det.bbox,
            })
        pred_all.append(current_pred)

        # Check Pass / Fail classification
        predicted_pass = "PASS" in res.pass_fail_status
        expected_pass = not is_defective_gt
        if predicted_pass == expected_pass:
            passed_correct += 1

    # Compute evaluation metrics
    mean_iou, precision, recall, f1, breakdown = evaluate_detections(
        ground_truths=gt_all,
        predictions=pred_all,
        iou_threshold=0.45,
    )

    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    fps = 1000.0 / avg_lat if avg_lat > 0 else 0.0
    qc_accuracy = (passed_correct / total_samples * 100.0) if total_samples > 0 else 0.0

    # Print Class Table
    print(f"\n{'CLASS NAME':<18} | {'PRECISION':<10} | {'RECALL':<8} | {'F1-SCORE':<9} | {'SUPPORT':<8}")
    print("-" * 65)
    for c_name, m in breakdown.items():
        grade = "🟢" if m.f1_score >= 0.70 else "🟡" if m.f1_score >= 0.40 else "⚪"
        print(f"{grade} {c_name:<16} | {m.precision:<10.2f} | {m.recall:<8.2f} | {m.f1_score:<9.2f} | {m.support:<8}")
    print("-" * 65)
    print(f"  {'OVERALL MICRO':<16} | {precision:<10.2f} | {recall:<8.2f} | {f1:<9.2f} | {sum(m.support for m in breakdown.values())}")

    print("\n" + "=" * 78)
    print("  OPERATIONAL & SPEED PERFORMANCE METRICS")
    print("=" * 78)
    print(f"  • Pass / Fail QC Accuracy          : {passed_correct}/{total_samples} ({qc_accuracy:.1f}%)")
    print(f"  • Matched True-Positive Mean IoU  : {mean_iou:.4f}")
    print(f"  • Average Inference Latency       : {avg_lat:.2f} ms")
    print(f"  • Throughput Capacity              : {fps:.1f} FPS (~{int(fps * 60)} boards/min)")
    print("=" * 78 + "\n")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "fps": fps,
        "qc_accuracy": qc_accuracy,
    }


if __name__ == "__main__":
    run_benchmark()
