"""Benchmark dataset generator for PCB Defect Detection platform.
Generates reproducible synthetic PCB boards with normalized YOLO ground-truth annotations.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.generator.pcb_synthesizer import PCBSynthesizer  # noqa: E402
from src.generator.defect_injector import DefectInjector  # noqa: E402
from src.models.schemas import DefectClass  # noqa: E402


def generate_benchmark_dataset(
    output_dir: Path | None = None,
    num_samples: int = 18,
    seed: int = 42,
) -> dict:
    """Generates synthetic PCB boards and YOLO annotations."""
    base_dir = output_dir or Path(__file__).parent
    img_dir = base_dir / "images"
    lbl_dir = base_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    synthesizer = PCBSynthesizer(width=800, height=800, seed=seed)
    injector = DefectInjector(seed=seed)

    manifest: list[dict] = []

    print(f"Generating {num_samples} benchmark PCB boards...")

    # Defect class combinations to guarantee coverage across all 6 classes
    classes_list = list(DefectClass)

    for i in range(1, num_samples + 1):
        board_name = f"pcb_sample_{i:03d}"
        img_path = img_dir / f"{board_name}.png"
        lbl_path = lbl_dir / f"{board_name}.txt"

        # Generate golden board
        golden_img, meta = synthesizer.generate_board()

        # Samples 1 to 4 are clean golden boards (Pass control group)
        if i <= 4:
            golden_img.save(img_path, format="PNG")
            lbl_path.write_text("", encoding="utf-8")  # empty label file = 0 defects
            manifest.append({
                "filename": f"{board_name}.png",
                "is_defective": False,
                "defects": [],
            })
            print(f"  [PASS] {board_name}.png (Clean Golden Board)")
        else:
            # Defective board
            # Cycle through specific defect types to guarantee full coverage
            target_class = classes_list[(i - 5) % len(classes_list)]
            num_defects = 2 if i % 2 == 0 else 3
            defective_img, gt_records = injector.inject_defects(
                golden_img,
                meta,
                target_defects=[target_class],
                num_defects=num_defects,
            )
            defective_img.save(img_path, format="PNG")

            # Write YOLO format label: <class_id> <cx> <cy> <w> <h>
            label_lines = []
            serializable_defects = []
            for r in gt_records:
                b = r["bbox"]
                label_lines.append(
                    f"{r['class_id']} {b.x_center_norm:.6f} {b.y_center_norm:.6f} {b.width_norm:.6f} {b.height_norm:.6f}"
                )
                serializable_defects.append({
                    "class_id": r["class_id"],
                    "class_name": r["class_name"],
                    "bbox": b.model_dump(),
                })

            lbl_path.write_text("\n".join(label_lines), encoding="utf-8")
            manifest.append({
                "filename": f"{board_name}.png",
                "is_defective": True,
                "defects": serializable_defects,
            })
            print(f"  [FAIL] {board_name}.png ({len(gt_records)} defects: {target_class.value})")

    # Save manifest
    manifest_file = base_dir / "ground_truth.json"
    manifest_file.write_text(json.dumps({"samples": manifest}, indent=2), encoding="utf-8")
    print(f"\nDataset generation complete. Saved to {base_dir}")
    return {"total": num_samples, "manifest": manifest_file}


if __name__ == "__main__":
    generate_benchmark_dataset()
