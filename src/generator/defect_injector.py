"""Defect Injector for synthetic PCB boards.
Injects 6 canonical IPC-A-610 defect classes with exact ground-truth bounding box coordinates.
"""
from __future__ import annotations

import random
from typing import Tuple
from PIL import Image, ImageDraw

from src.models.schemas import (
    DEFECT_NAME_TO_ID,
    BoundingBox,
    DefectClass,
)


class DefectInjector:
    """Injects canonical PCB defects onto a clean synthetic board."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def inject_defects(
        self,
        board_img: Image.Image,
        meta: dict,
        target_defects: list[DefectClass] | None = None,
        num_defects: int = 4,
    ) -> Tuple[Image.Image, list[dict]]:
        """Injects defects into a copy of the input board image.

        Args:
            board_img: Clean golden PCB PIL image.
            meta: Board features metadata (traces, vias, pads).
            target_defects: Specific defect types to inject (defaults to random subset).
            num_defects: Total count of defects to inject.

        Returns:
            (defective_board_image, list of ground_truth_defect_records).
        """
        img = board_img.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size

        # Mask substrate colors for carving away copper
        c_mask = board_img.getpixel((5, 5))
        c_copper = (212, 175, 55)
        c_copper_bright = (235, 195, 75)

        gt_records: list[dict] = []

        available_classes = target_defects or list(DefectClass)

        for _ in range(num_defects):
            defect_type = self.rng.choice(available_classes)

            if defect_type == DefectClass.MISSING_HOLE:
                rec = self._inject_missing_hole(draw, meta, w, h, c_copper_bright)
            elif defect_type == DefectClass.MOUSE_BITE:
                rec = self._inject_mouse_bite(draw, meta, w, h, c_mask)
            elif defect_type == DefectClass.OPEN_CIRCUIT:
                rec = self._inject_open_circuit(draw, meta, w, h, c_mask)
            elif defect_type == DefectClass.SHORT:
                rec = self._inject_short(draw, meta, w, h, c_copper)
            elif defect_type == DefectClass.SPUR:
                rec = self._inject_spur(draw, meta, w, h, c_copper)
            elif defect_type == DefectClass.SPURIOUS_COPPER:
                rec = self._inject_spurious_copper(draw, w, h, c_copper)
            else:
                continue

            if rec:
                gt_records.append(rec)

        return img, gt_records

    # ------------------------------------------------------------------
    # 0. Missing Hole: Annular copper pad with no drill hole inside
    # ------------------------------------------------------------------
    def _inject_missing_hole(self, draw: ImageDraw.ImageDraw, meta: dict, w: int, h: int, c_copper: tuple) -> dict | None:
        r = self.rng.choice([10, 12, 14])
        # Place on an existing via or in open routing zone
        if meta.get("vias") and self.rng.random() > 0.5:
            via = self.rng.choice(meta["vias"])
            cx, cy = via["x"], via["y"]
        else:
            cx = self.rng.randint(60, w - 60)
            cy = self.rng.randint(60, h - 60)

        # Draw solid copper circle with NO drill center
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=c_copper)

        pad_box = 4
        bbox = BoundingBox.from_xyxy(cx - r - pad_box, cy - r - pad_box, cx + r + pad_box, cy + r + pad_box, w, h)
        return {
            "class_id": DEFECT_NAME_TO_ID[DefectClass.MISSING_HOLE.value],
            "class_name": DefectClass.MISSING_HOLE.value,
            "bbox": bbox,
        }

    # ------------------------------------------------------------------
    # 1. Mouse Bite: Notch carved into edge of copper trace
    # ------------------------------------------------------------------
    def _inject_mouse_bite(self, draw: ImageDraw.ImageDraw, meta: dict, w: int, h: int, c_mask: tuple) -> dict | None:
        traces = [t for t in meta.get("traces", []) if t["width"] >= 4]
        if not traces:
            return None
        tr = self.rng.choice(traces)
        bite_r = self.rng.randint(4, 7)

        if tr["type"] == "horiz":
            bx = self.rng.randint(tr["x1"] + 20, tr["x2"] - 20)
            by = tr["y"] + (tr["width"] // 2 if self.rng.random() > 0.5 else -tr["width"] // 2)
        else:
            bx = tr["x"] + (tr["width"] // 2 if self.rng.random() > 0.5 else -tr["width"] // 2)
            by = self.rng.randint(tr["y1"] + 20, tr["y2"] - 20)

        # Carve bite with substrate color
        draw.ellipse([(bx - bite_r, by - bite_r), (bx + bite_r, by + bite_r)], fill=c_mask)

        pad_box = 4
        bbox = BoundingBox.from_xyxy(bx - bite_r - pad_box, by - bite_r - pad_box, bx + bite_r + pad_box, by + bite_r + pad_box, w, h)
        return {
            "class_id": DEFECT_NAME_TO_ID[DefectClass.MOUSE_BITE.value],
            "class_name": DefectClass.MOUSE_BITE.value,
            "bbox": bbox,
        }

    # ------------------------------------------------------------------
    # 2. Open Circuit: Complete break/gap cut across a trace
    # ------------------------------------------------------------------
    def _inject_open_circuit(self, draw: ImageDraw.ImageDraw, meta: dict, w: int, h: int, c_mask: tuple) -> dict | None:
        traces = meta.get("traces", [])
        if not traces:
            return None
        tr = self.rng.choice(traces)
        gap_size = self.rng.randint(8, 14)

        if tr["type"] == "horiz":
            cx = self.rng.randint(tr["x1"] + 30, tr["x2"] - 30)
            cy = tr["y"]
            draw.rectangle(
                [(cx - gap_size // 2, cy - tr["width"] - 2), (cx + gap_size // 2, cy + tr["width"] + 2)],
                fill=c_mask,
            )
            bbox = BoundingBox.from_xyxy(cx - gap_size // 2 - 4, cy - tr["width"] - 4, cx + gap_size // 2 + 4, cy + tr["width"] + 4, w, h)
        else:
            cx = tr["x"]
            cy = self.rng.randint(tr["y1"] + 30, tr["y2"] - 30)
            draw.rectangle(
                [(cx - tr["width"] - 2, cy - gap_size // 2), (cx + tr["width"] + 2, cy + gap_size // 2)],
                fill=c_mask,
            )
            bbox = BoundingBox.from_xyxy(cx - tr["width"] - 4, cy - gap_size // 2 - 4, cx + tr["width"] + 4, cy + gap_size // 2 + 4, w, h)

        return {
            "class_id": DEFECT_NAME_TO_ID[DefectClass.OPEN_CIRCUIT.value],
            "class_name": DefectClass.OPEN_CIRCUIT.value,
            "bbox": bbox,
        }

    # ------------------------------------------------------------------
    # 3. Short: Conductive bridge connecting two tracks or pads
    # ------------------------------------------------------------------
    def _inject_short(self, draw: ImageDraw.ImageDraw, meta: dict, w: int, h: int, c_copper: tuple) -> dict | None:
        # Find adjacent horizontal traces or pads
        h_traces = [t for t in meta.get("traces", []) if t["type"] == "horiz"]
        if len(h_traces) >= 2:
            t1 = self.rng.choice(h_traces)
            # Find closest adjacent trace
            others = [t for t in h_traces if t != t1]
            t2 = min(others, key=lambda t: abs(t["y"] - t1["y"]))
            if abs(t2["y"] - t1["y"]) <= 80:
                sx = self.rng.randint(max(t1["x1"], t2["x1"]) + 40, min(t1["x2"], t2["x2"]) - 40)
                y_min, y_max = min(t1["y"], t2["y"]), max(t1["y"], t2["y"])
                short_w = self.rng.randint(4, 7)
                draw.line([(sx, y_min), (sx, y_max)], fill=c_copper, width=short_w)
                pad_box = 4
                bbox = BoundingBox.from_xyxy(sx - short_w - pad_box, y_min - pad_box, sx + short_w + pad_box, y_max + pad_box, w, h)
                return {
                    "class_id": DEFECT_NAME_TO_ID[DefectClass.SHORT.value],
                    "class_name": DefectClass.SHORT.value,
                    "bbox": bbox,
                }

        # Fallback: draw localized bridge
        bx = self.rng.randint(100, w - 100)
        by = self.rng.randint(100, h - 100)
        draw.line([(bx, by), (bx + 18, by + 18)], fill=c_copper, width=5)
        bbox = BoundingBox.from_xyxy(bx - 4, by - 4, bx + 22, by + 22, w, h)
        return {
            "class_id": DEFECT_NAME_TO_ID[DefectClass.SHORT.value],
            "class_name": DefectClass.SHORT.value,
            "bbox": bbox,
        }

    # ------------------------------------------------------------------
    # 4. Spur: Unwanted burr/spike jutting out from trace edge
    # ------------------------------------------------------------------
    def _inject_spur(self, draw: ImageDraw.ImageDraw, meta: dict, w: int, h: int, c_copper: tuple) -> dict | None:
        traces = meta.get("traces", [])
        if not traces:
            return None
        tr = self.rng.choice(traces)
        spur_len = self.rng.randint(10, 18)
        spur_w = self.rng.randint(3, 5)

        if tr["type"] == "horiz":
            sx = self.rng.randint(tr["x1"] + 30, tr["x2"] - 30)
            sy = tr["y"]
            direction = 1 if self.rng.random() > 0.5 else -1
            draw.line([(sx, sy), (sx + self.rng.choice([-6, 0, 6]), sy + direction * spur_len)], fill=c_copper, width=spur_w)
            y_start = min(sy, sy + direction * spur_len)
            y_end = max(sy, sy + direction * spur_len)
            bbox = BoundingBox.from_xyxy(sx - 8, y_start - 4, sx + 8, y_end + 4, w, h)
        else:
            sx = tr["x"]
            sy = self.rng.randint(tr["y1"] + 30, tr["y2"] - 30)
            direction = 1 if self.rng.random() > 0.5 else -1
            draw.line([(sx, sy), (sx + direction * spur_len, sy + self.rng.choice([-6, 0, 6]))], fill=c_copper, width=spur_w)
            x_start = min(sx, sx + direction * spur_len)
            x_end = max(sx, sx + direction * spur_len)
            bbox = BoundingBox.from_xyxy(x_start - 4, sy - 8, x_end + 4, sy + 8, w, h)

        return {
            "class_id": DEFECT_NAME_TO_ID[DefectClass.SPUR.value],
            "class_name": DefectClass.SPUR.value,
            "bbox": bbox,
        }

    # ------------------------------------------------------------------
    # 5. Spurious Copper: Isolated un-etched copper fleck floating in space
    # ------------------------------------------------------------------
    def _inject_spurious_copper(self, draw: ImageDraw.ImageDraw, w: int, h: int, c_copper: tuple) -> dict:
        fx = self.rng.randint(70, w - 70)
        fy = self.rng.randint(70, h - 70)
        fw = self.rng.randint(8, 16)
        fh = self.rng.randint(8, 16)

        # Irregular polygon / blob
        shape_type = self.rng.choice(["ellipse", "rect", "triangle"])
        if shape_type == "ellipse":
            draw.ellipse([(fx, fy), (fx + fw, fy + fh)], fill=c_copper)
        elif shape_type == "rect":
            draw.rectangle([(fx, fy), (fx + fw, fy + fh)], fill=c_copper)
        else:
            draw.polygon([(fx, fy), (fx + fw, fy + fh // 2), (fx + fw // 3, fy + fh)], fill=c_copper)

        pad_box = 4
        bbox = BoundingBox.from_xyxy(fx - pad_box, fy - pad_box, fx + fw + pad_box, fy + fh + pad_box, w, h)
        return {
            "class_id": DEFECT_NAME_TO_ID[DefectClass.SPURIOUS_COPPER.value],
            "class_name": DefectClass.SPURIOUS_COPPER.value,
            "bbox": bbox,
        }
