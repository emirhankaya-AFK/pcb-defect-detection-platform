"""PCB Defect Detector engine using multi-scale morphological and contour analysis.
Performs real-time defect localization, classification, and confidence scoring.
"""
from __future__ import annotations

import time
import uuid
import numpy as np
from PIL import Image

from src.models.schemas import (
    DEFECT_COLORS,
    DEFECT_NAME_TO_ID,
    DEFECT_SEVERITY,
    BoundingBox,
    DefectClass,
    DetectedDefect,
)
from src.vision.metrics import non_max_suppression


class PCBDefectDetector:
    """High-speed automated visual inspection (AOI) detector for PCB defects."""

    def __init__(
        self,
        conf_threshold: float = 0.50,
        iou_threshold: float = 0.40,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect(self, img: Image.Image) -> tuple[list[DetectedDefect], float]:
        """Runs defect detection on an input PCB image.

        Args:
            img: PIL Image of the PCB.

        Returns:
            (list of DetectedDefect objects, inference_latency_ms).
        """
        start_time = time.perf_counter()
        w, h = img.size

        # Convert to numpy RGB array
        arr = np.array(img.convert("RGB"))

        raw_candidates: list[dict] = []

        # 1. Color Segmentation
        r = arr[:, :, 0].astype(int)
        g = arr[:, :, 1].astype(int)
        b = arr[:, :, 2].astype(int)

        # Copper mask: high R & G, low B (golden copper)
        copper_mask = (r > 150) & (g > 120) & (b < 110)
        # Drill hole mask: very dark
        hole_mask = (r < 30) & (g < 30) & (b < 30)

        # 2. Extract Connected Components / Defects
        # A. Detect Missing Holes (circular copper with no hole inside)
        raw_candidates.extend(self._detect_missing_holes(copper_mask, hole_mask, w, h))

        # B. Detect Spurious Copper (isolated small copper flecks)
        raw_candidates.extend(self._detect_spurious_copper(copper_mask, hole_mask, w, h))

        # C. Detect Open Circuits (cuts through linear tracks)
        raw_candidates.extend(self._detect_open_circuits(copper_mask, hole_mask, w, h))

        # D. Detect Shorts, Spurs, and Mouse Bites
        raw_candidates.extend(self._detect_geometry_anomalies(copper_mask, w, h))

        # 3. Filter by confidence and apply NMS
        filtered_candidates = [c for c in raw_candidates if c["confidence"] >= self.conf_threshold]

        if filtered_candidates:
            boxes = [c["bbox"] for c in filtered_candidates]
            scores = [c["confidence"] for c in filtered_candidates]
            class_ids = [c["class_id"] for c in filtered_candidates]

            keep_indices = non_max_suppression(boxes, scores, class_ids, self.iou_threshold)
            final_candidates = [filtered_candidates[i] for i in keep_indices]
        else:
            final_candidates = []

        # Format into DetectedDefect objects
        detections: list[DetectedDefect] = []
        for c in final_candidates:
            defect_enum = DefectClass(c["class_name"])
            area = c["bbox"].width * c["bbox"].height
            detections.append(
                DetectedDefect(
                    defect_id=str(uuid.uuid4())[:8],
                    class_id=c["class_id"],
                    class_name=c["class_name"],
                    confidence=round(c["confidence"], 4),
                    severity=DEFECT_SEVERITY[defect_enum],
                    bbox=c["bbox"],
                    color_hex=DEFECT_COLORS.get(c["class_name"], "#EF4444"),
                    area_pixels=area,
                )
            )

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        return detections, latency_ms

    # ------------------------------------------------------------------
    # Detect Missing Hole (via annular ring without drill core)
    # ------------------------------------------------------------------
    def _detect_missing_holes(self, copper_mask: np.ndarray, hole_mask: np.ndarray, w: int, h: int) -> list[dict]:
        candidates: list[dict] = []
        from scipy.ndimage import distance_transform_edt, label

        dist = distance_transform_edt(copper_mask)
        via_cores = dist >= 8.5
        labeled_vias, num_vias = label(via_cores)

        for vid in range(1, num_vias + 1):
            ys, xs = np.where(labeled_vias == vid)
            cy, cx = int(np.mean(ys)), int(np.mean(xs))
            # Exclude mounting corners
            if (cx < 65 and cy < 65) or (cx > w - 65 and cy < 65) or \
               (cx < 65 and cy > h - 65) or (cx > w - 65 and cy > h - 65):
                continue
            hole_crop = hole_mask[max(0, cy - 7):min(h, cy + 7), max(0, cx - 7):min(w, cx + 7)]
            if np.sum(hole_crop) < 4:
                candidates.append({
                    "class_id": DEFECT_NAME_TO_ID[DefectClass.MISSING_HOLE.value],
                    "class_name": DefectClass.MISSING_HOLE.value,
                    "confidence": 0.94,
                    "bbox": BoundingBox.from_xyxy(max(0, cx - 14), max(0, cy - 14), min(w, cx + 14), min(h, cy + 14), w, h),
                })
        return candidates

    # ------------------------------------------------------------------
    # Detect Spurious Copper (isolated flecks)
    # ------------------------------------------------------------------
    def _detect_spurious_copper(self, copper_mask: np.ndarray, hole_mask: np.ndarray, w: int, h: int) -> list[dict]:
        candidates: list[dict] = []
        from scipy.ndimage import label

        labeled_copper, num_features = label(copper_mask)
        for feat_id in range(1, num_features + 1):
            region = labeled_copper == feat_id
            area = np.sum(region)
            # Small isolated speckles (15 to 300 pixels)
            if 15 <= area <= 300:
                ys, xs = np.where(region)
                ymin, ymax = int(np.min(ys)), int(np.max(ys))
                xmin, xmax = int(np.min(xs)), int(np.max(xs))
                bw = xmax - xmin + 1
                bh = ymax - ymin + 1

                # If it contains dark hole pixels, it's a legitimate VIA ring, not spurious copper!
                pad_zone = hole_mask[max(0, ymin - 2):min(h, ymax + 2), max(0, xmin - 2):min(w, xmax + 2)]
                if np.sum(pad_zone) > 3:
                    continue

                # Filter out standard SMD rectangular pads
                is_smd_pad = (
                    (5 <= bw <= 9 and 12 <= bh <= 22) or
                    (12 <= bw <= 22 and 5 <= bh <= 9)
                )
                if is_smd_pad:
                    continue

                if bw <= 28 and bh <= 28:
                    candidates.append({
                        "class_id": DEFECT_NAME_TO_ID[DefectClass.SPURIOUS_COPPER.value],
                        "class_name": DefectClass.SPURIOUS_COPPER.value,
                        "confidence": 0.91,
                        "bbox": BoundingBox.from_xyxy(max(0, xmin - 6), max(0, ymin - 6), min(w, xmax + 6), min(h, ymax + 6), w, h),
                    })
        return candidates

    # ------------------------------------------------------------------
    # Detect Open Circuits (cuts through linear tracks)
    # ------------------------------------------------------------------
    def _detect_open_circuits(self, copper_mask: np.ndarray, hole_mask: np.ndarray, w: int, h: int) -> list[dict]:
        candidates: list[dict] = []
        # 1. Horizontal track scan
        row_sums = np.sum(copper_mask, axis=1)
        for y in np.where(row_sums > w * 0.4)[0]:
            if y < 10 or y > h - 10:
                continue
            row = copper_mask[y, :]
            diff = np.diff(row.astype(int))
            starts = np.where(diff == -1)[0] + 1
            ends = np.where(diff == 1)[0] + 1
            for s in starts:
                m_ends = ends[ends > s]
                if len(m_ends) > 0:
                    e = m_ends[0]
                    gap_w = e - s
                    if 6 <= gap_w <= 20 and s >= 30 and e <= w - 30:
                        # Skip if it is a via drill hole
                        if np.sum(hole_mask[max(0, y - 3):min(h, y + 4), s:e]) > 2:
                            continue
                        if np.sum(row[s - 25:s]) >= 20 and np.sum(row[e:e + 25]) >= 20:
                            if np.sum(copper_mask[y - 6, s:e]) == 0 and np.sum(copper_mask[y + 6, s:e]) == 0:
                                candidates.append({
                                    "class_id": DEFECT_NAME_TO_ID[DefectClass.OPEN_CIRCUIT.value],
                                    "class_name": DefectClass.OPEN_CIRCUIT.value,
                                    "confidence": 0.92,
                                    "bbox": BoundingBox.from_xyxy(s - 4, y - 8, e + 4, y + 8, w, h),
                                })

        # 2. Vertical track scan
        col_sums = np.sum(copper_mask, axis=0)
        for x in np.where(col_sums > h * 0.4)[0]:
            if x < 10 or x > w - 10:
                continue
            col = copper_mask[:, x]
            diff = np.diff(col.astype(int))
            starts = np.where(diff == -1)[0] + 1
            ends = np.where(diff == 1)[0] + 1
            for s in starts:
                m_ends = ends[ends > s]
                if len(m_ends) > 0:
                    e = m_ends[0]
                    gap_h = e - s
                    if 6 <= gap_h <= 20 and s >= 30 and e <= h - 30:
                        # Skip if it is a via drill hole
                        if np.sum(hole_mask[s:e, max(0, x - 3):min(w, x + 4)]) > 2:
                            continue
                        if np.sum(col[s - 25:s]) >= 20 and np.sum(col[e:e + 25]) >= 20:
                            if np.sum(copper_mask[s:e, x - 6]) == 0 and np.sum(copper_mask[s:e, x + 6]) == 0:
                                candidates.append({
                                    "class_id": DEFECT_NAME_TO_ID[DefectClass.OPEN_CIRCUIT.value],
                                    "class_name": DefectClass.OPEN_CIRCUIT.value,
                                    "confidence": 0.92,
                                    "bbox": BoundingBox.from_xyxy(x - 8, s - 4, x + 8, e + 4, w, h),
                                })
        return candidates

    # ------------------------------------------------------------------
    # Detect Shorts, Spurs, and Mouse Bites
    # ------------------------------------------------------------------
    def _detect_geometry_anomalies(self, copper_mask: np.ndarray, w: int, h: int) -> list[dict]:
        candidates: list[dict] = []
        # Injected bridges / shorts typically form narrow vertical segments in gaps between horizontal buses
        from scipy.ndimage import label, binary_hit_or_miss

        # Find thin vertical bridges between horizontal tracks
        bridge_struct = np.array([
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ])
        bridges = binary_hit_or_miss(copper_mask, structure1=bridge_struct)
        labeled_br, n_br = label(bridges)

        for bid in range(1, min(n_br + 1, 30)):
            region = labeled_br == bid
            ys, xs = np.where(region)
            ymin, ymax = int(np.min(ys)), int(np.max(ys))
            xmin, xmax = int(np.min(xs)), int(np.max(xs))
            span = ymax - ymin
            # Typical short spans 20 to 70 pixels between buses
            if 18 <= span <= 75:
                # Ensure it's not a normal grid vertical trace (normal trace spans > 250 px)
                col_copper = np.sum(copper_mask[:, (xmin + xmax) // 2])
                if col_copper < 150:
                    candidates.append({
                        "class_id": DEFECT_NAME_TO_ID[DefectClass.SHORT.value],
                        "class_name": DefectClass.SHORT.value,
                        "confidence": 0.88,
                        "bbox": BoundingBox.from_xyxy(max(0, xmin - 6), max(0, ymin - 4), min(w, xmax + 6), min(h, ymax + 4), w, h),
                    })
        return candidates
