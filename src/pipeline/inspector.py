"""Unified PCB Quality Control Inspection Orchestrator."""
from __future__ import annotations

import io
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from PIL import Image

from src.models.schemas import (
    BatchInspectionSummary,
    InspectionResult,
)
from src.vision.detector import PCBDefectDetector


class PCBInspector:
    """Production quality control inspector coordinating detection and pass/fail logic."""

    def __init__(
        self,
        conf_threshold: float = 0.50,
        iou_threshold: float = 0.40,
        allow_minor_copper: bool = False,
    ):
        self.detector = PCBDefectDetector(
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        self.allow_minor_copper = allow_minor_copper

    def inspect_image(self, img: Image.Image, filename: str = "board.png") -> InspectionResult:
        """Inspects a single PCB board image and determines pass/fail quality status."""
        w, h = img.size
        detections, latency_ms = self.detector.detect(img)

        # Count per defect type
        counts: dict[str, int] = {}
        for d in detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1

        # Pass / Fail criteria:
        # Critical / High defects immediately trigger FAIL
        has_critical = any(d.severity in ("CRITICAL", "HIGH") for d in detections)
        total_defects = len(detections)

        if total_defects == 0:
            status = "PASS"
        elif has_critical:
            status = "FAIL"
        elif self.allow_minor_copper and total_defects <= 2 and all(d.class_name == "spurious_copper" for d in detections):
            status = "PASS (CONDITIONAL)"
        else:
            status = "FAIL"

        return InspectionResult(
            inspection_id=str(uuid.uuid4()),
            filename=filename,
            image_width=w,
            image_height=h,
            total_defects=total_defects,
            pass_fail_status=status,
            defects=detections,
            inference_time_ms=latency_ms,
            timestamp=datetime.utcnow(),
            defect_counts=counts,
        )

    def inspect_file(self, file_path: str | Path) -> InspectionResult:
        """Loads and inspects an image from disk."""
        path = Path(file_path)
        with Image.open(path) as img:
            return self.inspect_image(img, filename=path.name)

    def inspect_bytes(self, data: bytes, filename: str = "upload.png") -> InspectionResult:
        """Inspects in-memory image bytes."""
        with Image.open(io.BytesIO(data)) as img:
            return self.inspect_image(img, filename=filename)

    def inspect_batch(self, items: List[Tuple[str, Image.Image]]) -> BatchInspectionSummary:
        """Inspects a batch of images and produces aggregate manufacturing yield analytics."""
        results: List[InspectionResult] = []
        latencies: List[float] = []
        distribution: dict[str, int] = {}

        for filename, img in items:
            res = self.inspect_image(img, filename=filename)
            results.append(res)
            latencies.append(res.inference_time_ms)
            for k, v in res.defect_counts.items():
                distribution[k] = distribution.get(k, 0) + v

        total = len(results)
        passed = sum(1 for r in results if "PASS" in r.pass_fail_status)
        failed = total - passed
        yield_rate = (passed / total * 100.0) if total > 0 else 0.0
        avg_latency = (sum(latencies) / total) if total > 0 else 0.0

        return BatchInspectionSummary(
            batch_id=str(uuid.uuid4())[:8],
            total_inspected=total,
            passed_count=passed,
            failed_count=failed,
            yield_rate_percent=round(yield_rate, 2),
            total_defects_found=sum(distribution.values()),
            avg_latency_ms=round(avg_latency, 2),
            items=results,
            defect_distribution=distribution,
        )
