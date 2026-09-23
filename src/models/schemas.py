"""Pydantic v2 schemas for PCB Defect Detection & Quality Control Platform."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class DefectClass(str, Enum):
    MISSING_HOLE = "missing_hole"
    MOUSE_BITE = "mouse_bite"
    OPEN_CIRCUIT = "open_circuit"
    SHORT = "short"
    SPUR = "spur"
    SPURIOUS_COPPER = "spurious_copper"


# Canonical mapping between ID and class name
DEFECT_CLASS_MAP: dict[int, DefectClass] = {
    0: DefectClass.MISSING_HOLE,
    1: DefectClass.MOUSE_BITE,
    2: DefectClass.OPEN_CIRCUIT,
    3: DefectClass.SHORT,
    4: DefectClass.SPUR,
    5: DefectClass.SPURIOUS_COPPER,
}

DEFECT_NAME_TO_ID: dict[str, int] = {v.value: k for k, v in DEFECT_CLASS_MAP.items()}

# Defect severity mapping for QC rules
DEFECT_SEVERITY: dict[DefectClass, str] = {
    DefectClass.OPEN_CIRCUIT: "CRITICAL",
    DefectClass.SHORT: "CRITICAL",
    DefectClass.MISSING_HOLE: "HIGH",
    DefectClass.MOUSE_BITE: "MEDIUM",
    DefectClass.SPUR: "MEDIUM",
    DefectClass.SPURIOUS_COPPER: "LOW",
}

# Hex color coding for bounding boxes
DEFECT_COLORS: dict[str, str] = {
    "missing_hole": "#EF4444",      # Red
    "mouse_bite": "#F59E0B",        # Amber
    "open_circuit": "#DC2626",      # Dark Red
    "short": "#8B5CF6",             # Purple
    "spur": "#3B82F6",              # Blue
    "spurious_copper": "#10B981",   # Green
}


class BoundingBox(BaseModel):
    """Bounding box coordinates (both absolute pixel and normalized values)."""
    x_min: int = Field(ge=0, description="Top-left x in pixels")
    y_min: int = Field(ge=0, description="Top-left y in pixels")
    x_max: int = Field(ge=0, description="Bottom-right x in pixels")
    y_max: int = Field(ge=0, description="Bottom-right y in pixels")
    width: int = Field(ge=0)
    height: int = Field(ge=0)
    x_center_norm: float = Field(ge=0.0, le=1.0)
    y_center_norm: float = Field(ge=0.0, le=1.0)
    width_norm: float = Field(ge=0.0, le=1.0)
    height_norm: float = Field(ge=0.0, le=1.0)

    @classmethod
    def from_xyxy(cls, x_min: int, y_min: int, x_max: int, y_max: int, img_w: int, img_h: int) -> BoundingBox:
        x_min = max(0, min(x_min, img_w))
        y_min = max(0, min(y_min, img_h))
        x_max = max(x_min, min(x_max, img_w))
        y_max = max(y_min, min(y_max, img_h))
        w = x_max - x_min
        h = y_max - y_min
        cx_norm = (x_min + w / 2.0) / img_w if img_w > 0 else 0.0
        cy_norm = (y_min + h / 2.0) / img_h if img_h > 0 else 0.0
        w_norm = w / img_w if img_w > 0 else 0.0
        h_norm = h / img_h if img_h > 0 else 0.0
        return cls(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            width=w,
            height=h,
            x_center_norm=round(cx_norm, 6),
            y_center_norm=round(cy_norm, 6),
            width_norm=round(w_norm, 6),
            height_norm=round(h_norm, 6),
        )

    @classmethod
    def from_yolo(cls, class_id: int, cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> BoundingBox:
        x_min = int((cx - w / 2.0) * img_w)
        y_min = int((cy - h / 2.0) * img_h)
        x_max = int((cx + w / 2.0) * img_w)
        y_max = int((cy + h / 2.0) * img_h)
        return cls.from_xyxy(x_min, y_min, x_max, y_max, img_w, img_h)


class DetectedDefect(BaseModel):
    """Individual defect detection record."""
    defect_id: str
    class_id: int = Field(ge=0, le=5)
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    severity: str
    bbox: BoundingBox
    color_hex: str
    area_pixels: int


class InspectionResult(BaseModel):
    """Complete inspection outcome for a single PCB board image."""
    inspection_id: str
    filename: str
    image_width: int
    image_height: int
    total_defects: int
    pass_fail_status: str = Field(description="PASS if 0 defects or only allowable minor, FAIL otherwise")
    defects: list[DetectedDefect] = Field(default_factory=list)
    inference_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    defect_counts: dict[str, int] = Field(default_factory=dict)


class BatchInspectionSummary(BaseModel):
    """Aggregated batch inspection analytics."""
    batch_id: str
    total_inspected: int
    passed_count: int
    failed_count: int
    yield_rate_percent: float
    total_defects_found: int
    avg_latency_ms: float
    items: list[InspectionResult] = Field(default_factory=list)
    defect_distribution: dict[str, int] = Field(default_factory=dict)


class ClassMetrics(BaseModel):
    class_name: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    support: int = 0


class BenchmarkReport(BaseModel):
    """Model evaluation benchmark output against ground-truth dataset."""
    total_images: int
    total_ground_truth_defects: int
    total_predicted_defects: int
    matched_tp_mean_iou: float = Field(description="Matched True-Positive Mean IoU")
    overall_precision: float
    overall_recall: float
    overall_f1: float
    avg_inference_latency_ms: float
    frames_per_second: float
    class_breakdown: dict[str, ClassMetrics] = Field(default_factory=dict)
    confusion_summary: dict[str, Any] = Field(default_factory=dict)
