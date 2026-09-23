"""FastAPI router for QC analytics, metrics, and defect taxonomy metadata."""
from __future__ import annotations

from fastapi import APIRouter
from src.models.schemas import DEFECT_CLASS_MAP, DEFECT_COLORS, DEFECT_SEVERITY

router = APIRouter()

# In-memory metrics counter for demo session
_METRICS = {
    "total_inspections": 0,
    "total_passed": 0,
    "total_failed": 0,
    "total_defects_found": 0,
}


@router.get("/classes")
async def get_defect_classes():
    """Returns the list of supported IPC-A-610 defect classes with descriptions and severities."""
    descriptions = {
        "missing_hole": "Drill cycle failure: missing mounting hole or via drill core.",
        "mouse_bite": "Breakout tab damage or mechanical notch carved into trace edge.",
        "open_circuit": "Discontinuity or cut through conductive copper routing line.",
        "short": "Unintended conductive bridge or solder web connecting adjacent traces/pads.",
        "spur": "Unwanted burr or triangular spike projecting from copper trace perimeter.",
        "spurious_copper": "Isolated un-etched copper fleck or parasitic island on FR4 mask.",
    }

    result = []
    for cid, enum_val in DEFECT_CLASS_MAP.items():
        name = enum_val.value
        result.append({
            "class_id": cid,
            "class_name": name,
            "severity": DEFECT_SEVERITY[enum_val],
            "color_hex": DEFECT_COLORS.get(name, "#EF4444"),
            "description": descriptions.get(name, ""),
        })
    return {"classes": result}


@router.get("/metrics")
async def get_inspection_metrics():
    """Returns runtime manufacturing analytics and yield statistics."""
    total = _METRICS["total_inspections"]
    passed = _METRICS["total_passed"]
    yield_rate = (passed / total * 100.0) if total > 0 else 100.0
    return {
        "total_inspections": total,
        "total_passed": passed,
        "total_failed": _METRICS["total_failed"],
        "yield_rate_percent": round(yield_rate, 2),
        "total_defects_found": _METRICS["total_defects_found"],
    }
