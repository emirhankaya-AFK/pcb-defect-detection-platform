# Walkthrough — PCB Defect Detection & Quality Control Platform

## Project Overview

**Repository:** `pcb-defect-detection-platform`  
**Purpose:** Industrial Visual Inspection AI (Automated Optical Inspection for Electronics Manufacturing)  
**Alignment:** Electrical & Electronics Engineering (EEE) + Computer Vision + Production Backend  
**Tech Stack:** Python 3.11 · SciPy (ndimage) · FastAPI · Streamlit · Docker · GitHub Actions

---

## 🛠️ Work Accomplished

### 1. High-Resolution Procedural PCB Synthesizer
- Implemented `PCBSynthesizer` (`src/generator/pcb_synthesizer.py`) generating photorealistic PCB layouts:
  - FR4 solder mask substrate texture (Green, Blue, Black palettes) with micro-grain noise.
  - Horizontal, vertical, and diagonal copper traces with varying trace widths.
  - SMD component pads arrays (QFP / SOIC clusters) and pin 1 indicators.
  - Annular copper via drill rings with dark inner drill holes.
  - Corner mounting holes with circular copper rings.

### 2. Defect Injector (6 Canonical IPC-A-610 Classes)
- Implemented `DefectInjector` (`src/generator/defect_injector.py`) injecting standard industrial PCB defects with exact ground-truth bounding box coordinates:
  - **`missing_hole`**: Solid annular ring with no drill hole.
  - **`mouse_bite`**: Edge erosion/notch carved into copper trace.
  - **`open_circuit`**: Conductor break cut across a trace.
  - **`short`**: Conductive bridge connecting adjacent traces.
  - **`spur`**: Sharp extraneous protrusion projecting from trace.
  - **`spurious_copper`**: Isolated parasitic copper flecks.

### 3. Computer Vision Detection Core
- Implemented `PCBDefectDetector` (`src/vision/detector.py`):
  - Euclidean distance transform separating via annular cores from tracks.
  - Scanline run-length track continuity analysis for open circuit detection.
  - Multi-scale morphology and binary hit-or-miss pattern matching.
  - Non-maximum suppression (`non_max_suppression`) with dynamic IoU overlap filtering.
  - False positive suppression (filtering out SMD pads and via drill holes).

### 4. Quality Control & Batch Inspector
- Implemented `PCBInspector` (`src/pipeline/inspector.py`):
  - Pass/Fail grading according to IPC-A-610 Class 3 standards.
  - Critical/High defect rejection.
  - Batch conveyor processing aggregating First Pass Yield (FPY) percentages.

### 5. Production REST API (FastAPI)
- `POST /api/v1/inspect`: Single image defect localization, bounding box metadata, and latency.
- `POST /api/v1/inspect/annotated`: Image rendering with colored bounding boxes and confidence tags.
- `POST /api/v1/inspect/batch`: Multi-board conveyor batch inspection with yield analytics.
- `GET /api/v1/classes`: IPC-A-610 defect taxonomy descriptions and severity levels.
- `GET /api/v1/metrics`: In-memory inspection counters and pass/fail distribution.
- `GET /health`: Service health check probe.

### 6. Operator QC Terminal (Streamlit)
- Implemented `dashboard/app.py` with custom high-contrast industrial dark CSS (`dashboard/theme.py`):
  - Live AOI inspection with sample board loader or custom upload.
  - Real-time parameter sliders (Confidence, IoU NMS threshold).
  - High-magnification Defect Zoom Cutout Gallery.
  - Continuous batch conveyor yield analytics with pass/fail charts.
  - IPC-A-610 defect taxonomy visual guide.

---

## 🧪 Verification & Audit Results

### 1. Pytest Suite
```
pytest tests/ -v --tb=short --ignore=tests/evaluation

28 passed in 1.58s
```
- `test_api.py`: 6 tests passing (health, classes, single inspect, annotated render, batch inspect).
- `test_detector.py`: 8 tests passing (golden board, missing hole, spurious copper, inspector, batch, visualizer, crops).
- `test_generator.py`: 6 tests passing (synthesizer, reproducibility, mask colors, defect injector, bounds).
- `test_metrics.py`: 8 tests passing (IoU identity, disjoint, overlap, NMS suppression, evaluation calculations).

### 2. Code Quality
```
ruff check .

All checks passed! (0 lint errors)
```

### 3. Empirical Accuracy & Latency Benchmark
```
python tests/evaluation/run_evaluation.py

• Pass / Fail QC Accuracy   : 16/18 (88.9%)
• Mean Bounding Box IoU    : 0.7807
• Average Inference Latency: 158.78 ms
• Throughput Capacity       : ~377 boards/min
```
