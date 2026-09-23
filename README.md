# 🔬 Synthetic PCB AOI Prototype & Benchmark Platform: Rule-Based Computer Vision & Quality Control Engine

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Taxonomy: IPC--A--610 Categories](https://img.shields.io/badge/Taxonomy-Simulated%20IPC--A--610-blue?style=for-the-badge)](https://www.ipc.org)
[![Pytest](https://img.shields.io/badge/Pytest-30%2F30%20Passed-10b981?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![Ruff](https://img.shields.io/badge/Code%20Style-Ruff%200%20Errors-purple.svg?style=for-the-badge)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

A lightweight automated optical inspection (AOI) prototype and evaluation testbed for printed circuit board (PCB) quality control. Built with classical morphological and geometric computer vision (NumPy, SciPy, Pillow), procedural FR4 board synthesis with exact ground-truth annotations, a production-hardened FastAPI microservice, and an interactive Streamlit operator terminal. Evaluates 6 simulated **IPC-A-610** surface defect categories with transparent baseline metrics.

---

## 🏗️ Architecture & Inspection Pipeline

```mermaid
flowchart TD
    subgraph Acquisition["1. Image Acquisition & Preprocessing"]
        A1[High-Res Optical Camera / PCB Upload] --> A2[Color Normalization & Masking]
        A2 --> A3[FR4 Solder Mask Substrate Segmentation]
        A2 --> A4[ENIG Gold / Copper Trace Channel]
        A2 --> A5[Via & Mounting Drill Core Channel]
    end

    subgraph InspectionCore["2. Multi-Scale Defect Detection Core"]
        A4 & A5 --> B1["Distance Transform (Via Annular Core vs Missing Hole)"]
        A4 --> B2["1D Scanline Run-Length Continuity (Open Circuits)"]
        A4 & A5 --> B3["Morphological Opening & Hit-or-Miss (Shorts & Bridges)"]
        A3 & A4 --> B4["Connected Components Spatial Isolation (Spurious Copper)"]
        B1 & B2 & B3 & B4 --> B5[Raw Candidate Bounding Boxes & Confidence Scores]
    end

    subgraph PostProcessing["3. Spatial Suppression & Quality Logic"]
        B5 --> C1[Non-Maximum Suppression NMS with Dynamic IoU]
        C1 --> C2{IPC-A-610 Class 3 Severity Engine}
        C2 -->|Critical / High Defect Found| C3["QUALITY VERDICT: REJECT / FAIL"]
        C2 -->|Zero Defects or Allowable Minor| C4["QUALITY VERDICT: ACCEPT / PASS"]
    end

    subgraph Delivery["4. Production Backend & QC Terminal"]
        C3 & C4 --> D1[FastAPI REST Microservice]
        D1 --> D2["POST /api/v1/inspect (JSON Metadata)"]
        D1 --> D3["POST /api/v1/inspect/annotated (Rendered Image)"]
        D1 --> D4["POST /api/v1/inspect/batch (Conveyor Analytics)"]
        C3 & C4 --> E1[Streamlit Industrial QC Terminal]
        E1 --> E2[Defect Zoom Cutout Gallery]
        E1 --> E3[First Pass Yield FPY Tracking]
    end
```

---

## 🎯 Simulated IPC-A-610 6-Defect Taxonomy

The platform models 6 standard industrial electronic assembly defect categories on procedural synthetic FR4 boards:

| Class ID | Defect Class | Industrial Risk & Description | Default Severity | Detection Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **0** | `missing_hole` | Drill cycle failure: mounting or via hole not drilled; causes broken multi-layer net connections | `CRITICAL` | Euclidean distance transform core radius check |
| **1** | `mouse_bite` | Breakout tab erosion: mechanical notch in trace reducing cross-sectional current capacity | `MEDIUM` | Edge erosion baseline / ML target |
| **2** | `open_circuit` | Trace discontinuity: complete physical break in copper conductor line; immediate circuit failure | `CRITICAL` | 1D run-length track continuity with substrate isolation |
| **3** | `short` | Conductive bridge: unintended copper bridge across adjacent traces or SMD pads; causes power shorts | `CRITICAL` | Trace proximity & bridge baseline / ML target |
| **4** | `spur` | Extraneous burr: sharp protrusion jutting from trace perimeter; violates clearance design rules | `MEDIUM` | Skeleton branch protrusion baseline / ML target |
| **5** | `spurious_copper` | Residual copper: un-etched copper fleck or parasitic island floating on solder mask | `LOW` | Connected component spatial isolation filter |

---

## 🧮 Mathematical & Algorithmic Core

### 1. Spatial Euclidean Distance Transform for Via Core Verification
Annular via pads are separated from thin routing traces by evaluating the distance transform of the copper binary mask:
$$D(x, y) = \min_{(x', y') \in \text{Substrate}} \sqrt{(x - x')^2 + (y - y')^2}$$
- **Routing traces**: $D(x, y) \le 4$ px
- **Via annular copper rings**: $D(x, y) \ge 8.5$ px
Missing drill holes are identified when $D(x, y) \ge 8.5$ coincides with zero dark drill hole pixels ($\text{Hole Area} < 4$ px).

### 2. Intersection over Union (IoU) & Non-Maximum Suppression (NMS)
Candidate bounding boxes $B_1$ and $B_2$ are matched and duplicate detections suppressed using IoU:
$$\text{IoU}(B_1, B_2) = \frac{\text{Area}(B_1 \cap B_2)}{\text{Area}(B_1 \cup B_2)}$$

### 3. First Pass Yield (FPY) Manufacturing Metric
$$\text{FPY} = \left( \frac{\text{Passed Boards}}{\text{Total Inspected Boards}} \right) \times 100\%$$

---

## 📊 Verification & Empirical Benchmark

Evaluation performed using `tests/evaluation/run_evaluation.py` on an 18-board synthetic benchmark dataset with normalized ground-truth YOLO annotations (`dataset/ground_truth.json`):

```bash
python tests/evaluation/run_evaluation.py
```

### Class-by-Class Detection Performance (IoU Threshold = 0.45)

| Defect Class | Precision | Recall | F1-Score | Support | Algorithmic Mechanism / Baseline Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`missing_hole`** | 0.89 | 0.89 | **0.89** | 9 | 🟢 Euclidean distance transform annular core check |
| **`open_circuit`** | 0.16 | 1.00 | **0.28** | 6 | 🟡 1D scanline continuity heuristic (high recall baseline) |
| **`spurious_copper`** | 0.11 | 0.25 | **0.15** | 4 | ⚪ Connected-component area isolation filter |
| **`mouse_bite`** | 0.00 | 0.00 | **0.00** | 6 | ⚪ Baseline target for deep learning / YOLO fine-tuning |
| **`short`** | 0.00 | 0.00 | **0.00** | 4 | ⚪ Baseline target for deep learning / YOLO fine-tuning |
| **`spur`** | 0.00 | 0.00 | **0.00** | 6 | ⚪ Baseline target for deep learning / YOLO fine-tuning |
| **OVERALL MICRO** | **0.27** | **0.43** | **0.33** | 35 | Evaluated across 18 synthetic benchmark boards |

### Operational & Speed Performance Metrics

| Verification Metric | Target Benchmark | Achieved Result | Status |
| :--- | :--- | :--- | :--- |
| **Unit Test Suite** | 30 Pytest cases covering API, generator, detector, metrics, security | **30 / 30 Passed** | ✅ Verified |
| **Code Style & Quality** | Strict linting with `ruff check .` | **0 Errors** | ✅ Verified |
| **Pass / Fail QC Accuracy** | Binary classification on clean vs defective boards | **88.9% (16/18)** | ✅ Verified |
| **Matched TP Mean IoU** | Spatial overlap against ground truth on matched detections | **0.7807** | ✅ Verified |
| **Inference Latency** | Average latency per 800×800 board | **~195 ms** | ✅ Verified |
| **Throughput Capacity** | Continuous conveyor inspection rate | **~305 boards/min (~5.1 FPS)** | ✅ Verified |

> [!NOTE]
> **Prototype Scope & ML Roadmap:** This platform uses heuristic morphological computer vision (NumPy, SciPy) designed as a fast, reproducible baseline and benchmark platform. The procedural generator exports YOLO-formatted ground truth (`dataset/labels/*.txt`), providing the exact training and evaluation pipeline for deep learning object detectors (e.g. YOLOv8 / YOLOv10).

---

## 💻 Tech Stack

- **Computer Vision Core**: NumPy, SciPy (ndimage, distance_transform_edt, binary_hit_or_miss), Pillow
- **Backend Microservice**: FastAPI, Uvicorn, Pydantic v2
- **Operator Terminal**: Streamlit, Custom High-Contrast Industrial Dark CSS
- **Dataset Synthesis**: Procedural Board Synthesizer & Canonical Defect Injector
- **Packaging & CI/CD**: Docker, Docker Compose, GitHub Actions

---

## 🚀 Quickstart

### Option 1: Docker Compose (Recommended)
```bash
git clone https://github.com/emirhankaya-AFK/pcb-defect-detection-platform.git
cd pcb-defect-detection-platform

docker-compose up --build
```
- **Operator QC Terminal**: [http://localhost:8501](http://localhost:8501)
- **FastAPI Interactive Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Option 2: Local Installation
```bash
# Clone and setup environment
git clone https://github.com/emirhankaya-AFK/pcb-defect-detection-platform.git
cd pcb-defect-detection-platform

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Generate benchmark dataset
python dataset/generate_dataset.py
```

#### Run Tests & Benchmark
```bash
# Execute unit test suite
pytest tests/ -v --tb=short --ignore=tests/evaluation

# Execute benchmark accuracy & latency report
python tests/evaluation/run_evaluation.py
```

#### Launch Services
```bash
# Terminal 1: FastAPI Service
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Streamlit QC Terminal
streamlit run dashboard/app.py --server.port 8501
```

---

## 📡 REST API Reference

```bash
# 1. Health check
curl http://localhost:8000/health

# 2. Get defect classes taxonomy
curl http://localhost:8000/api/v1/classes

# 3. Inspect single PCB board (returns detected defects + pass/fail status)
curl -X POST http://localhost:8000/api/v1/inspect \
  -F "file=@dataset/images/pcb_sample_005.png"

# 4. Inspect board and download annotated image with bounding boxes
curl -X POST http://localhost:8000/api/v1/inspect/annotated \
  -F "file=@dataset/images/pcb_sample_005.png" \
  --output annotated_board.png

# 5. Batch inspect multiple boards
curl -X POST http://localhost:8000/api/v1/inspect/batch \
  -F "files=@dataset/images/pcb_sample_001.png" \
  -F "files=@dataset/images/pcb_sample_005.png"
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
