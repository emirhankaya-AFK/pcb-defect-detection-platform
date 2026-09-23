"""Unit tests for FastAPI endpoints using TestClient."""
from __future__ import annotations

import io
import sys
from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from api.main import app  # noqa: E402
from src.generator.pcb_synthesizer import PCBSynthesizer  # noqa: E402

client = TestClient(app, raise_server_exceptions=False)


def _create_sample_png_bytes(width: int = 300, height: int = 300) -> bytes:
    synth = PCBSynthesizer(width=width, height=height, seed=42)
    img, _ = synth.generate_board()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestHealthAndMetadata:
    def test_health_check(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "HEALTHY"
        assert "version" in data

    def test_get_defect_classes(self):
        resp = client.get("/api/v1/classes")
        assert resp.status_code == 200
        data = resp.json()
        assert "classes" in data
        assert len(data["classes"]) == 6
        class_names = [c["class_name"] for c in data["classes"]]
        assert "missing_hole" in class_names
        assert "short" in class_names
        assert "open_circuit" in class_names

    def test_get_metrics(self):
        resp = client.get("/api/v1/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_inspections" in data
        assert "yield_rate_percent" in data


class TestInspectionEndpoints:
    def test_inspect_single_board_endpoint(self):
        png_bytes = _create_sample_png_bytes()
        resp = client.post(
            "/api/v1/inspect",
            files={"file": ("pcb_test.png", io.BytesIO(png_bytes), "image/png")},
            params={"conf_threshold": 0.40},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "inspection_id" in data
        assert "pass_fail_status" in data
        assert "defects" in data
        assert "inference_time_ms" in data
        assert data["image_width"] == 300
        assert data["image_height"] == 300

    def test_inspect_annotated_image_endpoint(self):
        png_bytes = _create_sample_png_bytes()
        resp = client.post(
            "/api/v1/inspect/annotated",
            files={"file": ("pcb_test.png", io.BytesIO(png_bytes), "image/png")},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        img = Image.open(io.BytesIO(resp.content))
        assert img.size == (300, 300)

    def test_inspect_batch_endpoint(self):
        png1 = _create_sample_png_bytes()
        png2 = _create_sample_png_bytes()

        files = [
            ("files", ("board_1.png", io.BytesIO(png1), "image/png")),
            ("files", ("board_2.png", io.BytesIO(png2), "image/png")),
        ]
        resp = client.post("/api/v1/inspect/batch", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_inspected"] == 2
        assert "yield_rate_percent" in data
        assert len(data["items"]) == 2
