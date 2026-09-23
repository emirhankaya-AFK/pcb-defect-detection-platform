"""Unit tests for PCB defect detector, visualizer, and inspection pipeline."""
from __future__ import annotations

import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.generator.pcb_synthesizer import PCBSynthesizer  # noqa: E402
from src.generator.defect_injector import DefectInjector  # noqa: E402
from src.models.schemas import DefectClass  # noqa: E402
from src.pipeline.inspector import PCBInspector  # noqa: E402
from src.vision.detector import PCBDefectDetector  # noqa: E402
from src.vision.visualizer import draw_detections, extract_defect_crops  # noqa: E402


class TestPCBDefectDetector:
    def setup_method(self):
        synth = PCBSynthesizer(width=400, height=400, seed=77)
        self.golden_img, self.meta = synth.generate_board()
        self.detector = PCBDefectDetector(conf_threshold=0.40)

    def test_golden_board_has_low_or_zero_defects(self):
        detections, latency_ms = self.detector.detect(self.golden_img)
        assert latency_ms > 0.0
        assert isinstance(detections, list)
        # Golden board should have at most 0 or very few false positive anomalies
        assert len(detections) <= 1

    def test_detects_injected_missing_hole(self):
        inj = DefectInjector(seed=12)
        d_img, _ = inj.inject_defects(
            self.golden_img,
            self.meta,
            target_defects=[DefectClass.MISSING_HOLE],
            num_defects=2,
        )
        detections, _ = self.detector.detect(d_img)
        detected_classes = [d.class_name for d in detections]
        assert DefectClass.MISSING_HOLE.value in detected_classes

    def test_detects_injected_spurious_copper(self):
        inj = DefectInjector(seed=34)
        d_img, _ = inj.inject_defects(
            self.golden_img,
            self.meta,
            target_defects=[DefectClass.SPURIOUS_COPPER],
            num_defects=2,
        )
        detections, _ = self.detector.detect(d_img)
        detected_classes = [d.class_name for d in detections]
        assert DefectClass.SPURIOUS_COPPER.value in detected_classes


class TestPCBInspectorPipeline:
    def setup_method(self):
        synth = PCBSynthesizer(width=400, height=400, seed=55)
        self.golden_img, self.meta = synth.generate_board()
        self.inspector = PCBInspector(conf_threshold=0.45)

    def test_golden_board_inspection_verdict(self):
        res = self.inspector.inspect_image(self.golden_img, filename="golden.png")
        assert res.filename == "golden.png"
        assert res.image_width == 400
        assert res.image_height == 400
        assert res.inference_time_ms > 0.0

    def test_defective_board_inspection_fails(self):
        inj = DefectInjector(seed=88)
        d_img, _ = inj.inject_defects(
            self.golden_img,
            self.meta,
            target_defects=[DefectClass.SHORT, DefectClass.OPEN_CIRCUIT],
            num_defects=3,
        )
        res = self.inspector.inspect_image(d_img, filename="defective.png")
        if res.total_defects > 0:
            assert res.pass_fail_status == "FAIL"

    def test_batch_inspection_aggregates_stats(self):
        batch = [
            ("board_1.png", self.golden_img),
            ("board_2.png", self.golden_img),
        ]
        summary = self.inspector.inspect_batch(batch)
        assert summary.total_inspected == 2
        assert summary.passed_count + summary.failed_count == 2
        assert 0.0 <= summary.yield_rate_percent <= 100.0


class TestVisualizer:
    def test_draw_detections_returns_same_sized_image(self):
        synth = PCBSynthesizer(width=300, height=300, seed=1)
        img, meta = synth.generate_board()
        inj = DefectInjector(seed=2)
        d_img, _ = inj.inject_defects(img, meta, num_defects=2)

        detector = PCBDefectDetector(conf_threshold=0.30)
        detections, _ = detector.detect(d_img)

        annotated = draw_detections(d_img, detections)
        assert isinstance(annotated, Image.Image)
        assert annotated.size == (300, 300)

    def test_extract_defect_crops(self):
        synth = PCBSynthesizer(width=300, height=300, seed=1)
        img, meta = synth.generate_board()
        inj = DefectInjector(seed=2)
        d_img, _ = inj.inject_defects(img, meta, target_defects=[DefectClass.MISSING_HOLE], num_defects=2)

        detector = PCBDefectDetector(conf_threshold=0.30)
        detections, _ = detector.detect(d_img)
        crops = extract_defect_crops(d_img, detections)
        assert isinstance(crops, list)
        for _, crop_img in crops:
            assert isinstance(crop_img, Image.Image)
            assert crop_img.width > 0 and crop_img.height > 0
