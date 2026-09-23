"""Unit tests for procedural PCB generator and defect injector."""
from __future__ import annotations

import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.generator.pcb_synthesizer import PCBSynthesizer  # noqa: E402
from src.generator.defect_injector import DefectInjector  # noqa: E402
from src.models.schemas import DefectClass  # noqa: E402


class TestPCBSynthesizer:
    def test_generates_valid_image(self):
        synth = PCBSynthesizer(width=400, height=400, seed=42)
        img, meta = synth.generate_board()

        assert isinstance(img, Image.Image)
        assert img.size == (400, 400)
        assert img.mode == "RGB"
        assert "traces" in meta
        assert "vias" in meta
        assert "pads" in meta
        assert len(meta["traces"]) > 0

    def test_reproducible_with_seed(self):
        s1 = PCBSynthesizer(width=200, height=200, seed=123)
        s2 = PCBSynthesizer(width=200, height=200, seed=123)
        img1, _ = s1.generate_board()
        img2, _ = s2.generate_board()

        assert list(img1.getdata()) == list(img2.getdata())

    def test_supports_alternative_mask_colors(self):
        s_blue = PCBSynthesizer(width=200, height=200, mask_color="blue", seed=1)
        s_black = PCBSynthesizer(width=200, height=200, mask_color="black", seed=1)
        img_b, _ = s_blue.generate_board()
        img_k, _ = s_black.generate_board()

        assert img_b.size == (200, 200)
        assert img_k.size == (200, 200)


class TestDefectInjector:
    def setup_method(self):
        self.synth = PCBSynthesizer(width=500, height=500, seed=99)
        self.golden_img, self.meta = self.synth.generate_board()
        self.injector = DefectInjector(seed=99)

    def test_injects_requested_number_of_defects(self):
        d_img, records = self.injector.inject_defects(self.golden_img, self.meta, num_defects=3)
        assert isinstance(d_img, Image.Image)
        assert len(records) == 3

    def test_defect_record_structure(self):
        _, records = self.injector.inject_defects(self.golden_img, self.meta, num_defects=2)
        for r in records:
            assert "class_id" in r
            assert "class_name" in r
            assert "bbox" in r
            box = r["bbox"]
            assert 0 <= box.x_min < box.x_max <= 500
            assert 0 <= box.y_min < box.y_max <= 500
            assert 0.0 <= box.x_center_norm <= 1.0
            assert 0.0 <= box.y_center_norm <= 1.0

    def test_can_inject_each_canonical_defect_class(self):
        for defect_cls in DefectClass:
            _, recs = self.injector.inject_defects(
                self.golden_img,
                self.meta,
                target_defects=[defect_cls],
                num_defects=1,
            )
            assert len(recs) == 1
            assert recs[0]["class_name"] == defect_cls.value
