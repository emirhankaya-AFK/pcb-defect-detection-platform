"""Procedural high-resolution PCB board generator.
Simulates FR4 solder mask, copper traces, SMD component pads, and via holes.
"""
from __future__ import annotations

import random
from typing import Tuple
import numpy as np
from PIL import Image, ImageDraw


class PCBSynthesizer:
    """Generates procedural, realistic synthetic PCB boards."""

    def __init__(
        self,
        width: int = 800,
        height: int = 800,
        seed: int | None = None,
        mask_color: str = "green",
    ):
        self.width = width
        self.height = height
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.mask_color = mask_color

        # Industrial color palettes
        if mask_color == "blue":
            self.c_mask = (15, 45, 80)
            self.c_mask_dark = (10, 32, 60)
        elif mask_color == "black":
            self.c_mask = (25, 28, 32)
            self.c_mask_dark = (16, 18, 20)
        else:  # classic green
            self.c_mask = (18, 68, 38)
            self.c_mask_dark = (12, 48, 26)

        self.c_copper = (212, 175, 55)       # ENIG Gold / Copper
        self.c_copper_bright = (235, 195, 75)
        self.c_via_hole = (8, 12, 10)         # Drill hole black
        self.c_silkscreen = (240, 245, 250)   # White silkscreen

    def generate_board(self) -> Tuple[Image.Image, dict]:
        """Generates a complete, defect-free golden reference PCB board image and its feature map.

        Returns:
            (PIL.Image in RGB, metadata dictionary with trace lines, pads, vias).
        """
        # 1. Base substrate texture
        img = Image.new("RGB", (self.width, self.height), self.c_mask)
        draw = ImageDraw.Draw(img)

        # Micro-texture noise to mimic real FR4 glass-fiber weave
        noise = self.np_rng.integers(-4, 5, (self.height, self.width, 3), dtype=np.int16)
        base_arr = np.array(img, dtype=np.int16) + noise
        base_arr = np.clip(base_arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(base_arr)
        draw = ImageDraw.Draw(img)

        meta: dict = {
            "traces": [],
            "vias": [],
            "pads": [],
            "silkscreens": [],
        }

        # 2. Draw ground grid / bus lines
        margin = 40
        step_y = 60
        step_x = 60

        # Horizontal and vertical routing buses
        for y in range(margin + 20, self.height - margin, step_y):
            y_jitter = y + self.rng.randint(-5, 5)
            x1, x2 = margin, self.width - margin
            trace_w = self.rng.choice([4, 6, 8])
            draw.line([(x1, y_jitter), (x2, y_jitter)], fill=self.c_copper, width=trace_w)
            meta["traces"].append({"type": "horiz", "y": y_jitter, "x1": x1, "x2": x2, "width": trace_w})

        for x in range(margin + 30, self.width - margin, step_x):
            x_jitter = x + self.rng.randint(-5, 5)
            y1, y2 = margin, self.height - margin
            trace_w = self.rng.choice([4, 6])
            draw.line([(x_jitter, y1), (x_jitter, y2)], fill=self.c_copper, width=trace_w)
            meta["traces"].append({"type": "vert", "x": x_jitter, "y1": y1, "y2": y2, "width": trace_w})

        # 3. SMD IC footprints (e.g. QFP / SOIC clusters)
        ic_centers = [
            (self.width // 4, self.height // 3),
            (3 * self.width // 4, self.height // 3),
            (self.width // 2, 2 * self.height // 3),
        ]

        for cx, cy in ic_centers:
            # IC body silkscreen boundary
            bw, bh = 80, 80
            draw.rectangle(
                [(cx - bw // 2, cy - bh // 2), (cx + bw // 2, cy + bh // 2)],
                outline=self.c_silkscreen,
                width=2,
            )
            # Pin 1 indicator dot
            draw.ellipse(
                [(cx - bw // 2 + 6, cy - bh // 2 + 6), (cx - bw // 2 + 12, cy - bh // 2 + 12)],
                fill=self.c_silkscreen,
            )
            # Chip label
            meta["silkscreens"].append({"x": cx, "y": cy, "text": "U_CHIP"})

            # SMD Pads around the chip
            pad_w, pad_h = 6, 16
            # Top & Bottom rows
            for px in range(cx - bw // 2 + 10, cx + bw // 2 - 10, 14):
                # Top pad
                draw.rectangle([(px, cy - bh // 2 - 20), (px + pad_w, cy - bh // 2 - 4)], fill=self.c_copper)
                meta["pads"].append({"x": px, "y": cy - bh // 2 - 20, "w": pad_w, "h": pad_h})
                # Bottom pad
                draw.rectangle([(px, cy + bh // 2 + 4), (px + pad_w, cy + bh // 2 + 20)], fill=self.c_copper)
                meta["pads"].append({"x": px, "y": cy + bh // 2 + 4, "w": pad_w, "h": pad_h})

            # Left & Right columns
            for py in range(cy - bh // 2 + 10, cy + bh // 2 - 10, 14):
                # Left pad
                draw.rectangle([(cx - bw // 2 - 20, py), (cx - bw // 2 - 4, py + pad_w)], fill=self.c_copper)
                meta["pads"].append({"x": cx - bw // 2 - 20, "y": py, "w": pad_h, "h": pad_w})
                # Right pad
                draw.rectangle([(cx + bw // 2 + 4, py), (cx + bw // 2 + 20, py + pad_w)], fill=self.c_copper)
                meta["pads"].append({"x": cx + bw // 2 + 4, "y": py, "w": pad_h, "h": pad_w})

        # 4. Via drill holes with copper annular rings
        for _ in range(35):
            vx = self.rng.randint(margin + 20, self.width - margin - 20)
            vy = self.rng.randint(margin + 20, self.height - margin - 20)
            r_outer = self.rng.choice([9, 11, 13])
            r_inner = r_outer // 2 + 1

            # Outer copper ring
            draw.ellipse(
                [(vx - r_outer, vy - r_outer), (vx + r_outer, vy + r_outer)],
                fill=self.c_copper_bright,
            )
            # Inner drill hole
            draw.ellipse(
                [(vx - r_inner, vy - r_inner), (vx + r_inner, vy + r_inner)],
                fill=self.c_via_hole,
            )
            meta["vias"].append({"x": vx, "y": vy, "r_outer": r_outer, "r_inner": r_inner})

        # 5. Mounting holes in corners
        for mx, my in [
            (margin, margin),
            (self.width - margin, margin),
            (margin, self.height - margin),
            (self.width - margin, self.height - margin),
        ]:
            draw.ellipse([(mx - 18, my - 18), (mx + 18, my + 18)], fill=self.c_copper)
            draw.ellipse([(mx - 10, my - 10), (mx + 10, my + 10)], fill=self.c_via_hole)

        return img, meta
