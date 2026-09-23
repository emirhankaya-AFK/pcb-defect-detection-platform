"""Visualizer utilities for PCB Defect Bounding Boxes and Operator Zoom Galleries."""
from __future__ import annotations

from typing import List, Tuple
from PIL import Image, ImageDraw
from src.models.schemas import DEFECT_COLORS, DetectedDefect


def draw_detections(
    image: Image.Image,
    detections: List[DetectedDefect],
    box_width: int = 3,
    show_labels: bool = True,
) -> Image.Image:
    """Renders color-coded bounding boxes and confidence badges on a PCB image.

    Returns:
        Annotated PIL Image copy.
    """
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size

    for det in detections:
        box = det.bbox
        color = DEFECT_COLORS.get(det.class_name, "#EF4444")

        # 1. Main bounding box
        draw.rectangle(
            [(box.x_min, box.y_min), (box.x_max, box.y_max)],
            outline=color,
            width=box_width,
        )

        # 2. Corner markers for precision aesthetic
        c_len = min(8, box.width // 4, box.height // 4)
        if c_len > 2:
            # Top-left corner
            draw.line([(box.x_min, box.y_min), (box.x_min + c_len, box.y_min)], fill="#FFFFFF", width=box_width + 1)
            draw.line([(box.x_min, box.y_min), (box.x_min, box.y_min + c_len)], fill="#FFFFFF", width=box_width + 1)
            # Bottom-right corner
            draw.line([(box.x_max, box.y_max), (box.x_max - c_len, box.y_max)], fill="#FFFFFF", width=box_width + 1)
            draw.line([(box.x_max, box.y_max), (box.x_max, box.y_max - c_len)], fill="#FFFFFF", width=box_width + 1)

        # 3. Label tag
        if show_labels:
            label_text = f"{det.class_name} {int(det.confidence * 100)}%"
            # Text bounding box
            tag_h = 16
            tag_w = len(label_text) * 7 + 8
            tag_y1 = max(0, box.y_min - tag_h)
            tag_y2 = box.y_min if box.y_min >= tag_h else box.y_min + tag_h

            draw.rectangle([(box.x_min, tag_y1), (box.x_min + tag_w, tag_y2)], fill=color)
            draw.text((box.x_min + 4, tag_y1 + 1), label_text, fill="#FFFFFF")

    return canvas


def extract_defect_crops(
    image: Image.Image,
    detections: List[DetectedDefect],
    padding: int = 20,
) -> List[Tuple[DetectedDefect, Image.Image]]:
    """Extracts zoomed cutouts around each detected defect for operator inspection."""
    crops: List[Tuple[DetectedDefect, Image.Image]] = []
    w, h = image.size

    for det in detections:
        box = det.bbox
        x1 = max(0, box.x_min - padding)
        y1 = max(0, box.y_min - padding)
        x2 = min(w, box.x_max + padding)
        y2 = min(h, box.y_max + padding)

        if x2 > x1 and y2 > y1:
            cropped = image.crop((x1, y1, x2, y2))
            crops.append((det, cropped))

    return crops
