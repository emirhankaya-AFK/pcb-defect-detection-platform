"""FastAPI router for single and batch PCB inspections."""
from __future__ import annotations

import io
from typing import List
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image

from src.models.schemas import BatchInspectionSummary, InspectionResult
from src.pipeline.inspector import PCBInspector
from src.vision.visualizer import draw_detections

router = APIRouter()

ALLOWED_MIME = {"image/png", "image/jpeg", "image/bmp", "image/webp"}
MAX_FILE_SIZE_MB = 15
MAX_BATCH_SIZE_MB = 50


@router.post("/inspect", response_model=InspectionResult)
async def inspect_single_board(
    file: UploadFile = File(...),
    conf_threshold: float = Query(0.50, ge=0.1, le=0.99, description="Confidence score threshold"),
    iou_threshold: float = Query(0.40, ge=0.1, le=0.95, description="IoU threshold for NMS"),
):
    """Inspects a single uploaded PCB image and returns detected defects and pass/fail status."""
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Allowed types: {', '.join(sorted(ALLOWED_MIME))}",
        )

    data = await file.read()
    if len(data) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds maximum size of {MAX_FILE_SIZE_MB} MB")

    try:
        inspector = PCBInspector(conf_threshold=conf_threshold, iou_threshold=iou_threshold)
        result = inspector.inspect_bytes(data, filename=file.filename or "board.png")
        return result
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Image decoding failed: {str(e)}")


@router.post("/inspect/annotated")
async def inspect_and_render_image(
    file: UploadFile = File(...),
    conf_threshold: float = Query(0.50, ge=0.1, le=0.99),
):
    """Inspects an image and returns the annotated PNG image with color-coded bounding boxes."""
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Allowed types: {', '.join(sorted(ALLOWED_MIME))}",
        )

    data = await file.read()
    if len(data) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds maximum size of {MAX_FILE_SIZE_MB} MB")

    try:
        inspector = PCBInspector(conf_threshold=conf_threshold)
        with Image.open(io.BytesIO(data)) as img:
            result = inspector.inspect_image(img, filename=file.filename or "board.png")
            annotated = draw_detections(img, result.defects)

            buf = io.BytesIO()
            annotated.save(buf, format="PNG")
            return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Image rendering failed: {str(e)}")


@router.post("/inspect/batch", response_model=BatchInspectionSummary)
async def inspect_batch_boards(
    files: List[UploadFile] = File(...),
    conf_threshold: float = Query(0.50, ge=0.1, le=0.99),
):
    """Inspects a batch of PCB images and returns aggregated yield rate and defect statistics."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 files allowed per batch")

    items = []
    total_bytes = 0

    for f in files:
        if f.content_type not in ALLOWED_MIME:
            raise HTTPException(
                status_code=415,
                detail=f"File '{f.filename}' has unsupported media type '{f.content_type}'. Allowed types: {', '.join(sorted(ALLOWED_MIME))}",
            )

        data = await f.read()
        file_size = len(data)
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File '{f.filename}' exceeds individual maximum size of {MAX_FILE_SIZE_MB} MB",
            )

        total_bytes += file_size
        if total_bytes > MAX_BATCH_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"Batch payload exceeds cumulative limit of {MAX_BATCH_SIZE_MB} MB",
            )

        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            items.append((f.filename or "board.png", img))
        except Exception:
            continue

    if not items:
        raise HTTPException(status_code=422, detail="No valid images found in batch")

    inspector = PCBInspector(conf_threshold=conf_threshold)
    summary = inspector.inspect_batch(items)
    return summary
