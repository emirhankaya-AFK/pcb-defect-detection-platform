"""FastAPI application entry point for the PCB Defect Detection & Quality Control Platform."""
from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import analytics, inspection

APP_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup log
    print(f"🚀 Industrial Visual Inspection AI v{APP_VERSION} initialized.")
    yield


app = FastAPI(
    title="PCB Defect Detection & Quality Control API",
    description=(
        "Industrial Visual Inspection AI microservice for automated optical inspection (AOI) of PCBs. "
        "Detects 6 canonical IPC-A-610 defect classes: missing hole, mouse bite, open circuit, short, spur, spurious copper."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(inspection.router, prefix="/api/v1", tags=["Inspection"])
app.include_router(analytics.router, prefix="/api/v1", tags=["Analytics"])


@app.get("/health", tags=["Health"])
async def health_check():
    """Service health probe."""
    return {
        "status": "HEALTHY",
        "service": "pcb-defect-detection-api",
        "version": APP_VERSION,
    }
