"""FastAPI inference and monitoring service.

Production characteristics:
- Lifespan management: Loads and warms up the YOLO model once during startup.
- Non-blocking inference: Offloads CPU-bound vision pipeline work to a worker
  thread via `asyncio.to_thread` so the async event loop is never starved.
- Separate health & readiness checks:
    /health -> Liveness probe (process is running)
    /ready  -> Readiness probe (model weights are loaded and ready to serve)
- Structured telemetry: /metrics provides uptime and request latency stats.
- Security & error handling: Upload size and MIME-type validation, with
  custom exception handlers preventing internal stack trace leakage.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from vision_monitor.config import AppSettings, load_settings
from vision_monitor.depth import SimulatedDepthProvider
from vision_monitor.detector import Detector
from vision_monitor.exceptions import VisionMonitorError
from vision_monitor.pipeline import VisionPipeline
from vision_monitor.schemas import Detection

logger = logging.getLogger(__name__)

# Maximum allowed upload file size (15 MB)
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"}


# ─── Response Schemas ───


class HealthResponse(BaseModel):
    status: str = "healthy"
    timestamp: float = Field(default_factory=time.time)


class ReadyResponse(BaseModel):
    ready: bool
    model_name: str
    device: str
    is_depth_simulated: bool


class TimingMetrics(BaseModel):
    preprocess_ms: float
    inference_ms: float
    depth_ms: float
    total_ms: float


class PredictResponse(BaseModel):
    detections: list[Detection]
    image_width: int
    image_height: int
    is_depth_simulated: bool
    timings: TimingMetrics


class MetricsResponse(BaseModel):
    requests_total: int
    requests_failed: int
    avg_latency_ms: float
    uptime_seconds: float
    total_bytes_processed: int = 0
    bandwidth_mbps: float = 0.0


# ─── State & Telemetry ───


class ServiceState:
    def __init__(self) -> None:
        self.pipeline: VisionPipeline | None = None
        self.detector: Detector | None = None
        self.settings: AppSettings | None = None
        self.start_time: float = time.time()
        self.requests_total: int = 0
        self.requests_failed: int = 0
        self.latencies_ms: list[float] = []
        self.total_bytes_processed: int = 0

    def record_request(
        self, latency_ms: float, success: bool = True, bytes_count: int = 0
    ) -> None:
        self.requests_total += 1
        if success:
            self.latencies_ms.append(latency_ms)
            self.total_bytes_processed += bytes_count
            if len(self.latencies_ms) > 1000:
                self.latencies_ms.pop(0)
        else:
            self.requests_failed += 1


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app_settings = settings or load_settings()
    state = ServiceState()
    state.settings = app_settings

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: Load YOLO model weights and initialize pipeline
        logger.info("Initializing Smart Vision Monitor inference service...")
        detector = Detector(app_settings.detector)
        detector.load()

        # Software-only Mode A depth provider
        depth_provider = SimulatedDepthProvider(default_distance_m=1.8)
        pipeline = VisionPipeline(
            detector=detector,
            depth_provider=depth_provider,
            alert_settings=app_settings.alerts,
        )

        state.detector = detector
        state.pipeline = pipeline
        logger.info("Service readiness achieved: Model loaded and warmed up.")
        yield
        # Shutdown cleanup
        logger.info("Shutting down Smart Vision Monitor service...")

    app = FastAPI(
        title="Smart Vision Monitor API",
        version="0.1.0",
        description="Production computer vision detection & distance monitoring API",
        lifespan=lifespan,
    )

    # ─── Global Exception Handler ───
    @app.exception_handler(VisionMonitorError)
    async def handle_vision_error(request, exc: VisionMonitorError):
        logger.error("Vision monitor internal error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Inference processing error."},
        )

    # ─── Endpoints ───

    @app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
    async def health_check() -> HealthResponse:
        """Liveness probe: verifies service process is running."""
        return HealthResponse()

    @app.get("/ready", response_model=ReadyResponse, tags=["Monitoring"])
    async def ready_check() -> ReadyResponse:
        """Readiness probe: verifies model is loaded and ready for traffic."""
        if state.detector is None or not state.detector.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not ready to receive traffic.",
            )
        return ReadyResponse(
            ready=True,
            model_name=app_settings.detector.model_name,
            device=app_settings.detector.device,
            is_depth_simulated=True,
        )

    @app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
    async def get_metrics() -> MetricsResponse:
        """Operational metrics and request latency summary."""
        avg_latency = (
            sum(state.latencies_ms) / len(state.latencies_ms) if state.latencies_ms else 0.0
        )
        uptime = time.time() - state.start_time
        bandwidth = (
            round((state.total_bytes_processed / 1_000_000) / uptime, 4)
            if uptime > 0
            else 0.0
        )
        return MetricsResponse(
            requests_total=state.requests_total,
            requests_failed=state.requests_failed,
            avg_latency_ms=round(avg_latency, 2),
            uptime_seconds=round(uptime, 1),
            total_bytes_processed=state.total_bytes_processed,
            bandwidth_mbps=bandwidth,
        )

    @app.post("/predict", response_model=PredictResponse, tags=["Inference"])
    async def predict_image(file: UploadFile = File(...)) -> PredictResponse:  # noqa: B008
        """Upload an image file to run detection, distance estimation, and alert evaluation."""
        content_type = (
            file.content_type.split(";")[0].strip().lower()
            if file.content_type
            else ""
        )
        if content_type not in ALLOWED_IMAGE_TYPES:
            state.record_request(0, success=False)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unsupported file type '{file.content_type}'. "
                    f"Must be an image ({ALLOWED_IMAGE_TYPES})."
                ),
            )

        content = await file.read()
        if len(content) > MAX_FILE_SIZE_BYTES:
            state.record_request(0, success=False)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File size exceeds maximum 15MB limit.",
            )

        # Decode image from memory bytes
        np_arr = np.frombuffer(content, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            state.record_request(0, success=False)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to decode image. Corrupted or invalid image byte stream.",
            )

        if state.pipeline is None:
            state.record_request(0, success=False)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Pipeline is not initialized.",
            )

        # Offload CPU inference to worker thread so event loop remains unblocked
        pipeline_output = await asyncio.to_thread(state.pipeline.process_frame, frame)

        state.record_request(
            pipeline_output.total_time_ms, success=True, bytes_count=len(content)
        )

        return PredictResponse(
            detections=pipeline_output.result.detections,
            image_width=pipeline_output.result.image_width,
            image_height=pipeline_output.result.image_height,
            is_depth_simulated=pipeline_output.is_depth_simulated,
            timings=TimingMetrics(
                preprocess_ms=pipeline_output.preprocess_time_ms,
                inference_ms=pipeline_output.inference_time_ms,
                depth_ms=pipeline_output.depth_time_ms,
                total_ms=pipeline_output.total_time_ms,
            ),
        )

    return app
