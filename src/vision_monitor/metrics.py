"""Performance telemetry, percentile statistics, and benchmark runner."""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import psutil

if TYPE_CHECKING:
    from vision_monitor.pipeline import VisionPipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LatencySummary:
    """Statistical summary of execution latencies."""

    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    throughput_fps: float


@dataclass(frozen=True)
class BenchmarkRecord:
    """A single benchmark trial record for CSV export and MATLAB ingest."""

    model_name: str
    device: str
    image_size: int
    iterations: int
    mean_inference_ms: float
    p50_inference_ms: float
    p95_inference_ms: float
    mean_total_ms: float
    throughput_fps: float
    peak_ram_mb: float
    frame_bytes: int  # Raw frame size in bytes (H * W * channels)
    bandwidth_mbps: float  # Data throughput: MB/s = frame_bytes * FPS / 1e6


def compute_latency_summary(latencies_ms: list[float]) -> LatencySummary:
    """Calculate descriptive percentiles and throughput from raw latency samples."""
    if not latencies_ms:
        raise ValueError("Cannot compute latency summary from an empty sample list")

    arr = np.array(latencies_ms, dtype=np.float64)
    mean_val = float(np.mean(arr))
    p50_val = float(np.percentile(arr, 50))
    p95_val = float(np.percentile(arr, 95))
    p99_val = float(np.percentile(arr, 99))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    fps = 1000.0 / mean_val if mean_val > 0 else 0.0

    return LatencySummary(
        count=len(latencies_ms),
        mean_ms=round(mean_val, 2),
        p50_ms=round(p50_val, 2),
        p95_ms=round(p95_val, 2),
        p99_ms=round(p99_val, 2),
        min_ms=round(min_val, 2),
        max_ms=round(max_val, 2),
        throughput_fps=round(fps, 1),
    )


def run_benchmark(
    pipeline: VisionPipeline,
    test_image: np.ndarray,
    iterations: int = 50,
    warmup_runs: int = 5,
    model_name: str = "yolo26n.pt",
    device: str = "cpu",
) -> BenchmarkRecord:
    """Execute a controlled benchmarking session."""
    logger.info("Warming up benchmark (%d runs)...", warmup_runs)
    for _ in range(warmup_runs):
        pipeline.process_frame(test_image)

    inference_latencies: list[float] = []
    total_latencies: list[float] = []

    process = psutil.Process(os.getpid())
    peak_ram = process.memory_info().rss / (1024 * 1024)

    logger.info("Executing %d benchmark iterations...", iterations)
    for _ in range(iterations):
        out = pipeline.process_frame(test_image)
        inference_latencies.append(out.inference_time_ms)
        total_latencies.append(out.total_time_ms)

        current_ram = process.memory_info().rss / (1024 * 1024)
        if current_ram > peak_ram:
            peak_ram = current_ram

    inf_summary = compute_latency_summary(inference_latencies)
    tot_summary = compute_latency_summary(total_latencies)

    h, w = test_image.shape[:2]
    img_size = max(h, w)

    # Bandwidth: raw frame bytes * throughput = data processed per second
    frame_bytes = test_image.nbytes  # H * W * channels (e.g. 640*640*3)
    bandwidth_mbps = round(frame_bytes * tot_summary.throughput_fps / 1_000_000, 2)

    return BenchmarkRecord(
        model_name=model_name,
        device=device,
        image_size=img_size,
        iterations=iterations,
        mean_inference_ms=inf_summary.mean_ms,
        p50_inference_ms=inf_summary.p50_ms,
        p95_inference_ms=inf_summary.p95_ms,
        mean_total_ms=tot_summary.mean_ms,
        throughput_fps=tot_summary.throughput_fps,
        peak_ram_mb=round(peak_ram, 1),
        frame_bytes=frame_bytes,
        bandwidth_mbps=bandwidth_mbps,
    )


def export_benchmark_csv(records: list[BenchmarkRecord], filepath: Path | str) -> None:
    """Save benchmark records to a standard CSV file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        return

    fieldnames = list(asdict(records[0]).keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))

    logger.info("Exported %d benchmark records to %s", len(records), path)
