from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision_monitor.config import load_settings
from vision_monitor.depth import SimulatedDepthProvider
from vision_monitor.detector import Detector
from vision_monitor.metrics import export_benchmark_csv, run_benchmark
from vision_monitor.pipeline import VisionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Smart Vision Monitor — Benchmarking Suite")
    parser.add_argument("--iterations", type=int, default=30, help="Benchmark sample iterations")
    parser.add_argument("--image", default="sample_data/bus.jpg", help="Reference test image")
    parser.add_argument("--out-csv", default="reports/benchmarks.csv", help="Output CSV path")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        logger.error("Test image '%s' not found. Run create_test_image.py first.", image_path)
        sys.exit(1)

    raw_frame = cv2.imread(str(image_path))
    if raw_frame is None:
        logger.error("Failed to decode image from %s", image_path)
        sys.exit(1)

    test_resolutions = [(320, 320), (640, 640)]
    records = []

    print("=" * 65)
    print("  Smart Vision Monitor — Performance Profiler")
    print("=" * 65)

    for w, h in test_resolutions:
        resized = cv2.resize(raw_frame, (w, h))

        settings = load_settings(
            detector={"model_name": "yolo26n.pt", "image_size": max(w, h), "device": "cpu"}
        )
        detector = Detector(settings.detector)
        detector.load()

        depth_provider = SimulatedDepthProvider(default_distance_m=2.0)
        pipeline = VisionPipeline(
            detector=detector,
            depth_provider=depth_provider,
            alert_settings=settings.alerts,
        )

        print(f"\nBenchmarking resolution: {w}x{h} (iterations={args.iterations})...")
        record = run_benchmark(
            pipeline=pipeline,
            test_image=resized,
            iterations=args.iterations,
            warmup_runs=5,
            model_name=settings.detector.model_name,
            device=settings.detector.device,
        )
        records.append(record)

        print(f"  Mean Total Latency : {record.mean_total_ms:.1f} ms")
        print(f"  Inference p50      : {record.p50_inference_ms:.1f} ms")
        print(f"  Inference p95      : {record.p95_inference_ms:.1f} ms")
        print(f"  Throughput         : {record.throughput_fps:.1f} FPS")
        print(f"  Peak Host RAM      : {record.peak_ram_mb:.1f} MB")
        print(f"  Frame Size         : {record.frame_bytes:,} bytes")
        print(f"  Bandwidth          : {record.bandwidth_mbps:.2f} MB/s")

    export_benchmark_csv(records, args.out_csv)
    print("\n" + "=" * 65)
    print(f"Benchmark run complete. Report saved to: {args.out_csv}")
    print("=" * 65)


if __name__ == "__main__":
    main()
