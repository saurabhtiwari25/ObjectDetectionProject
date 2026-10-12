from __future__ import annotations

import argparse
import contextlib
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision_monitor.camera import CameraSource
from vision_monitor.config import load_settings
from vision_monitor.depth import SimulatedDepthProvider
from vision_monitor.detector import Detector
from vision_monitor.pipeline import PipelineOutput, VisionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

COLORS = [
    (0, 255, 127),
    (255, 191, 0),
    (0, 165, 255),
    (255, 0, 255),
    (0, 255, 255),
    (255, 105, 180),
    (50, 205, 50),
    (255, 0, 0),
    (203, 192, 255),
]


def draw_pipeline_output(frame: np.ndarray, output: PipelineOutput) -> np.ndarray:
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    any_alert = any(det.alert for det in output.result.detections)

    for det in output.result.detections:
        depth_tag = " [SIM]" if output.is_depth_simulated else ""
        dist_str = f" | {det.distance_m:.1f}m{depth_tag}" if det.distance_m is not None else ""

        color_idx = hash(det.class_name) % len(COLORS)
        color = COLORS[color_idx]
        label = f"{det.class_name} {det.confidence:.0%}{dist_str}"

        x1, y1 = int(det.bbox.x1), int(det.bbox.y1)
        x2, y2 = int(det.bbox.x2), int(det.bbox.y2)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        rect_pt1 = (x1, max(0, y1 - th - 10))
        rect_pt2 = (x1 + tw + 4, max(th + 10, y1))
        cv2.rectangle(annotated, rect_pt1, rect_pt2, color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 2, max(th + 5, y1 - 4)),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
        )

    cv2.rectangle(annotated, (0, 0), (w, 40), (20, 20, 20), -1)

    mode_label = "MODE A (SIMULATED DEPTH)" if output.is_depth_simulated else "MODE B (OAK-D DEPTH)"
    cv2.putText(annotated, mode_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    latency_text = (
        f"Latency: {output.total_time_ms:.0f}ms (Infer: {output.inference_time_ms:.0f}ms)"
    )
    cv2.putText(
        annotated,
        latency_text,
        (250, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 127),
        1,
    )

    alert_banner = "PROXIMITY WARNING" if any_alert else "STATUS: CLEAR"
    alert_color = (0, 0, 255) if any_alert else (0, 255, 0)
    cv2.putText(
        annotated,
        alert_banner,
        (max(10, w - 230), 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        alert_color,
        2,
    )

    return annotated


def main():
    parser = argparse.ArgumentParser(description="Smart Vision Monitor — Pipeline Demo")
    parser.add_argument("--source", default=0, help="Camera index, image path, or video path")
    parser.add_argument("--confidence", type=float, default=0.25, help="Min confidence threshold")
    parser.add_argument("--img-size", type=int, default=640, help="YOLO input size")
    parser.add_argument("--alert-dist", type=float, default=2.0, help="Alert threshold in meters")
    args = parser.parse_args()

    source = args.source
    with contextlib.suppress(ValueError, TypeError):
        source = int(source)

    settings = load_settings(
        detector={
            "model_name": "yolo26n.pt",
            "confidence": args.confidence,
            "image_size": args.img_size,
        },
        alerts={"distance_threshold_m": args.alert_dist},
    )

    print("=" * 60)
    print("  Smart Vision Monitor — Pipeline & Alert Demo")
    print("=" * 60)
    print(f"  Model       : {settings.detector.model_name}")
    print(f"  Source      : {source}")
    print(f"  Alert Limit : {settings.alerts.distance_threshold_m:.1f} m")
    print("=" * 60)

    detector = Detector(settings.detector)
    print("\n[1/3] Loading YOLO model...")
    detector.load()

    depth_provider = SimulatedDepthProvider(default_distance_m=1.8)
    pipeline = VisionPipeline(
        detector=detector,
        depth_provider=depth_provider,
        alert_settings=settings.alerts,
    )

    print("[2/3] VisionPipeline initialized.")
    print("[3/3] Opening camera stream...\n")

    cam_source = CameraSource(
        source=source,
        width=settings.camera.width,
        height=settings.camera.height,
    )
    with cam_source as cam:
        print("Camera opened. Press 'q' or ESC to quit, 's' to save snapshot.\n")
        frame_count = 0
        fps_start = time.perf_counter()
        elapsed = 0.0
        fps = 0.0

        while True:
            try:
                frame = cam.read()
            except Exception:
                print("Frame read finished or source disconnected.")
                break

            output = pipeline.process_frame(frame)
            annotated = draw_pipeline_output(frame, output)

            frame_count += 1
            elapsed = time.perf_counter() - fps_start
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Smart Vision Monitor", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("\nQuitting...")
                break
            elif key == ord("s"):
                cv2.imwrite("snapshot.jpg", annotated)
                print("Snapshot saved: snapshot.jpg")

    cv2.destroyAllWindows()
    print(f"\nProcessed {frame_count} frames in {elapsed:.1f}s ({fps:.1f} FPS)")
    print("Done.")


if __name__ == "__main__":
    main()
