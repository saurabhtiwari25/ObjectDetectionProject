"""End-to-end tests for the FastAPI inference service.

Verifies:
- GET /health (liveness)
- GET /ready (readiness)
- GET /metrics (runtime telemetry)
- POST /predict (image upload, schema validation, structured output)
- Rejection of non-image uploads and corrupt byte payloads
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vision_monitor.api import create_app
from vision_monitor.config import load_settings


@pytest.fixture(scope="module")
def client():
    settings = load_settings(
        detector={"model_name": "yolo26n.pt", "confidence": 0.25, "device": "cpu"},
        alerts={"distance_threshold_m": 2.0},
    )
    app = create_app(settings)
    # Using TestClient as a context manager triggers the FastAPI lifespan startup/shutdown
    with TestClient(app) as c:
        yield c


class TestHealthAndReadyEndpoints:
    def test_health_check(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_ready_check(self, client: TestClient):
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True
        assert "model_name" in data

    def test_metrics_endpoint(self, client: TestClient):
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "requests_total" in data
        assert "avg_latency_ms" in data
        assert "total_bytes_processed" in data
        assert "bandwidth_mbps" in data


class TestPredictEndpoint:
    def test_predict_rejects_non_image_file(self, client: TestClient):
        files = {"file": ("test.txt", b"plain text is not an image", "text/plain")}
        response = client.post("/predict", files=files)
        assert response.status_code == 400
        assert "image" in response.json()["detail"].lower()

    def test_predict_rejects_corrupted_image(self, client: TestClient):
        files = {"file": ("broken.jpg", b"\x00\x01\x02\x03\x04", "image/jpeg")}
        response = client.post("/predict", files=files)
        assert response.status_code == 400
        assert "decode" in response.json()["detail"].lower()

    def test_predict_accepts_image_jpg_with_parameters(self, client: TestClient):
        sample_path = Path("sample_data/bus.jpg")
        if not sample_path.exists():
            pytest.skip("sample_data/bus.jpg not available")

        with open(sample_path, "rb") as f:
            image_bytes = f.read()

        files = {"file": ("bus.jpg", image_bytes, "image/jpg; charset=utf-8")}
        response = client.post("/predict", files=files)
        assert response.status_code == 200

    @pytest.mark.slow
    def test_predict_valid_sample_image(self, client: TestClient):
        sample_path = Path("sample_data/bus.jpg")
        if not sample_path.exists():
            pytest.skip("sample_data/bus.jpg not available")

        with open(sample_path, "rb") as f:
            image_bytes = f.read()

        files = {"file": ("bus.jpg", image_bytes, "image/jpeg")}
        response = client.post("/predict", files=files)

        assert response.status_code == 200
        payload = response.json()

        assert "detections" in payload
        assert "timings" in payload
        assert "image_width" in payload
        assert "image_height" in payload
        assert payload["is_depth_simulated"] is True

        # Ensure structured detections match schema
        assert len(payload["detections"]) > 0
        first_det = payload["detections"][0]
        assert "class_name" in first_det
        assert "confidence" in first_det
        assert "bbox" in first_det
        assert "distance_m" in first_det
        assert "alert" in first_det

        # Timings check
        assert payload["timings"]["total_ms"] > 0
        assert payload["timings"]["inference_ms"] > 0
