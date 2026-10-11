# Smart Vision Monitor

Production-Style Real-Time Computer Vision & Proximity Alert System.

Built with **OpenCV 5.0**, **PyTorch 2.14 (CPU)**, **Ultralytics YOLO26n (Nano)**, **DepthAI 3.x (Hardware Backend)**, **FastAPI**, **Pydantic**, and **MATLAB**.

---

## Architecture

```
smart-vision-monitor/
├── src/
│   └── vision_monitor/
│       ├── __init__.py
│       ├── config.py         # Type-safe, validated Pydantic settings
│       ├── schemas.py        # Shared data contracts (BoundingBox, Detection, DetectionResult)
│       ├── detector.py       # Decoupled YOLO26n inference wrapper with warm-up
│       ├── camera.py         # Unified OpenCV input abstraction (webcam, image, video)
│       ├── depth.py          # DepthProvider (Simulated Mode A & DepthAI 3.x Mode B) + ROI median
│       ├── alerts.py         # Pure proximity alert evaluation engine
│       ├── pipeline.py       # End-to-end vision pipeline orchestrator
│       ├── api.py            # Production FastAPI service with async threadpool inference
│       ├── metrics.py        # Latency statistics (p50, p95, p99), throughput (FPS), RAM profiler
│       └── exceptions.py     # Custom exception hierarchy
├── tests/
│   ├── unit/                 # Unit tests (schemas, config, camera, detector, alerts, depth, pipeline, metrics)
│   ├── integration/          # Integration tests with real image detection
│   └── e2e/                  # End-to-end API tests (health, ready, metrics, predict)
├── scripts/
│   ├── run_webcam.py         # Live webcam / image / video detection demo with HUD
│   ├── benchmark.py          # Latency & throughput profiler across resolutions
│   └── create_test_image.py  # Deterministic test pattern generator
├── matlab/
│   └── analyze_benchmarks.m  # MATLAB benchmark ingestion, speedup scaling & plots
├── reports/
│   └── benchmarks.csv        # Measured benchmark performance logs
├── sample_data/
│   ├── bus.jpg               # Standard test image with person & bus
│   └── test_synthetic.jpg    # Synthetic geometry test pattern
├── .github/workflows/
│   └── ci.yml                # GitHub Actions CI workflow (lint, format, test coverage >=80%)
├── Dockerfile                # Production non-root container with PyTorch CPU & pre-downloaded weights
├── compose.yaml              # Docker Compose deployment configuration
└── pyproject.toml            # PEP 621 build configuration & tool settings
```

---

## Operating Modes

- **Mode A: Software-Only Mode**:
  - Works with any webcam, video file, or static image without specialized hardware.
  - Generates clearly identified simulated depth measurements (`[SIM]`) so proximity alert logic can be tested deterministically.
- **Mode B: Hardware-Assisted Mode**:
  - Integrates DepthAI 3.x for OAK-D stereo depth cameras.
  - Captures real stereo disparity/depth maps, synchronized with RGB frames.

---

## Measured Performance Benchmarks (Intel/AMD CPU Steady-State)

*Measured on 10 controlled iterations after 5 warmup cycles:*

| Resolution | Mean Total Latency | Inference p50 | Inference p95 | Throughput | Peak Host RAM |
|---|---|---|---|---|---|
| **320 × 320** | **25.9 ms** | 24.9 ms | 28.8 ms | **38.6 FPS** | 360.5 MB |
| **640 × 640** | **63.4 ms** | 61.9 ms | 64.8 ms | **15.8 FPS** | 409.6 MB |

---

## Quickstart & Usage

### 1. Run Automated Test Suite (74 Tests, 85% Coverage)
```powershell
.venv\Scripts\pytest.exe tests/ -v --cov=vision_monitor --cov-report=term-missing
```

### 2. Live Webcam / Pipeline Demo
```powershell
# With sample image (no camera required)
.venv\Scripts\python.exe scripts\run_webcam.py --source sample_data/bus.jpg

# With live webcam (index 0)
.venv\Scripts\python.exe scripts\run_webcam.py --source 0 --alert-dist 2.0
```
- Red bounding box + `PROXIMITY WARNING` HUD banner when objects are closer than `--alert-dist`.
- Press `q` or `ESC` to quit, `s` to save `snapshot.jpg`.

### 3. Run Benchmark Suite
```powershell
.venv\Scripts\python.exe scripts\benchmark.py --iterations 30
```
Outputs results to `reports/benchmarks.csv`.

### 4. Run Production FastAPI Service
```powershell
.venv\Scripts\uvicorn.exe vision_monitor.api:create_app --factory --host 0.0.0.0 --port 8000
```
Endpoints:
- `GET http://localhost:8000/health`: Liveness probe (`{"status": "healthy"}`)
- `GET http://localhost:8000/ready`: Readiness probe (verifies YOLO weights are loaded)
- `GET http://localhost:8000/metrics`: Uptime & average latency metrics
- `POST http://localhost:8000/predict`: Upload image file (`multipart/form-data`) returning structured detections, estimated distance, alert flag, and latency breakdown.
### 5. Launch Streamlit Web Dashboard
```powershell
.venv\Scripts\streamlit.exe run app.py
```
Opens in your browser at `http://localhost:8501`. Features:
- **Image Upload & Live Camera Snapshot**: Run YOLO26n inference directly in browser.
- **Visual Bounding Boxes & Distance Overlay**: Visualizes detections and proximity flags.
- **Latency & Performance Metrics**: Real-time breakdown of inference and pipeline latency.
- **Interactive Controls**: Adjust confidence threshold and alert distance dynamically.

### 6. Docker Deployment
```bash
# Build and run with Docker Compose
docker compose up --build -d

# Check health
curl http://localhost:8000/health
```

### 7. MATLAB Analysis
In MATLAB, open and run:
```matlab
run('matlab/analyze_benchmarks.m')
```
Plots latency vs. image resolution, throughput FPS, and RAM consumption.

---

## Portfolio Summary (Resume / GitHub)
> **Smart Vision Monitor**: Engineered a production-grade, real-time computer vision system using OpenCV 5.0, Ultralytics YOLO26n, and PyTorch (CPU). Architected a decoupled pipeline supporting both software simulation (Mode A) and DepthAI 3.x stereo depth cameras (Mode B) with robust ROI-median distance estimation. Exposed inference via a non-blocking FastAPI service with health/readiness probes and automated metrics. Achieved 38+ FPS on CPU with 74 automated tests (85% coverage), GitHub Actions CI, Docker packaging, and MATLAB benchmarking integration.
