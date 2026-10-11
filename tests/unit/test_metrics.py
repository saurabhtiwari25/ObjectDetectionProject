"""Unit tests for metrics and latency statistics calculation."""

from __future__ import annotations

import pytest

from vision_monitor.metrics import compute_latency_summary


class TestComputeLatencySummary:
    def test_empty_latencies_raises_value_error(self):
        with pytest.raises(ValueError, match="empty sample list"):
            compute_latency_summary([])

    def test_single_value_produces_identical_percentiles(self):
        summary = compute_latency_summary([42.0])
        assert summary.mean_ms == 42.0
        assert summary.p50_ms == 42.0
        assert summary.p95_ms == 42.0
        assert summary.p99_ms == 42.0
        assert summary.min_ms == 42.0
        assert summary.max_ms == 42.0

    def test_multi_sample_distribution(self):
        # 100 samples from 1ms to 100ms
        latencies = list(range(1, 101))
        summary = compute_latency_summary(latencies)

        assert summary.mean_ms == pytest.approx(50.5, rel=1e-2)
        assert summary.p50_ms == pytest.approx(50.5, abs=1.0)
        assert summary.p95_ms == pytest.approx(95.0, abs=2.0)
        assert summary.p99_ms == pytest.approx(99.0, abs=2.0)
        assert summary.min_ms == 1.0
        assert summary.max_ms == 100.0
        assert summary.count == 100
        assert summary.throughput_fps > 0
