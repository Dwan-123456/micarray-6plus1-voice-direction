from __future__ import annotations

import pytest

from common.config import load_config
from scripts.benchmark_l3_l4 import PROJECT_ROOT, _resolve_device, _summary


def test_benchmark_summary_has_locked_latency_and_throughput_units() -> None:
    summary = _summary([1.0, 2.0, 3.0, 4.0])

    assert summary["n"] == 4
    assert summary["avg_ms"] == pytest.approx(2.5)
    assert summary["median_ms"] == pytest.approx(2.5)
    assert summary["p95_ms"] == pytest.approx(3.85)
    assert summary["throughput_hz"] == pytest.approx(400.0)


def test_benchmark_auto_device_follows_the_single_runtime_config() -> None:
    config = load_config(PROJECT_ROOT / "config/config.yaml", environ={})

    assert _resolve_device("auto", config) == config.runtime.preferred_device.casefold()
