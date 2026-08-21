from __future__ import annotations

from gui.dev_test_ui.app import (
    _format_processing_pipeline_status,
    _format_processing_pipeline_tooltip,
)


class _ParallelRuntime:
    processing_status = {
        "queue_depths": {"l2": 1, "l3": 2, "l5": 0, "completion": 3},
        "queue_capacities": {"l2": 4, "l3": 3, "l5": 3, "completion": 8},
        "stage_alive": {"l2": True, "l3": True, "l5": False, "commit": True},
        "cache_bytes": 3 * 1024 * 1024,
        "cache_max_bytes": 64 * 1024 * 1024,
        "inflight_windows": 5,
        "completed_counts": {"l2": 11, "l3": 10, "l5": 9, "commit": 8},
        "error_counts": {"l2": 0, "l3": 1, "l5": 0, "commit": 0},
        "latest_errors": {"l2": None, "l3": "test failure", "l5": None, "commit": None},
        "processing_drops": 2,
    }


def test_parallel_pipeline_status_uses_only_public_snapshot():
    runtime = _ParallelRuntime()

    text = _format_processing_pipeline_status(runtime)
    tooltip = _format_processing_pipeline_tooltip(runtime)

    assert "L2 1/4 RUN #11" in text
    assert "L3 2/3 RUN #10 !1" in text
    assert "L5 0/3 STOP #9" in text
    assert "JOIN 3/8 RUN #8" in text
    assert "flight 5" in text
    assert "cache 3.0/64.0 MiB" in text
    assert "入口丢窗累计：2" in tooltip
    assert "l3: test failure" in tooltip


def test_pipeline_status_has_safe_fallback_without_public_snapshot():
    class _LegacyRuntime:
        pass

    runtime = _LegacyRuntime()
    assert _format_processing_pipeline_status(runtime) == "pipeline telemetry unavailable"
    assert "未提供分层流水线诊断" in _format_processing_pipeline_tooltip(runtime)
