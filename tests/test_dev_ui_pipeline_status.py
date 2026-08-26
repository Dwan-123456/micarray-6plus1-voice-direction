from __future__ import annotations

from types import SimpleNamespace

from gui.dev_test_ui.app import (
    _format_l12_segment_timing,
    _is_current_monotonic_l2_snapshot,
    _format_processing_pipeline_status,
    _format_processing_pipeline_tooltip,
)


def test_l2_ui_snapshot_filter_rejects_old_epoch_and_non_monotonic_windows() -> None:
    current = ("session-new", 2)
    latest = SimpleNamespace(
        session_id="session-new", stream_epoch=2, window_id=11, decision_sample=18_240,
    )
    previous = ("session-new", 2, 10, 17_280)
    assert _is_current_monotonic_l2_snapshot(latest, current, previous)
    assert not _is_current_monotonic_l2_snapshot(
        SimpleNamespace(
            session_id="session-old", stream_epoch=1,
            window_id=999, decision_sample=999_000,
        ),
        current,
        previous,
    )
    assert not _is_current_monotonic_l2_snapshot(
        SimpleNamespace(
            session_id="session-new", stream_epoch=2,
            window_id=10, decision_sample=17_280,
        ),
        current,
        previous,
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
        "input_health": {
            "input_overflow_count": 3,
            "handoff_drop_count": 4,
            "discontinuity_count": 2,
            "last_discontinuity": {"reason": "health_event:input_overflow"},
        },
        "dev_center_reference_writer": {
            "queue_depth": 4,
            "queue_capacity": 128,
            "accepted": 300,
            "completed": 296,
            "dropped": 1,
            "error": None,
        },
        "l12_segment_timing": {
            "by_gate": {
                "open": {
                    "total_count": 50,
                    "imcra": {"avg_ms": 1.25, "p95_ms": 1.75},
                    "music": {"avg_ms": 5.5, "p95_ms": 8.25},
                    "id_tracking": {"avg_ms": 0.2, "p95_ms": 0.35},
                    "l2_total": {"avg_ms": 7.1, "p95_ms": 10.0},
                },
                "closed": {
                    "total_count": 25,
                    "imcra": {"avg_ms": 1.1, "p95_ms": 1.5},
                    "music": {"avg_ms": None, "p95_ms": None},
                    "id_tracking": {"avg_ms": 0.15, "p95_ms": 0.25},
                    "l2_total": {"avg_ms": 0.6, "p95_ms": 0.9},
                },
            }
        },
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
    assert "IN ov3 hd4 ep2" in text
    assert "入口丢窗累计：2" in tooltip
    assert "Center试听旁路：排队 4/128" in tooltip
    assert "仅试听丢块 1" in tooltip
    assert "overflow 3，handoff丢块 4，epoch重置 2" in tooltip
    assert "最近原因 health_event:input_overflow" in tooltip
    assert "l3: test failure" in tooltip
    timing = _format_l12_segment_timing(runtime.processing_status)
    assert "OPEN n50 I 1.25/1.75 M 5.50/8.25 ID 0.20/0.35 T 7.10/10.00" in timing
    assert "CLOSED n25 I 1.10/1.50 M — ID 0.15/0.25 T 0.60/0.90" in timing


def test_pipeline_status_has_safe_fallback_without_public_snapshot():
    class _LegacyRuntime:
        pass

    runtime = _LegacyRuntime()
    assert _format_processing_pipeline_status(runtime) == "pipeline telemetry unavailable"
    assert "未提供分层流水线诊断" in _format_processing_pipeline_tooltip(runtime)
