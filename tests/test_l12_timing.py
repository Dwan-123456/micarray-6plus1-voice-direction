from __future__ import annotations

import pytest

from app.l12_timing import L12SegmentTimingTelemetry


def test_l12_timing_separates_gate_states_and_correlates_imcra_endpoint() -> None:
    timing = L12SegmentTimingTelemetry()
    timing.record_imcra("session", 0, (960,), 1.0)
    timing.record_l2(
        "session", 0, 960,
        gate_state="open", music_ms=5.0, id_tracking_ms=0.25, total_ms=6.5,
    )
    timing.record_imcra("session", 0, (1920,), 1.2)
    timing.record_l2(
        "session", 0, 1920,
        gate_state="closed", music_ms=None, id_tracking_ms=0.15, total_ms=0.8,
    )

    snapshot = timing.snapshot()

    opened = snapshot["by_gate"]["open"]
    closed = snapshot["by_gate"]["closed"]
    assert opened["total_count"] == 1
    assert opened["imcra"]["avg_ms"] == pytest.approx(1.0)
    assert opened["music"]["avg_ms"] == pytest.approx(5.0)
    assert opened["id_tracking"]["avg_ms"] == pytest.approx(0.25)
    assert opened["l2_total"]["avg_ms"] == pytest.approx(6.5)
    assert closed["total_count"] == 1
    assert closed["imcra"]["avg_ms"] == pytest.approx(1.2)
    assert closed["music"]["count"] == 0
    assert closed["music"]["avg_ms"] is None
    assert closed["id_tracking"]["avg_ms"] == pytest.approx(0.15)
    assert closed["l2_total"]["avg_ms"] == pytest.approx(0.8)


def test_l12_timing_resets_at_stream_boundary_without_retaining_audio() -> None:
    timing = L12SegmentTimingTelemetry()
    timing.record_imcra("old", 0, (960,), 2.0)
    timing.record_l2(
        "old", 0, 960,
        gate_state="open", music_ms=3.0, id_tracking_ms=0.2, total_ms=4.0,
    )

    timing.record_imcra("new", 1, (960,), 1.0)
    timing.record_l2(
        "new", 1, 960,
        gate_state="closed", music_ms=None, id_tracking_ms=None, total_ms=0.5,
    )
    snapshot = timing.snapshot()

    assert snapshot["session_id"] == "new"
    assert snapshot["stream_epoch"] == 1
    assert snapshot["by_gate"]["open"]["total_count"] == 0
    assert snapshot["by_gate"]["closed"]["total_count"] == 1
