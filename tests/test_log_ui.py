from __future__ import annotations

import os
from pathlib import Path

import pytest

from gui.log_ui import Availability, PublicApiAdapter, StageState, StatisticsEngine
from gui.log_ui.cache import BoundedLru
from gui.log_ui.controller import CancelledError, page
from gui.log_ui.standalone import StandaloneUnavailableProvider


def _direction(track_id: int, theta: float, *, new: bool = False) -> dict[str, object]:
    return {
        "track_id": track_id,
        "measured_theta_deg": theta,
        "theta_deg": theta,
        "track_state": "confirmed",
        "is_observed": True,
        "is_new_track": new,
    }


def _decision(
    window_id: int,
    sample: int,
    *,
    epoch: int = 0,
    schema: str = "decision_record_v4",
    state: str = "completed",
) -> dict[str, object]:
    direction = _direction(7, 359.0 if window_id == 1 else 1.0, new=window_id == 1)
    return {
        "schema_version": schema,
        "session_id": "session-a",
        "stream_epoch": epoch,
        "window_id": window_id,
        "decision_sample": sample,
        "doa_range": [sample - 960, sample],
        "stage_statuses": {"l2": state, "l3": state, "l4": state},
        "stage_timings_ms": {"l2": 2.0 + window_id, "l3": 4.0, "l4": 5.0},
        "stage_queue_wait_ms": {"l2": 1.0, "l3": 2.0},
        "candidates": [direction],
        "active_tracks": [direction],
        "detections": [{"track_id": 7, "theta_deg": direction["theta_deg"], "probability": 0.8, "is_voice": True}],
        "enhanced_audio": [{"track_id": 7, "theta_deg": direction["theta_deg"], "path": "private.wav"}],
        "normalized_scores": [0.25] * 360,
        "model_order": {"estimated_sources": 1},
        "music_diagnostics": {"valid_frequency_count": 32},
    }


class AuditedProvider:
    def __init__(self, decisions=None):
        self.calls: list[tuple[object, ...]] = []
        self.decisions = list(decisions or (_decision(1, 960), _decision(2, 1920)))

    def runtime_sessions(self):
        self.calls.append(("runtime_sessions",))
        return [{
            "id": "session-a",
            "status": "complete",
            "schema_version": "audio_session_v2",
            "started_at": "2026-08-19T00:00:00Z",
            "ended_at": "2026-08-19T00:00:01Z",
            "path": "C:/sensitive/runtime/session-a",
            "metadata_json": (
                '{"device_format":{"sample_rate":48000},"config_hash":"cfg",'
                '"calibration_hash":"cal","path":"C:/nested/sensitive"}'
            ),
        }]

    def session_decisions(self, session_id, *, include_v3=True):
        self.calls.append(("session_decisions", session_id, include_v3))
        return list(self.decisions)

    def runtime_session_tracks(self, session_id, *, stream_epoch=None):
        self.calls.append(("runtime_session_tracks", session_id, stream_epoch))
        return []

    def track_timeline(self, session_id, stream_epoch, track_id):
        self.calls.append(("track_timeline", session_id, stream_epoch, track_id))
        return []

    def track_audio_assets(self, session_id, stream_epoch, track_id):
        self.calls.append(("track_audio_assets", session_id, stream_epoch, track_id))
        return [{"absolute_path": "C:/validated/track.wav", "sha256": "abc"}]

    def session_audio_assets(self, session_id, kind):
        self.calls.append(("session_audio_assets", session_id, kind))
        return []


def test_adapter_probes_only_public_capabilities_and_redacts_paths() -> None:
    provider = AuditedProvider()
    adapter = PublicApiAdapter(provider)
    descriptor = adapter.list_sessions()[0]

    assert adapter.capabilities.offline_review
    assert descriptor.sample_rate == 48_000
    assert descriptor.config_hash == "cfg"
    assert descriptor.raw_public["path"] == "[redacted]"
    assert descriptor.raw_public["metadata_json"]["path"] == "[redacted]"
    assert provider.calls == [("runtime_sessions",)]


def test_loading_uses_no_track_asset_or_runtime_mailbox_and_preserves_public_ids() -> None:
    provider = AuditedProvider()
    session = PublicApiAdapter(provider).load_session("session-a")

    assert len(session.windows) == 2
    assert {(item.stream_epoch, item.track_id) for item in session.tracks} == {(0, 7)}
    assert [item.theta_deg for item in session.tracks] == [359.0, 1.0]
    assert session.windows[0].raw_public["enhanced_audio"][0]["path"] == "[redacted]"
    assert not any(call[0] in {"track_audio_assets", "track_timeline"} for call in provider.calls)
    assert not any(call[0] in {"latest_dev_ui", "latest_l4_dev_ui", "latest_l1", "latest_windows"} for call in provider.calls)


def test_track_audio_is_requested_only_on_demand() -> None:
    provider = AuditedProvider()
    adapter = PublicApiAdapter(provider)
    adapter.load_session("session-a")
    assert not any(call[0] == "track_audio_assets" for call in provider.calls)

    assets = adapter.track_assets("session-a", 0, 7)

    assert assets[0]["absolute_path"].endswith("track.wav")
    assert provider.calls[-1] == ("track_audio_assets", "session-a", 0, 7)


def test_missing_capability_is_na_instead_of_zero() -> None:
    class SessionsOnly:
        def runtime_sessions(self):
            return [{"id": "session-a", "status": "complete", "schema_version": "audio_session_v2"}]

    session = PublicApiAdapter(SessionsOnly()).load_session("session-a")
    stats = StatisticsEngine().calculate(session)

    assert session.decision_availability == Availability.NOT_PROVIDED
    assert stats.track_count is None
    assert stats.stage["l2"].completed_hz is None


def test_standalone_desktop_provider_opens_without_catalog_or_runtime_capabilities() -> None:
    adapter = PublicApiAdapter(StandaloneUnavailableProvider())

    assert not adapter.capabilities.offline_review
    assert adapter.list_sessions() == ()


def test_unknown_schema_fails_closed_and_does_not_enter_statistics() -> None:
    provider = AuditedProvider([_decision(1, 960, schema="decision_record_v99")])
    session = PublicApiAdapter(provider).load_session("session-a")

    assert session.decision_availability == Availability.UNSUPPORTED_SCHEMA
    assert session.windows == ()
    assert session.anomalies[0].category == "schema"


def test_v3_remains_visible_without_inventing_public_track_ids() -> None:
    row = _decision(1, 960, schema="decision_record_v3")
    row["candidates"] = [{"theta_deg": 20.0}]
    row["active_tracks"] = []
    row["detections"] = [{"theta_deg": 20.0, "probability": 0.8}]
    session = PublicApiAdapter(AuditedProvider([row])).load_session("session-a")

    assert len(session.windows) == 1
    assert session.tracks == ()
    assert StatisticsEngine().calculate(session).track_count == 0


def test_completed_hz_uses_full_authoritative_epoch_ranges_and_completed_only() -> None:
    rows = [
        _decision(1, 960, state="completed"),
        _decision(2, 1920, state="dropped"),
        _decision(3, 2880, state="completed"),
        _decision(4, 960, epoch=1, state="completed"),
    ]
    session = PublicApiAdapter(AuditedProvider(rows)).load_session("session-a")
    stats = StatisticsEngine().calculate(session).stage["l2"]

    # epoch 0 covers [0,2880), epoch 1 covers [0,960): 0.08 s total.
    assert stats.duration_seconds == pytest.approx(0.08)
    assert stats.completed_hz == pytest.approx(37.5)
    assert stats.counts[StageState.DROPPED] == 1
    assert stats.applicable == 4


def test_percentiles_keep_n_and_missing_separate() -> None:
    row = _decision(2, 1920)
    row["stage_timings_ms"] = {"l3": 8.0}
    session = PublicApiAdapter(AuditedProvider([_decision(1, 960), row])).load_session("session-a")
    metric = StatisticsEngine().calculate(session).stage["l2"].compute

    assert metric.n == 1 and metric.missing == 1
    assert metric.p50 == pytest.approx(3.0)
    assert metric.missing_rate == pytest.approx(0.5)


def test_sample_gap_and_terminal_failures_are_explicit_anomalies() -> None:
    session = PublicApiAdapter(AuditedProvider([
        _decision(1, 960),
        _decision(3, 2880, state="timed_out"),
    ])).load_session("session-a")
    categories = {item.category for item in session.anomalies}

    assert "sample_gap" in categories
    assert "timed_out" in categories


def test_static_fixture_tree_is_byte_identical_and_no_sqlite_files_are_created(tmp_path: Path) -> None:
    fixture = tmp_path / "sealed"
    fixture.mkdir()
    record = fixture / "public_snapshot.json"
    record.write_text("immutable", encoding="utf-8")
    before = {item.relative_to(tmp_path): item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()}

    adapter = PublicApiAdapter(AuditedProvider())
    adapter.list_sessions()
    adapter.load_session("session-a")

    after = {item.relative_to(tmp_path): item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()}
    assert after == before
    assert not list(tmp_path.rglob("*.sqlite*"))
    assert not list(tmp_path.rglob("*-wal"))
    assert not list(tmp_path.rglob("*-shm"))


def test_lru_and_paging_are_bounded() -> None:
    cache = BoundedLru[int, int](2)
    assert cache.get_or_load(1, lambda: 10) == 10
    cache.get_or_load(2, lambda: 20)
    cache.get_or_load(3, lambda: 30)
    assert len(cache) == 2
    current = page(tuple(range(100_000)), 99_900, 100)
    assert len(current.items) == 100 and current.total == 100_000


def test_large_normalization_can_be_cancelled_without_partial_result() -> None:
    provider = AuditedProvider([_decision(1, 960)] * 100_000)
    calls = 0

    def cancelled() -> bool:
        nonlocal calls
        calls += 1
        return calls > 10

    with pytest.raises(CancelledError):
        PublicApiAdapter(provider).load_session("session-a", cancelled=cancelled)
    assert calls == 11


@pytest.mark.skipif(os.environ.get("CI_NO_QT") == "1", reason="Qt disabled")
def test_five_page_window_renders_offscreen_without_control_actions(monkeypatch) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QPushButton

    from gui.log_ui.app import PipelineLogWindow

    application = QApplication.instance() or QApplication([])
    window = PipelineLogWindow(AuditedProvider())
    session = window.adapter.load_session("session-a")
    window._loaded(session, None)
    application.processEvents()

    assert [window.tabs.tabText(index) for index in range(window.tabs.count())] == [
        "记录列表", "会话总览", "Pipeline 时间线", "单窗详情", "ID 与异常",
    ]
    labels = {button.text() for button in window.findChildren(QPushButton)}
    assert not labels.intersection({"启动", "停止", "暂停", "删除", "恢复", "导出", "重建Catalog"})
    assert window.timeline.table.rowCount() == 3  # explicit epoch boundary + two windows
    assert window.tracks.track_table.rowCount() == 1
    window.close()
