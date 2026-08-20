from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import pytest

from common.data_types import IngestedAudioBlock
from data_management import (
    Catalog,
    DataManagerService,
    DecisionRecord,
    RecordingStore,
    ResultWatermark,
    SessionMetadata,
)
from data_management.manifests import sha256_file, write_manifest
from data_management.timeline import iter_session_decisions


def _block(session_id: str, start_sample: int) -> IngestedAudioBlock:
    physical = np.full((960, 7), 0.01, np.float32)
    native = np.column_stack((physical[:, :6], np.zeros(960, np.float32), physical[:, 6]))
    logical = np.column_stack((physical, np.zeros(960, np.float32)))
    return IngestedAudioBlock(
        session_id,
        0,
        start_sample,
        start_sample + 960,
        48_000,
        start_sample // 960,
        start_sample / 48_000,
        logical,
        native,
    )


def _direction(track_id: int, theta: float, *, new: bool = False) -> dict[str, object]:
    return {
        "track_id": track_id,
        "rank": track_id - 1,
        "measured_theta_deg": theta,
        "theta_deg": theta,
        "raw_score": 2.0,
        "normalized_score": 0.8,
        "track_state": "confirmed",
        "is_observed": True,
        "is_new_track": new,
        "first_seen_sample": 960,
        "last_observed_sample": 48_000,
        "missed_samples": 0,
        "kalman_applied": True,
    }


def _decision(session_id: str) -> DecisionRecord:
    directions = (_direction(1, 20.0, new=True), _direction(2, 200.0, new=True))
    detections = tuple(
        {
            "track_id": item["track_id"],
            "theta_deg": item["theta_deg"],
            "probability": 0.9 - 0.1 * index,
            "is_voice": True,
            "model_id": "primary",
        }
        for index, item in enumerate(directions)
    )
    enhanced = tuple(
        {
            "track_id": item["track_id"],
            "theta_deg": item["theta_deg"],
            "backend": "test",
            "start_sample": 40_320,
            "end_sample": 48_000,
        }
        for item in directions
    )
    return DecisionRecord(
        session_id,
        0,
        8,
        48_000,
        (46_080, 48_000),
        (40_320, 48_000),
        "ok",
        candidates=directions,
        active_tracks=directions,
        detections=detections,
        voice_direction_count=2,
        raw_scores=np.ones(360, np.float32),
        normalized_scores=np.ones(360, np.float32),
        enhanced_audio=enhanced,
        enhanced_waveforms=(
            np.full(7_680, 0.1, np.float32),
            np.full(7_680, -0.1, np.float32),
        ),
        music_algorithm_version="normmusic_incremental_v1",
        model_order={"algorithm_version": "mdl_wax_kailath_v1", "estimated_sources": 2, "age_ms": 0},
        music_diagnostics={
            "valid_frequency_count": 43,
            "covariance_rank_min": 7,
            "covariance_condition_max": 125.0,
        },
        kalman_applied=True,
        config_revision=5,
        config_hash="config-hash-v5",
        calibration_revision=3,
        calibration_version="cal-v3",
        calibration_hash="calibration-hash-v3",
    )


def test_v4_rejects_cross_layer_track_id_mismatch() -> None:
    with pytest.raises(ValueError, match="L3 track_id"):
        DecisionRecord(
            "session",
            0,
            0,
            15_360,
            (13_440, 15_360),
            (7_680, 15_360),
            "error",
            candidates=(_direction(1, 20.0),),
            enhanced_audio=({"track_id": 2, "theta_deg": 20.0},),
            enhanced_waveforms=(np.zeros(7_680, np.float32),),
        )


def test_v4_per_id_assets_catalog_and_service_queries(tmp_path: Path) -> None:
    session_id = str(uuid.uuid4())
    store = RecordingStore(tmp_path, chunk_seconds=1, min_free_storage_gb=0)
    store.start_session(
        session_id,
        SessionMetadata(
            "session-config-hash",
            "session-calibration-hash",
            config_revision=5,
            calibration_revision=3,
            calibration_version="cal-v3",
        ),
    )
    store.set_recording_mode("continuous")
    for start in range(0, 48_000, 960):
        store.append_audio(_block(session_id, start))
    store.append_result(_decision(session_id))
    store.advance_result_watermark(ResultWatermark(session_id, 0, 48_000))

    manifest = store.stop_session()
    root = next(tmp_path.glob(f"runtime_sessions/*/*/{session_id}"))
    enhanced = [
        item
        for item in manifest["chunks"][0]["assets"]
        if item["kind"] == "enhanced_audio"
    ]
    assert len(enhanced) == 2
    assert {item["track_id"] for item in enhanced} == {1, 2}
    assert len({item["path"] for item in enhanced}) == 2
    assert all(f"track{item['track_id']:06d}" in item["path"] for item in enhanced)
    assert manifest["config_revision"] == 5
    assert manifest["calibration_revision"] == 3

    rows = list(iter_session_decisions(root))
    assert len(rows) == 1 and rows[0]["schema_version"] == "decision_record_v4"
    assert rows[0]["music_algorithm_version"] == "normmusic_incremental_v1"
    assert rows[0]["model_order"]["estimated_sources"] == 2
    assert rows[0]["music_diagnostics"]["valid_frequency_count"] == 43

    summaries = store.catalog.session_track_summaries(session_id)
    assert [(item["stream_epoch"], item["track_id"]) for item in summaries] == [(0, 1), (0, 2)]
    assert all(item["audio_asset_count"] == 1 for item in summaries)
    assert summaries[0]["latest_l4_probability"] == pytest.approx(0.9)
    timeline = store.catalog.track_timeline(session_id, 0, 2)
    assert len(timeline) == 1 and timeline[0]["enhanced_asset_path"].endswith(".wav")

    service = DataManagerService(tmp_path)
    assert len(service.runtime_session_tracks(session_id)) == 2
    assert len(service.track_timeline(session_id, 0, 1)) == 1
    assert len(service.track_audio_assets(session_id, 0, 1)) == 1
    assert len(service.session_audio_assets(session_id, "logical_8ch")) == 1
    service.close()
    store.close()


def test_v3_session_is_read_only_and_not_projected_as_public_ids(tmp_path: Path) -> None:
    session_id = str(uuid.uuid4())
    root = tmp_path / "runtime_sessions" / "2026" / "08" / session_id
    results = root / "results" / "legacy.jsonl"
    results.parent.mkdir(parents=True)
    legacy_row = {
        "record_type": "decision",
        "schema_version": "decision_record_v3",
        "session_id": session_id,
        "stream_epoch": 0,
        "window_id": 1,
        "decision_sample": 960,
        "candidates": [{"theta_deg": 20.0}],
        "detections": [{"theta_deg": 20.0, "probability": 0.8, "is_voice": True}],
    }
    results.write_text(json.dumps(legacy_row, ensure_ascii=False) + "\n", encoding="utf-8")
    before = results.read_bytes()
    manifest = {
        "schema_version": "audio_session_v2",
        "session_id": session_id,
        "status": "complete",
        "started_at_utc": "2026-08-19T00:00:00Z",
        "ended_at_utc": "2026-08-19T00:00:01Z",
        "current_mode": "continuous",
        "chunks": [{
            "stream_epoch": 0,
            "start_sample": 0,
            "end_sample": 960,
            "assets": [{
                "kind": "results",
                "path": str(results.relative_to(root)),
                "sha256": sha256_file(results),
            }],
        }],
    }
    write_manifest(root / "session_manifest.json", manifest)
    catalog = Catalog(tmp_path / "catalog.sqlite")
    catalog.upsert_session(manifest, root)

    assert list(iter_session_decisions(root))[0]["schema_version"] == "decision_record_v3"
    assert catalog.session_track_summaries(session_id) == []
    assert results.read_bytes() == before
    store = RecordingStore(tmp_path / "new", min_free_storage_gb=0)
    store.start_session("new-session", SessionMetadata("a", "b"))
    store.set_recording_mode("continuous")
    store.append_audio(_block("new-session", 0))
    with pytest.raises(ValueError, match="仅支持读取"):
        store.append_result({**legacy_row, "session_id": "new-session"})
    store.close()
    catalog.close()


def test_v4_enhanced_journal_recovery_keeps_track_identity(tmp_path: Path) -> None:
    session_id = str(uuid.uuid4())
    root = tmp_path / "runtime_sessions" / "2026" / "08" / session_id
    partial = root / "enhanced_audio" / "track000003.wav.partial"
    final = root / "enhanced_audio" / "track000003.wav"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"prepared-track-audio")
    journal = root / "enhanced_asset_commit.json"
    write_manifest(journal, {
        "schema_version": "enhanced_asset_commit_v2",
        "session_id": session_id,
        "entries": [{
            "partial_path": str(partial.relative_to(root)),
            "final_path": str(final.relative_to(root)),
            "sha256": sha256_file(partial),
            "stream_epoch": 0,
            "window_id": 1,
            "decision_sample": 960,
            "track_id": 3,
        }],
    })
    partial.replace(final)

    store = RecordingStore(tmp_path, min_free_storage_gb=0)
    recovered = store.recover_partials()
    assert len(recovered) == 2
    assert not final.exists() and not journal.exists()
    assert any(path.read_bytes() == b"prepared-track-audio" for path in recovered)
    store.close()
