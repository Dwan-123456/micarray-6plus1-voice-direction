from __future__ import annotations

import json
import threading
import time
import uuid
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np

from common.data_types import ImcraHopSnapshot, IngestedAudioBlock
from layer1_input.interface import NoiseSpectrumRecord
from data_management import (
    Annotation,
    Catalog,
    CorpusStore,
    DecisionRecord,
    RecordingMetadata,
    RecordingStore,
    SessionMetadata,
)
from data_management.export import export_assets, verify_export
from data_management.qa import leakage_check
from data_management.retention import move_to_trash, restore_from_trash
from data_management.dedicated_recording import DedicatedRecordingController, WizardPhase
from data_management.experiments import ExperimentStore
from data_management.wizard import WizardInput
from data_management.service import DataManagerService


def block(session: str, start: int, *, epoch: int = 0, frames: int = 960) -> IngestedAudioBlock:
    physical = np.full((frames, 7), 0.1, np.float32)
    native = np.column_stack((physical[:, :6], np.full(frames, 0.2, np.float32), physical[:, 6]))
    logical = np.column_stack((physical, np.full(frames, 0.2, np.float32)))
    return IngestedAudioBlock(
        session, epoch, start, start + frames, 48000, start // 960, float(start / 48000), logical, native
    )


def metadata(dataset: str) -> RecordingMetadata:
    return RecordingMetadata(
        dataset,
        "dedicated",
        "2026-01-01T00:00:00Z",
        "quiet",
        "room-a",
        "pose-a",
        1,
        ("human_voice",),
        {
            "consent_status": "granted",
            "license_id": "internal",
            "allowed_uses": ["research", "ml_training"],
            "expires_at_utc": None,
        },
        (30.0,),
        (1.0,),
        ("speaker-01",),
        ("zh",),
    )


def test_runtime_recording_assets_results_and_catalog_rebuild(tmp_path: Path):
    session = str(uuid.uuid4())
    store = RecordingStore(tmp_path, chunk_seconds=1, min_free_storage_gb=0)
    store.start_session(session, SessionMetadata("a" * 64, "b" * 64))
    store.set_recording_mode("manual")
    store.append_audio(block(session, 0))
    store.start_recording()
    for start in range(0, 48000, 960):
        store.append_audio(block(session, start))
    result = DecisionRecord(
        session,
        0,
        1,
        960,
        (0, 960),
        (0, 960),
        "ok",
        candidates=({"theta_deg": 0.0},),
        detections=({
            "theta_deg": 0.0,
            "probability": 0.9,
            "is_voice": True,
            "model_id": "test",
        },),
        voice_direction_count=1,
        raw_scores=np.ones(360),
        normalized_scores=np.ones(360) * 0.5,
    )
    store.append_result(result)
    from data_management import ResultWatermark
    store.advance_result_watermark(ResultWatermark(session, 0, 48000))
    manifest = store.stop_session("normal")
    assert manifest["status"] == "complete" and len(manifest["chunks"]) == 1
    chunk = manifest["chunks"][0]
    assert chunk["frame_count"] == 48000 and chunk["result_count"] == 1
    root = next(tmp_path.glob(f"runtime_sessions/*/*/{session}"))
    assets = {x["kind"]: root / x["path"] for x in chunk["assets"]}
    with wave.open(str(assets["native_8ch"]), "rb") as wav:
        assert (wav.getnchannels(), wav.getnframes()) == (8, 48000)
    with wave.open(str(assets["logical_8ch"]), "rb") as wav:
        assert (wav.getnchannels(), wav.getnframes()) == (8, 48000)
        logical_pcm = np.frombuffer(wav.readframes(1), dtype="<i2")
    with wave.open(str(assets["native_8ch"]), "rb") as wav:
        native_pcm = np.frombuffer(wav.readframes(1), dtype="<i2")
    assert logical_pcm[-1] == native_pcm[6]
    assert logical_pcm[6] == native_pcm[7]
    assert np.load(assets["physical_float"]).shape == (48000, 7)
    assert len(assets["results"].read_text(encoding="utf-8").splitlines()) == 2
    with np.load(assets["spatial_response"]) as spatial:
        assert spatial["raw_scores"].shape == (1, 360)
    store.catalog.connection.execute("DELETE FROM sessions")
    store.catalog.connection.commit()
    assert store.catalog.rebuild(tmp_path)["sessions"] == 1
    store.close()


def test_terminal_drop_watermark_is_audited_without_duplicate_result_row(tmp_path: Path):
    from data_management import ResultWatermark

    session = str(uuid.uuid4())
    store = RecordingStore(tmp_path, chunk_seconds=1, min_free_storage_gb=0)
    store.start_session(session, SessionMetadata("a", "b"))
    store.set_recording_mode("continuous")
    for start in range(0, 48_000, 960):
        store.append_audio(block(session, start))
    store.append_result(DecisionRecord(
        session, 0, 0, 48_000, (46_080, 48_000), (32_640, 48_000), "error",
        diagnostics=("l2_stage=dropped:l2_admission_queue_overflow",),
        stage_statuses={"l2": "dropped", "l3": "dropped", "l4": "dropped"},
        terminal_reason="l2_admission_queue_overflow",
    ))
    store.advance_result_watermark(ResultWatermark(
        session, 0, 48_000,
        ({"window_id": 0, "decision_sample": 48_000,
          "reason": "l2_admission_queue_overflow"},),
    ))

    manifest = store.stop_session("normal")
    chunk = manifest["chunks"][0]
    assert chunk["result_count"] == 1
    root = next(tmp_path.glob(f"runtime_sessions/*/*/{session}"))
    result_path = root / next(
        item["path"] for item in chunk["assets"] if item["kind"] == "results"
    )
    rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[1]["stage_statuses"] == {
        "l2": "dropped", "l3": "dropped", "l4": "dropped",
    }
    assert rows[1]["schema_version"] == "decision_record_v4"
    store.close()


def test_runtime_records_exact_algorithm_sidecars_on_absolute_timeline(tmp_path: Path):
    from data_management import ResultWatermark

    session = str(uuid.uuid4())
    store = RecordingStore(tmp_path, chunk_seconds=1, min_free_storage_gb=0)
    store.start_session(session, SessionMetadata("a", "b", algorithm_versions={"imcra": "imcra_v1"}))
    store.set_recording_mode("continuous")
    for start in range(0, 48000, 960):
        store.append_audio(block(session, start))
    waveform = np.linspace(-0.25, 0.25, 15360, dtype=np.float32)
    store.append_result(DecisionRecord(
        session, 0, 7, 48000, (46080, 48000), (32640, 48000), "ok",
        candidates=({"track_id": 7, "measured_theta_deg": 29.0, "theta_deg": 30.0,
                     "track_state": "confirmed", "is_observed": True, "is_new_track": False,
                     "kalman_applied": True, "raw_score": 2.0, "normalized_score": 0.8},),
        active_tracks=({"track_id": 7, "measured_theta_deg": 29.0, "theta_deg": 30.0,
                        "track_state": "confirmed", "is_observed": True},),
        detections=({"track_id": 7, "theta_deg": 30.0, "probability": 0.9,
                     "is_voice": True, "model_id": "primary"},),
        voice_direction_count=1,
        raw_scores=np.arange(360, dtype=np.float32),
        normalized_scores=np.linspace(0, 1, 360, dtype=np.float32),
        gate_decision={
            "backend": "mean_2x20ms_v1", "state": "open", "sound_present": True,
            "reason": "foreground", "diagnostics": [],
            "source_hops": [
                {"start_sample": 46080, "end_sample": 47040, "array_source_probability_20ms": 0.7},
                {"start_sample": 47040, "end_sample": 48000, "array_source_probability_20ms": 0.8},
            ],
        },
        search_diagnostics={"mode": "single_pass", "algorithm_version": "srp_phat_single_pass_v1"},
        enhanced_audio=({
            "track_id": 7, "theta_deg": 30.0, "backend": "frequency_hybrid", "fallback_reason": None,
            "diagnostics": [], "sample_rate": 48000, "start_sample": 32640, "end_sample": 48000,
        },),
        enhanced_waveforms=(waveform,),
        l4_result={
            "primary_model_id": "primary", "threshold": 0.5,
            "predictions": [{"model_id": "primary", "probabilities": [0.9], "latency_ms": 1.2, "metadata": {}}],
            },
        music_algorithm_version="normmusic_incremental_v1",
        model_order={"estimated_sources": 1, "algorithm": "mdl_wax_kailath_v1"},
        music_diagnostics={"valid_frequency_count": 43, "covariance_condition_max": 120.0},
        kalman_applied=True,
        config_revision=3,
        config_hash="config-v4",
        calibration_revision=2,
        calibration_version="calibration-v2",
        calibration_hash="calibration-hash-v2",
    ))
    store.advance_result_watermark(ResultWatermark(session, 0, 48000))
    manifest = store.stop_session()
    assert manifest["status"] == "complete"
    assert manifest["channel_layouts"]["logical_from_native"] == [0, 1, 2, 3, 4, 5, 7, 6]
    chunk = manifest["chunks"][0]
    assert chunk["result_status"] == "complete"
    root = next(tmp_path.glob(f"runtime_sessions/*/*/{session}"))
    assets = chunk["assets"]
    kinds = {item["kind"] for item in assets}
    assert {"gate", "spatial_response", "enhanced_audio", "l4"} <= kinds
    gate_path = root / next(item["path"] for item in assets if item["kind"] == "gate")
    gate = json.loads(gate_path.read_text(encoding="utf-8").splitlines()[1])
    assert gate["decision_sample"] == 48000 and gate["sound_present"] is True
    assert gate["source_hops"][0]["start_sample"] == 46080
    spatial_path = root / next(item["path"] for item in assets if item["kind"] == "spatial_response")
    with np.load(spatial_path) as spatial:
        assert spatial["doa_start_samples"].tolist() == [46080]
        assert spatial["doa_end_samples"].tolist() == [48000]
        assert spatial["theta_degrees"].tolist() == list(range(360))
    enhanced = next(item for item in assets if item["kind"] == "enhanced_audio")
    assert enhanced["track_id"] == 7 and "track000007" in enhanced["path"]
    assert (enhanced["start_sample"], enhanced["end_sample"]) == (32640, 48000)
    with wave.open(str(root / enhanced["path"]), "rb") as wav:
        assert (wav.getnchannels(), wav.getnframes()) == (1, 15360)


def test_result_watermark_gap_is_explicit_not_audio_corruption(tmp_path: Path):
    session = str(uuid.uuid4())
    store = RecordingStore(tmp_path, chunk_seconds=1, min_free_storage_gb=0)
    store.start_session(session, SessionMetadata("a", "b"))
    store.set_recording_mode("continuous")
    store.append_audio(block(session, 0))
    store.append_result(DecisionRecord(session, 0, 1, 960, (0, 960), (0, 960), "ok"))
    manifest = store.stop_session()
    assert manifest["status"] == "result_incomplete"
    assert manifest["chunks"][0]["result_status"] == "result_incomplete"
    assert manifest["result_gaps"][0]["reason"] == "watermark_before_chunk_end"


def test_runtime_records_noise_spectrum_as_audio_aligned_asset(tmp_path: Path):
    session = str(uuid.uuid4())
    frequencies = np.fft.rfftfreq(2048, 1 / 48000).astype(np.float32)
    noise = NoiseSpectrumRecord(np.full((7, frequencies.size), 0.01, np.float32), frequencies, 48000, 2048, 0, 0.0)
    source = block(session, 0)
    source = replace(source, noise_spectrum=noise)
    store = RecordingStore(tmp_path, chunk_seconds=1, min_free_storage_gb=0)
    store.start_session(session, SessionMetadata("a", "b"))
    store.set_recording_mode("continuous")
    store.append_audio(source)
    manifest = store.stop_session()
    root = next(tmp_path.glob(f"runtime_sessions/*/*/{session}"))
    asset = next(x for x in manifest["chunks"][0]["assets"] if x["kind"] == "noise_spectrum")
    with np.load(root / asset["path"]) as recorded:
        assert recorded["psd"].shape == (1, 7, 1025)
        assert recorded["start_samples"].tolist() == [0]
        assert recorded["end_samples"].tolist() == [960]


def test_runtime_records_l1_imcra_exact_fields_without_invented_statistics(tmp_path: Path):
    session = str(uuid.uuid4())
    source = block(session, 0)
    frequencies = np.fft.rfftfreq(2048, 1 / 48000).astype(np.float32)
    frequencies = frequencies[frequencies <= 8_000.0]
    spectral = (7, frequencies.size)
    hop = ImcraHopSnapshot(
        session, 0, 0, 960, (0,), "cohen_imcra_2003_l1_v2", "ready",
        frequencies,
        np.ones(spectral, np.float32),
        np.ones(spectral, np.float32) * 2,
        np.ones(spectral, np.float32) * 1.5,
        np.ones(spectral, np.float32) * 0.5,
        np.ones(spectral, np.float32) * 0.4,
        np.ones(spectral, np.float32) * 0.25,
        np.ones(spectral, np.float32) * 0.75,
        np.ones(spectral, np.float32) * 2,
        np.ones(spectral, np.float32),
        np.ones((7, 4), np.float32),
        np.arange(7, dtype=np.float32),
        np.linspace(0.1, 0.7, 7, dtype=np.float32),
        0.4,
    )
    source = replace(source, imcra_hop=hop)
    store = RecordingStore(tmp_path, min_free_storage_gb=0)
    store.start_session(session, SessionMetadata("a", "b"))
    store.set_recording_mode("continuous")
    store.append_audio(source)
    manifest = store.stop_session()
    root = next(tmp_path.glob(f"runtime_sessions/*/*/{session}"))
    asset = next(item for item in manifest["chunks"][0]["assets"] if item["kind"] == "imcra")
    with np.load(root / asset["path"]) as values:
        assert values["start_samples"].tolist() == [0]
        assert values["end_samples"].tolist() == [960]
        assert values["frequencies_hz"].shape == (342,)
        assert values["noise_psd"].shape == (1, 7, 342)
        assert values["noise_level_db"].shape == (1, 7)
        assert values["source_probability_per_mic"].shape == (1, 7)
        assert values["array_source_probability_20ms"].tolist() == [np.float32(0.4)]
        assert {
            "noise_psd", "smoothed_psd", "conditional_smoothed_psd", "minimum_psd",
            "conditional_minimum_psd", "spp", "speech_absence_probability",
            "posterior_snr", "prior_snr", "noise_features",
        } <= set(values.files)


def test_chunk_streams_large_float_noise_and_imcra_arrays_with_constant_ram_buffer(
    tmp_path: Path,
):
    from data_management.recording_store import _Chunk

    session = str(uuid.uuid4())
    imcra_frequencies = np.fft.rfftfreq(2048, 1 / 48000).astype(np.float32)
    imcra_frequencies = imcra_frequencies[
        imcra_frequencies <= 8_000.0
    ]
    spectral = np.ones((7, imcra_frequencies.size), np.float32)
    noise_frequencies = np.fft.rfftfreq(64, 1 / 48000).astype(np.float32)
    noise_psd = np.ones((7, noise_frequencies.size), np.float32)
    settings = {
        "record_native_8ch": False,
        "record_logical_8ch": False,
        "record_physical_7ch": False,
        "record_physical_float32": True,
        "record_results_jsonl": False,
        "record_spatial_response": False,
        "record_hotmaps": False,
        "record_imcra": True,
        "record_noise_spectrum": True,
    }

    def enriched(start: int) -> IngestedAudioBlock:
        source = block(session, start)
        noise = NoiseSpectrumRecord(
            noise_psd,
            noise_frequencies,
            48_000,
            64,
            start // 960,
            start / 48_000,
        )
        hop = ImcraHopSnapshot(
            session,
            0,
            start,
            start + 960,
            (start // 960,),
            "cohen_imcra_2003_l1_v2",
            "ready",
            imcra_frequencies,
            spectral,
            spectral * 2,
            spectral * 1.5,
            spectral * 0.5,
            spectral * 0.4,
            spectral * 0.25,
            spectral * 0.75,
            spectral * 2,
            spectral,
            np.ones((7, 4), np.float32),
            np.arange(7, dtype=np.float32),
            np.linspace(0.1, 0.7, 7, dtype=np.float32),
            0.4,
        )
        return replace(source, noise_spectrum=noise, imcra_hop=hop)

    first = enriched(0)
    root = tmp_path / "session"
    chunk = _Chunk(root, first, 60 * 48_000, settings, "a", "b")
    for index in range(64):
        chunk.append(first if index == 0 else enriched(index * 960))
        # Only the two shared frequency axes stay resident; every per-hop
        # tensor and physical sample has already been written to a partial.
        assert chunk.resident_array_buffer_bytes < 8_192
        assert not any(
            isinstance(value, np.ndarray)
            for item in chunk.imcra_hops
            for value in item.values()
        )

    info = chunk.close(session)
    assets = {item["kind"]: root / item["path"] for item in info["assets"]}
    assert np.load(assets["physical_float"]).shape == (64 * 960, 7)
    with np.load(assets["noise_spectrum"]) as noise:
        assert noise["psd"].shape == (64, 7, noise_frequencies.size)
    with np.load(assets["imcra"]) as imcra:
        assert imcra["noise_psd"].shape == (64, 7, imcra_frequencies.size)
        assert imcra["noise_features"].shape == (64, 7, 4)
    assert not list(root.rglob("*.partial"))
    chunk.mark_manifest_committed()


def test_imcra_streaming_closes_under_long_windows_session_path(tmp_path: Path):
    from data_management.recording_store import _Chunk

    session = str(uuid.uuid4())
    frequencies = np.fft.rfftfreq(2048, 1 / 48_000).astype(np.float32)
    frequencies = frequencies[frequencies <= 8_000.0]
    spectral = np.ones((7, frequencies.size), np.float32)
    hop = ImcraHopSnapshot(
        session_id=session,
        stream_epoch=0,
        start_sample=0,
        end_sample=960,
        source_sequence_ids=(0,),
        algorithm_version="cohen_imcra_2003_l1_v2",
        state="ready",
        frequencies_hz=frequencies,
        noise_psd=spectral,
        smoothed_psd=spectral,
        conditional_smoothed_psd=spectral,
        minimum_psd=spectral,
        conditional_minimum_psd=spectral,
        spp=spectral * 0.5,
        speech_absence_probability=spectral * 0.5,
        posterior_snr=spectral,
        prior_snr=spectral,
        noise_features=np.ones((7, 4), np.float32),
        noise_level_db=np.zeros(7, np.float32),
        source_probability_per_mic=np.full(7, 0.5, np.float32),
        array_source_probability_20ms=0.5,
    )
    source = replace(block(session, 0), imcra_hop=hop)
    settings = {
        "record_native_8ch": False,
        "record_logical_8ch": False,
        "record_physical_7ch": False,
        "record_physical_float32": False,
        "record_results_jsonl": False,
        "record_spatial_response": False,
        "record_hotmaps": False,
        "record_imcra": True,
        "record_noise_spectrum": False,
    }
    # This depth left enough room for the public final filename but exceeded
    # legacy MAX_PATH with the former descriptive IMCRA staging filenames.
    root = tmp_path / ("runtime-root-" + "x" * 24) / ("session-" + "y" * 24)
    chunk = _Chunk(root, source, 48_000, settings, "a", "b")
    chunk.append(source)
    assert max(len(str(spool.path)) for spool in chunk._imcra_spools.values()) < 240
    info = chunk.close(session)
    asset = next(item for item in info["assets"] if item["kind"] == "imcra")
    with np.load(root / asset["path"]) as archive:
        assert archive["speech_absence_probability"].shape == (1, 7, frequencies.size)
    chunk.mark_manifest_committed()


def test_manual_pause_creates_real_intervals(tmp_path: Path):
    session = str(uuid.uuid4())
    store = RecordingStore(tmp_path, chunk_seconds=60, min_free_storage_gb=0)
    store.start_session(session, SessionMetadata("a", "b"))
    store.set_recording_mode("manual")
    store.start_recording()
    store.append_audio(block(session, 0))
    store.pause_recording()
    store.append_audio(block(session, 960))
    store.start_recording()
    store.append_audio(block(session, 1920))
    manifest = store.stop_session()
    assert [(x["start_sample"], x["end_sample"]) for x in manifest["recorded_intervals"]] == [(0, 960), (1920, 2880)]


def test_promote_annotations_qa_trash_and_export(tmp_path: Path):
    session = str(uuid.uuid4())
    catalog = Catalog(tmp_path / "catalog.sqlite")
    store = RecordingStore(tmp_path, catalog=catalog, chunk_seconds=1, min_free_storage_gb=0)
    store.start_session(session, SessionMetadata("a", "b"))
    store.set_recording_mode("continuous")
    for start in range(0, 48000, 960):
        store.append_audio(block(session, start))
    store.stop_session()
    corpus = CorpusStore(tmp_path, catalog=catalog)
    dataset = str(uuid.uuid4())
    rid = corpus.promote_runtime_segment(session, 0, 0, 9600, metadata(dataset))
    root = tmp_path / "test_corpus" / dataset / "recordings" / rid
    manifest = json.loads((root / "recording_manifest.json").read_text(encoding="utf-8"))
    assert manifest["lineage"]["start_sample"] == 0 and manifest["duration_samples"] == 9600
    annotation = Annotation(str(uuid.uuid4()), rid, 0, 4800, "voice_activity", "voice", 30, 0.9, "ann-a", "v0001")
    corpus.add_annotations(rid, (annotation,))
    assert (root / "annotations" / "v0001.jsonl").exists()
    archive = export_assets([root], tmp_path / "export.zip")
    assert verify_export(archive)
    trash = move_to_trash(tmp_path, root, entity_type="recording", entity_id=rid, catalog=catalog)
    assert not root.exists()
    assert catalog.list_recordings() == []
    assert restore_from_trash(trash.parent, catalog=catalog) == root
    assert [row["id"] for row in catalog.list_recordings()] == [rid]


def test_service_reconciles_recordings_trashed_by_older_builds(tmp_path: Path):
    service = DataManagerService(tmp_path)
    dataset = str(uuid.uuid4())
    root = tmp_path / "test_corpus" / dataset / "recordings" / str(uuid.uuid4())
    root.mkdir(parents=True)
    manifest = {
        "schema_version": "test_recording_v1",
        "dataset_id": dataset,
        "recording_id": root.name,
        "source_type": "dedicated",
        "quality_status": "pending",
        "split": "unset",
        "assets": [],
    }
    from data_management.manifests import write_manifest

    write_manifest(root / "recording_manifest.json", manifest)
    service.catalog.upsert_dataset(dataset, root.parents[1])
    service.catalog.upsert_recording(manifest, root)
    service.trash("recording", root.name)
    service.catalog.mark_restored("recording", root.name)  # Reproduce the stale row written by the old implementation.
    service.close()

    reopened = DataManagerService(tmp_path)
    assert reopened.recordings() == []
    assert len(reopened.trash_operations()) == 1
    reopened.close()


def test_leakage_detects_room_session_and_speaker():
    rows = [
        {"split": "train", "capture_session_id": "s1", "room_id": "r1", "speaker_ids_anonymous": ["p1"]},
        {"split": "test", "capture_session_id": "s2", "room_id": "r1", "speaker_ids_anonymous": []},
    ]
    report = leakage_check(rows)
    assert not report["passed"] and report["leaks"][0]["field"] == "room_id"


def test_event_requires_valid_voice_and_writes_preroll(tmp_path: Path):
    session = str(uuid.uuid4())
    store = RecordingStore(tmp_path, chunk_seconds=60, pre_roll_seconds=2, post_roll_seconds=3, min_free_storage_gb=0)
    store.start_session(session, SessionMetadata("a", "b"))
    store.set_recording_mode("event")
    for start in range(0, 3 * 48000, 960):
        store.append_audio(block(session, start))
    invalid = DecisionRecord(
        session, 0, 1, 3 * 48000, (0, 1), (0, 1), "ok",
        candidates=({"theta_deg": 0.0},),
        detections=({
            "theta_deg": 0.0,
            "probability": 0.1,
            "is_voice": False,
            "model_id": "test",
        },),
        voice_direction_count=0,
    )
    store.trigger_event(invalid)
    store.trigger_event(replace(
        invalid,
        window_id=2,
        detections=({
            "theta_deg": 0.0,
            "probability": 0.9,
            "is_voice": True,
            "model_id": "test",
        },),
        voice_direction_count=1,
    ))
    for start in range(3 * 48000, 6 * 48000, 960):
        store.append_audio(block(session, start))
    manifest = store.stop_session()
    assert len(manifest["event_triggers"]) == 1
    assert manifest["recorded_intervals"][0] == {
        "stream_epoch": 0,
        "start_sample": 48000,
        "end_sample": 6 * 48000,
    }


def test_event_triggers_merge_audit_and_scan_storage_once_per_segment(tmp_path: Path):
    session = str(uuid.uuid4())
    store = RecordingStore(
        tmp_path,
        chunk_seconds=60,
        pre_roll_seconds=2,
        post_roll_seconds=3,
        min_free_storage_gb=0,
    )
    store.start_session(session, SessionMetadata("a", "b"))
    store.set_recording_mode("event")
    scans = 0

    def count_scan() -> None:
        nonlocal scans
        scans += 1

    store._ensure_storage = count_scan

    def trigger(window_id: int, decision_sample: int) -> None:
        store.trigger_event({
            "session_id": session,
            "stream_epoch": 0,
            "window_id": window_id,
            "decision_sample": decision_sample,
            "status": "ok",
            "voice_direction_count": 1,
        })

    trigger(1, 3 * 48_000)
    for window_id in range(2, 102):
        trigger(window_id, 3 * 48_000 + (window_id - 1) * 960)

    audit = store._manifest["event_triggers"]
    assert scans == 1
    assert len(audit) == 1
    assert audit[0]["first_window_id"] == 1
    assert audit[0]["last_window_id"] == 101
    assert audit[0]["trigger_count"] == 101

    # Once the next trigger's pre-roll no longer overlaps the current event,
    # it starts one new segment and performs exactly one new capacity scan.
    next_decision = int(audit[0]["end_sample"]) + 2 * 48_000 + 960
    trigger(102, next_decision)
    assert scans == 2
    assert len(store._manifest["event_triggers"]) == 2
    store.stop_session()
    store.close()


def test_merged_event_replays_ring_gap_after_previous_postroll(tmp_path: Path):
    session = str(uuid.uuid4())
    store = RecordingStore(
        tmp_path,
        chunk_seconds=60,
        pre_roll_seconds=2,
        post_roll_seconds=3,
        min_free_storage_gb=0,
    )
    store.start_session(session, SessionMetadata("a", "b"))
    store.set_recording_mode("event")
    for start in range(0, 3 * 48_000, 960):
        store.append_audio(block(session, start))

    first = {
        "session_id": session,
        "stream_epoch": 0,
        "window_id": 1,
        "decision_sample": 3 * 48_000,
        "status": "ok",
        "voice_direction_count": 1,
    }
    store.trigger_event(first)
    # Audio from 6 to 7 seconds is initially outside the first post-roll and
    # therefore exists only in the event ring.
    for start in range(3 * 48_000, 7 * 48_000, 960):
        store.append_audio(block(session, start))
    store.trigger_event({**first, "window_id": 2, "decision_sample": 7 * 48_000})
    for start in range(7 * 48_000, 10 * 48_000, 960):
        store.append_audio(block(session, start))

    manifest = store.stop_session()
    assert len(manifest["event_triggers"]) == 1
    assert manifest["event_triggers"][0]["trigger_count"] == 2
    assert manifest["recorded_intervals"] == [{
        "stream_epoch": 0,
        "start_sample": 48_000,
        "end_sample": 10 * 48_000,
    }]
    store.close()


def test_off_session_persists_zero_asset_lifecycle_manifest_and_catalog(tmp_path: Path):
    session = str(uuid.uuid4())
    store = RecordingStore(tmp_path, min_free_storage_gb=0)
    store.start_session(session, SessionMetadata("a", "b"))
    store.set_recording_mode("off")
    store.append_audio(block(session, 0))
    manifest = store.stop_session()
    assert not manifest["chunks"]
    root = next(tmp_path.glob(f"runtime_sessions/*/*/{session}"))
    persisted = json.loads((root / "session_manifest.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "complete" and persisted["chunks"] == []
    row = next(item for item in store.catalog.list_sessions() if item["id"] == session)
    assert row["status"] == "complete"
    store.close()


def _result_retention_test_settings(*, hotmaps: bool = False) -> dict[str, bool]:
    return {
        "record_native_8ch": False,
        "record_logical_8ch": False,
        "record_physical_7ch": False,
        "record_physical_float32": False,
        "record_results_jsonl": True,
        "record_spatial_response": False,
        "record_hotmaps": hotmaps,
        "record_imcra": False,
        "record_noise_spectrum": False,
    }


def _decision_mapping(session: str, window_id: int, decision: int, *, waveform_samples: int = 0):
    waveform = np.ones(waveform_samples, np.float32) if waveform_samples else None
    return {
        "session_id": session,
        "stream_epoch": 0,
        "window_id": window_id,
        "decision_sample": decision,
        "doa_range": (max(0, decision - 1920), decision),
        "context_range": (max(0, decision - 15360), decision),
        "status": "ok",
        "enhanced_audio": () if waveform is None else ({"track_id": 1, "theta_deg": 0.0},),
        "enhanced_waveforms": () if waveform is None else (waveform,),
    }


def test_results_stream_to_complete_chunks_and_release_waveform_memory(tmp_path: Path):
    from data_management import ResultWatermark

    session = str(uuid.uuid4())
    store = RecordingStore(
        tmp_path,
        chunk_seconds=1,
        min_free_storage_gb=0,
        settings=_result_retention_test_settings(),
    )
    store.start_session(session, SessionMetadata("a", "b"))
    store.set_recording_mode("continuous")
    for index, start in enumerate(range(0, 2 * 48_000, 960)):
        store.append_audio(block(session, start))
        decision = start + 960
        assert store.append_result_with_watermark(
            _decision_mapping(session, index, decision, waveform_samples=32),
            ResultWatermark(session, 0, decision),
        )

    deadline = time.monotonic() + 5.0
    while (
        len(store._result_assets_finalized) < 2
        and store._worker_error is None
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    assert store._worker_error is None
    assert len(store._result_assets_finalized) == 2
    assert store._all_results == []

    manifest = store.stop_session()
    assert [chunk["result_count"] for chunk in manifest["chunks"]] == [50, 50]
    assert all(chunk["result_status"] == "complete" for chunk in manifest["chunks"])
    root = next(tmp_path.glob(f"runtime_sessions/*/*/{session}"))
    for chunk in manifest["chunks"]:
        result_asset = next(item for item in chunk["assets"] if item["kind"] == "results")
        assert len((root / result_asset["path"]).read_text(encoding="utf-8").splitlines()) == 51
    store.close()


def test_off_results_are_discarded_and_event_preroll_is_bounded(tmp_path: Path):
    from data_management import ResultWatermark

    session = str(uuid.uuid4())
    store = RecordingStore(
        tmp_path,
        pre_roll_seconds=2,
        min_free_storage_gb=0,
        settings=_result_retention_test_settings(),
    )
    store.start_session(session, SessionMetadata("a", "b"))
    large = _decision_mapping(session, 0, 960, waveform_samples=15_360)
    for _ in range(200):
        assert not store.append_result(large)
    assert store.result_queue.empty()
    assert store._all_results == []

    store.set_recording_mode("event")
    accepted_count = 0
    for index in range(300):
        decision = (index + 1) * 960
        accepted_count += int(store.append_result_with_watermark(
            _decision_mapping(session, index, decision, waveform_samples=32),
            ResultWatermark(session, 0, decision),
        ))
        if index % 25 == 0:
            time.sleep(0.01)
    deadline = time.monotonic() + 2.0
    while not store.result_queue.empty() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert store.result_queue.empty()
    assert accepted_count > 0
    assert len(store._all_results) <= 100
    if accepted_count < 300:
        assert store._manifest["result_gaps"][-1]["reason"] == "result_overflow"
    store.stop_session()
    store.close()


def test_atomic_result_watermark_never_advances_on_queue_overflow(tmp_path: Path):
    from data_management import ResultWatermark

    session = str(uuid.uuid4())
    store = RecordingStore(tmp_path, result_queue_capacity=1, min_free_storage_gb=0)
    # Keep the idle writer from consuming the deliberately filled slot.
    store._drain_results = lambda: None
    store.start_session(session, SessionMetadata("a", "b"))
    store.set_recording_mode("continuous")
    store._accept_interval(0, 0, 960)
    store.result_queue.put_nowait(("result", {"window_id": -1}))
    accepted = store.append_result_with_watermark(
        _decision_mapping(session, 1, 960),
        ResultWatermark(session, 0, 960),
    )
    assert not accepted
    assert store._last_watermark == {}
    gap = store._manifest["result_gaps"][0]
    assert gap["reason"] == "result_overflow"
    assert (gap["gap_start_sample"], gap["gap_end_sample"]) == (-1, 960)
    store.stop_session()
    store.close()


def test_hotmaps_are_streamed_instead_of_retained_for_the_session(tmp_path: Path):
    session = str(uuid.uuid4())
    store = RecordingStore(
        tmp_path,
        chunk_seconds=1,
        min_free_storage_gb=0,
        settings=_result_retention_test_settings(hotmaps=True),
    )
    store.start_session(session, SessionMetadata("a", "b"))
    store.set_recording_mode("continuous")
    hotmap = np.arange(256, dtype=np.uint8).reshape(16, 16)
    for start in range(0, 2 * 48_000, 960):
        store.append_audio(replace(block(session, start), hotmap=hotmap))
    manifest = store.stop_session()
    assert manifest["hotmaps"]["count"] == 100
    root = next(tmp_path.glob(f"runtime_sessions/*/*/{session}"))
    hotmap_path = root / manifest["hotmaps"]["path"]
    assert len(hotmap_path.read_text(encoding="utf-8").splitlines()) == 100
    assert store._hotmap_sequences == set()
    assert not hasattr(store, "_hotmaps")
    store.close()


def test_hotmap_is_deduplicated_by_sequence(tmp_path: Path):
    session = str(uuid.uuid4())
    store = RecordingStore(tmp_path, min_free_storage_gb=0)
    store.start_session(session, SessionMetadata("a", "b"))
    store.set_recording_mode("continuous")
    source = block(session, 0)
    hot = np.arange(256, dtype=np.uint8).reshape(16, 16)
    first = IngestedAudioBlock(
        source.session_id,
        source.stream_epoch,
        source.start_sample,
        source.end_sample,
        source.sample_rate,
        source.sequence_id,
        source.timestamp,
        source.samples,
        source.native_samples,
        hot,
    )
    store.append_audio(first)
    manifest = store.stop_session()
    assert manifest["hotmaps"]["count"] == 1
    root = next(tmp_path.glob(f"runtime_sessions/*/*/{session}"))
    assert len((root / "hotmaps.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def wizard_block(session: str, start: int, rng: np.random.Generator) -> IngestedAudioBlock:
    physical = rng.normal(0, 0.02, (960, 7)).astype(np.float32)
    mix = rng.normal(0, 0.02, 960).astype(np.float32)
    native = np.column_stack((physical[:, :6], mix, physical[:, 6]))
    logical = np.column_stack((physical, mix))
    from layer1_input.interface import CdcHotmapFrame

    sequence = start // 960
    hotmap = CdcHotmapFrame(np.full((16, 16), sequence, dtype=np.uint8), sequence, start / 48000)
    return IngestedAudioBlock(
        session, 0, start, start + 960, 48000, sequence, start / 48000,
        logical, native, hotmap,
    )


def test_dedicated_recording_streams_complete_raw_input_with_pause_resume(tmp_path: Path):
    catalog = Catalog(tmp_path / "catalog.sqlite")
    controller = DedicatedRecordingController(tmp_path, catalog)
    dataset = str(uuid.uuid4())
    controller.begin(WizardInput(
        dataset, "room", "quiet", "pose", 1, "granted", ("research",),
        (30.0,), (1.0,), recording_name="诊室 · 1个声源 · 20260819-120000",
        source_categories=("人声",), source_movements=("静止",), noise_source="空调",
    ))
    rng = np.random.default_rng(42)
    session = str(uuid.uuid4())
    sample = 0
    for _ in range(20):
        controller.append(wizard_block(session, sample, rng))
        sample += 960
    assert controller.phase == WizardPhase.RECORDING
    controller.pause()
    sample += 5 * 960
    assert controller.phase == WizardPhase.PAUSED
    controller.resume()
    for _ in range(10):
        controller.append(wizard_block(session, sample, rng))
        sample += 960
    controller.finish()
    assert controller.phase == WizardPhase.FINALIZING
    recording_id = controller.finalize()
    root = tmp_path / "test_corpus" / dataset / "recordings" / recording_id
    assert controller.phase == WizardPhase.COMPLETE
    assert not (root / "qa_report.json").exists()
    with wave.open(str(root / "native_8ch.wav"), "rb") as recorded:
        assert recorded.getnchannels() == 8
        assert recorded.getnframes() == 30 * 960
    hotmaps = [json.loads(line) for line in (root / "hotmaps.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(hotmaps) == 30
    assert hotmaps[20]["playback_sample"] == 20 * 960
    manifest = json.loads((root / "recording_manifest.json").read_text(encoding="utf-8"))
    assert manifest["quality_status"] == "pending"
    assert manifest["display_name"] == "诊室 · 1个声源 · 20260819-120000"
    assert manifest["environment_id"] == "quiet"
    assert manifest["source_count"] == 1
    assert manifest["source_categories"] == ["人声"]
    assert manifest["source_movements"] == ["静止"]
    assert manifest["noise_source"] == "空调"
    assert manifest["algorithm_direction_ids"] == {
        "status": "not_available",
        "reason": "l1_only_recording",
        "display_text": "无算法方向ID",
    }
    assert {item["kind"] for item in manifest["assets"]} == {"native_8ch", "cdc_hotmaps", "labels"}
    labels = json.loads((root / "labels.json").read_text(encoding="utf-8"))
    assert labels["schema_version"] == "test_recording_labels_v3"
    assert labels["environment"] == "quiet"
    assert labels["sources"] == [{"index": 1, "type": "人声", "movement": "静止"}]
    assert labels["noise_source"] == "空调"
    assert labels["duration_seconds"] == 30 * 960 / 48_000
    assert len(labels["recorded_intervals"]) == 2


def test_experiment_snapshot_locks_dataset_and_recording(tmp_path: Path):
    catalog = Catalog(tmp_path / "catalog.sqlite")
    dataset = str(uuid.uuid4())
    root = tmp_path / "test_corpus" / dataset / "recordings" / str(uuid.uuid4())
    root.mkdir(parents=True)
    manifest = {
        "schema_version": "test_recording_v1",
        "dataset_id": dataset,
        "recording_id": root.name,
        "source_type": "dedicated",
        "quality_status": "passed",
        "split": "train",
        "assets": [],
    }
    from data_management.manifests import write_manifest

    write_manifest(root / "recording_manifest.json", manifest)
    catalog.upsert_dataset(dataset, root.parents[1])
    catalog.upsert_recording(manifest, root)
    store = ExperimentStore(tmp_path, catalog)
    experiment = store.create_snapshot(
        name="baseline",
        dataset_id=dataset,
        dataset_version="1.0.0",
        config_hash="a" * 64,
        model_version="voice-v1",
        recording_ids=(root.name,),
    )
    assert (tmp_path / "experiments" / experiment / "experiment_manifest.json").exists()
    assert catalog.get_dataset(dataset)["locked"] == 1
    assert catalog.recording_is_experiment_locked(root.name)


def test_runtime_recording_stop_is_bounded_after_writer_failure_with_full_queue(tmp_path: Path):
    session = str(uuid.uuid4())
    store = RecordingStore(
        tmp_path, audio_queue_seconds=0, min_free_storage_gb=0,
    )
    store.start_session(session, SessionMetadata("a", "b"))
    store.set_recording_mode("continuous")

    def fail_write(_block):
        raise OSError("simulated recording writer failure")

    store._write_audio = fail_write
    store.append_audio(block(session, 0))
    deadline = time.monotonic() + 2.0
    while store._worker_error is None and time.monotonic() < deadline:
        time.sleep(0.005)
    assert isinstance(store._worker_error, OSError)
    assert store._thread is not None and not store._thread.is_alive()

    # Reproduce the previous deadlock condition: the dead writer can no
    # longer consume this final slot, so a sentinel put would block forever.
    store.append_audio(block(session, 960))
    assert store.audio_queue.full()
    started = time.monotonic()
    manifest = store.stop_session("writer_failure")

    assert time.monotonic() - started < 1.0
    assert manifest["status"] == "corrupt"
    assert "simulated recording writer failure" in manifest["writer_error"]
    assert store.audio_queue.empty()

    # Stale commands from the failed session must not poison a restart.
    replacement = str(uuid.uuid4())
    store.start_session(replacement, SessionMetadata("c", "d"))
    restarted = store.stop_session("normal")
    assert restarted["session_id"] == replacement
    assert restarted["status"] == "complete"
    store.close()


def test_direct_queue_capacities_are_bounded_and_full_result_skips_waveform_copy(
    tmp_path: Path,
):
    small = RecordingStore(tmp_path / "small", result_queue_capacity=0, min_free_storage_gb=0)
    large = RecordingStore(tmp_path / "large", result_queue_capacity=100_000, min_free_storage_gb=0)
    assert small.result_queue.maxsize == 1
    assert large.result_queue.maxsize == RecordingStore.MAX_RESULT_QUEUE_ITEMS == 256
    small.close()
    large.close()

    session = str(uuid.uuid4())
    store = RecordingStore(tmp_path / "active", result_queue_capacity=1, min_free_storage_gb=0)
    store._drain_results = lambda: None
    store.start_session(session, SessionMetadata("a", "b"))
    store.set_recording_mode("continuous")
    store._accept_interval(0, 0, 960)
    store.result_queue.put_nowait(("result", {"window_id": 0}))
    store._prepare_result = lambda _result: (_ for _ in ()).throw(
        AssertionError("full queue must be rejected before copying waveforms")
    )
    accepted = store.append_result(
        _decision_mapping(session, 1, 960, waveform_samples=15_360)
    )
    assert accepted is False
    assert store._manifest["result_gaps"][-1]["reason"] == "result_overflow"
    store.stop_session()
    store.close()


def test_oversized_audio_block_is_split_to_bounded_queue_and_overflow_is_coalesced(
    tmp_path: Path,
):
    session = str(uuid.uuid4())
    store = RecordingStore(
        tmp_path,
        audio_queue_seconds=0,
        min_free_storage_gb=0,
        settings=_result_retention_test_settings(),
    )
    write_started = threading.Event()
    release = threading.Event()

    def slow_write(_block):
        write_started.set()
        release.wait(2.0)

    store._write_audio = slow_write
    store.start_session(session, SessionMetadata("a", "b"))
    store.set_recording_mode("continuous")
    store.append_audio(block(session, 0, frames=8 * 960))
    assert write_started.wait(1.0)
    with store.audio_queue.mutex:
        queued = tuple(store.audio_queue.queue)
    assert all(
        kind != "audio" or value.end_sample - value.start_sample <= 960
        for kind, value in queued
    )
    missing = store._manifest["missing_intervals"]
    # The writer may consume one extra hop during append_audio.  Accepted
    # audio separates real missing ranges, but 6-7 rejected hops must never
    # create one manifest row per hop.
    assert 1 <= len(missing) <= 2
    assert all(item["reason"] == "recording_overflow" for item in missing)
    assert sum(item["end_sample"] - item["start_sample"] for item in missing) >= 6 * 960
    release.set()
    store.stop_session()
    store.close()


def test_sealed_chunk_persists_open_manifest_before_session_stop(tmp_path: Path):
    session = str(uuid.uuid4())
    store = RecordingStore(tmp_path, chunk_seconds=1, min_free_storage_gb=0)
    store.start_session(session, SessionMetadata("a", "b"))
    store.set_recording_mode("continuous")
    for start in range(0, 48_000, 960):
        store.append_audio(block(session, start))
    root = next(tmp_path.glob(f"runtime_sessions/*/*/{session}"))
    manifest_path = root / "session_manifest.json"
    deadline = time.monotonic() + 3.0
    open_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    while not open_manifest["chunks"] and time.monotonic() < deadline:
        time.sleep(0.01)
        try:
            open_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # Windows may briefly deny/read an atomically replaced manifest.
            # Retry until the writer's checkpoint transaction is visible.
            continue
    assert open_manifest["status"] == "open"
    assert len(open_manifest["chunks"]) == 1
    for asset in open_manifest["chunks"][0]["assets"]:
        if "_partial_path" not in asset:
            assert (root / asset["path"]).exists()
    store.stop_session()
    store.close()


def test_recording_store_close_never_hides_a_stuck_writer(
    tmp_path: Path, monkeypatch,
) -> None:
    import pytest

    store = RecordingStore(tmp_path, min_free_storage_gb=0)
    store._session_id = "stuck-session"

    class _StuckWriter:
        @staticmethod
        def is_alive() -> bool:
            return True

    store._thread = _StuckWriter()  # type: ignore[assignment]
    monkeypatch.setattr(store, "stop_session", lambda _reason="normal": {"status": "incomplete"})

    with pytest.raises(RuntimeError, match="writer/session is still active"):
        store.close()

    # Explicit test cleanup: close() correctly left the Catalog open while a
    # writer was reported alive, so the fixture now releases the fake owner.
    store._thread = None
    store._session_id = None
    store.catalog.close()


def test_full_dead_writer_never_blocks_mode_boundary_or_audio_ingest(tmp_path: Path):
    session = str(uuid.uuid4())
    store = RecordingStore(tmp_path, audio_queue_seconds=0, min_free_storage_gb=0)
    store.start_session(session, SessionMetadata("a", "b"))
    store.set_recording_mode("continuous")

    def fail_write(_block):
        raise OSError("dead writer")

    store._write_audio = fail_write
    store.append_audio(block(session, 0))
    deadline = time.monotonic() + 2.0
    while store._worker_error is None and time.monotonic() < deadline:
        time.sleep(0.005)
    assert isinstance(store._worker_error, OSError)

    # Fill the only queue slot after the consumer has died.
    store.append_audio(block(session, 960))
    assert store.audio_queue.full()

    started = time.monotonic()
    store.set_recording_mode("off")
    assert time.monotonic() - started < 0.1
    assert any(
        item["reason"] == "recording_boundary_overflow"
        for item in store._manifest["missing_intervals"]
    )

    store.set_recording_mode("continuous")
    started = time.monotonic()
    store.append_audio(block(session, 1920))
    assert time.monotonic() - started < 0.1
    assert any(
        item["reason"] == "recording_overflow"
        for item in store._manifest["missing_intervals"]
    )
    store.stop_session("dead_writer")
    store.close()


def test_storage_capacity_scan_never_holds_the_ingest_hot_lock(tmp_path: Path):
    session = str(uuid.uuid4())
    store = RecordingStore(tmp_path, min_free_storage_gb=0)
    store.start_session(session, SessionMetadata("a", "b"))
    store.set_recording_mode("manual")
    original_check = store._ensure_storage

    scan_started = threading.Event()
    release_scan = threading.Event()

    def slow_check():
        scan_started.set()
        assert release_scan.wait(2.0)

    store._ensure_storage = slow_check
    starter = threading.Thread(target=store.start_recording)
    starter.start()
    assert scan_started.wait(1.0)
    started = time.monotonic()
    store.append_audio(block(session, 0))
    assert time.monotonic() - started < 0.1
    release_scan.set()
    starter.join(1.0)
    assert not starter.is_alive() and store.manual_active

    store._ensure_storage = original_check
    store.pause_recording()
    store.set_recording_mode("event")
    scan_started.clear()
    release_scan.clear()
    store._ensure_storage = slow_check
    event_result = {
        "session_id": session,
        "stream_epoch": 0,
        "window_id": 1,
        "decision_sample": 1920,
        "status": "ok",
        "voice_direction_count": 1,
    }
    trigger = threading.Thread(target=store.trigger_event, args=(event_result,))
    trigger.start()
    assert scan_started.wait(1.0)
    started = time.monotonic()
    store.append_audio(block(session, 960))
    assert time.monotonic() - started < 0.1
    release_scan.set()
    trigger.join(1.0)
    assert not trigger.is_alive()
    assert len(store._manifest["event_triggers"]) == 1
    store._ensure_storage = original_check
    store.stop_session()
    store.close()


def test_enhanced_wav_stays_partial_until_manifest_transaction(tmp_path: Path):
    from data_management import ResultWatermark

    session = str(uuid.uuid4())
    store = RecordingStore(
        tmp_path,
        chunk_seconds=1,
        min_free_storage_gb=0,
        settings=_result_retention_test_settings(),
    )
    store.start_session(session, SessionMetadata("a", "b"))
    store.set_recording_mode("continuous")
    for start in range(0, 48_000, 960):
        store.append_audio(block(session, start))
    assert store.append_result_with_watermark(
        _decision_mapping(session, 1, 48_000, waveform_samples=320),
        ResultWatermark(session, 0, 48_000),
    )
    deadline = time.monotonic() + 3.0
    while not store._result_assets_finalized and time.monotonic() < deadline:
        time.sleep(0.01)
    assert store._result_assets_finalized
    root = next(tmp_path.glob(f"runtime_sessions/*/*/{session}"))
    asset = next(
        item
        for item in store._manifest["chunks"][0]["assets"]
        if item["kind"] == "enhanced_audio"
    )
    final_path = root / asset["path"]
    partial_path = root / asset["_partial_path"]
    assert partial_path.exists() and not final_path.exists()

    manifest = store.stop_session()
    final_asset = next(
        item
        for item in manifest["chunks"][0]["assets"]
        if item["kind"] == "enhanced_audio"
    )
    assert "_partial_path" not in final_asset
    assert final_path.exists() and not partial_path.exists()
    assert not (root / "enhanced_asset_commit.json").exists()
    store.close()


def test_recovery_quarantines_renamed_enhanced_asset_without_manifest(tmp_path: Path):
    from data_management.manifests import atomic_json

    root = tmp_path / "runtime_sessions" / "2026" / "08" / str(uuid.uuid4())
    final_path = root / "enhanced_audio" / "orphan.wav"
    partial_path = Path(str(final_path) + ".partial")
    partial_path.parent.mkdir(parents=True)
    partial_path.write_bytes(b"prepared-wave")
    journal = root / "enhanced_asset_commit.json"
    atomic_json(journal, {
        "schema_version": "enhanced_asset_commit_v1",
        "entries": [{
            "partial_path": str(partial_path.relative_to(root)),
            "final_path": str(final_path.relative_to(root)),
            "sha256": "unused",
        }],
    })
    partial_path.replace(final_path)  # crash after rename, before session manifest

    store = RecordingStore(tmp_path, min_free_storage_gb=0)
    recovered = store.recover_partials()
    assert len(recovered) == 2
    assert not final_path.exists() and not journal.exists()
    assert any(path.read_bytes() == b"prepared-wave" for path in recovered)
    store.close()


def test_chunk_commit_journal_quarantines_all_assets_after_nth_rename_failure(
    tmp_path: Path,
    monkeypatch,
):
    import pytest

    from data_management.manifests import write_manifest
    from data_management.recording_store import _Chunk

    session = str(uuid.uuid4())
    root = tmp_path / "runtime_sessions" / "2026" / "08" / session
    write_manifest(root / "session_manifest.json", {
        "schema_version": "audio_session_v2",
        "session_id": session,
        "status": "open",
        "chunks": [],
        "missing_intervals": [],
        "result_gaps": [],
    })
    settings = {
        "record_native_8ch": True,
        "record_logical_8ch": False,
        "record_physical_7ch": True,
        "record_physical_float32": True,
        "record_results_jsonl": False,
        "record_spatial_response": False,
        "record_hotmaps": False,
        "record_imcra": False,
        "record_noise_spectrum": False,
    }
    source = block(session, 0)
    chunk = _Chunk(root, source, 48_000, settings, "a", "b")
    chunk.append(source)
    original = _Chunk._rename_prepared_asset
    rename_count = 0

    def fail_third_rename(partial: Path, final: Path) -> None:
        nonlocal rename_count
        rename_count += 1
        if rename_count == 3:
            raise OSError("simulated power loss on third asset rename")
        original(partial, final)

    monkeypatch.setattr(_Chunk, "_rename_prepared_asset", staticmethod(fail_third_rename))
    with pytest.raises(OSError, match="third asset rename"):
        chunk.close(session)
    journal = next(root.glob(".cj_*.json"))
    transaction = json.loads(journal.read_text(encoding="utf-8"))
    assert rename_count == 3 and len(transaction["entries"]) == 3

    store = RecordingStore(tmp_path, min_free_storage_gb=0)
    recovered = store.recover_partials()
    assert recovered
    for entry in transaction["entries"]:
        assert not (root / entry["partial_path"]).exists()
        assert not (root / entry["final_path"]).exists()
    assert not journal.exists()
    repaired = json.loads((root / "session_manifest.json").read_text(encoding="utf-8"))
    assert repaired["status"] == "incomplete"
    row = next(item for item in store.catalog.list_sessions() if item["id"] == session)
    assert row["status"] == "incomplete"
    store.close()


def test_open_session_recovery_removes_provisional_and_invalid_assets(tmp_path: Path):
    from data_management.manifests import write_manifest

    session = str(uuid.uuid4())
    root = tmp_path / "runtime_sessions" / "2026" / "08" / session
    enhanced = root / "enhanced_audio" / "pending.wav.partial"
    enhanced.parent.mkdir(parents=True)
    enhanced.write_bytes(b"pending")
    invalid = root / "physical_7ch" / "bad.wav"
    invalid.parent.mkdir(parents=True)
    invalid.write_bytes(b"corrupt")
    write_manifest(root / "session_manifest.json", {
        "schema_version": "audio_session_v2",
        "session_id": session,
        "status": "open",
        "chunks": [{
            "stream_epoch": 0,
            "start_sample": 0,
            "end_sample": 960,
            "assets": [
                {
                    "kind": "enhanced_audio",
                    "path": "enhanced_audio/pending.wav",
                    "_partial_path": "enhanced_audio/pending.wav.partial",
                    "sha256": "unused",
                    "stream_epoch": 0,
                    "window_id": 1,
                    "decision_sample": 960,
                },
                {
                    "kind": "physical_7ch",
                    "path": "physical_7ch/bad.wav",
                    "sha256": "0" * 64,
                },
            ],
        }],
        "missing_intervals": [],
        "result_gaps": [],
    })

    store = RecordingStore(tmp_path, min_free_storage_gb=0)
    recovered = store.recover_partials()
    repaired = json.loads((root / "session_manifest.json").read_text(encoding="utf-8"))
    assert repaired["status"] == "incomplete"
    assert repaired["stop_reason"] == "crash_recovery"
    assert repaired["chunks"][0]["assets"] == []
    assert any(
        item["reason"] == "crash_recovery_uncommitted_enhanced_asset"
        for item in repaired["result_gaps"]
    )
    assert any(
        item["reason"] == "crash_recovery_asset_hash_mismatch"
        for item in repaired["missing_intervals"]
    )
    assert not enhanced.exists() and not invalid.exists()
    assert len(recovered) >= 2
    row = next(item for item in store.catalog.list_sessions() if item["id"] == session)
    assert row["status"] == "incomplete"
    store.close()


def test_start_session_catalog_failure_rolls_back_root_and_state(tmp_path: Path, monkeypatch):
    import pytest

    session = str(uuid.uuid4())
    store = RecordingStore(tmp_path, min_free_storage_gb=0)

    def fail_catalog(_manifest, _root):
        raise OSError("catalog unavailable")

    monkeypatch.setattr(store.catalog, "upsert_session", fail_catalog)
    with pytest.raises(OSError, match="catalog unavailable"):
        store.start_session(session, SessionMetadata("a", "b"))
    assert store._session_id is None
    assert store._root is None
    assert store._thread is None
    assert not list(tmp_path.glob(f"runtime_sessions/*/*/{session}"))
    assert not any(item["id"] == session for item in store.catalog.list_sessions())
    store.close()


def test_stop_session_timeout_raises_and_preserves_retry_state(tmp_path: Path):
    import pytest

    store = RecordingStore(tmp_path, min_free_storage_gb=0)
    store._session_id = "still-running"
    store._manifest = {"status": "open"}

    class _StuckWriter:
        @staticmethod
        def join(timeout: float | None = None) -> None:
            del timeout

        @staticmethod
        def is_alive() -> bool:
            return True

    thread = _StuckWriter()
    store._thread = thread  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="did not stop"):
        store.stop_session("shutdown")
    assert store._session_id == "still-running"
    assert store._thread is thread
    assert store._manifest["status"] == "incomplete"

    # Explicit fixture cleanup: the production object correctly retained the
    # session so a later call can retry once its writer has actually exited.
    store._thread = None
    store._session_id = None
    store.catalog.close()


def test_locked_dataset_blocks_annotation_and_trash(tmp_path: Path):
    service = DataManagerService(tmp_path)
    dataset = str(uuid.uuid4())
    root = tmp_path / "test_corpus" / dataset / "recordings" / str(uuid.uuid4())
    root.mkdir(parents=True)
    manifest = {
        "schema_version": "test_recording_v1",
        "dataset_id": dataset,
        "recording_id": root.name,
        "source_type": "dedicated",
        "quality_status": "passed",
        "split": "unset",
        "assets": [],
        "capture_session_id": "capture-a",
        "room_id": "room-a",
        "speaker_ids_anonymous": [],
        "duration_samples": 48000,
    }
    from data_management.manifests import write_manifest

    write_manifest(root / "recording_manifest.json", manifest)
    service.catalog.upsert_dataset(dataset, root.parents[1])
    service.catalog.upsert_recording(manifest, root)
    locked = service.assign_and_lock_dataset(dataset, "1.0.0")
    assert locked["locked"] and (root.parents[1] / "dataset_manifest.json").exists()
    annotation = Annotation(str(uuid.uuid4()), root.name, 0, 100, "voice_activity", "voice", None, 1, "a", "v0001")
    import pytest

    with pytest.raises(PermissionError):
        service.add_annotation(annotation)
    with pytest.raises(PermissionError):
        service.trash("recording", root.name)


def test_dataset_split_preview_is_non_mutating(tmp_path: Path):
    service = DataManagerService(tmp_path)
    dataset = str(uuid.uuid4())
    root = tmp_path / "test_corpus" / dataset / "recordings" / str(uuid.uuid4())
    root.mkdir(parents=True)
    manifest = {
        "schema_version": "test_recording_v1",
        "dataset_id": dataset,
        "recording_id": root.name,
        "source_type": "dedicated",
        "quality_status": "passed",
        "split": "unset",
        "assets": [],
        "capture_session_id": "capture-a",
        "room_id": "room-a",
        "speaker_ids_anonymous": [],
        "duration_samples": 48000,
    }
    from data_management.manifests import write_manifest

    write_manifest(root / "recording_manifest.json", manifest)
    service.catalog.upsert_dataset(dataset, root.parents[1])
    service.catalog.upsert_recording(manifest, root)
    preview = service.preview_dataset_split(dataset)
    assert preview["recording_count"] == 1 and preview["leakage_report"]["passed"]
    assert service.catalog.get_dataset(dataset)["locked"] == 0
