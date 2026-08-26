from __future__ import annotations

from types import SimpleNamespace
import wave

import numpy as np

import gui.dev_test_ui.offline_l4_store as l4_store_module
from gui.dev_test_ui.offline_l4_store import OfflineLayer4UiStore
from gui.dev_test_ui.offline_l6_store import OfflineLayer6UiStore


def _hop_audio(hops: int, *, sign: float = 1.0) -> np.ndarray:
    """Small 20 ms-hop fixture; 4,097 hops are about 82 s, not 30 min."""

    waveform = np.empty(hops * 320, dtype=np.float32)
    amplitudes = sign * (0.05 + 0.8 * (np.arange(hops) % 31) / 30.0)
    waveform.reshape(hops, 320)[:] = amplitudes[:, None]
    return waveform


def _prefixes(total_hops: int, *, first: int = 257, step: int = 512) -> tuple[int, ...]:
    values = list(range(first, total_hops, step))
    if not values or values[-1] != total_hops:
        values.append(total_hops)
    return tuple(values)


def _observe_wav_updates(monkeypatch, store_type):
    writes: list[tuple[str, int]] = []
    appends: list[tuple[str, int]] = []
    original_write = store_type._write_waveform
    original_append = store_type._append_waveform

    def write(_cls, path, waveform):
        writes.append((path.name, len(waveform)))
        return original_write(path, waveform)

    def append(_cls, path, waveform):
        appends.append((path.name, len(waveform)))
        return original_append(path, waveform)

    monkeypatch.setattr(store_type, "_write_waveform", classmethod(write))
    monkeypatch.setattr(store_type, "_append_waveform", classmethod(append))
    return writes, appends


def _l4_processed(waveform: np.ndarray):
    source = SimpleNamespace(
        session_id="long-session",
        stream_epoch=0,
        track_id=1,
        theta_deg=35.0,
        start_sample=0,
        end_sample=len(waveform) * 3,
    )
    return SimpleNamespace(
        request_id="realtime-track-1",
        source=source,
        output_kind="merged",
        output_asset_id="long-session:track-1:merged",
        waveform_16k=waveform,
        metadata={
            "mos_score": 0.8,
            "realtime_provisional": True,
            "stable_branch_id": 0,
        },
    )


def _l5_result(processed, probabilities, decisions):
    return SimpleNamespace(
        source=processed.source,
        output_asset_id=processed.output_asset_id,
        l5_probabilities_20ms=probabilities,
        l5_is_voice_20ms=decisions,
        l5_model_id="bounded-l5",
        metadata={"l5_threshold": 0.7},
    )


def test_l4_l5_cumulative_snapshots_append_only_and_keep_ui_state_bounded(monkeypatch):
    total_hops = OfflineLayer4UiStore._MAX_WAVEFORM_BINS + 1
    audio = _hop_audio(total_hops)
    probabilities = tuple(float(index % 101) / 100.0 for index in range(total_hops))
    decisions = tuple(value >= 0.7 for value in probabilities)
    prefixes = _prefixes(total_hops)
    writes, appends = _observe_wav_updates(monkeypatch, OfflineLayer4UiStore)
    created_annotations = []
    annotation_type = l4_store_module.TrackVoiceAnnotation

    def tracked_annotation(*args, **kwargs):
        value = annotation_type(*args, **kwargs)
        created_annotations.append(value)
        return value

    monkeypatch.setattr(l4_store_module, "TrackVoiceAnnotation", tracked_annotation)
    store = OfflineLayer4UiStore()
    try:
        stored_identity = None
        previous_hops = 0
        for hops in prefixes:
            processed = _l4_processed(audio[:hops * 320])
            store.set_processed((processed,))
            store.apply_l5((
                _l5_result(
                    processed,
                    probabilities[:hops],
                    decisions[:hops],
                ),
            ))
            current_identity = id(store._tracks[1])
            stored_identity = current_identity if stored_identity is None else stored_identity
            assert current_identity == stored_identity
            assert store._tracks[1].l5_applied_hops == hops
            previous_hops = hops

        assert previous_hops == total_hops
        assert writes == [("l4_track_000001_merged.wav", prefixes[0] * 320)]
        assert [samples for _path, samples in appends] == [
            (right - left) * 320
            for left, right in zip(prefixes, prefixes[1:])
        ]
        assert len(created_annotations) == total_hops

        snapshot = store.snapshots()[0]
        assert len(snapshot.waveform_envelope) <= OfflineLayer4UiStore._MAX_WAVEFORM_BINS
        assert len(snapshot.voice_annotations_20ms) == len(snapshot.waveform_envelope)
        assert all(value is not None for value in snapshot.voice_annotations_20ms)
        assert store._tracks[1].envelope_hops == total_hops
        assert store._tracks[1].envelope_bin_hops == 2
        path = store.audio_path(1)
        assert path is not None
        with wave.open(str(path), "rb") as reader:
            assert reader.getnframes() == total_hops * 320
    finally:
        store.close()


def _l6_output(speaker_id: int, waveform: np.ndarray):
    return SimpleNamespace(
        speaker_id=speaker_id,
        label=f"Speaker {chr(64 + speaker_id)}",
        waveform_16k=waveform,
        source_track_ids=(speaker_id,),
        fragment_ids=(f"fragment-{speaker_id}",),
        mean_quality=0.8,
    )


def _l6_result(outputs, *, changed=(), append_only=()):
    return SimpleNamespace(
        session_id="long-session",
        outputs=tuple(outputs),
        metadata={
            "incremental_changed_speaker_ids": tuple(changed),
            "incremental_append_only_speaker_ids": tuple(append_only),
        },
    )


def test_l6_incremental_contract_appends_rewrites_and_removes_speakers(monkeypatch):
    writes, appends = _observe_wav_updates(monkeypatch, OfflineLayer6UiStore)
    speaker_1 = _hop_audio(4)
    speaker_2 = _hop_audio(2, sign=-1.0)
    store = OfflineLayer6UiStore()
    try:
        store.set_result(_l6_result((
            _l6_output(1, speaker_1[:2 * 320]),
            _l6_output(2, speaker_2),
        )))
        speaker_2_path = store.audio_path(2)
        assert speaker_2_path is not None and speaker_2_path.is_file()

        store.set_result(_l6_result(
            (_l6_output(1, speaker_1), _l6_output(2, speaker_2)),
            append_only=(1,),
        ))
        assert appends == [("l6_id_1.wav", 2 * 320)]
        assert writes == [("l6_id_1.wav", 2 * 320), ("l6_id_2.wav", 2 * 320)]

        replacement = np.full(4 * 320, -0.75, dtype=np.float32)
        store.set_result(_l6_result(
            (_l6_output(1, replacement), _l6_output(2, speaker_2)),
            changed=(1,),
        ))
        assert writes[-1] == ("l6_id_1.wav", 4 * 320)
        assert len(appends) == 1
        with wave.open(str(store.audio_path(1)), "rb") as reader:
            first_sample = np.frombuffer(reader.readframes(1), dtype="<i2")[0]
        assert first_sample == round(-0.75 * 32768)

        store.set_result(_l6_result((_l6_output(1, replacement),)))
        assert store.audio_path(2) is None
        assert not speaker_2_path.exists()
        assert tuple(item.track_id for item in store.snapshots()) == (1,)
    finally:
        store.close()


def test_l6_long_append_only_envelope_never_exceeds_4096(monkeypatch):
    total_hops = OfflineLayer6UiStore._MAX_WAVEFORM_BINS + 1
    audio = _hop_audio(total_hops)
    prefixes = _prefixes(total_hops)
    writes, appends = _observe_wav_updates(monkeypatch, OfflineLayer6UiStore)
    store = OfflineLayer6UiStore()
    try:
        for index, hops in enumerate(prefixes):
            store.set_result(_l6_result(
                (_l6_output(1, audio[:hops * 320]),),
                append_only=() if index == 0 else (1,),
            ))

        assert writes == [("l6_id_1.wav", prefixes[0] * 320)]
        assert [samples for _path, samples in appends] == [
            (right - left) * 320
            for left, right in zip(prefixes, prefixes[1:])
        ]
        snapshot = store.snapshots()[0]
        assert len(snapshot.waveform_envelope) <= 4_096
        assert len(snapshot.waveform_envelope) <= OfflineLayer6UiStore._MAX_WAVEFORM_BINS
        assert store._stored[1].envelope_hops == total_hops
        assert store._stored[1].envelope_bin_hops == 2
        path = store.audio_path(1)
        assert path is not None
        with wave.open(str(path), "rb") as reader:
            assert reader.getnframes() == total_hops * 320
    finally:
        store.close()
