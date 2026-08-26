from __future__ import annotations

import hashlib
import json
from pathlib import Path
import wave

import numpy as np

import layer4_speech_separation.offline as offline_module
from layer4_speech_separation.contracts import (
    Layer4LongAudioInput,
    Layer4OfflineResult,
    SpeakerCountDecision,
)
from layer4_speech_separation.offline import persist_offline_results


def _result(frame_count: int) -> Layer4OfflineResult:
    source_waveform = np.ascontiguousarray(
        np.linspace(-0.25, 0.25, frame_count * 960, dtype=np.float32),
    )
    source = Layer4LongAudioInput(
        asset_id="asset",
        sha256=hashlib.sha256(source_waveform.tobytes()).hexdigest(),
        session_id="session",
        stream_epoch=0,
        track_id=7,
        theta_deg=120.0,
        start_sample=0,
        sample_rate=48_000,
        waveform=source_waveform,
        l2_direction_counts=((960, 1),),
    )
    decision = SpeakerCountDecision("asset", 1, 1.0, "test-count", {})
    output = np.ascontiguousarray(
        np.linspace(-1.25, 1.25, frame_count * 320, dtype=np.float32),
    )
    probabilities = np.linspace(0.1, 0.9, frame_count, dtype=np.float32)
    decisions = tuple(bool(value >= 0.7) for value in probabilities)
    return Layer4OfflineResult(
        request_id="request",
        source=source,
        speaker_count=decision,
        path="single_speaker_bypass",
        selected=None,
        l5_probability=0.9,
        l5_is_voice=True,
        l5_model_id="l5-test",
        l5_probabilities_20ms=probabilities,
        l5_is_voice_20ms=decisions,
        output_asset_id="output",
        output_sha256=hashlib.sha256(output.tobytes()).hexdigest(),
        metadata={"output_waveform_16k": output, "kept": "metadata"},
    )


def _session_root(tmp_path: Path) -> Path:
    (tmp_path / "session_manifest.json").write_text(
        json.dumps({"status": "complete", "session_id": "session"}),
        encoding="utf-8",
    )
    return tmp_path


def _expected_pcm(result: Layer4OfflineResult) -> np.ndarray:
    waveform = np.asarray(result.metadata["output_waveform_16k"], dtype=np.float32)
    return np.clip(
        np.rint(waveform * 32768.0), -32768, 32767,
    ).astype("<i2")


def _read_pcm(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as reader:
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        assert reader.getframerate() == 16_000
        return np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")


def test_short_offline_result_keeps_inline_frames_and_exact_pcm(tmp_path: Path) -> None:
    result = _result(4)

    manifest_path = persist_offline_results(_session_root(tmp_path), (result,))

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = payload["results"][0]
    assert isinstance(row["l5_probabilities_20ms"], list)
    assert isinstance(row["l5_is_voice_20ms"], list)
    np.testing.assert_array_equal(
        np.asarray(row["l5_probabilities_20ms"], dtype=np.float32),
        result.l5_probabilities_20ms,
    )
    assert tuple(row["l5_is_voice_20ms"]) == result.l5_is_voice_20ms
    assert row["metadata"] == {"kept": "metadata"}
    assert not tuple(manifest_path.parent.glob("*.bin"))

    wav_path = manifest_path.parent / row["output_path"]
    np.testing.assert_array_equal(_read_pcm(wav_path), _expected_pcm(result))
    assert row["output_sha256"] == hashlib.sha256(wav_path.read_bytes()).hexdigest()


def test_long_offline_result_uses_bounded_binary_frame_sidecars(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result = _result(7)
    rint_lengths: list[int] = []
    original_rint = np.rint

    def bounded_rint(values, *args, **kwargs):
        rint_lengths.append(len(values))
        return original_rint(values, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(offline_module, "_PERSIST_INLINE_FRAME_LIMIT", 2)
        scoped.setattr(offline_module, "_PERSIST_FRAME_CHUNK_ITEMS", 2)
        scoped.setattr(offline_module, "_PERSIST_WAV_CHUNK_SAMPLES", 320)
        scoped.setattr(offline_module.np, "rint", bounded_rint)

        def forbidden_read_bytes(_path):
            raise AssertionError("offline persistence must hash files as streams")

        scoped.setattr(Path, "read_bytes", forbidden_read_bytes)
        manifest_path = persist_offline_results(_session_root(tmp_path), (result,))

    assert rint_lengths == [320] * 7
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = payload["results"][0]
    descriptors = (
        (
            row["l5_probabilities_20ms"],
            np.dtype("<f4"),
            np.asarray(result.l5_probabilities_20ms, dtype=np.float32),
        ),
        (
            row["l5_is_voice_20ms"],
            np.dtype(np.bool_),
            np.asarray(result.l5_is_voice_20ms, dtype=np.bool_),
        ),
    )
    for descriptor, dtype, expected in descriptors:
        assert descriptor["storage"] == "binary_sidecar_v1"
        assert descriptor["dtype"] == dtype.str
        assert descriptor["frame_count"] == 7
        relative = Path(descriptor["path"])
        assert not relative.is_absolute()
        sidecar = manifest_path.parent / relative
        assert descriptor["sha256"] == hashlib.sha256(sidecar.read_bytes()).hexdigest()
        np.testing.assert_array_equal(np.fromfile(sidecar, dtype=dtype), expected)

    wav_path = manifest_path.parent / row["output_path"]
    np.testing.assert_array_equal(_read_pcm(wav_path), _expected_pcm(result))
    assert row["output_sha256"] == hashlib.sha256(wav_path.read_bytes()).hexdigest()
    assert not tuple(manifest_path.parent.glob("*.partial"))
