from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from layer1_input.calibration import ChannelCalibrator
from layer1_input.configuration import CalibrationConfig
from layer1_input.pipeline import InputPipeline
from layer1_input.recording_replay import RecordingReplaySource


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def recording(tmp_path: Path, *, blocks: int = 3) -> Path:
    root = tmp_path / "recording"
    root.mkdir()
    audio = root / "native_8ch.wav"
    values = np.arange(blocks * 960 * 8, dtype=np.int16).reshape(-1, 8)
    with wave.open(str(audio), "wb") as output:
        output.setnchannels(8)
        output.setsampwidth(2)
        output.setframerate(48_000)
        output.writeframes(values.astype("<i2").tobytes())
    hotmaps = root / "hotmaps.jsonl"
    hotmaps.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": "recorded_hotmap_v1",
                    "sequence_id": index,
                    "timestamp": index * 0.02,
                    "received_at": None,
                    "playback_sample": index * 960,
                    "matrix": np.full((16, 16), index, dtype=np.uint8).tolist(),
                }
            )
            + "\n"
            for index in range(blocks)
        ),
        encoding="utf-8",
    )
    manifest = root / "recording_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "display_name": "我命名的测试音频",
                "assets": [
                    {"kind": "native_8ch", "path": audio.name, "sha256": _sha256(audio)},
                    {"kind": "cdc_hotmaps", "path": hotmaps.name, "sha256": _sha256(hotmaps)},
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def source(path: Path, *, autoplay: bool = False) -> RecordingReplaySource:
    return RecordingReplaySource(
        path,
        logical_channel_map=(0, 1, 2, 3, 4, 5, 7, 6),
        autoplay=autoplay,
    )


def test_complete_recording_replays_audio_without_loading_hotmap(tmp_path: Path):
    replay = source(recording(tmp_path))
    pipeline = InputPipeline(
        replay,
        ChannelCalibrator(CalibrationConfig((1.0,) * 7, (1,) * 7, (0,) * 7)),
    )
    pipeline.start()
    assert replay.display_name == "我命名的测试音频"
    assert replay.status().state == "paused"
    replay.resume()
    first = pipeline.read(timeout=0.1)
    assert first is not None
    assert first.native_samples is not None and first.native_samples.shape == (960, 8)
    assert first.hotmap is None

    replay.pause()
    before = replay.status().current_sample
    assert pipeline.read(timeout=0.01) is None
    assert replay.status().current_sample == before
    replay.resume()
    second = pipeline.read(timeout=0.1)
    assert second is not None and second.sequence_id == 1
    assert second.hotmap is None
    pipeline.stop()


def test_replay_rewinds_all_inputs_and_waits_at_eof(tmp_path: Path):
    replay = source(recording(tmp_path, blocks=2), autoplay=True)
    replay.start()
    first = replay.read(timeout=0.1)
    second = replay.read(timeout=0.1)
    assert first is not None and second is not None
    assert replay.read(timeout=0.1) is None
    assert replay.status().state == "ended"
    assert replay.exhausted is False

    replay.replay()
    restarted = replay.read(timeout=0.1)
    assert restarted is not None
    assert restarted.sequence_id == 0 and restarted.timestamp == 0.0
    assert replay.status().generation == 1
    replay.stop()


def test_replay_does_not_open_or_validate_recorded_hotmap_asset(tmp_path: Path):
    manifest = recording(tmp_path)
    (manifest.parent / "hotmaps.jsonl").write_text("tampered\n", encoding="utf-8")
    replay = source(manifest, autoplay=True)
    replay.start()
    frame = replay.read(timeout=0.1)
    assert frame is not None
    replay.stop()


def test_replay_after_runtime_stop_reopens_from_sample_zero(tmp_path: Path):
    replay = source(recording(tmp_path, blocks=2), autoplay=True)
    replay.start()
    assert replay.read(timeout=0.1) is not None
    generation = replay.status().generation
    replay.stop()

    replay.replay()
    assert replay.status().state == "ready"
    assert replay.status().current_sample == 0
    assert replay.status().generation == generation + 1
    replay.start()
    restarted = replay.read(timeout=0.1)
    assert restarted is not None
    assert restarted.sequence_id == 0 and restarted.timestamp == 0.0
    replay.stop()


def test_replay_rejects_modified_audio_asset(tmp_path: Path):
    manifest = recording(tmp_path)
    (manifest.parent / "native_8ch.wav").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="校验失败"):
        source(manifest)


def test_replay_accepts_manifest_without_hotmap_asset(tmp_path: Path):
    manifest = recording(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["assets"] = [item for item in payload["assets"] if item["kind"] == "native_8ch"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    replay = source(manifest, autoplay=True)
    replay.start()
    assert replay.read(timeout=0.1) is not None
    replay.stop()
