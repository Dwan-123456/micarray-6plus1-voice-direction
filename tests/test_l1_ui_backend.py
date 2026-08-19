import numpy as np

from common.data_types import IngestedAudioBlock
from gui.dev_test_ui import L1Meter
from ingest import BlockFanout
from gui.dev_test_ui.scratch_recorder import ScratchRecorder
import json
import threading
import wave
import time
import pytest
from gui.dev_test_ui.preview_player import PreviewPlayer


def block(samples, *, sequence=0):
    return IngestedAudioBlock("session", 0, 0, len(samples), 48_000, sequence, 0.0, samples)


def test_fanout_delivers_same_object_to_window_and_recording_consumers():
    fanout = BlockFanout()
    windows, recording = fanout.subscribe(1), fanout.subscribe(1)
    value = block(np.zeros((10, 8), np.float32))
    fanout.publish(value)
    assert windows.get_nowait() is value
    assert recording.get_nowait() is value


def test_fanout_is_bounded_and_latest_value_wins():
    fanout, receiver = BlockFanout(), None
    receiver = fanout.subscribe(1)
    first = block(np.zeros((1, 8), np.float32), sequence=0)
    second = block(np.ones((1, 8), np.float32), sequence=1)
    fanout.publish(first)
    fanout.publish(second)
    assert receiver.get_nowait() is second
    assert fanout.dropped_by_subscriber == 1


def test_meter_uses_eight_independent_latest_960_sample_channels():
    samples = np.zeros((960, 8), np.float32)
    samples[:, 0] = 1.0
    samples[:, 1] = 0.5
    snapshot = L1Meter().add(block(samples))
    assert snapshot.rms_dbfs[0] == 0
    assert np.isclose(snapshot.rms_dbfs[1], -6.0206, atol=0.001)
    assert snapshot.rms_dbfs[2] == -120
    assert snapshot.clipped.tolist() == [True, False, False, False, False, False, False, False]


def test_scratch_record_pause_resume_finish_preserves_real_ranges(tmp_path):
    recorder = ScratchRecorder("data/dev_test_ui/scratch/current", project_root=tmp_path)
    recorder.record()
    first_samples = np.zeros((10, 8), np.float32)
    first = IngestedAudioBlock("session", 0, 0, 10, 48_000, 0, 0.0, first_samples, np.zeros((10, 8), np.float32))
    recorder.append(first)
    recorder.pause()
    recorder.resume()
    second = IngestedAudioBlock(
        "session", 0, 20, 25, 48_000, 2, 20 / 48_000, np.ones((5, 8), np.float32), np.ones((5, 8), np.float32)
    )
    recorder.append(second)
    manifest_path = recorder.finish()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [(item["start_sample"], item["end_sample"]) for item in manifest["segments"]] == [(0, 10), (20, 25)]
    assert recorder.state == "complete"
    for item in manifest["segments"]:
        physical = manifest_path.parent / next(
            file["path"] for file in item["files"] if "physical_7ch.wav" in file["path"]
        )
        with wave.open(str(physical), "rb") as source:
            assert source.getnframes() == item["frame_count"]
        float_path = manifest_path.parent / next(
            file["path"] for file in item["files"] if "physical_7ch_float.npy" in file["path"]
        )
        values = np.load(float_path, allow_pickle=False)
        assert values.shape == (item["frame_count"], 7) and values.dtype == np.float32


def test_new_recording_removes_previous_current(tmp_path):
    recorder = ScratchRecorder("data/dev_test_ui/scratch/current", project_root=tmp_path)
    recorder.record()
    recorder.append(block(np.zeros((2, 8), np.float32)))
    recorder.finish()
    old = recorder.current_root / "old-marker"
    old.write_text("old")
    recorder.record()
    assert not old.exists()
    assert recorder.state == "recording"


def test_scratch_streaming_memory_and_queue_are_bounded(tmp_path):
    recorder = ScratchRecorder("data/dev_test_ui/scratch/current", project_root=tmp_path, queue_blocks=8)
    recorder.record()
    samples = np.zeros((960, 8), np.float32)
    for index in range(250):
        recorder.append(
            IngestedAudioBlock(
                "session", 0, index * 960, (index + 1) * 960, 48_000, index, index * .02, samples
            )
        )
        while recorder.queued_blocks >= 8:
            time.sleep(.001)
    manifest = recorder.finish()
    values = np.load(manifest.parent / "segments/segment_000_physical_7ch_float.npy", mmap_mode="r")
    assert values.shape == (250 * 960, 7)
    assert recorder.queued_blocks == 0
    assert not hasattr(recorder._open, "float_chunks")
    recorder.shutdown()


def test_scratch_shutdown_can_delete_all_test_ui_recordings(tmp_path):
    recorder = ScratchRecorder("data/dev_test_ui/scratch/current", project_root=tmp_path)
    recorder.record()
    recorder.append(block(np.zeros((2, 8), np.float32)))
    recorder.shutdown(delete_files=True)

    assert not (tmp_path / "data/dev_test_ui/scratch").exists()


def test_scratch_cleanup_does_not_delete_formal_recordings_or_ui_settings(tmp_path):
    formal = tmp_path / "data/runtime_sessions/keep.wav"
    settings = tmp_path / "data/dev_test_ui/settings.json"
    formal.parent.mkdir(parents=True)
    settings.parent.mkdir(parents=True)
    formal.write_bytes(b"formal")
    settings.write_text("{}", encoding="utf-8")
    recorder = ScratchRecorder("data/dev_test_ui/scratch/current", project_root=tmp_path)
    recorder.record()
    recorder.append(block(np.zeros((2, 8), np.float32)))
    recorder.shutdown(delete_files=True)

    assert formal.read_bytes() == b"formal"
    assert settings.read_text(encoding="utf-8") == "{}"


def test_l3_preview_player_releases_synthesized_audio_on_close(tmp_path):
    player = PreviewPlayer(sample_rate=48_000, volume=0.25, loop_gap_ms=80, autoplay=False)
    cache = tmp_path / "preview.f32"
    np.ones(15_360, dtype=np.float32).tofile(cache)
    player.load_file(cache)
    assert player._audio.size == 15_360
    assert isinstance(player._audio, np.memmap)

    player.close()

    assert player._audio.size == 0
    assert player._playing is False


def test_preview_player_reports_sample_accurate_progress_and_resets_on_stop():
    player = PreviewPlayer(sample_rate=48_000, volume=1.0, loop_gap_ms=0, autoplay=False)
    player.load(np.arange(100, dtype=np.float32))
    player._playing = True
    output = np.zeros((25, 1), dtype=np.float32)

    player._callback(output, 25, None, None)
    assert player.playback_progress == pytest.approx(0.25)

    player.pause()
    assert player.playback_progress == pytest.approx(0.25)
    player.stop()
    assert player.playback_progress == 0.0
    player.close()


def test_track_preview_uses_one_bounded_gain_for_the_complete_file(tmp_path):
    player = PreviewPlayer(
        sample_rate=48_000, volume=1.0, loop_gap_ms=80, autoplay=False,
        peak_dbfs=-6.0,
    )
    time_axis = np.arange(48_000, dtype=np.float64) / 48_000
    audio = np.ascontiguousarray(0.005 * np.sin(2 * np.pi * 1_000 * time_axis), dtype=np.float32)
    cache = tmp_path / "quiet-track.f32"
    audio.tofile(cache)

    player.load_file(cache, target_rms_dbfs=-28.0, max_gain_db=18.0)

    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(audio)))
    expected = min(
        10 ** ((-28.0 - 20 * np.log10(rms)) / 20),
        10 ** (18.0 / 20),
        10 ** (-6.0 / 20) / peak,
    )
    assert player._gain == pytest.approx(expected)
    player.close()


def test_preview_player_rebuilds_output_when_default_device_changes(monkeypatch):
    class FakeStream:
        def __init__(self, *, device, **_kwargs):
            self.device = device
            self.active = False
            self.closed = False

        def start(self):
            self.active = True

        def stop(self):
            self.active = False

        def close(self):
            self.closed = True

    devices = [7, 8]
    created = []
    monkeypatch.setattr(
        PreviewPlayer, "_default_output_device", staticmethod(lambda: devices[0]),
    )
    monkeypatch.setattr(
        "gui.dev_test_ui.preview_player.sd.OutputStream",
        lambda **kwargs: created.append(FakeStream(**kwargs)) or created[-1],
    )
    player = PreviewPlayer(sample_rate=48_000, volume=1.0, loop_gap_ms=0, autoplay=False)
    player.load(np.arange(960, dtype=np.float32))
    assert player.play() is True
    assert created[-1].device == 7

    player.pause()
    devices[0] = 8
    assert player.play() is True
    assert len(created) == 2
    assert created[0].closed is True
    assert created[1].device == 8
    player.close()


def test_preview_player_reports_output_open_failure(monkeypatch):
    monkeypatch.setattr(
        PreviewPlayer, "_default_output_device", staticmethod(lambda: 7),
    )

    def fail(**_kwargs):
        raise RuntimeError("device disconnected")

    monkeypatch.setattr("gui.dev_test_ui.preview_player.sd.OutputStream", fail)
    player = PreviewPlayer(sample_rate=48_000, volume=1.0, loop_gap_ms=0, autoplay=False)
    player.load(np.arange(960, dtype=np.float32))
    assert player.play() is False
    assert "device disconnected" in (player.take_error() or "")
    assert player.playing is False
    player.close()


def test_scratch_full_command_queue_enters_error_and_shutdown_still_purges(tmp_path):
    recorder = ScratchRecorder(
        "data/dev_test_ui/scratch/current", project_root=tmp_path, queue_blocks=1,
    )
    recorder.record()
    entered, release = threading.Event(), threading.Event()
    original_append = recorder._append_on_worker

    def slow_append(value):
        entered.set()
        if not release.wait(5.0):
            raise TimeoutError("test did not release scratch writer")
        original_append(value)

    recorder._append_on_worker = slow_append
    samples = np.zeros((2, 8), np.float32)
    recorder.append(block(samples, sequence=0))
    assert entered.wait(1.0)
    recorder.append(block(samples, sequence=1))
    assert recorder.queued_blocks == 1

    try:
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="scratch writer命令队列已满"):
            recorder.pause()
        assert time.monotonic() - started < 3.0
        assert recorder.state == "error"
        assert recorder.last_error == "scratch writer命令队列已满"
    finally:
        release.set()
        recorder.shutdown(delete_files=True)

    assert not (tmp_path / "data/dev_test_ui/scratch").exists()
