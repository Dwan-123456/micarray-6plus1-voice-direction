from __future__ import annotations

import os
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from data_management.catalog import Catalog
from data_management.dedicated_recording import DedicatedRecordingController, WizardPhase
from data_management.wizard import WizardInput
from gui.production_ui.capture_host import CaptureHost
from layer1_input.interface import DecodedAudio
from PySide6.QtWidgets import QApplication


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QT_APP = QApplication.instance() or QApplication([])


def process_signals() -> None:
    QT_APP.processEvents()


def wizard_input() -> WizardInput:
    return WizardInput(
        "dataset", "room", "quiet", "pose", 1, "granted", ("research",),
        recording_name="阵列原始输入",
    )


def test_wizard_abort_discards_take_and_allows_new_session(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite")
    controller = DedicatedRecordingController(tmp_path, catalog)
    controller.begin(wizard_input())
    samples = np.random.default_rng(7).normal(0, 0.02, (960, 8)).astype(np.float32)
    from common.data_types import IngestedAudioBlock

    controller.append(IngestedAudioBlock(
        "old", 0, 0, 960, 48_000, 0, 0.0, samples, native_samples=samples,
    ))
    status = controller.abort("麦克风断开，本次录制已中止")
    assert status.phase == WizardPhase.ERROR
    assert "麦克风断开" in status.message
    assert status.sample_count == 0

    controller.begin(wizard_input())
    status = controller.append(IngestedAudioBlock(
        "new", 0, 0, 960, 48_000, 0, 0.0, samples, native_samples=samples,
    ))
    assert status.phase == WizardPhase.RECORDING
    controller.reset()
    assert controller.status().phase == WizardPhase.IDLE
    catalog.close()


class _FakePipeline:
    def __init__(self, frames: list[DecodedAudio], *, keep_alive: bool = False):
        self.frames = list(frames)
        self.keep_alive = keep_alive
        self.stopped = threading.Event()

    def start(self) -> None:
        return None

    def read(self, timeout: float | None = None):
        del timeout
        if self.frames:
            return self.frames.pop(0)
        if self.keep_alive and not self.stopped.wait(0.002):
            return None
        return None

    def take_health_events(self):
        return ()

    def stop(self) -> None:
        self.stopped.set()


def _service(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite")
    return SimpleNamespace(
        data_root=tmp_path,
        catalog=catalog,
        wizard=DedicatedRecordingController(tmp_path, catalog),
    )


def _audio(sequence: int) -> DecodedAudio:
    rng = np.random.default_rng(sequence)
    physical = rng.normal(0, 0.02, (960, 7)).astype(np.float32)
    mix = rng.normal(0, 0.02, 960).astype(np.float32)
    native = np.column_stack((physical[:, :6], mix, physical[:, 6]))
    logical = np.column_stack((physical, mix))
    from layer1_input.interface import CdcHotmapFrame

    return DecodedAudio(
        logical,
        48_000,
        sequence,
        sequence * 0.02,
        native_samples=native,
        hotmap=CdcHotmapFrame(np.full((16, 16), sequence, dtype=np.uint8), sequence, sequence * 0.02),
    )


def test_wizard_append_error_does_not_disconnect_runtime_capture(tmp_path):
    service = _service(tmp_path)
    service.wizard.begin(wizard_input())
    calls = 0

    def broken_append(_block, _hotmaps=()):
        nonlocal calls
        calls += 1
        raise ValueError("模拟向导错误")

    service.wizard.append = broken_append
    pipeline = _FakePipeline([_audio(0), _audio(1)], keep_alive=True)
    host = CaptureHost(PROJECT_ROOT, service)
    host._make_pipeline = lambda: pipeline
    errors: list[str] = []
    host.error.connect(errors.append)
    host.start()

    deadline = time.monotonic() + 2
    while not errors and time.monotonic() < deadline:
        process_signals()
        time.sleep(0.005)
    process_signals()
    assert calls == 1
    assert host.connected
    assert service.wizard.phase == WizardPhase.ERROR
    assert any("运行录音和麦克风采集仍在继续" in message for message in errors)
    host.close()
    assert host._thread is None
    service.catalog.close()


def test_runtime_status_is_throttled_and_close_waits_for_worker(tmp_path):
    service = _service(tmp_path)
    pipeline = _FakePipeline([_audio(index) for index in range(20)], keep_alive=True)
    host = CaptureHost(PROJECT_ROOT, service)
    host._make_pipeline = lambda: pipeline
    statuses: list[dict] = []
    host.runtime_status.connect(statuses.append)
    host.start()

    deadline = time.monotonic() + 2
    while pipeline.frames and time.monotonic() < deadline:
        process_signals()
        time.sleep(0.005)
    process_signals()
    assert not pipeline.frames
    # One forced connection update plus, at most, one periodic update while
    # twenty blocks are consumed much faster than the 250 ms UI interval.
    assert 1 <= len(statuses) <= 2
    host.close()
    assert pipeline.stopped.is_set()
    assert host._thread is None
    assert host._pipeline is None
    assert host._session_id is None
    assert (tmp_path / "logs" / "audio_data_manager_capture.log").exists()
    service.catalog.close()


def test_standalone_capture_records_formal_l1_imcra_sidecar(tmp_path):
    service = _service(tmp_path)
    pipeline = _FakePipeline([_audio(index) for index in range(3)], keep_alive=True)
    host = CaptureHost(PROJECT_ROOT, service)
    host._make_pipeline = lambda: pipeline
    host.handle_command("mode:continuous")
    host.start()
    deadline = time.monotonic() + 2
    while pipeline.frames and time.monotonic() < deadline:
        process_signals()
        time.sleep(0.005)
    assert not pipeline.frames
    host.close()
    manifest_path = next(tmp_path.glob("runtime_sessions/*/*/*/session_manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    imcra = next(item for item in manifest["chunks"][0]["assets"] if item["kind"] == "imcra")
    with np.load(manifest_path.parent / imcra["path"]) as values:
        assert values["start_samples"].tolist() == [0, 960, 1920]
        assert values["noise_psd"].shape == (3, 7, 338)
        assert values["spp"].shape == (3, 7, 338)
    service.catalog.close()


def test_capture_stop_uses_mode_specific_control_without_continuous_pause_bug(tmp_path):
    service = _service(tmp_path)
    host = CaptureHost(PROJECT_ROOT, service)
    calls: list[tuple[str, str | None]] = []
    host.recording_store = SimpleNamespace(
        set_recording_mode=lambda mode: calls.append(("mode", mode)),
        pause_recording=lambda: calls.append(("pause", None)),
        start_recording=lambda: calls.append(("record", None)),
    )
    host._thread = threading.current_thread()
    host._pipeline = SimpleNamespace()

    host.handle_command("mode:continuous")
    host.handle_command("stop")
    assert calls == [("mode", "continuous"), ("mode", "off")]

    calls.clear()
    host.handle_command("mode:manual")
    host.handle_command("stop")
    assert calls == [("mode", "manual"), ("pause", None)]
    host._thread = None
    host._pipeline = None
    service.catalog.close()
