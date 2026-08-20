from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from common.config import load_config
from common.data_types import IngestedAudioBlock
from gui.dev_test_ui.meter import L1Meter
from gui.l1_spectrum_ui import CENTER_CHANNEL_INDEX, CHANNEL_NAMES, L1SpectrumAnalyzer
from gui.l1_spectrum_ui.host import L1SpectrumHost
from layer1_input.imcra import Layer1Imcra


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "config" / "config.yaml"


def _block(samples: np.ndarray, *, sequence: int = 0) -> IngestedAudioBlock:
    start = sequence * 960
    return IngestedAudioBlock(
        "l1-spectrum-test", 0, start, start + len(samples), 48_000,
        sequence, start / 48_000, samples,
    )


def test_channel_contract_matches_real_six_plus_center_mapping() -> None:
    assert CHANNEL_NAMES == ("MIC0", "MIC1", "MIC2", "MIC3", "MIC4", "MIC5", "Center", "Mix")
    assert CENTER_CHANNEL_INDEX == 6


def test_analyzer_uses_one_20ms_hop_and_finds_tone() -> None:
    time_axis = np.arange(960, dtype=np.float64) / 48_000
    samples = np.zeros((960, 8), np.float32)
    samples[:, 6] = 0.5 * np.sin(2.0 * np.pi * 1_000.0 * time_axis)
    block = _block(samples)
    frame = L1SpectrumAnalyzer().analyze(block, L1Meter().add(block))

    assert frame.channel_levels_dbfs.shape == (8, 427)
    assert frame.frequencies_hz[0] == 0.0
    assert 9_980.0 < frame.frequencies_hz[-1] <= 10_000.0
    peak = frame.frequencies_hz[int(np.argmax(frame.channel_levels_dbfs[6]))]
    assert peak == pytest.approx(1_000.0, abs=24.0)
    assert frame.noise_levels_dbfs is None


def test_analyzer_exposes_aligned_imcra_noise_for_seven_physical_mics() -> None:
    config = load_config(CONFIG)
    samples = np.random.default_rng(7).normal(0.0, 0.01, (960, 8)).astype(np.float32)
    raw = _block(samples)
    hop = Layer1Imcra.from_project(config).process(raw)[0]
    block = IngestedAudioBlock(
        raw.session_id, raw.stream_epoch, raw.start_sample, raw.end_sample,
        raw.sample_rate, raw.sequence_id, raw.timestamp, raw.samples, imcra_hop=hop,
    )
    frame = L1SpectrumAnalyzer().analyze(block, L1Meter().add(block))

    assert frame.noise_levels_dbfs is not None
    assert frame.noise_levels_dbfs.shape == (7, 342)
    assert 7_970.0 < frame.noise_frequencies_hz[-1] <= 8_000.0
    assert np.isfinite(frame.noise_levels_dbfs).all()


def test_l1_spectrum_host_has_no_downstream_runtime_dependencies() -> None:
    source = (ROOT / "gui/l1_spectrum_ui/host.py").read_text(encoding="utf-8")
    for forbidden in (
        "ApplicationRuntime", "WindowAssembler", "layer2_source_detection",
        "layer3_direction_signal", "layer4_voice_classifier", "RecordingStore",
    ):
        assert forbidden not in source


def test_light_control_uses_existing_official_cdc_commands() -> None:
    class FakeSerial:
        def __init__(self):
            self.packets = []

        def write(self, packet):
            self.packets.append(packet)
            return len(packet)

        def stop(self):
            pass

    serial = FakeSerial()
    host = L1SpectrumHost(load_config(CONFIG), serial_device=serial)
    states = []
    host.light_state_changed.connect(states.append)

    host._write_light(True)
    host._write_light(False)

    assert serial.packets == [b"E", b"e"]
    assert states == ["on", "off"]
    assert host.light_state == "off"


def test_window_defaults_center_and_snapshot_is_frozen() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QObject, Signal
    from PySide6.QtWidgets import QApplication

    from gui.l1_spectrum_ui.app import L1SpectrumWindow

    class FakeHost(QObject):
        frame_ready = Signal(object)
        state_changed = Signal(str)
        light_state_changed = Signal(str)
        error = Signal(str)

        def __init__(self):
            super().__init__()
            self.pre_denoise_enabled = False
            self.light_state = "unknown"

        def start(self):
            pass

        def stop(self, timeout=5.0):
            del timeout
            return True

        def set_pre_denoise_enabled(self, enabled):
            self.pre_denoise_enabled = bool(enabled)

        def set_light(self, enabled):
            self.light_state = "on" if enabled else "off"
            self.light_state_changed.emit(self.light_state)

    application = QApplication.instance() or QApplication([])
    window = L1SpectrumWindow(load_config(CONFIG), host=FakeHost(), auto_start=False)
    try:
        assert window.channel_group.checkedId() == CENTER_CHANNEL_INDEX
        assert sum(button.isChecked() for button in window.channel_buttons) == 1

        samples = np.zeros((960, 8), np.float32)
        samples[:, 6] = 0.25 * np.sin(2 * np.pi * 1_000 * np.arange(960) / 48_000)
        first_block = _block(samples)
        first = L1SpectrumAnalyzer().analyze(first_block, L1Meter().add(first_block))
        window._on_frame(first)
        window._capture_snapshot()
        frozen = window.snapshot_plot._levels.copy()

        samples[:, 6] = 0.25 * np.sin(2 * np.pi * 2_000 * np.arange(960) / 48_000)
        second_block = _block(samples, sequence=1)
        second = L1SpectrumAnalyzer().analyze(second_block, L1Meter().add(second_block))
        window._on_frame(second)
        application.processEvents()
        assert np.array_equal(window.snapshot_plot._levels, frozen)
        assert not np.array_equal(window.current_plot._levels, frozen)
    finally:
        window.close()
