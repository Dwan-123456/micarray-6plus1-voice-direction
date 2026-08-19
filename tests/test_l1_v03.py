from __future__ import annotations

import numpy as np
import pytest

from common.config import load_config
from common.data_types import IngestedAudioBlock
from ingest import IngestCoordinator
from layer1_input.calibration import ChannelCalibrator
from layer1_input.configuration import CalibrationConfig
from layer1_input.imcra import Layer1Imcra
from layer1_input.interface import DecodedAudio
from layer1_input.sources import map_logical_channels
from windowing import WindowAssembler


def _block(samples: np.ndarray, *, epoch: int, start: int, sequence: int) -> IngestedAudioBlock:
    return IngestedAudioBlock(
        "session", epoch, start, start + len(samples), 48_000, sequence,
        start / 48_000, samples,
    )


def test_native_to_logical_mapping_is_exact_and_hardware_mix_is_last():
    native = np.arange(16, dtype=np.float32).reshape(2, 8)
    logical = map_logical_channels(native, (0, 1, 2, 3, 4, 5, 7, 6))
    assert np.array_equal(logical, native[:, [0, 1, 2, 3, 4, 5, 7, 6]])
    assert np.array_equal(logical[:, :7], native[:, [0, 1, 2, 3, 4, 5, 7]])
    assert np.array_equal(logical[:, 7], native[:, 6])


def test_calibration_changes_only_the_seven_physical_microphones():
    logical = np.arange(32, dtype=np.float32).reshape(4, 8)
    frame = DecodedAudio(logical, 48_000, 0, 0.0)
    calibrator = ChannelCalibrator(CalibrationConfig((2.0,) * 7, (1,) * 7, (0,) * 7))
    output = calibrator.process(frame)
    assert output.samples.shape == (4, 8)
    assert np.array_equal(output.samples[:, :7], logical[:, :7] * 2.0)
    assert np.array_equal(output.samples[:, 7], logical[:, 7])


def test_imcra_emits_exact_20ms_hops_for_arbitrary_input_chunking():
    config = load_config("config/config.yaml")
    estimator = Layer1Imcra(config.layer1_imcra.model_copy(update={"warmup_seconds": 0.04}))
    rng = np.random.default_rng(7)
    samples = rng.normal(0.0, 0.01, (2_400, 8)).astype(np.float32)
    parts = (137, 1_100, 1_163)
    hops, start = [], 0
    for sequence, size in enumerate(parts):
        hops.extend(estimator.process(_block(samples[start : start + size], epoch=0, start=start, sequence=sequence)))
        start += size
    assert [(item.start_sample, item.end_sample) for item in hops] == [(0, 960), (960, 1_920)]
    assert hops[1].source_sequence_ids == (1, 2)
    ready = hops[1]
    assert ready.state == "ready" and ready.array_source_probability_20ms is not None
    assert ready.frequencies_hz.size == 342
    assert ready.frequencies_hz[0] == 0.0 and ready.frequencies_hz[-1] <= 8_000.0
    assert ready.noise_psd.shape == ready.smoothed_psd.shape == ready.minimum_psd.shape == ready.spp.shape == (7, 342)
    assert ready.conditional_smoothed_psd.shape == ready.conditional_minimum_psd.shape == (7, 342)
    assert ready.speech_absence_probability.shape == ready.posterior_snr.shape == ready.prior_snr.shape == (7, 342)
    assert ready.noise_features.shape == (7, 4)
    assert np.isfinite(ready.noise_features).all()
    assert np.all((ready.spp >= 0.0) & (ready.spp <= 1.0))
    assert np.all((ready.speech_absence_probability >= 0.0) & (ready.speech_absence_probability <= 1.0))
    with pytest.raises(ValueError):
        ready.noise_psd.setflags(write=True)


def test_cohen_2003_table_parameters_and_two_pass_state_are_exposed():
    config = load_config("config/config.yaml").layer1_imcra
    assert config.algorithm_version == "cohen_imcra_2003_l1_v2"
    assert (
        config.frequency_smoothing_half_width,
        config.spectrum_smoothing,
        config.minimum_history_subwindows,
        config.minimum_subwindow_frames,
        config.minimum_bias,
        config.gamma0,
        config.gamma1,
        config.zeta0,
        config.prior_snr_smoothing,
        config.noise_smoothing,
        config.bias_compensation,
    ) == (1, 0.9, 8, 15, 1.66, 4.6, 3.0, 1.67, 0.92, 0.85, 1.47)
    first = Layer1Imcra(config).process(
        _block(np.ones((960, 8), np.float32) * 0.01, epoch=0, start=0, sequence=0)
    )[0]
    assert np.array_equal(first.smoothed_psd, first.conditional_smoothed_psd)
    assert np.array_equal(first.minimum_psd, first.conditional_minimum_psd)
    assert np.all(first.speech_absence_probability == 1.0)
    assert np.all(first.spp == 0.0)


def test_cohen_posterior_spp_detects_a_strong_tonal_component_after_minimum_warmup():
    config = load_config("config/config.yaml").layer1_imcra
    estimator = Layer1Imcra(config)
    rng = np.random.default_rng(44)
    start = 0
    for sequence in range(120):
        noise = rng.normal(0.0, 0.01, (960, 8)).astype(np.float32)
        estimator.process(_block(noise, epoch=0, start=start, sequence=sequence))
        start += 960
    time_axis = np.arange(960, dtype=np.float64) / 48_000
    tone = (0.5 * np.sin(2.0 * np.pi * 1_000.0 * time_axis)).astype(np.float32)
    signal = rng.normal(0.0, 0.01, (960, 8)).astype(np.float32)
    signal[:, :7] += tone[:, None]
    hop = estimator.process(_block(signal, epoch=0, start=start, sequence=120))[0]
    tone_bin = int(np.argmin(np.abs(hop.frequencies_hz - 1_000.0)))
    assert np.all(hop.speech_absence_probability[:, tone_bin] == 0.0)
    assert np.all(hop.spp[:, tone_bin] > 0.99)


def test_imcra_resets_on_epoch_and_ignores_hardware_mix():
    config = load_config("config/config.yaml")
    short = config.layer1_imcra.model_copy(update={"warmup_seconds": 0.02})
    left, right = Layer1Imcra(short), Layer1Imcra(short)
    rng = np.random.default_rng(11)
    physical = rng.normal(0.0, 0.01, (960, 7)).astype(np.float32)
    a = np.column_stack((physical, np.zeros(960, np.float32)))
    b = np.column_stack((physical, np.full(960, 1_000.0, np.float32)))
    hop_a = left.process(_block(a, epoch=0, start=0, sequence=0))[0]
    hop_b = right.process(_block(b, epoch=0, start=0, sequence=0))[0]
    assert np.array_equal(hop_a.noise_psd, hop_b.noise_psd)
    assert np.array_equal(hop_a.spp, hop_b.spp)
    reset = left.process(_block(a, epoch=1, start=0, sequence=1))[0]
    assert reset.start_sample == 0
    assert np.all(reset.spp == 0.0)


def test_coordinator_imcra_and_window_share_one_sample_axis():
    config = load_config("config/config.yaml")
    coordinator = IngestCoordinator(session_id="session")
    estimator = Layer1Imcra(config.layer1_imcra.model_copy(update={"warmup_seconds": 0.02}))
    assembler = WindowAssembler()
    windows = []
    for sequence in range(16):
        values = np.zeros((960, 8), np.float32)
        frame = DecodedAudio(values, 48_000, sequence, sequence * 0.02)
        block = coordinator.ingest(frame)
        hops = estimator.process(block)
        windows.extend(assembler.add(block, hops))
    assert len(windows) == 1
    window = windows[0]
    assert window.samples.shape == (15_360, 8)
    assert len(window.imcra_hops) == 16
    assert [(hop.start_sample, hop.end_sample) for hop in window.imcra_hops[-2:]] == [
        (window.doa_start_sample, window.doa_start_sample + 960),
        (window.doa_start_sample + 960, window.doa_end_sample),
    ]
