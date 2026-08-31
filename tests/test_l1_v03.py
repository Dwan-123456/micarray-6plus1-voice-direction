from __future__ import annotations

import numpy as np
import pytest

from common.config import load_config
from common.data_types import CalibrationMetadata, IngestedAudioBlock
from ingest import IngestCoordinator
from layer1_input.calibration import ChannelCalibrator
from layer1_input.configuration import CalibrationConfig
from layer1_input.imcra import Layer1Imcra
from layer1_input.interface import DecodedAudio
from layer1_input.serial_device import SerialDevice
from layer1_input.sources import map_logical_channels
from layer1_input.speech_spectrum import speech_gate_band_weights
from windowing import WindowAssembler


def _block(
    samples: np.ndarray,
    *,
    epoch: int,
    start: int,
    sequence: int,
    session: str = "session",
    calibration: CalibrationMetadata | None = None,
) -> IngestedAudioBlock:
    return IngestedAudioBlock(
        session, epoch, start, start + len(samples), 48_000, sequence,
        start / 48_000, samples,
        calibration=calibration or CalibrationMetadata.unverified_identity(),
    )


def test_native_to_logical_mapping_is_exact_and_hardware_mix_is_last():
    native = np.arange(16, dtype=np.float32).reshape(2, 8)
    logical = map_logical_channels(native, (0, 1, 2, 3, 4, 5, 7, 6))
    assert np.array_equal(logical, native[:, [0, 1, 2, 3, 4, 5, 7, 6]])
    assert np.array_equal(logical[:, :7], native[:, [0, 1, 2, 3, 4, 5, 7]])
    assert np.array_equal(logical[:, 7], native[:, 6])


def test_unused_cdc_hotmap_queue_retains_only_the_latest_frame():
    device = SerialDevice("unused", 2_000_000)
    for value in range(3):
        payload = bytes([value]) * device.HOTMAP_PAYLOAD_SIZE
        device._publish(device.HOTMAP_HEADER + payload)

    frames = device.take_hotmap_frames()
    assert len(frames) == 1
    assert frames[0].sequence_id == 2
    assert np.all(frames[0].matrix == 2)
    assert device.latest_hotmap_frame() is frames[0]
    assert device.status()["hotmap_frames"] == 3


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
    assert ready.frequencies_hz.size == 201
    assert ready.frequencies_hz[0] == 0.0 and ready.frequencies_hz[-1] <= 10_000.0
    assert ready.noise_psd.shape == ready.smoothed_psd.shape == ready.minimum_psd.shape == ready.spp.shape == (7, 201)
    assert not hasattr(ready, "noise_covariance")
    assert ready.conditional_smoothed_psd.shape == ready.conditional_minimum_psd.shape == (7, 201)
    assert ready.speech_absence_probability.shape == ready.posterior_snr.shape == ready.prior_snr.shape == (7, 201)
    assert ready.noise_features.shape == (7, 4)
    assert np.isfinite(ready.noise_features).all()
    assert np.all((ready.spp >= 0.0) & (ready.spp <= 1.0))
    assert np.all((ready.speech_absence_probability >= 0.0) & (ready.speech_absence_probability <= 1.0))
    with pytest.raises(ValueError):
        ready.noise_psd.setflags(write=True)


def test_adapted_cohen_parameters_and_two_pass_state_are_exposed():
    config = load_config("config/config.yaml").layer1_imcra
    assert config.algorithm_version == "cohen_imcra_2003_l1_v9"
    assert config.n_fft == config.hop_samples == 960
    assert (config.frequency_min_hz, config.frequency_max_hz) == (250.0, 3_400.0)
    estimator = Layer1Imcra(config)
    gate_frequencies = estimator._all_frequencies_hz[estimator._gate_band]
    assert np.all((gate_frequencies >= 250.0) & (gate_frequencies <= 3_400.0))
    assert gate_frequencies[0] == pytest.approx(250.0)
    assert gate_frequencies[-1] == pytest.approx(3_400.0)
    weights = speech_gate_band_weights(gate_frequencies)
    assert weights.shape == gate_frequencies.shape
    assert np.sum(weights) == pytest.approx(1.0)
    assert np.all(weights > 0.0)
    expected_band_weights = (
        (250.0, 700.0, 0.15),
        (700.0, 1_600.0, 0.35),
        (1_600.0, 3_400.1, 0.50),
    )
    for low, high, expected in expected_band_weights:
        selected = (gate_frequencies >= low) & (gate_frequencies < high)
        assert np.sum(weights[selected]) == pytest.approx(expected, abs=1e-6)
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
    ) == (1, 0.77, 10, 5, 1.66, 4.6, 3.0, 1.67, 0.81, 0.66, 1.47)
    assert config.minimum_history_subwindows * config.minimum_subwindow_frames == 50
    assert config.warmup_seconds == 1.0
    first = estimator.process(
        _block(np.ones((960, 8), np.float32) * 0.01, epoch=0, start=0, sequence=0)
    )[0]
    assert np.array_equal(first.smoothed_psd, first.conditional_smoothed_psd)
    assert np.array_equal(first.minimum_psd, first.conditional_minimum_psd)
    assert np.all(first.speech_absence_probability == 1.0)
    assert np.all(first.spp == 0.0)


def test_imcra_frequency_smoother_matches_edge_padded_reference() -> None:
    config = load_config("config/config.yaml").layer1_imcra
    estimator = Layer1Imcra(config)
    values = np.random.default_rng(827_2026).random((7, 41))
    padded = np.pad(values, ((0, 0), (1, 1)), mode="edge")
    reference = (
        0.25 * padded[:, :-2]
        + 0.5 * padded[:, 1:-1]
        + 0.25 * padded[:, 2:]
    )
    np.testing.assert_allclose(estimator._frequency_smooth(values), reference)
    assert estimator._window_energy == pytest.approx(float(np.sum(estimator._window**2)))


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


def test_imcra_preserves_ready_statistics_across_epoch_and_ignores_hardware_mix():
    config = load_config("config/config.yaml")
    short = config.layer1_imcra.model_copy(update={"warmup_seconds": 0.04})
    left, right = Layer1Imcra(short), Layer1Imcra(short)
    rng = np.random.default_rng(11)
    physical = rng.normal(0.0, 0.01, (960, 7)).astype(np.float32)
    a = np.column_stack((physical, np.zeros(960, np.float32)))
    b = np.column_stack((physical, np.full(960, 1_000.0, np.float32)))
    hop_a = left.process(_block(a, epoch=0, start=0, sequence=0))[0]
    hop_b = right.process(_block(b, epoch=0, start=0, sequence=0))[0]
    assert np.array_equal(hop_a.noise_psd, hop_b.noise_psd)
    assert np.array_equal(hop_a.spp, hop_b.spp)
    assert left.process(_block(a, epoch=0, start=960, sequence=1))[0].state == "ready"

    recovered = left.process(_block(a, epoch=1, start=0, sequence=2))[0]

    assert recovered.start_sample == 0
    assert recovered.state == "ready"
    assert recovered.array_source_probability_20ms is not None


def test_imcra_new_session_still_requires_a_fresh_warmup():
    config = load_config("config/config.yaml").layer1_imcra.model_copy(
        update={"warmup_seconds": 0.04}
    )
    estimator = Layer1Imcra(config)
    samples = np.zeros((960, 8), dtype=np.float32)
    estimator.process(_block(samples, epoch=0, start=0, sequence=0))
    assert estimator.process(
        _block(samples, epoch=0, start=960, sequence=1)
    )[0].state == "ready"

    new_session = estimator.process(
        _block(samples, epoch=0, start=0, sequence=0, session="new-session")
    )[0]

    assert new_session.state == "warming_up"
    assert new_session.array_source_probability_20ms is None
    assert np.all(new_session.spp == 0.0)


def test_imcra_calibration_change_still_requires_a_fresh_warmup():
    config = load_config("config/config.yaml").layer1_imcra.model_copy(
        update={"warmup_seconds": 0.04}
    )
    estimator = Layer1Imcra(config)
    samples = np.zeros((960, 8), dtype=np.float32)
    estimator.process(_block(samples, epoch=0, start=0, sequence=0))
    assert estimator.process(
        _block(samples, epoch=0, start=960, sequence=1)
    )[0].state == "ready"
    changed_calibration = CalibrationMetadata(
        status="verified",
        version="calibration-v2",
        calibration_hash="1" * 64,
        correction_model="gain_polarity_integer_delay_v1",
    )

    changed = estimator.process(
        _block(
            samples,
            epoch=1,
            start=0,
            sequence=2,
            calibration=changed_calibration,
        )
    )[0]

    assert changed.state == "warming_up"
    assert changed.array_source_probability_20ms is None
    assert np.all(changed.spp == 0.0)


def test_coordinator_imcra_and_window_share_one_sample_axis():
    config = load_config("config/config.yaml")
    coordinator = IngestCoordinator(session_id="session")
    estimator = Layer1Imcra(config.layer1_imcra.model_copy(update={"warmup_seconds": 0.02}))
    assembler = WindowAssembler()
    windows = []
    for sequence in range(8):
        values = np.zeros((960, 8), np.float32)
        frame = DecodedAudio(values, 48_000, sequence, sequence * 0.02)
        block = coordinator.ingest(frame)
        hops = estimator.process(block)
        windows.extend(assembler.add(block, hops))
    assert len(windows) == 1
    window = windows[0]
    assert window.samples.shape == (7_680, 8)
    assert len(window.imcra_hops) == 8
    assert [(hop.start_sample, hop.end_sample) for hop in window.imcra_hops[-2:]] == [
        (window.doa_start_sample, window.doa_start_sample + 960),
        (window.doa_start_sample + 960, window.doa_end_sample),
    ]


def test_one_sequence_gap_preserves_ready_imcra_and_probability_recovers_immediately():
    config = load_config("config/config.yaml")
    coordinator = IngestCoordinator(session_id="single-gap-warmup")
    estimator = Layer1Imcra(config.layer1_imcra)
    assembler = WindowAssembler()
    samples = np.zeros((960, 8), dtype=np.float32)

    warmup_hops = int(np.ceil(config.layer1_imcra.warmup_seconds / 0.02))
    for sequence in range(warmup_hops):
        decoded = DecodedAudio(samples, 48_000, sequence, sequence * 0.02)
        block = coordinator.ingest(decoded)
        hops = estimator.process(block)
        assembler.add(block, hops)
    assert hops[0].state == "ready"

    recovered_states: list[str] = []
    recovered_windows = []
    # One block is genuinely absent.  The sequence and timestamp gaps describe
    # the same discontinuity; the next epoch starts at sample zero.
    for sequence in range(warmup_hops + 1, warmup_hops + 10):
        decoded = DecodedAudio(samples, 48_000, sequence, sequence * 0.02)
        block = coordinator.ingest(decoded)
        hops = estimator.process(block)
        recovered_windows.extend(assembler.add(block, hops))
        recovered_states.append(hops[0].state)

    assert coordinator.stream_epoch == 1
    assert len(coordinator.discontinuities) == 1
    assert coordinator.discontinuities[0].reason == "sequence_gap"
    assert recovered_states == ["ready"] * len(recovered_states)
    assert recovered_windows
    assert all(hop.state == "ready" for hop in recovered_windows[0].imcra_hops[-2:])
    assert assembler.status.stream_epoch == 1
    assert assembler.status.state == "running"
