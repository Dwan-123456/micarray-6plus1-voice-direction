from pathlib import Path

import numpy as np
import pytest

from common.config import load_config
from common.data_types import ImcraHopSnapshot, IngestedAudioBlock
from layer1_input.pre_denoise import ImcraWienerPreDenoiser


CONFIG = Path(__file__).parents[1] / "config" / "config.yaml"


def _block(
    samples: np.ndarray,
    index: int,
    hop: ImcraHopSnapshot | None = None,
    *,
    epoch: int = 0,
) -> IngestedAudioBlock:
    start = index * 960
    return IngestedAudioBlock(
        "session", epoch, start, start + len(samples), 48_000, index, index * 0.02,
        np.asarray(samples, np.float32), imcra_hop=hop,
    )


def _hop(index: int, *, spp_by_mic: np.ndarray, state: str = "ready") -> ImcraHopSnapshot:
    frequencies = np.fft.rfftfreq(960, 1.0 / 48_000).astype(np.float32)
    frequencies = frequencies[frequencies <= 10_000.0]
    shape = (7, frequencies.size)
    ones = np.ones(shape, np.float32)
    spp = np.broadcast_to(np.asarray(spp_by_mic, np.float32)[:, None], shape).copy()
    probability = np.mean(spp, axis=1).astype(np.float32)
    start = index * 960
    return ImcraHopSnapshot(
        "session", 0, start, start + 960, (index,), "cohen_imcra_2003_l1_v11", state,
        frequencies, ones, ones, ones, ones, ones, spp, 1.0 - spp, ones,
        np.zeros(shape, np.float32),
        np.column_stack((np.zeros(7), np.zeros(7), np.zeros(7), probability)).astype(np.float32),
        np.zeros(7, np.float32), probability,
        float(np.median(probability)) if state == "ready" else None,
    )


def test_wola_identity_is_continuous_during_imcra_warmup() -> None:
    denoiser = ImcraWienerPreDenoiser.from_project(load_config(CONFIG, environ={}))
    rng = np.random.default_rng(41)
    source = rng.normal(0.0, 0.05, (4 * 960, 8)).astype(np.float32)
    output = []
    for index in range(4):
        hop = _hop(index, spp_by_mic=np.zeros(7), state="warming_up")
        output.extend(denoiser.process(
            _block(source[index * 960 : (index + 1) * 960], index, hop)
        ))
    output.extend(denoiser.flush())
    assert len(output) == 4
    assert [(item.raw.start_sample, item.raw.end_sample) for item in output] == [
        (index * 960, (index + 1) * 960) for index in range(4)
    ]
    assert all(item.denoised.samples.shape == (960, 8) for item in output)
    np.testing.assert_allclose(
        np.concatenate([item.denoised.samples for item in output]),
        source,
        atol=2.0e-7,
        rtol=0.0,
    )


def test_gain_smoothing_updates_once_per_20ms_block() -> None:
    denoiser = ImcraWienerPreDenoiser.from_project(load_config(CONFIG, environ={}))
    hop = _hop(0, spp_by_mic=np.zeros(7))
    denoiser.process(_block(np.zeros((960, 8), np.float32), 0, hop))

    alpha = denoiser.config.gain_smoothing
    expected = alpha + (1.0 - alpha) * denoiser._minimum_gain
    np.testing.assert_allclose(
        denoiser._previous_gain[denoiser._output_band],
        expected,
        atol=1.0e-12,
        rtol=0.0,
    )


def test_epoch_change_flushes_old_block_without_cross_epoch_wola() -> None:
    denoiser = ImcraWienerPreDenoiser.from_project(load_config(CONFIG, environ={}))
    rng = np.random.default_rng(57)
    old_audio = rng.normal(0.0, 0.05, (2, 960, 8)).astype(np.float32)
    new_audio = rng.normal(0.0, 0.05, (2, 960, 8)).astype(np.float32)

    output = []
    output.extend(denoiser.process(_block(old_audio[0], 0, epoch=0)))
    output.extend(denoiser.process(_block(old_audio[1], 1, epoch=0)))
    output.extend(denoiser.process(_block(new_audio[0], 0, epoch=1)))
    output.extend(denoiser.process(_block(new_audio[1], 1, epoch=1)))
    output.extend(denoiser.flush())

    expected_identity = [
        (0, 0, 960, 0),
        (0, 960, 1_920, 1),
        (1, 0, 960, 0),
        (1, 960, 1_920, 1),
    ]
    assert [
        (item.raw.stream_epoch, item.raw.start_sample, item.raw.end_sample, item.raw.sequence_id)
        for item in output
    ] == expected_identity
    assert [
        (
            item.denoised.stream_epoch,
            item.denoised.start_sample,
            item.denoised.end_sample,
            item.denoised.sequence_id,
        )
        for item in output
    ] == expected_identity
    np.testing.assert_allclose(
        np.stack([item.denoised.samples for item in output[:2]]),
        old_audio,
        atol=2.0e-7,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.stack([item.denoised.samples for item in output[2:]]),
        new_audio,
        atol=2.0e-7,
        rtol=0.0,
    )
    assert denoiser.flush() == ()


def test_each_microphone_uses_its_own_mask_and_hardware_mix_is_untouched() -> None:
    denoiser = ImcraWienerPreDenoiser.from_project(load_config(CONFIG, environ={}))
    time = np.arange(960, dtype=np.float64) / 48_000.0
    tone = (0.2 * np.sin(2.0 * np.pi * 1_000.0 * time)).astype(np.float32)
    source = np.repeat(tone[:, None], 8, axis=1)
    output = []
    spp = np.asarray((1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), np.float32)
    for index in range(30):
        output.extend(item.denoised.samples for item in denoiser.process(
            _block(source, index, _hop(index, spp_by_mic=spp))
        ))
    output.extend(item.denoised.samples for item in denoiser.flush())
    settled = np.concatenate(output)[-5 * 960 :]
    rms = np.sqrt(np.mean(np.square(settled, dtype=np.float64), axis=0))
    assert rms[0] > 0.95 * rms[7]
    assert rms[1] < 0.20 * rms[0]
    np.testing.assert_allclose(settled[:, 7], np.tile(tone, 5), atol=2.0e-7, rtol=0.0)


def test_pre_denoiser_applies_imcra_gain_from_dc_through_10000_hz() -> None:
    denoiser = ImcraWienerPreDenoiser.from_project(load_config(CONFIG, environ={}))
    assert (denoiser.frame_samples, denoiser.hop_samples, denoiser.config.n_fft) == (
        960, 480, 960,
    )
    assert denoiser._frequencies[denoiser._output_band][0] == 0.0
    assert denoiser._frequencies[denoiser._output_band][-1] <= 10_000.0
    assert np.count_nonzero(denoiser._output_band) == 201

    source = np.full((960, 8), 0.1, np.float32)
    output = []
    for index in range(30):
        output.extend(item.denoised.samples for item in denoiser.process(
            _block(source, index, _hop(index, spp_by_mic=np.zeros(7)))
        ))
    output.extend(item.denoised.samples for item in denoiser.flush())
    settled = np.concatenate(output)[-5 * 960 :]
    assert np.sqrt(np.mean(np.square(settled[:, 0], dtype=np.float64))) < 0.02
    np.testing.assert_allclose(settled[:, 7], 0.1, atol=2.0e-7, rtol=0.0)


def test_pre_denoiser_rejects_wrong_block_size_and_discontinuity() -> None:
    denoiser = ImcraWienerPreDenoiser.from_project(load_config(CONFIG, environ={}))
    with pytest.raises(ValueError, match="exact 20 ms"):
        denoiser.process(_block(np.zeros((480, 8), np.float32), 0))
    denoiser.process(_block(np.zeros((960, 8), np.float32), 0))
    with pytest.raises(ValueError, match="discontinuity"):
        denoiser.process(_block(np.zeros((960, 8), np.float32), 2))
