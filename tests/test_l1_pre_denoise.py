from pathlib import Path

import numpy as np
import pytest

from common.config import load_config
from common.data_types import ImcraHopSnapshot, IngestedAudioBlock
from layer1_input.pre_denoise import ImcraWienerPreDenoiser


CONFIG = Path(__file__).parents[1] / "config" / "config.yaml"


def _block(samples: np.ndarray, index: int, hop: ImcraHopSnapshot | None = None) -> IngestedAudioBlock:
    start = index * 960
    return IngestedAudioBlock(
        "session", 0, start, start + len(samples), 48_000, index, index * 0.02,
        np.asarray(samples, np.float32), imcra_hop=hop,
    )


def _hop(index: int, *, spp_by_mic: np.ndarray, state: str = "ready") -> ImcraHopSnapshot:
    frequencies = np.fft.rfftfreq(2048, 1.0 / 48_000).astype(np.float32)
    frequencies = frequencies[frequencies <= 8_000.0]
    shape = (7, frequencies.size)
    ones = np.ones(shape, np.float32)
    spp = np.broadcast_to(np.asarray(spp_by_mic, np.float32)[:, None], shape).copy()
    probability = np.mean(spp, axis=1).astype(np.float32)
    start = index * 960
    return ImcraHopSnapshot(
        "session", 0, start, start + 960, (index,), "cohen_imcra_2003_l1_v2", state,
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
        output.extend(item.denoised.samples for item in denoiser.process(
            _block(source[index * 960 : (index + 1) * 960], index, hop)
        ))
    output.extend(item.denoised.samples for item in denoiser.flush())
    np.testing.assert_allclose(np.concatenate(output), source, atol=2.0e-7, rtol=0.0)


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


def test_pre_denoiser_applies_imcra_gain_from_dc_through_8000_hz() -> None:
    denoiser = ImcraWienerPreDenoiser.from_project(load_config(CONFIG, environ={}))
    assert denoiser._frequencies[denoiser._output_band][0] == 0.0
    assert denoiser._frequencies[denoiser._output_band][-1] <= 8_000.0
    assert np.count_nonzero(denoiser._output_band) == 342

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
