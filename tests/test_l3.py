from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from common.config import load_config
from common.data_types import CandidateDirection, DecisionWindow, ImcraHopSnapshot
from common.geometry import physical_6plus1_geometry
from layer3_direction_signal import (
    L3_MODE_CONSTANT_BEAMWIDTH,
    L3_MODE_DS_BASELINE,
    Layer3Processor,
)
from layer3_direction_signal.configuration import SpatialSeparationConfig
from layer3_direction_signal.constant_beamwidth import constant_beamwidth_weights
from layer3_direction_signal.steering import steering_vectors


CONFIG = Path(__file__).parents[1] / "config/config.yaml"


def _imcra_hops(
    *, noise_by_mic: np.ndarray | None = None, spp_by_hop: np.ndarray | None = None,
) -> tuple[ImcraHopSnapshot, ...]:
    frequencies = np.fft.rfftfreq(2048, 1 / 48_000).astype(np.float32)
    frequencies = frequencies[frequencies <= 8_000.0]
    spectral = (7, len(frequencies))
    noise_by_mic = np.ones(7, np.float32) if noise_by_mic is None else np.asarray(noise_by_mic, np.float32)
    noise = np.broadcast_to(noise_by_mic[:, None], spectral).copy()
    ones = np.ones(spectral, np.float32)
    spp_by_hop = np.full(8, 0.2, np.float32) if spp_by_hop is None else np.asarray(spp_by_hop, np.float32)
    return tuple(
        ImcraHopSnapshot(
            "session", 0, index * 960, (index + 1) * 960, (index,),
            "cohen_imcra_2003_l1_v2", "ready", frequencies,
            noise, ones * 2.0, ones * 1.5, ones * 0.5, ones * 0.4,
            np.full(spectral, spp_by_hop[index], np.float32),
            np.full(spectral, 1.0 - spp_by_hop[index], np.float32),
            ones * 4.0, ones * 3.0,
            np.ones((7, 4), np.float32),
            10.0 * np.log10(noise_by_mic),
            np.full(7, spp_by_hop[index], np.float32), float(spp_by_hop[index]),
        )
        for index in range(8)
    )


def _window(
    samples: np.ndarray, *, ready_imcra: bool = True, noise_by_mic: np.ndarray | None = None,
    spp_by_hop: np.ndarray | None = None,
) -> DecisionWindow:
    hops = _imcra_hops(noise_by_mic=noise_by_mic, spp_by_hop=spp_by_hop) if ready_imcra else ()
    return DecisionWindow(
        "session", 0, 4, 7_680, 5_760, 7_680, 0, 7_680, 48_000, samples, (0,), hops,
    )


def _candidate(theta: float) -> CandidateDirection:
    return CandidateDirection("session", 0, 4, 7_680, 5_760, 7_680, theta, 1.0, 0.8)


def test_l3_empty_candidates_skips_outputs():
    processor = Layer3Processor(load_config(CONFIG, environ={}))
    output = processor.process(_window(np.zeros((7_680, 8), np.float32)), (), physical_6plus1_geometry())
    assert output.enhanced_audio == ()


def test_l3_outputs_one_48khz_mono_audio_per_candidate():
    rng = np.random.default_rng(42)
    processor = Layer3Processor(load_config(CONFIG, environ={}))
    output = processor.process(
        _window(rng.normal(0, 0.03, (7_680, 8)).astype(np.float32)),
        (_candidate(0.0), _candidate(359.0)), physical_6plus1_geometry(),
    )
    assert len(output.enhanced_audio) == 2
    assert tuple(item.theta_deg for item in output.enhanced_audio) == (0.0, 359.0)
    for item in output.enhanced_audio:
        assert item.sample_rate == 48_000
        assert item.enhanced_audio.shape == (7_680,)
        assert item.enhanced_audio.dtype == np.float32
        assert not item.enhanced_audio.flags.writeable
        assert np.isfinite(item.enhanced_audio).all()
        assert item.diagnostics[0] == "backend=imcra_spatial_separation"
        assert not hasattr(item, "stft_complex")


def test_l3_rejects_candidate_from_another_window():
    processor = Layer3Processor(load_config(CONFIG, environ={}))
    wrong = CandidateDirection("session", 0, 5, 8_640, 6_720, 8_640, 30.0, 1.0, 0.8)
    with pytest.raises(RuntimeError, match="同一窗口"):
        processor.process(_window(np.zeros((7_680, 8), np.float32)), (wrong,), physical_6plus1_geometry())


def test_l3_rejects_old_seven_channel_input_contract():
    with pytest.raises(ValueError, match="7680, 8"):
        _window(np.zeros((7_680, 7), np.float32))


def test_l3_accepts_three_candidates_and_rejects_four():
    processor = Layer3Processor(load_config(CONFIG, environ={}))
    output = processor.process(
        _window(np.zeros((7_680, 8), np.float32)),
        (_candidate(0.0), _candidate(90.0), _candidate(180.0)),
        physical_6plus1_geometry(),
    )
    assert len(output.enhanced_audio) == 3
    with pytest.raises(RuntimeError, match="0、1、2或3"):
        processor.process(
            _window(np.zeros((7_680, 8), np.float32)),
            (_candidate(0.0), _candidate(90.0), _candidate(180.0), _candidate(270.0)),
            physical_6plus1_geometry(),
        )


def test_hardware_mix_channel_never_changes_l3_output():
    rng = np.random.default_rng(123)
    physical = rng.normal(0, 0.03, (7_680, 7)).astype(np.float32)
    first = np.column_stack((physical, np.zeros(7_680, np.float32)))
    second = np.column_stack((physical, rng.normal(0, 10.0, 7_680).astype(np.float32)))
    processor = Layer3Processor(load_config(CONFIG, environ={}))
    geometry = physical_6plus1_geometry()

    a = processor.process(_window(first), (_candidate(42.0),), geometry).enhanced_audio[0]
    b = processor.process(_window(second), (_candidate(42.0),), geometry).enhanced_audio[0]

    np.testing.assert_array_equal(a.enhanced_audio, b.enhanced_audio)


def test_l3_config_has_spatial_p_lookup_branches():
    layer3 = load_config(CONFIG, environ={}).layer3
    assert (layer3.frequency_min_hz, layer3.frequency_max_hz) == (80.0, 8_000.0)
    assert (layer3.rho_lcmv_max, layer3.rho_soft_null_max) == (0.3, 0.7)
    assert layer3.main_backend == "imcra_spatial_separation"


def test_missing_imcra_context_degrades_the_complete_window_to_das():
    processor = Layer3Processor(load_config(CONFIG, environ={}))
    item = processor.process(
        _window(np.zeros((7_680, 8), np.float32), ready_imcra=False),
        (_candidate(20.0),), physical_6plus1_geometry(),
    ).enhanced_audio[0]
    assert item.algorithm == "das"
    assert item.fallback_reason is not None and "8个hop" in item.fallback_reason


def test_single_candidate_uses_imcra_loaded_mvdr():
    rng = np.random.default_rng(77)
    processor = Layer3Processor(load_config(CONFIG, environ={}))
    item = processor.process(
        _window(rng.normal(0, 0.02, (7_680, 8)).astype(np.float32)),
        (_candidate(20.0),), physical_6plus1_geometry(),
    ).enhanced_audio[0]
    assert item.algorithm == "imcra_spatial_separation"
    assert any("loaded_mvdr=" in message for message in item.diagnostics)


def test_ds_baseline_uses_seven_channel_delay_and_sum_without_imcra_or_spatial_p():
    rng = np.random.default_rng(770)
    samples = rng.normal(0, 0.02, (7_680, 8)).astype(np.float32)
    processor = Layer3Processor(load_config(CONFIG, environ={}))
    output = processor.process(
        _window(samples, ready_imcra=False),
        (_candidate(20.0), _candidate(80.0)),
        physical_6plus1_geometry(),
        mode=L3_MODE_DS_BASELINE,
    )

    assert len(output.enhanced_audio) == 2
    for item in output.enhanced_audio:
        assert item.algorithm == "ds_baseline"
        assert item.fallback_reason is None
        assert item.diagnostics == (
            "backend=ds_baseline",
            "comparison_only=true",
            "physical_channels=7",
            "imcra=unused",
            "spatial_p=unused",
        )
        assert np.isfinite(item.enhanced_audio).all()


def test_constant_beamwidth_baseline_targets_30_degree_fnbw_without_imcra_or_p():
    rng = np.random.default_rng(771)
    processor = Layer3Processor(load_config(CONFIG, environ={}))
    output = processor.process(
        _window(rng.normal(0, 0.02, (7_680, 8)).astype(np.float32), ready_imcra=False),
        (_candidate(20.0), _candidate(110.0)),
        physical_6plus1_geometry(),
        mode=L3_MODE_CONSTANT_BEAMWIDTH,
    )

    assert len(output.enhanced_audio) == 2
    for item in output.enhanced_audio:
        assert item.algorithm == "constant_beamwidth_baseline"
        assert item.diagnostics[0] == "backend=constant_beamwidth_baseline"
        assert "target_fnbw_deg=30.0" in item.diagnostics
        assert "imcra=unused" in item.diagnostics
        assert "spatial_p=unused" in item.diagnostics
        assert any(value.startswith("das_fallback_bins=") for value in item.diagnostics)
        assert np.isfinite(item.enhanced_audio).all()


def test_constant_beamwidth_has_30_degree_first_null_where_uca_can_realize_it():
    project = load_config(CONFIG, environ={})
    config = SpatialSeparationConfig.from_project(project)
    geometry = physical_6plus1_geometry()
    frequencies = torch.fft.rfftfreq(1024, d=1.0 / 48_000)
    theta = torch.tensor([0.0])
    target = steering_vectors(frequencies, theta, geometry)
    solved = constant_beamwidth_weights(frequencies, target, theta, geometry, config)

    # Around 6 kHz the 4 cm UCA can safely realize the requested +/-15 degree
    # first null. Low frequencies cannot and must remain the exact DAS fallback.
    six_khz = round(6_000 / (48_000 / 1024))
    probes = steering_vectors(frequencies, torch.tensor([0.0, 15.0, 345.0]), geometry)
    response = torch.einsum(
        "c,mc->m", solved.weights_mfc[0, six_khz].conj(), probes[:, six_khz],
    ).abs()
    assert response[0] == pytest.approx(1.0, abs=1e-4)
    assert response[1] < 1e-3
    assert response[2] < 1e-3

    five_hundred_hz = round(500 / (48_000 / 1024))
    np.testing.assert_allclose(
        solved.weights_mfc[0, five_hundred_hz].numpy(),
        (target[0, five_hundred_hz] / 7.0).numpy(),
        rtol=0,
        atol=0,
    )


def test_l3_rejects_unknown_processing_mode():
    processor = Layer3Processor(load_config(CONFIG, environ={}))
    with pytest.raises(ValueError, match="未知L3处理模式"):
        processor.process(
            _window(np.zeros((7_680, 8), np.float32)),
            (_candidate(20.0),),
            physical_6plus1_geometry(),
            mode="unknown",
        )


def test_imcra_noise_psd_changes_the_adaptive_solution():
    rng = np.random.default_rng(88)
    samples = rng.normal(0, 0.02, (7_680, 8)).astype(np.float32)
    processor = Layer3Processor(load_config(CONFIG, environ={}))
    geometry = physical_6plus1_geometry()
    balanced = processor.process(
        _window(samples), (_candidate(35.0),), geometry,
    ).enhanced_audio[0].enhanced_audio
    noisy_mic0 = processor.process(
        _window(samples, noise_by_mic=np.asarray((100.0, 1, 1, 1, 1, 1, 1))),
        (_candidate(35.0),), geometry,
    ).enhanced_audio[0].enhanced_audio
    assert not np.allclose(balanced, noisy_mic0, rtol=1e-5, atol=1e-7)


def test_imcra_spp_controls_noise_covariance_updates():
    rng = np.random.default_rng(99)
    samples = np.zeros((7_680, 8), np.float32)
    samples[:3_840, 0] = rng.normal(0, 0.08, 3_840)
    samples[3_840:, 1] = rng.normal(0, 0.08, 3_840)
    first_is_speech = np.asarray((0.95,) * 4 + (0.05,) * 4, np.float32)
    second_is_speech = first_is_speech[::-1].copy()
    processor = Layer3Processor(load_config(CONFIG, environ={}))
    geometry = physical_6plus1_geometry()

    first = processor.process(
        _window(samples, spp_by_hop=first_is_speech), (_candidate(40.0),), geometry,
    ).enhanced_audio[0].enhanced_audio
    second = processor.process(
        _window(samples, spp_by_hop=second_is_speech), (_candidate(40.0),), geometry,
    ).enhanced_audio[0].enhanced_audio

    assert not np.allclose(first, second, rtol=1e-5, atol=1e-7)
