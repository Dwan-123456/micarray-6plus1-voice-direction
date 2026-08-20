from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from common.config import load_config
from common.data_types import CandidateDirection, DecisionWindow, ImcraHopSnapshot, TrackedDirection
from common.window_key import WindowKey
from common.geometry import physical_6plus1_geometry
from layer3_direction_signal import (
    BeamformedL3Batch,
    L3_MODE_DS_BASELINE,
    L3_MODE_SUBBAND_ROBUST,
    Layer3Processor,
)
from layer3_direction_signal.configuration import SpatialSeparationConfig
from layer3_direction_signal.subband_robust import subband_robust_weights
from layer3_direction_signal.steering import steering_vectors


CONFIG = Path(__file__).parents[1] / "config/config.yaml"


def _imcra_hops(
    *, noise_by_mic: np.ndarray | None = None, spp_by_hop: np.ndarray | None = None,
) -> tuple[ImcraHopSnapshot, ...]:
    frequencies = np.fft.rfftfreq(2048, 1 / 48_000).astype(np.float32)
    frequencies = frequencies[(frequencies >= 80.0) & (frequencies <= 8_000.0)]
    spectral = (7, len(frequencies))
    noise_by_mic = np.ones(7, np.float32) if noise_by_mic is None else np.asarray(noise_by_mic, np.float32)
    noise = np.broadcast_to(noise_by_mic[:, None], spectral).copy()
    ones = np.ones(spectral, np.float32)
    spp_by_hop = np.full(8, 0.2, np.float32) if spp_by_hop is None else np.asarray(spp_by_hop, np.float32)
    return tuple(
        ImcraHopSnapshot(
            "session", 0, index * 960, (index + 1) * 960, (index,),
            "cohen_imcra_2003_l1_v1", "ready", frequencies,
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


def _candidate(
    theta: float,
    *,
    track_id: int | None = None,
    rank: int | None = None,
    observed: bool = True,
    allow_prediction: bool = False,
) -> TrackedDirection:
    identity = int(round(theta * 10.0)) + 1 if track_id is None else track_id
    original_rank = identity if rank is None else rank
    last_observed = 7_680 if observed else 6_720
    return TrackedDirection(
        "session", 0, 4, 7_680, 5_760, 7_680,
        identity, original_rank, theta if observed else None, theta, 1.0, 0.8,
        "confirmed" if observed else "coasting", observed, False,
        0, last_observed, 7_680 - last_observed, False, allow_prediction,
    )


def test_l3_empty_candidates_skips_outputs():
    processor = Layer3Processor(load_config(CONFIG, environ={}))
    output = processor.process(_window(np.zeros((7_680, 8), np.float32)), (), physical_6plus1_geometry())
    assert output.enhanced_audio == ()
    assert output.window_key == WindowKey("session", 0, 4, 7_680)


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


@pytest.mark.parametrize("mode", ("optimized", "ds_baseline", "subband_robust_baseline"))
def test_all_three_modes_preserve_authoritative_ids_angles_and_original_order(mode):
    processor = Layer3Processor(load_config(CONFIG, environ={}))
    directions = (
        _candidate(359.5, track_id=73, rank=2),
        _candidate(0.5, track_id=11, rank=1),
        _candidate(180.0, track_id=42, rank=3),
    )
    output = processor.process(
        _window(np.zeros((7_680, 8), np.float32)),
        directions,
        physical_6plus1_geometry(),
        mode=mode,
    )
    assert tuple(item.track_id for item in output.enhanced_audio) == (73, 11, 42)
    assert tuple(item.rank for item in output.enhanced_audio) == (2, 1, 3)
    assert tuple(item.theta_deg for item in output.enhanced_audio) == (359.5, 0.5, 180.0)
    assert all(item.window_key == output.window_key for item in output.enhanced_audio)


def test_l3_rejects_duplicate_and_missing_public_ids_without_angle_repair():
    processor = Layer3Processor(load_config(CONFIG, environ={}))
    window = _window(np.zeros((7_680, 8), np.float32))
    geometry = physical_6plus1_geometry()
    with pytest.raises(RuntimeError, match="track_ids must be unique"):
        processor.process(
            window,
            (_candidate(5.0, track_id=9), _candidate(95.0, track_id=9)),
            geometry,
        )
    legacy = CandidateDirection("session", 0, 4, 7_680, 5_760, 7_680, 5.0, 1.0, 0.8)
    with pytest.raises(RuntimeError, match="only public L2 TrackedDirection"):
        processor.process(window, (legacy,), geometry)  # type: ignore[arg-type]


def test_l3_allows_only_explicit_short_prediction_and_rejects_long_coasting():
    processor = Layer3Processor(load_config(CONFIG, environ={}))
    window = _window(np.zeros((7_680, 8), np.float32), ready_imcra=False)
    geometry = physical_6plus1_geometry()
    predicted = _candidate(12.0, track_id=5, observed=False, allow_prediction=True)
    output = processor.process(window, (predicted,), geometry, mode=L3_MODE_DS_BASELINE)
    assert output.enhanced_audio[0].track_id == 5
    long_coasting = _candidate(12.0, track_id=5, observed=False, allow_prediction=False)
    with pytest.raises(RuntimeError, match="long-coasting"):
        processor.process(window, (long_coasting,), geometry, mode=L3_MODE_DS_BASELINE)


def test_l3_exit_rejects_beamformed_id_order_corruption(monkeypatch):
    processor = Layer3Processor(load_config(CONFIG, environ={}))
    window = _window(np.zeros((7_680, 8), np.float32), ready_imcra=False)
    directions = (
        _candidate(25.0, track_id=4, rank=2),
        _candidate(120.0, track_id=8, rank=1),
    )
    geometry = physical_6plus1_geometry()
    prepared = processor.prepare(window, mode=L3_MODE_DS_BASELINE)
    valid = processor.beamformer.process_prepared_batch(prepared, directions, geometry)
    corrupted = BeamformedL3Batch(
        valid.window_key,
        valid.spectra_mft,
        tuple(reversed(valid.track_ids)),
        valid.ranks,
        valid.theta_degrees,
        valid.backends,
        valid.fallback_reasons,
        valid.diagnostics,
    )
    monkeypatch.setattr(
        processor.beamformer,
        "process_prepared_batch",
        lambda *_args, **_kwargs: corrupted,
    )
    with pytest.raises(RuntimeError, match="track_id set or order"):
        processor.process_prepared(prepared, directions, geometry)


def test_l3_rejects_candidate_from_another_window():
    processor = Layer3Processor(load_config(CONFIG, environ={}))
    wrong = TrackedDirection(
        "session", 0, 5, 8_640, 6_720, 8_640, 301, 1,
        30.0, 30.0, 1.0, 0.8, "confirmed", True, False,
        0, 8_640, 0, False,
    )
    with pytest.raises(RuntimeError, match="WindowKey"):
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
    with pytest.raises(RuntimeError, match="zero to three"):
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


def test_subband_robust_baseline_uses_imcra_five_bands_without_spatial_p():
    rng = np.random.default_rng(771)
    processor = Layer3Processor(load_config(CONFIG, environ={}))
    output = processor.process(
        _window(rng.normal(0, 0.02, (7_680, 8)).astype(np.float32)),
        (_candidate(20.0), _candidate(110.0)),
        physical_6plus1_geometry(),
        mode=L3_MODE_SUBBAND_ROBUST,
    )

    assert len(output.enhanced_audio) == 2
    for item in output.enhanced_audio:
        assert item.algorithm == "subband_robust_baseline"
        assert item.diagnostics[0] == "backend=subband_robust_baseline"
        assert any(value.startswith("imcra=cohen_imcra_2003_l1_v1") for value in item.diagnostics)
        assert "rtf_source=free_field_steering_proxy_v1" in item.diagnostics
        assert "source_scm=rank1_direction_fit_v1" in item.diagnostics
        assert any(value.startswith("bands:80-500=") for value in item.diagnostics)
        assert "low=mild_interference_mvdr+wiener" in item.diagnostics
        assert "low_mid=wng_constrained_soft_lcmv" in item.diagnostics
        assert "mid_core=strong_lcmv" in item.diagnostics
        assert "high=alias_loaded_mvdr" in item.diagnostics
        assert "spatial_p=unused" in item.diagnostics
        assert any(value.startswith("das_fallback_bins=") for value in item.diagnostics)
        assert np.isfinite(item.enhanced_audio).all()


def test_subband_robust_missing_imcra_falls_back_without_affecting_other_modes():
    processor = Layer3Processor(load_config(CONFIG, environ={}))
    item = processor.process(
        _window(np.zeros((7_680, 8), np.float32), ready_imcra=False),
        (_candidate(20.0),),
        physical_6plus1_geometry(),
        mode=L3_MODE_SUBBAND_ROBUST,
    ).enhanced_audio[0]
    assert item.algorithm == "das"
    assert item.fallback_reason is not None and "IMCRA adaptive BF unavailable" in item.fallback_reason


def test_subband_robust_output_responds_to_imcra_noise_psd():
    rng = np.random.default_rng(773)
    samples = rng.normal(0, 0.02, (7_680, 8)).astype(np.float32)
    geometry = physical_6plus1_geometry()
    processor = Layer3Processor(load_config(CONFIG, environ={}))
    balanced = processor.process(
        _window(samples),
        (_candidate(30.0), _candidate(120.0)),
        geometry,
        mode=L3_MODE_SUBBAND_ROBUST,
    ).enhanced_audio[0].enhanced_audio
    noisy_mic0 = processor.process(
        _window(samples, noise_by_mic=np.asarray((100.0, 1, 1, 1, 1, 1, 1))),
        (_candidate(30.0), _candidate(120.0)),
        geometry,
        mode=L3_MODE_SUBBAND_ROBUST,
    ).enhanced_audio[0].enhanced_audio
    assert not np.allclose(balanced, noisy_mic0, rtol=1e-5, atol=1e-7)


def test_subband_robust_wng_guard_and_low_frequency_wiener_gain():
    project = load_config(CONFIG, environ={})
    config = SpatialSeparationConfig.from_project(project)
    geometry = physical_6plus1_geometry()
    frequencies = torch.fft.rfftfreq(1024, d=1.0 / 48_000)
    target = steering_vectors(frequencies, torch.tensor([0.0, 120.0]), geometry)
    generator = torch.Generator().manual_seed(772)
    spectrum = torch.complex(
        torch.randn((513, 7, 17), generator=generator),
        torch.randn((513, 7, 17), generator=generator),
    ).to(torch.complex64)
    covariance = torch.eye(7, dtype=torch.complex64).expand(513, 7, 7).clone()
    solved = subband_robust_weights(
        covariance,
        spectrum,
        target,
        frequencies,
        torch.ones(513),
        config,
    )
    low_mid = (frequencies >= 500) & (frequencies < 900)
    wng = torch.reciprocal(solved.weights_mfc.abs().square().sum(dim=-1))
    assert torch.all(wng[:, low_mid] >= 1.0 - 1e-5)
    assert torch.all(solved.postfilter_mf[:, frequencies >= 500] == 1.0)
    assert torch.all(solved.postfilter_mf[:, frequencies < 500] <= 1.0)
    assert torch.any(solved.postfilter_mf[:, frequencies < 500] < 1.0)
    assert sum(solved.band_bins) == int(((frequencies >= 80) & (frequencies <= 8_000)).sum())
    assert torch.isfinite(solved.weights_mfc).all()


def test_removed_constant_beamwidth_mode_is_rejected():
    processor = Layer3Processor(load_config(CONFIG, environ={}))
    with pytest.raises(ValueError, match="未知L3处理模式"):
        processor.process(
            _window(np.zeros((7_680, 8), np.float32)),
            (_candidate(20.0),),
            physical_6plus1_geometry(),
            mode="constant_beamwidth_baseline",
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
