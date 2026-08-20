from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
import torch

from common.config import load_config
from common.data_types import CandidateDirection, DecisionWindow, ImcraHopSnapshot
from common.geometry import physical_6plus1_geometry
from layer3_direction_signal import Layer3Processor, PreparedL3Context
from layer3_direction_signal.configuration import SpatialSeparationConfig, StftSettings
from layer3_direction_signal.noise_context import BeamformerNoiseContext, RollingNoiseStatisticsCache
from layer3_direction_signal.shared_stft import RollingStftCache, shared_stft


CONFIG = Path(__file__).parents[1] / "config/config.yaml"


def _hop(index: int) -> ImcraHopSnapshot:
    frequencies = np.fft.rfftfreq(2048, 1 / 48_000).astype(np.float32)
    frequencies = frequencies[frequencies <= 10_000.0]
    shape = (7, len(frequencies))
    scale = np.float32(1.0 + index * 0.01)
    noise = np.full(shape, scale, np.float32)
    spp_value = np.float32(0.1 + (index % 8) * 0.05)
    spp = np.full(shape, spp_value, np.float32)
    ones = np.ones(shape, np.float32)
    return ImcraHopSnapshot(
        "cache-session",
        0,
        index * 960,
        (index + 1) * 960,
        (index,),
        "cohen_imcra_2003_l1_v3",
        "ready",
        frequencies,
        noise,
        ones * (2.0 + scale),
        ones * (1.5 + scale),
        ones * 0.5,
        ones * 0.4,
        spp,
        1.0 - spp,
        ones * (4.0 + index * 0.02),
        ones * (3.0 + index * 0.01),
        np.ones((7, 4), np.float32),
        np.full(7, np.float32(index * 0.01), np.float32),
        np.full(7, spp_value, np.float32),
        float(spp_value),
    )


def _window(
    continuous: np.ndarray,
    hops: tuple[ImcraHopSnapshot, ...],
    index: int,
    *,
    session_id: str = "cache-session",
    epoch: int = 0,
) -> DecisionWindow:
    start = index * 960
    end = start + 7_680
    return DecisionWindow(
        session_id,
        epoch,
        index,
        end,
        end - 1_920,
        end,
        start,
        end,
        48_000,
        continuous[start:end],
        tuple(range(index, index + 8)),
        hops[index:index + 8] if session_id == "cache-session" and epoch == 0 else (),
    )


def _candidates(window: DecisionWindow) -> tuple[CandidateDirection, ...]:
    return tuple(
        CandidateDirection(
            window.session_id,
            window.stream_epoch,
            window.window_id,
            window.decision_sample,
            window.doa_start_sample,
            window.doa_end_sample,
            theta,
            1.0,
            0.8,
        )
        for theta in (20.0, 120.0)
    )


def test_rolling_stft_reuses_overlap_and_is_bit_exact():
    rng = np.random.default_rng(20260818)
    continuous = rng.normal(0.0, 0.02, (8_640, 8)).astype(np.float32)
    hops = tuple(_hop(index) for index in range(9))
    first, second = _window(continuous, hops, 0), _window(continuous, hops, 1)
    settings = StftSettings.from_project(load_config(CONFIG, environ={}))
    cache = RollingStftCache(device=torch.device("cpu"))

    cache.process(first, settings)
    actual = cache.process(second, settings)
    expected = shared_stft(second, settings, device=torch.device("cpu"))

    assert torch.equal(actual, expected)
    snapshot = cache.snapshot()
    assert (snapshot.reused_frames, snapshot.recomputed_frames) == (
        settings.frame_count - 4, 4,
    )
    assert snapshot.temporal_hops == settings.window_hops
    assert snapshot.max_temporal_hops == 50


@pytest.mark.parametrize("hop_gap", (1, 2, 4, 7))
def test_rolling_l3_matches_a_fresh_full_recalculation(hop_gap: int):
    rng = np.random.default_rng(991)
    total_hops = 8 + hop_gap
    continuous = rng.normal(0.0, 0.02, (total_hops * 960, 8)).astype(np.float32)
    hops = tuple(_hop(index) for index in range(total_hops))
    first = _window(continuous, hops, 0)
    second = _window(continuous, hops, hop_gap)
    config = load_config(CONFIG, environ={})
    settings = StftSettings.from_project(config)
    geometry = physical_6plus1_geometry()

    rolling = Layer3Processor(config, device="cpu")
    rolling.process(first, _candidates(first), geometry)
    actual = rolling.process(second, _candidates(second), geometry)
    expected = Layer3Processor(config, device="cpu").process(second, _candidates(second), geometry)

    for cached, full in zip(actual.enhanced_audio, expected.enhanced_audio):
        np.testing.assert_allclose(cached.enhanced_audio, full.enhanced_audio, rtol=2e-4, atol=2e-5)
    snapshot = rolling.cache_snapshot()
    expected = (
        (0, settings.frame_count)
        if hop_gap >= settings.window_hops
        else (settings.frame_count - 2 - 2 * hop_gap, 2 + 2 * hop_gap)
    )
    assert (snapshot.stft_reused_frames, snapshot.stft_recomputed_frames) == expected
    assert snapshot.covariance_rolled == (hop_gap < settings.window_hops)
    assert snapshot.stft_temporal_hops == snapshot.imcra_temporal_hops == settings.window_hops
    assert snapshot.max_temporal_hops == 50


@pytest.mark.parametrize("hop_gap", (2, 4, 7))
def test_rolling_stft_reuses_absolute_sample_overlap(hop_gap: int):
    rng = np.random.default_rng(12)
    total_hops = 8 + hop_gap
    continuous = rng.normal(0.0, 0.02, (total_hops * 960, 8)).astype(np.float32)
    hops = tuple(_hop(index) for index in range(total_hops))
    settings = StftSettings.from_project(load_config(CONFIG, environ={}))
    cache = RollingStftCache(device=torch.device("cpu"))

    cache.process(_window(continuous, hops, 0), settings)
    current = _window(continuous, hops, hop_gap)
    actual = cache.process(current, settings)
    expected = shared_stft(current, settings, device=torch.device("cpu"))

    assert torch.equal(actual, expected)
    expected = (
        (0, settings.frame_count)
        if hop_gap >= settings.window_hops
        else (settings.frame_count - 2 - 2 * hop_gap, 2 + 2 * hop_gap)
    )
    assert (cache.snapshot().reused_frames, cache.snapshot().recomputed_frames) == expected


@pytest.mark.parametrize("hop_gap", (2, 4, 7))
def test_rolling_noise_statistics_match_fresh_recalculation_after_gap(hop_gap: int):
    rng = np.random.default_rng(212)
    total_hops = 8 + hop_gap
    continuous = rng.normal(0.0, 0.02, (total_hops * 960, 8)).astype(np.float32)
    hops = tuple(_hop(index) for index in range(total_hops))
    first = _window(continuous, hops, 0)
    current = _window(continuous, hops, hop_gap)
    project = load_config(CONFIG, environ={})
    settings = StftSettings.from_project(project)
    config = SpatialSeparationConfig.from_project(project)
    device = torch.device("cpu")
    frequencies = torch.fft.rfftfreq(settings.n_fft, 1.0 / 48_000, device=device)
    first_spectrum = shared_stft(first, settings, device=device).permute(1, 0, 2).contiguous()
    current_spectrum = shared_stft(current, settings, device=device).permute(1, 0, 2).contiguous()

    rolling = RollingNoiseStatisticsCache()
    rolling.estimate_window(
        first, first_spectrum, frequencies, config, settings, allow_rolling=True,
    )
    actual, actual_version = rolling.estimate_window(
        current, current_spectrum, frequencies, config, settings, allow_rolling=True,
    )
    expected, expected_version = RollingNoiseStatisticsCache().estimate_window(
        current, current_spectrum, frequencies, config, settings, allow_rolling=True,
    )

    assert rolling.snapshot().rolled == (hop_gap < settings.window_hops)
    assert actual_version == expected_version
    torch.testing.assert_close(actual.covariance_fcc, expected.covariance_fcc, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual.noise_confidence_f, expected.noise_confidence_f)
    torch.testing.assert_close(actual.frequency_gain_f, expected.frequency_gain_f)


@pytest.mark.parametrize("hop_gap", (2, 7))
def test_explicit_noise_context_rolls_across_multi_hop_gap(hop_gap: int):
    rng = np.random.default_rng(313)
    total_hops = 8 + hop_gap
    continuous = rng.normal(0.0, 0.02, (total_hops * 960, 8)).astype(np.float32)
    hops = tuple(_hop(index) for index in range(total_hops))
    first = _window(continuous, hops, 0)
    current = _window(continuous, hops, hop_gap)
    project = load_config(CONFIG, environ={})
    settings = StftSettings.from_project(project)
    config = SpatialSeparationConfig.from_project(project)
    device = torch.device("cpu")
    frequencies = torch.fft.rfftfreq(settings.n_fft, 1.0 / 48_000, device=device)
    first_spectrum = shared_stft(first, settings, device=device).permute(1, 0, 2).contiguous()
    current_spectrum = shared_stft(current, settings, device=device).permute(1, 0, 2).contiguous()
    first_context = BeamformerNoiseContext.from_window(first, settings)
    current_context = BeamformerNoiseContext.from_window(current, settings)

    rolling = RollingNoiseStatisticsCache()
    rolling.estimate(
        first_context, first_spectrum, frequencies, config, settings, allow_rolling=True,
    )
    actual = rolling.estimate(
        current_context, current_spectrum, frequencies, config, settings, allow_rolling=True,
    )
    expected = RollingNoiseStatisticsCache().estimate(
        current_context, current_spectrum, frequencies, config, settings, allow_rolling=True,
    )

    assert rolling.snapshot().rolled == (hop_gap < settings.window_hops)
    torch.testing.assert_close(actual.covariance_fcc, expected.covariance_fcc, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual.noise_confidence_f, expected.noise_confidence_f)
    torch.testing.assert_close(actual.frequency_gain_f, expected.frequency_gain_f)


def test_no_overlap_or_stream_identity_change_forces_complete_rebuild():
    rng = np.random.default_rng(121)
    continuous = rng.normal(0.0, 0.02, (16 * 960, 8)).astype(np.float32)
    hops = tuple(_hop(index) for index in range(8))
    settings = StftSettings.from_project(load_config(CONFIG, environ={}))
    cache = RollingStftCache(device=torch.device("cpu"))

    cache.process(_window(continuous, hops, 0), settings)
    cache.process(_window(continuous, hops, 8), settings)
    assert (cache.snapshot().reused_frames, cache.snapshot().recomputed_frames) == (
        0, settings.frame_count,
    )

    changed_stream = _window(continuous, hops, 8, session_id="new-session", epoch=1)
    cache.process(changed_stream, settings)
    assert (cache.snapshot().reused_frames, cache.snapshot().recomputed_frames) == (
        0, settings.frame_count,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cuda_rolling_caches_reuse_a_multi_hop_gap():
    hop_gap = 7
    rng = np.random.default_rng(717)
    continuous = rng.normal(0.0, 0.02, ((8 + hop_gap) * 960, 8)).astype(np.float32)
    hops = tuple(_hop(index) for index in range(8 + hop_gap))
    project = load_config(CONFIG, environ={})
    settings = StftSettings.from_project(project)
    config = SpatialSeparationConfig.from_project(project)
    device = torch.device("cuda")
    cache = RollingStftCache(device=device)

    first = _window(continuous, hops, 0)
    first_spectrum = cache.process(first, settings).permute(1, 0, 2).contiguous()
    current = _window(continuous, hops, hop_gap)
    actual = cache.process(current, settings)
    expected = shared_stft(current, settings, device=device)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert (cache.snapshot().reused_frames, cache.snapshot().recomputed_frames) == (
        0, settings.frame_count,
    )

    frequencies = torch.fft.rfftfreq(settings.n_fft, 1.0 / 48_000, device=device)
    noise_cache = RollingNoiseStatisticsCache()
    noise_cache.estimate_window(
        first, first_spectrum, frequencies, config, settings, allow_rolling=True,
    )
    actual_noise, _ = noise_cache.estimate_window(
        current,
        actual.permute(1, 0, 2).contiguous(),
        frequencies,
        config,
        settings,
        allow_rolling=True,
    )
    expected_noise, _ = RollingNoiseStatisticsCache().estimate_window(
        current,
        expected.permute(1, 0, 2).contiguous(),
        frequencies,
        config,
        settings,
        allow_rolling=True,
    )
    assert not noise_cache.snapshot().rolled
    torch.testing.assert_close(
        actual_noise.covariance_fcc,
        expected_noise.covariance_fcc,
        rtol=2e-5,
        atol=2e-6,
    )


def test_temporal_and_angle_caches_have_hard_bounded_capacity():
    rng = np.random.default_rng(111)
    continuous = rng.normal(0.0, 0.02, (7_680, 8)).astype(np.float32)
    hops = tuple(_hop(index) for index in range(8))
    window = _window(continuous, hops, 0)
    config = load_config(CONFIG, environ={})
    processor = Layer3Processor(config, device="cpu")
    geometry = physical_6plus1_geometry()

    for index in range(24):
        first = float(index)
        candidates = tuple(
            CandidateDirection(
                window.session_id,
                window.stream_epoch,
                window.window_id,
                window.decision_sample,
                window.doa_start_sample,
                window.doa_end_sample,
                theta,
                1.0,
                0.8,
            )
            for theta in (first, first + 90.0)
        )
        processor.process(window, candidates, geometry)

    snapshot = processor.cache_snapshot()
    assert snapshot.max_temporal_hops == 50
    assert snapshot.stft_temporal_hops <= config.downstream_audio_window.decision_hops
    assert snapshot.imcra_temporal_hops <= config.downstream_audio_window.decision_hops
    assert snapshot.steering_entries <= 16
    assert snapshot.p_entries <= 16
    assert snapshot.persistent_tensor_bytes < 8 * 1024 * 1024


def test_candidate_independent_prepare_matches_compatible_process_api():
    rng = np.random.default_rng(20260819)
    samples = rng.normal(0.0, 0.02, (7_680, 8)).astype(np.float32)
    hops = tuple(_hop(index) for index in range(8))
    window = _window(samples, hops, 0)
    processor = Layer3Processor(load_config(CONFIG, environ={}), device="cpu")
    geometry = physical_6plus1_geometry()

    prepared = processor.prepare(window)
    assert isinstance(prepared, PreparedL3Context)
    assert prepared.window_key == (
        window.session_id, window.stream_epoch, window.window_id, window.decision_sample,
    )
    assert prepared.spectrum_fct.shape == (
        513, 7, processor.window_spec.stft_frames,
    )
    assert not prepared.spectrum_fct.requires_grad
    with pytest.raises(FrozenInstanceError):
        prepared.window_id = 99  # type: ignore[misc]

    staged = processor.process_prepared(prepared, _candidates(window), geometry)
    compatible = processor.process(window, _candidates(window), geometry)
    for actual, expected in zip(staged.enhanced_audio, compatible.enhanced_audio):
        np.testing.assert_array_equal(actual.enhanced_audio, expected.enhanced_audio)


def test_prepared_context_cache_is_bounded_and_clearable():
    rng = np.random.default_rng(771)
    continuous = rng.normal(0.0, 0.02, (9_600, 8)).astype(np.float32)
    hops = tuple(_hop(index) for index in range(10))
    processor = Layer3Processor(load_config(CONFIG, environ={}), device="cpu")

    for index in range(3):
        processor.prepare(_window(continuous, hops, index))
    snapshot = processor.cache_snapshot()
    assert snapshot.prepared_entries == snapshot.prepared_entry_limit == 2
    assert snapshot.prepared_tensor_bytes > 0
    assert snapshot.persistent_tensor_bytes < 8 * 1024 * 1024

    processor.clear_cache()
    cleared = processor.cache_snapshot()
    assert cleared.prepared_entries == 0
    assert cleared.prepared_tensor_bytes == 0
    assert cleared.stft_temporal_hops == cleared.imcra_temporal_hops == 0
    assert cleared.steering_entries == cleared.p_entries == 0


def test_two_candidates_use_one_batched_inverse_stft(monkeypatch: pytest.MonkeyPatch):
    import layer3_direction_signal.engine as engine_module

    rng = np.random.default_rng(181)
    samples = rng.normal(0.0, 0.02, (7_680, 8)).astype(np.float32)
    hops = tuple(_hop(index) for index in range(8))
    window = _window(samples, hops, 0)
    processor = Layer3Processor(load_config(CONFIG, environ={}), device="cpu")
    calls: list[tuple[int, ...]] = []
    original = engine_module.inverse_stft

    def recording_inverse(spectrum, settings, *, length=7_680):
        calls.append(tuple(spectrum.shape))
        return original(spectrum, settings, length=length)

    monkeypatch.setattr(engine_module, "inverse_stft", recording_inverse)
    result = processor.process(window, _candidates(window), physical_6plus1_geometry())

    assert len(result.enhanced_audio) == 2
    assert calls == [(2, 513, processor.window_spec.stft_frames)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_prepared_context_accepts_unindexed_cuda_device_alias():
    rng = np.random.default_rng(814)
    samples = rng.normal(0.0, 0.02, (7_680, 8)).astype(np.float32)
    hops = tuple(_hop(index) for index in range(8))
    window = _window(samples, hops, 0)
    processor = Layer3Processor(load_config(CONFIG, environ={}), device="cuda")

    prepared = processor.prepare(window)
    result = processor.process_prepared(
        prepared, (_candidates(window)[0],), physical_6plus1_geometry(),
    )

    assert prepared.spectrum_fct.device.type == "cuda"
    assert len(result.enhanced_audio) == 1
