from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
import torch

from common.config import load_config
from common.data_types import DecisionWindow, ImcraHopSnapshot, TrackedDirection
from common.geometry import physical_6plus1_geometry
from common.window_key import WindowKey
from layer3_direction_signal import Layer3Processor, PreparedL3Context
from layer3_direction_signal.configuration import StftSettings
from layer3_direction_signal.shared_stft import RollingStftCache, shared_stft


CONFIG = Path(__file__).parents[1] / "config/config.yaml"


def _hop(index: int) -> ImcraHopSnapshot:
    frequencies = np.fft.rfftfreq(2048, 1 / 48_000).astype(np.float32)
    frequencies = frequencies[(frequencies >= 80.0) & (frequencies <= 8_000.0)]
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
        "cohen_imcra_2003_l1_v1",
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
    end = start + 15_360
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
        tuple(range(index, index + 16)),
        hops[index:index + 16] if session_id == "cache-session" and epoch == 0 else (),
    )


def _candidates(window: DecisionWindow) -> tuple[TrackedDirection, ...]:
    return tuple(
        TrackedDirection(
            window.session_id,
            window.stream_epoch,
            window.window_id,
            window.decision_sample,
            window.doa_start_sample,
            window.doa_end_sample,
            index + 1,
            index + 1,
            theta,
            theta,
            1.0,
            0.8,
            "confirmed",
            True,
            False,
            window.context_start_sample,
            window.decision_sample,
            0,
            False,
        )
        for index, theta in enumerate((20.0, 120.0))
    )


def test_rolling_stft_reuses_29_frames_and_is_bit_exact():
    rng = np.random.default_rng(20260818)
    continuous = rng.normal(0.0, 0.02, (16_320, 8)).astype(np.float32)
    hops = tuple(_hop(index) for index in range(17))
    first, second = _window(continuous, hops, 0), _window(continuous, hops, 1)
    settings = StftSettings.from_project(load_config(CONFIG, environ={}))
    cache = RollingStftCache(device=torch.device("cpu"))

    cache.process(first, settings)
    actual = cache.process(second, settings)
    expected = shared_stft(second, settings, device=torch.device("cpu"))

    assert torch.equal(actual, expected)
    snapshot = cache.snapshot()
    assert (snapshot.reused_frames, snapshot.recomputed_frames) == (29, 4)
    assert snapshot.temporal_hops == 16
    assert snapshot.max_temporal_hops == 50


def test_rolling_l3_matches_a_fresh_full_recalculation():
    rng = np.random.default_rng(991)
    continuous = rng.normal(0.0, 0.02, (16_320, 8)).astype(np.float32)
    hops = tuple(_hop(index) for index in range(17))
    first, second = _window(continuous, hops, 0), _window(continuous, hops, 1)
    config = load_config(CONFIG, environ={})
    geometry = physical_6plus1_geometry()

    rolling = Layer3Processor(config, device="cpu")
    rolling.process(first, _candidates(first), geometry)
    actual = rolling.process(second, _candidates(second), geometry)
    expected = Layer3Processor(config, device="cpu").process(second, _candidates(second), geometry)

    for cached, full in zip(actual.enhanced_audio, expected.enhanced_audio):
        np.testing.assert_allclose(cached.enhanced_audio, full.enhanced_audio, rtol=2e-4, atol=2e-5)
    snapshot = rolling.cache_snapshot()
    assert snapshot.stft_reused_frames == 29
    assert snapshot.stft_recomputed_frames == 4
    assert snapshot.covariance_rolled
    assert snapshot.stft_temporal_hops == snapshot.imcra_temporal_hops == 16
    assert snapshot.max_temporal_hops == 50


def test_stream_identity_and_time_discontinuity_force_complete_rebuild():
    rng = np.random.default_rng(12)
    continuous = rng.normal(0.0, 0.02, (17_280, 8)).astype(np.float32)
    hops = tuple(_hop(index) for index in range(18))
    settings = StftSettings.from_project(load_config(CONFIG, environ={}))
    cache = RollingStftCache(device=torch.device("cpu"))

    cache.process(_window(continuous, hops, 0), settings)
    cache.process(_window(continuous, hops, 2), settings)
    assert (cache.snapshot().reused_frames, cache.snapshot().recomputed_frames) == (0, 33)

    changed_stream = _window(continuous, hops, 2, session_id="new-session", epoch=1)
    cache.process(changed_stream, settings)
    assert (cache.snapshot().reused_frames, cache.snapshot().recomputed_frames) == (0, 33)


def test_temporal_and_angle_caches_have_hard_bounded_capacity():
    rng = np.random.default_rng(111)
    continuous = rng.normal(0.0, 0.02, (15_360, 8)).astype(np.float32)
    hops = tuple(_hop(index) for index in range(16))
    window = _window(continuous, hops, 0)
    config = load_config(CONFIG, environ={})
    processor = Layer3Processor(config, device="cpu")
    geometry = physical_6plus1_geometry()

    for index in range(24):
        first = float(index)
        candidates = tuple(
            TrackedDirection(
                window.session_id,
                window.stream_epoch,
                window.window_id,
                window.decision_sample,
                window.doa_start_sample,
                window.doa_end_sample,
                candidate_index + 1,
                candidate_index + 1,
                theta,
                theta,
                1.0,
                0.8,
                "confirmed",
                True,
                False,
                window.context_start_sample,
                window.decision_sample,
                0,
                False,
            )
            for candidate_index, theta in enumerate((first, first + 90.0))
        )
        processor.process(window, candidates, geometry)

    snapshot = processor.cache_snapshot()
    assert snapshot.max_temporal_hops == 50
    assert snapshot.stft_temporal_hops <= 16
    assert snapshot.imcra_temporal_hops <= 16
    assert snapshot.steering_entries <= 16
    assert snapshot.p_entries <= 16
    assert snapshot.persistent_tensor_bytes < 8 * 1024 * 1024


def test_candidate_independent_prepare_matches_compatible_process_api():
    rng = np.random.default_rng(20260819)
    samples = rng.normal(0.0, 0.02, (15_360, 8)).astype(np.float32)
    hops = tuple(_hop(index) for index in range(16))
    window = _window(samples, hops, 0)
    processor = Layer3Processor(load_config(CONFIG, environ={}), device="cpu")
    geometry = physical_6plus1_geometry()

    prepared = processor.prepare(window)
    assert isinstance(prepared, PreparedL3Context)
    assert prepared.window_key == WindowKey.from_window(window)
    assert prepared.spectrum_fct.shape == (513, 7, 33)
    assert not prepared.spectrum_fct.requires_grad
    with pytest.raises(FrozenInstanceError):
        prepared.window_id = 99  # type: ignore[misc]

    staged = processor.process_prepared(prepared, _candidates(window), geometry)
    compatible = processor.process(window, _candidates(window), geometry)
    for actual, expected in zip(staged.enhanced_audio, compatible.enhanced_audio):
        np.testing.assert_array_equal(actual.enhanced_audio, expected.enhanced_audio)


def test_prepared_context_cache_is_bounded_and_clearable():
    rng = np.random.default_rng(771)
    continuous = rng.normal(0.0, 0.02, (17_280, 8)).astype(np.float32)
    hops = tuple(_hop(index) for index in range(18))
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
    samples = rng.normal(0.0, 0.02, (15_360, 8)).astype(np.float32)
    hops = tuple(_hop(index) for index in range(16))
    window = _window(samples, hops, 0)
    processor = Layer3Processor(load_config(CONFIG, environ={}), device="cpu")
    calls: list[tuple[int, ...]] = []
    original = engine_module.inverse_stft

    def recording_inverse(spectrum, settings, *, length=15_360):
        calls.append(tuple(spectrum.shape))
        return original(spectrum, settings, length=length)

    monkeypatch.setattr(engine_module, "inverse_stft", recording_inverse)
    result = processor.process(window, _candidates(window), physical_6plus1_geometry())

    assert len(result.enhanced_audio) == 2
    assert calls == [(2, 513, 33)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_prepared_context_accepts_unindexed_cuda_device_alias():
    rng = np.random.default_rng(814)
    samples = rng.normal(0.0, 0.02, (15_360, 8)).astype(np.float32)
    hops = tuple(_hop(index) for index in range(16))
    window = _window(samples, hops, 0)
    processor = Layer3Processor(load_config(CONFIG, environ={}), device="cuda")

    prepared = processor.prepare(window)
    result = processor.process_prepared(
        prepared, (_candidates(window)[0],), physical_6plus1_geometry(),
    )

    assert prepared.spectrum_fct.device.type == "cuda"
    assert len(result.enhanced_audio) == 1
