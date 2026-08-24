from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from common.config import load_config
import layer3_direction_signal.adaptive_separation as adaptive_module
from layer3_direction_signal.adaptive_separation import (
    adaptive_separation_weights,
    loaded_mvdr_weights,
)
from layer3_direction_signal.configuration import SpatialSeparationConfig


CONFIG = Path(__file__).parents[1] / "config/config.yaml"


def _steering_for_correlations(correlations: tuple[float, ...]) -> torch.Tensor:
    channels = torch.arange(7, dtype=torch.float32)
    target = torch.ones(7, dtype=torch.complex64)
    orthogonal = torch.exp(2j * torch.pi * channels / 7).to(torch.complex64)
    competitors = tuple(
        correlation * target + np.sqrt(1.0 - correlation**2) * orthogonal
        for correlation in correlations
    )
    return torch.stack((target.expand(len(correlations), -1), torch.stack(competitors)), dim=0)


def _config() -> SpatialSeparationConfig:
    return SpatialSeparationConfig.from_project(load_config(CONFIG, environ={}))


def _eager_loaded_mvdr_reference(
    covariance: torch.Tensor,
    steering: torch.Tensor,
    frequencies: torch.Tensor,
    confidence: torch.Tensor,
    config: SpatialSeparationConfig,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Pre-optimization fixed-shape retry path used as a numeric oracle."""

    static = adaptive_module.adaptive_static_data(frequencies, config)
    output = adaptive_module.das_weights(steering).clone()
    candidate_count, _, channel_count = steering.shape
    valid = torch.zeros(steering.shape[:2], dtype=torch.bool)
    valid[:, ~static.speech_band_f] = True
    scale = torch.diagonal(covariance, dim1=-2, dim2=-1).real.mean(dim=-1).clamp_min(1e-8)
    uncertainty = 1.0 + (1.0 - confidence) * config.uncertainty_loading_multiplier
    retry = torch.as_tensor(config.loading_retry_factors, dtype=scale.dtype)
    loading = retry[:, None] * uncertainty[None] * static.alias_multiplier_f[None] * scale[None]
    loaded = covariance[None] + loading[..., None, None] * static.identity_cc
    factor, matrix_ok = adaptive_module._factorize(loaded, config.condition_number_limit)
    steering_fcm = steering.permute(1, 2, 0).contiguous()
    solved = torch.cholesky_solve(
        steering_fcm[None].expand(len(retry), -1, -1, -1), factor,
    )
    weights, retry_valid = adaptive_module._mvdr_from_solved(
        solved, steering_fcm, matrix_ok, config.constraint_tolerance,
    )
    retry_valid &= static.speech_band_f[None, None, :]
    solved_valid = retry_valid.any(dim=0)
    first_valid = retry_valid.to(torch.int8).argmax(dim=0)
    selected = torch.gather(
        weights,
        0,
        first_valid[None, ..., None].expand(1, -1, -1, channel_count),
    )[0]
    output = torch.where(solved_valid[..., None], selected, output)
    valid |= solved_valid
    fallback = static.speech_band_f[None] & ~valid
    return output, tuple(int(item) for item in fallback.sum(dim=1).tolist())


def _eager_adaptive_reference(
    covariance: torch.Tensor,
    steering: torch.Tensor,
    frequencies: torch.Tensor,
    confidence: torch.Tensor,
    config: SpatialSeparationConfig,
    spatial_p: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Pre-optimization two-target fixed-shape path used as a numeric oracle."""

    static = adaptive_module.adaptive_static_data(frequencies, config)
    output = adaptive_module.das_weights(steering).clone()
    candidate_count, _, channel_count = steering.shape
    speech_band = static.speech_band_f
    lcmv_mask = speech_band & (spatial_p < config.rho_lcmv_max)
    soft_mask = (
        speech_band
        & (spatial_p >= config.rho_lcmv_max)
        & (spatial_p < config.rho_soft_null_max)
    )
    scale = torch.diagonal(covariance, dim1=-2, dim2=-1).real.mean(dim=-1).clamp_min(1e-8)
    uncertainty = 1.0 + (1.0 - confidence) * config.uncertainty_loading_multiplier
    retry = torch.as_tensor(config.loading_retry_factors, dtype=scale.dtype)
    loading = retry[:, None] * uncertainty[None] * static.alias_multiplier_f[None] * scale[None]
    loaded = covariance[None] + loading[..., None, None] * static.identity_cc
    factor, matrix_ok = adaptive_module._factorize(loaded, config.condition_number_limit)
    steering_fcm = steering.permute(1, 2, 0).contiguous()
    solved = torch.cholesky_solve(
        steering_fcm[None].expand(len(retry), -1, -1, -1), factor,
    )
    mvdr_weights, mvdr_valid = adaptive_module._mvdr_from_solved(
        solved, steering_fcm, matrix_ok, config.constraint_tolerance,
    )
    lcmv_weights, lcmv_valid = adaptive_module._dual_lcmv_from_solved(
        solved,
        steering_fcm,
        matrix_ok,
        config.condition_number_limit,
        config.constraint_tolerance,
    )
    soft_weights, soft_valid = adaptive_module._soft_null_mvdr_batched(
        loaded,
        steering,
        config.soft_null_strength,
        config.condition_number_limit,
        config.constraint_tolerance,
    )
    routed_weights = torch.where(
        lcmv_mask[None, None, :, None],
        lcmv_weights,
        torch.where(soft_mask[None, None, :, None], soft_weights, mvdr_weights),
    )
    routed_valid = torch.where(
        lcmv_mask[None, None, :],
        lcmv_valid,
        torch.where(soft_mask[None, None, :], soft_valid, mvdr_valid),
    )
    routed_valid &= speech_band[None, None]
    valid = routed_valid.any(dim=0)
    first_valid = routed_valid.to(torch.int8).argmax(dim=0)
    selected = torch.gather(
        routed_weights,
        0,
        first_valid[None, ..., None].expand(1, -1, -1, channel_count),
    )[0]
    output = torch.where(valid[..., None], selected, output)
    fallback = speech_band[None] & ~valid
    return output, tuple(int(item) for item in fallback.sum(dim=1).tolist())


def test_rho_routes_bins_to_lcmv_soft_null_and_loaded_mvdr():
    steering = _steering_for_correlations((0.1, 0.5, 0.9))
    covariance = torch.eye(7, dtype=torch.complex64).expand(3, 7, 7).clone()
    result = adaptive_separation_weights(
        covariance, steering, torch.full((3,), 2_000.0), torch.ones(3), _config(),
        spatial_p_f=torch.tensor((0.1, 0.5, 0.9)),
    )

    assert torch.allclose(result.rho_f, torch.tensor((0.1, 0.5, 0.9)), atol=1e-5)
    assert (result.lcmv_bins, result.soft_null_bins, result.loaded_mvdr_bins) == (1, 1, 1)
    assert result.fallback_bins == (0, 0)

    lcmv_response = result.weights_mfc[:, 0].conj() @ steering[:, 0].T
    assert torch.allclose(lcmv_response, torch.eye(2, dtype=torch.complex64), atol=2e-3)
    target_response = torch.einsum("mfc,mfc->mf", result.weights_mfc.conj(), steering)
    assert torch.allclose(target_response, torch.ones_like(target_response), atol=2e-3)


def test_precomputed_spatial_p_overrides_runtime_steering_correlation_for_routing():
    steering = _steering_for_correlations((0.9, 0.9, 0.9))
    covariance = torch.eye(7, dtype=torch.complex64).expand(3, 7, 7).clone()
    result = adaptive_separation_weights(
        covariance,
        steering,
        torch.full((3,), 2_000.0),
        torch.ones(3),
        _config(),
        spatial_p_f=torch.tensor((0.1, 0.5, 0.9)),
    )

    assert torch.equal(result.rho_f, torch.tensor((0.1, 0.5, 0.9)))
    assert (result.lcmv_bins, result.soft_null_bins, result.loaded_mvdr_bins) == (1, 1, 1)


def test_two_candidate_solver_rejects_missing_precomputed_spatial_p():
    steering = _steering_for_correlations((0.1, 0.5, 0.9))
    covariance = torch.eye(7, dtype=torch.complex64).expand(3, 7, 7).clone()
    with pytest.raises(ValueError, match="requires precomputed spatial p"):
        adaptive_separation_weights(
            covariance, steering, torch.full((3,), 2_000.0), torch.ones(3), _config(),
        )


def test_numerically_rejected_bins_fall_back_to_das():
    steering = _steering_for_correlations((0.1, 0.5, 0.9))
    covariance = torch.eye(7, dtype=torch.complex64).expand(3, 7, 7).clone()
    strict = replace(_config(), condition_number_limit=0.5)
    result = adaptive_separation_weights(
        covariance, steering, torch.full((3,), 2_000.0), torch.ones(3), strict,
        spatial_p_f=torch.tensor((0.1, 0.5, 0.9)),
    )
    assert result.fallback_bins == (3, 3)
    expected = steering / 7.0
    assert torch.allclose(result.weights_mfc, expected)


def test_all_three_solvers_depend_on_the_supplied_noise_covariance():
    steering = _steering_for_correlations((0.1, 0.5, 0.9))
    balanced = torch.eye(7, dtype=torch.complex64).expand(3, 7, 7).clone()
    asymmetric = balanced.clone()
    asymmetric[:, 0, 0] = 100.0
    args = (steering, torch.full((3,), 2_000.0), torch.ones(3), _config())
    spatial_p = torch.tensor((0.1, 0.5, 0.9))

    first = adaptive_separation_weights(balanced, *args, spatial_p_f=spatial_p)
    second = adaptive_separation_weights(asymmetric, *args, spatial_p_f=spatial_p)

    assert first.fallback_bins == second.fallback_bins == (0, 0)
    for frequency_index in range(3):
        assert not torch.allclose(
            first.weights_mfc[:, frequency_index], second.weights_mfc[:, frequency_index],
        )


def test_loaded_mvdr_baseline_uses_current_batched_cholesky_solver():
    steering = _steering_for_correlations((0.2, 0.5, 0.8))
    covariance = torch.eye(7, dtype=torch.complex64).expand(3, 7, 7).clone()
    result = loaded_mvdr_weights(
        covariance, steering, torch.full((3,), 2_000.0), torch.ones(3), _config(),
    )

    assert result.loaded_mvdr_bins == 3
    assert result.fallback_bins == (0, 0)
    target_response = torch.einsum("mfc,mfc->mf", result.weights_mfc.conj(), steering)
    assert torch.allclose(target_response, torch.ones_like(target_response), atol=2e-3)
    assert torch.isfinite(result.weights_mfc).all()


@pytest.mark.parametrize("solver_name", ("loaded", "adaptive"))
def test_active_band_lazy_retry_only_revisits_unresolved_bins(
    solver_name: str, monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[int, ...]] = []
    original = torch.linalg.cholesky_ex

    def recording_cholesky(matrix: torch.Tensor, *args, **kwargs):
        calls.append(tuple(matrix.shape))
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky_ex", recording_cholesky)
    frequencies = torch.tensor((0.0, 2_000.0, 3_000.0, 10_000.0))
    covariance = torch.eye(7, dtype=torch.complex64).expand(4, 7, 7).clone()
    covariance[2] = torch.diag(
        torch.tensor((1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    ).to(torch.complex64)
    steering = torch.ones((1, 4, 7), dtype=torch.complex64)
    config = replace(_config(), condition_number_limit=100.0)
    args = (covariance, steering, frequencies, torch.ones(4), config)

    result = (
        loaded_mvdr_weights(*args)
        if solver_name == "loaded"
        else adaptive_separation_weights(*args)
    )

    # Only two passband bins are factored.  The well-conditioned bin succeeds
    # immediately; the singular bin alone proceeds through retries two/three.
    assert calls == [(2, 7, 7), (1, 7, 7), (1, 7, 7)]
    assert result.fallback_bins == (0,)
    expected_das = steering / 7.0
    assert torch.equal(result.weights_mfc[:, (0, 3)], expected_das[:, (0, 3)])


def test_lazy_loaded_mvdr_is_numerically_equal_to_eager_fixed_retries():
    frequencies = torch.tensor((0.0, 2_000.0, 2_500.0, 3_000.0, 10_000.0))
    covariance = torch.eye(7, dtype=torch.complex64).expand(5, 7, 7).clone()
    covariance[2] = torch.diag(
        torch.tensor((1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    ).to(torch.complex64)
    steering = _steering_for_correlations((0.2, 0.3, 0.4, 0.5, 0.6))
    confidence = torch.ones(5)
    config = replace(_config(), condition_number_limit=100.0)

    expected, expected_fallback = _eager_loaded_mvdr_reference(
        covariance, steering, frequencies, confidence, config,
    )
    actual = loaded_mvdr_weights(
        covariance, steering, frequencies, confidence, config,
    )

    assert actual.fallback_bins == expected_fallback
    assert torch.allclose(actual.weights_mfc, expected, atol=2e-5, rtol=2e-5)


def test_lazy_adaptive_is_numerically_equal_to_eager_fixed_retries():
    frequencies = torch.tensor((0.0, 2_000.0, 2_500.0, 3_000.0, 10_000.0))
    covariance = torch.eye(7, dtype=torch.complex64).expand(5, 7, 7).clone()
    covariance[2] = torch.diag(
        torch.tensor((1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    ).to(torch.complex64)
    steering = _steering_for_correlations((0.9, 0.1, 0.5, 0.9, 0.9))
    confidence = torch.ones(5)
    spatial_p = torch.tensor((0.9, 0.1, 0.5, 0.9, 0.9))
    config = replace(_config(), condition_number_limit=100.0)

    expected, expected_fallback = _eager_adaptive_reference(
        covariance, steering, frequencies, confidence, config, spatial_p,
    )
    actual = adaptive_separation_weights(
        covariance,
        steering,
        frequencies,
        confidence,
        config,
        spatial_p_f=spatial_p,
    )

    assert actual.fallback_bins == expected_fallback
    assert torch.allclose(actual.weights_mfc, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("candidate_count", (1, 3))
def test_loaded_mvdr_cholesky_path_matches_direct_solve(candidate_count: int):
    generator = torch.Generator().manual_seed(20260819 + candidate_count)
    frequency_count = 5
    random_matrix = (
        torch.randn(frequency_count, 7, 7, generator=generator)
        + 1j * torch.randn(frequency_count, 7, 7, generator=generator)
    ).to(torch.complex64)
    covariance = random_matrix @ random_matrix.mH / 7.0 + 0.2 * torch.eye(7)
    steering = (
        torch.randn(candidate_count, frequency_count, 7, generator=generator)
        + 1j * torch.randn(candidate_count, frequency_count, 7, generator=generator)
    ).to(torch.complex64)
    steering *= np.sqrt(7.0) / torch.linalg.vector_norm(steering, dim=-1, keepdim=True)
    config = replace(_config(), loading_retry_factors=(0.01,))
    frequencies = torch.full((frequency_count,), 2_000.0)

    result = adaptive_separation_weights(
        covariance, steering, frequencies, torch.ones(frequency_count), config,
    )

    scale = torch.diagonal(covariance, dim1=-2, dim2=-1).real.mean(dim=-1)
    loaded = covariance + 0.01 * scale[:, None, None] * torch.eye(7)
    constraints = steering.permute(1, 2, 0).contiguous()
    solved = torch.linalg.solve(loaded, constraints)
    denominator = torch.einsum("fcm,fcm->fm", constraints.conj(), solved)
    expected = (solved / denominator[:, None, :]).permute(2, 0, 1)

    assert result.fallback_bins == (0,) * candidate_count
    assert torch.allclose(result.weights_mfc, expected, atol=2e-5, rtol=2e-5)


def test_two_candidate_cholesky_paths_match_direct_lcmv_soft_null_and_mvdr_solves():
    steering = _steering_for_correlations((0.1, 0.5, 0.9))
    covariance = torch.eye(7, dtype=torch.complex64).expand(3, 7, 7).clone()
    covariance[:, 0, 0] = 4.0
    covariance[:, 1, 1] = 2.0
    config = replace(_config(), loading_retry_factors=(0.01,))
    result = adaptive_separation_weights(
        covariance,
        steering,
        torch.full((3,), 2_000.0),
        torch.ones(3),
        config,
        spatial_p_f=torch.tensor((0.1, 0.5, 0.9)),
    )

    covariance_scale = torch.diagonal(covariance, dim1=-2, dim2=-1).real.mean(dim=-1)
    loaded = covariance + 0.01 * covariance_scale[:, None, None] * torch.eye(7)
    expected = torch.empty_like(steering)

    constraints = steering[:, 0].T.contiguous()
    solved = torch.linalg.solve(loaded[0], constraints)
    gram = constraints.mH @ solved
    expected[:, 0] = (solved @ torch.linalg.solve(gram, torch.eye(2, dtype=torch.complex64))).T

    loaded_scale = torch.diagonal(loaded, dim1=-2, dim2=-1).real.mean(dim=-1)
    for target, interferer in ((0, 1), (1, 0)):
        interference = steering[interferer, 1]
        penalty = interference[:, None] * interference.conj()[None, :]
        soft_covariance = loaded[1] + config.soft_null_strength * loaded_scale[1] * penalty
        soft_solved = torch.linalg.solve(soft_covariance, steering[target, 1])
        expected[target, 1] = soft_solved / torch.vdot(steering[target, 1], soft_solved)

    mvdr_constraints = steering[:, 2].T.contiguous()
    mvdr_solved = torch.linalg.solve(loaded[2], mvdr_constraints)
    mvdr_denominator = torch.einsum("cm,cm->m", mvdr_constraints.conj(), mvdr_solved)
    expected[:, 2] = (mvdr_solved / mvdr_denominator[None, :]).T

    assert result.fallback_bins == (0, 0)
    assert torch.allclose(result.weights_mfc, expected, atol=2e-5, rtol=2e-5)


def test_fixed_shape_retries_select_the_first_numerically_valid_loading():
    covariance = torch.diag(torch.tensor((1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))).to(torch.complex64)[None]
    steering = torch.ones((1, 1, 7), dtype=torch.complex64)
    config = replace(_config(), condition_number_limit=100.0)

    result = adaptive_separation_weights(
        covariance, steering, torch.tensor((2_000.0,)), torch.ones(1), config,
    )

    scale = torch.diagonal(covariance, dim1=-2, dim2=-1).real.mean(dim=-1)
    loaded = covariance + 0.1 * scale[:, None, None] * torch.eye(7)
    solved = torch.linalg.solve(loaded, steering.permute(1, 2, 0))
    expected = (solved / torch.einsum("fcm,fcm->fm", steering.permute(1, 2, 0).conj(), solved))
    expected = expected.permute(2, 0, 1)

    assert result.fallback_bins == (0,)
    assert torch.allclose(result.weights_mfc, expected, atol=2e-5, rtol=2e-5)


def test_two_candidate_solver_batches_one_lazy_retry_and_soft_null_dimensions(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[int, ...]] = []
    original = torch.linalg.cholesky_ex

    def recording_cholesky(matrix: torch.Tensor, *args, **kwargs):
        calls.append(tuple(matrix.shape))
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky_ex", recording_cholesky)
    steering = _steering_for_correlations((0.1, 0.5, 0.9))
    covariance = torch.eye(7, dtype=torch.complex64).expand(3, 7, 7).clone()
    adaptive_separation_weights(
        covariance,
        steering,
        torch.full((3,), 2_000.0),
        torch.ones(3),
        _config(),
        spatial_p_f=torch.tensor((0.1, 0.5, 0.9)),
    )

    assert calls == [(3, 7, 7), (1, 3, 2, 2), (1, 2, 3, 7, 7)]
