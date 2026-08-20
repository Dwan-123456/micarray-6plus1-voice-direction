from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from common.config import load_config
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


def test_loaded_mvdr_baseline_is_distortionless_and_uses_no_cross_direction_null():
    steering = _steering_for_correlations((0.2, 0.5, 0.8))
    covariance = torch.eye(7, dtype=torch.complex64).expand(3, 7, 7).clone()
    result = loaded_mvdr_weights(
        covariance, steering, torch.full((3,), 2_000.0), torch.ones(3), _config(),
    )

    assert result.loaded_mvdr_bins == 3
    assert result.fallback_bins == (0, 0)
    target_response = torch.einsum("mfc,mfc->mf", result.weights_mfc.conj(), steering)
    assert torch.allclose(target_response, torch.ones_like(target_response), atol=2e-3)
    cross_response = torch.einsum(
        "fc,fc->f", result.weights_mfc[0].conj(), steering[1],
    ).abs()
    assert torch.all(cross_response > 0.05)
