from __future__ import annotations

from dataclasses import dataclass

import torch

from .configuration import SpatialSeparationConfig
from .das import das_weights


@dataclass(frozen=True, slots=True)
class AdaptiveWeightResult:
    weights_mfc: torch.Tensor
    rho_f: torch.Tensor
    lcmv_bins: int
    soft_null_bins: int
    loaded_mvdr_bins: int
    fallback_bins: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LoadedMvdrWeightResult:
    weights_mfc: torch.Tensor
    loaded_mvdr_bins: int
    fallback_bins: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AdaptiveStaticData:
    speech_band_f: torch.Tensor
    identity_cc: torch.Tensor
    alias_multiplier_f: torch.Tensor


def adaptive_static_data(
    frequencies_hz: torch.Tensor,
    config: SpatialSeparationConfig,
    *,
    channel_count: int = 7,
) -> AdaptiveStaticData:
    speech_band = (
        (frequencies_hz >= config.frequency_min_hz)
        & (frequencies_hz <= config.frequency_max_hz)
    )
    identity = torch.eye(channel_count, dtype=torch.complex64, device=frequencies_hz.device)
    alias = torch.where(
        frequencies_hz >= config.alias_guard_hz,
        torch.full_like(frequencies_hz, config.alias_loading_multiplier),
        torch.ones_like(frequencies_hz),
    )
    return AdaptiveStaticData(speech_band, identity, alias)


def _factorize(
    matrix_xcc: torch.Tensor,
    condition_limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Factor a Hermitian positive-definite batch without an SVD.

    Loaded covariance and LCMV Gram matrices are Hermitian positive definite. Their
    spectral condition number is therefore the largest eigenvalue divided by the
    smallest one, so ``eigvalsh`` preserves the previous 2-norm condition guard at
    substantially lower cost than ``linalg.cond``'s general complex SVD.
    """
    try:
        factor, info = torch.linalg.cholesky_ex(matrix_xcc, check_errors=False)
        eigenvalues = torch.linalg.eigvalsh(matrix_xcc)
    except RuntimeError:
        identity = torch.eye(
            matrix_xcc.shape[-1], dtype=matrix_xcc.dtype, device=matrix_xcc.device,
        )
        factor = identity.expand_as(matrix_xcc)
        return factor, torch.zeros(matrix_xcc.shape[:-2], dtype=torch.bool, device=matrix_xcc.device)

    smallest = eigenvalues[..., 0]
    largest = eigenvalues[..., -1]
    valid = (
        info.eq(0)
        & torch.isfinite(eigenvalues).all(dim=-1)
        & (smallest > 0)
        & (largest <= float(condition_limit) * smallest)
    )
    identity = torch.eye(
        matrix_xcc.shape[-1], dtype=matrix_xcc.dtype, device=matrix_xcc.device,
    )
    safe_factor = torch.where(info.eq(0)[..., None, None], factor, identity)
    return safe_factor, valid


def _mvdr_from_solved(
    solved_rfcm: torch.Tensor,
    steering_fcm: torch.Tensor,
    matrix_ok_rf: torch.Tensor,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    denominator = torch.einsum("fcm,rfcm->rfm", steering_fcm.conj(), solved_rfcm)
    safe = torch.where(torch.abs(denominator) > 1e-12, denominator, torch.ones_like(denominator))
    columns = solved_rfcm / safe[:, :, None, :]
    weights = columns.permute(0, 3, 1, 2).contiguous()
    steering_mfc = steering_fcm.permute(2, 0, 1)
    response = torch.einsum("rmfc,mfc->rmf", weights.conj(), steering_mfc)
    valid = (
        matrix_ok_rf[:, None, :]
        & torch.isfinite(weights).all(dim=-1)
        & torch.isfinite(denominator).permute(0, 2, 1)
        & (torch.abs(response - 1.0) <= tolerance)
    )
    return weights, valid


def _dual_lcmv_from_solved(
    solved_rfcm: torch.Tensor,
    constraints_fcm: torch.Tensor,
    covariance_ok_rf: torch.Tensor,
    condition_limit: float,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    gram_rfmm = torch.einsum("fcm,rfcn->rfmn", constraints_fcm.conj(), solved_rfcm)
    factor, gram_ok = _factorize(gram_rfmm, condition_limit)
    identity = torch.eye(2, dtype=gram_rfmm.dtype, device=gram_rfmm.device)
    transform = torch.cholesky_solve(identity.expand_as(gram_rfmm), factor)
    columns = solved_rfcm @ transform
    weights = columns.permute(0, 3, 1, 2).contiguous()
    response = torch.einsum("rfcm,fcn->rfmn", columns.conj(), constraints_fcm)
    response_error = torch.amax(torch.abs(response - identity), dim=(-2, -1))
    per_frequency = (
        covariance_ok_rf
        & gram_ok
        & torch.isfinite(weights).all(dim=(1, 3))
        & (response_error <= tolerance)
    )
    return weights, per_frequency[:, None, :].expand(-1, 2, -1)


def _soft_null_mvdr_batched(
    loaded_rfcc: torch.Tensor,
    steering_mfc: torch.Tensor,
    strength: float,
    condition_limit: float,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    loaded_scale = (
        torch.diagonal(loaded_rfcc, dim1=-2, dim2=-1).real.mean(dim=-1).clamp_min(1e-8)
    )
    interference = steering_mfc.flip(0)
    penalty_mfcc = interference[..., :, None] * interference.conj()[..., None, :]
    covariance_rmfcc = (
        loaded_rfcc[:, None]
        + float(strength) * loaded_scale[:, None, :, None, None] * penalty_mfcc[None]
    )
    factor, matrix_ok = _factorize(covariance_rmfcc, condition_limit)
    right_hand_side = steering_mfc[None, ..., None].expand(len(loaded_rfcc), -1, -1, -1, -1)
    solved = torch.cholesky_solve(right_hand_side, factor)[..., 0]
    denominator = torch.einsum("mfc,rmfc->rmf", steering_mfc.conj(), solved)
    safe = torch.where(torch.abs(denominator) > 1e-12, denominator, torch.ones_like(denominator))
    weights = solved / safe[..., None]
    response = torch.einsum("rmfc,mfc->rmf", weights.conj(), steering_mfc)
    valid = (
        matrix_ok
        & torch.isfinite(weights).all(dim=-1)
        & torch.isfinite(denominator)
        & (torch.abs(response - 1.0) <= tolerance)
    )
    return weights, valid


def adaptive_separation_weights(
    covariance_fcc: torch.Tensor,
    steering_mfc: torch.Tensor,
    frequencies_hz: torch.Tensor,
    noise_confidence_f: torch.Tensor,
    config: SpatialSeparationConfig,
    *,
    spatial_p_f: torch.Tensor | None = None,
    static: AdaptiveStaticData | None = None,
) -> AdaptiveWeightResult:
    candidate_count, frequency_count, channel_count = steering_mfc.shape
    if candidate_count not in {1, 2, 3} or channel_count != 7:
        raise ValueError("adaptive separation requires one, two, or three 7-channel steering vectors")
    das = das_weights(steering_mfc)
    output = das.clone()
    static = static or adaptive_static_data(frequencies_hz, config, channel_count=channel_count)
    speech_band = static.speech_band_f
    if (
        speech_band.shape != (frequency_count,)
        or static.identity_cc.shape != (channel_count, channel_count)
        or static.alias_multiplier_f.shape != (frequency_count,)
    ):
        raise ValueError("adaptive static cache shapes do not match the current solve")
    if candidate_count in {1, 3}:
        rho = torch.ones(frequency_count, dtype=torch.float32, device=steering_mfc.device)
    else:
        if spatial_p_f is None:
            raise ValueError("two-candidate adaptive separation requires precomputed spatial p lookup values")
        rho = spatial_p_f
        if rho.shape != (frequency_count,):
            raise ValueError("spatial p must be finite [frequency] values in [0,1]")
        if bool((~torch.isfinite(rho) | (rho < 0) | (rho > 1)).any()):
            raise ValueError("spatial p must be finite [frequency] values in [0,1]")
    rho = rho.to(device=steering_mfc.device, dtype=torch.float32)
    if candidate_count in {1, 3}:
        lcmv_mask = torch.zeros_like(speech_band)
        soft_mask = torch.zeros_like(speech_band)
        mvdr_mask = speech_band
    else:
        lcmv_mask = speech_band & (rho < config.rho_lcmv_max)
        soft_mask = speech_band & (rho >= config.rho_lcmv_max) & (rho < config.rho_soft_null_max)
        mvdr_mask = speech_band & (rho >= config.rho_soft_null_max)

    eye = static.identity_cc
    scale = torch.diagonal(covariance_fcc, dim1=-2, dim2=-1).real.mean(dim=-1).clamp_min(1e-8)
    uncertainty = 1.0 + (1.0 - noise_confidence_f) * config.uncertainty_loading_multiplier
    loading_multiplier = uncertainty * static.alias_multiplier_f

    retry_factors = torch.as_tensor(
        config.loading_retry_factors, dtype=scale.dtype, device=scale.device,
    )
    loading = retry_factors[:, None] * loading_multiplier[None, :] * scale[None, :]
    loaded_rfcc = covariance_fcc[None] + loading[..., None, None] * eye
    covariance_factor, covariance_ok = _factorize(
        loaded_rfcc, config.condition_number_limit,
    )
    steering_fcm = steering_mfc.permute(1, 2, 0).contiguous()
    solved_rfcm = torch.cholesky_solve(
        steering_fcm[None].expand(len(retry_factors), -1, -1, -1), covariance_factor,
    )
    mvdr_weights, mvdr_valid = _mvdr_from_solved(
        solved_rfcm, steering_fcm, covariance_ok, config.constraint_tolerance,
    )

    if candidate_count == 2:
        lcmv_weights, lcmv_valid = _dual_lcmv_from_solved(
            solved_rfcm,
            steering_fcm,
            covariance_ok,
            config.condition_number_limit,
            config.constraint_tolerance,
        )
        soft_weights, soft_valid = _soft_null_mvdr_batched(
            loaded_rfcc,
            steering_mfc,
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
    else:
        routed_weights = mvdr_weights
        routed_valid = mvdr_valid

    routed_valid &= speech_band[None, None, :]
    valid = routed_valid.any(dim=0)
    first_valid_retry = routed_valid.to(torch.int8).argmax(dim=0)
    selected = torch.gather(
        routed_weights,
        0,
        first_valid_retry[None, ..., None].expand(1, -1, -1, channel_count),
    )[0]
    output = torch.where(valid[..., None], selected, output)
    fallback = speech_band[None, :] & ~valid
    output = torch.where(fallback[..., None], das, output)
    diagnostic_values = torch.cat(
        (
            torch.stack((lcmv_mask.sum(), soft_mask.sum(), mvdr_mask.sum())),
            fallback.sum(dim=1),
            torch.isfinite(output).all().reshape(1),
        )
    ).tolist()
    if not diagnostic_values[-1]:
        output = das
        fallback_count = sum(int(item) for item in diagnostic_values[:3])
        fallback_counts = (fallback_count,) * candidate_count
    else:
        fallback_counts = tuple(int(item) for item in diagnostic_values[3:-1])
    return AdaptiveWeightResult(
        output,
        rho,
        int(diagnostic_values[0]),
        int(diagnostic_values[1]),
        int(diagnostic_values[2]),
        fallback_counts,
    )


def loaded_mvdr_weights(
    covariance_fcc: torch.Tensor,
    steering_mfc: torch.Tensor,
    frequencies_hz: torch.Tensor,
    noise_confidence_f: torch.Tensor,
    config: SpatialSeparationConfig,
    *,
    static: AdaptiveStaticData | None = None,
) -> LoadedMvdrWeightResult:
    """Independent diagonal-loaded MVDR for every supplied target direction."""
    candidate_count, frequency_count, channel_count = steering_mfc.shape
    if candidate_count not in {1, 2, 3} or channel_count != 7:
        raise ValueError("loaded MVDR baseline requires one, two, or three 7-channel directions")
    static = static or adaptive_static_data(frequencies_hz, config, channel_count=channel_count)
    if (
        covariance_fcc.shape != (frequency_count, channel_count, channel_count)
        or noise_confidence_f.shape != (frequency_count,)
        or static.speech_band_f.shape != (frequency_count,)
    ):
        raise ValueError("loaded MVDR baseline inputs do not share one frequency axis")

    das = das_weights(steering_mfc)
    output = das.clone()
    valid = torch.zeros(
        (candidate_count, frequency_count), dtype=torch.bool, device=steering_mfc.device,
    )
    valid[:, ~static.speech_band_f] = True
    scale = torch.diagonal(covariance_fcc, dim1=-2, dim2=-1).real.mean(dim=-1).clamp_min(1e-8)
    uncertainty = 1.0 + (1.0 - noise_confidence_f) * config.uncertainty_loading_multiplier
    loading_multiplier = uncertainty * static.alias_multiplier_f

    for factor in config.loading_retry_factors:
        pending = static.speech_band_f[None, :] & ~valid
        if not bool(pending.any()):
            break
        loading = float(factor) * loading_multiplier * scale
        loaded = covariance_fcc + loading[:, None, None] * static.identity_cc[None]
        indices = torch.nonzero(pending.any(dim=0), as_tuple=False).flatten()
        weights, ok = _mvdr(
            loaded[indices], steering_mfc[:, indices],
            config.condition_number_limit, config.constraint_tolerance,
        )
        accept = ok & pending[:, indices]
        output[:, indices] = torch.where(accept[..., None], weights, output[:, indices])
        valid[:, indices] |= accept

    fallback = static.speech_band_f[None, :] & ~valid
    output = torch.where(fallback[..., None], das, output)
    if not torch.isfinite(output).all():
        output = das
        fallback = static.speech_band_f[None, :].expand(candidate_count, -1)
    return LoadedMvdrWeightResult(
        output,
        int(static.speech_band_f.sum().item()),
        tuple(int(item) for item in fallback.sum(dim=1).tolist()),
    )
