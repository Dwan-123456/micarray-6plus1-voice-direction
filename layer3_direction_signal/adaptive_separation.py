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


def _condition_ok(matrix: torch.Tensor, limit: float) -> torch.Tensor:
    try:
        condition = torch.linalg.cond(matrix)
    except RuntimeError:
        return torch.zeros(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    return torch.isfinite(condition) & (condition <= limit)


def _mvdr(
    covariance_fcc: torch.Tensor,
    steering_mfc: torch.Tensor,
    condition_limit: float,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    steering_fcm = steering_mfc.permute(1, 2, 0).contiguous()
    try:
        solved = torch.linalg.solve(covariance_fcc, steering_fcm)
    except RuntimeError:
        return torch.zeros_like(steering_mfc), torch.zeros(
            steering_mfc.shape[:2], dtype=torch.bool, device=steering_mfc.device,
        )
    denominator = torch.einsum("fcm,fcm->fm", steering_fcm.conj(), solved)
    safe = torch.where(torch.abs(denominator) > 1e-12, denominator, torch.ones_like(denominator))
    weights = (solved / safe[:, None, :]).permute(2, 0, 1).contiguous()
    response = torch.einsum("mfc,mfc->mf", weights.conj(), steering_mfc)
    matrix_ok = _condition_ok(covariance_fcc, condition_limit)
    valid = (
        matrix_ok[None, :]
        & torch.isfinite(weights).all(dim=-1)
        & torch.isfinite(denominator).transpose(0, 1)
        & (torch.abs(response - 1.0) <= tolerance)
    )
    return weights, valid


def _dual_lcmv(
    covariance_fcc: torch.Tensor,
    steering_mfc: torch.Tensor,
    condition_limit: float,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    constraints_fcm = steering_mfc.permute(1, 2, 0).contiguous()
    try:
        solved = torch.linalg.solve(covariance_fcc, constraints_fcm)
        gram = constraints_fcm.mH @ solved
        transform = torch.linalg.solve(
            gram,
            torch.eye(2, dtype=torch.complex64, device=gram.device).expand(len(gram), 2, 2),
        )
        columns = solved @ transform
    except RuntimeError:
        return torch.zeros_like(steering_mfc), torch.zeros(
            steering_mfc.shape[:2], dtype=torch.bool, device=steering_mfc.device,
        )
    weights = columns.permute(2, 0, 1).contiguous()
    response = columns.mH @ constraints_fcm
    identity = torch.eye(2, dtype=torch.complex64, device=response.device)
    response_error = torch.amax(torch.abs(response - identity[None, :, :]), dim=(-2, -1))
    matrix_ok = _condition_ok(covariance_fcc, condition_limit) & _condition_ok(gram, condition_limit)
    per_frequency = matrix_ok & torch.isfinite(weights).all(dim=(0, 2)) & (response_error <= tolerance)
    return weights, per_frequency[None, :].expand(2, -1)


def _soft_null_mvdr(
    loaded_fcc: torch.Tensor,
    steering_mfc: torch.Tensor,
    strength: float,
    condition_limit: float,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.zeros_like(steering_mfc)
    valid = torch.zeros(steering_mfc.shape[:2], dtype=torch.bool, device=steering_mfc.device)
    scale = torch.diagonal(loaded_fcc, dim1=-2, dim2=-1).real.mean(dim=-1).clamp_min(1e-8)
    for target, interferer in ((0, 1), (1, 0)):
        interference = steering_mfc[interferer]
        penalty = interference[..., :, None] * interference.conj()[..., None, :]
        covariance = loaded_fcc + float(strength) * scale[:, None, None] * penalty
        weights, ok = _mvdr(covariance, steering_mfc[target:target + 1], condition_limit, tolerance)
        output[target] = weights[0]
        valid[target] = ok[0]
    return output, valid


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
    if candidate_count not in {1, 2} or channel_count != 7:
        raise ValueError("adaptive separation requires one or two 7-channel steering vectors")
    das = das_weights(steering_mfc)
    output = das.clone()
    valid = torch.zeros((candidate_count, frequency_count), dtype=torch.bool, device=steering_mfc.device)
    static = static or adaptive_static_data(frequencies_hz, config, channel_count=channel_count)
    speech_band = static.speech_band_f
    if (
        speech_band.shape != (frequency_count,)
        or static.identity_cc.shape != (channel_count, channel_count)
        or static.alias_multiplier_f.shape != (frequency_count,)
    ):
        raise ValueError("adaptive static cache shapes do not match the current solve")
    valid[:, ~speech_band] = True
    if candidate_count == 1:
        rho = torch.ones(frequency_count, dtype=torch.float32, device=steering_mfc.device)
    elif spatial_p_f is None:
        raise ValueError("two-candidate adaptive separation requires precomputed spatial p lookup values")
    else:
        rho = spatial_p_f
    if rho.shape != (frequency_count,) or not torch.isfinite(rho).all() or bool(((rho < 0) | (rho > 1)).any()):
        raise ValueError("spatial p must be finite [frequency] values in [0,1]")
    rho = rho.to(device=steering_mfc.device, dtype=torch.float32)
    if candidate_count == 1:
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

    for factor in config.loading_retry_factors:
        if bool(valid[:, speech_band].all()):
            break
        loading = float(factor) * loading_multiplier * scale
        loaded = covariance_fcc + loading[:, None, None] * eye

        lcmv_pending = lcmv_mask & ~valid.all(dim=0)
        if candidate_count == 2 and bool(lcmv_pending.any()):
            indices = torch.nonzero(lcmv_pending, as_tuple=False).flatten()
            weights, ok = _dual_lcmv(
                loaded[indices], steering_mfc[:, indices],
                config.condition_number_limit, config.constraint_tolerance,
            )
            accept = ok & ~valid[:, indices]
            output[:, indices] = torch.where(accept[..., None], weights, output[:, indices])
            valid[:, indices] |= accept

        soft_pending = soft_mask & ~valid.all(dim=0)
        if candidate_count == 2 and bool(soft_pending.any()):
            indices = torch.nonzero(soft_pending, as_tuple=False).flatten()
            weights, ok = _soft_null_mvdr(
                loaded[indices], steering_mfc[:, indices], config.soft_null_strength,
                config.condition_number_limit, config.constraint_tolerance,
            )
            accept = ok & ~valid[:, indices]
            output[:, indices] = torch.where(accept[..., None], weights, output[:, indices])
            valid[:, indices] |= accept

        mvdr_pending = mvdr_mask & ~valid.all(dim=0)
        if bool(mvdr_pending.any()):
            indices = torch.nonzero(mvdr_pending, as_tuple=False).flatten()
            weights, ok = _mvdr(
                loaded[indices], steering_mfc[:, indices],
                config.condition_number_limit, config.constraint_tolerance,
            )
            accept = ok & ~valid[:, indices]
            output[:, indices] = torch.where(accept[..., None], weights, output[:, indices])
            valid[:, indices] |= accept

    fallback = speech_band[None, :] & ~valid
    output = torch.where(fallback[..., None], das, output)
    if not torch.isfinite(output).all():
        output = das
        fallback = speech_band[None, :].expand(candidate_count, -1)
    return AdaptiveWeightResult(
        output,
        rho,
        int(lcmv_mask.sum().item()),
        int(soft_mask.sum().item()),
        int(mvdr_mask.sum().item()),
        tuple(int(item) for item in fallback.sum(dim=1).tolist()),
    )
