from __future__ import annotations

from dataclasses import dataclass

import torch

from .adaptive_separation import AdaptiveStaticData, _mvdr, adaptive_static_data
from .configuration import SpatialSeparationConfig
from .das import das_weights


@dataclass(frozen=True, slots=True)
class SubbandRobustWeightResult:
    weights_mfc: torch.Tensor
    postfilter_mf: torch.Tensor
    band_bins: tuple[int, int, int, int, int]
    fallback_bins: tuple[int, ...]
    minimum_wng_db: tuple[float, ...]


def _source_powers(
    spectrum_fct: torch.Tensor,
    noise_covariance_fcc: torch.Tensor,
    steering_mfc: torch.Tensor,
) -> torch.Tensor:
    """Fit non-negative rank-1 source SCM powers for the current directions."""
    snapshots = max(int(spectrum_fct.shape[-1]), 1)
    mixture = torch.einsum(
        "fct,fdt->fcd", spectrum_fct, spectrum_fct.conj(),
    ) / float(snapshots)
    residual = 0.5 * (
        mixture - noise_covariance_fcc
        + (mixture - noise_covariance_fcc).mH
    )
    projected = torch.einsum(
        "mfc,fcd,mfd->fm", steering_mfc.conj(), residual, steering_mfc,
    ).real
    overlap = torch.einsum(
        "mfc,nfc->fmn", steering_mfc.conj(), steering_mfc,
    ).abs().square()
    scale = torch.diagonal(overlap, dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-8)
    identity = torch.eye(
        steering_mfc.shape[0], dtype=overlap.dtype, device=overlap.device,
    )
    regularized = overlap + 1e-3 * scale[:, None, None] * identity[None]
    try:
        powers = torch.linalg.solve(regularized, projected)
    except RuntimeError:
        powers = torch.zeros_like(projected)
    powers = torch.where(torch.isfinite(powers), powers, torch.zeros_like(powers))
    return powers.clamp_min(0.0).transpose(0, 1).contiguous()


def _source_covariances(
    powers_mf: torch.Tensor,
    steering_mfc: torch.Tensor,
) -> torch.Tensor:
    return (
        powers_mf[..., None, None]
        * steering_mfc[..., :, None]
        * steering_mfc.conj()[..., None, :]
    )


def _condition_ok(matrix: torch.Tensor, limit: float) -> torch.Tensor:
    try:
        condition = torch.linalg.cond(matrix)
    except RuntimeError:
        return torch.zeros(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    return torch.isfinite(condition) & (condition <= limit)


def _hard_lcmv_all_targets(
    covariance_fcc: torch.Tensor,
    steering_mfc: torch.Tensor,
    condition_limit: float,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    constraints_fcm = steering_mfc.permute(1, 2, 0).contiguous()
    try:
        solved = torch.linalg.solve(covariance_fcc, constraints_fcm)
        gram = constraints_fcm.mH @ solved
        identity = torch.eye(
            steering_mfc.shape[0], dtype=torch.complex64, device=covariance_fcc.device,
        ).expand(len(covariance_fcc), -1, -1)
        transform = torch.linalg.solve(gram, identity)
        columns = solved @ transform
    except RuntimeError:
        empty = torch.zeros_like(steering_mfc)
        invalid = torch.zeros(len(covariance_fcc), dtype=torch.bool, device=covariance_fcc.device)
        return empty, invalid
    weights = columns.permute(2, 0, 1).contiguous()
    response = columns.mH @ constraints_fcm
    identity = torch.eye(
        steering_mfc.shape[0], dtype=torch.complex64, device=covariance_fcc.device,
    )
    error = torch.amax(torch.abs(response - identity[None]), dim=(-2, -1))
    valid = (
        _condition_ok(covariance_fcc, condition_limit)
        & _condition_ok(gram, condition_limit)
        & torch.isfinite(weights).all(dim=(0, 2))
        & (error <= tolerance)
    )
    return weights, valid


def _wng_constrained_blend(
    hard_weights_mfc: torch.Tensor,
    das_weights_mfc: torch.Tensor,
    hard_valid_f: torch.Tensor,
    mask_f: torch.Tensor,
    minimum_wng_db: float,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose the deepest hard-LCMV/DAS blend that meets the WNG floor."""
    alpha = torch.linspace(
        0.0, 1.0, steps + 1, dtype=torch.float32, device=hard_weights_mfc.device,
    )
    blended = (
        alpha[:, None, None, None] * hard_weights_mfc[None]
        + (1.0 - alpha[:, None, None, None]) * das_weights_mfc[None]
    )
    wng = torch.reciprocal(blended.abs().square().sum(dim=-1).clamp_min(1e-12))
    threshold = 10.0 ** (float(minimum_wng_db) / 10.0)
    eligible = (
        hard_valid_f[None, None, :]
        & mask_f[None, None, :]
        & (wng >= threshold)
    )
    scores = torch.where(
        eligible,
        alpha[:, None, None],
        torch.full_like(wng, -1.0),
    )
    selected = torch.argmax(scores, dim=0)
    source_index = torch.arange(hard_weights_mfc.shape[0], device=selected.device)[:, None]
    frequency_index = torch.arange(hard_weights_mfc.shape[1], device=selected.device)[None, :]
    weights = blended[selected, source_index, frequency_index]
    valid = scores.max(dim=0).values >= 0.0
    return weights, valid


def _loaded_covariance(
    covariance_fcc: torch.Tensor,
    noise_confidence_f: torch.Tensor,
    static: AdaptiveStaticData,
    config: SpatialSeparationConfig,
    factor: float,
) -> torch.Tensor:
    scale = torch.diagonal(covariance_fcc, dim1=-2, dim2=-1).real.mean(dim=-1).clamp_min(1e-8)
    uncertainty = 1.0 + (1.0 - noise_confidence_f) * config.uncertainty_loading_multiplier
    loading = float(factor) * uncertainty * static.alias_multiplier_f * scale
    return covariance_fcc + loading[:, None, None] * static.identity_cc[None]


def _accept_mvdr(
    output: torch.Tensor,
    valid: torch.Tensor,
    covariance_fcc: torch.Tensor,
    steering_mfc: torch.Tensor,
    mask_f: torch.Tensor,
    config: SpatialSeparationConfig,
) -> None:
    pending = mask_f[None, :] & ~valid
    if not bool(pending.any()):
        return
    indices = torch.nonzero(mask_f, as_tuple=False).flatten()
    weights, ok = _mvdr(
        covariance_fcc[indices], steering_mfc[:, indices],
        config.condition_number_limit, config.constraint_tolerance,
    )
    accept = ok & pending[:, indices]
    output[:, indices] = torch.where(accept[..., None], weights, output[:, indices])
    valid[:, indices] |= accept


def subband_robust_weights(
    noise_covariance_fcc: torch.Tensor,
    spectrum_fct: torch.Tensor,
    steering_mfc: torch.Tensor,
    frequencies_hz: torch.Tensor,
    noise_confidence_f: torch.Tensor,
    config: SpatialSeparationConfig,
    *,
    static: AdaptiveStaticData | None = None,
) -> SubbandRobustWeightResult:
    """Five-band robust BF using free-field steering as the initial RTF proxy."""
    source_count, frequency_count, channel_count = steering_mfc.shape
    if source_count not in {1, 2, 3} or channel_count != 7:
        raise ValueError("subband robust BF requires one, two, or three 7-channel directions")
    if spectrum_fct.shape[:2] != (frequency_count, channel_count):
        raise ValueError("subband robust BF spectrum shape does not match steering")
    static = static or adaptive_static_data(frequencies_hz, config, channel_count=channel_count)
    das = das_weights(steering_mfc)
    output = das.clone()
    valid = torch.zeros(
        (source_count, frequency_count), dtype=torch.bool, device=steering_mfc.device,
    )
    valid[:, ~static.speech_band_f] = True

    edge0, edge1, edge2, edge3 = config.subband_frequency_edges_hz
    low = static.speech_band_f & (frequencies_hz < edge0)
    low_mid = static.speech_band_f & (frequencies_hz >= edge0) & (frequencies_hz < edge1)
    mid = static.speech_band_f & (frequencies_hz >= edge1) & (frequencies_hz < edge2)
    core = static.speech_band_f & (frequencies_hz >= edge2) & (frequencies_hz <= edge3)
    high = static.speech_band_f & (frequencies_hz > edge3)
    band_bins = tuple(int(item.sum().item()) for item in (low, low_mid, mid, core, high))

    powers = _source_powers(spectrum_fct, noise_covariance_fcc, steering_mfc)
    source_covariance = _source_covariances(powers, steering_mfc)
    total_source_covariance = source_covariance.sum(dim=0)

    for factor in config.loading_retry_factors:
        loaded_noise = _loaded_covariance(
            noise_covariance_fcc, noise_confidence_f, static, config, factor,
        )

        if bool(low.any()):
            for target in range(source_count):
                interference = total_source_covariance - source_covariance[target]
                mild_covariance = (
                    loaded_noise
                    + config.subband_mild_interference_scale * interference
                )
                target_mask = low & ~valid[target]
                if not bool(target_mask.any()):
                    continue
                indices = torch.nonzero(target_mask, as_tuple=False).flatten()
                weights, ok = _mvdr(
                    mild_covariance[indices], steering_mfc[target:target + 1, indices],
                    config.condition_number_limit, config.constraint_tolerance,
                )
                output[target, indices] = torch.where(
                    ok[0, :, None], weights[0], output[target, indices],
                )
                valid[target, indices] |= ok[0]

        if source_count == 1:
            _accept_mvdr(
                output, valid, loaded_noise, steering_mfc, low_mid | mid | core | high, config,
            )
            continue

        # Compute the multi-constraint hard-null solution once, then continuously
        # blend it toward distortionless DAS only as far as each band's WNG floor
        # requires.  This avoids a solve per target and per soft-null step.
        hard_lcmv, hard_valid = _hard_lcmv_all_targets(
            loaded_noise,
            steering_mfc,
            config.condition_number_limit,
            config.constraint_tolerance,
        )
        for band, minimum_wng_db in zip(
            (low_mid, mid, core), config.subband_wng_floors_db, strict=True,
        ):
            if not bool(band.any()):
                continue
            weights, ok = _wng_constrained_blend(
                hard_lcmv,
                das,
                hard_valid,
                band,
                minimum_wng_db,
                config.subband_soft_null_steps,
            )
            accept = ok & ~valid
            output = torch.where(accept[..., None], weights, output)
            valid |= accept

        _accept_mvdr(output, valid, loaded_noise, steering_mfc, high, config)

    fallback = static.speech_band_f[None, :] & ~valid
    output = torch.where(fallback[..., None], das, output)
    if not torch.isfinite(output).all():
        output = das
        fallback = static.speech_band_f[None, :].expand(source_count, -1)

    postfilter = torch.ones(
        (source_count, frequency_count), dtype=torch.float32, device=steering_mfc.device,
    )
    if bool(low.any()):
        response = torch.einsum("mfc,nfc->mnf", output.conj(), steering_mfc)
        source_at_output = powers[None, :, :] * response.abs().square()
        desired = torch.diagonal(source_at_output, dim1=0, dim2=1).transpose(0, 1)
        interference = source_at_output.sum(dim=1) - desired
        noise_at_output = torch.einsum(
            "mfc,fcd,mfd->mf", output.conj(), noise_covariance_fcc, output,
        ).real.clamp_min(0.0)
        gain = desired / (desired + interference + noise_at_output).clamp_min(1e-12)
        gain = gain.clamp(config.subband_wiener_min_gain, 1.0)
        postfilter[:, low] = gain[:, low]

    wng = torch.reciprocal(output.abs().square().sum(dim=-1).clamp_min(1e-12))
    speech_wng = torch.where(
        static.speech_band_f[None, :], wng, torch.full_like(wng, torch.inf),
    )
    minimum_wng = tuple(
        float(10.0 * torch.log10(item.clamp_min(1e-12)).item())
        for item in speech_wng.min(dim=1).values
    )
    return SubbandRobustWeightResult(
        output,
        postfilter,
        band_bins,
        tuple(int(item) for item in fallback.sum(dim=1).tolist()),
        minimum_wng,
    )
