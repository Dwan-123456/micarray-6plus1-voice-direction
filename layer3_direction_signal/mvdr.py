from __future__ import annotations

import torch

from common.geometry import MicGeometry


def sample_covariance(spectrum_fct: torch.Tensor) -> torch.Tensor:
    return spectrum_fct @ spectrum_fct.mH / spectrum_fct.shape[-1]


def diffuse_covariance(frequencies_hz: torch.Tensor, geometry: MicGeometry) -> torch.Tensor:
    positions = torch.tensor(geometry.positions_m.copy(), dtype=torch.float64, device=frequencies_hz.device)
    distances = torch.linalg.vector_norm(positions[:, None, :] - positions[None, :, :], dim=-1)
    argument = 2.0 * frequencies_hz.to(torch.float64)[:, None, None] * distances / geometry.speed_of_sound_mps
    # torch.sinc(x) is sin(pi*x)/(pi*x), hence argument is 2*f*r/c.
    return torch.sinc(argument).to(torch.complex64)


def _loaded_solve(covariance: torch.Tensor, steering: torch.Tensor, loading: float) -> tuple[torch.Tensor, torch.Tensor]:
    channels = covariance.shape[-1]
    eye = torch.eye(channels, dtype=torch.complex64, device=covariance.device)
    scale = torch.clamp(torch.diagonal(covariance, dim1=-2, dim2=-1).real.sum(-1) / channels, min=1e-8)
    loaded = covariance + float(loading) * scale[..., None, None] * eye
    try:
        value = torch.linalg.solve(loaded, steering.unsqueeze(-1)).squeeze(-1)
    except RuntimeError:
        return torch.zeros_like(steering), torch.zeros(steering.shape[:-1], dtype=torch.bool, device=steering.device)
    denominator = torch.einsum("...c,...c->...", steering.conj(), value)
    valid = (
        torch.isfinite(value).all(dim=-1) & torch.isfinite(denominator)
        & (denominator.real > 1e-8)
        & (torch.abs(denominator.imag) <= 1e-4 * denominator.real)
    )
    safe_denominator = torch.where(valid, denominator.real, torch.ones_like(denominator.real))
    weights = value / safe_denominator.unsqueeze(-1)
    response = torch.einsum("...c,...c->...", weights.conj(), steering)
    valid &= torch.isfinite(weights).all(dim=-1) & (torch.abs(response - 1.0) <= 1e-3)
    return weights, valid


def adaptive_mvdr_weights(
    covariance_fcc: torch.Tensor, steering_mfc: torch.Tensor, loading_factors: tuple[float, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.zeros_like(steering_mfc)
    valid = torch.zeros(steering_mfc.shape[:2], dtype=torch.bool, device=steering_mfc.device)
    for loading in loading_factors:
        for candidate in range(steering_mfc.shape[0]):
            pending = ~valid[candidate]
            if not bool(pending.any()):
                continue
            weights, ok = _loaded_solve(covariance_fcc[pending], steering_mfc[candidate, pending], loading)
            indices = torch.nonzero(pending, as_tuple=False).flatten()
            output[candidate, indices[ok]] = weights[ok]
            valid[candidate, indices[ok]] = True
    return output, valid


def robust_superdirective_weights(
    diffuse_fcc: torch.Tensor, steering_mfc: torch.Tensor, loading_factors: tuple[float, ...], wng_floor_db: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.zeros_like(steering_mfc)
    valid = torch.zeros(steering_mfc.shape[:2], dtype=torch.bool, device=steering_mfc.device)
    wng_floor = 10.0 ** (wng_floor_db / 10.0)
    candidates = (0.0,) + tuple(loading_factors)
    for loading in candidates:
        for candidate in range(steering_mfc.shape[0]):
            pending = ~valid[candidate]
            if not bool(pending.any()):
                continue
            weights, ok = _loaded_solve(diffuse_fcc[pending], steering_mfc[candidate, pending], loading)
            wng = 1.0 / torch.clamp(torch.sum(torch.abs(weights) ** 2, dim=-1), min=1e-12)
            accepted = ok & (wng >= wng_floor)
            indices = torch.nonzero(pending, as_tuple=False).flatten()
            output[candidate, indices[accepted]] = weights[accepted]
            valid[candidate, indices[accepted]] = True
    return output, valid
