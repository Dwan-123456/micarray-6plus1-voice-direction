from __future__ import annotations

from dataclasses import dataclass

import torch

from common.geometry import MicGeometry

from .configuration import SpatialSeparationConfig
from .das import das_weights
from .steering import steering_vectors


@dataclass(frozen=True, slots=True)
class ConstantBeamwidthSolution:
    weights_mfc: torch.Tensor
    fallback_bins: tuple[int, ...]
    minimum_wng_db: tuple[float, ...]


def constant_beamwidth_weights(
    frequencies_hz: torch.Tensor,
    target_steering_mfc: torch.Tensor,
    theta_degrees: torch.Tensor,
    geometry: MicGeometry,
    config: SpatialSeparationConfig,
) -> ConstantBeamwidthSolution:
    """Numerical UCA adaptation of a constant-FNBW fixed beamformer.

    The Frank/Ben-Kish/Cohen construction is derived for a ULA.  For this
    project's 6+1 UCA, each in-band frequency instead fits the same desired
    30-degree first-null response on the physical array manifold.  The fit is
    distortionless at the requested look direction and falls back to DAS when
    its white-noise gain would be unsafe.
    """
    if target_steering_mfc.ndim != 3 or target_steering_mfc.shape[-1] != 7:
        raise ValueError("constant-beamwidth steering must be [M,F,7]")
    if theta_degrees.shape != (target_steering_mfc.shape[0],):
        raise ValueError("constant-beamwidth target angles do not match steering")

    weights = das_weights(target_steering_mfc).clone()
    passband = (
        (frequencies_hz >= config.frequency_min_hz)
        & (frequencies_hz <= config.frequency_max_hz)
    )
    bin_indices = torch.nonzero(passband, as_tuple=False).flatten()
    if bin_indices.numel() == 0:
        count = target_steering_mfc.shape[0]
        return ConstantBeamwidthSolution(weights, tuple(0 for _ in range(count)),
                                         tuple(float("inf") for _ in range(count)))

    grid_step = config.constant_beamwidth_design_grid_deg
    grid_angles = torch.arange(
        0.0, 360.0, grid_step, dtype=torch.float32, device=frequencies_hz.device,
    )
    selected_frequencies = frequencies_hz[bin_indices]
    # [F,A,C]. This is a temporary design tensor and is never retained in a cache.
    manifold_fac = steering_vectors(selected_frequencies, grid_angles, geometry).permute(1, 0, 2)
    # Fit d(theta)^H w to a real desired response.  The conjugation order is
    # significant: rows of the least-squares design matrix are d(theta)^H.
    gram_fcc = manifold_fac.transpose(-2, -1) @ manifold_fac.conj()
    scale_f = gram_fcc.diagonal(dim1=-2, dim2=-1).real.mean(dim=-1)
    identity = torch.eye(7, dtype=torch.complex64, device=frequencies_hz.device)
    gram_fcc = gram_fcc + (
        config.constant_beamwidth_regularization * scale_f[:, None, None] * identity[None]
    )

    fallback_counts: list[int] = []
    minimum_wng: list[float] = []
    half_width = 0.5 * config.constant_beamwidth_fnbw_deg
    min_wng_linear = 10.0 ** (config.constant_beamwidth_min_wng_db / 10.0)
    for target_index, theta in enumerate(theta_degrees):
        delta = torch.remainder(grid_angles - theta + 180.0, 360.0) - 180.0
        inside = delta.abs() < half_width
        desired = torch.zeros_like(delta)
        # A smooth, real-valued lobe: unity at boresight and its first target
        # zeros at +/- FNBW/2. Outside the lobe the desired response is zero.
        desired[inside] = torch.cos(
            0.5 * torch.pi * delta[inside] / half_width,
        ).square()
        rhs_fc = torch.einsum(
            "fac,a->fc", manifold_fac, desired.to(torch.complex64),
        )
        unconstrained_fc = torch.linalg.solve(gram_fcc, rhs_fc.unsqueeze(-1)).squeeze(-1)

        edge_angles = torch.stack((theta, theta - half_width, theta + half_width)).to(torch.float32)
        constraints_fcg = steering_vectors(
            selected_frequencies, edge_angles, geometry,
        ).permute(1, 2, 0)
        inverse_constraints_fcg = torch.linalg.solve(gram_fcc, constraints_fcg)
        constraint_gram_fgg = (
            constraints_fcg.conj().transpose(-2, -1) @ inverse_constraints_fcg
        )
        desired_constraints = torch.tensor(
            [1.0, 0.0, 0.0], dtype=torch.complex64, device=frequencies_hz.device,
        ).expand(bin_indices.numel(), -1)
        current_constraints_fg = torch.einsum(
            "fcg,fc->fg", constraints_fcg.conj(), unconstrained_fc,
        )
        correction_fg = torch.einsum(
            "fgh,fh->fg",
            torch.linalg.pinv(constraint_gram_fgg, rtol=1.0e-5, hermitian=True),
            desired_constraints - current_constraints_fg,
        )
        fitted_weights_fc = unconstrained_fc + torch.einsum(
            "fcg,fg->fc", inverse_constraints_fcg, correction_fg,
        )
        achieved_constraints = torch.einsum(
            "fcg,fc->fg", constraints_fcg.conj(), fitted_weights_fc,
        )
        constraint_error_f = (achieved_constraints - desired_constraints).abs().amax(dim=-1)
        wng_f = 1.0 / fitted_weights_fc.abs().square().sum(dim=-1).clamp_min(1.0e-12)
        stable = (
            torch.isfinite(fitted_weights_fc).all(dim=-1)
            & (constraint_error_f <= 0.05)
            & (wng_f >= min_wng_linear)
        )
        weights[target_index, bin_indices] = torch.where(
            stable[:, None], fitted_weights_fc, weights[target_index, bin_indices],
        )
        fallback_counts.append(int((~stable).sum().item()))
        finite_wng = wng_f[stable & torch.isfinite(wng_f)]
        minimum_wng.append(
            float(10.0 * torch.log10(finite_wng.min()).item()) if finite_wng.numel() else float("-inf")
        )

    return ConstantBeamwidthSolution(weights, tuple(fallback_counts), tuple(minimum_wng))
