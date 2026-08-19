from __future__ import annotations

import torch

from common.geometry import MicGeometry


def steering_vectors(
    frequencies_hz: torch.Tensor, theta_degrees: torch.Tensor, geometry: MicGeometry,
) -> torch.Tensor:
    radians = torch.deg2rad(theta_degrees.to(torch.float64))
    directions = torch.stack((torch.cos(radians), torch.sin(radians)), dim=-1)
    positions = torch.tensor(geometry.positions_m.copy(), dtype=torch.float64, device=frequencies_hz.device)
    delays = -(directions @ positions.T) / geometry.speed_of_sound_mps
    phase = -2.0 * torch.pi * frequencies_hz.to(torch.float64)[None, :, None] * delays[:, None, :]
    result = torch.exp(1j * phase).to(torch.complex64)
    # Center is the exact phase reference, including at DC.
    result[..., 6] = 1.0 + 0.0j
    return result
