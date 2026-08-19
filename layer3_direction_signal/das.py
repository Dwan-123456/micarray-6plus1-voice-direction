from __future__ import annotations

import torch


def das_weights(steering: torch.Tensor) -> torch.Tensor:
    return steering / steering.shape[-1]


def apply_weights(weights: torch.Tensor, spectrum_fct: torch.Tensor) -> torch.Tensor:
    return torch.einsum("mfc,fct->mft", weights.conj(), spectrum_fct)
