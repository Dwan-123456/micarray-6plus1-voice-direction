"""Minimal inference exports used by the vendored TIGER speech model.

The upstream package imports training-only librosa/asteroid modules here.
Offline inference needs only the two local registries imported directly by
``look2hear.models.tiger``.
"""

from . import activations, normalizations

__all__ = ["activations", "normalizations"]
