from __future__ import annotations

import numpy as np


_VALIDATION_CHUNK_ELEMENTS = 1_048_576


class _ValidatedFloat32Vector(np.ndarray):
    """Private marker for immutable vectors already checked by a contract."""

    _finite_float32_trusted = True


def is_trusted_finite_float32(value: object) -> bool:
    """Recognize internal immutable arrays and validated disk audio views."""

    current = value
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if bool(getattr(current, "_finite_float32_trusted", False)) or bool(
            getattr(current, "_disk_audio_trusted", False)
        ):
            return True
        current = getattr(current, "base", None)
    return False


def validate_finite_float32_vector(
    value: object,
    *,
    name: str,
    allow_empty: bool = False,
) -> np.ndarray:
    """Validate with a bounded temporary working set, even for multi-hour audio."""

    array = np.asarray(value)
    if (
        array.ndim != 1
        or array.dtype != np.float32
        or not array.flags.c_contiguous
        or (not allow_empty and not len(array))
    ):
        raise ValueError(f"{name} must be C-contiguous float32 mono data")
    if not is_trusted_finite_float32(value):
        for start in range(0, len(array), _VALIDATION_CHUNK_ELEMENTS):
            if not np.isfinite(array[start:start + _VALIDATION_CHUNK_ELEMENTS]).all():
                raise ValueError(f"{name} must contain only finite values")
    return array


def validate_probability_vector(
    value: object,
    *,
    name: str,
    allow_empty: bool = False,
) -> np.ndarray:
    """Validate finite [0,1] float32 probabilities with bounded temporaries."""

    array = validate_finite_float32_vector(
        value,
        name=name,
        allow_empty=allow_empty,
    )
    for start in range(0, len(array), _VALIDATION_CHUNK_ELEMENTS):
        chunk = array[start:start + _VALIDATION_CHUNK_ELEMENTS]
        if np.any((chunk < 0.0) | (chunk > 1.0)):
            raise ValueError(f"{name} must contain values in [0,1]")
    return array


def readonly_validated_probability_vector(
    value: object,
    *,
    name: str,
    allow_empty: bool = False,
) -> np.ndarray:
    """Return immutable finite [0,1] probabilities with bounded validation."""

    array = validate_probability_vector(value, name=name, allow_empty=allow_empty)
    if not array.flags.writeable and is_trusted_finite_float32(value):
        return value if isinstance(value, np.ndarray) else array
    result = np.frombuffer(array.tobytes(), dtype=np.float32).view(
        _ValidatedFloat32Vector
    )
    result.flags.writeable = False
    return result


def readonly_validated_float32_vector(value: object, *, name: str) -> np.ndarray:
    """Return immutable validated audio without copying trusted disk-backed data."""

    array = validate_finite_float32_vector(value, name=name)
    if not array.flags.writeable and is_trusted_finite_float32(value):
        return value if isinstance(value, np.ndarray) else array
    result = np.frombuffer(array.tobytes(), dtype=np.float32).view(
        _ValidatedFloat32Vector
    )
    result.flags.writeable = False
    return result
