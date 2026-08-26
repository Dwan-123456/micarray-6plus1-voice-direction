from __future__ import annotations

import gc

import numpy as np
import pytest

from common.disk_audio import DiskAudioStore
from common.validated_array import (
    readonly_validated_float32_vector,
    readonly_validated_probability_vector,
)
from layer5_voice_classifier.contracts import Layer5AudioSegment


def test_external_readonly_bytes_are_still_validated() -> None:
    waveform = np.zeros(320, dtype=np.float32)
    waveform[17] = np.nan
    external_readonly = np.frombuffer(waveform.tobytes(), dtype=np.float32)

    with pytest.raises(ValueError, match="finite"):
        readonly_validated_float32_vector(
            external_readonly,
            name="external audio",
        )


def test_probability_validation_does_not_freeze_mutable_caller() -> None:
    source = np.array([0.1, 0.9], dtype=np.float32)

    result = readonly_validated_probability_vector(source, name="probabilities")

    assert source.flags.writeable
    assert not result.flags.writeable
    assert not np.shares_memory(source, result)
    np.testing.assert_array_equal(result, source)


def test_probability_validation_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        readonly_validated_probability_vector(
            np.array([0.2, 1.1], dtype=np.float32),
            name="probabilities",
        )


def test_l5_reuses_trusted_disk_audio_view_without_full_track_copy() -> None:
    store = DiskAudioStore(prefix="test_l5_validated_disk_view_")
    spool = store.create_spool("audio")
    spool.append(np.linspace(-0.5, 0.5, 640, dtype=np.float32))
    view = spool.view(0, 640)

    segment = Layer5AudioSegment(
        "session",
        0,
        1,
        640,
        10.0,
        16_000,
        view,
    )

    assert segment.waveform is view
    store.retire()
    del segment, view
    gc.collect()
