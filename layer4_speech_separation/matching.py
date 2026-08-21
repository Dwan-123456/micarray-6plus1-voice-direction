from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .contracts import (
    L4_MATCH_FREQUENCY_MAX_HZ,
    L4_MATCH_FREQUENCY_MIN_HZ,
    L4_MODEL_SAMPLE_RATE,
    Layer4CandidatePair,
    Layer4LongAudioInput,
    Layer4PrimarySelection,
)


class BandMagnitudeMatcher:
    """Select a track's dominant separated source using its L3 BF 2--4 kHz signature."""

    algorithm_version = "l3_bf_2_4khz_magnitude_cosine_v1"

    def __init__(self, *, n_fft: int = 512, hop_length: int = 160) -> None:
        if n_fft != 512 or hop_length != 160:
            raise ValueError("Layer 4 v1 matching standard fixes n_fft=512 and hop_length=160")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self._window = np.hanning(n_fft).astype(np.float32)
        frequencies = np.fft.rfftfreq(n_fft, 1.0 / L4_MODEL_SAMPLE_RATE)
        self._bins = (frequencies >= L4_MATCH_FREQUENCY_MIN_HZ) & (
            frequencies <= L4_MATCH_FREQUENCY_MAX_HZ
        )

    def _magnitude(self, waveform: NDArray[np.float32]) -> NDArray[np.float64]:
        value = np.asarray(waveform, dtype=np.float32)
        if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
            raise ValueError("matching audio must be finite non-empty mono audio")
        if len(value) < self.n_fft:
            value = np.pad(value, (0, self.n_fft - len(value)))
        frame_count = 1 + (len(value) - self.n_fft) // self.hop_length
        frames = np.lib.stride_tricks.sliding_window_view(value, self.n_fft)[
            : frame_count * self.hop_length : self.hop_length
        ]
        return np.abs(np.fft.rfft(frames * self._window, axis=1)[:, self._bins])

    def _score(self, reference: NDArray[np.float32], candidate: NDArray[np.float32]) -> float:
        if len(reference) != len(candidate):
            raise ValueError("matching reference and candidate must be time-aligned and equal length")
        left = self._magnitude(reference)
        right = self._magnitude(candidate)
        numerator = np.sum(left * right, axis=1)
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        valid = denominator > np.finfo(np.float64).eps
        if not np.any(valid):
            return 0.0
        weights = np.sum(left[valid] ** 2, axis=1)
        scores = np.clip(numerator[valid] / denominator[valid], 0.0, 1.0)
        if not np.any(weights > 0.0):
            return float(np.mean(scores))
        return float(np.average(scores, weights=weights))

    def select(
        self,
        *,
        parent: Layer4LongAudioInput,
        reference_16k: NDArray[np.float32],
        candidates: Layer4CandidatePair,
    ) -> Layer4PrimarySelection:
        if candidates.request_id == "":
            raise ValueError("candidate request identity is required")
        reference = np.asarray(reference_16k, dtype=np.float32)
        if reference.ndim != 1 or not reference.flags.c_contiguous or not np.isfinite(reference).all():
            raise ValueError("16 kHz L3 reference must be finite C-contiguous float32 mono audio")
        if len(reference) != len(candidates.sources[0]):
            raise ValueError("resampled L3 reference and separated candidates must have equal duration")
        scores = tuple(self._score(reference, source) for source in candidates.sources)
        selected = 0 if scores[0] >= scores[1] else 1
        return Layer4PrimarySelection(
            request_id=candidates.request_id,
            parent_asset_id=parent.asset_id,
            session_id=parent.session_id,
            stream_epoch=parent.stream_epoch,
            track_id=parent.track_id,
            theta_deg=parent.theta_deg,
            sample_rate=L4_MODEL_SAMPLE_RATE,
            selected_source_index=selected,
            candidate_scores=(scores[0], scores[1]),
            score_margin=abs(scores[0] - scores[1]),
            matching_algorithm=self.algorithm_version,
            waveform=candidates.sources[selected],
        )
