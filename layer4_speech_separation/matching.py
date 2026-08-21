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
    """Select the source phase-coherent with its L3 BF 2--4 kHz reference."""

    algorithm_version = "l3_bf_2_4khz_complex_coherence_v2"
    minimum_reliable_samples = 2 * L4_MODEL_SAMPLE_RATE
    minimum_reliable_score = 0.50
    minimum_score_margin = 0.025

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

    def _frames(self, waveform: NDArray[np.float32]) -> NDArray[np.float32]:
        value = np.asarray(waveform, dtype=np.float32)
        if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
            raise ValueError("matching audio must be finite non-empty mono audio")
        frame_count = max(1, 1 + int(np.ceil((len(value) - self.n_fft) / self.hop_length)))
        required = (frame_count - 1) * self.hop_length + self.n_fft
        if len(value) < required:
            value = np.pad(value, (0, required - len(value)))
        return np.lib.stride_tricks.sliding_window_view(value, self.n_fft)[
            : frame_count * self.hop_length : self.hop_length
        ]

    def _score(self, reference: NDArray[np.float32], candidate: NDArray[np.float32]) -> float:
        if len(reference) != len(candidate):
            raise ValueError("matching reference and candidate must be time-aligned and equal length")
        left_frames = self._frames(reference)
        right_frames = self._frames(candidate)
        weighted_score = 0.0
        total_weight = 0.0
        unweighted_score = 0.0
        valid_count = 0
        # Bound peak memory for session-length recordings while preserving the
        # exact complete-audio statistic.
        for start in range(0, len(left_frames), 4096):
            stop = min(len(left_frames), start + 4096)
            left = np.fft.rfft(
                left_frames[start:stop] * self._window, axis=1,
            )[:, self._bins]
            right = np.fft.rfft(
                right_frames[start:stop] * self._window, axis=1,
            )[:, self._bins]
            # Magnitude-only cosine similarity cannot distinguish two speakers
            # with similar speech spectra. Complex coherence retains the
            # phase/time signature of the directional L3 reference while the
            # absolute inner product remains invariant to a harmless global
            # polarity flip in a separated source.
            numerator = np.abs(np.sum(np.conjugate(left) * right, axis=1))
            denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
            valid = denominator > np.finfo(np.float64).eps
            if not np.any(valid):
                continue
            scores = np.clip(numerator[valid] / denominator[valid], 0.0, 1.0)
            weights = np.sum(np.abs(left[valid]) ** 2, axis=1)
            weighted_score += float(np.sum(scores * weights))
            total_weight += float(np.sum(weights))
            unweighted_score += float(np.sum(scores))
            valid_count += len(scores)
        if valid_count == 0:
            return 0.0
        return weighted_score / total_weight if total_weight > 0.0 else unweighted_score / valid_count

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
        fallback_reason = None
        if len(reference) < self.minimum_reliable_samples:
            fallback_reason = "shorter_than_2_seconds"
        elif scores[selected] < self.minimum_reliable_score:
            fallback_reason = "low_complex_coherence"
        elif abs(scores[0] - scores[1]) < self.minimum_score_margin:
            fallback_reason = "ambiguous_candidate_scores"
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
            waveform=reference if fallback_reason is not None else candidates.sources[selected],
            used_reference_fallback=fallback_reason is not None,
            fallback_reason=fallback_reason,
        )
