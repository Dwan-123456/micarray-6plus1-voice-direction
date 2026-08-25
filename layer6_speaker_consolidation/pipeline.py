from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Protocol

import numpy as np

from layer4_speech_separation import Layer4OfflineResult

from .contracts import (
    Layer6Fragment,
    Layer6Result,
    Layer6SpeakerAudio,
)
from .models import CampPlusEmbedder


_MINIMUM_VOICEPRINT_SAMPLES = 8_000
_VOICEPRINT_SEGMENT_SAMPLES = 32_000
_MINIMUM_TRACK_MATCH_COUNT = 2
_TRACK_MATCH_COVERAGE = 0.30
_OUTLIER_MAD_SCALE = 2.5
_OUTLIER_MINIMUM_MARGIN = 0.05


class Layer6Configuration(Protocol):
    maximum_speakers: int
    speaker_similarity_threshold: float
    secondary_candidate_match_gap_max: float
    secondary_candidate_match_min: float
    secondary_candidate_mos_min: float
    maximum_internal_silence_ms: int


@dataclass(slots=True)
class _VoiceprintAudio:
    result: Layer4OfflineResult
    branch_index: int
    match_score: float
    mos_score: float
    waveform: np.ndarray
    embedding: np.ndarray
    segment_embeddings: np.ndarray
    voice_sample_count: int
    segment_count: int
    retained_segment_count: int

    @property
    def audio_id(self) -> str:
        return self.result.output_asset_id

    @property
    def start_sample_48k(self) -> int:
        return self.result.source.start_sample

    @property
    def end_sample_48k(self) -> int:
        return self.result.source.end_sample


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-12)
    return float(np.dot(left, right) / denominator)


def _score(metadata: Mapping[str, object], name: str, low: float, high: float) -> float:
    try:
        value = float(metadata[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"L6 requires L4 metadata {name}") from exc
    if not np.isfinite(value) or not low <= value <= high:
        raise ValueError(f"L6 metadata {name} must be between {low} and {high}")
    return value


def _retain_consistent_segments(embeddings: np.ndarray) -> np.ndarray:
    """Drop 2 s embeddings whose within-track centrality is a robust low outlier."""

    if len(embeddings) <= 2:
        return embeddings
    similarities = embeddings @ embeddings.T
    centrality = np.asarray([
        np.median(np.delete(similarities[index], index))
        for index in range(len(embeddings))
    ], dtype=np.float32)
    center = float(np.median(centrality))
    mad = float(np.median(np.abs(centrality - center)))
    cutoff = center - max(_OUTLIER_MINIMUM_MARGIN, _OUTLIER_MAD_SCALE * mad)
    keep = centrality >= cutoff
    if int(np.count_nonzero(keep)) < 2:
        keep[np.argsort(centrality)[-2:]] = True
    return np.ascontiguousarray(embeddings[keep], dtype=np.float32)


def _track_similarity(left: np.ndarray, right: np.ndarray) -> tuple[float, int, int]:
    """Return the weakest required one-to-one 2 s match and its evidence counts."""

    similarities = left @ right.T
    available = np.asarray(similarities, dtype=np.float32).copy()
    matched: list[float] = []
    for _ in range(min(available.shape)):
        flat_index = int(np.argmax(available))
        left_index, right_index = np.unravel_index(flat_index, available.shape)
        matched.append(float(available[left_index, right_index]))
        available[left_index, :] = -np.inf
        available[:, right_index] = -np.inf
    required = max(
        _MINIMUM_TRACK_MATCH_COUNT,
        int(np.ceil(_TRACK_MATCH_COVERAGE * min(len(left), len(right)))),
    )
    if len(matched) < required:
        return -1.0, len(matched), required
    return float(matched[required - 1]), len(matched), required


def _pairwise_similarities(
    items: tuple[_VoiceprintAudio, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.eye(len(items), dtype=np.float32)
    matched_counts = np.zeros((len(items), len(items)), dtype=np.int32)
    required_counts = np.zeros((len(items), len(items)), dtype=np.int32)
    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            value, matched, required = _track_similarity(
                items[left].segment_embeddings,
                items[right].segment_embeddings,
            )
            matrix[left, right] = matrix[right, left] = value
            matched_counts[left, right] = matched_counts[right, left] = matched
            required_counts[left, right] = required_counts[right, left] = required
    return matrix, matched_counts, required_counts


def _cluster(
    similarities: np.ndarray,
    threshold: float,
    maximum_speakers: int,
) -> tuple[int, ...]:
    """Complete-link AHC over multi-segment evidence for each selected L4 track."""

    clusters: list[list[int]] = [[index] for index in range(len(similarities))]

    def complete_link_similarity(left: list[int], right: list[int]) -> float:
        return float(np.min([
            similarities[a, b]
            for a in left
            for b in right
        ]))

    while len(clusters) > 1:
        options = [
            (complete_link_similarity(clusters[left], clusters[right]), left, right)
            for left in range(len(clusters))
            for right in range(left + 1, len(clusters))
        ]
        score, left, right = max(options)
        if len(clusters) <= maximum_speakers and score < threshold:
            break
        clusters[left].extend(clusters.pop(right))

    assignments = [-1] * len(similarities)
    for cluster_index, members in enumerate(clusters):
        for member in members:
            assignments[member] = cluster_index
    return tuple(assignments)


def _compress_silence(
    waveform: np.ndarray,
    occupied: np.ndarray,
    maximum_silence_frames: int,
) -> np.ndarray:
    """Trim edge silence and cap every internal silent run at the configured length."""

    active = np.flatnonzero(occupied)
    if not len(active):
        return np.empty(0, dtype=np.float32)
    first = int(active[0])
    last = int(active[-1]) + 1
    audio_frames = waveform.reshape(-1, 320)[first:last]
    active_frames = occupied[first:last]
    parts: list[np.ndarray] = []
    cursor = 0
    while cursor < len(active_frames):
        state = bool(active_frames[cursor])
        end = cursor + 1
        while end < len(active_frames) and bool(active_frames[end]) == state:
            end += 1
        if state:
            parts.append(audio_frames[cursor:end])
        else:
            keep = min(end - cursor, maximum_silence_frames)
            parts.append(np.zeros((keep, 320), dtype=np.float32))
        cursor = end
    return np.ascontiguousarray(np.concatenate(parts).reshape(-1), dtype=np.float32)


class OfflineLayer6Pipeline:
    """Manual offline complete-track voiceprint clustering and MOS-based merge."""

    def __init__(self, embedder: CampPlusEmbedder, config: Layer6Configuration) -> None:
        self.embedder = embedder
        self.config = config

    @staticmethod
    def _validate_results(
        results: tuple[Layer4OfflineResult, ...],
    ) -> tuple[Layer4OfflineResult, ...]:
        results = tuple(results)
        if not results:
            raise ValueError("manual L6 requires completed L4/L5 results")
        if len({item.source.session_id for item in results}) != 1:
            raise ValueError("one L6 job cannot mix capture sessions")
        if any(item.output_kind not in {"candidate_0", "candidate_1"} for item in results):
            raise ValueError("L6 complete-track clustering requires unmerged L4 A/B outputs")
        return results

    def _select_tracks(
        self,
        results: tuple[Layer4OfflineResult, ...],
    ) -> tuple[tuple[Layer4OfflineResult, int, float, float], ...]:
        groups: dict[tuple[str, int, str], dict[int, Layer4OfflineResult]] = {}
        for result in results:
            branch = int(result.output_kind[-1])
            key = (
                result.source.session_id,
                result.source.stream_epoch,
                result.source.asset_id,
            )
            branches = groups.setdefault(key, {})
            if branch in branches:
                raise ValueError("L6 received a duplicate L4 A/B branch")
            branches[branch] = result

        selected: list[tuple[Layer4OfflineResult, int, float, float]] = []
        for branches in groups.values():
            if 0 not in branches:
                raise ValueError("L6 requires an A track for every L4 source")
            a = branches[0]
            a_match = _score(a.metadata, "candidate_match_score", 0.0, 1.0)
            a_mos = _score(a.metadata, "mos_score", 0.0, 1.0)
            selected.append((a, 0, a_match, a_mos))
            b = branches.get(1)
            if b is None:
                continue
            b_match = _score(b.metadata, "candidate_match_score", 0.0, 1.0)
            b_mos = _score(b.metadata, "mos_score", 0.0, 1.0)
            if (
                abs(a_match - b_match)
                <= self.config.secondary_candidate_match_gap_max + 1e-12
                and b_match > self.config.secondary_candidate_match_min
                and b_mos > self.config.secondary_candidate_mos_min
            ):
                selected.append((b, 1, b_match, b_mos))
        return tuple(selected)

    def _voiceprint_audio(
        self,
        result: Layer4OfflineResult,
        branch_index: int,
        match_score: float,
        mos_score: float,
    ) -> _VoiceprintAudio | None:
        waveform = np.asarray(result.metadata.get("output_waveform_16k"))
        if (
            waveform.dtype != np.float32
            or waveform.ndim != 1
            or not waveform.flags.c_contiguous
            or len(waveform) != len(result.l5_probabilities_20ms) * 320
        ):
            raise ValueError("L6 requires each L4 result to retain aligned 16 kHz audio")
        voice_mask = np.asarray(result.l5_is_voice_20ms, dtype=bool)
        voice_waveform = np.ascontiguousarray(
            waveform.reshape(-1, 320)[voice_mask].reshape(-1),
            dtype=np.float32,
        )
        if len(voice_waveform) < _MINIMUM_VOICEPRINT_SAMPLES:
            return None
        segments = []
        for start in range(0, len(voice_waveform), _VOICEPRINT_SEGMENT_SAMPLES):
            segment = np.ascontiguousarray(
                voice_waveform[start:start + _VOICEPRINT_SEGMENT_SAMPLES],
                dtype=np.float32,
            )
            if len(segment) < _MINIMUM_VOICEPRINT_SAMPLES:
                continue
            segments.append(segment)
        if not segments:
            return None
        batch_embed = getattr(self.embedder, "embed_batch", None)
        if callable(batch_embed):
            segment_embeddings = tuple(batch_embed(tuple(segments)))
        else:
            segment_embeddings = tuple(self.embedder.embed(segment) for segment in segments)
        all_embeddings = np.ascontiguousarray(np.stack(segment_embeddings), dtype=np.float32)
        retained_embeddings = _retain_consistent_segments(all_embeddings)
        embedding = np.mean(retained_embeddings, axis=0)
        embedding /= max(float(np.linalg.norm(embedding)), 1e-12)
        embedding = np.ascontiguousarray(embedding, dtype=np.float32)
        return _VoiceprintAudio(
            result=result,
            branch_index=branch_index,
            match_score=match_score,
            mos_score=mos_score,
            waveform=np.ascontiguousarray(waveform, dtype=np.float32),
            embedding=embedding,
            segment_embeddings=retained_embeddings,
            voice_sample_count=len(voice_waveform),
            segment_count=len(all_embeddings),
            retained_segment_count=len(retained_embeddings),
        )

    @staticmethod
    def _cluster_order(
        items: tuple[_VoiceprintAudio, ...],
        assignments: tuple[int, ...],
        included_clusters: set[int],
    ) -> tuple[int, ...]:
        return tuple(sorted(included_clusters, key=lambda cluster: min(
            (
                items[index].start_sample_48k,
                items[index].result.source.track_id,
                items[index].branch_index,
            )
            for index, assignment in enumerate(assignments)
            if assignment == cluster
        )))

    def process(self, results: tuple[Layer4OfflineResult, ...]) -> Layer6Result:
        started = perf_counter()
        results = self._validate_results(results)
        session_id = results[0].source.session_id
        recording_start = min(item.source.start_sample for item in results)
        recording_end = max(item.source.end_sample for item in results)
        if (
            (recording_end - recording_start) % 960
            or any((item.source.start_sample - recording_start) % 960 for item in results)
        ):
            raise ValueError("L6 source timelines must align to absolute 20 ms frames")
        selected = self._select_tracks(results)
        extracted: list[_VoiceprintAudio] = []
        insufficient_voice_audio_ids: list[str] = []
        for result, branch, match_score, mos_score in selected:
            voiceprint = self._voiceprint_audio(result, branch, match_score, mos_score)
            if voiceprint is None:
                insufficient_voice_audio_ids.append(result.output_asset_id)
            else:
                extracted.append(voiceprint)
        voiceprints = tuple(extracted)

        similarities, matched_counts, required_counts = _pairwise_similarities(voiceprints)
        raw_assignments = _cluster(
            similarities,
            self.config.speaker_similarity_threshold,
            self.config.maximum_speakers,
        )
        active_clusters = {
            raw
            for item, raw in zip(voiceprints, raw_assignments)
            if any(item.result.l5_is_voice_20ms)
        }
        order = self._cluster_order(voiceprints, raw_assignments, active_clusters)
        if not order:
            matrix = tuple(tuple(float(value) for value in row) for row in similarities)
            return Layer6Result(session_id, 0, (), (), {
                "algorithm": "campplus_2s_segment_consistency_complete_link_v5",
                "recording_start_sample_48k": recording_start,
                "recording_end_sample_48k": recording_end,
                "extracted_audio_ids": tuple(item.audio_id for item in voiceprints),
                "pairwise_audio_ids": tuple(item.audio_id for item in voiceprints),
                "pairwise_similarity_matrix": matrix,
                "pairwise_matched_segment_counts": tuple(
                    tuple(int(value) for value in row) for row in matched_counts
                ),
                "pairwise_required_segment_counts": tuple(
                    tuple(int(value) for value in row) for row in required_counts
                ),
                "voiceprint_voice_sample_counts": {
                    item.audio_id: item.voice_sample_count for item in voiceprints
                },
                "voiceprint_segment_counts": {
                    item.audio_id: item.segment_count for item in voiceprints
                },
                "voiceprint_retained_segment_counts": {
                    item.audio_id: item.retained_segment_count for item in voiceprints
                },
                "silent_voiceprint_audio_ids": tuple(item.audio_id for item in voiceprints),
                "insufficient_voice_audio_ids": tuple(insufficient_voice_audio_ids),
                "elapsed_ms": (perf_counter() - started) * 1_000.0,
            })
        remap = {raw: index + 1 for index, raw in enumerate(order)}
        speaker_ids = tuple(remap.get(raw, 0) for raw in raw_assignments)
        centroids: dict[int, np.ndarray] = {}
        for speaker_id in range(1, len(order) + 1):
            centroid = np.mean([
                item.embedding
                for item, assigned in zip(voiceprints, speaker_ids)
                if assigned == speaker_id
            ], axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
            centroids[speaker_id] = centroid

        assignments: list[Layer6Fragment] = []
        for item, speaker_id in zip(voiceprints, speaker_ids):
            if speaker_id == 0:
                continue
            probabilities = tuple(float(value) for value in item.result.l5_probabilities_20ms)
            speaker_similarity = float(np.clip(
                _cosine(item.embedding, centroids[speaker_id]), 0.0, 1.0,
            ))
            assignments.append(Layer6Fragment(
                item.audio_id,
                item.result.source.asset_id,
                item.result.source.track_id,
                item.result.source.theta_deg,
                item.branch_index,
                item.branch_index == 0,
                item.start_sample_48k,
                item.end_sample_48k,
                item.waveform,
                probabilities,
                tuple(item.result.l5_is_voice_20ms),
                item.embedding,
                speaker_id,
                item.match_score,
                item.mos_score,
                speaker_similarity,
            ))

        outputs = tuple(
            output
            for speaker_id in range(1, len(order) + 1)
            if (output := self._merge_speaker(
                speaker_id,
                tuple(assignments),
                recording_start,
                recording_end,
            )) is not None
        )
        if len(outputs) != len(order):
            raise ValueError("L6 voiceprint cluster contains no active audio")
        matrix = tuple(tuple(float(value) for value in row) for row in similarities)
        return Layer6Result(session_id, len(outputs), outputs, tuple(assignments), {
            "algorithm": "campplus_2s_segment_consistency_complete_link_v5",
            "recording_start_sample_48k": recording_start,
            "recording_end_sample_48k": recording_end,
            "secondary_candidate_gate": {
                "match_gap_max": self.config.secondary_candidate_match_gap_max,
                "match_min": self.config.secondary_candidate_match_min,
                "mos_min": self.config.secondary_candidate_mos_min,
            },
            "maximum_internal_silence_ms": self.config.maximum_internal_silence_ms,
            "extracted_audio_ids": tuple(item.audio_id for item in voiceprints),
            "voiceprint_voice_sample_counts": {
                item.audio_id: item.voice_sample_count for item in voiceprints
            },
            "voiceprint_segment_counts": {
                item.audio_id: item.segment_count for item in voiceprints
            },
            "voiceprint_retained_segment_counts": {
                item.audio_id: item.retained_segment_count for item in voiceprints
            },
            "insufficient_voice_audio_ids": tuple(insufficient_voice_audio_ids),
            "silent_voiceprint_audio_ids": tuple(
                item.audio_id
                for item, speaker_id in zip(voiceprints, speaker_ids)
                if speaker_id == 0
            ),
            "pairwise_audio_ids": tuple(item.audio_id for item in voiceprints),
            "pairwise_similarity_matrix": matrix,
            "pairwise_matched_segment_counts": tuple(
                tuple(int(value) for value in row) for row in matched_counts
            ),
            "pairwise_required_segment_counts": tuple(
                tuple(int(value) for value in row) for row in required_counts
            ),
            "voiceprint_audio_ids": {
                speaker_id: tuple(
                    item.audio_id
                    for item, assigned in zip(voiceprints, speaker_ids)
                    if assigned == speaker_id
                )
                for speaker_id in range(1, len(order) + 1)
            },
            "speaker_similarity_threshold": self.config.speaker_similarity_threshold,
            "elapsed_ms": (perf_counter() - started) * 1_000.0,
        })

    def _merge_speaker(
        self,
        speaker_id: int,
        assignments: tuple[Layer6Fragment, ...],
        recording_start: int,
        recording_end: int,
    ) -> Layer6SpeakerAudio | None:
        items = tuple(item for item in assignments if item.speaker_id == speaker_id)
        frame_count = (recording_end - recording_start) // 960
        waveform = np.zeros(frame_count * 320, dtype=np.float32)
        occupied = np.zeros(frame_count, dtype=bool)
        priorities = sorted(
            items,
            key=lambda item: (
                -item.mos_score,
                -item.match_score,
                -item.speaker_similarity,
                item.branch_index,
                item.fragment_id,
            ),
        )
        for item in priorities:
            first_output_frame = (item.start_sample_48k - recording_start) // 960
            frames = item.waveform_16k.reshape(-1, 320)
            for source_frame, active in enumerate(item.voice_is_active_20ms):
                output_frame = first_output_frame + source_frame
                if not active or occupied[output_frame]:
                    continue
                target = output_frame * 320
                waveform[target:target + 320] = frames[source_frame]
                occupied[output_frame] = True
        compressed = _compress_silence(
            waveform,
            occupied,
            self.config.maximum_internal_silence_ms // 20,
        )
        if not len(compressed):
            return None
        return Layer6SpeakerAudio(
            speaker_id,
            f"Speaker {chr(64 + speaker_id)}",
            16_000,
            recording_start,
            recording_end,
            compressed,
            tuple(sorted({item.source_track_id for item in items})),
            tuple(item.fragment_id for item in items),
            float(np.mean([item.mos_score for item in items])),
        )
