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
from .matching import TrackMatchFeatures, hungarian_track_features
from .multistage import MultiStageSnapshot, MultiStageVoiceprintClusterer, SegmentEvidence


_MINIMUM_VOICEPRINT_SAMPLES = 8_000
_VOICEPRINT_SEGMENT_SAMPLES = 32_000
_MINIMUM_TRACK_MATCH_COUNT = 1
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
    clustering_backend: str
    multistage_fallback_distance: float
    multistage_l: int
    multistage_u1: int
    multistage_u2: int


@dataclass(slots=True)
class _VoiceprintAudio:
    result: Layer4OfflineResult
    branch_index: int
    match_score: float
    mos_score: float
    waveform: np.ndarray
    embedding: np.ndarray
    all_segment_embeddings: np.ndarray
    segment_embeddings: np.ndarray
    segment_sample_counts: tuple[int, ...]
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


def _track_match_features(
    left: np.ndarray,
    right: np.ndarray,
    threshold: float = 0.62,
) -> TrackMatchFeatures:
    return hungarian_track_features(
        left,
        right,
        threshold=threshold,
        minimum_match_count=_MINIMUM_TRACK_MATCH_COUNT,
        required_coverage=_TRACK_MATCH_COVERAGE,
    )


def _track_similarity(left: np.ndarray, right: np.ndarray) -> tuple[float, int, int]:
    """Compatibility wrapper returning the Hungarian decision score and counts."""

    features = _track_match_features(left, right)
    return features.decision_score, features.matched_count, features.required_count


def _pairwise_similarities(
    items: tuple[_VoiceprintAudio, ...],
    threshold: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[tuple[dict[str, float | int] | None, ...], ...],
]:
    matrix = np.eye(len(items), dtype=np.float32)
    matched_counts = np.zeros((len(items), len(items)), dtype=np.int32)
    required_counts = np.zeros((len(items), len(items)), dtype=np.int32)
    features: list[list[dict[str, float | int] | None]] = [
        [None] * len(items) for _ in items
    ]
    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            pair = _track_match_features(
                items[left].segment_embeddings,
                items[right].segment_embeddings,
                threshold,
            )
            matrix[left, right] = matrix[right, left] = pair.decision_score
            matched_counts[left, right] = matched_counts[right, left] = pair.matched_count
            required_counts[left, right] = required_counts[right, left] = pair.required_count
            features[left][right] = features[right][left] = pair.as_dict()
    return matrix, matched_counts, required_counts, tuple(tuple(row) for row in features)


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
    """Offline complete-track voiceprint clustering and MOS-based merge."""

    def __init__(self, embedder: CampPlusEmbedder, config: Layer6Configuration) -> None:
        self.embedder = embedder
        self.config = config
        self._streaming_clusterer: MultiStageVoiceprintClusterer | None = None
        self._streaming_session_id: str | None = None

    @staticmethod
    def _validate_results(
        results: tuple[Layer4OfflineResult, ...],
    ) -> tuple[Layer4OfflineResult, ...]:
        results = tuple(results)
        if not results:
            raise ValueError("L6 requires completed L4/L5 results")
        if len({item.source.session_id for item in results}) != 1:
            raise ValueError("one L6 job cannot mix capture sessions")
        if any(
            item.output_kind == "merged" and item.path != "single_speaker_bypass"
            for item in results
        ):
            raise ValueError(
                "L6 requires unmerged L4 A/B outputs or single-speaker bypasses"
            )
        return results

    def _select_tracks(
        self,
        results: tuple[Layer4OfflineResult, ...],
    ) -> tuple[tuple[Layer4OfflineResult, int, float, float], ...]:
        groups: dict[tuple[str, int, str], dict[int, Layer4OfflineResult]] = {}
        for result in results:
            branch = 0 if result.output_kind == "merged" else int(result.output_kind[-1])
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
            a_match = (
                1.0
                if a.output_kind == "merged"
                else _score(a.metadata, "candidate_match_score", 0.0, 1.0)
            )
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
        *,
        include_partial_segment: bool,
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
        segment_sample_counts: list[int] = []
        for start in range(0, len(voice_waveform), _VOICEPRINT_SEGMENT_SAMPLES):
            segment = np.ascontiguousarray(
                voice_waveform[start:start + _VOICEPRINT_SEGMENT_SAMPLES],
                dtype=np.float32,
            )
            if len(segment) < _MINIMUM_VOICEPRINT_SAMPLES:
                continue
            if len(segment) < _VOICEPRINT_SEGMENT_SAMPLES and not include_partial_segment:
                continue
            segments.append(segment)
            segment_sample_counts.append(len(segment))
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
            all_segment_embeddings=all_embeddings,
            segment_embeddings=retained_embeddings,
            segment_sample_counts=tuple(segment_sample_counts),
            voice_sample_count=len(voice_waveform),
            segment_count=len(all_embeddings),
            retained_segment_count=len(retained_embeddings),
        )

    @staticmethod
    def _multistage_evidence(
        items: tuple[_VoiceprintAudio, ...],
    ) -> tuple[SegmentEvidence, ...]:
        evidence: list[tuple[tuple[int, int, int, int], SegmentEvidence]] = []
        for item in items:
            source = item.result.source
            track_key = (
                f"{source.session_id}:epoch{source.stream_epoch}:"
                f"track{source.track_id}:branch{item.branch_index}"
            )
            for segment_index, (embedding, sample_count) in enumerate(zip(
                item.all_segment_embeddings,
                item.segment_sample_counts,
                strict=True,
            )):
                evidence_id = f"{track_key}:segment{segment_index}"
                order = (
                    item.start_sample_48k,
                    source.track_id,
                    item.branch_index,
                    segment_index,
                )
                evidence.append((order, SegmentEvidence(
                    evidence_id,
                    track_key,
                    embedding,
                    sample_count,
                )))
        return tuple(value for _, value in sorted(evidence, key=lambda pair: pair[0]))

    @staticmethod
    def _multistage_track_assignments(
        items: tuple[_VoiceprintAudio, ...],
        snapshot: MultiStageSnapshot,
    ) -> tuple[int, ...]:
        assignments: list[int] = []
        for item in items:
            source = item.result.source
            track_key = (
                f"{source.session_id}:epoch{source.stream_epoch}:"
                f"track{source.track_id}:branch{item.branch_index}"
            )
            weighted: dict[int, int] = {}
            first_index: dict[int, int] = {}
            for segment_index, sample_count in enumerate(item.segment_sample_counts):
                label = snapshot.labels_by_evidence_id[f"{track_key}:segment{segment_index}"]
                weighted[label] = weighted.get(label, 0) + sample_count
                first_index.setdefault(label, segment_index)
            assignments.append(max(
                weighted,
                key=lambda label: (weighted[label], -first_index[label], -label),
            ))
        return tuple(assignments)

    def reset_streaming(self) -> None:
        self._streaming_clusterer = None
        self._streaming_session_id = None

    def process_streaming(
        self,
        results: tuple[Layer4OfflineResult, ...],
        *,
        final: bool = False,
    ) -> Layer6Result:
        backend = str(getattr(self.config, "clustering_backend", "complete_link"))
        if backend != "multistage":
            return self.process(results)
        results = self._validate_results(results)
        session_id = results[0].source.session_id
        if self._streaming_session_id not in {None, session_id}:
            raise ValueError("streaming L6 pipeline cannot mix capture sessions")
        if self._streaming_clusterer is None:
            self._streaming_clusterer = MultiStageVoiceprintClusterer(self.config)
            self._streaming_session_id = session_id
        return self._process(
            results,
            multistage=self._streaming_clusterer,
            include_partial_segment=final,
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
        backend = str(getattr(self.config, "clustering_backend", "complete_link"))
        multistage = (
            MultiStageVoiceprintClusterer(self.config)
            if backend == "multistage"
            else None
        )
        return self._process(
            results,
            multistage=multistage,
            include_partial_segment=True,
        )

    def _process(
        self,
        results: tuple[Layer4OfflineResult, ...],
        *,
        multistage: MultiStageVoiceprintClusterer | None,
        include_partial_segment: bool,
    ) -> Layer6Result:
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
            voiceprint = self._voiceprint_audio(
                result,
                branch,
                match_score,
                mos_score,
                include_partial_segment=include_partial_segment,
            )
            if voiceprint is None:
                insufficient_voice_audio_ids.append(result.output_asset_id)
            else:
                extracted.append(voiceprint)
        voiceprints = tuple(extracted)

        similarities, matched_counts, required_counts, match_features = _pairwise_similarities(
            voiceprints, self.config.speaker_similarity_threshold,
        )
        multistage_snapshot: MultiStageSnapshot | None = None
        if multistage is None:
            raw_assignments = _cluster(
                similarities,
                self.config.speaker_similarity_threshold,
                self.config.maximum_speakers,
            )
            algorithm = "campplus_2s_hungarian_features_complete_link_experiment_v6"
        else:
            multistage_snapshot = multistage.update(self._multistage_evidence(voiceprints))
            raw_assignments = self._multistage_track_assignments(
                voiceprints,
                multistage_snapshot,
            ) if voiceprints else ()
            algorithm = "campplus_2s_multistage_streaming_v1"
        active_clusters = {
            raw
            for item, raw in zip(voiceprints, raw_assignments)
            if any(item.result.l5_is_voice_20ms)
        }
        order = self._cluster_order(voiceprints, raw_assignments, active_clusters)
        if not order:
            matrix = tuple(tuple(float(value) for value in row) for row in similarities)
            return Layer6Result(session_id, 0, (), (), {
                "algorithm": algorithm,
                "multistage": (
                    None if multistage_snapshot is None else {
                        "evidence_count": multistage_snapshot.evidence_count,
                        "cluster_count": multistage_snapshot.cluster_count,
                        "stage": multistage_snapshot.stage,
                    }
                ),
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
                "pairwise_match_features": match_features,
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
            "algorithm": algorithm,
            "multistage": (
                None if multistage_snapshot is None else {
                    "evidence_count": multistage_snapshot.evidence_count,
                    "cluster_count": multistage_snapshot.cluster_count,
                    "stage": multistage_snapshot.stage,
                }
            ),
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
            "pairwise_match_features": match_features,
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
