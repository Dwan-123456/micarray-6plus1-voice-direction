from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from time import perf_counter
from typing import Protocol

import numpy as np

from layer4_speech_separation import Layer4OfflineResult

from .contracts import Layer6Fragment, Layer6Result, Layer6SpeakerAudio
from .models import CampPlusEmbedder, DnsMosScorer
from .quality import score_quality


class Layer6Configuration(Protocol):
    maximum_speakers: int
    speaker_similarity_threshold: float
    minimum_embedding_speech_ms: int
    minimum_voice_fragment_ms: int
    merge_voice_gap_ms: int
    selection_switch_margin: float
    crossfade_ms: int


@dataclass(slots=True)
class _PendingInput:
    fragment_id: str
    source_asset_id: str
    source_track_id: int
    source_theta_deg: float
    branch_index: int
    selected_for_parent: bool
    start_sample_48k: int
    end_sample_48k: int
    waveform: np.ndarray
    probabilities: tuple[float, ...]
    decisions: tuple[bool, ...]
    embedding_audio: np.ndarray
    embedding_speech_ms: int
    noise_rms: float
    quality_key: tuple[str, int, int, int]
    quality_audio: np.ndarray


@dataclass(slots=True)
class _PendingFragment:
    fragment_id: str
    source_asset_id: str
    source_track_id: int
    source_theta_deg: float
    branch_index: int
    selected_for_parent: bool
    start_sample_48k: int
    end_sample_48k: int
    waveform: np.ndarray
    probabilities: tuple[float, ...]
    decisions: tuple[bool, ...]
    embedding: np.ndarray
    embedding_speech_ms: int
    noise_rms: float
    quality_key: tuple[str, int, int, int]


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-12))


_MINIMUM_CONCURRENT_OVERLAP_SAMPLES_48K = 24_000
_CROSS_TRACK_DUPLICATE_SIMILARITY = 0.85
_WEAK_EXTRA_CLUSTER_MARGIN = 0.15
_SPEAKER_ASSIGNMENT_WINDOW_MS = 500


def _cannot_share_speaker(left: _PendingFragment, right: _PendingFragment) -> bool:
    """Keep two sustained, simultaneous L2 directions in different speaker classes."""

    if left.source_track_id == right.source_track_id:
        return False
    overlap = min(left.end_sample_48k, right.end_sample_48k) - max(
        left.start_sample_48k, right.start_sample_48k,
    )
    shorter = min(
        left.end_sample_48k - left.start_sample_48k,
        right.end_sample_48k - right.start_sample_48k,
    )
    return (
        overlap >= _MINIMUM_CONCURRENT_OVERLAP_SAMPLES_48K
        and overlap * 2 >= shorter
        and _cosine(left.embedding, right.embedding) < _CROSS_TRACK_DUPLICATE_SIMILARITY
    )


def _cluster(
    fragments: tuple[_PendingFragment, ...],
    threshold: float,
    maximum: int,
    minimum_embedding_speech_ms: int,
) -> tuple[int, ...]:
    """Constrained AHC over reliable windows, then attach short residual speech.

    Repeating a 200 ms residual until CAMPPlus accepts it produces an embedding,
    but not enough evidence to establish a new person.  Only windows containing
    the configured minimum amount of real audio seed clusters.  Short residuals
    are assigned to the closest established centroid afterwards.
    """

    minimum_samples = minimum_embedding_speech_ms * 48
    reliable = [
        index
        for index, item in enumerate(fragments)
        if item.embedding_speech_ms * 48 >= minimum_samples
    ]
    if not reliable:
        reliable = [max(
            range(len(fragments)),
            key=lambda index: fragments[index].end_sample_48k - fragments[index].start_sample_48k,
        )]
    clusters: list[list[int]] = [[index] for index in reliable]

    def similarity(left: list[int], right: list[int]) -> float:
        return float(np.mean([
            _cosine(fragments[a].embedding, fragments[b].embedding)
            for a in left for b in right
        ]))

    def constrained_similarity(left: list[int], right: list[int]) -> float:
        score = similarity(left, right)
        if any(
            _cannot_share_speaker(fragments[a], fragments[b])
            for a in left for b in right
        ):
            score -= 0.25
        return score

    while len(clusters) > 1:
        options = [
            (constrained_similarity(clusters[left], clusters[right]), left, right)
            for left in range(len(clusters))
            for right in range(left + 1, len(clusters))
        ]
        score, left, right = max(options)
        if len(clusters) <= maximum and score < threshold:
            break
        clusters[left].extend(clusters.pop(right))

    # Directional L2 tracks provide a conservative speaker floor.  AHC may
    # still add another class for a real speaker change inside one track, but
    # only when that class is clearly separated.  This removes unstable third
    # classes caused by separator leakage and noisy per-window embeddings.
    track_ids = sorted({fragments[index].source_track_id for index in reliable})
    incompatible_tracks = {
        (left, right)
        for left, right in combinations(track_ids, 2)
        if any(
            _cannot_share_speaker(fragments[a], fragments[b])
            for a in reliable for b in reliable
            if fragments[a].source_track_id == left
            and fragments[b].source_track_id == right
        )
    }
    speaker_floor = 1
    for size in range(2, min(maximum, len(track_ids)) + 1):
        if any(
            all(tuple(sorted(pair)) in incompatible_tracks for pair in combinations(group, 2))
            for group in combinations(track_ids, size)
        ):
            speaker_floor = size
    while len(clusters) > max(1, speaker_floor):
        centroids = []
        for members in clusters:
            centroid = np.mean([fragments[index].embedding for index in members], axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
            centroids.append(centroid)
        margins = []
        for cluster_index, members in enumerate(clusters):
            own = float(np.mean([
                _cosine(fragments[index].embedding, centroids[cluster_index])
                for index in members
            ]))
            alternate = max(
                float(np.mean([
                    _cosine(fragments[index].embedding, centroids[other])
                    for index in members
                ]))
                for other in range(len(clusters)) if other != cluster_index
            )
            margins.append(own - alternate)
        weakest = int(np.argmin(margins))
        if margins[weakest] >= _WEAK_EXTRA_CLUSTER_MARGIN:
            break
        destination = max(
            (index for index in range(len(clusters)) if index != weakest),
            key=lambda index: similarity(clusters[weakest], clusters[index]),
        )
        clusters[destination].extend(clusters.pop(weakest))
    assignments = [-1] * len(fragments)
    for cluster_index, members in enumerate(clusters):
        for member in members:
            assignments[member] = cluster_index
    for index, item in enumerate(fragments):
        if assignments[index] >= 0:
            continue
        centroids = [
            np.mean([fragments[member].embedding for member in members], axis=0)
            for members in clusters
        ]
        allowed = [
            cluster_index
            for cluster_index, members in enumerate(clusters)
            if not any(_cannot_share_speaker(item, fragments[member]) for member in members)
        ]
        candidates = allowed or list(range(len(clusters)))
        selected = max(
            candidates,
            key=lambda cluster_index: _cosine(item.embedding, centroids[cluster_index]),
        )
        assignments[index] = selected
        clusters[selected].append(index)
    return tuple(assignments)


class OfflineLayer6Pipeline:
    """Manual offline speaker consolidation after complete L4 and L5 results."""

    def __init__(self, embedder: CampPlusEmbedder, dnsmos: DnsMosScorer, config: Layer6Configuration) -> None:
        self.embedder = embedder
        self.dnsmos = dnsmos
        self.config = config

    def _regions(
        self, decisions: tuple[bool, ...],
    ) -> tuple[tuple[int, int, int, int, int, int], ...]:
        gap = self.config.merge_voice_gap_ms // 20
        runs: list[list[int]] = []
        start: int | None = None
        for index, active in enumerate((*decisions, False)):
            if active and start is None:
                start = index
            elif not active and start is not None:
                runs.append([start, index])
                start = None
        merged: list[list[int]] = []
        for run in runs:
            if merged and run[0] - merged[-1][1] <= gap:
                merged[-1][1] = run[1]
            else:
                merged.append(run)
        minimum = self.config.minimum_voice_fragment_ms // 20
        regions = tuple((start, end) for start, end in merged if end - start >= minimum)
        # A VAD run is not a speaker turn. Assign the timeline in 500 ms pieces,
        # while every piece receives up to 1.5 s of embedding context from only
        # its own merged voice region. A short isolated residual therefore never
        # borrows reliable speech from a different region to seed a new ID.
        assignment_window = _SPEAKER_ASSIGNMENT_WINDOW_MS // 20
        embedding_window = self.config.minimum_embedding_speech_ms // 20
        split: list[tuple[int, int, int, int, int, int]] = []
        for region_start, region_end in regions:
            assignments: list[list[int]] = []
            cursor = region_start
            while region_end - cursor > assignment_window:
                assignments.append([cursor, cursor + assignment_window])
                cursor += assignment_window
            if region_end - cursor >= minimum:
                assignments.append([cursor, region_end])
            elif assignments:
                assignments[-1][1] = region_end
            for first, last in assignments:
                center = (first + last) // 2
                embedding_first = max(
                    region_start,
                    min(center - embedding_window // 2, region_end - embedding_window),
                )
                embedding_last = min(region_end, embedding_first + embedding_window)
                split.append((
                    first, last, embedding_first, embedding_last, region_start, region_end,
                ))
        return tuple(split)

    def process(self, results: tuple[Layer4OfflineResult, ...]) -> Layer6Result:
        started = perf_counter()
        results = tuple(results)
        if not results:
            raise ValueError("manual L6 requires completed L4/L5 results")
        sessions = {item.source.session_id for item in results}
        if len(sessions) != 1:
            raise ValueError("one L6 job cannot mix capture sessions")
        pending_inputs: list[_PendingInput] = []
        for result in results:
            output_kind = getattr(result, "output_kind", "merged")
            if output_kind not in {"merged", "candidate_0", "candidate_1"}:
                raise ValueError("L6 received an unknown L4 output kind")
            branch_index = 0 if output_kind == "merged" else int(output_kind[-1])
            selected_for_parent = output_kind == "merged"
            branch_waveform = np.asarray(result.metadata.get("output_waveform_16k"))
            if (
                branch_waveform.dtype != np.float32
                or branch_waveform.ndim != 1
                or len(branch_waveform) != len(result.l5_probabilities_20ms) * 320
            ):
                raise ValueError("L6 requires each L4 result to retain its aligned 16 kHz waveform")
            frame_audio = branch_waveform.reshape(-1, 320)
            inactive = [
                frame_audio[index]
                for index, active in enumerate(result.l5_is_voice_20ms)
                if not active
            ]
            noise_rms = (
                float(np.sqrt(np.mean(np.square(np.concatenate(inactive), dtype=np.float64)) + 1e-12))
                if inactive else 1e-4
            )
            for region_index, (
                first, last, context_first, context_last, region_first, region_last,
            ) in enumerate(
                self._regions(result.l5_is_voice_20ms)
            ):
                waveform = np.ascontiguousarray(branch_waveform[first * 320:last * 320], dtype=np.float32)
                embedding_audio = np.ascontiguousarray(
                    branch_waveform[context_first * 320:context_last * 320], dtype=np.float32,
                )
                target = self.config.minimum_embedding_speech_ms * 16
                if len(embedding_audio) < target:
                    embedding_audio = np.resize(embedding_audio, target).astype(np.float32)
                pending_inputs.append(_PendingInput(
                    fragment_id=f"{result.source.asset_id}:b{branch_index}:v{region_index}",
                    source_asset_id=result.source.asset_id,
                    source_track_id=result.source.track_id,
                    source_theta_deg=result.source.theta_deg,
                    branch_index=branch_index,
                    selected_for_parent=selected_for_parent,
                    start_sample_48k=result.source.start_sample + first * 960,
                    end_sample_48k=result.source.start_sample + last * 960,
                    waveform=waveform,
                    probabilities=result.l5_probabilities_20ms[first:last],
                    decisions=result.l5_is_voice_20ms[first:last],
                    embedding_audio=np.ascontiguousarray(embedding_audio),
                    embedding_speech_ms=(context_last - context_first) * 20,
                    noise_rms=noise_rms,
                    quality_key=(result.source.asset_id, branch_index, region_first, region_last),
                    quality_audio=np.ascontiguousarray(
                        branch_waveform[region_first * 320:region_last * 320], dtype=np.float32,
                    ),
                ))
        if not pending_inputs:
            return Layer6Result(next(iter(sessions)), 0, (), (), {
                "algorithm": "campplus_batched_constrained_ahc_dnsmos_region_cache_timeline_v3",
                "elapsed_ms": (perf_counter() - started) * 1_000.0,
            })
        embed_many = getattr(self.embedder, "embed_many", None)
        if callable(embed_many):
            embeddings = tuple(embed_many(tuple(item.embedding_audio for item in pending_inputs)))
        else:
            embeddings = tuple(self.embedder.embed(item.embedding_audio) for item in pending_inputs)
        if len(embeddings) != len(pending_inputs):
            raise RuntimeError("CAMPPlus batch output count does not match L6 fragments")
        pending = [
            _PendingFragment(
                item.fragment_id, item.source_asset_id, item.source_track_id,
                item.source_theta_deg, item.branch_index, item.selected_for_parent,
                item.start_sample_48k, item.end_sample_48k, item.waveform,
                item.probabilities, item.decisions, embedding, item.embedding_speech_ms,
                item.noise_rms, item.quality_key,
            )
            for item, embedding in zip(pending_inputs, embeddings, strict=True)
        ]
        dnsmos_scores: dict[tuple[str, int, int, int], tuple[float, float, float]] = {}
        for item in pending_inputs:
            if item.quality_key not in dnsmos_scores:
                dnsmos_scores[item.quality_key] = self.dnsmos.score(item.quality_audio)
        raw_assignments = _cluster(
            tuple(pending),
            self.config.speaker_similarity_threshold,
            self.config.maximum_speakers,
            self.config.minimum_embedding_speech_ms,
        )
        raw_ids = sorted(set(raw_assignments), key=lambda value: min(
            item.start_sample_48k for item, assignment in zip(pending, raw_assignments) if assignment == value
        ))
        remap = {raw: index + 1 for index, raw in enumerate(raw_ids)}
        centroids: dict[int, np.ndarray] = {}
        for raw in raw_ids:
            centroid = np.mean([item.embedding for item, assignment in zip(pending, raw_assignments) if assignment == raw], axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
            centroids[remap[raw]] = centroid
        fragments: list[Layer6Fragment] = []
        for item, raw in zip(pending, raw_assignments):
            speaker_id = remap[raw]
            quality = score_quality(
                item.waveform,
                item.probabilities,
                _cosine(item.embedding, centroids[speaker_id]),
                item.noise_rms,
                dnsmos_scores[item.quality_key],
            )
            fragments.append(Layer6Fragment(
                item.fragment_id, item.source_asset_id, item.source_track_id, item.source_theta_deg,
                item.branch_index, item.selected_for_parent, item.start_sample_48k, item.end_sample_48k,
                item.waveform, item.probabilities, item.decisions, item.embedding, speaker_id, quality,
            ))
        outputs = tuple(self._stitch(speaker_id, tuple(fragments)) for speaker_id in range(1, len(raw_ids) + 1))
        return Layer6Result(next(iter(sessions)), len(outputs), outputs, tuple(fragments), {
            "algorithm": "campplus_batched_constrained_ahc_dnsmos_region_cache_timeline_v3",
            "speaker_assignment_window_ms": _SPEAKER_ASSIGNMENT_WINDOW_MS,
            "speaker_embedding_context_ms": self.config.minimum_embedding_speech_ms,
            "embedding_device": str(getattr(self.embedder, "device", "custom")),
            "embedding_batch_size": int(getattr(self.embedder, "batch_size", 1)),
            "dnsmos_evaluation_count": len(dnsmos_scores),
            "minimum_concurrent_overlap_ms": _MINIMUM_CONCURRENT_OVERLAP_SAMPLES_48K // 48,
            "cross_track_duplicate_similarity": _CROSS_TRACK_DUPLICATE_SIMILARITY,
            "weak_extra_cluster_margin": _WEAK_EXTRA_CLUSTER_MARGIN,
            "speaker_similarity_threshold": self.config.speaker_similarity_threshold,
            "quality_weights": {"voice": 0.30, "speaker": 0.30, "mos": 0.20, "snr": 0.10, "continuity": 0.10},
            "elapsed_ms": (perf_counter() - started) * 1_000.0,
        })

    def _stitch(self, speaker_id: int, fragments: tuple[Layer6Fragment, ...]) -> Layer6SpeakerAudio:
        items = tuple(item for item in fragments if item.speaker_id == speaker_id)
        start = min(item.start_sample_48k for item in items)
        end = max(item.end_sample_48k for item in items)
        waveform = np.zeros((end - start) // 3, dtype=np.float32)
        occupied = np.zeros(len(waveform) // 320, dtype=bool)
        priorities = sorted(
            items,
            key=lambda item: item.quality.total + (
                self.config.selection_switch_margin if item.selected_for_parent else 0.0
            ),
            reverse=True,
        )
        for item in priorities:
            offset = (item.start_sample_48k - start) // 3
            for frame, active in enumerate(item.voice_is_active_20ms):
                if not active:
                    continue
                output_frame = offset // 320 + frame
                if occupied[output_frame]:
                    continue
                target = output_frame * 320
                waveform[target:target + 320] = item.waveform_16k[frame * 320:(frame + 1) * 320]
                occupied[output_frame] = True
        fade = self.config.crossfade_ms * 16
        if fade:
            transitions = np.flatnonzero(np.diff(np.pad(occupied.astype(np.int8), (1, 1))))
            for left, right in zip(transitions[0::2], transitions[1::2]):
                run_start, run_end = left * 320, right * 320
                count = min(fade, (run_end - run_start) // 2)
                if count:
                    waveform[run_start:run_start + count] *= np.linspace(0.0, 1.0, count, dtype=np.float32)
                    waveform[run_end - count:run_end] *= np.linspace(1.0, 0.0, count, dtype=np.float32)
        return Layer6SpeakerAudio(
            speaker_id, f"Speaker {chr(64 + speaker_id)}", 16_000, start, end,
            np.ascontiguousarray(waveform), tuple(sorted({item.source_track_id for item in items})),
            tuple(item.fragment_id for item in items), float(np.mean([item.quality.total for item in items])),
        )
