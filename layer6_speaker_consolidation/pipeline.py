from __future__ import annotations

from dataclasses import dataclass
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
    noise_rms: float


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-12))


def _cluster(embeddings: tuple[np.ndarray, ...], threshold: float, maximum: int) -> tuple[int, ...]:
    clusters: list[list[int]] = [[index] for index in range(len(embeddings))]

    def similarity(left: list[int], right: list[int]) -> float:
        return float(np.mean([_cosine(embeddings[a], embeddings[b]) for a in left for b in right]))

    while len(clusters) > 1:
        options = [
            (similarity(clusters[left], clusters[right]), left, right)
            for left in range(len(clusters))
            for right in range(left + 1, len(clusters))
        ]
        score, left, right = max(options)
        if len(clusters) <= maximum and score < threshold:
            break
        clusters[left].extend(clusters.pop(right))
    assignments = [0] * len(embeddings)
    for cluster_index, members in enumerate(clusters):
        for member in members:
            assignments[member] = cluster_index
    return tuple(assignments)


class OfflineLayer6Pipeline:
    """Manual offline speaker consolidation after complete L4 and L5 results."""

    def __init__(self, embedder: CampPlusEmbedder, dnsmos: DnsMosScorer, config: Layer6Configuration) -> None:
        self.embedder = embedder
        self.dnsmos = dnsmos
        self.config = config

    def _regions(self, decisions: tuple[bool, ...]) -> tuple[tuple[int, int], ...]:
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
        return tuple((start, end) for start, end in merged if end - start >= minimum)

    def process(self, results: tuple[Layer4OfflineResult, ...]) -> Layer6Result:
        started = perf_counter()
        results = tuple(results)
        if not results:
            raise ValueError("manual L6 requires completed L4/L5 results")
        sessions = {item.source.session_id for item in results}
        if len(sessions) != 1:
            raise ValueError("one L6 job cannot mix capture sessions")
        pending: list[_PendingFragment] = []
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
            for region_index, (first, last) in enumerate(self._regions(result.l5_is_voice_20ms)):
                waveform = np.ascontiguousarray(branch_waveform[first * 320:last * 320], dtype=np.float32)
                embedding_audio = waveform
                target = self.config.minimum_embedding_speech_ms * 16
                if len(embedding_audio) < target:
                    embedding_audio = np.resize(embedding_audio, target).astype(np.float32)
                pending.append(_PendingFragment(
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
                    embedding=self.embedder.embed(np.ascontiguousarray(embedding_audio)),
                    noise_rms=noise_rms,
                ))
        if not pending:
            return Layer6Result(next(iter(sessions)), 0, (), (), {
                "algorithm": "campplus_ahc_dnsmos_timeline_v1", "elapsed_ms": (perf_counter() - started) * 1_000.0,
            })
        raw_assignments = _cluster(
            tuple(item.embedding for item in pending),
            self.config.speaker_similarity_threshold,
            self.config.maximum_speakers,
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
                self.dnsmos,
            )
            fragments.append(Layer6Fragment(
                item.fragment_id, item.source_asset_id, item.source_track_id, item.source_theta_deg,
                item.branch_index, item.selected_for_parent, item.start_sample_48k, item.end_sample_48k,
                item.waveform, item.probabilities, item.decisions, item.embedding, speaker_id, quality,
            ))
        outputs = tuple(self._stitch(speaker_id, tuple(fragments)) for speaker_id in range(1, len(raw_ids) + 1))
        return Layer6Result(next(iter(sessions)), len(outputs), outputs, tuple(fragments), {
            "algorithm": "campplus_ahc_dnsmos_timeline_v1",
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
