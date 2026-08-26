from __future__ import annotations

import heapq
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


class MultiStageConfiguration(Protocol):
    multistage_fallback_distance: float
    multistage_l: int
    multistage_u1: int
    multistage_u2: int
    maximum_speakers: int


@dataclass(frozen=True, slots=True)
class SegmentEvidence:
    """One immutable CAMPPlus observation submitted to streaming clustering."""

    evidence_id: str
    track_key: str
    embedding: np.ndarray
    weight_samples_16k: int

    def __post_init__(self) -> None:
        embedding = np.asarray(self.embedding, dtype=np.float32)
        if (
            not self.evidence_id
            or not self.track_key
            or embedding.ndim != 1
            or not len(embedding)
            or not np.isfinite(embedding).all()
            or self.weight_samples_16k <= 0
        ):
            raise ValueError("invalid multi-stage speaker evidence")
        norm = float(np.linalg.norm(embedding))
        if norm <= 1e-12:
            raise ValueError("multi-stage speaker embedding cannot be zero")
        normalized = np.ascontiguousarray(embedding / norm, dtype=np.float32)
        normalized.flags.writeable = False
        object.__setattr__(self, "embedding", normalized)


@dataclass(frozen=True, slots=True)
class MultiStageSnapshot:
    labels_by_evidence_id: Mapping[str, int]
    evidence_count: int
    cluster_count: int
    stage: str
    assignments_by_track_key: Mapping[str, int] = field(default_factory=dict)


class _ArrayBackedLabelMap(Mapping[str, int]):
    """Immutable label snapshot without rebuilding a Python dict of history.

    Evidence and its ID index are append-only for one clusterer session.  A
    frozen length hides later appends from older snapshots, while the label
    array belongs exclusively to this snapshot.  This preserves historical
    snapshot semantics with one compact native array copy already required by
    cluster limiting, instead of allocating one dict entry per old segment on
    every four-second refresh.
    """

    __slots__ = ("_evidence", "_index_by_id", "_labels", "_length")

    def __init__(
        self,
        evidence: list[SegmentEvidence],
        index_by_id: dict[str, int],
        labels: np.ndarray,
    ) -> None:
        self._evidence = evidence
        self._index_by_id = index_by_id
        self._labels = labels
        self._length = len(labels)

    def __getitem__(self, evidence_id: str) -> int:
        try:
            index = self._index_by_id[evidence_id]
        except KeyError:
            raise KeyError(evidence_id) from None
        if index >= self._length:
            raise KeyError(evidence_id)
        return int(self._labels[index])

    def __iter__(self) -> Iterator[str]:
        for index in range(self._length):
            yield self._evidence[index].evidence_id

    def __len__(self) -> int:
        return self._length


class _StreamingBackend(Protocol):
    def streaming_predict(self, embedding: np.ndarray) -> np.ndarray: ...


def _official_backend(config: MultiStageConfiguration) -> _StreamingBackend:
    from spectralcluster import (
        Deflicker,
        ICASSP2018_REFINEMENT_SEQUENCE,
        MultiStageClusterer,
        RefinementOptions,
        SpectralClusterer,
        ThresholdType,
    )

    main = SpectralClusterer(
        min_clusters=1,
        max_clusters=int(config.maximum_speakers),
        refinement_options=RefinementOptions(
            gaussian_blur_sigma=1,
            p_percentile=0.95,
            thresholding_soft_multiplier=0.01,
            thresholding_type=ThresholdType.RowMax,
            refinement_sequence=ICASSP2018_REFINEMENT_SEQUENCE,
        ),
        custom_dist="cosine",
    )
    return MultiStageClusterer(
        main_clusterer=main,
        fallback_threshold=float(config.multistage_fallback_distance),
        L=int(config.multistage_l),
        U1=int(config.multistage_u1),
        U2=int(config.multistage_u2),
        deflicker=Deflicker.Hungarian,
    )


class MultiStageVoiceprintClusterer:
    """Session-local adapter around SpectralCluster's streaming implementation."""

    def __init__(
        self,
        config: MultiStageConfiguration,
        *,
        backend: _StreamingBackend | None = None,
    ) -> None:
        if not 1 <= int(config.multistage_l) <= int(config.multistage_u1):
            raise ValueError("multi-stage L must be in 1..U1")
        if not int(config.maximum_speakers) < int(config.multistage_u1) < int(config.multistage_u2):
            raise ValueError("multi-stage limits must satisfy maximum_speakers < U1 < U2")
        if not 0.0 < float(config.multistage_fallback_distance) <= 2.0:
            raise ValueError("multi-stage fallback cosine distance must be in (0,2]")
        self.config = config
        self.backend = backend if backend is not None else _official_backend(config)
        self._evidence: list[SegmentEvidence] = []
        self._by_id: dict[str, SegmentEvidence] = {}
        self._index_by_id: dict[str, int] = {}
        self._labels = np.empty(0, dtype=np.int32)
        self._limited_labels = np.empty(0, dtype=np.int32)
        self._track_label_weights: dict[str, dict[int, int]] = {}
        self._track_label_members: dict[str, dict[int, list[int]]] = {}
        self._track_label_heap_entries: dict[str, dict[int, set[int]]] = {}
        self._assignments_by_track_key: dict[str, int] = {}
        self._snapshot_dirty = True
        self._last_snapshot: MultiStageSnapshot | None = None

    @property
    def evidence_count(self) -> int:
        return len(self._evidence)

    def _limit_cluster_count(self, labels: np.ndarray) -> np.ndarray:
        """Merge nearest weighted centroids until the configured cap is met."""

        limited = np.asarray(labels, dtype=np.int32).copy()
        maximum = int(self.config.maximum_speakers)
        while len(np.unique(limited)) > maximum:
            cluster_ids = tuple(int(value) for value in np.unique(limited))
            centroids: dict[int, np.ndarray] = {}
            for cluster_id in cluster_ids:
                indices = np.flatnonzero(limited == cluster_id)
                weights = np.asarray(
                    [self._evidence[index].weight_samples_16k for index in indices],
                    dtype=np.float64,
                )
                vectors = np.stack(
                    [self._evidence[index].embedding for index in indices],
                ).astype(np.float64, copy=False)
                centroid = np.average(vectors, axis=0, weights=weights)
                centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
                centroids[cluster_id] = centroid
            target, source = max(
                (
                    (
                        float(np.dot(centroids[left], centroids[right])),
                        -left,
                        -right,
                        left,
                        right,
                    )
                    for index, left in enumerate(cluster_ids)
                    for right in cluster_ids[index + 1:]
                ),
            )[3:]
            limited[limited == source] = target
        return limited

    def _update_track_assignments(self, limited_labels: np.ndarray) -> None:
        """Apply only new or historically corrected labels to track votes."""

        previous_count = len(self._limited_labels)
        common = min(previous_count, len(limited_labels))
        changed = np.flatnonzero(
            self._limited_labels[:common] != limited_labels[:common]
        )
        if len(limited_labels) > common:
            changed = np.concatenate((
                changed,
                np.arange(common, len(limited_labels), dtype=np.int64),
            ))
        affected: set[str] = set()
        for raw_index in changed:
            index = int(raw_index)
            item = self._evidence[index]
            track_key = item.track_key
            weight = int(item.weight_samples_16k)
            weights = self._track_label_weights.setdefault(track_key, {})
            members = self._track_label_members.setdefault(track_key, {})
            heap_entries = self._track_label_heap_entries.setdefault(track_key, {})
            if index < previous_count:
                old_label = int(self._limited_labels[index])
                remaining = weights.get(old_label, 0) - weight
                if remaining > 0:
                    weights[old_label] = remaining
                else:
                    weights.pop(old_label, None)
            new_label = int(limited_labels[index])
            weights[new_label] = weights.get(new_label, 0) + weight
            present = heap_entries.setdefault(new_label, set())
            if index not in present:
                heapq.heappush(members.setdefault(new_label, []), index)
                present.add(index)
            affected.add(track_key)

        self._limited_labels = limited_labels
        for track_key in affected:
            weights = self._track_label_weights[track_key]
            members = self._track_label_members[track_key]
            heap_entries = self._track_label_heap_entries[track_key]
            first_by_label: dict[int, int] = {}
            for label in weights:
                heap = members[label]
                while heap and int(limited_labels[heap[0]]) != label:
                    heap_entries[label].remove(heapq.heappop(heap))
                if not heap:
                    raise RuntimeError("multi-stage track label index became empty")
                first_by_label[label] = heap[0]
            self._assignments_by_track_key[track_key] = max(
                weights,
                key=lambda label: (
                    weights[label], -first_by_label[label], -label,
                ),
            )

    def _snapshot(self) -> MultiStageSnapshot:
        limited_labels = self._limit_cluster_count(self._labels)
        limited_labels.flags.writeable = False
        self._update_track_assignments(limited_labels)
        count = int(np.unique(limited_labels).size)
        evidence_count = len(self._evidence)
        if evidence_count < int(self.config.multistage_l):
            stage = "fallback_ahc"
        elif evidence_count <= int(self.config.multistage_u1):
            stage = "spectral"
        else:
            stage = "preclustered_spectral"
        snapshot = MultiStageSnapshot(
            _ArrayBackedLabelMap(
                self._evidence,
                self._index_by_id,
                limited_labels,
            ),
            evidence_count,
            count,
            stage,
            dict(self._assignments_by_track_key),
        )
        self._snapshot_dirty = False
        self._last_snapshot = snapshot
        return snapshot

    def update(self, evidence: tuple[SegmentEvidence, ...]) -> MultiStageSnapshot:
        for item in evidence:
            previous = self._by_id.get(item.evidence_id)
            if previous is not None:
                if not np.array_equal(previous.embedding, item.embedding):
                    raise ValueError("immutable multi-stage evidence changed in place")
                continue
            labels = np.asarray(self.backend.streaming_predict(item.embedding), dtype=np.int32)
            expected = len(self._evidence) + 1
            if labels.shape != (expected,) or np.any(labels < 0):
                raise RuntimeError("multi-stage backend returned invalid historical labels")
            index = len(self._evidence)
            self._evidence.append(item)
            self._by_id[item.evidence_id] = item
            self._index_by_id[item.evidence_id] = index
            self._labels = labels
            self._snapshot_dirty = True
        if not self._snapshot_dirty and self._last_snapshot is not None:
            return self._last_snapshot
        return self._snapshot()
