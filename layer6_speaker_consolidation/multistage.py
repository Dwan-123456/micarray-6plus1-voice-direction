from __future__ import annotations

from dataclasses import dataclass
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
    labels_by_evidence_id: dict[str, int]
    evidence_count: int
    cluster_count: int
    stage: str


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
        self._labels = np.empty(0, dtype=np.int32)

    @property
    def evidence_count(self) -> int:
        return len(self._evidence)

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
            self._evidence.append(item)
            self._by_id[item.evidence_id] = item
            self._labels = labels
        labels_by_id = {
            item.evidence_id: int(label)
            for item, label in zip(self._evidence, self._labels, strict=True)
        }
        count = len(set(labels_by_id.values()))
        if len(self._evidence) < int(self.config.multistage_l):
            stage = "fallback_ahc"
        elif len(self._evidence) <= int(self.config.multistage_u1):
            stage = "spectral"
        else:
            stage = "preclustered_spectral"
        return MultiStageSnapshot(labels_by_id, len(self._evidence), count, stage)
