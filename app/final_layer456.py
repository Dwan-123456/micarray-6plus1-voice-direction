from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from layer4_speech_separation import (
    Layer4LongAudioInput,
    Layer4OfflineResult,
    Layer4ProcessedAudio,
)

from .realtime_postprocessing import RealtimePostprocessingSnapshot


TrackKey = tuple[str, int, int]


def _key(value: object) -> TrackKey:
    source = getattr(value, "source", value)
    return (
        str(getattr(source, "session_id")),
        int(getattr(source, "stream_epoch")),
        int(getattr(source, "track_id")),
    )


def _expected_kinds(source: Layer4LongAudioInput) -> set[str]:
    count = min(2, max(value for _, value in source.l2_direction_counts))
    return {"merged"} if count == 1 else {"candidate_0", "candidate_1"}


def _source_matches(
    candidate: Layer4LongAudioInput,
    sealed: Layer4LongAudioInput,
) -> bool:
    return (
        _key(candidate) == _key(sealed)
        and candidate.start_sample == sealed.start_sample
        and candidate.end_sample == sealed.end_sample
        and candidate.sha256 == sealed.sha256
    )


def _canonical_metadata(
    metadata: Mapping[str, object],
    *,
    source: str,
) -> dict[str, object]:
    return {
        **dict(metadata),
        "canonical": True,
        "realtime_provisional": False,
        "canonical_source": source,
        "canonical_l4_reused": True,
        "finality_scope": "validated_realtime_tail_flushed",
    }


def _promote_track(
    sealed: Layer4LongAudioInput,
    processed: tuple[Layer4ProcessedAudio, ...],
    l5_results: tuple[Layer4OfflineResult, ...],
) -> tuple[tuple[Layer4ProcessedAudio, ...], tuple[Layer4OfflineResult, ...]]:
    by_kind = {item.output_kind: item for item in l5_results}
    promoted_l4: list[Layer4ProcessedAudio] = []
    promoted_l5: list[Layer4OfflineResult] = []
    for item in sorted(processed, key=lambda value: value.output_kind):
        result = by_kind[item.output_kind]
        decision = replace(
            item.speaker_count,
            asset_id=sealed.asset_id,
            metadata={
                **dict(item.speaker_count.metadata),
                "canonical": True,
                "canonical_source": "validated_realtime_tail_flushed",
            },
        )
        request_id = f"{sealed.asset_id}:promoted-realtime"
        output_id = f"{sealed.asset_id}:l4:promoted:{item.output_kind}"
        promoted = replace(
            item,
            request_id=request_id,
            source=sealed,
            speaker_count=decision,
            output_asset_id=output_id,
            metadata=_canonical_metadata(
                item.metadata,
                source="validated_realtime_tail_flushed",
            ),
        )
        promoted_result = replace(
            result,
            request_id=request_id,
            source=sealed,
            speaker_count=decision,
            output_asset_id=output_id,
            metadata={
                **_canonical_metadata(
                    result.metadata,
                    source="validated_realtime_tail_flushed",
                ),
                "output_waveform_16k": promoted.waveform_16k,
            },
        )
        promoted_l4.append(promoted)
        promoted_l5.append(promoted_result)
    return tuple(promoted_l4), tuple(promoted_l5)


@dataclass(frozen=True, slots=True)
class FinalReusePlan:
    reused_l4: tuple[Layer4ProcessedAudio, ...]
    reused_l5: tuple[Layer4OfflineResult, ...]
    missing_sources: tuple[Layer4LongAudioInput, ...]
    reused_track_keys: tuple[TrackKey, ...]
    rejected: tuple[tuple[TrackKey, str], ...]

    @property
    def exact_fast_path(self) -> bool:
        return bool(self.reused_track_keys) and not self.missing_sources


@dataclass(frozen=True, slots=True)
class FinalLayer456Outcome:
    backend_id: str
    pipeline: object
    l4_processed: tuple[Layer4ProcessedAudio, ...]
    l5_results: tuple[Layer4OfflineResult, ...]
    l6_result: object | None
    reused_track_keys: tuple[TrackKey, ...]
    recomputed_track_keys: tuple[TrackKey, ...]
    rejected: tuple[tuple[TrackKey, str], ...]
    exact_fast_path: bool
    stage_durations_seconds: tuple[tuple[str, float], ...]
    diagnostics: Mapping[str, object]


def plan_final_reuse(
    snapshot: RealtimePostprocessingSnapshot | None,
    sealed_sources: tuple[Layer4LongAudioInput, ...],
    *,
    backend_id: str,
) -> FinalReusePlan:
    """Promote exact, complete realtime tracks and isolate only unsafe fallbacks."""

    sources = tuple(sealed_sources)
    empty_keys = tuple(_key(source) for source in sources)
    if snapshot is None:
        return FinalReusePlan((), (), sources, (), tuple((key, "no_final_snapshot") for key in empty_keys))
    if any(source.session_id != snapshot.session_id for source in sources):
        return FinalReusePlan((), (), sources, (), tuple((key, "session_mismatch") for key in empty_keys))

    l4_by_key: dict[TrackKey, list[Layer4ProcessedAudio]] = {}
    for item in snapshot.l4_processed:
        if isinstance(item, Layer4ProcessedAudio):
            l4_by_key.setdefault(_key(item), []).append(item)
    l5_by_key: dict[TrackKey, list[Layer4OfflineResult]] = {}
    for item in snapshot.l5_results:
        if isinstance(item, Layer4OfflineResult):
            l5_by_key.setdefault(_key(item), []).append(item)

    reused_l4: list[Layer4ProcessedAudio] = []
    reused_l5: list[Layer4OfflineResult] = []
    missing: list[Layer4LongAudioInput] = []
    reused_keys: list[TrackKey] = []
    rejected: list[tuple[TrackKey, str]] = []
    for source in sources:
        key = _key(source)
        processed = tuple(l4_by_key.get(key, ()))
        results = tuple(l5_by_key.get(key, ()))
        expected = _expected_kinds(source)
        reason: str | None = None
        if (
            len(processed) != len(expected)
            or {item.output_kind for item in processed} != expected
        ):
            reason = "l4_branch_set_incomplete"
        elif (
            len(results) != len(expected)
            or {item.output_kind for item in results} != expected
        ):
            reason = "l5_branch_set_incomplete"
        elif not snapshot.is_final and any(
            not bool(item.metadata.get("realtime_track_final"))
            for item in processed + results
        ):
            reason = "track_not_final"
        elif any(not _source_matches(item.source, source) for item in processed + results):
            reason = "source_coverage_or_hash_mismatch"
        elif any(str(item.metadata.get("backend")) != backend_id for item in processed):
            reason = "backend_mismatch"
        elif any(bool(item.metadata.get("realtime_preview_degraded")) for item in processed):
            reason = "realtime_preview_degraded"
        elif any(
            not bool(item.metadata.get("dnsmos_finalized_without_full_rerun"))
            for item in processed
        ):
            reason = "dnsmos_not_finalized"
        elif any(
            int(item.metadata.get("realtime_l4_valid_through_sample_48k", -1))
            < source.end_sample
            for item in processed
        ):
            reason = "l4_watermark_incomplete"
        elif any(
            int(item.metadata.get("realtime_l5_valid_through_sample_48k", -1))
            != source.end_sample
            for item in results
        ):
            reason = "l5_watermark_incomplete"
        elif any(
            item.output_sha256
            != next(value.output_sha256 for value in processed if value.output_kind == item.output_kind)
            for item in results
        ):
            reason = "l4_l5_output_hash_mismatch"
        if reason is not None:
            missing.append(source)
            rejected.append((key, reason))
            continue
        promoted_l4, promoted_l5 = _promote_track(source, processed, results)
        reused_l4.extend(promoted_l4)
        reused_l5.extend(promoted_l5)
        reused_keys.append(key)
    return FinalReusePlan(
        tuple(reused_l4),
        tuple(reused_l5),
        tuple(missing),
        tuple(reused_keys),
        tuple(rejected),
    )


def track_load_diagnostics(
    sources: tuple[Layer4LongAudioInput, ...],
    *,
    chunk_samples_48k: int,
) -> dict[str, object]:
    if not sources:
        return {
            "sealed_track_count": 0,
            "shorter_than_chunk_count": 0,
            "source_audio_seconds": 0.0,
            "timeline_span_seconds": 0.0,
            "overlap_load_ratio": 0.0,
        }
    total = sum(len(source.waveform) for source in sources)
    start = min(source.start_sample for source in sources)
    end = max(source.end_sample for source in sources)
    span = max(1, end - start)
    return {
        "sealed_track_count": len(sources),
        "shorter_than_chunk_count": sum(
            len(source.waveform) < chunk_samples_48k for source in sources
        ),
        "source_audio_seconds": total / 48_000.0,
        "timeline_span_seconds": span / 48_000.0,
        "overlap_load_ratio": total / span,
    }
