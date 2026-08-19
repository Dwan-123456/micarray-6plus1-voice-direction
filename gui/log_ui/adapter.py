from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Callable, Protocol, runtime_checkable

from .models import (
    Anomaly,
    Availability,
    CapabilitySet,
    SessionDescriptor,
    SessionReadModel,
    StageObservation,
    StageState,
    TrackObservation,
    WindowKey,
    WindowObservation,
)


_STAGES = ("l1", "gate", "l2", "l3", "l4", "commit")
_SUPPORTED_SCHEMAS = {"decision_record_v3", "decision_record_v4"}


@runtime_checkable
class ReadOnlyProvider(Protocol):
    """Public query surface injected by an owning, already-running host.

    The Log UI intentionally has no constructor that accepts a data directory.
    This prevents it from implicitly creating/opening Catalog or WAL files.
    """

    def runtime_sessions(self) -> list[dict[str, Any]]: ...

    def session_decisions(self, session_id: str, *, include_v3: bool = True) -> list[dict[str, Any]]: ...

    def runtime_session_tracks(
        self, session_id: str, *, stream_epoch: int | None = None
    ) -> list[dict[str, Any]]: ...

    def track_timeline(self, session_id: str, stream_epoch: int, track_id: int) -> list[dict[str, Any]]: ...

    def track_audio_assets(self, session_id: str, stream_epoch: int, track_id: int) -> list[dict[str, Any]]: ...

    def session_audio_assets(self, session_id: str, kind: str) -> list[dict[str, Any]]: ...


def _public_copy(value: object) -> object:
    """Detach public DTO data and redact paths/private implementation fields."""
    if isinstance(value, Mapping):
        output: dict[str, object] = {}
        for key, item in value.items():
            name = str(key)
            if name.startswith("_"):
                continue
            if name.casefold() in {"path", "absolute_path", "operation_root"}:
                output[name] = "[redacted]"
            elif name.casefold().endswith("_json") and isinstance(item, str):
                try:
                    output[name] = _public_copy(json.loads(item))
                except json.JSONDecodeError:
                    output[name] = "[invalid public JSON]"
            else:
                output[name] = _public_copy(item)
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_public_copy(item) for item in value]
    return deepcopy(value)


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) and result >= 0.0 else None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


class PublicApiAdapter:
    """Capability-probed adapter over public, read-only project methods only."""

    _METHODS = {
        "sessions": "runtime_sessions",
        "decisions": "session_decisions",
        "tracks": "runtime_session_tracks",
        "track_timeline": "track_timeline",
        "track_audio": "track_audio_assets",
        "session_audio": "session_audio_assets",
        "runtime_status": "processing_status",
    }

    def __init__(self, provider: object, *, cache_capacity: int = 4):
        if provider is None:
            raise ValueError("Pipeline Log UI requires an injected public read-only provider")
        self._provider = provider
        from .cache import BoundedLru

        self._cache: BoundedLru[str, SessionReadModel] = BoundedLru(cache_capacity)
        available = {
            capability: callable(getattr(provider, method, None))
            for capability, method in self._METHODS.items()
        }
        self.capabilities = CapabilitySet(**available)

    def _call(self, capability: str, *args: object, **kwargs: object) -> Any:
        method_name = self._METHODS[capability]
        method = getattr(self._provider, method_name, None)
        if not callable(method):
            raise NotImplementedError(f"public capability unavailable: {capability}")
        return method(*args, **kwargs)

    def list_sessions(self) -> tuple[SessionDescriptor, ...]:
        if not self.capabilities.sessions:
            return ()
        rows = self._call("sessions")
        if not isinstance(rows, Sequence):
            raise TypeError("runtime_sessions() must return a sequence")
        return tuple(self._descriptor(_mapping(row)) for row in rows)

    def _descriptor(self, row: Mapping[str, Any]) -> SessionDescriptor:
        metadata = _mapping(row.get("metadata"))
        if not metadata and isinstance(row.get("metadata_json"), str):
            try:
                metadata = _mapping(json.loads(str(row["metadata_json"])))
            except json.JSONDecodeError:
                metadata = {}
        device = _mapping(metadata.get("device_format"))
        sample_rate_value = row.get("sample_rate", device.get("sample_rate"))
        sample_rate = int(sample_rate_value) if isinstance(sample_rate_value, int) and sample_rate_value > 0 else None
        duration_samples = _duration_samples(row, metadata)
        algorithm_versions = _mapping(metadata.get("algorithm_versions"))
        status = str(row.get("status", "unknown"))
        warning = None
        availability = Availability.AVAILABLE
        if status.casefold() == "open":
            warning = "录制中 / 数据可能不完整；尚未封存字段不计为零或失败"
            availability = Availability.NOT_SEALED
        return SessionDescriptor(
            session_id=str(row.get("id") or row.get("session_id") or ""),
            status=status,
            schema_version=str(row.get("schema_version", "unknown")),
            started_at=None if row.get("started_at") is None else str(row["started_at"]),
            ended_at=None if row.get("ended_at") is None else str(row["ended_at"]),
            mode=None if row.get("mode") is None else str(row["mode"]),
            sample_rate=sample_rate,
            duration_seconds=(duration_samples / sample_rate if duration_samples is not None and sample_rate else None),
            project_version=_text(row.get("project_version", metadata.get("project_version"))),
            algorithm_version=(", ".join(f"{key}={value}" for key, value in sorted(algorithm_versions.items())) or None),
            config_hash=_text(row.get("config_hash", metadata.get("config_hash"))),
            calibration_hash=_text(row.get("calibration_hash", metadata.get("calibration_hash"))),
            data_integrity=_text(row.get("data_integrity", metadata.get("data_integrity"))),
            capabilities=self.capabilities,
            availability=availability,
            warning=warning,
            raw_public=_public_copy(row),
        )

    def load_session(
        self,
        session_id: str,
        *,
        refresh: bool = False,
        cancelled: Callable[[], bool] | None = None,
    ) -> SessionReadModel:
        if refresh:
            self._cache.clear()
        if cancelled is not None:
            return self._load_session(session_id, cancelled=cancelled)
        return self._cache.get_or_load(session_id, lambda: self._load_session(session_id))

    def _load_session(
        self, session_id: str, *, cancelled: Callable[[], bool] | None = None
    ) -> SessionReadModel:
        descriptor = next((item for item in self.list_sessions() if item.session_id == session_id), None)
        if descriptor is None:
            raise FileNotFoundError(session_id)
        if not self.capabilities.decisions:
            return SessionReadModel(descriptor, (), (), (), Availability.NOT_PROVIDED)
        try:
            rows = self._call("decisions", session_id, include_v3=True)
        except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
            anomaly = Anomaly("record_validation", f"公开记录校验失败：{exc}")
            return SessionReadModel(descriptor, (), (), (anomaly,), Availability.VALIDATION_FAILED)
        if not isinstance(rows, Sequence):
            raise TypeError("session_decisions() must return a sequence")
        windows: list[WindowObservation] = []
        anomalies: list[Anomaly] = []
        for index, row in enumerate(rows):
            if cancelled is not None and cancelled():
                from .controller import CancelledError

                raise CancelledError("session normalization cancelled")
            try:
                window = self._window(session_id, _mapping(row))
            except (TypeError, ValueError) as exc:
                anomalies.append(Anomaly("record_validation", f"记录 {index}：{exc}"))
                continue
            if window.schema_version not in _SUPPORTED_SCHEMAS:
                anomalies.append(Anomaly("schema", f"不支持的DecisionRecord：{window.schema_version}", window.key))
                continue
            windows.append(window)
        windows.sort(key=lambda item: item.key)
        anomalies.extend(self._window_anomalies(windows))
        tracks, track_anomalies = self._tracks(windows)
        anomalies.extend(track_anomalies)
        availability = Availability.AVAILABLE
        if rows and not windows:
            availability = Availability.UNSUPPORTED_SCHEMA
        elif not rows:
            availability = Availability.NOT_RECORDED
        return SessionReadModel(descriptor, tuple(windows), tracks, tuple(anomalies), availability)

    @staticmethod
    def _window(session_id: str, row: Mapping[str, Any]) -> WindowObservation:
        row_session = str(row.get("session_id", ""))
        if row_session != session_id:
            raise ValueError("DecisionRecord session_id不匹配")
        for field in ("stream_epoch", "window_id", "decision_sample"):
            if type(row.get(field)) is not int or int(row[field]) < 0:
                raise ValueError(f"{field}必须是非负整数")
        key = WindowKey(row_session, int(row["stream_epoch"]), int(row["window_id"]), int(row["decision_sample"]))
        statuses = _mapping(row.get("stage_statuses"))
        timings = _mapping(row.get("stage_timings_ms"))
        waits = _mapping(row.get("stage_queue_wait_ms"))
        ages = _mapping(row.get("stage_end_to_end_ms"))
        reason = _text(row.get("terminal_reason"))
        stages = {
            name: StageObservation(
                StageState.from_public(statuses.get(name)),
                _finite(timings.get(name)),
                _finite(waits.get(name)),
                _finite(ages.get(name)),
                reason if name in statuses and StageState.from_public(statuses.get(name)) not in {
                    StageState.COMPLETED, StageState.SKIPPED,
                } else None,
            )
            for name in _STAGES
        }
        sample_range = _sample_range(row.get("doa_range"))
        scores = row.get("normalized_scores")
        normalized = None
        if isinstance(scores, Sequence) and not isinstance(scores, (str, bytes)) and len(scores) == 360:
            converted = tuple(_optional_float(item) for item in scores)
            normalized = tuple(float(item) for item in converted) if all(item is not None for item in converted) else None
        return WindowObservation(
            key=key,
            schema_version=str(row.get("schema_version", "unknown")),
            sample_range=sample_range,
            stages=stages,
            candidates=tuple(_public_copy(_mapping(item)) for item in _sequence(row.get("candidates"))),
            active_tracks=tuple(_public_copy(_mapping(item)) for item in _sequence(row.get("active_tracks"))),
            detections=tuple(_public_copy(_mapping(item)) for item in _sequence(row.get("detections"))),
            enhanced_assets=tuple(_public_copy(_mapping(item)) for item in _sequence(row.get("enhanced_audio"))),
            gate=_optional_mapping(row.get("gate_decision")),
            model_order=_optional_mapping(row.get("model_order")),
            music_diagnostics=_optional_mapping(row.get("music_diagnostics")),
            normalized_scores=normalized,
            terminal_reason=reason,
            raw_public=_public_copy(row),
        )

    @staticmethod
    def _window_anomalies(windows: Sequence[WindowObservation]) -> list[Anomaly]:
        result: list[Anomaly] = []
        previous: WindowObservation | None = None
        for window in windows:
            if previous is not None and window.key.stream_epoch == previous.key.stream_epoch:
                if window.key.decision_sample <= previous.key.decision_sample:
                    result.append(Anomaly("timeline", "decision_sample未严格递增", window.key))
                elif window.key.decision_sample - previous.key.decision_sample > 960:
                    result.append(Anomaly("sample_gap", "权威sample时间线存在缺口", window.key))
            for stage, observation in window.stages.items():
                if observation.state in {StageState.DROPPED, StageState.TIMED_OUT, StageState.FAILED, StageState.CANCELLED}:
                    result.append(Anomaly(observation.state.value, f"{stage.upper()} {observation.state.value}", window.key))
            previous = window
        return result

    @staticmethod
    def _tracks(windows: Sequence[WindowObservation]) -> tuple[tuple[TrackObservation, ...], list[Anomaly]]:
        rows: list[TrackObservation] = []
        anomalies: list[Anomaly] = []
        for window in windows:
            detections = {item.get("track_id"): item for item in window.detections if type(item.get("track_id")) is int}
            enhanced = {item.get("track_id"): item for item in window.enhanced_assets if type(item.get("track_id")) is int}
            candidates = window.active_tracks or window.candidates
            candidate_ids = tuple(item.get("track_id") for item in window.candidates if type(item.get("track_id")) is int)
            detection_ids = tuple(item.get("track_id") for item in window.detections if type(item.get("track_id")) is int)
            if candidate_ids and detection_ids and candidate_ids != detection_ids:
                anomalies.append(Anomaly("id_alignment", "L2与L4 track_id集合或顺序不一致", window.key))
            for item in candidates:
                track_id = item.get("track_id")
                if type(track_id) is not int or track_id <= 0:
                    continue
                detection = detections.get(track_id, {})
                rows.append(TrackObservation(
                    window.key.session_id, window.key.stream_epoch, track_id,
                    window.key.window_id, window.key.decision_sample,
                    _optional_float(item.get("measured_theta_deg")),
                    _optional_float(item.get("theta_deg")),
                    _text(item.get("track_state")),
                    item.get("is_observed") if type(item.get("is_observed")) is bool else None,
                    item.get("is_new_track") if type(item.get("is_new_track")) is bool else None,
                    _optional_float(detection.get("probability")),
                    detection.get("is_voice") if type(detection.get("is_voice")) is bool else None,
                    enhanced.get(track_id),
                ))
        rows.sort(key=lambda item: (item.stream_epoch, item.track_id, item.decision_sample, item.window_id))
        return tuple(rows), anomalies

    def track_assets(self, session_id: str, stream_epoch: int, track_id: int) -> tuple[Mapping[str, Any], ...]:
        if not self.capabilities.track_audio:
            raise NotImplementedError("public capability unavailable: track_audio")
        rows = self._call("track_audio", session_id, int(stream_epoch), int(track_id))
        return tuple(dict(item) for item in rows)

    def session_assets(self, session_id: str, kind: str) -> tuple[Mapping[str, Any], ...]:
        if not self.capabilities.session_audio:
            raise NotImplementedError("public capability unavailable: session_audio")
        rows = self._call("session_audio", session_id, kind)
        return tuple(dict(item) for item in rows)


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else ()


def _optional_mapping(value: object) -> Mapping[str, Any] | None:
    return _public_copy(value) if isinstance(value, Mapping) else None


def _sample_range(value: object) -> tuple[int, int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return None
    start, end = value
    if type(start) is not int or type(end) is not int or start < 0 or end <= start:
        return None
    return int(start), int(end)


def _duration_samples(row: Mapping[str, Any], metadata: Mapping[str, Any]) -> int | None:
    direct = row.get("duration_samples", metadata.get("duration_samples"))
    if type(direct) is int and direct >= 0:
        return int(direct)
    ranges = []
    for chunk in _sequence(metadata.get("chunks")):
        item = _mapping(chunk)
        start, end = item.get("start_sample"), item.get("end_sample")
        if type(start) is int and type(end) is int and end > start >= 0:
            ranges.append((int(item.get("stream_epoch", 0)), start, end))
    if not ranges:
        return None
    by_epoch: dict[int, list[tuple[int, int]]] = {}
    for epoch, start, end in ranges:
        by_epoch.setdefault(epoch, []).append((start, end))
    return sum(max(end for _, end in values) - min(start for start, _ in values) for values in by_epoch.values())
