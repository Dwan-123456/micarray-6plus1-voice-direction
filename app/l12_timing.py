from __future__ import annotations

from collections import OrderedDict, deque
from math import isfinite
from threading import Lock
from time import monotonic

import numpy as np


class L12SegmentTimingTelemetry:
    """Bounded, audio-free timing telemetry for one live L1/L2 stream."""

    _RETAINED_WINDOWS = 30_000
    _IMCRA_ENDPOINTS = 512

    def __init__(self) -> None:
        self._lock = Lock()
        self._stream: tuple[str, int] | None = None
        self._imcra_ms: OrderedDict[int, float] = OrderedDict()
        self._records: dict[str, deque[tuple[float | None, float | None, float | None, float]]] = {
            "open": deque(maxlen=self._RETAINED_WINDOWS),
            "closed": deque(maxlen=self._RETAINED_WINDOWS),
        }
        self._total_counts = {"open": 0, "closed": 0}
        self._latest: dict[str, object] | None = None
        self._cached_snapshot: dict[str, object] | None = None
        self._cached_at = 0.0

    def _ensure_stream(self, session_id: str, stream_epoch: int) -> None:
        stream = (session_id, stream_epoch)
        if stream == self._stream:
            return
        self._stream = stream
        self._imcra_ms.clear()
        for records in self._records.values():
            records.clear()
        self._total_counts = {"open": 0, "closed": 0}
        self._latest = None
        self._cached_snapshot = None
        self._cached_at = 0.0

    def record_imcra(
        self,
        session_id: str,
        stream_epoch: int,
        endpoints: tuple[int, ...],
        elapsed_ms: float,
    ) -> None:
        if not endpoints:
            return
        value = float(elapsed_ms) / len(endpoints)
        if not isfinite(value) or value < 0.0:
            return
        with self._lock:
            self._ensure_stream(session_id, stream_epoch)
            for endpoint in endpoints:
                self._imcra_ms[int(endpoint)] = value
                self._imcra_ms.move_to_end(int(endpoint))
            while len(self._imcra_ms) > self._IMCRA_ENDPOINTS:
                self._imcra_ms.popitem(last=False)

    def record_l2(
        self,
        session_id: str,
        stream_epoch: int,
        decision_sample: int,
        *,
        gate_state: str,
        music_ms: float | None,
        id_tracking_ms: float | None,
        total_ms: float,
    ) -> None:
        state = str(gate_state).casefold()
        if state not in self._records:
            return
        values = (music_ms, id_tracking_ms, total_ms)
        if any(value is not None and (not isfinite(float(value)) or float(value) < 0.0) for value in values):
            return
        with self._lock:
            self._ensure_stream(session_id, stream_epoch)
            imcra_ms = self._imcra_ms.pop(int(decision_sample), None)
            record = (
                imcra_ms,
                None if music_ms is None else float(music_ms),
                None if id_tracking_ms is None else float(id_tracking_ms),
                float(total_ms),
            )
            self._records[state].append(record)
            self._total_counts[state] += 1
            self._latest = {
                "gate_state": state,
                "decision_sample": int(decision_sample),
                "imcra_ms": imcra_ms,
                "music_ms": record[1],
                "id_tracking_ms": record[2],
                "l2_total_ms": record[3],
            }

    @staticmethod
    def _metric(records: tuple[tuple[float | None, ...], ...], index: int) -> dict[str, object]:
        values = np.asarray(
            [item[index] for item in records if item[index] is not None], dtype=np.float64
        )
        if values.size == 0:
            return {"count": 0, "avg_ms": None, "p95_ms": None, "max_ms": None}
        return {
            "count": int(values.size),
            "avg_ms": float(np.mean(values)),
            "p95_ms": float(np.percentile(values, 95)),
            "max_ms": float(np.max(values)),
        }

    def snapshot(self) -> dict[str, object]:
        now = monotonic()
        with self._lock:
            if self._cached_snapshot is not None and now - self._cached_at < 0.5:
                return self._cached_snapshot
            stream = self._stream
            latest = None if self._latest is None else dict(self._latest)
            total_counts = dict(self._total_counts)
            copied = {state: tuple(records) for state, records in self._records.items()}
        # Percentiles can be noticeably more expensive than appending one
        # timing tuple. Compute them after releasing the producer lock so UI
        # diagnostics can never stall the 20 ms L1/L2 workers.
        summaries: dict[str, object] = {}
        for state, records in copied.items():
            summaries[state] = {
                "total_count": total_counts[state],
                "retained_count": len(records),
                "imcra": self._metric(records, 0),
                "music": self._metric(records, 1),
                "id_tracking": self._metric(records, 2),
                "l2_total": self._metric(records, 3),
            }
        snapshot = {
            "session_id": None if stream is None else stream[0],
            "stream_epoch": None if stream is None else stream[1],
            "latest": latest,
            "by_gate": summaries,
            "retained_window_limit": self._RETAINED_WINDOWS,
        }
        with self._lock:
            # A stream boundary invalidates the copied data. Do not publish a
            # previous stream after reset, even if percentile calculation
            # overlapped the first new record.
            stream_changed = self._stream != stream
            if not stream_changed:
                self._cached_snapshot = snapshot
                self._cached_at = now
        return self.snapshot() if stream_changed else snapshot
