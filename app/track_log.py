from __future__ import annotations

import queue
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class _TrackLogBatch:
    session_id: str
    stream_epoch: int
    decision_sample: int
    tracks: tuple[tuple[int, str, int, int, float], ...]
    reset_history: bool = False


class TrackHistoryLogger:
    """Bounded, best-effort track diagnostics isolated from the L2 worker."""

    def __init__(
        self,
        path: str | Path,
        *,
        sample_rate: int = 48_000,
        submit_interval_samples: int = 48_000,
        trajectory_interval_samples: int = 240_000,
        max_tracks: int = 64,
        max_points_per_track: int = 240,
    ) -> None:
        self.path = Path(path)
        self.sample_rate = int(sample_rate)
        self.submit_interval_samples = int(submit_interval_samples)
        self.trajectory_interval_samples = int(trajectory_interval_samples)
        self.max_tracks = int(max_tracks)
        self.max_points_per_track = int(max_points_per_track)
        self._queue: queue.Queue[_TrackLogBatch] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_submitted_sample = -self.submit_interval_samples
        self._last_stream_key: tuple[str, int] | None = None
        self._consumed_stream_key: tuple[str, int] | None = None
        self._history: OrderedDict[int, dict[str, object]] = OrderedDict()
        self._status_lock = threading.Lock()
        self._last_error: str | None = None

    @property
    def last_error(self) -> str | None:
        with self._status_lock:
            return self._last_error

    @property
    def history_size(self) -> int:
        return len(self._history)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._last_submitted_sample = -self.submit_interval_samples
        self._last_stream_key = None
        self._consumed_stream_key = None
        self._history.clear()
        with self._status_lock:
            self._last_error = None
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._thread = threading.Thread(
            target=self._run,
            name="l2-track-log",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(max(0.0, float(timeout)))
        if thread is None or not thread.is_alive():
            self._thread = None

    def submit(
        self,
        tracks: tuple[object, ...],
        decision_sample: int,
        *,
        session_id: str,
        stream_epoch: int,
    ) -> None:
        stream_key = (str(session_id), int(stream_epoch))
        reset_history = (
            self._last_stream_key is not None and stream_key != self._last_stream_key
        ) or decision_sample < self._last_submitted_sample
        if reset_history:
            self._last_submitted_sample = -self.submit_interval_samples
        self._last_stream_key = stream_key
        if decision_sample - self._last_submitted_sample < self.submit_interval_samples:
            return
        self._last_submitted_sample = int(decision_sample)
        batch = _TrackLogBatch(
            stream_key[0],
            stream_key[1],
            int(decision_sample),
            tuple(
                (
                    int(track.track_id),
                    str(track.track_state),
                    int(track.first_seen_sample),
                    int(track.last_observed_sample),
                    float(track.theta_deg),
                )
                for track in tracks
            ),
            reset_history,
        )
        try:
            self._queue.put_nowait(batch)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(batch)

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                batch = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._consume(batch)
                self._write_snapshot()
                with self._status_lock:
                    self._last_error = None
            except Exception as exc:  # Diagnostic I/O must never stop realtime L2.
                with self._status_lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"

    def _consume(self, batch: _TrackLogBatch) -> None:
        batch_stream_key = (
            (str(batch.session_id), int(batch.stream_epoch))
            if hasattr(batch, "session_id") and hasattr(batch, "stream_epoch")
            else None
        )
        stream_changed = (
            batch_stream_key is not None
            and self._consumed_stream_key is not None
            and batch_stream_key != self._consumed_stream_key
        )
        if getattr(batch, "reset_history", False) or stream_changed:
            self._history.clear()
        if batch_stream_key is not None:
            self._consumed_stream_key = batch_stream_key
        for track_id, state, first, last, theta in batch.tracks:
            entry = self._history.get(track_id)
            if entry is None:
                entry = {
                    "first": first,
                    "last": last,
                    "state": state,
                    "trajectory": deque(maxlen=self.max_points_per_track),
                }
                self._history[track_id] = entry
            entry["last"] = max(int(entry["last"]), last)
            entry["state"] = state
            trajectory = entry["trajectory"]
            assert isinstance(trajectory, deque)
            if (
                not trajectory
                or batch.decision_sample - int(trajectory[-1][0])
                >= self.trajectory_interval_samples
            ):
                trajectory.append((batch.decision_sample, round(theta, 1)))
            self._history.move_to_end(track_id)
        while len(self._history) > self.max_tracks:
            self._history.popitem(last=False)

    def _write_snapshot(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "track_id\tstate\tfirst_sample\tlast_observed_sample\tduration_s\t"
            "trajectory(sample:deg)"
        ]
        for track_id, entry in sorted(self._history.items()):
            duration = (int(entry["last"]) - int(entry["first"])) / self.sample_rate
            trajectory = ",".join(
                f"{sample}:{angle:.1f}" for sample, angle in entry["trajectory"]
            )
            lines.append(
                f"{track_id}\t{entry['state']}\t{entry['first']}\t{entry['last']}\t"
                f"{duration:.3f}\t{trajectory}"
            )
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary.replace(self.path)
