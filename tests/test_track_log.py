from __future__ import annotations

from types import SimpleNamespace

from app.track_log import TrackHistoryLogger


def _track(track_id: int, sample: int) -> object:
    return SimpleNamespace(
        track_id=track_id,
        track_state="confirmed",
        first_seen_sample=max(0, sample - 48_000),
        last_observed_sample=sample,
        theta_deg=float(track_id),
    )


def test_track_logger_bounds_history_and_writes_latest_snapshot(tmp_path) -> None:
    logger = TrackHistoryLogger(
        tmp_path / "tracks.txt",
        submit_interval_samples=1,
        trajectory_interval_samples=1,
        max_tracks=2,
        max_points_per_track=2,
    )
    logger._consume(SimpleNamespace(decision_sample=1, tracks=((1, "confirmed", 0, 1, 1.0),)))
    logger._consume(SimpleNamespace(decision_sample=2, tracks=((2, "confirmed", 0, 2, 2.0),)))
    logger._consume(SimpleNamespace(decision_sample=3, tracks=((3, "confirmed", 0, 3, 3.0),)))
    logger._write_snapshot()

    text = (tmp_path / "tracks.txt").read_text(encoding="utf-8")
    assert logger.history_size == 2
    assert "\n1\t" not in text
    assert "\n2\t" in text and "\n3\t" in text


def test_track_logger_io_error_is_recorded_without_raising(tmp_path) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("file", encoding="utf-8")
    logger = TrackHistoryLogger(blocked / "tracks.txt", submit_interval_samples=1)
    logger.start()
    logger.submit(
        (_track(1, 48_000),),
        48_000,
        session_id="session-a",
        stream_epoch=0,
    )
    logger.stop(timeout=2.0)

    assert logger.last_error is not None


def test_track_logger_resets_history_when_epoch_changes_at_same_sample(tmp_path) -> None:
    logger = TrackHistoryLogger(tmp_path / "tracks.txt", submit_interval_samples=1)
    logger.submit(
        (_track(1, 7_680),),
        7_680,
        session_id="session-a",
        stream_epoch=0,
    )
    first = logger._queue.get_nowait()
    logger._consume(first)
    logger.submit(
        (_track(2, 7_680),),
        7_680,
        session_id="session-a",
        stream_epoch=1,
    )
    second = logger._queue.get_nowait()
    assert second.reset_history
    logger._consume(second)

    assert logger.history_size == 1


def test_consumer_detects_epoch_when_reset_batch_was_replaced(tmp_path) -> None:
    logger = TrackHistoryLogger(tmp_path / "tracks.txt", submit_interval_samples=1)
    logger.submit(
        (_track(1, 7_680),),
        7_680,
        session_id="session-a",
        stream_epoch=0,
    )
    logger._consume(logger._queue.get_nowait())
    logger.submit(
        (_track(2, 7_680),),
        7_680,
        session_id="session-a",
        stream_epoch=1,
    )
    logger.submit(
        (_track(2, 7_681),),
        7_681,
        session_id="session-a",
        stream_epoch=1,
    )
    replacement = logger._queue.get_nowait()

    assert not replacement.reset_history
    logger._consume(replacement)
    assert logger.history_size == 1
