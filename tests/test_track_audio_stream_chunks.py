from types import SimpleNamespace
from threading import Event, Thread

import numpy as np
import pytest

from layer5_voice_classifier import InputGainCompensationSettings
from track_audio_stream import TrackAudioStreamHub, TrackAudioWindow


_HOP = 960


def _identity(decision_sample: int) -> tuple[str, int, int, int]:
    return ("session", 0, decision_sample // _HOP, decision_sample)


def _track(track_id: int, *, state: str = "confirmed") -> SimpleNamespace:
    return SimpleNamespace(
        track_id=track_id,
        track_state=state,
        theta_deg=float(20 + track_id),
    )


def _window(
    decision_sample: int,
    track_id: int,
    *,
    level: float,
    processing_mode: str = "optimized",
) -> TrackAudioWindow:
    waveform = np.full(4 * _HOP, level, dtype=np.float32)
    return TrackAudioWindow(
        "session",
        0,
        decision_sample // _HOP,
        decision_sample,
        track_id,
        float(20 + track_id),
        waveform,
        (0.9,) * 4,
        processing_mode,
    )


def _observe(
    hub: TrackAudioStreamHub,
    decision_sample: int,
    tracks: tuple[SimpleNamespace, ...],
    *,
    processing_mode: str = "optimized",
    direction_count: int | None = None,
) -> None:
    hub.observe_l2(
        identity=_identity(decision_sample),
        active_tracks=tracks,
        processing_mode=processing_mode,
        l2_direction_count=len(tracks) if direction_count is None else direction_count,
    )


def test_streaming_chunks_are_independent_contiguous_and_never_repeated() -> None:
    hub = TrackAudioStreamHub(
        InputGainCompensationSettings(enabled=False), context_ms=160,
    )
    tracks = (_track(7), _track(8))
    decisions = tuple(7_680 + index * _HOP for index in range(4))
    for index, decision in enumerate(decisions):
        _observe(hub, decision, tracks)
        hub.process(
            (
                _window(decision, 7, level=0.01 + index * 0.001),
                _window(decision, 8, level=0.02 + index * 0.001),
            ),
            active_track_ids=(7, 8),
            identity=_identity(decision),
            l2_direction_count=2,
        )

    key7 = ("session", 0, 7)
    key8 = ("session", 0, 8)
    first = hub.take_streaming_chunks(
        chunk_samples=2 * _HOP, ready_track_keys={key7},
    )
    assert tuple(item.track_id for item in first) == (7, 7)
    assert tuple((item.start_sample, item.end_sample) for item in first) == (
        (5_760, 7_680),
        (7_680, 9_600),
    )
    assert hub.take_streaming_chunks(
        chunk_samples=2 * _HOP, ready_track_keys={key7},
    ) == ()

    second = hub.take_streaming_chunks(
        chunk_samples=2 * _HOP, ready_track_keys={key7, key8},
    )
    assert tuple(item.track_id for item in second) == (8, 8)
    assert tuple((item.start_sample, item.end_sample) for item in second) == (
        (5_760, 7_680),
        (7_680, 9_600),
    )

    sealed = {item.track_id: item for item in hub.seal()}
    np.testing.assert_array_equal(
        np.concatenate([item.waveform for item in first]), sealed[7].waveform,
    )
    np.testing.assert_array_equal(
        np.concatenate([item.waveform for item in second]), sealed[8].waveform,
    )


def test_ready_track_gate_waits_for_backfill_before_creating_the_cursor() -> None:
    hub = TrackAudioStreamHub(
        InputGainCompensationSettings(enabled=False), context_ms=160,
    )
    track = _track(7)
    key = ("session", 0, 7)
    _observe(hub, 7_680, (track,))
    hub.process(
        (_window(7_680, 7, level=0.03),),
        active_track_ids=(7,),
        identity=_identity(7_680),
        l2_direction_count=1,
    )

    assert hub.take_streaming_chunks(
        chunk_samples=3 * _HOP, ready_track_keys=set(),
    ) == ()
    assert key not in hub._streaming_cursors

    inserted = hub.insert_backfill(
        (
            _window(5_760, 7, level=0.01),
            _window(6_720, 7, level=0.02),
        ),
        l2_direction_count=1,
    )
    chunk = hub.take_streaming_chunks(
        chunk_samples=3 * _HOP, ready_track_keys={key},
    )

    assert len(inserted) == 2
    assert len(chunk) == 1
    assert (chunk[0].start_sample, chunk[0].end_sample) == (3_840, 6_720)
    np.testing.assert_array_equal(
        chunk[0].waveform,
        np.concatenate([item.waveform for item in (*inserted,)] + [
            hub._archive[key][-1].waveform,
        ]),
    )
    assert hub.take_streaming_chunks(
        chunk_samples=3 * _HOP, ready_track_keys={key},
    ) == ()


def test_streaming_claim_advances_only_after_acceptance_and_retries_exactly() -> None:
    hub = TrackAudioStreamHub(
        InputGainCompensationSettings(enabled=False), context_ms=160,
    )
    track = _track(7)
    key = ("session", 0, 7)
    for index, decision in enumerate((7_680, 8_640)):
        _observe(hub, decision, (track,))
        hub.process(
            (_window(decision, 7, level=0.01 + index * 0.01),),
            active_track_ids=(7,),
            identity=_identity(decision),
            l2_direction_count=1,
        )

    first = hub.claim_streaming_chunks(
        chunk_samples=2 * _HOP,
        ready_track_keys={key},
        max_chunks=1,
    )
    assert len(first) == 1
    assert key not in hub._streaming_cursors
    assert hub.claim_streaming_chunks(
        chunk_samples=2 * _HOP,
        ready_track_keys={key},
        max_chunks=1,
    ) == ()

    hub.resolve_streaming_chunk(first[0], accepted=False)
    retried = hub.claim_streaming_chunks(
        chunk_samples=2 * _HOP,
        ready_track_keys={key},
        max_chunks=1,
    )
    assert len(retried) == 1
    assert retried[0].start_sample == first[0].start_sample
    assert retried[0].sha256 == first[0].sha256

    hub.resolve_streaming_chunk(retried[0], accepted=True)
    assert hub._streaming_cursors[key] == retried[0].end_sample
    assert hub.claim_streaming_chunks(
        chunk_samples=2 * _HOP,
        ready_track_keys={key},
        max_chunks=1,
    ) == ()


def test_streaming_chunk_materialization_does_not_hold_the_hub_write_lock(
    monkeypatch,
) -> None:
    hub = TrackAudioStreamHub(
        InputGainCompensationSettings(enabled=False), context_ms=160,
    )
    track = _track(7)
    key = ("session", 0, 7)
    for index, decision in enumerate((7_680, 8_640)):
        _observe(hub, decision, (track,))
        hub.process(
            (_window(decision, 7, level=0.01 + index * 0.01),),
            active_track_ids=(7,),
            identity=_identity(decision),
            l2_direction_count=1,
        )

    build_started = Event()
    release_build = Event()
    writer_done = Event()
    claimed: list[object] = []
    original = TrackAudioStreamHub._build_streaming_chunk

    def slow_build(plan):
        build_started.set()
        assert release_build.wait(2.0)
        return original(plan)

    monkeypatch.setattr(
        TrackAudioStreamHub,
        "_build_streaming_chunk",
        staticmethod(slow_build),
    )
    claimant = Thread(
        target=lambda: claimed.extend(hub.claim_streaming_chunks(
            chunk_samples=2 * _HOP,
            ready_track_keys={key},
            max_chunks=1,
        )),
    )
    claimant.start()
    assert build_started.wait(1.0)

    def observe_next() -> None:
        _observe(hub, 9_600, (track,))
        writer_done.set()

    writer = Thread(target=observe_next)
    writer.start()
    assert writer_done.wait(0.5)
    release_build.set()
    claimant.join(2.0)
    writer.join(2.0)

    assert len(claimed) == 1
    hub.resolve_streaming_chunk(claimed[0], accepted=True)


def test_streaming_flush_uses_seal_gap_rules_and_emits_a_complete_hop_tail() -> None:
    hub = TrackAudioStreamHub(
        InputGainCompensationSettings(enabled=False), context_ms=160,
    )
    track = _track(7)
    decisions = tuple(7_680 + index * _HOP for index in range(5))
    for decision in decisions:
        _observe(hub, decision, (track,))
    live = hub.process(
        (_window(decisions[0], 7, level=0.05),),
        active_track_ids=(7,),
        identity=_identity(decisions[0]),
        l2_direction_count=1,
    )

    assert hub.take_streaming_chunks(chunk_samples=2 * _HOP) == ()
    chunks = hub.take_streaming_chunks(chunk_samples=2 * _HOP, flush=True)

    assert tuple(len(item.waveform) for item in chunks) == (2 * _HOP, 2 * _HOP, _HOP)
    assert tuple((item.start_sample, item.end_sample) for item in chunks) == (
        (5_760, 7_680),
        (7_680, 9_600),
        (9_600, 10_560),
    )
    waveform = np.concatenate([item.waveform for item in chunks])
    np.testing.assert_array_equal(waveform[:_HOP], live.emitted_hops[0].waveform)
    np.testing.assert_array_equal(waveform[_HOP:], 0.0)
    assert tuple(
        value
        for chunk in chunks
        for value in chunk.l2_direction_counts
    ) == tuple((decision - _HOP, 1) for decision in decisions)
    assert hub.take_streaming_chunks(chunk_samples=2 * _HOP, flush=True) == ()


def test_streaming_cursor_is_cleared_by_mode_reset_reset_and_seal_discard() -> None:
    hub = TrackAudioStreamHub(
        InputGainCompensationSettings(enabled=False), context_ms=160,
    )
    key = ("session", 0, 7)
    track = _track(7)
    _observe(hub, 7_680, (track,))
    hub.process(
        (_window(7_680, 7, level=0.01),),
        active_track_ids=(7,), identity=_identity(7_680), l2_direction_count=1,
    )
    assert len(hub.take_streaming_chunks(chunk_samples=_HOP)) == 1
    assert key in hub._streaming_cursors

    _observe(hub, 8_640, (track,), processing_mode="ds_baseline")
    assert key not in hub._streaming_cursors
    hub.process(
        (_window(8_640, 7, level=0.02, processing_mode="ds_baseline"),),
        active_track_ids=(7,), identity=_identity(8_640), l2_direction_count=1,
    )
    replacement = hub.take_streaming_chunks(chunk_samples=_HOP)
    assert len(replacement) == 1
    assert key in hub._streaming_cursors

    hub.seal(allowed_track_keys=set())
    assert key not in hub._streaming_cursors

    hub.reset()
    assert hub._streaming_cursors == {}


@pytest.mark.parametrize("chunk_samples", [0, -_HOP, 1, _HOP + 1, True])
def test_streaming_chunk_size_must_be_a_positive_complete_hop(chunk_samples: int) -> None:
    hub = TrackAudioStreamHub(InputGainCompensationSettings(enabled=False))
    with pytest.raises(ValueError, match="positive 20 ms multiple"):
        hub.take_streaming_chunks(chunk_samples=chunk_samples)
