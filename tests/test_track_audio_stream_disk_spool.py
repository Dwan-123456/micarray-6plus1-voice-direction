from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from layer5_voice_classifier import InputGainCompensationSettings
from track_audio_stream import TrackAudioStreamHub, TrackAudioWindow


_HOP = 960
_SAMPLE_RATE = 48_000


def _identity(decision_sample: int) -> tuple[str, int, int, int]:
    return ("session", 0, decision_sample // _HOP, decision_sample)


def _track(
    track_id: int,
    *,
    state: str = "confirmed",
    theta_deg: float | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        track_id=track_id,
        track_state=state,
        theta_deg=float(20 + track_id if theta_deg is None else theta_deg),
    )


def _window(
    decision_sample: int,
    track_id: int,
    *,
    level: float,
    processing_mode: str = "optimized",
) -> TrackAudioWindow:
    return TrackAudioWindow(
        "session",
        0,
        decision_sample // _HOP,
        decision_sample,
        track_id,
        float(20 + track_id),
        np.full(4 * _HOP, level, dtype=np.float32),
        (0.9,) * 4,
        processing_mode,
    )


def _observe(
    hub: TrackAudioStreamHub,
    decision_sample: int,
    tracks: tuple[SimpleNamespace, ...],
    *,
    processing_mode: str = "optimized",
    direction_count: int = 1,
) -> None:
    hub.observe_l2(
        identity=_identity(decision_sample),
        active_tracks=tracks,
        processing_mode=processing_mode,
        l2_direction_count=direction_count,
    )


def test_90k_hop_logical_session_keeps_only_window_audio_in_python() -> None:
    hub = TrackAudioStreamHub(
        InputGainCompensationSettings(enabled=False),
        context_ms=60,
    )
    key = ("session", 0, 7)
    processed_hops = 12
    for index in range(processed_hops):
        decision = 7_680 + index * _HOP
        hub.process(
            (_window(decision, 7, level=0.01 + index * 0.001),),
            active_track_ids=(7,),
            identity=_identity(decision),
            l2_direction_count=1,
        )
        assert len(hub._archive[key]) <= hub.max_hops

    state = hub._tracks[key]
    archive = hub._archive[key]
    assert len(archive) == len(state.audio) == hub.max_hops
    archive_ids = {id(item.waveform) for item in archive}
    state_ids = {id(item) for item in state.audio}
    assert archive_ids == state_ids
    retained = {id(value): value for value in state.audio}
    if state.future_audio is not None:
        retained[id(state.future_audio)] = state.future_audio
    assert sum(value.nbytes for value in retained.values()) <= (
        hub.max_hops + 1
    ) * _HOP * np.dtype(np.float32).itemsize

    # A sparse extension exercises the exact 30-minute/90,000-hop logical
    # dimensions without allocating or constructing 90,000 Python waveforms.
    logical_hops = 90_000
    audio = hub._archive_audio[key]
    directions = hub._direction_counts[key]
    presence = hub._audio_presence[key]
    audio.ensure_length(logical_hops * _HOP)
    directions.set(logical_hops - 1, 1)
    presence.set(logical_hops - 1, 1)

    float_bytes = np.dtype(np.float32).itemsize
    assert audio.resident_bytes == 0
    assert audio.disk_bytes == logical_hops * _HOP * float_bytes
    assert audio.path.stat().st_size == audio.disk_bytes
    assert directions.disk_bytes == presence.disk_bytes == logical_hops
    assert (
        audio.disk_bytes + directions.disk_bytes + presence.disk_bytes
        == logical_hops * (_HOP * float_bytes + 2)
    )
    assert len(hub._archive[key]) == hub.max_hops
    hub.reset()


def test_claimed_waveform_and_sha_are_immutable_after_backfill_and_live_mutation() -> None:
    hub = TrackAudioStreamHub(
        InputGainCompensationSettings(enabled=False),
        context_ms=160,
    )
    key = ("session", 0, 7)
    track = _track(7)
    _observe(hub, 7_680, (track,))
    hub.process(
        (_window(7_680, 7, level=0.3),),
        active_track_ids=(7,),
        identity=_identity(7_680),
        l2_direction_count=1,
    )
    claimed = hub.claim_streaming_chunks(
        chunk_samples=_HOP,
        ready_track_keys={key},
        max_chunks=1,
    )[0]
    published = claimed.waveform.tobytes()
    published_sha = claimed.sha256

    inserted = hub.insert_backfill(
        (
            _window(5_760, 7, level=0.1),
            _window(6_720, 7, level=0.2),
        ),
        l2_direction_count=1,
    )
    assert len(inserted) == 2
    active_spool = hub._archive_audio[key]
    relative_start = claimed.start_sample - hub._archive_origins[key]
    active_spool.write_at(
        relative_start,
        np.full(_HOP, -0.75, dtype=np.float32),
    )

    assert claimed.waveform.tobytes() == published
    assert claimed.sha256 == published_sha
    assert hashlib.sha256(claimed.waveform.tobytes()).hexdigest() == published_sha
    hub.resolve_streaming_chunk(claimed, accepted=True)
    hub.reset()


def test_mode_switch_keeps_the_first_new_mode_direction_count() -> None:
    hub = TrackAudioStreamHub(
        InputGainCompensationSettings(enabled=False),
        context_ms=160,
    )
    key = ("session", 0, 7)
    track = _track(7)
    _observe(hub, 7_680, (track,), direction_count=1)
    hub.process(
        (_window(7_680, 7, level=0.1),),
        active_track_ids=(7,),
        identity=_identity(7_680),
        l2_direction_count=1,
    )

    _observe(
        hub,
        8_640,
        (track,),
        processing_mode="ds_baseline",
        direction_count=3,
    )
    new_batch = hub.process(
        (_window(8_640, 7, level=0.8, processing_mode="ds_baseline"),),
        active_track_ids=(7,),
        identity=_identity(8_640),
        # The authoritative L2 count written by observe_l2 must win this
        # deliberately different compatibility fallback.
        l2_direction_count=1,
    )

    assert hub._direction_counts[key].read_range(0, 1) == bytes((3,))
    sealed = hub.seal()[0]
    assert tuple(sealed.l2_direction_counts) == ((sealed.end_sample, 3),)
    np.testing.assert_array_equal(
        sealed.waveform,
        new_batch.emitted_hops[0].waveform,
    )
    hub.reset()


def test_late_track_spool_is_relative_to_its_own_origin() -> None:
    hub = TrackAudioStreamHub(
        InputGainCompensationSettings(enabled=False),
        context_ms=160,
    )
    first_track = _track(7)
    _observe(hub, 7_680, (first_track,))
    hub.process(
        (_window(7_680, 7, level=0.1),),
        active_track_ids=(7,),
        identity=_identity(7_680),
        l2_direction_count=1,
    )

    late_decision = 90_000 * _HOP + 7_680
    late_track = _track(8)
    late_key = ("session", 0, 8)
    _observe(hub, late_decision, (late_track,))
    hub.process(
        (_window(late_decision, 8, level=0.8),),
        active_track_ids=(8,),
        identity=_identity(late_decision),
        l2_direction_count=1,
    )

    spool = hub._archive_audio[late_key]
    assert hub._archive_origins[late_key] == late_decision - 2 * _HOP
    assert spool.length_samples == _HOP
    assert spool.disk_bytes == spool.path.stat().st_size == (
        _HOP * np.dtype(np.float32).itemsize
    )
    assert hub._direction_counts[late_key].disk_bytes == 1
    assert hub._audio_presence[late_key].disk_bytes == 1
    late_output = next(item for item in hub.seal() if item.track_id == 8)
    assert late_output.start_sample == late_decision - 2 * _HOP
    assert len(late_output.waveform) == _HOP
    hub.reset()


def test_seal_is_idempotent_even_if_a_later_allow_list_differs() -> None:
    hub = TrackAudioStreamHub(
        InputGainCompensationSettings(enabled=False),
        context_ms=160,
    )
    key = ("session", 0, 7)
    track = _track(7)
    for index, decision in enumerate((7_680, 8_640)):
        _observe(hub, decision, (track,), direction_count=index + 1)
        hub.process(
            (_window(decision, 7, level=0.1 + index * 0.1),),
            active_track_ids=(7,),
            identity=_identity(decision),
            l2_direction_count=index + 1,
        )

    first = hub.seal(allowed_track_keys={key})
    active_views = hub._archive_store.active_views
    waveform_sha = hashlib.sha256(first[0].waveform.tobytes()).hexdigest()
    second = hub.seal(allowed_track_keys=set())

    assert second is first
    assert hub.sealed_tracks is first
    assert second[0] is first[0]
    assert second[0].sha256 == waveform_sha
    assert hub._archive_store.active_views == active_views
    hub.reset()


@pytest.mark.parametrize("chunk_seconds", [3, 4, 15])
def test_configured_chunk_durations_preserve_one_contiguous_time_axis(
    chunk_seconds: int,
) -> None:
    hub = TrackAudioStreamHub(
        InputGainCompensationSettings(enabled=False),
        context_ms=160,
    )
    key = ("session", 0, 7)
    track = _track(7)
    first_decision = 7_680
    _observe(hub, first_decision, (track,), direction_count=1)
    timeline = hub._l2_timelines[key]
    start_sample = timeline.first_start_sample
    chunk_hops = chunk_seconds * _SAMPLE_RATE // _HOP
    tail_hops = 3
    total_hops = chunk_hops + tail_hops
    end_sample = start_sample + total_hops * _HOP
    timeline.end_sample = end_sample
    timeline.observed_through_sample = end_sample + _HOP

    levels = np.arange(total_hops, dtype=np.float32) / max(total_hops, 1)
    waveform = np.repeat(levels, _HOP)
    direction_values = bytes((index % 3) + 1 for index in range(total_hops))
    hub._audio_spool(key).write_at(0, waveform)
    hub._direction_spool(key).write_range(0, direction_values)
    hub._presence_spool(key).write_range(0, bytes((1,)) * total_hops)
    hub._emitted_ends[key] = end_sample

    chunks = hub.take_streaming_chunks(
        chunk_samples=chunk_seconds * _SAMPLE_RATE,
        ready_track_keys={key},
        flush=True,
    )

    assert tuple(len(item.waveform) for item in chunks) == (
        chunk_seconds * _SAMPLE_RATE,
        tail_hops * _HOP,
    )
    cursor = start_sample
    waveform_offset = 0
    direction_offset = 0
    for chunk in chunks:
        assert chunk.start_sample == cursor
        assert chunk.end_sample == cursor + len(chunk.waveform)
        expected_waveform = waveform[
            waveform_offset:waveform_offset + len(chunk.waveform)
        ]
        np.testing.assert_array_equal(chunk.waveform, expected_waveform)
        assert chunk.sha256 == hashlib.sha256(expected_waveform.tobytes()).hexdigest()
        chunk_hop_count = len(chunk.waveform) // _HOP
        assert chunk.l2_direction_counts == tuple(
            (
                cursor + (index + 1) * _HOP,
                direction_values[direction_offset + index],
            )
            for index in range(chunk_hop_count)
        )
        cursor = chunk.end_sample
        waveform_offset += len(chunk.waveform)
        direction_offset += chunk_hop_count

    assert cursor == end_sample
    assert waveform_offset == len(waveform)
    assert direction_offset == total_hops
    assert hub.take_streaming_chunks(
        chunk_samples=chunk_seconds * _SAMPLE_RATE,
        ready_track_keys={key},
        flush=True,
    ) == ()
    hub.reset()
