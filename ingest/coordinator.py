from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from common.data_types import CalibrationMetadata, IngestedAudioBlock
from layer1_input.continuity import continuity_decision
from layer1_input.interface import DecodedAudio, InputHealthEvent


@dataclass(frozen=True, slots=True)
class Discontinuity:
    session_id: str
    old_epoch: int
    new_epoch: int
    reason: str


class IngestCoordinator:
    """The sole authority for stream epochs and absolute sample boundaries."""

    def __init__(
        self, *, sample_rate: int = 48_000, timestamp_tolerance_ms: float = 5.0, session_id: str | None = None
    ):
        if sample_rate != 48_000 or timestamp_tolerance_ms < 0:
            raise ValueError("Coordinator固定使用48 kHz且timestamp容差不能为负")
        self.sample_rate = sample_rate
        self.timestamp_tolerance = timestamp_tolerance_ms / 1000.0
        self.session_id = session_id or str(uuid4())
        self.stream_epoch = 0
        self._next_sample = 0
        self._previous: DecodedAudio | None = None
        self._pending_events: list[InputHealthEvent] = []
        self._seen_event_ids: set[int] = set()
        self._pending_reset_reason: str | None = None
        self.discontinuities: list[Discontinuity] = []

    def publish_health_event(self, event: InputHealthEvent) -> None:
        if event.event_id not in self._seen_event_ids:
            self._seen_event_ids.add(event.event_id)
            self._pending_events.append(event)

    def _continuity_reason(self, frame: DecodedAudio) -> str | None:
        return continuity_decision(
            self._previous, frame, timestamp_tolerance_ms=self.timestamp_tolerance * 1000.0
        ).reason

    def _reset(self, reason: str) -> None:
        old = self.stream_epoch
        self.stream_epoch += 1
        self._next_sample = 0
        self._previous = None
        self.discontinuities.append(Discontinuity(self.session_id, old, self.stream_epoch, reason))

    def ingest(self, frame: DecodedAudio, health_events: tuple[InputHealthEvent, ...] = ()) -> IngestedAudioBlock:
        for event in health_events:
            self.publish_health_event(event)
        if frame.sample_rate != self.sample_rate:
            raise ValueError(f"输入采样率必须为{self.sample_rate}")
        if self._pending_events:
            kinds = ",".join(event.kind for event in self._pending_events)
            self._pending_events.clear()
            self._pending_reset_reason = f"health_event:{kinds}"
        continuity_reason = self._continuity_reason(frame)
        reason = self._pending_reset_reason or continuity_reason
        if reason is not None:
            # A health event and the matching sequence gap describe one break.
            self._reset(reason)
            self._pending_reset_reason = None
        start = self._next_sample
        end = start + frame.frame_count
        block = IngestedAudioBlock(
            self.session_id,
            self.stream_epoch,
            start,
            end,
            frame.sample_rate,
            frame.sequence_id,
            frame.timestamp,
            frame.samples,
            frame.native_samples,
            frame.hotmap,
            frame.noise_spectrum,
            None,
            frame.calibration or CalibrationMetadata.unverified_identity(),
        )
        self._next_sample = end
        self._previous = frame
        return block
