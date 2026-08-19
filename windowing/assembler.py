from __future__ import annotations

from collections import deque

import numpy as np

from common.data_types import (
    CalibrationMetadata,
    DecisionWindow,
    ImcraHopSnapshot,
    IngestedAudioBlock,
    PipelineStatus,
)


class WindowAssembler:
    def __init__(self, *, context_samples: int = 15_360, doa_window_samples: int = 1_920, hop_samples: int = 960):
        if (context_samples, doa_window_samples, hop_samples) != (15_360, 1_920, 960):
            raise ValueError("v0.2窗口参数固定为15360/1920/960")
        self.context_samples, self.doa_window_samples, self.hop_samples = (
            context_samples,
            doa_window_samples,
            hop_samples,
        )
        self._session_id: str | None = None
        self._epoch: int | None = None
        self._buffer = np.empty((0, 8), dtype=np.float32)
        self._segments: deque[tuple[int, int, int]] = deque()
        self._imcra_hops: deque[ImcraHopSnapshot] = deque()
        self._calibration: CalibrationMetadata | None = None
        self._next_decision = context_samples
        self._next_window_id = 0
        self._epoch_window_count = 0
        self.discarded_tail_samples = 0

    def _start_epoch(self, block: IngestedAudioBlock) -> None:
        if self._epoch is not None:
            next_hop_start = max(0, self._next_decision - self.hop_samples)
            self.discarded_tail_samples += max(0, block.start_sample - next_hop_start)
        self._session_id, self._epoch = block.session_id, block.stream_epoch
        self._buffer = np.empty((0, 8), dtype=np.float32)
        self._segments.clear()
        self._imcra_hops.clear()
        self._calibration = block.calibration
        self._next_decision = self.context_samples
        self._epoch_window_count = 0

    def add(
        self, block: IngestedAudioBlock, imcra_hops: tuple[ImcraHopSnapshot, ...] = ()
    ) -> tuple[DecisionWindow, ...]:
        if self._session_id != block.session_id or self._epoch != block.stream_epoch:
            self._start_epoch(block)
        elif block.calibration != self._calibration:
            raise ValueError("WindowAssembler calibration boundary changed without a new epoch")
        expected_start = 0 if not self._segments and not len(self._buffer) else self._segments[-1][1]
        if block.start_sample != expected_start:
            raise ValueError("WindowAssembler收到非连续block；连续性只能由Coordinator重建epoch")
        # IngestedAudioBlock is already in logical order:
        # MIC0..MIC5, Center, HardwareMix. Preserve all eight channels so L3
        # receives its public input unchanged.
        self._buffer = np.concatenate((self._buffer, block.samples), axis=0)
        self._segments.append((block.start_sample, block.end_sample, block.sequence_id))
        incoming_hops = tuple(imcra_hops) or (() if block.imcra_hop is None else (block.imcra_hop,))
        for hop in incoming_hops:
            if (hop.session_id, hop.stream_epoch) != (block.session_id, block.stream_epoch):
                raise ValueError("WindowAssembler IMCRA hop belongs to another stream")
            if self._imcra_hops and hop.start_sample != self._imcra_hops[-1].end_sample:
                raise ValueError("WindowAssembler IMCRA hops must be continuous")
            self._imcra_hops.append(hop)
        windows: list[DecisionWindow] = []
        while block.end_sample >= self._next_decision:
            context_start = self._next_decision - self.context_samples
            # Buffer always starts at max(0, next_decision-context); after a
            # window it is trimmed to retain exactly the needed overlap.
            buffer_start = self._segments[0][0]
            offset = context_start - buffer_start
            data = self._buffer[offset : offset + self.context_samples]
            sequence_ids = tuple(
                seq for start, end, seq in self._segments if end > context_start and start < self._next_decision
            )
            imcra_hops = tuple(
                hop for hop in self._imcra_hops
                if hop.end_sample > context_start and hop.start_sample < self._next_decision
            )
            windows.append(
                DecisionWindow(
                    block.session_id,
                    block.stream_epoch,
                    self._next_window_id,
                    self._next_decision,
                    self._next_decision - self.doa_window_samples,
                    self._next_decision,
                    context_start,
                    self._next_decision,
                    block.sample_rate,
                    data,
                    sequence_ids,
                    imcra_hops,
                    block.calibration,
                )
            )
            self._next_window_id += 1
            self._epoch_window_count += 1
            self._next_decision += self.hop_samples
            keep_start = self._next_decision - self.context_samples
            trim = keep_start - buffer_start
            if trim > 0:
                self._buffer = self._buffer[trim:]
                while self._segments and self._segments[0][1] <= keep_start:
                    self._segments.popleft()
                if self._segments and self._segments[0][0] < keep_start:
                    start, end, seq = self._segments.popleft()
                    self._segments.appendleft((keep_start, end, seq))
                while self._imcra_hops and self._imcra_hops[0].end_sample <= keep_start:
                    self._imcra_hops.popleft()
        return tuple(windows)

    @property
    def status(self) -> PipelineStatus:
        if self._session_id is None or self._epoch is None:
            raise RuntimeError("尚未收到音频block")
        buffered = min(self._next_decision - self.context_samples + len(self._buffer), self.context_samples)
        ready = self._epoch_window_count > 0
        return PipelineStatus(
            "running" if ready else "warming_up",
            self._session_id,
            self._epoch,
            buffered,
            self.context_samples,
            "Ready" if ready else "Warming",
        )
