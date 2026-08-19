from __future__ import annotations

import queue
import threading

from common.data_types import IngestedAudioBlock


class BlockFanout:
    """Publishes the exact same immutable block object to bounded consumers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: set[queue.Queue[IngestedAudioBlock]] = set()
        self.dropped_by_subscriber = 0

    def subscribe(self, capacity: int) -> queue.Queue[IngestedAudioBlock]:
        if capacity <= 0:
            raise ValueError("subscriber capacity必须为正数")
        receiver: queue.Queue[IngestedAudioBlock] = queue.Queue(maxsize=capacity)
        with self._lock:
            self._subscribers.add(receiver)
        return receiver

    def unsubscribe(self, receiver: queue.Queue[IngestedAudioBlock]) -> None:
        with self._lock:
            self._subscribers.discard(receiver)

    def publish(self, block: IngestedAudioBlock) -> None:
        with self._lock:
            subscribers = tuple(self._subscribers)
        for receiver in subscribers:
            try:
                receiver.put_nowait(block)
            except queue.Full:
                try:
                    receiver.get_nowait()
                    receiver.put_nowait(block)
                    self.dropped_by_subscriber += 1
                except queue.Empty:
                    pass
