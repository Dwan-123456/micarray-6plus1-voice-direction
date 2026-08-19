from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from threading import RLock
from typing import Generic, TypeVar


K = TypeVar("K")
V = TypeVar("V")


class BoundedLru(Generic[K, V]):
    """Small, deterministic in-memory LRU; no project-side cache files."""

    def __init__(self, capacity: int = 8):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("LRU capacity must be a positive integer")
        self.capacity = capacity
        self._items: OrderedDict[K, V] = OrderedDict()
        self._lock = RLock()

    def get_or_load(self, key: K, loader: Callable[[], V]) -> V:
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        value = loader()
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
        return value

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
