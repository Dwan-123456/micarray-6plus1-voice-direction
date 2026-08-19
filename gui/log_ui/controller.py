from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event, RLock
from typing import Generic, TypeVar

from .adapter import PublicApiAdapter
from .models import SessionDescriptor, SessionReadModel


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Page(Generic[T]):
    items: tuple[T, ...]
    offset: int
    total: int


def page(items: Sequence[T], offset: int, limit: int) -> Page[T]:
    if offset < 0 or limit <= 0:
        raise ValueError("offset must be non-negative and limit must be positive")
    return Page(tuple(items[offset:offset + limit]), offset, len(items))


class LogUiController:
    """Cancellable background loader used by the Qt layer and headless tests."""

    def __init__(self, adapter: PublicApiAdapter, *, workers: int = 1):
        if workers != 1:
            raise ValueError("Log UI uses one bounded read worker")
        self.adapter = adapter
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline-log-read")
        self._cancel = Event()
        self._lock = RLock()
        self._future: Future[SessionReadModel] | None = None

    def sessions(self, offset: int = 0, limit: int = 200) -> Page[SessionDescriptor]:
        return page(self.adapter.list_sessions(), offset, limit)

    def load_session(
        self,
        session_id: str,
        callback: Callable[[SessionReadModel | None, BaseException | None], None],
    ) -> Future[SessionReadModel]:
        self.cancel()
        self._cancel = Event()
        cancel = self._cancel

        def load() -> SessionReadModel:
            if cancel.is_set():
                raise CancelledError("session load cancelled")
            result = self.adapter.load_session(session_id, cancelled=cancel.is_set)
            if cancel.is_set():
                raise CancelledError("session load cancelled")
            return result

        future = self._executor.submit(load)

        def done(completed: Future[SessionReadModel]) -> None:
            try:
                result = completed.result()
            except BaseException as exc:  # callback receives validation/cancellation details
                callback(None, exc)
            else:
                callback(result, None)

        future.add_done_callback(done)
        with self._lock:
            self._future = future
        return future

    def cancel(self) -> None:
        with self._lock:
            self._cancel.set()
            if self._future is not None:
                self._future.cancel()

    def close(self) -> None:
        self.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)


class CancelledError(RuntimeError):
    pass
