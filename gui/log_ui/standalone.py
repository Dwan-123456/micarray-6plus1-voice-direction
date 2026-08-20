from __future__ import annotations

from dataclasses import dataclass

from .app import launch_log_ui


@dataclass(frozen=True, slots=True)
class StandaloneUnavailableProvider:
    """Intentional no-capability provider for the independent desktop process.

    The standalone process has no cross-process public read-only port in the
    current architecture.  Keeping this object method-free makes capability
    probing render ``Unavailable`` without opening Catalog, SQLite, WAL, or a
    Runtime mailbox.
    """


def main() -> int:
    return launch_log_ui(StandaloneUnavailableProvider())


__all__ = ["StandaloneUnavailableProvider", "main"]
