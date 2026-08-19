"""Independent, read-only Pipeline Log UI.

The package deliberately accepts an already-created public query provider.  It
never constructs ``DataManagerService`` or opens the project's Catalog.
"""

from .adapter import PublicApiAdapter, ReadOnlyProvider
from .models import (
    Anomaly,
    Availability,
    CapabilitySet,
    SessionReadModel,
    StageState,
    WindowKey,
)
from .statistics import SessionStatistics, StatisticsEngine

__all__ = [
    "Anomaly",
    "Availability",
    "CapabilitySet",
    "PublicApiAdapter",
    "ReadOnlyProvider",
    "SessionReadModel",
    "SessionStatistics",
    "StageState",
    "StatisticsEngine",
    "WindowKey",
]
