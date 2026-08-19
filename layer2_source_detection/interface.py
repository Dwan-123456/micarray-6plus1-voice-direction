from __future__ import annotations

from typing import Protocol

from common.data_types import CandidateDirection, DecisionWindow, SpatialResponse
from common.geometry import MicGeometry

from .configuration import DirectionScanConfig
from .iterative import CandidateSearchDiagnostics


class DirectionScanner(Protocol):
    def scan(
        self,
        window: DecisionWindow,
        geometry: MicGeometry,
        config: DirectionScanConfig,
    ) -> tuple[SpatialResponse, tuple[CandidateDirection, ...]]: ...


class DetailedDirectionScanner(Protocol):
    def scan_detailed(
        self,
        window: DecisionWindow,
        geometry: MicGeometry,
        config: DirectionScanConfig,
        config_revision: int = 0,
    ) -> tuple[SpatialResponse, tuple[CandidateDirection, ...], CandidateSearchDiagnostics]: ...


class DirectionScanError(RuntimeError):
    """The current window cannot produce a contract-valid spatial response."""
