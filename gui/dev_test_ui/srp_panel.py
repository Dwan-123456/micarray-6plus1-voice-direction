from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

import numpy as np

from common.data_types import CandidateDirection, SpatialResponse, TrackedDirection


_TRACK_COLOURS = ("#ff3b30", "#2ecc71", "#ffb000", "#af7ac5", "#00bcd4", "#ff7f50")
_TENTATIVE_TRACK_COLOUR = "#aab2bb"


class TrackColourPool:
    """Lease the six formal-ID colours until an authoritative track dies."""

    def __init__(self) -> None:
        self._stream: tuple[str, int] | None = None
        self._leases: dict[int, int] = {}

    def sync(
        self,
        stream: tuple[str, int] | None,
        active_tracks: tuple[TrackedDirection, ...],
    ) -> None:
        if stream != self._stream:
            self._stream = stream
            self._leases.clear()
        formal_ids = tuple(dict.fromkeys(
            int(track.track_id)
            for track in active_tracks
            if track.track_state != "tentative"
        ))
        live = set(formal_ids)
        self._leases = {
            track_id: colour_index
            for track_id, colour_index in self._leases.items()
            if track_id in live
        }
        available = [
            index for index in range(len(_TRACK_COLOURS))
            if index not in self._leases.values()
        ]
        for track_id in formal_ids:
            if track_id not in self._leases and available:
                self._leases[track_id] = available.pop(0)

    def colour(self, track_id: int) -> str | None:
        index = self._leases.get(int(track_id))
        return None if index is None else _TRACK_COLOURS[index]


_TRACK_COLOUR_POOL = TrackColourPool()


def sync_track_colours(
    stream: tuple[str, int] | None,
    active_tracks: tuple[TrackedDirection, ...],
) -> None:
    _TRACK_COLOUR_POOL.sync(stream, tuple(active_tracks))


def track_colour_hex(track_id: int) -> str:
    """Return the stable UI colour assigned to one authoritative L2 ID."""
    leased = _TRACK_COLOUR_POOL.colour(track_id)
    if leased is not None:
        return leased
    # Historical/non-live rows do not own a lease. Keep their legacy colour
    # deterministic without allowing them to reserve a live-ID colour.
    return _TRACK_COLOURS[(int(track_id) - 1) % len(_TRACK_COLOURS)]


def _track_marker_style(track: TrackedDirection) -> tuple[str, float]:
    """Keep tentative IDs neutral; reserve stable colours for formal IDs."""

    if track.track_state == "tentative":
        return _TENTATIVE_TRACK_COLOUR, 10.0
    diameter = 24.0 if track.is_observed else 10.0
    return track_colour_hex(track.track_id), diameter


@dataclass(frozen=True, slots=True)
class MusicPanelSnapshot:
    """One immutable, authoritative Layer-2 DOA/UI projection."""

    response: SpatialResponse
    directions: tuple[TrackedDirection, ...]
    active_tracks: tuple[TrackedDirection, ...]
    published_monotonic: float
    effective_order: int | None = None
    raw_peaks: tuple[CandidateDirection, ...] = ()
    direction_id_tracking_enabled: bool = True

    def __post_init__(self) -> None:
        directions = tuple(self.directions)
        active = tuple(self.active_tracks)
        raw_peaks = tuple(self.raw_peaks)
        identity = (
            self.response.session_id,
            self.response.stream_epoch,
            self.response.window_id,
            self.response.decision_sample,
        )
        if not np.isfinite(self.published_monotonic):
            raise ValueError("published_monotonic must be finite")
        if len(directions) > 3:
            raise ValueError("MUSIC directions are limited to three")
        if self.effective_order is not None and self.effective_order not in {0, 1, 2, 3}:
            raise ValueError("effective MUSIC order must be 0..3 or None")
        if type(self.direction_id_tracking_enabled) is not bool:
            raise TypeError("MUSIC panel ID tracking flag must be bool")
        if len(raw_peaks) > 3:
            raise ValueError("MUSIC raw peaks are limited to three")
        if any(
            (item.session_id, item.stream_epoch, item.window_id, item.decision_sample)
            != identity for item in raw_peaks
        ):
            raise ValueError("MUSIC response and raw peaks must belong to one window")
        if any((item.session_id, item.stream_epoch, item.window_id, item.decision_sample) != identity for item in directions):
            raise ValueError("MUSIC response and directions must belong to one window")
        if any((item.session_id, item.stream_epoch) != identity[:2] for item in active):
            raise ValueError("active L2 tracks must belong to the MUSIC stream")
        if len({item.track_id for item in active}) != len(active):
            raise ValueError("active L2 track IDs must be unique")
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "active_tracks", active)
        object.__setattr__(self, "raw_peaks", raw_peaks)

    @property
    def age_ms(self) -> float:
        return max(0.0, (monotonic() - self.published_monotonic) * 1_000.0)


try:
    from PySide6.QtCore import QPointF, Qt, Signal
    from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
    from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem, QWidget
except ImportError:  # pragma: no cover
    QWidget = None


if QWidget is not None:

    class MusicPolarPanel(QWidget):
        track_selected = Signal(int, int)
        candidate_selected = Signal(float, int)

        def __init__(self, stale_after_ms: int = 500, parent: QWidget | None = None):
            super().__init__(parent)
            self.stale_after_ms = int(stale_after_ms)
            self._snapshot: MusicPanelSnapshot | None = None
            self._live = False
            self._selected_track_id: int | None = None
            self._stream_key: tuple[str, int] | None = None
            self._gate_closed_tracks: tuple[TrackedDirection, ...] = ()
            self._gate_closed_window_id: int | None = None
            self.setMinimumSize(0, 0)

        def set_snapshot(self, snapshot: MusicPanelSnapshot | None, *, live: bool = True) -> None:
            stream = None if snapshot is None else (snapshot.response.session_id, snapshot.response.stream_epoch)
            if stream != self._stream_key:
                self._stream_key = stream
            self._snapshot = snapshot
            self._gate_closed_tracks = ()
            self._gate_closed_window_id = None
            self._live = bool(live)
            self.update()

        def set_gate_closed_tracks(
            self,
            active_tracks: tuple[TrackedDirection, ...],
            *,
            window_id: int,
            live: bool = True,
        ) -> None:
            """Render the geometry and live IDs without a stale MUSIC curve."""

            self._snapshot = None
            self._gate_closed_tracks = tuple(active_tracks)
            self._gate_closed_window_id = int(window_id)
            self._live = bool(live)
            self.update()

        def set_live(self, live: bool) -> None:
            self._live = bool(live)
            self.update()

        @staticmethod
        def _point(center: QPointF, radius: float, theta_deg: float) -> QPointF:
            angle = np.deg2rad(theta_deg)
            return QPointF(center.x() + radius * np.cos(angle), center.y() - radius * np.sin(angle))

        @staticmethod
        def _response_radius(outer_radius: float, score: float) -> float:
            return outer_radius * (0.035 + 0.93 * float(np.clip(score, 0.0, 1.0)))

        def paintEvent(self, _event) -> None:  # noqa: N802
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillRect(self.rect(), QColor("#11161d"))
            painter.setPen(QColor("#dce7f2"))
            painter.setFont(QFont("Sans Serif", 11))
            snapshot = self._snapshot
            gate_closed = self._gate_closed_window_id is not None
            if (
                not gate_closed
                and (snapshot is None or snapshot.response.model_order is None)
            ):
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "DOA UNAVAILABLE")
                return

            margin = 34.0
            radius = max(20.0, min(self.width(), self.height()) / 2.0 - margin)
            center = QPointF(self.width() / 2.0, self.height() / 2.0)
            painter.setPen(QPen(QColor("#526273"), 1.6))
            painter.drawEllipse(center, radius, radius)
            painter.drawEllipse(center, 3.0, 3.0)
            for theta in range(0, 360, 30):
                outer = self._point(center, radius, theta)
                inner = self._point(center, radius - (10 if theta % 90 == 0 else 5), theta)
                painter.drawLine(inner, outer)
                if theta % 90 == 0:
                    label = self._point(center, radius + 20, theta)
                    painter.drawText(QPointF(label.x() - 12, label.y() + 5), f"{theta}°")

            if not gate_closed:
                assert snapshot is not None
                scores = snapshot.response.normalized_scores
                polygon = QPolygonF([
                    self._point(center, self._response_radius(radius, scores[theta]), theta)
                    for theta in range(360)
                ] + [self._point(center, self._response_radius(radius, scores[0]), 0)])
                painter.setPen(QPen(QColor("#42b8ff"), 2.2))
                painter.drawPolyline(polygon)

            tracks = self._gate_closed_tracks if gate_closed else snapshot.active_tracks
            id_tracking_enabled = gate_closed or snapshot.direction_id_tracking_enabled
            if id_tracking_enabled:
                for track in tracks:
                    point = self._point(center, radius, track.theta_deg)
                    colour, diameter = _track_marker_style(track)
                    painter.setBrush(QColor(colour))
                    painter.setPen(QPen(QColor("white") if track.track_id == self._selected_track_id else QColor("#11161d"), 2.5))
                    painter.drawEllipse(point, diameter / 2.0, diameter / 2.0)
            else:
                painter.setBrush(QColor("#aab2bb"))
                painter.setPen(QPen(QColor("#11161d"), 1.5))
                for peak in snapshot.raw_peaks:
                    point = self._point(center, radius, peak.theta_deg)
                    painter.drawEllipse(point, 5.0, 5.0)

        def mousePressEvent(self, event) -> None:  # noqa: N802
            snapshot = self._snapshot
            gate_closed = self._gate_closed_window_id is not None
            tracks = self._gate_closed_tracks if gate_closed else (
                () if snapshot is None else snapshot.active_tracks
            )
            if not tracks or (not gate_closed and not snapshot.direction_id_tracking_enabled):
                return
            center = QPointF(self.width() / 2.0, self.height() / 2.0)
            dx, dy = event.position().x() - center.x(), center.y() - event.position().y()
            clicked = float(np.rad2deg(np.arctan2(dy, dx)) % 360.0)
            nearest = min(tracks, key=lambda item: abs((item.theta_deg - clicked + 180) % 360 - 180))
            if abs((nearest.theta_deg - clicked + 180) % 360 - 180) <= 8.0:
                self._selected_track_id = nearest.track_id
                window_id = (
                    self._gate_closed_window_id
                    if gate_closed
                    else snapshot.response.window_id
                )
                self.track_selected.emit(nearest.track_id, window_id)
                self.candidate_selected.emit(nearest.theta_deg, window_id)
                self.update()


    class DirectionTrackTable(QTableWidget):
        ROW_COUNT = 3
        HEADERS = ("track_id", "观测角", "输出角", "score", "状态", "新建", "观测")

        def __init__(self, parent: QWidget | None = None):
            super().__init__(self.ROW_COUNT, len(self.HEADERS), parent)
            self.setHorizontalHeaderLabels(self.HEADERS)
            self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            self.horizontalHeader().setStretchLastSection(True)
            self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.setAlternatingRowColors(False)
            for row in range(self.ROW_COUNT):
                for column in range(len(self.HEADERS)):
                    self.setItem(row, column, QTableWidgetItem(""))

        def set_snapshot(self, snapshot: MusicPanelSnapshot | None) -> None:
            tracks = () if snapshot is None else snapshot.active_tracks
            self.setUpdatesEnabled(False)
            try:
                for row in range(self.ROW_COUNT):
                    track = tracks[row] if row < len(tracks) else None
                    if track is None:
                        values = ("",) * len(self.HEADERS)
                        colour = self.palette().text().color()
                    else:
                        values = (
                            str(track.track_id),
                            "—" if track.measured_theta_deg is None else f"{track.measured_theta_deg:.1f}°",
                            f"{track.theta_deg:.1f}°",
                            f"{track.normalized_score:.3f}",
                            track.track_state,
                            "是" if track.is_new_track else "否",
                            "是" if track.is_observed else "否",
                        )
                        colour = QColor(track_colour_hex(track.track_id))
                    for column, value in enumerate(values):
                        item = self.item(row, column)
                        item.setText(value)
                        item.setForeground(colour)
            finally:
                self.setUpdatesEnabled(True)
                self.viewport().update()

else:

    class MusicPolarPanel:  # pragma: no cover
        def __init__(self, *_args, **_kwargs):
            raise ImportError("安装PySide6后才能使用MusicPolarPanel")

    class DirectionTrackTable:  # pragma: no cover
        def __init__(self, *_args, **_kwargs):
            raise ImportError("安装PySide6后才能使用DirectionTrackTable")
