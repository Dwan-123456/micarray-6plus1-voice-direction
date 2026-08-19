from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from types import MappingProxyType
from typing import Mapping

import numpy as np

from common.data_types import SpatialResponse, TrackedDirection


_TRACK_COLOURS = ("#ff3b30", "#2ecc71", "#ffb000", "#af7ac5", "#00bcd4", "#ff7f50")


@dataclass(frozen=True, slots=True)
class MusicPanelSnapshot:
    """One immutable, authoritative Layer-2 MUSIC/UI projection."""

    response: SpatialResponse
    directions: tuple[TrackedDirection, ...]
    active_tracks: tuple[TrackedDirection, ...]
    published_monotonic: float
    l4_probability_by_track: Mapping[int, float] = MappingProxyType({})

    def __post_init__(self) -> None:
        directions = tuple(self.directions)
        active = tuple(self.active_tracks)
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
        if any((item.session_id, item.stream_epoch, item.window_id, item.decision_sample) != identity for item in directions):
            raise ValueError("MUSIC response and directions must belong to one window")
        if any((item.session_id, item.stream_epoch) != identity[:2] for item in active):
            raise ValueError("active L2 tracks must belong to the MUSIC stream")
        if len({item.track_id for item in active}) != len(active):
            raise ValueError("active L2 track IDs must be unique")
        probabilities = {int(key): float(value) for key, value in self.l4_probability_by_track.items()}
        if any(key <= 0 or not 0.0 <= value <= 1.0 for key, value in probabilities.items()):
            raise ValueError("L4 probabilities must be keyed by positive L2 track IDs")
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "active_tracks", active)
        object.__setattr__(self, "l4_probability_by_track", MappingProxyType(probabilities))

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
            self.setMinimumSize(0, 0)

        def set_snapshot(self, snapshot: MusicPanelSnapshot | None, *, live: bool = True) -> None:
            stream = None if snapshot is None else (snapshot.response.session_id, snapshot.response.stream_epoch)
            if stream != self._stream_key:
                self._stream_key = stream
            self._snapshot = snapshot
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

        def _track_colour(self, track_id: int) -> QColor:
            return QColor(_TRACK_COLOURS[(int(track_id) - 1) % len(_TRACK_COLOURS)])

        def paintEvent(self, _event) -> None:  # noqa: N802
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillRect(self.rect(), QColor("#11161d"))
            painter.setPen(QColor("#dce7f2"))
            painter.setFont(QFont("Sans Serif", 11))
            painter.drawText(16, 24, "DOA / MUSIC 360°")
            snapshot = self._snapshot
            if snapshot is None or snapshot.response.model_order is None:
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "MUSIC UNAVAILABLE")
                return

            margin, footer = 36.0, 72.0
            radius = max(20.0, min(self.width(), self.height() - footer) / 2.0 - margin)
            center = QPointF(self.width() / 2.0, (self.height() - footer) / 2.0 + 28.0)
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

            scores = snapshot.response.normalized_scores
            polygon = QPolygonF([
                self._point(center, self._response_radius(radius, scores[theta]), theta)
                for theta in range(360)
            ] + [self._point(center, self._response_radius(radius, scores[0]), 0)])
            painter.setPen(QPen(QColor("#42b8ff"), 2.2))
            painter.drawPolyline(polygon)

            for track in snapshot.active_tracks:
                point = self._point(center, radius, track.theta_deg)
                diameter = 24.0 if track.is_observed and track.track_state != "coasting" else 10.0
                painter.setBrush(self._track_colour(track.track_id))
                painter.setPen(QPen(QColor("white") if track.track_id == self._selected_track_id else QColor("#11161d"), 2.5))
                painter.drawEllipse(point, diameter / 2.0, diameter / 2.0)

            model = snapshot.response.model_order
            state = "STALE" if self._live and snapshot.age_ms > self.stale_after_ms else ("LIVE" if self._live else "STOPPED")
            painter.setPen(QColor("#9fb2c5"))
            painter.drawText(
                16,
                self.height() - 18,
                f"order={model.estimated_sources}  valid={snapshot.response.valid_frequency_bins}  "
                f"status={snapshot.response.numerical_status}  {state}",
            )

        def mousePressEvent(self, event) -> None:  # noqa: N802
            snapshot = self._snapshot
            if snapshot is None or not snapshot.active_tracks:
                return
            footer = 72.0
            center = QPointF(self.width() / 2.0, (self.height() - footer) / 2.0 + 28.0)
            dx, dy = event.position().x() - center.x(), center.y() - event.position().y()
            clicked = float(np.rad2deg(np.arctan2(dy, dx)) % 360.0)
            nearest = min(snapshot.active_tracks, key=lambda item: abs((item.theta_deg - clicked + 180) % 360 - 180))
            if abs((nearest.theta_deg - clicked + 180) % 360 - 180) <= 8.0:
                self._selected_track_id = nearest.track_id
                self.track_selected.emit(nearest.track_id, snapshot.response.window_id)
                self.candidate_selected.emit(nearest.theta_deg, snapshot.response.window_id)
                self.update()


    class DirectionTrackTable(QTableWidget):
        HEADERS = ("track_id", "观测角", "输出角", "score", "状态", "新建", "观测", "L4概率")

        def __init__(self, parent: QWidget | None = None):
            super().__init__(0, len(self.HEADERS), parent)
            self.setHorizontalHeaderLabels(self.HEADERS)
            self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            self.horizontalHeader().setStretchLastSection(True)
            self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.setAlternatingRowColors(True)

        def set_snapshot(self, snapshot: MusicPanelSnapshot | None) -> None:
            tracks = () if snapshot is None else snapshot.active_tracks
            self.setRowCount(len(tracks))
            for row, track in enumerate(tracks):
                probability = None if snapshot is None else snapshot.l4_probability_by_track.get(track.track_id)
                values = (
                    str(track.track_id),
                    "—" if track.measured_theta_deg is None else f"{track.measured_theta_deg:.1f}°",
                    f"{track.theta_deg:.1f}°",
                    f"{track.normalized_score:.3f}",
                    track.track_state,
                    "是" if track.is_new_track else "否",
                    "是" if track.is_observed else "否",
                    "—" if probability is None else f"{probability:.3f}",
                )
                colour = QColor(_TRACK_COLOURS[(track.track_id - 1) % len(_TRACK_COLOURS)])
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setForeground(colour)
                    self.setItem(row, column, item)

else:

    class MusicPolarPanel:  # pragma: no cover
        def __init__(self, *_args, **_kwargs):
            raise ImportError("安装PySide6后才能使用MusicPolarPanel")

    class DirectionTrackTable:  # pragma: no cover
        def __init__(self, *_args, **_kwargs):
            raise ImportError("安装PySide6后才能使用DirectionTrackTable")
