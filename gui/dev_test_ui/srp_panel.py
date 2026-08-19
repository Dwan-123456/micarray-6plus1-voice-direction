from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

import numpy as np

from common.data_types import CandidateDirection, SpatialResponse
from layer2_source_detection.iterative import CandidateSearchDiagnostics


@dataclass(frozen=True, slots=True)
class SrpPanelSnapshot:
    response: SpatialResponse
    candidates: tuple[CandidateDirection, ...]
    published_monotonic: float
    search_diagnostics: CandidateSearchDiagnostics | None = None
    candidate_track_ids: tuple[int | None, ...] = ()
    candidate_is_prediction: tuple[bool, ...] = ()
    candidate_track_is_formal: tuple[bool, ...] = ()
    candidate_track_is_new: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        track_ids = tuple(self.candidate_track_ids) or (None,) * len(self.candidates)
        prediction_flags = tuple(self.candidate_is_prediction) or (False,) * len(self.candidates)
        formal_flags = tuple(self.candidate_track_is_formal) or (False,) * len(self.candidates)
        new_flags = tuple(self.candidate_track_is_new) or (False,) * len(self.candidates)
        if not (
            len(track_ids) == len(prediction_flags) == len(formal_flags)
            == len(new_flags) == len(self.candidates)
        ):
            raise ValueError("SRP候选身份显示信息必须与候选逐项对齐")
        if any(formal and track_id is None for formal, track_id in zip(formal_flags, track_ids)):
            raise ValueError("正式SRP候选必须携带ID")
        object.__setattr__(self, "candidate_track_ids", track_ids)
        object.__setattr__(self, "candidate_is_prediction", prediction_flags)
        object.__setattr__(self, "candidate_track_is_formal", formal_flags)
        object.__setattr__(self, "candidate_track_is_new", new_flags)
        if not np.isfinite(self.published_monotonic):
            raise ValueError("published_monotonic必须finite")
        if len(self.candidates) > 2:
            raise ValueError("SRP面板最多显示2个L2正式候选")
        identity = (
            self.response.session_id,
            self.response.stream_epoch,
            self.response.window_id,
            self.response.decision_sample,
            self.response.doa_start_sample,
            self.response.doa_end_sample,
        )
        seen: set[float] = set()
        previous_score = float("inf")
        for candidate in self.candidates:
            if (
                candidate.session_id,
                candidate.stream_epoch,
                candidate.window_id,
                candidate.decision_sample,
                candidate.doa_start_sample,
                candidate.doa_end_sample,
            ) != identity:
                raise ValueError("SRP响应与候选不属于同一window")
            if candidate.theta_deg in seen or candidate.normalized_score > previous_score:
                raise ValueError("候选必须角度唯一并保持Layer 2 rank顺序")
            seen.add(candidate.theta_deg)
            previous_score = candidate.normalized_score

    @property
    def age_ms(self) -> float:
        return max(0.0, (monotonic() - self.published_monotonic) * 1_000.0)


try:
    from PySide6.QtCore import QPointF, Qt, Signal
    from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
    from PySide6.QtWidgets import QWidget
except ImportError:  # pragma: no cover - UI dependency is optional for headless tests
    QWidget = None


if QWidget is not None:

    class SrpPolarPanel(QWidget):
        """Right-top development panel for one immutable Layer 2 snapshot."""

        candidate_selected = Signal(float, int)

        def __init__(self, stale_after_ms: int = 500, parent: QWidget | None = None):
            super().__init__(parent)
            self.stale_after_ms = int(stale_after_ms)
            self._snapshot: SrpPanelSnapshot | None = None
            self._selected_theta: float | None = None
            self._live = False
            self._stream_key: tuple[str, int] | None = None
            self._formal_color_slots: dict[int, int] = {}
            # The parent owns the strict 50% x 50% quadrant geometry. A hard
            # canvas minimum would make the top row steal height and jitter
            # when surrounding labels change their size hints.
            self.setMinimumSize(0, 0)

        def set_snapshot(self, snapshot: SrpPanelSnapshot | None, *, live: bool = True) -> None:
            unchanged = self._snapshot is snapshot and self._live == live
            stream_key = None if snapshot is None else (
                snapshot.response.session_id,
                snapshot.response.stream_epoch,
            )
            if stream_key != self._stream_key:
                self._formal_color_slots.clear()
                self._stream_key = stream_key
            if snapshot is not None:
                self._reserve_formal_color_slots(snapshot)
            self._snapshot = snapshot
            self._live = live
            if not unchanged:
                self.update()

        def set_live(self, live: bool) -> None:
            if self._live != live:
                self._live = live
                self.update()

        @staticmethod
        def _point(center: QPointF, radius: float, theta_deg: float) -> QPointF:
            radians = np.deg2rad(theta_deg)
            return QPointF(center.x() + radius * np.cos(radians), center.y() - radius * np.sin(radians))

        @staticmethod
        def _candidate_style(
            track_id: int | None, *, is_prediction: bool, is_formal: bool,
            is_new: bool = False, formal_color_slot: int | None = None,
        ) -> tuple[QColor, float]:
            """Return identity colour and evidence size for one L2 point.

            Colour describes identity: unassigned/pending is grey and formal
            IDs alternate deterministically between red and green.  Size
            describes current evidence: a formal Kalman-only prediction is
            small, while a currently observed formal ID is large.
            """

            small_diameter, large_diameter = 10.0, 24.0
            if not is_formal:
                return QColor("#929daa"), small_diameter if track_id is None or is_new else large_diameter
            slot = (int(track_id) - 1) % 2 if formal_color_slot is None else formal_color_slot
            colour = QColor("#ff3b30") if slot == 0 else QColor("#2ecc71")
            return colour, small_diameter if is_prediction else large_diameter

        def _reserve_formal_color_slots(self, snapshot: SrpPanelSnapshot) -> None:
            """Keep identity colours stable and distinct for the two live formal IDs."""

            current_ids = tuple(dict.fromkeys(
                int(track_id)
                for track_id, is_formal in zip(
                    snapshot.candidate_track_ids,
                    snapshot.candidate_track_is_formal,
                    strict=True,
                )
                if is_formal and track_id is not None
            ))
            used: set[int] = set()
            unresolved: list[int] = []
            for track_id in current_ids:
                previous = self._formal_color_slots.get(track_id)
                if previous in {0, 1} and previous not in used:
                    used.add(previous)
                else:
                    unresolved.append(track_id)
            for track_id in unresolved:
                slot = next((item for item in (0, 1) if item not in used), 0)
                self._formal_color_slots[track_id] = slot
                used.add(slot)

        @staticmethod
        def _response_radius(outer_radius: float, normalized_score: float) -> float:
            """Map Norm=0 near the center marker and Norm=1 near the rim."""
            score = float(np.clip(normalized_score, 0.0, 1.0))
            return outer_radius * (0.035 + 0.93 * score)

        def paintEvent(self, _event) -> None:  # noqa: N802
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillRect(self.rect(), QColor("#11161d"))
            painter.setPen(QColor("#dce7f2"))
            painter.setFont(QFont("Sans Serif", 11))
            painter.drawText(16, 24, "SRP-PHAT 360°")
            snapshot = self._snapshot
            if snapshot is None:
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "UNAVAILABLE")
                return

            # Leave less unused margin so the grey mathematical-angle rim is
            # larger, while retaining room for cardinal labels/candidates.
            margin, footer = 36.0, 72.0
            radius = max(20.0, min(self.width(), self.height() - footer) / 2.0 - margin)
            center = QPointF(self.width() / 2.0, (self.height() - footer) / 2.0 + 28.0)
            painter.setPen(QPen(QColor("#526273"), 1.0))
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
            polygon = QPolygonF(
                [self._point(center, self._response_radius(radius, scores[theta]), theta) for theta in range(360)]
                + [self._point(center, self._response_radius(radius, scores[0]), 0)]
            )
            painter.setPen(QPen(QColor("#42b8ff"), 2.0))
            painter.drawPolyline(polygon)

            for candidate, track_id, is_prediction, is_formal, is_new in zip(
                snapshot.candidates,
                snapshot.candidate_track_ids,
                snapshot.candidate_is_prediction,
                snapshot.candidate_track_is_formal,
                snapshot.candidate_track_is_new,
                strict=True,
            ):
                point = self._point(center, radius, candidate.theta_deg)
                colour, diameter = self._candidate_style(
                    track_id,
                    is_prediction=is_prediction,
                    is_formal=is_formal,
                    is_new=is_new,
                    formal_color_slot=self._formal_color_slots.get(track_id),
                )
                painter.setBrush(colour)
                pen = QPen(QColor("white") if candidate.theta_deg == self._selected_theta else QColor("#11161d"), 2.5)
                painter.setPen(pen)
                painter.drawEllipse(point, diameter / 2.0, diameter / 2.0)
                label = self._point(center, radius + 34.0, candidate.theta_deg)
                painter.setPen(QColor("#f3f7fb"))
                painter.drawText(
                    QPointF(label.x() - 28, label.y() + 5),
                    f"{candidate.theta_deg:.0f}° N={candidate.normalized_score:.3f}",
                )

            age_ms = snapshot.age_ms if self._live else None
            stale = age_ms is not None and age_ms > self.stale_after_ms
            state = "STOPPED" if not self._live else (
                "STALE" if stale else ("NO CANDIDATE" if not snapshot.candidates else "LIVE")
            )
            painter.setPen(QColor("#ffb54c") if stale else QColor("#9fb2c5"))
            age_text = "—" if age_ms is None else f"{age_ms:.0f} ms"
            painter.drawText(
                16, self.height() - 18, f"window={snapshot.response.window_id}  age={age_text}  {state}"
            )

        def mousePressEvent(self, event) -> None:  # noqa: N802
            if self._snapshot is None or not self._snapshot.candidates:
                return
            footer = 72.0
            center = QPointF(self.width() / 2.0, (self.height() - footer) / 2.0 + 28.0)
            dx, dy = event.position().x() - center.x(), center.y() - event.position().y()
            clicked = float(np.rad2deg(np.arctan2(dy, dx)) % 360.0)
            nearest = min(
                self._snapshot.candidates, key=lambda candidate: abs((candidate.theta_deg - clicked + 180) % 360 - 180)
            )
            if abs((nearest.theta_deg - clicked + 180) % 360 - 180) <= 8.0:
                self._selected_theta = nearest.theta_deg
                self.candidate_selected.emit(nearest.theta_deg, self._snapshot.response.window_id)
                self.update()
else:

    class SrpPolarPanel:  # pragma: no cover - informative fallback
        def __init__(self, *_args, **_kwargs):
            raise ImportError("安装PySide6后才能使用SrpPolarPanel")
