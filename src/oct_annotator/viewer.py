"""Custom QGraphicsView for displaying M-scans and collecting seed points."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt6.QtGui import (
    QImage, QPixmap, QPen, QColor, QBrush, QPainterPath, QMouseEvent,
    QWheelEvent, QResizeEvent,
)
from PyQt6.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsEllipseItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsLineItem,
)

from oct_annotator.engine import to_preview_uint8


SEED_RADIUS = 2
SEED_HITBOX_RADIUS_PX = 12
SEED_COLOR = QColor(255, 50, 50, 140)
SPLINE_COLOR = QColor(0, 255, 100, 110)
REFINED_COLOR = QColor(255, 40, 40, 235)
_CLASS_CURVE_COLORS: dict[int, QColor] = {
    1: QColor(0, 190, 90, 220),
    2: QColor(230, 190, 0, 220),
    3: QColor(210, 40, 40, 220),
}
# Palette of (fill, edge) colors cycled across successive NaN windows
_NAN_PALETTE: list[tuple[QColor, QColor]] = [
    (QColor(220,   0, 255,  55), QColor(220,   0, 255, 210)),  # magenta
    (QColor(  0, 210, 210,  55), QColor(  0, 210, 210, 210)),  # cyan
    (QColor(230, 210,   0,  55), QColor(230, 210,   0, 210)),  # yellow
    (QColor(255, 100,   0,  55), QColor(255, 100,   0, 210)),  # orange
    (QColor(255,   0, 110,  55), QColor(255,   0, 110, 210)),  # hot-pink
]


# ---------------------------------------------------------------------------
#  Draggable vertical bar (edge of a NaN-exclusion window)
# ---------------------------------------------------------------------------

class _DraggableVLine(QGraphicsLineItem):
    """A vertical line spanning the full image height, draggable along x."""

    def __init__(self, x: float, height: float, parent_viewer: "MScanViewer",
                 edge_color: QColor | None = None):
        super().__init__(x, 0, x, height)
        self.setPen(QPen(edge_color if edge_color is not None else _NAN_PALETTE[0][1], 2))
        self.setFlag(QGraphicsLineItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsLineItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self._height = height
        self._viewer = parent_viewer
        self._image_width: float = 0.0

    def set_image_width(self, w: float) -> None:
        self._image_width = w

    def itemChange(self, change, value):
        if change == QGraphicsLineItem.GraphicsItemChange.ItemPositionChange:
            # value is the proposed new pos() (scene offset).
            # The line itself is drawn at line().x1() in local coords, so the
            # actual screen x = value.x() + line().x1().  Clamping value.x()
            # directly would prevent bars from moving left of their initial
            # position, so we clamp the actual screen x instead.
            actual_x = value.x() + self.line().x1()
            clamped_x = max(0.0, min(actual_x, self._image_width - 1))
            self._viewer._on_nan_bar_moved()
            return QPointF(clamped_x - self.line().x1(), 0.0)
        return super().itemChange(change, value)

    @property
    def x_pos(self) -> float:
        return self.pos().x() + self.line().x1()


class NaNWindow:
    """A pair of vertical bars defining an exclusion region."""

    def __init__(self, left_bar: _DraggableVLine, right_bar: _DraggableVLine,
                 fill_rect: QGraphicsRectItem):
        self.left = left_bar
        self.right = right_bar
        self.fill = fill_rect

    @property
    def x_range(self) -> Tuple[int, int]:
        lx = int(round(self.left.x_pos))
        rx = int(round(self.right.x_pos))
        return (min(lx, rx), max(lx, rx))


# ---------------------------------------------------------------------------
#  Main viewer
# ---------------------------------------------------------------------------

class MScanViewer(QGraphicsView):
    """Interactive viewer that displays a numpy M-scan and lets users place seed points."""

    seeds_changed = pyqtSignal()
    nan_windows_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self._pixmap_item = None
        self._image_shape: Tuple[int, int] | None = None  # (rows, cols)

        # Seed points as (x, y, class_id) image coordinates
        self._seeds: List[Tuple[float, float, int]] = []
        self._seed_items: List[QGraphicsEllipseItem] = []
        self._current_seed_class: int = 1

        # Curve overlays
        self._spline_path_items: List[QGraphicsPathItem] = []
        self._refined_path_items: List[QGraphicsPathItem] = []
        self._spline_color: QColor = QColor(SPLINE_COLOR)
        self._refined_color: QColor = QColor(REFINED_COLOR)

        # Pan state (both mouse buttons held)
        self._panning: bool = False
        self._pan_start = None

        # NaN exclusion windows
        self._nan_windows: List[NaNWindow] = []

    # ---- public API ---------------------------------------------------

    def set_image(self, data: NDArray) -> None:
        """Display a 2-D float array as a grayscale image."""
        self.clear_overlays()
        self.clear_nan_windows()
        self._seeds.clear()
        self._seed_items.clear()
        self._scene.clear()
        self._pixmap_item = None
        self._spline_path_items = []
        self._refined_path_items = []

        rows, cols = data.shape
        self._image_shape = (rows, cols)

        # Match DB-analyzer preprocessing for preview contrast.
        normed = to_preview_uint8(data)

        qimage = QImage(
            normed.data.tobytes(),
            cols,
            rows,
            cols,  # bytes-per-line for 8-bit grayscale
            QImage.Format.Format_Grayscale8,
        )
        pixmap = QPixmap.fromImage(qimage)
        self._pixmap_item = self._scene.addPixmap(pixmap)
        # Keep an explicit finite scene rect so view scrollbars stay responsive.
        self._scene.setSceneRect(QRectF(0, 0, cols, rows))
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    @property
    def seeds(self) -> List[Tuple[float, float]]:
        return [(x, y) for x, y, _ in self._seeds]

    @property
    def seed_classes(self) -> List[int]:
        return [cls for _, _, cls in self._seeds]

    def set_current_seed_class(self, class_id: int) -> None:
        self._current_seed_class = int(class_id)

    @property
    def image_width(self) -> int:
        if self._image_shape is None:
            return 0
        return self._image_shape[1]

    @property
    def image_height(self) -> int:
        if self._image_shape is None:
            return 0
        return self._image_shape[0]

    def draw_spline(self, y_indices: NDArray, nan_mask: NDArray[np.bool_] | None = None) -> None:
        """Overlay a spline curve on the image, with gaps for NaN regions."""
        self._clear_path_items(self._spline_path_items)
        self._spline_path_items = [self._add_curve(y_indices, self._spline_color, 1.5, nan_mask)]

    def draw_refined(self, y_indices: NDArray, nan_mask: NDArray[np.bool_] | None = None) -> None:
        """Overlay a refined boundary curve on the image, with gaps for NaN regions."""
        self._clear_path_items(self._refined_path_items)
        self._refined_path_items = [self._add_curve(y_indices, self._refined_color, 2.5, nan_mask)]

    def draw_spline_classified(
        self,
        y_indices: NDArray,
        class_by_column: NDArray[np.int32],
        nan_mask: NDArray[np.bool_] | None = None,
    ) -> None:
        """Overlay a spline curve with per-class colors along x."""
        self._clear_path_items(self._spline_path_items)
        items: List[QGraphicsPathItem] = []
        for class_id, color in _CLASS_CURVE_COLORS.items():
            items.append(self._add_curve_for_class(y_indices, class_by_column, class_id, color, 1.5, nan_mask))
        self._spline_path_items = items

    def draw_refined_classified(
        self,
        y_indices: NDArray,
        class_by_column: NDArray[np.int32],
        nan_mask: NDArray[np.bool_] | None = None,
    ) -> None:
        """Overlay a refined curve with per-class colors along x."""
        self._clear_path_items(self._refined_path_items)
        items: List[QGraphicsPathItem] = []
        for class_id, color in _CLASS_CURVE_COLORS.items():
            items.append(self._add_curve_for_class(y_indices, class_by_column, class_id, color, 2.5, nan_mask))
        self._refined_path_items = items

    def set_annotation_colors(self, spline_color: QColor, refined_color: QColor | None = None) -> None:
        """Set colors used for subsequent spline/refined rendering."""
        self._spline_color = QColor(spline_color)
        self._refined_color = QColor(refined_color if refined_color is not None else spline_color)

    def clear_overlays(self) -> None:
        self._clear_path_items(self._spline_path_items)
        self._clear_path_items(self._refined_path_items)
        self._spline_path_items = []
        self._refined_path_items = []

    def clear_refined(self) -> None:
        """Remove only the refined (blue) overlay, keeping the spline."""
        self._clear_path_items(self._refined_path_items)
        self._refined_path_items = []

    def clear_seeds(self) -> None:
        for item in self._seed_items:
            self._scene.removeItem(item)
        self._seed_items.clear()
        self._seeds.clear()
        self.seeds_changed.emit()

    def remove_last_seed(self) -> bool:
        if not self._seed_items:
            return False
        self._scene.removeItem(self._seed_items.pop())
        self._seeds.pop()
        self.seeds_changed.emit()
        return True

    def remove_seed_item(self, item: QGraphicsEllipseItem) -> bool:
        if item not in self._seed_items:
            return False
        idx = self._seed_items.index(item)
        self._scene.removeItem(self._seed_items.pop(idx))
        self._seeds.pop(idx)
        self.seeds_changed.emit()
        return True

    def zoom_fit(self) -> None:
        """Reset zoom to fit the entire image."""
        if self._pixmap_item is not None:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    # ---- NaN exclusion windows ----------------------------------------

    def add_nan_window(self) -> None:
        """Add a new NaN-exclusion window (two draggable vertical bars)."""
        if self._image_shape is None:
            return
        rows, cols = self._image_shape
        # Place at 40%–60% of image width by default
        lx = int(cols * 0.4)
        rx = int(cols * 0.6)

        fill_color, edge_color = _NAN_PALETTE[len(self._nan_windows) % len(_NAN_PALETTE)]

        left_bar = _DraggableVLine(lx, rows, self, edge_color)
        left_bar.set_image_width(cols)
        right_bar = _DraggableVLine(rx, rows, self, edge_color)
        right_bar.set_image_width(cols)

        fill = self._scene.addRect(
            QRectF(lx, 0, rx - lx, rows),
            QPen(Qt.PenStyle.NoPen),
            QBrush(fill_color),
        )
        fill.setZValue(-1)  # behind curves / seeds

        self._scene.addItem(left_bar)
        self._scene.addItem(right_bar)

        win = NaNWindow(left_bar, right_bar, fill)
        self._nan_windows.append(win)
        self._refresh_nan_fill(win)
        self.nan_windows_changed.emit()

    def remove_last_nan_window(self) -> None:
        """Remove the most-recently added NaN window."""
        if not self._nan_windows:
            return
        win = self._nan_windows.pop()
        self._remove_item(win.left)
        self._remove_item(win.right)
        self._remove_item(win.fill)
        self.nan_windows_changed.emit()

    def clear_nan_windows(self) -> None:
        for win in self._nan_windows:
            self._remove_item(win.left)
            self._remove_item(win.right)
            self._remove_item(win.fill)
        self._nan_windows.clear()

    def get_nan_column_mask(self, width: int) -> NDArray[np.bool_]:
        """Return a boolean mask where True = column is inside a NaN window."""
        mask = np.zeros(width, dtype=np.bool_)
        for win in self._nan_windows:
            lo, hi = win.x_range
            lo = max(0, lo)
            hi = min(width - 1, hi)
            mask[lo:hi + 1] = True
        return mask

    @property
    def nan_windows(self) -> List[NaNWindow]:
        return list(self._nan_windows)

    def _on_nan_bar_moved(self) -> None:
        """Called when any NaN bar is dragged — update fill rectangles."""
        for win in self._nan_windows:
            self._refresh_nan_fill(win)
        self.nan_windows_changed.emit()

    def _refresh_nan_fill(self, win: NaNWindow) -> None:
        lx = win.left.x_pos
        rx = win.right.x_pos
        x0 = min(lx, rx)
        x1 = max(lx, rx)
        rows = self._image_shape[0] if self._image_shape else 100
        win.fill.setRect(QRectF(x0, 0, x1 - x0, rows))

    # ---- interaction --------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent):
        # Both buttons held → start panning
        both = Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton) and event.buttons() == both:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() == Qt.MouseButton.RightButton:
            seed_idx = self._find_seed_index_near_view_pos(float(event.pos().x()), float(event.pos().y()))
            if seed_idx is not None:
                self.remove_seed_item(self._seed_items[seed_idx])
                return
            self.remove_last_seed()
            return

        # Middle-click: reset zoom
        if event.button() == Qt.MouseButton.MiddleButton:
            self.zoom_fit()
            return

        ## Annotate single data point
        # Left-click: place seed (only if not clicking on a draggable bar) 
        if (
            event.button() == Qt.MouseButton.LeftButton
            and event.buttons() == Qt.MouseButton.LeftButton
            and not self._panning
            and self._pixmap_item is not None
        ):
            # Let scene handle movable items first
            item_under = self.itemAt(event.pos())
            if isinstance(item_under, _DraggableVLine):
                super().mousePressEvent(event)
                return
            scene_pos = self.mapToScene(event.pos())
            x, y = scene_pos.x(), scene_pos.y()
            rows, cols = self._image_shape
            if 0 <= x < cols and 0 <= y < rows:
                x_col = int(round(x))
                if any(int(round(seed_x)) == x_col for seed_x, _, _ in self._seeds):
                    return
                self._seeds.append((x, y, self._current_seed_class))
                seed_color = _CLASS_CURVE_COLORS.get(self._current_seed_class, SEED_COLOR)
                item = self._scene.addEllipse(
                    x - SEED_RADIUS,
                    y - SEED_RADIUS,
                    SEED_RADIUS * 2,
                    SEED_RADIUS * 2,
                    QPen(seed_color),
                    QBrush(seed_color),
                )
                self._seed_items.append(item)
                self.seeds_changed.emit()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._panning and self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._panning:
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        """Zoom with scroll wheel."""
        factor = 1.15
        if event.angleDelta().y() > 0:
            self.scale(factor, factor)
        else:
            self.scale(1 / factor, 1 / factor)

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)

    # ---- helpers ------------------------------------------------------

    def _add_curve(self, y_indices: NDArray, color: QColor, width: float,
                   nan_mask: NDArray[np.bool_] | None = None) -> QGraphicsPathItem:
        path = QPainterPath()
        in_segment = False
        for x in range(len(y_indices)):
            if nan_mask is not None and nan_mask[x]:
                in_segment = False
                continue
            if not in_segment:
                path.moveTo(QPointF(float(x), float(y_indices[x])))
                in_segment = True
            else:
                path.lineTo(QPointF(float(x), float(y_indices[x])))
        pen = QPen(color, width)
        pen.setCosmetic(True)  # constant screen-width regardless of zoom
        item = self._scene.addPath(path, pen)
        return item

    def _add_curve_for_class(
        self,
        y_indices: NDArray,
        class_by_column: NDArray[np.int32],
        class_id: int,
        color: QColor,
        width: float,
        nan_mask: NDArray[np.bool_] | None = None,
    ) -> QGraphicsPathItem:
        path = QPainterPath()
        in_segment = False
        for x in range(len(y_indices)):
            if nan_mask is not None and nan_mask[x]:
                in_segment = False
                continue
            if int(class_by_column[x]) != class_id:
                in_segment = False
                continue
            if not in_segment:
                path.moveTo(QPointF(float(x), float(y_indices[x])))
                in_segment = True
            else:
                path.lineTo(QPointF(float(x), float(y_indices[x])))
        pen = QPen(color, width)
        pen.setCosmetic(True)
        return self._scene.addPath(path, pen)

    def _find_seed_index_near_view_pos(self, view_x: float, view_y: float) -> int | None:
        """Return the nearest seed index within the on-screen hitbox, if any."""
        if not self._seeds:
            return None
        best_idx = None
        best_dist2 = float("inf")
        for i, (seed_x, seed_y, _) in enumerate(self._seeds):
            vp = self.mapFromScene(QPointF(seed_x, seed_y))
            dx = float(vp.x()) - view_x
            dy = float(vp.y()) - view_y
            dist2 = dx * dx + dy * dy
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_idx = i
        if best_idx is None:
            return None
        if best_dist2 <= float(SEED_HITBOX_RADIUS_PX * SEED_HITBOX_RADIUS_PX):
            return best_idx
        return None

    def _remove_item(self, item):
        if item is not None and item.scene() is not None:
            self._scene.removeItem(item)

    def _clear_path_items(self, items: List[QGraphicsPathItem]) -> None:
        for item in items:
            self._remove_item(item)
