"""Custom QGraphicsView for displaying M-scans and collecting seed points."""

from __future__ import annotations

from typing import List, Tuple, Optional

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
    QRubberBand,
)


SEED_RADIUS = 4
SEED_COLOR = QColor(255, 50, 50)
SPLINE_COLOR = QColor(0, 255, 100)
REFINED_COLOR = QColor(50, 150, 255)
NAN_FILL_COLOR = QColor(255, 120, 0, 60)   # semi-transparent orange
NAN_EDGE_COLOR = QColor(255, 120, 0, 200)


# ---------------------------------------------------------------------------
#  Draggable vertical bar (edge of a NaN-exclusion window)
# ---------------------------------------------------------------------------

class _DraggableVLine(QGraphicsLineItem):
    """A vertical line spanning the full image height, draggable along x."""

    def __init__(self, x: float, height: float, parent_viewer: "MScanViewer"):
        super().__init__(x, 0, x, height)
        self.setPen(QPen(NAN_EDGE_COLOR, 2))
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
            # Constrain to horizontal movement only, clamp to image
            new_x = max(0.0, min(value.x(), self._image_width - 1))
            clamped = QPointF(new_x, 0.0)
            # Notify viewer so it can redraw the fill rect
            self._viewer._on_nan_bar_moved()
            return clamped
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

        # Seed points as (x, y) image coordinates
        self._seeds: List[Tuple[float, float]] = []
        self._seed_items: List[QGraphicsEllipseItem] = []

        # Curve overlays
        self._spline_path_item: QGraphicsPathItem | None = None
        self._refined_path_item: QGraphicsPathItem | None = None

        # Rubber-band zoom state (right-click drag)
        self._rubber_band: Optional[QRubberBand] = None
        self._rb_origin = None  # viewport point

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
        self._spline_path_item = None
        self._refined_path_item = None

        rows, cols = data.shape
        self._image_shape = (rows, cols)

        # Normalise to 0-255 uint8
        lo, hi = float(data.min()), float(data.max())
        if hi - lo > 0:
            normed = ((data - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            normed = np.zeros_like(data, dtype=np.uint8)

        qimage = QImage(
            normed.data.tobytes(),
            cols,
            rows,
            cols,  # bytes-per-line for 8-bit grayscale
            QImage.Format.Format_Grayscale8,
        )
        pixmap = QPixmap.fromImage(qimage)
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    @property
    def seeds(self) -> List[Tuple[float, float]]:
        return list(self._seeds)

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

    def draw_spline(self, y_indices: NDArray) -> None:
        """Overlay a spline curve on the image."""
        self._remove_item(self._spline_path_item)
        self._spline_path_item = self._add_curve(y_indices, SPLINE_COLOR, 1.5)

    def draw_refined(self, y_indices: NDArray) -> None:
        """Overlay a refined boundary curve on the image."""
        self._remove_item(self._refined_path_item)
        self._refined_path_item = self._add_curve(y_indices, REFINED_COLOR, 1.5)

    def clear_overlays(self) -> None:
        self._remove_item(self._spline_path_item)
        self._spline_path_item = None
        self._remove_item(self._refined_path_item)
        self._refined_path_item = None

    def clear_refined(self) -> None:
        """Remove only the refined (blue) overlay, keeping the spline."""
        self._remove_item(self._refined_path_item)
        self._refined_path_item = None

    def clear_seeds(self) -> None:
        for item in self._seed_items:
            self._scene.removeItem(item)
        self._seed_items.clear()
        self._seeds.clear()
        self.seeds_changed.emit()

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

        left_bar = _DraggableVLine(lx, rows, self)
        left_bar.set_image_width(cols)
        right_bar = _DraggableVLine(rx, rows, self)
        right_bar.set_image_width(cols)

        fill = self._scene.addRect(
            QRectF(lx, 0, rx - lx, rows),
            QPen(Qt.PenStyle.NoPen),
            QBrush(NAN_FILL_COLOR),
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
        # Right-click: start rubber-band zoom
        if event.button() == Qt.MouseButton.RightButton:
            self._rb_origin = event.pos()
            if self._rubber_band is None:
                self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
            self._rubber_band.setGeometry(self._rb_origin.x(), self._rb_origin.y(), 0, 0)
            self._rubber_band.show()
            return

        # Middle-click: reset zoom
        if event.button() == Qt.MouseButton.MiddleButton:
            self.zoom_fit()
            return

        # Left-click: place seed (only if not clicking on a draggable bar)
        if event.button() == Qt.MouseButton.LeftButton and self._pixmap_item is not None:
            # Let scene handle movable items first
            item_under = self.itemAt(event.pos())
            if isinstance(item_under, _DraggableVLine):
                super().mousePressEvent(event)
                return
            scene_pos = self.mapToScene(event.pos())
            x, y = scene_pos.x(), scene_pos.y()
            rows, cols = self._image_shape
            if 0 <= x < cols and 0 <= y < rows:
                self._seeds.append((x, y))
                item = self._scene.addEllipse(
                    x - SEED_RADIUS,
                    y - SEED_RADIUS,
                    SEED_RADIUS * 2,
                    SEED_RADIUS * 2,
                    QPen(SEED_COLOR),
                    QBrush(SEED_COLOR),
                )
                self._seed_items.append(item)
                self.seeds_changed.emit()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._rubber_band is not None and self._rb_origin is not None and self._rubber_band.isVisible():
            self._rubber_band.setGeometry(
                min(self._rb_origin.x(), event.pos().x()),
                min(self._rb_origin.y(), event.pos().y()),
                abs(event.pos().x() - self._rb_origin.x()),
                abs(event.pos().y() - self._rb_origin.y()),
            )
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.RightButton and self._rubber_band is not None and self._rb_origin is not None:
            rect = self._rubber_band.geometry()
            self._rubber_band.hide()
            self._rb_origin = None
            if rect.width() > 5 and rect.height() > 5:
                scene_rect = self.mapToScene(rect).boundingRect()
                self.fitInView(scene_rect, Qt.AspectRatioMode.KeepAspectRatio)
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

    def _add_curve(self, y_indices: NDArray, color: QColor, width: float) -> QGraphicsPathItem:
        path = QPainterPath()
        path.moveTo(QPointF(0, float(y_indices[0])))
        for x in range(1, len(y_indices)):
            path.lineTo(QPointF(float(x), float(y_indices[x])))
        pen = QPen(color, width)
        pen.setCosmetic(True)  # constant screen-width regardless of zoom
        item = self._scene.addPath(path, pen)
        return item

    def _remove_item(self, item):
        if item is not None and item.scene() is not None:
            self._scene.removeItem(item)
