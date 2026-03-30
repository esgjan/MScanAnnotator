"""Custom QGraphicsView for displaying M-scans and collecting seed points."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from PyQt6.QtCore import Qt, pyqtSignal, QPointF
from PyQt6.QtGui import QImage, QPixmap, QPen, QColor, QBrush, QPainterPath
from PyQt6.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsEllipseItem,
    QGraphicsPathItem,
)


SEED_RADIUS = 4
SEED_COLOR = QColor(255, 50, 50)
SPLINE_COLOR = QColor(0, 255, 100)
REFINED_COLOR = QColor(50, 150, 255)


class MScanViewer(QGraphicsView):
    """Interactive viewer that displays a numpy M-scan and lets users place seed points."""

    seeds_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(
            self.renderHints()
        )
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

    # ---- public API ---------------------------------------------------

    def set_image(self, data: NDArray) -> None:
        """Display a 2-D float array as a grayscale image."""
        self.clear_overlays()
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

    # ---- interaction --------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._pixmap_item is not None:
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

    def wheelEvent(self, event):
        """Zoom with scroll wheel."""
        factor = 1.15
        if event.angleDelta().y() > 0:
            self.scale(factor, factor)
        else:
            self.scale(1 / factor, 1 / factor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap_item is not None:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

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
