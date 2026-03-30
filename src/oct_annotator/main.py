"""Main application window and entry point for the OCT M-Scan Annotator."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import numpy as np
from numpy.typing import NDArray
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QComboBox,
    QStatusBar,
    QMessageBox,
)

from oct_annotator.viewer import MScanViewer
from oct_annotator.engine import fit_spline, refine_boundary, render_annotation_png

# Sentinel value written into uint16 annotations for NaN / excluded columns
NAN_SENTINEL: np.uint16 = np.uint16(65535)


class MainWindow(QMainWindow):
    def __init__(self, directory: str | None = None):
        super().__init__()
        self.setWindowTitle("OCT M-Scan Annotator")
        self.resize(1200, 700)

        # State
        self._npy_files: List[Path] = []
        self._current_data: NDArray | None = None
        self._spline_indices: NDArray | None = None
        self._refined_indices: NDArray | None = None

        self._build_ui()
        self._connect_signals()

        if directory:
            self._load_directory(directory)

    # ---- UI construction -----------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Top toolbar row
        toolbar = QHBoxLayout()
        root.addLayout(toolbar)

        self._btn_open = QPushButton("Open Folder…")
        toolbar.addWidget(self._btn_open)

        toolbar.addWidget(QLabel("File:"))
        self._combo_files = QComboBox()
        self._combo_files.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        toolbar.addWidget(self._combo_files, stretch=1)

        self._btn_fit = QPushButton("Fit Spline")
        self._btn_fit.setEnabled(False)
        toolbar.addWidget(self._btn_fit)

        self._btn_refine = QPushButton("Fine-tune")
        self._btn_refine.setEnabled(False)
        toolbar.addWidget(self._btn_refine)

        self._btn_reset_refine = QPushButton("Reset Fine-tuning")
        self._btn_reset_refine.setEnabled(False)
        toolbar.addWidget(self._btn_reset_refine)

        self._btn_save = QPushButton("Save")
        self._btn_save.setEnabled(False)
        toolbar.addWidget(self._btn_save)

        self._btn_clear = QPushButton("Clear Seeds")
        toolbar.addWidget(self._btn_clear)

        # Second toolbar row — zoom & NaN windows
        toolbar2 = QHBoxLayout()
        root.addLayout(toolbar2)

        self._btn_zoom_fit = QPushButton("Zoom Fit")
        self._btn_zoom_fit.setToolTip("Reset zoom to fit entire image (also: middle-click)")
        toolbar2.addWidget(self._btn_zoom_fit)

        toolbar2.addStretch(1)

        self._btn_add_nan = QPushButton("Add NaN Window")
        self._btn_add_nan.setEnabled(False)
        self._btn_add_nan.setToolTip("Add a pair of draggable vertical bars to mark excluded columns")
        toolbar2.addWidget(self._btn_add_nan)

        self._btn_remove_nan = QPushButton("Remove Last NaN Window")
        self._btn_remove_nan.setEnabled(False)
        toolbar2.addWidget(self._btn_remove_nan)

        self._btn_clear_nan = QPushButton("Clear All NaN Windows")
        self._btn_clear_nan.setEnabled(False)
        toolbar2.addWidget(self._btn_clear_nan)

        self._lbl_nan_info = QLabel("")
        toolbar2.addWidget(self._lbl_nan_info)

        # Viewer
        self._viewer = MScanViewer()
        root.addWidget(self._viewer, stretch=1)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage(
            "Open a folder containing .npy M-scan files to begin.  "
            "Right-drag to zoom into a region · Middle-click to reset zoom."
        )

    # ---- Signals -------------------------------------------------------

    def _connect_signals(self):
        self._btn_open.clicked.connect(self._on_open)
        self._combo_files.currentIndexChanged.connect(self._on_file_selected)
        self._btn_fit.clicked.connect(self._on_fit_spline)
        self._btn_refine.clicked.connect(self._on_refine)
        self._btn_reset_refine.clicked.connect(self._on_reset_refine)
        self._btn_save.clicked.connect(self._on_save)
        self._btn_clear.clicked.connect(self._on_clear)
        self._btn_zoom_fit.clicked.connect(self._viewer.zoom_fit)
        self._btn_add_nan.clicked.connect(self._on_add_nan_window)
        self._btn_remove_nan.clicked.connect(self._on_remove_nan_window)
        self._btn_clear_nan.clicked.connect(self._on_clear_nan_windows)
        self._viewer.seeds_changed.connect(self._on_seeds_changed)
        self._viewer.nan_windows_changed.connect(self._on_nan_windows_changed)

    # ---- Slots ---------------------------------------------------------

    def _on_open(self):
        directory = QFileDialog.getExistingDirectory(self, "Select M-scan directory")
        if directory:
            self._load_directory(directory)

    def _load_directory(self, directory: str):
        folder = Path(directory)
        self._npy_files = sorted(folder.glob("*.npy"))
        self._combo_files.blockSignals(True)
        self._combo_files.clear()
        for f in self._npy_files:
            self._combo_files.addItem(f.name)
        self._combo_files.blockSignals(False)

        if self._npy_files:
            self._combo_files.setCurrentIndex(0)
            self._on_file_selected(0)
        else:
            self._status.showMessage("No .npy files found in the selected folder.")

    def _on_file_selected(self, index: int):
        if index < 0 or index >= len(self._npy_files):
            return
        path = self._npy_files[index]
        try:
            self._current_data = np.load(str(path)).astype(np.float64)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return

        self._spline_indices = None
        self._refined_indices = None
        self._btn_fit.setEnabled(False)
        self._btn_refine.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._btn_add_nan.setEnabled(True)
        self._btn_remove_nan.setEnabled(False)
        self._btn_clear_nan.setEnabled(False)
        self._lbl_nan_info.setText("")

        self._viewer.set_image(self._current_data)
        self._status.showMessage(
            f"Loaded {path.name}  —  shape {self._current_data.shape}  |  "
            "Click on the image to place seed points, then Fit Spline."
        )

    def _on_seeds_changed(self):
        n = len(self._viewer.seeds)
        self._btn_fit.setEnabled(n >= 2)
        self._status.showMessage(f"{n} seed point(s) placed.")

    def _on_fit_spline(self):
        seeds = self._viewer.seeds
        if len(seeds) < 2:
            return
        xs = np.array([s[0] for s in seeds])
        ys = np.array([s[1] for s in seeds])
        width = self._viewer.image_width
        try:
            self._spline_indices = fit_spline(xs, ys, width)
        except Exception as exc:
            QMessageBox.warning(self, "Spline error", str(exc))
            return
        nan_mask = self._viewer.get_nan_column_mask(width)
        self._viewer.draw_spline(self._spline_indices, nan_mask)
        self._viewer.clear_refined()
        self._btn_refine.setEnabled(True)
        self._btn_save.setEnabled(True)
        self._btn_reset_refine.setEnabled(False)
        self._refined_indices = None
        self._status.showMessage("Spline fitted. Press Fine-tune or Save.")

    def _on_refine(self):
        if self._spline_indices is None or self._current_data is None:
            return
        self._refined_indices = refine_boundary(
            self._current_data, self._spline_indices
        )
        nan_mask = self._viewer.get_nan_column_mask(self._viewer.image_width)
        self._viewer.draw_refined(self._refined_indices, nan_mask)
        self._btn_save.setEnabled(True)
        self._btn_reset_refine.setEnabled(True)
        self._status.showMessage("Boundary refined via gradient snap. Press Save to export.")

    def _on_reset_refine(self):
        """Discard the refined curve and revert to the raw spline."""
        self._refined_indices = None
        self._viewer.clear_refined()
        self._btn_reset_refine.setEnabled(False)
        if self._spline_indices is not None:
            self._btn_save.setEnabled(True)
            self._status.showMessage("Fine-tuning reset. Showing raw spline.")
        else:
            self._btn_save.setEnabled(False)

    def _on_save(self):
        indices = self._refined_indices if self._refined_indices is not None else self._spline_indices
        if indices is None or self._current_data is None:
            return

        current_idx = self._combo_files.currentIndex()
        src_path = self._npy_files[current_idx]
        m_scan = self._current_data
        rows, cols = m_scan.shape

        # Ensure annotation length matches the number of A-scans (columns)
        ann = indices.astype(np.uint16).reshape(-1)
        if len(ann) != cols:
            ann_resized = np.full(cols, NAN_SENTINEL, dtype=np.uint16)
            n = min(len(ann), cols)
            ann_resized[:n] = ann[:n]
            ann = ann_resized

        # Apply NaN-window sentinel
        nan_mask = self._viewer.get_nan_column_mask(cols)
        ann[nan_mask] = NAN_SENTINEL
        to_save = ann.reshape(-1, 1)

        # Save .npy annotation
        out_npy = src_path.parent / (src_path.stem + "_annotations.npy")
        np.save(str(out_npy), to_save)

        # Save .png visual overlay
        out_png = src_path.parent / (src_path.stem + "_annotations.png")
        render_annotation_png(m_scan, ann, nan_mask, str(out_png))

        n_nan = int(nan_mask.sum())
        nan_note = f"  ({n_nan} cols excluded)" if n_nan else ""
        self._status.showMessage(
            f"Saved \u2192 {out_npy.name} + {out_png.name}  "
            f"(shape {to_save.shape}){nan_note}"
        )

    # ---- NaN window slots ----------------------------------------------

    def _on_add_nan_window(self):
        self._viewer.add_nan_window()

    def _on_remove_nan_window(self):
        self._viewer.remove_last_nan_window()

    def _on_clear_nan_windows(self):
        self._viewer.clear_nan_windows()
        self._viewer.nan_windows_changed.emit()

    def _on_nan_windows_changed(self):
        n = len(self._viewer.nan_windows)
        self._btn_remove_nan.setEnabled(n > 0)
        self._btn_clear_nan.setEnabled(n > 0)
        self._lbl_nan_info.setText(f"{n} NaN window(s)" if n else "")
        # Redraw active curves with updated NaN gaps
        self._redraw_curves()

    def _redraw_curves(self):
        """Re-render spline/refined overlays respecting current NaN mask."""
        width = self._viewer.image_width
        nan_mask = self._viewer.get_nan_column_mask(width) if width > 0 else None
        if self._spline_indices is not None:
            self._viewer.draw_spline(self._spline_indices, nan_mask)
        if self._refined_indices is not None:
            self._viewer.draw_refined(self._refined_indices, nan_mask)

    def _on_clear(self):
        self._viewer.clear_seeds()
        self._viewer.clear_overlays()
        self._viewer.clear_nan_windows()
        self._spline_indices = None
        self._refined_indices = None
        self._btn_fit.setEnabled(False)
        self._btn_refine.setEnabled(False)
        self._btn_reset_refine.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._btn_remove_nan.setEnabled(False)
        self._btn_clear_nan.setEnabled(False)
        self._lbl_nan_info.setText("")
        self._status.showMessage("Seeds cleared.")


def run_app():
    """CLI entry point."""
    app = QApplication(sys.argv)
    directory = sys.argv[1] if len(sys.argv) > 1 else None
    win = MainWindow(directory=directory)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()
