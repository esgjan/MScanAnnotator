"""Main application window and entry point for the OCT M-Scan Annotator."""

from __future__ import annotations

import os
import shutil
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

if __package__ in (None, ""):
    # Support running this file directly: `python src/oct_annotator/main.py`.
    package_root = Path(__file__).resolve().parents[1]
    package_root_str = str(package_root)
    if package_root_str not in sys.path:
        sys.path.insert(0, package_root_str)

from oct_annotator.viewer import MScanViewer
from oct_annotator.engine import fit_spline, refine_boundary, render_annotation_png

# Sentinel value written into uint16 annotations for NaN / excluded columns
NAN_SENTINEL: np.uint16 = np.uint16(65535)
DEFAULT_SCAN_DIRECTORY = Path(r"D:\iiOCT_data\npy_raw_snippets")


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

    @staticmethod
    def _is_source_scan(path: Path) -> bool:
        return path.suffix == ".npy" and not path.stem.endswith("_annotations")

    @staticmethod
    def _annotation_output_dir(source_path: Path) -> Path:
        return source_path.parent / "annotated"

    @staticmethod
    def _too_hard_output_dir(source_path: Path) -> Path:
        # Keep flagged files alongside the current experiment folder.
        return source_path.parent / "2hard2label"

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

        self._btn_refine = QPushButton("Fine-tune")
        self._btn_refine.setEnabled(False)
        self._btn_refine.setShortcut("A")
        self._btn_refine.setToolTip("Fine-tune boundary (A)")
        toolbar.addWidget(self._btn_refine)

        self._btn_reset_refine = QPushButton("Reset Fine-tuning")
        self._btn_reset_refine.setEnabled(False)
        toolbar.addWidget(self._btn_reset_refine)

        self._btn_save = QPushButton("Save")
        self._btn_save.setEnabled(False)
        self._btn_save.setShortcut("D")
        self._btn_save.setToolTip("Save annotation (D)")
        toolbar.addWidget(self._btn_save)

        self._btn_flag = QPushButton("Too Hard (F)")
        self._btn_flag.setEnabled(False)
        self._btn_flag.setShortcut("F")
        self._btn_flag.setToolTip("Flag scan as too hard to label – copies it to 2hard2label/ and skips to next (F)")
        self._btn_flag.setStyleSheet("color: #c0392b; font-weight: bold;")
        toolbar.addWidget(self._btn_flag)

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
            "Right-click removes the last seed · Middle-click resets zoom."
        )

    # ---- Signals -------------------------------------------------------

    def _connect_signals(self):
        self._btn_open.clicked.connect(self._on_open)
        self._combo_files.currentIndexChanged.connect(self._on_file_selected)
        self._btn_refine.clicked.connect(self._on_refine)
        self._btn_reset_refine.clicked.connect(self._on_reset_refine)
        self._btn_save.clicked.connect(self._on_save)
        self._btn_flag.clicked.connect(self._on_flag_too_hard)
        self._btn_clear.clicked.connect(self._on_clear)
        self._btn_zoom_fit.clicked.connect(self._viewer.zoom_fit)
        self._btn_add_nan.clicked.connect(self._on_add_nan_window)
        self._btn_remove_nan.clicked.connect(self._on_remove_nan_window)
        self._btn_clear_nan.clicked.connect(self._on_clear_nan_windows)
        self._viewer.seeds_changed.connect(self._on_seeds_changed)
        self._viewer.nan_windows_changed.connect(self._on_nan_windows_changed)

    # ---- Slots ---------------------------------------------------------

    def _on_open(self):
        start_dir = self._initial_open_directory()
        directory = QFileDialog.getExistingDirectory(self, "Select M-scan directory", str(start_dir))
        if directory:
            self._load_directory(directory)

    def _initial_open_directory(self) -> Path:
        if self._npy_files:
            return self._npy_files[0].parent
        if DEFAULT_SCAN_DIRECTORY.exists():
            return DEFAULT_SCAN_DIRECTORY
        return Path.home()

    def _load_directory(self, directory: str):
        folder = Path(directory)
        self._npy_files = sorted(path for path in folder.glob("*.npy") if self._is_source_scan(path))
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

    def _find_next_experiment_directory(self, current_directory: Path) -> Path | None:
        parent = current_directory.parent
        sibling_dirs = sorted(path for path in parent.iterdir() if path.is_dir())

        try:
            current_index = sibling_dirs.index(current_directory)
        except ValueError:
            return None

        for candidate in sibling_dirs[current_index + 1:]:
            has_source_scans = any(
                self._is_source_scan(path) for path in candidate.glob("*.npy")
            )
            if has_source_scans:
                return candidate
        return None

    def _on_file_selected(self, index: int):
        if index < 0 or index >= len(self._npy_files):
            return
        self._btn_flag.setEnabled(True)
        path = self._npy_files[index]
        try:
            self._current_data = np.load(str(path)).astype(np.float64)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return

        self._spline_indices = None
        self._refined_indices = None
        self._btn_refine.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._btn_add_nan.setEnabled(True)
        self._btn_remove_nan.setEnabled(False)
        self._btn_clear_nan.setEnabled(False)
        self._lbl_nan_info.setText("")

        self._viewer.set_image(self._current_data)
        self._status.showMessage(
            f"Loaded {path.name}  —  shape {self._current_data.shape}  |  "
            "Click on the image to place seed points. Spline updates automatically."
        )

    def _on_seeds_changed(self):
        n = len(self._viewer.seeds)
        if n >= 2:
            self._on_fit_spline()
            return
        self._spline_indices = None
        self._refined_indices = None
        self._viewer.clear_overlays()
        self._btn_refine.setEnabled(False)
        self._btn_reset_refine.setEnabled(False)
        self._btn_save.setEnabled(False)
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
            self._status.showMessage(f"Spline update skipped: {exc}")
            return
        nan_mask = self._viewer.get_nan_column_mask(width)
        self._viewer.draw_spline(self._spline_indices, nan_mask)
        self._viewer.clear_refined()
        self._btn_refine.setEnabled(True)
        self._btn_save.setEnabled(True)
        self._btn_reset_refine.setEnabled(False)
        self._refined_indices = None
        self._status.showMessage("Spline updated automatically. Press Fine-tune or Save.")

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
        out_dir = self._annotation_output_dir(src_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_npy = out_dir / (src_path.stem + "_annotations.npy")
        np.save(str(out_npy), to_save)

        # Save .png visual overlay
        out_png = out_dir / (src_path.stem + "_annotations.png")
        render_annotation_png(m_scan, ann, nan_mask, str(out_png))

        n_nan = int(nan_mask.sum())
        nan_note = f"  ({n_nan} cols excluded)" if n_nan else ""
        self._status.showMessage(
            f"Saved \u2192 {out_dir.name}/{out_npy.name} + {out_png.name}  "
            f"(shape {to_save.shape}){nan_note}"
        )

        self._load_next_file(current_idx)

    def _refresh_file_list(self) -> None:
        """Re-scan the current directory and update the dropdown, keeping selection."""
        if not self._npy_files:
            return
        folder = self._npy_files[0].parent
        current_name = self._combo_files.currentText()
        self._npy_files = sorted(path for path in folder.glob("*.npy") if self._is_source_scan(path))
        self._combo_files.blockSignals(True)
        self._combo_files.clear()
        restore_idx = 0
        for i, f in enumerate(self._npy_files):
            self._combo_files.addItem(f.name)
            if f.name == current_name:
                restore_idx = i
        self._combo_files.setCurrentIndex(restore_idx)
        self._combo_files.blockSignals(False)

    def _on_flag_too_hard(self):
        current_idx = self._combo_files.currentIndex()
        if current_idx < 0 or current_idx >= len(self._npy_files):
            return
        src_path = self._npy_files[current_idx]
        dest_dir = self._too_hard_output_dir(src_path)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Copy original NPY into the experiment-local 2hard2label folder.
        dest_path = dest_dir / src_path.name
        shutil.copy2(str(src_path), str(dest_path))

        # Export an all-NaN annotation so this sample is marked explicitly as "too hard".
        if self._current_data is not None:
            cols = self._current_data.shape[1]
            blank_ann = np.full(cols, NAN_SENTINEL, dtype=np.uint16)
            out_ann_npy = dest_dir / (src_path.stem + "_annotations.npy")
            np.save(str(out_ann_npy), blank_ann.reshape(-1, 1))

            nan_mask = np.zeros(cols, dtype=np.bool_)
            out_png = dest_dir / (src_path.stem + "_annotations.png")
            render_annotation_png(self._current_data, blank_ann, nan_mask, str(out_png))
            self._status.showMessage(
                f"Flagged → 2hard2label/{src_path.name} + {out_ann_npy.name} + {out_png.name}  – skipping to next file."
            )
        else:
            self._status.showMessage(f"Flagged → 2hard2label/{src_path.name}  – skipping to next file.")

        self._load_next_file(current_idx)

    def _load_next_file(self, current_idx: int) -> None:
        if current_idx + 1 >= len(self._npy_files):
            if not self._npy_files:
                return
            current_directory = self._npy_files[0].parent
            next_directory = self._find_next_experiment_directory(current_directory)
            if next_directory is None:
                self._status.showMessage("Saved last file in this experiment. No next experiment folder found.")
                return
            self._load_directory(str(next_directory))
            self._status.showMessage(
                f"Finished {current_directory.name}. Opened next experiment {next_directory.name} at first snippet."
            )
            return
        self._combo_files.setCurrentIndex(current_idx + 1)

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
