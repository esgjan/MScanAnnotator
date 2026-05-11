"""Main application window and entry point for the OCT M-Scan Annotator."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import List

import numpy as np
from numpy.typing import NDArray
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QColor, QShortcut, QKeySequence
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
    QDoubleSpinBox,
    QLineEdit,
    QSpinBox,
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
from oct_annotator.engine import fit_spline, refine_boundary, render_annotation_tiff

# Sentinel value written into uint16 annotations for NaN / excluded columns
NAN_SENTINEL: np.uint16 = np.uint16(65535)
DEFAULT_FINE_TUNE_RADIUS = 5
DEFAULT_PREVIEW_CONTRAST = 1.0
DEFAULT_PREVIEW_GAMMA = 1.0

NAN_LABEL_TEXT = "nan"

# class_id -> (name, UI color, PNG RGB color, label-mask value)
_CLASS_STYLE: dict[int, tuple[str, QColor, tuple[int, int, int], float]] = {
    1: ("Class 1", QColor(0, 190, 90), (0, 190, 90), 1.0),
    2: ("Class 2", QColor(245, 205, 0), (245, 205, 0), 2.0),
    3: ("Class 3", QColor(255, 165, 50), (255, 165, 50), 3.0),
    4: ("Class 4", QColor(40, 120, 255), (40, 120, 255), 4.0),
    5: (NAN_LABEL_TEXT, QColor(210, 40, 40), (210, 40, 40), float("nan")),
}


def _format_label_value(class_value: float) -> str:
    return NAN_LABEL_TEXT if np.isnan(class_value) else str(int(class_value))


class MainWindow(QMainWindow):
    def __init__(self, directory: str | None = None):
        super().__init__()
        self.setWindowTitle("OCT M-Scan Annotator")
        self.resize(1200, 700)
        self._settings = QSettings("MScanAnnotator", "oct-annotator")

        # State
        self._npy_files: List[Path] = []
        self._current_data: NDArray | None = None
        self._spline_indices: NDArray | None = None
        self._refined_indices: NDArray | None = None
        self._active_class: int = 1
        self._fixed_spline_mode: bool = False
        self._fine_tune_radius: int = int(
            self._settings.value("preview/fine_tune_radius", DEFAULT_FINE_TUNE_RADIUS, type=int)
        )
        self._preview_contrast: float = float(
            self._settings.value("preview/contrast", DEFAULT_PREVIEW_CONTRAST, type=float)
        )
        self._preview_gamma: float = float(
            self._settings.value("preview/gamma", DEFAULT_PREVIEW_GAMMA, type=float)
        )
        self._output_root_directory: Path = Path.home()
        saved_output_dir = self._settings.value("paths/output_root", "", type=str)
        if saved_output_dir:
            self._output_root_directory = Path(saved_output_dir)

        self._build_ui()
        self._connect_signals()

        if directory:
            self._load_directory(directory)

    @staticmethod
    def _is_source_scan(path: Path) -> bool:
        return path.suffix == ".npy" and not path.stem.endswith("_annotations")

    @staticmethod
    def _experiment_output_parts(source_path: Path) -> tuple[str, ...]:
        """Extract the experiment folder structure from source path.
        
        Returns the parent folder (experiment name) and original subfolder name
        to preserve hierarchy in output.
        """
        source_dir = source_path.parent
        parent_dir = source_dir.parent
        if parent_dir == source_dir:
            return (source_dir.name,)
        if not parent_dir.name:
            return (source_dir.name,)
        return (parent_dir.name, source_dir.name)

    def _output_dir_for_source(self, source_path: Path) -> Path:
        """Get output directory preserving source hierarchy under selected save root."""
        return self._output_root_directory.joinpath(*self._experiment_output_parts(source_path))

    def _set_output_root_directory(self, directory: Path) -> None:
        self._output_root_directory = directory
        if hasattr(self, "_edit_output_dir"):
            self._edit_output_dir.setText(str(directory))

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

        self._btn_fixed_splines = QPushButton("Fixed Splines")
        self._btn_fixed_splines.setCheckable(True)
        self._btn_fixed_splines.setShortcut("S")
        self._btn_fixed_splines.setToolTip(
            "Seeds placed while enabled keep their spline regions fixed during fine-tuning (S)."
        )
        toolbar.addWidget(self._btn_fixed_splines)

        self._lbl_class = QLabel("")
        self._lbl_class.setMinimumWidth(160)
        toolbar.addWidget(self._lbl_class)

        self._shortcut_class_1 = QShortcut(QKeySequence("1"), self)
        self._shortcut_class_1.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._shortcut_class_1.activated.connect(lambda: self._set_active_class(1))
        self._shortcut_class_2 = QShortcut(QKeySequence("2"), self)
        self._shortcut_class_2.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._shortcut_class_2.activated.connect(lambda: self._set_active_class(2))
        self._shortcut_class_3 = QShortcut(QKeySequence("3"), self)
        self._shortcut_class_3.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._shortcut_class_3.activated.connect(lambda: self._set_active_class(3))
        self._shortcut_class_4 = QShortcut(QKeySequence("4"), self)
        self._shortcut_class_4.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._shortcut_class_4.activated.connect(lambda: self._set_active_class(4))
        self._shortcut_class_5 = QShortcut(QKeySequence("5"), self)
        self._shortcut_class_5.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._shortcut_class_5.activated.connect(lambda: self._set_active_class(5))

        # Second toolbar row — zoom & NaN windows
        toolbar2 = QHBoxLayout()
        root.addLayout(toolbar2)

        toolbar2.addWidget(QLabel("Save root:"))
        self._edit_output_dir = QLineEdit()
        self._edit_output_dir.setReadOnly(True)
        self._edit_output_dir.setText(str(self._output_root_directory))
        self._edit_output_dir.setCursor(Qt.CursorShape.PointingHandCursor)
        self._edit_output_dir.setToolTip("Click to choose save root folder")
        toolbar2.addWidget(self._edit_output_dir, stretch=1)

        self._btn_zoom_fit = QPushButton("Zoom Fit")
        self._btn_zoom_fit.setToolTip("Reset zoom to fit entire image (also: middle-click)")
        toolbar2.addWidget(self._btn_zoom_fit)

        toolbar2.addWidget(QLabel("Fine-tune band: +/-"))
        self._spin_fine_tune_radius = QSpinBox()
        self._spin_fine_tune_radius.setMinimum(0)
        self._spin_fine_tune_radius.setMaximum(100)
        self._spin_fine_tune_radius.setValue(self._fine_tune_radius)
        self._spin_fine_tune_radius.setToolTip(
            "Limit gradient fine-tuning to this many pixels above and below the spline."
        )
        toolbar2.addWidget(self._spin_fine_tune_radius)
        toolbar2.addWidget(QLabel("px"))

        toolbar2.addWidget(QLabel("Preview contrast:"))
        self._spin_preview_contrast = QDoubleSpinBox()
        self._spin_preview_contrast.setDecimals(2)
        self._spin_preview_contrast.setMinimum(0.10)
        self._spin_preview_contrast.setMaximum(4.00)
        self._spin_preview_contrast.setSingleStep(0.10)
        self._spin_preview_contrast.setValue(self._preview_contrast)
        self._spin_preview_contrast.setToolTip(
            "Adjust preview/export contrast live. 1.0 keeps the default display contrast."
        )
        toolbar2.addWidget(self._spin_preview_contrast)

        toolbar2.addWidget(QLabel("Gamma:"))
        self._spin_preview_gamma = QDoubleSpinBox()
        self._spin_preview_gamma.setDecimals(2)
        self._spin_preview_gamma.setMinimum(0.10)
        self._spin_preview_gamma.setMaximum(5.00)
        self._spin_preview_gamma.setSingleStep(0.10)
        self._spin_preview_gamma.setValue(self._preview_gamma)
        self._spin_preview_gamma.setToolTip(
            "Gamma correction: >1 brightens dark regions, <1 darkens them. 1.0 = no correction."
        )
        toolbar2.addWidget(self._spin_preview_gamma)

        # Viewer
        self._viewer = MScanViewer()
        self._viewer.set_preview_contrast(self._preview_contrast)
        self._viewer.set_preview_gamma(self._preview_gamma)
        self._viewer.set_class_colors(
            {class_id: color_qt for class_id, (_, color_qt, _, _) in _CLASS_STYLE.items()}
        )
        root.addWidget(self._viewer, stretch=1)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage(
            "Open a folder containing .npy M-scan files to begin.  "
            "Right-click removes the last seed · Middle-click resets zoom."
        )
        self._apply_active_class_style()

    # ---- Signals -------------------------------------------------------

    def _connect_signals(self):
        self._btn_open.clicked.connect(self._on_open)
        self._edit_output_dir.mousePressEvent = lambda _: self._on_choose_output_directory()
        self._combo_files.currentIndexChanged.connect(self._on_file_selected)
        self._btn_refine.clicked.connect(self._on_refine)
        self._btn_reset_refine.clicked.connect(self._on_reset_refine)
        self._btn_save.clicked.connect(self._on_save)
        self._btn_flag.clicked.connect(self._on_flag_too_hard)
        self._btn_clear.clicked.connect(self._on_clear)
        self._btn_fixed_splines.toggled.connect(self._on_toggle_fixed_splines)
        self._btn_zoom_fit.clicked.connect(self._viewer.zoom_fit)
        self._viewer.seeds_changed.connect(self._on_seeds_changed)
        self._spin_fine_tune_radius.valueChanged.connect(self._on_fine_tune_radius_changed)
        self._spin_preview_contrast.valueChanged.connect(self._on_preview_contrast_changed)
        self._spin_preview_gamma.valueChanged.connect(self._on_preview_gamma_changed)

    # ---- Slots ---------------------------------------------------------

    def _on_open(self):
        start_dir = self._initial_open_directory()
        directory = QFileDialog.getExistingDirectory(self, "Select M-scan directory", str(start_dir))
        if directory:
            self._load_directory(directory)

    def _on_choose_output_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select save root directory",
            str(self._output_root_directory),
        )
        if directory:
            self._set_output_root_directory(Path(directory))
            self._settings.setValue("paths/output_root", str(self._output_root_directory))
            self._status.showMessage(f"Save root set to {self._output_root_directory}")

    def _initial_open_directory(self) -> Path:
        if self._npy_files:
            return self._npy_files[0].parent
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

        self._apply_display_data()
        self._apply_active_class_style()

    def _on_fine_tune_radius_changed(self, value: int) -> None:
        self._fine_tune_radius = int(value)
        self._settings.setValue("preview/fine_tune_radius", self._fine_tune_radius)
        if self._spline_indices is not None:
            self._on_refine()
        elif self._current_data is not None:
            self._status.showMessage(
                f"Fine-tune band set to +/-{self._fine_tune_radius} px."
            )

    def _on_preview_contrast_changed(self, value: float) -> None:
        self._preview_contrast = float(value)
        self._settings.setValue("preview/contrast", self._preview_contrast)
        self._viewer.set_preview_contrast(self._preview_contrast)
        if self._current_data is not None:
            self._status.showMessage(
                f"Contrast: {self._preview_contrast:.2f}  Gamma: {self._preview_gamma:.2f}  – TIFF exports will match."
            )

    def _on_preview_gamma_changed(self, value: float) -> None:
        self._preview_gamma = float(value)
        self._settings.setValue("preview/gamma", self._preview_gamma)
        self._viewer.set_preview_gamma(self._preview_gamma)
        if self._current_data is not None:
            self._status.showMessage(
                f"Contrast: {self._preview_contrast:.2f}  Gamma: {self._preview_gamma:.2f}  – TIFF exports will match."
            )

    def _on_toggle_fixed_splines(self, enabled: bool) -> None:
        self._fixed_spline_mode = bool(enabled)
        self._viewer.set_current_seed_fixed(enabled)
        self._status.showMessage(
            "Fixed spline mode enabled." if enabled else "Fixed spline mode disabled."
        )

    def _apply_display_data(self) -> None:
        """Display the current M-scan and refresh the viewer."""
        if self._current_data is None:
            return
        self._viewer.set_image(self._current_data)
        path = self._npy_files[self._combo_files.currentIndex()]
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
        class_map = self._build_class_map(width)
        self._viewer.draw_spline_classified(self._spline_indices, class_map, nan_mask)
        self._btn_refine.setEnabled(True)
        self._btn_save.setEnabled(True)
        self._btn_reset_refine.setEnabled(False)
        self._refined_indices = None
        self._status.showMessage("Spline updated. Running fine-tune…")
        self._on_refine()

    def _on_refine(self):
        if self._spline_indices is None or self._current_data is None:
            return
        width = self._viewer.image_width
        fixed_mask = self._build_fixed_spline_mask(width)
        self._refined_indices = refine_boundary(
            self._current_data,
            self._spline_indices,
            search_radius=self._fine_tune_radius,
            fixed_mask=fixed_mask,
        )
        nan_mask = self._viewer.get_nan_column_mask(width)
        class_map = self._build_class_map(width)
        self._viewer.draw_refined_classified(self._refined_indices, class_map, nan_mask, fixed_mask)
        self._btn_save.setEnabled(True)
        self._btn_reset_refine.setEnabled(True)
        self._status.showMessage(
            f"Boundary refined via gradient snap within +/-{self._fine_tune_radius} px. Press Save to export."
        )

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
        orig_cols = m_scan.shape[1]
        class_map = self._build_class_map(orig_cols)
        ann = indices.astype(np.uint16).reshape(-1)
        if len(ann) != orig_cols:
            ann_resized = np.full(orig_cols, NAN_SENTINEL, dtype=np.uint16)
            n = min(len(ann), orig_cols)
            ann_resized[:n] = ann[:n]
            ann = ann_resized

        nan_mask = self._viewer.get_nan_column_mask(orig_cols)
        class5_mask = class_map == 5
        ann[class5_mask] = NAN_SENTINEL
        ann[nan_mask] = NAN_SENTINEL

        # Save class label mask .npy with per-region values from seed classes.
        label_mask = np.empty(orig_cols, dtype=np.float32)
        label_mask[class_map == 1] = 1.0
        label_mask[class_map == 2] = 2.0
        label_mask[class_map == 3] = 3.0
        label_mask[class_map == 4] = 4.0
        label_mask[class_map == 5] = np.nan
        label_mask[nan_mask] = np.nan

        # Combine annotations and labels into a single 2D array
        # Column 0: boundary indices, Column 1: class labels
        combined_data = np.column_stack([ann.astype(np.float32), label_mask])

        out_dir = self._output_dir_for_source(src_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Copy original snippet to output folder
        copied_snippet = out_dir / src_path.name
        if src_path.resolve() != copied_snippet.resolve():
            shutil.copy2(src_path, copied_snippet)

        out_combined_npy = out_dir / (src_path.stem + "_annotations.npy")
        np.save(str(out_combined_npy), combined_data)

        # Save .tiff visual overlay with class colors per seeded region.
        out_tiff = out_dir / (src_path.stem + "_annotations.tiff")
        png_class_colors = {
            class_id: rgb
            for class_id, (_, _, rgb, _) in _CLASS_STYLE.items()
        }
        render_annotation_tiff(
            m_scan,
            ann,
            nan_mask,
            str(out_tiff),
            contrast=self._preview_contrast,
            gamma=self._preview_gamma,
            class_by_column=class_map,
            class_colors=png_class_colors,
        )

        n_nan = int(nan_mask.sum()) + int(class5_mask.sum())
        nan_note = f"  ({n_nan} cols {NAN_LABEL_TEXT})" if n_nan else ""
        self._status.showMessage(
            f"Saved \u2192 {out_dir}/{copied_snippet.name} + {out_combined_npy.name} + {out_tiff.name}  "
            f"(shape {combined_data.shape}){nan_note}"
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

        # Create all-NaN annotation for this OCT without copying to 2hard2label folder.
        if self._current_data is not None:
            cols = self._current_data.shape[1]
            blank_ann = np.full(cols, NAN_SENTINEL, dtype=np.uint16)
            
            # Save to output directory using experiment structure
            out_dir = self._output_dir_for_source(src_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy original snippet
            copied_snippet = out_dir / src_path.name
            if src_path.resolve() != copied_snippet.resolve():
                shutil.copy2(src_path, copied_snippet)
            
            out_ann_npy = out_dir / (src_path.stem + "_annotations.npy")
            np.save(str(out_ann_npy), blank_ann.reshape(-1, 1))

            nan_mask = np.zeros(cols, dtype=np.bool_)
            out_tiff = out_dir / (src_path.stem + "_annotations.tiff")
            render_annotation_tiff(
                self._current_data,
                blank_ann,
                nan_mask,
                str(out_tiff),
                contrast=self._preview_contrast,
                gamma=self._preview_gamma,
            )
            self._status.showMessage(
                f"Marked as too hard – all boundaries set to NaN. Saved {copied_snippet.name}, {out_ann_npy.name}, and {out_tiff.name} to {out_dir}  – skipping to next file."
            )
        else:
            self._status.showMessage("No data loaded – cannot mark as too hard.")

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

    def _redraw_curves(self):
        """Re-render spline/refined overlays respecting current NaN mask."""
        width = self._viewer.image_width
        nan_mask = self._viewer.get_nan_column_mask(width) if width > 0 else None
        class_map = self._build_class_map(width) if width > 0 else None
        fixed_mask = self._build_fixed_spline_mask(width) if width > 0 else None
        if self._spline_indices is not None:
            if class_map is not None:
                self._viewer.draw_spline_classified(self._spline_indices, class_map, nan_mask)
            else:
                self._viewer.draw_spline(self._spline_indices, nan_mask)
        if self._refined_indices is not None:
            if class_map is not None:
                self._viewer.draw_refined_classified(self._refined_indices, class_map, nan_mask, fixed_mask)
            else:
                self._viewer.draw_refined(self._refined_indices, nan_mask, fixed_mask)

    def _on_clear(self):
        self._viewer.clear_seeds()
        self._viewer.clear_overlays()
        self._viewer.clear_nan_windows()
        self._spline_indices = None
        self._refined_indices = None
        self._btn_refine.setEnabled(False)
        self._btn_reset_refine.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._status.showMessage("Seeds cleared.")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_1:
            self._set_active_class(1)
            return
        if event.key() == Qt.Key.Key_2:
            self._set_active_class(2)
            return
        if event.key() == Qt.Key.Key_3:
            self._set_active_class(3)
            return
        if event.key() == Qt.Key.Key_4:
            self._set_active_class(4)
            return
        if event.key() == Qt.Key.Key_5:
            self._set_active_class(5)
            return
        super().keyPressEvent(event)

    def _set_active_class(self, cls: int) -> None:
        if cls not in _CLASS_STYLE:
            return
        self._active_class = cls
        self._apply_active_class_style()
        name, _, _, class_value = _CLASS_STYLE[cls]
        value_text = _format_label_value(class_value)
        self._status.showMessage(f"Active label set to {name}. Saved label-mask value: {value_text}.")

    def _apply_active_class_style(self) -> None:
        name, color_qt, _, class_value = _CLASS_STYLE[self._active_class]
        value_text = _format_label_value(class_value)
        self._lbl_class.setText(f"Active Seed Label: {name} -> {value_text}")
        self._lbl_class.setStyleSheet(f"color: {color_qt.name()}; font-weight: bold;")
        self._viewer.set_current_seed_class(self._active_class)
        self._redraw_curves()

    def _build_class_map(self, width: int) -> NDArray[np.int32]:
        """Build a per-column class map from seeded class labels.

        Each column inherits the class of the seed immediately to its left.
        The color/label changes exactly at the seed's x-position.
        """
        class_map = np.ones(width, dtype=np.int32)
        seeds = self._viewer.seeds
        classes = self._viewer.seed_classes
        if not seeds or not classes:
            return class_map

        xs = np.array([s[0] for s in seeds], dtype=np.float64)
        cls = np.array(classes, dtype=np.int32)
        order = np.argsort(xs)
        xs = xs[order]
        cls = cls[order]

        if len(xs) == 1:
            class_map[:] = int(cls[0])
            return class_map

        # Use seed x-positions as breakpoints: placing a seed at x causes the
        # color to change starting from the PREVIOUS seed's position.
        x_grid = np.arange(width, dtype=np.float64)
        region_idx = np.clip(np.searchsorted(xs, x_grid, side="right"), 0, len(cls) - 1)
        class_map[:] = cls[region_idx]
        return class_map

    def _build_fixed_spline_mask(self, width: int) -> NDArray[np.bool_]:
        """Build a per-column mask for spline regions locked against fine-tuning."""
        fixed_mask = np.zeros(width, dtype=np.bool_)
        seeds = self._viewer.seeds
        fixed_flags = self._viewer.seed_fixed_flags
        if not seeds or not fixed_flags:
            return fixed_mask

        xs = np.array([s[0] for s in seeds], dtype=np.float64)
        fixed = np.array(fixed_flags, dtype=np.bool_)
        order = np.argsort(xs)
        xs = xs[order]
        fixed = fixed[order]

        if len(xs) == 1:
            fixed_mask[:] = bool(fixed[0])
            return fixed_mask

        x_grid = np.arange(width, dtype=np.float64)
        region_idx = np.clip(np.searchsorted(xs, x_grid, side="right"), 0, len(fixed) - 1)
        fixed_mask[:] = fixed[region_idx]
        return fixed_mask


def run_app():
    """CLI entry point."""
    app = QApplication(sys.argv)
    directory = sys.argv[1] if len(sys.argv) > 1 else None
    win = MainWindow(directory=directory)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()
