"""Main application window and entry point for the OCT M-Scan Annotator."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import List

import cv2
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

# class_id -> (name, UI color, label-mask value)
_CLASS_STYLE: dict[int, tuple[str, QColor, float]] = {
    1: ("Class 1", QColor(0, 190, 90), 1.0),
    2: ("Class 2", QColor(245, 205, 0), 2.0),
    3: ("Class 3", QColor(145, 92, 43), 3.0),
    4: ("Class 4", QColor(40, 120, 255), 4.0),
    5: (NAN_LABEL_TEXT, QColor(210, 40, 40), float("nan")),
}

def _fix_mode_class_colors() -> dict[int, QColor]:
    colors = {class_id: QColor(color_qt) for class_id, (_, color_qt, _) in _CLASS_STYLE.items()}
    colors[5] = QColor(0, 0, 0, 0)
    return colors


def _class_overlay_rgb_map() -> dict[int, tuple[int, int, int]]:
    """Build export RGB colors from the same QColor values used by the UI."""
    return {
        class_id: (color_qt.red(), color_qt.green(), color_qt.blue())
        for class_id, (_, color_qt, _) in _CLASS_STYLE.items()
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
        self._current_directory: Path | None = None
        self._current_data: NDArray | None = None
        self._spline_indices: NDArray | None = None
        self._refined_indices: NDArray | None = None
        self._active_class: int = 1
        self._fixed_spline_mode: bool = False
        self._anchor_erase_mode: bool = False
        self._fix_mode_enabled: bool = True
        self._loaded_fix_spline_indices: NDArray | None = None
        self._loaded_fix_class_map: NDArray[np.int32] | None = None
        self._loaded_fix_anchor_xs: NDArray[np.float64] | None = None
        self._loaded_fix_anchor_ys: NDArray[np.float64] | None = None
        self._loaded_fix_fixed_mask: NDArray[np.bool_] | None = None
        self._loaded_fix_annotation_path: Path | None = None
        self._fix_mode_seeds_imported: bool = False
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
        suffix = path.suffix.lower()
        stem = path.stem
        if suffix == ".npy":
            return not (stem.endswith("_annotations") or stem.endswith("_still"))
        if suffix in (".tif", ".tiff"):
            return not stem.endswith("_annotations")
        return False

    @staticmethod
    def _list_source_scans(folder: Path) -> List[Path]:
        if not folder.is_dir():
            return []
        return sorted(path for path in folder.iterdir() if path.is_file() and MainWindow._is_source_scan(path))

    @staticmethod
    def _load_source_array(path: Path) -> NDArray[np.float64]:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            data = np.load(str(path), allow_pickle=False)
            if data.ndim != 2:
                raise ValueError(f"Unsupported .npy shape for source scan: {data.shape}")
            return data.astype(np.float64)

        if suffix in (".tif", ".tiff"):
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Failed to load image: {path}")
            if image.ndim == 2:
                return image.astype(np.float64)
            if image.ndim == 3:
                # Keep red-line guidance prominent when labeling overlay TIFFs.
                red_channel = image[:, :, 2]
                return red_channel.astype(np.float64)
            raise ValueError(f"Unsupported image shape: {image.shape}")

        raise ValueError(f"Unsupported file type: {path.suffix}")

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
        out_dir = self._output_root_directory.joinpath(*self._experiment_output_parts(source_path))
        return self._ensure_output_under_root(out_dir)

    def _ensure_output_under_root(self, output_dir: Path) -> Path:
        """Guarantee resolved output path stays within configured save root."""
        root = self._output_root_directory.resolve()
        resolved_out = output_dir.resolve()
        try:
            resolved_out.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                f"Output path {resolved_out} is outside save root {root}."
            ) from exc
        return resolved_out

    def _set_output_root_directory(self, directory: Path) -> None:
        self._output_root_directory = directory
        if hasattr(self, "_edit_output_dir"):
            self._edit_output_dir.setText(str(directory))

    def _active_viewer_class_colors(self) -> dict[int, QColor]:
        if self._fix_mode_enabled:
            return _fix_mode_class_colors()
        return {class_id: color for class_id, (_, color, _) in _CLASS_STYLE.items()}

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

        self._btn_anchor_erase = QPushButton("Remove Anchors (R)")
        self._btn_anchor_erase.setCheckable(True)
        self._btn_anchor_erase.setToolTip(
            "When enabled in fix mode, left-click removes anchors in a 10 px radius."
        )
        toolbar.addWidget(self._btn_anchor_erase)

        self._btn_fix_mode = QPushButton("Fix Mode")
        self._btn_fix_mode.setCheckable(True)
        self._btn_fix_mode.setChecked(False)
        self._btn_fix_mode.setToolTip(
            "Load existing *_annotations.npy as baseline anchors for fixing."
        )
        toolbar.addWidget(self._btn_fix_mode)

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
            {class_id: color_qt for class_id, (_, color_qt, _) in _CLASS_STYLE.items()}
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
        self._btn_anchor_erase.toggled.connect(self._on_toggle_anchor_erase_mode)
        self._btn_fix_mode.toggled.connect(self._on_toggle_fix_mode)
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
        self._current_directory = folder
        raw_candidates = self._list_source_scans(folder)
        candidates: list[Path] = []
        skipped_names: list[str] = []
        for candidate in raw_candidates:
            try:
                _ = self._load_source_array(candidate)
                candidates.append(candidate)
            except Exception:
                skipped_names.append(candidate.name)
        self._npy_files = candidates
        self._combo_files.blockSignals(True)
        self._combo_files.clear()
        for f in self._npy_files:
            self._combo_files.addItem(f.name)
        self._combo_files.blockSignals(False)

        if self._npy_files:
            self._combo_files.setCurrentIndex(0)
            self._on_file_selected(0)
            if skipped_names:
                self._status.showMessage(
                    f"Opened {len(self._npy_files)} source files; skipped {len(skipped_names)} unreadable files."
                )
        else:
            if skipped_names:
                self._status.showMessage(
                    f"No loadable source files found in the selected folder. Skipped {len(skipped_names)} unreadable files."
                )
            else:
                self._status.showMessage("No supported source files found in the selected folder.")

    def _find_next_experiment_directory(self, current_directory: Path) -> Path | None:
        parent = current_directory.parent
        sibling_dirs = sorted(path for path in parent.iterdir() if path.is_dir())

        try:
            current_index = sibling_dirs.index(current_directory)
        except ValueError:
            return None

        for candidate in sibling_dirs[current_index + 1:]:
            has_source_scans = len(self._list_source_scans(candidate)) > 0
            if has_source_scans:
                return candidate
        return None

    def _on_file_selected(self, index: int):
        if index < 0 or index >= len(self._npy_files):
            return
        self._btn_flag.setEnabled(True)
        path = self._npy_files[index]
        try:
            self._current_data = self._load_source_array(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return

        self._spline_indices = None
        self._refined_indices = None
        self._btn_refine.setEnabled(False)
        self._btn_save.setEnabled(False)

        self._apply_display_data()
        fix_message = self._load_fix_annotation_for_source(path)
        self._sync_fix_mode_seeds()
        if self._fix_mode_seeds_imported:
            self._on_seeds_changed()
            if fix_message is not None:
                self._status.showMessage(fix_message)
            self._apply_active_class_style()
            return
        if self._apply_loaded_fix_baseline():
            self._status.showMessage(fix_message or f"Loaded {path.name}.")
        elif fix_message is not None:
            self._status.showMessage(fix_message)
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
        if enabled:
            self._sync_fix_mode_seeds()
            if self._fix_mode_seeds_imported:
                self._on_seeds_changed()
        self._status.showMessage(
            "Fixed spline mode enabled." if enabled else "Fixed spline mode disabled."
        )

    def _on_toggle_anchor_erase_mode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled and not self._fix_mode_enabled:
            self._btn_anchor_erase.blockSignals(True)
            self._btn_anchor_erase.setChecked(False)
            self._btn_anchor_erase.blockSignals(False)
            self._status.showMessage("Remove anchors mode is only available while Fix Mode is enabled.")
            return

        self._anchor_erase_mode = enabled
        self._viewer.set_anchor_erase_mode(enabled)
        self._status.showMessage(
            "Remove anchors mode enabled (click removes anchors within 10 px)."
            if enabled
            else "Remove anchors mode disabled."
        )

    def _on_toggle_fix_mode(self, enabled: bool) -> None:
        self._fix_mode_enabled = bool(enabled)
        if not self._fix_mode_enabled and self._anchor_erase_mode:
            self._btn_anchor_erase.blockSignals(True)
            self._btn_anchor_erase.setChecked(False)
            self._btn_anchor_erase.blockSignals(False)
            self._anchor_erase_mode = False
            self._viewer.set_anchor_erase_mode(False)
        if self._current_directory is not None:
            self._load_directory(str(self._current_directory))
            return
        self._status.showMessage("Fix mode enabled." if enabled else "Fix mode disabled.")

    def _annotation_path_for_source(self, source_path: Path) -> Path:
        return source_path.with_name(source_path.stem + "_annotations.npy")

    def _clear_loaded_fix_annotation(self) -> None:
        self._loaded_fix_spline_indices = None
        self._loaded_fix_class_map = None
        self._loaded_fix_anchor_xs = None
        self._loaded_fix_anchor_ys = None
        self._loaded_fix_fixed_mask = None
        self._loaded_fix_annotation_path = None
        self._fix_mode_seeds_imported = False

    def _build_fix_anchor_seed_points(self) -> list[tuple[float, float, int, bool]]:
        if self._loaded_fix_spline_indices is None or self._loaded_fix_class_map is None:
            return []

        width = len(self._loaded_fix_spline_indices)
        if width == 0:
            return []

        eligible = np.isin(self._loaded_fix_class_map, [1, 2, 3, 4])
        if not np.any(eligible):
            return []

        selected_indices: set[int] = set()
        class_map = self._loaded_fix_class_map
        idx = 0
        while idx < width:
            if not eligible[idx]:
                idx += 1
                continue

            run_start = idx
            while idx < width and eligible[idx]:
                idx += 1
            run_end = idx - 1

            # Use every eligible pixel as a fixed anchor.
            for j in range(run_start, run_end + 1):
                selected_indices.add(j)

            # Add a NaN anchor exactly one column right of the last valid
            # anchor, when that immediate next column is NaN class.
            next_idx = run_end + 1
            if next_idx < width and int(class_map[next_idx]) == 5:
                selected_indices.add(int(next_idx))

        seeds: list[tuple[float, float, int, bool]] = []
        for i in sorted(selected_indices):
            cls = int(self._loaded_fix_class_map[i])
            if cls not in (1, 2, 3, 4, 5):
                continue
            y = float(self._loaded_fix_spline_indices[i])
            seeds.append((float(i), y, cls, True))
        return seeds

    def _sync_fix_mode_seeds(self) -> None:
        if not self._fix_mode_enabled:
            self._fix_mode_seeds_imported = False
            return

        if len(self._viewer.seeds) > 0:
            self._fix_mode_seeds_imported = False
            return

        seeds = self._build_fix_anchor_seed_points()
        if not seeds:
            self._fix_mode_seeds_imported = False
            return

        self._viewer.set_seed_points(seeds, emit_signal=False)
        self._fix_mode_seeds_imported = True

    @staticmethod
    def _labels_to_class_map(labels: NDArray[np.float64]) -> NDArray[np.int32]:
        class_map = np.ones(labels.shape[0], dtype=np.int32)
        nan_like_mask = np.isnan(labels) | (labels == float(NAN_SENTINEL))
        class_map[nan_like_mask] = 5
        finite_mask = np.isfinite(labels)
        rounded = np.zeros(labels.shape[0], dtype=np.int32)
        rounded[finite_mask] = np.round(labels[finite_mask]).astype(np.int32)
        for class_id in (1, 2, 3, 4):
            class_map[finite_mask & (rounded == class_id)] = class_id
        return class_map

    @staticmethod
    def _fill_boundary_gaps(boundary: NDArray[np.float64], width: int) -> NDArray[np.int64] | None:
        valid_mask = np.isfinite(boundary) & (boundary != float(NAN_SENTINEL))
        if int(valid_mask.sum()) < 2:
            return None
        x_valid = np.flatnonzero(valid_mask).astype(np.float64)
        y_valid = boundary[valid_mask].astype(np.float64)
        x_full = np.arange(width, dtype=np.float64)
        filled = np.interp(x_full, x_valid, y_valid)
        return np.round(filled).astype(np.int64)

    def _load_fix_annotation_for_source(self, source_path: Path) -> str | None:
        self._clear_loaded_fix_annotation()
        if (not self._fix_mode_enabled) or self._current_data is None:
            return None

        annotation_path = self._annotation_path_for_source(source_path)
        if not annotation_path.exists():
            return f"Fix mode: no existing annotation for {source_path.name}."

        annotation_data = np.load(str(annotation_path), allow_pickle=False)
        if annotation_data.ndim != 2 or annotation_data.shape[1] < 2:
            return f"Fix mode skipped: unsupported annotation shape {annotation_data.shape}."
        if annotation_data.shape[0] != self._current_data.shape[1]:
            return (
                "Fix mode skipped: annotation width mismatch "
                f"({annotation_data.shape[0]} vs {self._current_data.shape[1]})."
            )

        boundary = annotation_data[:, 0].astype(np.float64, copy=False)
        labels = annotation_data[:, 1].astype(np.float64, copy=False)
        class_map = self._labels_to_class_map(labels)
        valid_boundary = np.isfinite(boundary) & (boundary != float(NAN_SENTINEL))
        anchor_mask = valid_boundary & (class_map != 5)

        self._loaded_fix_class_map = class_map
        self._loaded_fix_fixed_mask = anchor_mask.astype(np.bool_, copy=False)
        self._loaded_fix_annotation_path = annotation_path
        if int(anchor_mask.sum()) >= 2:
            self._loaded_fix_anchor_xs = np.flatnonzero(anchor_mask).astype(np.float64)
            self._loaded_fix_anchor_ys = boundary[anchor_mask].astype(np.float64)

        self._loaded_fix_spline_indices = self._fill_boundary_gaps(boundary, self._current_data.shape[1])
        if self._loaded_fix_spline_indices is None:
            return f"Fix mode: loaded {annotation_path.name}, but no usable spline anchors found."

        return f"Fix mode: loaded {annotation_path.name} as baseline anchors."

    def _apply_loaded_fix_baseline(self) -> bool:
        if self._loaded_fix_spline_indices is None:
            return False
        width = self._viewer.image_width
        nan_mask = self._viewer.get_nan_column_mask(width)
        class_map = self._build_class_map(width)
        self._spline_indices = self._loaded_fix_spline_indices.copy()
        self._refined_indices = None
        self._viewer.draw_spline_classified(self._spline_indices, class_map, nan_mask)
        self._btn_refine.setEnabled(True)
        self._btn_save.setEnabled(True)
        self._btn_reset_refine.setEnabled(False)
        return True

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
        if self._viewer.image_width > 0:
            pass
        self._status.showMessage(f"{n} seed point(s) placed.")

    def _on_fit_spline(self):
        seeds = self._viewer.seeds
        seed_xs: list[float] = [seed[0] for seed in seeds]
        seed_ys: list[float] = [seed[1] for seed in seeds]
        if len(seed_xs) < 2:
            return
        xs = np.array(seed_xs)
        ys = np.array(seed_ys)
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

        try:
            out_dir = self._output_dir_for_source(src_path)
        except RuntimeError as exc:
            QMessageBox.critical(self, "Save error", str(exc))
            return
        out_dir.mkdir(parents=True, exist_ok=True)

        # Copy original snippet to output folder
        copied_snippet = out_dir / src_path.name
        if src_path.resolve() != copied_snippet.resolve():
            shutil.copy2(src_path, copied_snippet)

        out_combined_npy = out_dir / (src_path.stem + "_annotations.npy")
        np.save(str(out_combined_npy), combined_data)

        # Save .tiff visual overlay with class colors per seeded region.
        out_tiff = out_dir / (src_path.stem + "_annotations.tiff")
        overlay_class_colors = _class_overlay_rgb_map()
        render_annotation_tiff(
            m_scan,
            ann,
            nan_mask,
            str(out_tiff),
            contrast=self._preview_contrast,
            gamma=self._preview_gamma,
            class_by_column=class_map,
            class_colors=overlay_class_colors,
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
        candidates = self._list_source_scans(folder)
        self._npy_files = candidates
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
            try:
                out_dir = self._output_dir_for_source(src_path)
            except RuntimeError as exc:
                QMessageBox.critical(self, "Save error", str(exc))
                return
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
        if event.key() == Qt.Key.Key_R:
            self._btn_anchor_erase.setChecked(not self._btn_anchor_erase.isChecked())
            return
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
        if self._anchor_erase_mode:
            self._btn_anchor_erase.setChecked(False)
        self._active_class = cls
        self._apply_active_class_style()
        name, _, class_value = _CLASS_STYLE[cls]
        value_text = _format_label_value(class_value)
        self._status.showMessage(f"Active label set to {name}. Saved label-mask value: {value_text}.")

    def _apply_active_class_style(self) -> None:
        name, color_qt, class_value = _CLASS_STYLE[self._active_class]
        value_text = _format_label_value(class_value)
        self._lbl_class.setText(f"Active Seed Label: {name} -> {value_text}")
        self._lbl_class.setStyleSheet(f"color: {color_qt.name()}; font-weight: bold;")
        self._viewer.set_current_seed_class(self._active_class)
        self._viewer.set_class_colors(self._active_viewer_class_colors())
        self._redraw_curves()

    def _build_class_map(self, width: int) -> NDArray[np.int32]:
        """Build a per-column class map from seeded class labels.

        Each column inherits the class of the seed immediately to its left.
        The color/label changes exactly at the seed's x-position.
        """
        if self._loaded_fix_class_map is not None and len(self._loaded_fix_class_map) == width:
            class_map = self._loaded_fix_class_map.copy()
        else:
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
            start = int(np.clip(np.floor(xs[0]), 0, width - 1))
            class_map[start:] = int(cls[0])
            return class_map

        # Use seed x-positions as breakpoints: a seed controls from its own
        # x-position to the right, up to (but excluding) the next seed.
        x_grid = np.arange(width, dtype=np.float64)
        region_idx = np.clip(np.searchsorted(xs, x_grid, side="right") - 1, 0, len(cls) - 1)
        overwrite_mask = x_grid >= xs[0]
        class_map[overwrite_mask] = cls[region_idx[overwrite_mask]]
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
            start = int(np.clip(np.floor(xs[0]), 0, width - 1))
            fixed_mask[start:] = bool(fixed[0])
            return fixed_mask

        x_grid = np.arange(width, dtype=np.float64)
        region_idx = np.clip(np.searchsorted(xs, x_grid, side="right") - 1, 0, len(fixed) - 1)
        overwrite_mask = x_grid >= xs[0]
        fixed_mask[overwrite_mask] = fixed[region_idx[overwrite_mask]]
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
