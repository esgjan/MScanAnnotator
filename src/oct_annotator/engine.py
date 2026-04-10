"""Spline interpolation and gradient-based boundary refinement engine."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.interpolate import make_interp_spline
import cv2


_OCT_CLIP_MIN = 0.0
_OCT_CLIP_MAX = 4.0
_REFINE_DELTA = 5
_FIRST_LAYER_ABOVE_RELATIVE_THRESHOLD = 0.55
_FIRST_LAYER_BELOW_RELATIVE_THRESHOLD = 0.85
_FIRST_LAYER_MIN_BRIGHTNESS = 0.08
_FIRST_LAYER_EARLY_SELECTION_RATIO = 0.6
_REFINE_GRADIENT_SMOOTHING = 5
_REFINE_MEDIAN_SIZE = 7
_REFINE_OUTPUT_SMOOTHING = 9


def fit_spline(
    seed_xs: NDArray[np.float64],
    seed_ys: NDArray[np.float64],
    width: int,
) -> NDArray[np.int64]:
    """Fit an interpolating B-spline y=f(x) through seed points.

    Uses `scipy.interpolate.make_interp_spline` which guarantees that
    the resulting curve passes exactly through every seed point.  This
    avoids the parametric-smoothing problem of `splprep` where steep
    jumps between nearby seeds are under-represented.

    Parameters
    ----------
    seed_xs : array of x-coordinates (column indices) of user clicks.
    seed_ys : array of y-coordinates (row indices) of user clicks.
    width   : total number of A-scan columns to interpolate across.

    Returns
    -------
    y_indices : int array of shape (width,) with the row index for every column.
    """
    if len(seed_xs) < 2:
        raise ValueError("At least 2 seed points are required for spline fitting.")

    # Sort by x so that the spline is a proper function y=f(x)
    order = np.argsort(seed_xs)
    xs = seed_xs[order].astype(np.float64)
    ys = seed_ys[order].astype(np.float64)

    # Degree is at most len(points)-1, capped at 3 (cubic)
    k = min(3, len(xs) - 1)

    spline = make_interp_spline(xs, ys, k=k)

    # Evaluate at every integer column across the full image width
    x_eval = np.arange(width, dtype=np.float64)
    y_eval = spline(x_eval)

    y_indices = np.round(y_eval).astype(np.int64)
    return y_indices


def refine_boundary(
    m_scan: NDArray,
    spline_indices: NDArray[np.int64],
) -> NDArray[np.uint16]:
    """Snap spline indices to a strong first-layer vertical gradient.

    Uses a fixed local search window around each spline index and computes
    positive vertical gradients (dark-to-bright transitions). For each column,
    it selects the earliest strong peak near the spline and then smooths the
    resulting boundary across neighboring A-scans. This behavior is tuned for
    the top retinal layer, which is usually the first strong positive edge and
    should vary smoothly across columns.

    Parameters
    ----------
    m_scan         : 2-D array (rows × columns), the M-scan image.
    spline_indices : int array (columns,), one row-index per A-scan.

    Returns
    -------
    refined : uint16 array (columns,), refined row indices.
    """
    rows, cols = m_scan.shape
    delta = _REFINE_DELTA
    disp = normalized_preview_float(m_scan)

    # First layer is usually a dark-to-bright transition, so keep positive
    # vertical gradients only to avoid snapping to opposite edges.
    grad = np.maximum(np.gradient(m_scan.astype(np.float32, copy=False), axis=0), 0.0)
    grad = uniform_filter1d(grad, size=_REFINE_GRADIENT_SMOOTHING, axis=1, mode="nearest")
    disp = uniform_filter1d(disp, size=_REFINE_GRADIENT_SMOOTHING, axis=1, mode="nearest")

    # Vectorised: build a (2*delta+1, cols) window for all columns at once
    y_init = np.clip(spline_indices.astype(np.int64), delta, rows - delta - 1)
    offsets = np.arange(-delta, delta + 1)  # shape (2*delta+1,)
    # row indices: (2*delta+1, cols)
    row_idx = y_init[np.newaxis, :] + offsets[:, np.newaxis]
    col_idx = np.arange(cols)[np.newaxis, :]  # (1, cols)
    windows = grad[row_idx, col_idx]  # (2*delta+1, cols)
    post_row_idx = np.clip(row_idx + 1, 0, rows - 1)
    post_intensity = disp[post_row_idx, col_idx]

    local_max = np.max(windows, axis=0)
    relative_thresholds = np.where(
        offsets[:, np.newaxis] <= 0,
        _FIRST_LAYER_ABOVE_RELATIVE_THRESHOLD,
        _FIRST_LAYER_BELOW_RELATIVE_THRESHOLD,
    )
    strong_mask = windows >= (relative_thresholds * local_max[np.newaxis, :])
    strong_mask &= post_intensity >= _FIRST_LAYER_MIN_BRIGHTNESS

    # If a column is flat, keep the spline position there.
    has_signal = local_max > 0
    raw_refined = y_init.copy()

    # If a plausible top-rim candidate exists at or above the spline, only
    # consider those candidates. Deeper edges are only considered as fallback.
    above_mask = strong_mask & (offsets[:, np.newaxis] <= 0)
    preferred_mask = np.where(np.any(above_mask, axis=0)[np.newaxis, :], above_mask, strong_mask)

    candidate_columns = np.any(preferred_mask, axis=0) & has_signal
    if np.any(candidate_columns):
        candidate_scores = np.where(
            preferred_mask[:, candidate_columns],
            windows[:, candidate_columns],
            -np.inf,
        )
        best_scores = np.max(candidate_scores, axis=0)
        earliest_good_mask = candidate_scores >= (
            _FIRST_LAYER_EARLY_SELECTION_RATIO * best_scores[np.newaxis, :]
        )
        best_idx = np.argmax(earliest_good_mask, axis=0)
        raw_refined[candidate_columns] = y_init[candidate_columns] + offsets[best_idx]

    # If no bright-tissue candidate exists near the spline, keep the spline in
    # that column. This handles scan ends where the top retinal layer fades into
    # black background instead of forcing a snap to a deeper edge.

    # Smooth the final boundary across A-scans: median removes outliers,
    # uniform averaging reduces jitter while preserving the overall shape.
    refined = median_filter(raw_refined, size=_REFINE_MEDIAN_SIZE, mode="nearest")
    refined = uniform_filter1d(refined.astype(np.float32), size=_REFINE_OUTPUT_SMOOTHING, mode="nearest")
    refined = np.round(refined).astype(np.int64)

    # Keep the final result inside the allowed +/- delta band around the spline.
    min_allowed = np.clip(spline_indices.astype(np.int64) - delta, 0, rows - 1)
    max_allowed = np.clip(spline_indices.astype(np.int64) + delta, 0, rows - 1)
    refined = np.clip(refined, min_allowed, max_allowed)

    return refined.astype(np.uint16)


# ---------------------------------------------------------------------------
#  PNG export
# ---------------------------------------------------------------------------

_NAN_SENTINEL = np.uint16(65535)


def to_preview_uint8(m_scan: NDArray) -> NDArray[np.uint8]:
    """Convert raw OCT image to uint8 using the same pipeline as npy2png.py."""
    bgra = db_equivalent_bgra_from_raw(m_scan)
    return bgra[:, :, 0]


def normalized_preview_float(raw_img: NDArray) -> NDArray[np.float32]:
    """Return the clip+normalize display image as float32 in [0, 1]."""
    img = np.clip(raw_img.astype(np.float32, copy=False), _OCT_CLIP_MIN, _OCT_CLIP_MAX)
    vmin = float(img.min())
    vmax = float(img.max())
    if vmax > vmin:
        return (img - vmin) / (vmax - vmin)
    return np.zeros_like(img, dtype=np.float32)


def db_equivalent_bgra_from_raw(raw_img: NDArray) -> NDArray[np.uint8]:
    """Exact clip/normalize/BGRA pipeline used by dataloader2/npy2png.py."""
    disp = normalized_preview_float(raw_img)
    bgra = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGRA)
    return (bgra * 255.0).astype(np.uint8)


def render_annotation_png(
    m_scan: NDArray,
    annotation: NDArray[np.uint16],
    nan_mask: NDArray[np.bool_],
    out_path: str,
    line_color: tuple = (0, 255, 100),
    class_by_column: NDArray[np.int32] | None = None,
    class_colors: dict[int, tuple[int, int, int]] | None = None,
    line_thickness: int = 2,
) -> None:
    """Save a PNG showing the M-scan with only the final annotation boundary.

    Parameters
    ----------
    m_scan       : 2-D float array (rows x cols), the M-scan.
    annotation   : 1-D uint16 array (cols,); sentinel columns are skipped.
    nan_mask     : bool array (cols,); True = excluded column.
    out_path     : file path for the output PNG.
    line_color   : RGB tuple for the boundary line (fallback/default).
    class_by_column : optional int array (cols,), class id per column.
    class_colors : optional mapping class_id -> RGB tuple.
    line_thickness : pixel width of the boundary line.
    """
    rows, cols = m_scan.shape

    # Use the exact same clip+normalize+BGR conversion as npy2png.py.
    bgra = db_equivalent_bgra_from_raw(m_scan)

    # Draw the boundary line (skip NaN columns)
    ann = annotation.reshape(-1)
    default_r, default_g, default_b = line_color
    default_bgr = (default_b, default_g, default_r)
    half = line_thickness // 2
    for x in range(cols):
        if nan_mask[x] or ann[x] == _NAN_SENTINEL:
            continue
        bgr = default_bgr
        if class_by_column is not None and class_colors is not None:
            cls = int(class_by_column[x])
            rgb = class_colors.get(cls)
            if rgb is not None:
                r, g, b = rgb
                bgr = (b, g, r)
        y_center = int(ann[x])
        y_lo = max(0, y_center - half)
        y_hi = min(rows, y_center + half + 1)
        bgra[y_lo:y_hi, x, :3] = bgr

    ok = cv2.imwrite(out_path, bgra)
    if not ok:
        raise RuntimeError(f"Failed to write PNG: {out_path}")
