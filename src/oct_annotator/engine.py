"""Spline interpolation and gradient-based boundary refinement engine."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import make_interp_spline


_OCT_CLIP_MIN = 1.0
_OCT_CLIP_MAX = 4.0


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


_REFINE_DELTA = 5  # fixed half-window for gradient search


def refine_boundary(
    m_scan: NDArray,
    spline_indices: NDArray[np.int64],
) -> NDArray[np.uint16]:
    """Snap spline indices to the nearest strong vertical gradient.

    Uses a fixed ±5-pixel search window around each spline index to find
    the local gradient maximum.  This keeps the refinement tight to the
    spline while still snapping to real tissue boundaries.

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
    grad = np.abs(np.gradient(m_scan, axis=0))  # vertical gradient

    # Vectorised: build a (2*delta, cols) window for all columns at once
    y_init = np.clip(spline_indices.astype(np.int64), delta, rows - delta - 1)
    offsets = np.arange(-delta, delta)  # shape (2*delta,)
    # row indices: (2*delta, cols)
    row_idx = y_init[np.newaxis, :] + offsets[:, np.newaxis]
    col_idx = np.arange(cols)[np.newaxis, :]  # (1, cols)
    windows = grad[row_idx, col_idx]  # (2*delta, cols)
    best_offset = np.argmax(windows, axis=0)  # (cols,)
    refined = y_init + offsets[best_offset]

    return refined.astype(np.uint16)


# ---------------------------------------------------------------------------
#  PNG export
# ---------------------------------------------------------------------------

_NAN_SENTINEL = np.uint16(65535)


def to_preview_uint8(m_scan: NDArray) -> NDArray[np.uint8]:
    """Convert raw OCT image to uint8 using DB-analyzer clip+normalize."""
    img = np.clip(m_scan.astype(np.float64, copy=False), _OCT_CLIP_MIN, _OCT_CLIP_MAX)
    vmin, vmax = float(img.min()), float(img.max())
    if vmax > vmin:
        disp = (img - vmin) / (vmax - vmin)
    else:
        disp = np.zeros_like(img)
    return (disp * 255.0).astype(np.uint8)


def render_annotation_png(
    m_scan: NDArray,
    annotation: NDArray[np.uint16],
    nan_mask: NDArray[np.bool_],
    out_path: str,
    line_color: tuple = (0, 255, 100),
    line_thickness: int = 2,
) -> None:
    """Save a PNG showing the M-scan with only the final annotation boundary.

    Parameters
    ----------
    m_scan       : 2-D float array (rows x cols), the M-scan.
    annotation   : 1-D uint16 array (cols,); sentinel columns are skipped.
    nan_mask     : bool array (cols,); True = excluded column.
    out_path     : file path for the output PNG.
    line_color   : RGB tuple for the boundary line.
    line_thickness : pixel width of the boundary line.
    """
    rows, cols = m_scan.shape

    # Use the same clip+normalize pipeline as DB analyzer and UI preview.
    gray = to_preview_uint8(m_scan)
    rgb = np.stack([gray, gray, gray], axis=-1)  # (rows, cols, 3)

    # Draw the boundary line (skip NaN columns)
    ann = annotation.reshape(-1)
    r, g, b = line_color
    half = line_thickness // 2
    for x in range(cols):
        if nan_mask[x] or ann[x] == _NAN_SENTINEL:
            continue
        y_center = int(ann[x])
        y_lo = max(0, y_center - half)
        y_hi = min(rows, y_center + half + 1)
        rgb[y_lo:y_hi, x] = [r, g, b]

    # Write PNG using PyQt6's QImage (avoids extra dependencies)
    from PyQt6.QtGui import QImage
    img_data = np.ascontiguousarray(rgb)
    qimage = QImage(
        img_data.data,
        cols,
        rows,
        cols * 3,
        QImage.Format.Format_RGB888,
    )
    qimage.save(out_path, "PNG")
