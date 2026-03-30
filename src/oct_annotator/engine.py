"""Spline interpolation and gradient-based boundary refinement engine."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import make_interp_spline


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
