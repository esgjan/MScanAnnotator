"""Spline interpolation and gradient-based boundary refinement engine."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import splprep, splev


def fit_spline(
    seed_xs: NDArray[np.float64],
    seed_ys: NDArray[np.float64],
    width: int,
    smoothing: float = 0.0,
) -> NDArray[np.int64]:
    """Fit a B-spline through seed points and evaluate at every A-scan column.

    Parameters
    ----------
    seed_xs : array of x-coordinates (column indices) of user clicks.
    seed_ys : array of y-coordinates (row indices) of user clicks.
    width   : total number of A-scan columns to interpolate across.
    smoothing : smoothing factor for splprep (0 = interpolating).

    Returns
    -------
    y_indices : int array of shape (width,) with the row index for every column.
    """
    if len(seed_xs) < 2:
        raise ValueError("At least 2 seed points are required for spline fitting.")

    # Sort by x to avoid self-intersecting splines
    order = np.argsort(seed_xs)
    xs = seed_xs[order].astype(np.float64)
    ys = seed_ys[order].astype(np.float64)

    # Degree is at most len(points)-1, capped at 3 (cubic)
    k = min(3, len(xs) - 1)

    tck, _ = splprep([xs, ys], s=smoothing, k=k)

    # Evaluate at `width` evenly-spaced parameter values
    u_new = np.linspace(0, 1, width)
    x_fit, y_fit = splev(u_new, tck)

    # Map the parametric output back to integer column order
    y_indices = np.round(y_fit).astype(np.int64)
    return y_indices


def refine_boundary(
    m_scan: NDArray,
    spline_indices: NDArray[np.int64],
    delta: int = 5,
) -> NDArray[np.uint16]:
    """Snap spline indices to the nearest strong vertical gradient.

    Parameters
    ----------
    m_scan         : 2-D array (rows × columns), the M-scan image.
    spline_indices : int array (columns,), one row-index per A-scan.
    delta          : half-window size for local gradient search.

    Returns
    -------
    refined : uint16 array (columns,), refined row indices.
    """
    rows, cols = m_scan.shape
    grad = np.abs(np.gradient(m_scan, axis=0))  # vertical gradient

    refined = np.copy(spline_indices).astype(np.int64)
    for x in range(cols):
        y_init = int(spline_indices[x])
        lo = max(0, y_init - delta)
        hi = min(rows, y_init + delta)
        if hi <= lo:
            continue
        window = grad[lo:hi, x]
        refined[x] = lo + int(np.argmax(window))

    return refined.astype(np.uint16)
