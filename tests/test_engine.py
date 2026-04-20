"""Tests for the OCT annotator engine (spline + gradient refinement)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from oct_annotator.engine import fit_spline, refine_boundary, render_annotation_png, to_preview_uint8


# ---------------------------------------------------------------------------
# Unit tests – spline fitting
# ---------------------------------------------------------------------------

class TestFitSpline:
    """Validate spline fitting with controlled seed points."""

    def test_returns_correct_width(self):
        xs = np.array([0.0, 250.0, 499.0])
        ys = np.array([100.0, 120.0, 110.0])
        result = fit_spline(xs, ys, width=500)
        assert result.shape == (500,)

    def test_dtype_is_int(self):
        xs = np.array([0.0, 100.0, 200.0])
        ys = np.array([50.0, 60.0, 55.0])
        result = fit_spline(xs, ys, width=300)
        assert np.issubdtype(result.dtype, np.integer)

    def test_two_points_linear(self):
        """Two points should produce a linear (degree-1) spline."""
        xs = np.array([0.0, 99.0])
        ys = np.array([10.0, 20.0])
        result = fit_spline(xs, ys, width=100)
        # Endpoints should be close to the seed y-values
        assert abs(int(result[0]) - 10) <= 1
        assert abs(int(result[-1]) - 20) <= 1

    def test_raises_on_single_point(self):
        with pytest.raises(ValueError, match="At least 2"):
            fit_spline(np.array([5.0]), np.array([5.0]), width=100)

    def test_values_within_image_bounds(self):
        """Spline values should stay near the provided y-range."""
        xs = np.array([0.0, 50.0, 100.0])
        ys = np.array([200.0, 210.0, 205.0])
        result = fit_spline(xs, ys, width=101)
        assert result.min() >= 190  # reasonable bound
        assert result.max() <= 220


# ---------------------------------------------------------------------------
# Unit tests – gradient refinement
# ---------------------------------------------------------------------------

class TestRefineBoundary:
    """Validate gradient-based boundary snapping."""

    def test_snaps_to_edge(self):
        """Create an image with a clear horizontal edge and verify snapping."""
        rows, cols = 100, 200
        m_scan = np.zeros((rows, cols), dtype=np.float64)
        edge_row = 50
        m_scan[edge_row:, :] = 1.0  # strong edge at row 50

        # Start with indices offset by 3 rows from the true edge
        initial = np.full(cols, edge_row + 3, dtype=np.int64)
        refined = refine_boundary(m_scan, initial)

        # The gradient peaks at the pixel just before the step (row 49),
        # because np.gradient computes the central difference
        np.testing.assert_array_equal(refined, edge_row - 1)

    def test_output_dtype(self):
        m_scan = np.random.rand(64, 64)
        indices = np.full(64, 32, dtype=np.int64)
        refined = refine_boundary(m_scan, indices)
        assert refined.dtype == np.uint16

    def test_output_shape(self):
        m_scan = np.random.rand(128, 256)
        indices = np.full(256, 64, dtype=np.int64)
        refined = refine_boundary(m_scan, indices)
        assert refined.shape == (256,)

    def test_prefers_first_layer_when_multiple_peaks_exist(self):
        """Prefer the earliest strong edge when two nearby peaks are similar."""
        rows, cols = 120, 80
        m_scan = np.zeros((rows, cols), dtype=np.float64)

        # Two nearby step edges: first at row 40, second at row 46.
        # The second is slightly stronger, but the first should still be picked
        # by the first-strong-edge policy.
        m_scan[40:, :] += 1.0
        m_scan[46:, :] += 1.02

        initial = np.full(cols, 45, dtype=np.int64)
        refined = refine_boundary(m_scan, initial)

        # The true edge response is at 39, but the hard +/-5 pixel bound around
        # the spline clamps the result to 40 here.
        np.testing.assert_array_equal(refined, 40)

    def test_smooths_single_column_outlier(self):
        """An isolated stronger deeper edge should not pull one column down."""
        rows, cols = 120, 81
        m_scan = np.zeros((rows, cols), dtype=np.float64)

        # Consistent first retinal layer.
        m_scan[40:, :] += 1.0
        # Add a deeper edge only in one column.
        m_scan[48:, cols // 2] += 2.0

        initial = np.full(cols, 44, dtype=np.int64)
        refined = refine_boundary(m_scan, initial)

        # The refined curve should stay on the top layer even at the outlier column.
        assert int(refined[cols // 2]) == 39
        assert np.max(np.abs(refined.astype(np.int32) - 39)) <= 1

    def test_prefers_top_layer_even_when_deeper_edge_is_stronger(self):
        """A stronger deeper edge should not win if a plausible top layer exists."""
        rows, cols = 120, 80
        m_scan = np.zeros((rows, cols), dtype=np.float64)

        # Visible top retinal layer.
        m_scan[40:, :] += 1.0
        # Stronger deeper structure that previously could pull refinement down.
        m_scan[46:, :] += 1.35

        initial = np.full(cols, 44, dtype=np.int64)
        refined = refine_boundary(m_scan, initial)

        np.testing.assert_array_equal(refined, 39)

    def test_ignores_early_edge_that_stays_almost_black(self):
        """Prefer the first edge whose post-edge region is visibly bright."""
        rows, cols = 120, 80
        m_scan = np.zeros((rows, cols), dtype=np.float64)

        # Early weak layer: edge exists, but region remains almost black.
        m_scan[34:, :] += 0.05
        # Actual top retinal layer becomes clearly visible later.
        m_scan[40:, :] += 0.95

        initial = np.full(cols, 38, dtype=np.int64)
        refined = refine_boundary(m_scan, initial)

        np.testing.assert_array_equal(refined, 39)

    def test_keeps_spline_when_top_layer_fades_to_black_background(self):
        """At scan ends, keep the spline if no plausible bright top layer remains."""
        rows, cols = 120, 80
        m_scan = np.zeros((rows, cols), dtype=np.float64)

        # Left side contains a visible top layer.
        m_scan[40:, :40] += 1.0
        # Right side fades to black background with only a deeper edge present.
        m_scan[65:, 40:] += 1.0

        initial = np.full(cols, 40, dtype=np.int64)
        refined = refine_boundary(m_scan, initial)

        # Left half should detect the top layer; right half should stay near spline.
        assert np.all(refined[:35] == 39)
        assert np.max(np.abs(refined[45:].astype(np.int32) - 40)) <= 1

    def test_refinement_stays_within_five_pixels_of_spline(self):
        """Refinement must stay inside the hard +/-5 pixel band around the spline."""
        rows, cols = 120, 80
        m_scan = np.random.rand(rows, cols).astype(np.float64)
        spline = np.full(cols, 60, dtype=np.int64)

        refined = refine_boundary(m_scan, spline)

        assert np.max(np.abs(refined.astype(np.int32) - spline)) <= 5

    def test_fixed_mask_keeps_spline_columns_unchanged(self):
        """Columns marked fixed must stay exactly on the spline."""
        rows, cols = 100, 40
        m_scan = np.zeros((rows, cols), dtype=np.float64)
        m_scan[50:, :] = 1.0
        spline = np.full(cols, 54, dtype=np.int64)
        fixed_mask = np.zeros(cols, dtype=np.bool_)
        fixed_mask[10:20] = True

        refined = refine_boundary(m_scan, spline, fixed_mask=fixed_mask)

        assert np.all(refined[fixed_mask] == spline[fixed_mask])
        assert np.all(refined[~fixed_mask] == 49)


# ---------------------------------------------------------------------------
# Integration test – mock file round-trip
# ---------------------------------------------------------------------------

class TestIntegrationSaveLoad:
    """Simulate the full annotate-and-save workflow with a temp file."""

    def test_output_filename_convention(self, tmp_path: Path):
        # Create a mock M-scan .npy
        data = np.random.rand(100, 200).astype(np.float64)
        src = tmp_path / "scan_001_m-scan.npy"
        np.save(str(src), data)

        # Simulate annotation
        xs = np.array([0.0, 100.0, 199.0])
        ys = np.array([50.0, 55.0, 52.0])
        spline = fit_spline(xs, ys, width=200)
        refined = refine_boundary(data, spline)

        # Save as the app would
        out_dir = tmp_path / "annotated"
        out_dir.mkdir()
        out_name = src.stem + "_annotations.npy"
        out_path = out_dir / out_name
        np.save(str(out_path), refined.astype(np.uint16).reshape(-1, 1))

        # Verify
        assert out_path.exists()
        assert out_path.parent == out_dir
        assert "_annotations.npy" in out_path.name
        loaded = np.load(str(out_path))
        assert loaded.shape == (200, 1)
        assert loaded.dtype == np.uint16


# ---------------------------------------------------------------------------
# Performance test – gradient refinement < 100 ms for 1000-wide scan
# ---------------------------------------------------------------------------

class TestPerformance:
    def test_refine_under_100ms(self):
        m_scan = np.random.rand(1024, 1000).astype(np.float64)
        indices = np.full(1000, 512, dtype=np.int64)

        start = time.perf_counter()
        refine_boundary(m_scan, indices)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100, f"Refinement took {elapsed_ms:.1f} ms (limit 100 ms)"


# ---------------------------------------------------------------------------
# NaN-sentinel integration test
# ---------------------------------------------------------------------------

NAN_SENTINEL = np.uint16(65535)


class TestNaNSentinel:
    """Verify that NaN-window columns get the 65535 sentinel in saved output."""

    def test_sentinel_applied_to_masked_columns(self, tmp_path: Path):
        data = np.random.rand(100, 200).astype(np.float64)
        xs = np.array([0.0, 100.0, 199.0])
        ys = np.array([50.0, 55.0, 52.0])
        spline = fit_spline(xs, ys, width=200)
        refined = refine_boundary(data, spline)

        # Simulate a NaN mask covering columns 80-120
        to_save = refined.astype(np.uint16).reshape(-1)
        nan_mask = np.zeros(200, dtype=np.bool_)
        nan_mask[80:121] = True
        to_save[nan_mask] = NAN_SENTINEL
        to_save = to_save.reshape(-1, 1)

        out_path = tmp_path / "test_annotations.npy"
        np.save(str(out_path), to_save)

        loaded = np.load(str(out_path))
        assert loaded.dtype == np.uint16
        assert loaded.shape == (200, 1)
        # Masked columns should be sentinel
        assert np.all(loaded[80:121, 0] == NAN_SENTINEL)
        # Non-masked columns should NOT be sentinel (extremely unlikely by chance)
        assert not np.any(loaded[:80, 0] == NAN_SENTINEL)
        assert not np.any(loaded[121:, 0] == NAN_SENTINEL)


# ---------------------------------------------------------------------------
# PNG export test
# ---------------------------------------------------------------------------

class TestRenderAnnotationPng:
    """Verify PNG export produces a file."""

    def test_png_is_created(self, tmp_path: Path):
        rows, cols = 100, 200
        m_scan = np.random.rand(rows, cols).astype(np.float64)
        ann = np.full(cols, 50, dtype=np.uint16)
        nan_mask = np.zeros(cols, dtype=np.bool_)
        nan_mask[80:100] = True
        ann[nan_mask] = NAN_SENTINEL

        out = tmp_path / "test_overlay.png"
        render_annotation_png(m_scan, ann, nan_mask, str(out))

        assert out.exists()
        assert out.stat().st_size > 0


class TestPreviewConversion:
    """Verify DB-style clip+normalize conversion used for preview/export."""

    def test_clip_and_normalize_bounds(self):
        data = np.array([[0.0, 1.0, 2.5, 4.0, 6.0]], dtype=np.float64)
        out = to_preview_uint8(data)

        assert out.dtype == np.uint8
        assert int(out[0, 0]) == 0
        assert int(out[0, 1]) == 63
        assert int(out[0, 2]) == 159
        assert int(out[0, 3]) == 255
        assert int(out[0, 4]) == 255


# ---------------------------------------------------------------------------
# Annotation-length matching test
# ---------------------------------------------------------------------------

class TestAnnotationLength:
    """Ensure spline output length always matches the requested width."""

    def test_spline_matches_width(self):
        xs = np.array([0.0, 256.0, 511.0])
        ys = np.array([100.0, 120.0, 110.0])
        for width in [512, 1000, 256]:
            result = fit_spline(xs, ys, width=width)
            assert result.shape == (width,), f"Expected length {width}, got {result.shape}"
