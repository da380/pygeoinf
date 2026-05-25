"""
Tests for exact quadratic-slice rendering of Ball and Ellipsoid in SubspaceSlicePlotter.

Phase 1 tests: Ball exact slices in 1D and 2D.
Phase 2 tests: Ellipsoid exact slices in 1D and 2D.
"""

import warnings

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.pyplot as plt
import numpy as np
import pytest

import pygeoinf as inf
from pygeoinf.plot import plot_slice
from pygeoinf.subsets import Ball, Ellipsoid
from pygeoinf.subspaces import AffineSubspace


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after every test to prevent resource leaks."""
    yield
    plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _space2():
    return inf.EuclideanSpace(2)


def _make_subspace_2d(translation=None):
    """Full 2D subspace (the whole plane) on R²."""
    space = _space2()
    basis = [space.basis_vector(0), space.basis_vector(1)]
    origin = translation if translation is not None else np.zeros(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AffineSubspace.from_tangent_basis(space, basis, translation=origin)


def _make_line_subspace(axis=0, translation=None):
    """1D subspace (a line) along `axis` with optional translation offset."""
    space = _space2()
    basis = [space.basis_vector(axis)]
    origin = translation if translation is not None else np.zeros(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AffineSubspace.from_tangent_basis(space, basis, translation=origin)


def _make_ball(center=None, radius=1.0):
    space = _space2()
    if center is None:
        center = np.zeros(2)
    return Ball(space, center, radius, open_set=False)


def _make_ellipsoid(center=None, radius=1.0, A_mat=None):
    space = _space2()
    if center is None:
        center = np.zeros(2)
    if A_mat is None:
        # Default: moderately elongated SPD matrix
        A_mat = np.array([[2.0, 0.5], [0.5, 1.0]])
    A_op = inf.LinearOperator.from_matrix(space, space, A_mat)
    return Ellipsoid(space, center, radius, A_op, open_set=False)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Ball exact slices — 1D and 2D
# ─────────────────────────────────────────────────────────────────────────────

class TestBallExact1D:
    """Ball sliced by a line through the center."""

    def test_returns_interval_payload_shape(self):
        ball = _make_ball(center=np.zeros(2), radius=1.0)
        subspace = _make_line_subspace(axis=0)
        _, _, payload = plot_slice(ball, subspace, bounds=(-2.0, 2.0), show_plot=False)
        assert isinstance(payload, np.ndarray)
        assert payload.shape == (2,)

    def test_interval_endpoints_match_radius_horizontal(self):
        """Ball(center=0, r=1) sliced by x-axis → interval [-1, 1]."""
        ball = _make_ball(center=np.zeros(2), radius=1.0)
        subspace = _make_line_subspace(axis=0)
        _, _, payload = plot_slice(ball, subspace, bounds=(-2.0, 2.0), show_plot=False)
        np.testing.assert_allclose(payload[0], -1.0, atol=1e-10)
        np.testing.assert_allclose(payload[1], 1.0, atol=1e-10)

    def test_interval_endpoints_match_radius_vertical(self):
        """Ball(center=0, r=1) sliced by y-axis → interval [-1, 1]."""
        ball = _make_ball(center=np.zeros(2), radius=1.0)
        subspace = _make_line_subspace(axis=1)
        _, _, payload = plot_slice(ball, subspace, bounds=(-2.0, 2.0), show_plot=False)
        np.testing.assert_allclose(payload[0], -1.0, atol=1e-10)
        np.testing.assert_allclose(payload[1], 1.0, atol=1e-10)

    def test_off_center_ball_shifts_interval(self):
        """Ball centered at (0.5, 0) sliced by x-axis → interval [-0.5, 1.5]."""
        ball = _make_ball(center=np.array([0.5, 0.0]), radius=1.0)
        subspace = _make_line_subspace(axis=0)
        _, _, payload = plot_slice(ball, subspace, bounds=(-2.0, 2.0), show_plot=False)
        np.testing.assert_allclose(payload[0], -0.5, atol=1e-10)
        np.testing.assert_allclose(payload[1], 1.5, atol=1e-10)

    def test_slice_outside_ball_raises_value_error(self):
        """Slice through y=2 of a unit ball (radius=1) should raise ValueError (empty)."""
        ball = _make_ball(center=np.zeros(2), radius=1.0)
        # Translation y=2 puts the line outside the ball
        subspace = _make_line_subspace(axis=0, translation=np.array([0.0, 2.0]))
        with pytest.raises(ValueError, match="empty"):
            plot_slice(ball, subspace, bounds=(-2.0, 2.0), show_plot=False)

    def test_bypasses_membership_sampling(self, monkeypatch):
        """The exact path must not call is_element at all."""
        ball = _make_ball()
        subspace = _make_line_subspace(axis=0)
        call_count = [0]
        original = ball.is_element
        def counting(x, /, **kwargs):
            call_count[0] += 1
            return original(x, **kwargs)
        monkeypatch.setattr(ball, "is_element", counting)
        plot_slice(ball, subspace, bounds=(-2.0, 2.0), show_plot=False)
        assert call_count[0] == 0, f"Expected 0 is_element calls, got {call_count[0]}"

    def test_bounds_clipping_applied(self):
        """Very large ball: plot bounds must clip the rendered interval."""
        ball = _make_ball(center=np.zeros(2), radius=100.0)
        subspace = _make_line_subspace(axis=0)
        _, _, payload = plot_slice(ball, subspace, bounds=(-1.0, 1.0), show_plot=False)
        assert payload[0] >= -1.0 - 1e-12
        assert payload[1] <= 1.0 + 1e-12


class TestBallExact2D:
    """Ball sliced by the full 2D plane."""

    def test_returns_boundary_points_payload(self):
        ball = _make_ball(center=np.zeros(2), radius=1.0)
        subspace = _make_subspace_2d()
        _, _, payload = plot_slice(ball, subspace, bounds=(-2, 2, -2, 2), show_plot=False)
        assert isinstance(payload, np.ndarray)
        assert payload.ndim == 2
        assert payload.shape[1] == 2

    def test_boundary_points_at_unit_distance(self):
        """All boundary points of a unit ball should be at radius 1 from origin."""
        ball = _make_ball(center=np.zeros(2), radius=1.0)
        subspace = _make_subspace_2d()
        _, _, payload = plot_slice(ball, subspace, bounds=(-2, 2, -2, 2), show_plot=False)
        norms = np.linalg.norm(payload, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_boundary_points_at_correct_radius_large_ball(self):
        """Boundary points of a radius-2 ball should be at radius 2."""
        ball = _make_ball(center=np.zeros(2), radius=2.0)
        subspace = _make_subspace_2d()
        _, _, payload = plot_slice(ball, subspace, bounds=(-3, 3, -3, 3), show_plot=False)
        norms = np.linalg.norm(payload, axis=1)
        np.testing.assert_allclose(norms, 2.0, atol=1e-6)

    def test_boundary_points_respect_offset_center(self):
        """Boundary points of a ball at c=[0.3, -0.2] should cluster around c."""
        center = np.array([0.3, -0.2])
        ball = _make_ball(center=center, radius=1.0)
        subspace = _make_subspace_2d()
        _, _, payload = plot_slice(ball, subspace, bounds=(-3, 3, -3, 3), show_plot=False)
        dists = np.linalg.norm(payload - center, axis=1)
        np.testing.assert_allclose(dists, 1.0, atol=1e-6)

    def test_bypasses_membership_sampling_2d(self, monkeypatch):
        ball = _make_ball()
        subspace = _make_subspace_2d()
        call_count = [0]
        original = ball.is_element
        def counting(x, /, **kwargs):
            call_count[0] += 1
            return original(x, **kwargs)
        monkeypatch.setattr(ball, "is_element", counting)
        plot_slice(ball, subspace, bounds=(-2, 2, -2, 2), show_plot=False)
        assert call_count[0] == 0

    def test_returns_figure_and_axes(self):
        ball = _make_ball()
        subspace = _make_subspace_2d()
        fig, ax, _ = plot_slice(ball, subspace, bounds=(-2, 2, -2, 2), show_plot=False)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

    def test_slice_outside_ball_1d_raises_value_error(self):
        """Slice a 2D space along a horizontal line at y=2: that line misses a unit ball at origin."""
        ball = _make_ball(center=np.zeros(2), radius=1.0)
        # Horizontal line at y=2 — outside the unit ball
        subspace = _make_line_subspace(axis=0, translation=np.array([0.0, 2.0]))
        with pytest.raises(ValueError, match="empty"):
            plot_slice(ball, subspace, bounds=(-2, 2), show_plot=False)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Ellipsoid exact slices — 1D and 2D
# ─────────────────────────────────────────────────────────────────────────────

class TestEllipsoidExact1D:
    """Ellipsoid sliced by a line through its center."""

    def test_returns_interval_payload_shape(self):
        ellipsoid = _make_ellipsoid(center=np.zeros(2))
        subspace = _make_line_subspace(axis=0)
        _, _, payload = plot_slice(ellipsoid, subspace, bounds=(-5.0, 5.0), show_plot=False)
        assert isinstance(payload, np.ndarray)
        assert payload.shape == (2,)

    def test_interval_endpoints_match_quadratic_roots_axis0(self):
        """
        Ellipsoid {x : x^T A x <= r^2} with A = diag(a, b).
        Sliced by x-axis (y=0): x1² * a <= r² → x1 in [-r/sqrt(a), r/sqrt(a)].
        """
        A_mat = np.array([[4.0, 0.0], [0.0, 1.0]])  # eigenvalues 4, 1
        r = 1.0
        ellipsoid = _make_ellipsoid(center=np.zeros(2), radius=r, A_mat=A_mat)
        subspace = _make_line_subspace(axis=0)  # x-axis, y=0 in slice
        _, _, payload = plot_slice(ellipsoid, subspace, bounds=(-3.0, 3.0), show_plot=False)
        # x1 in [-1/sqrt(4), 1/sqrt(4)] = [-0.5, 0.5]
        expected_half_len = r / np.sqrt(4.0)
        np.testing.assert_allclose(payload[0], -expected_half_len, atol=1e-10)
        np.testing.assert_allclose(payload[1], expected_half_len, atol=1e-10)

    def test_interval_endpoints_match_quadratic_roots_axis1(self):
        A_mat = np.array([[4.0, 0.0], [0.0, 1.0]])
        r = 1.0
        ellipsoid = _make_ellipsoid(center=np.zeros(2), radius=r, A_mat=A_mat)
        subspace = _make_line_subspace(axis=1)  # y-axis, x=0 in slice
        _, _, payload = plot_slice(ellipsoid, subspace, bounds=(-3.0, 3.0), show_plot=False)
        # x2 in [-1/sqrt(1), 1/sqrt(1)] = [-1, 1]
        np.testing.assert_allclose(payload[0], -1.0, atol=1e-10)
        np.testing.assert_allclose(payload[1], 1.0, atol=1e-10)

    def test_slice_outside_ellipsoid_raises_value_error(self):
        ellipsoid = _make_ellipsoid(center=np.zeros(2), radius=1.0)
        # Translate line far from the ellipsoid
        subspace = _make_line_subspace(axis=0, translation=np.array([0.0, 5.0]))
        with pytest.raises(ValueError, match="empty"):
            plot_slice(ellipsoid, subspace, bounds=(-5.0, 5.0), show_plot=False)

    def test_bypasses_membership_sampling(self, monkeypatch):
        ellipsoid = _make_ellipsoid()
        subspace = _make_line_subspace(axis=0)
        call_count = [0]
        original = ellipsoid.is_element
        def counting(x, /, **kwargs):
            call_count[0] += 1
            return original(x, **kwargs)
        monkeypatch.setattr(ellipsoid, "is_element", counting)
        plot_slice(ellipsoid, subspace, bounds=(-3.0, 3.0), show_plot=False)
        assert call_count[0] == 0


class TestEllipsoidExact2D:
    """Ellipsoid sliced by the full 2D plane."""

    def test_returns_boundary_points_payload(self):
        ellipsoid = _make_ellipsoid(center=np.zeros(2))
        subspace = _make_subspace_2d()
        _, _, payload = plot_slice(ellipsoid, subspace, bounds=(-3, 3, -3, 3), show_plot=False)
        assert isinstance(payload, np.ndarray)
        assert payload.ndim == 2
        assert payload.shape[1] == 2

    def test_boundary_points_satisfy_quadratic_form(self):
        """All boundary points x should satisfy ⟨Ax, x⟩ ≈ r²."""
        A_mat = np.array([[2.0, 0.5], [0.5, 1.0]])
        r = 1.5
        center = np.array([0.3, -0.2])
        ellipsoid = _make_ellipsoid(center=center, radius=r, A_mat=A_mat)
        subspace = _make_subspace_2d()
        _, _, payload = plot_slice(ellipsoid, subspace, bounds=(-5, 5, -5, 5), show_plot=False)
        for pt in payload:
            d = pt - center
            quad_val = d @ A_mat @ d
            np.testing.assert_allclose(quad_val, r**2, atol=1e-6,
                                        err_msg=f"Point {pt} not on ellipsoid boundary")

    def test_boundary_points_for_identity_match_ball_boundary(self):
        """Ellipsoid with A=I and radius r should give the same boundary as Ball."""
        r = 1.3
        center = np.array([0.1, 0.2])
        A_mat = np.eye(2)
        ellipsoid = _make_ellipsoid(center=center, radius=r, A_mat=A_mat)
        ball = _make_ball(center=center, radius=r)
        subspace = _make_subspace_2d()
        _, _, pts_ell = plot_slice(ellipsoid, subspace, bounds=(-4, 4, -4, 4), show_plot=False)
        _, _, pts_ball = plot_slice(ball, subspace, bounds=(-4, 4, -4, 4), show_plot=False)
        # Both should have the same number of boundary points and radii
        norms_ell = np.linalg.norm(pts_ell - center, axis=1)
        norms_ball = np.linalg.norm(pts_ball - center, axis=1)
        np.testing.assert_allclose(norms_ell, r, atol=1e-6)
        np.testing.assert_allclose(norms_ball, r, atol=1e-6)

    def test_bypasses_membership_sampling_2d(self, monkeypatch):
        ellipsoid = _make_ellipsoid()
        subspace = _make_subspace_2d()
        call_count = [0]
        original = ellipsoid.is_element
        def counting(x, /, **kwargs):
            call_count[0] += 1
            return original(x, **kwargs)
        monkeypatch.setattr(ellipsoid, "is_element", counting)
        plot_slice(ellipsoid, subspace, bounds=(-3, 3, -3, 3), show_plot=False)
        assert call_count[0] == 0

    def test_degenerate_slice_raises_value_error(self):
        """Slice line outside the ellipsoid should raise ValueError explicitly."""
        ellipsoid = _make_ellipsoid(center=np.zeros(2), radius=1.0)
        # A horizontal line at y=5 is well outside this ellipsoid
        subspace = _make_line_subspace(axis=0, translation=np.array([0.0, 5.0]))
        with pytest.raises(ValueError, match="empty"):
            plot_slice(ellipsoid, subspace, bounds=(-5.0, 5.0), show_plot=False)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-checks: exact path does not break non-quadratic subsets
# ─────────────────────────────────────────────────────────────────────────────

class TestSampledPathPreserved:
    """Non-Ball/Ellipsoid subsets must still go through the sampled raster path."""

    def test_polyhedral_still_uses_exact_path(self):
        """PolyhedralSet must still use its own exact path (no sampling)."""
        from pygeoinf.subsets import HalfSpace, PolyhedralSet
        space = _space2()
        hs1 = HalfSpace(space, space.basis_vector(0), 1.0)
        hs2 = HalfSpace(space, -space.basis_vector(0), 1.0)
        hs3 = HalfSpace(space, space.basis_vector(1), 1.0)
        hs4 = HalfSpace(space, -space.basis_vector(1), 1.0)
        polytope = PolyhedralSet(space, [hs1, hs2, hs3, hs4])
        subspace = _make_subspace_2d()
        fig, ax, payload = plot_slice(polytope, subspace, bounds=(-2, 2, -2, 2), show_plot=False)
        # The polyhedral fast-path payload is vertex coordinates (not boundary points)
        assert isinstance(payload, np.ndarray)
