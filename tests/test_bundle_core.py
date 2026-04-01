"""
Tests for Phase 1 bundle method core data structures.

Tests cover: Cut, Bundle, QPResult, QPSolver Protocol, SciPyQPSolver, BundleResult.
"""

from __future__ import annotations

import numpy as np
import pytest

np.random.seed(42)

from pygeoinf.hilbert_space import EuclideanSpace


# ---------------------------------------------------------------------------
# Cut tests
# ---------------------------------------------------------------------------


def test_cut_fields():
    """Cut stores all fields correctly."""
    from pygeoinf.convex_optimisation import Cut

    space = EuclideanSpace(2)
    x = np.array([1.0, 2.0])
    g = np.array([0.5, -1.0])
    cut = Cut(x=x, f=3.7, g=g, iteration=5)

    np.testing.assert_array_equal(cut.x, x)
    assert cut.f == pytest.approx(3.7)
    np.testing.assert_array_equal(cut.g, g)
    assert cut.iteration == 5


# ---------------------------------------------------------------------------
# Bundle tests
# ---------------------------------------------------------------------------


def test_bundle_add_and_len():
    """Adding 3 cuts gives __len__ == 3."""
    from pygeoinf.convex_optimisation import Bundle, Cut

    space = EuclideanSpace(2)
    bundle = Bundle()
    for i in range(3):
        x = np.array([float(i), 0.0])
        g = np.array([1.0, 0.0])
        bundle.add_cut(Cut(x=x, f=float(i + 1), g=g, iteration=i))
    assert len(bundle) == 3


def test_bundle_upper_bound():
    """upper_bound() returns the minimum f value among all cuts."""
    from pygeoinf.convex_optimisation import Bundle, Cut

    bundle = Bundle()
    f_vals = [5.0, 2.0, 8.0]
    for i, f in enumerate(f_vals):
        x = np.array([float(i)])
        g = np.array([1.0])
        bundle.add_cut(Cut(x=x, f=f, g=g, iteration=i))
    assert bundle.upper_bound() == pytest.approx(2.0)


def test_bundle_best_point_matches_best_f():
    """best_point() returns the x corresponding to the cut with the smallest f."""
    from pygeoinf.convex_optimisation import Bundle, Cut

    bundle = Bundle()
    f_vals = [5.0, 2.0, 8.0]
    xs = [np.array([1.0]), np.array([3.0]), np.array([0.0])]
    for i, (x, f) in enumerate(zip(xs, f_vals)):
        g = np.array([0.0])
        bundle.add_cut(Cut(x=x, f=f, g=g, iteration=i))
    # best f == 2.0 at xs[1] == [3.0]
    np.testing.assert_array_equal(bundle.best_point(), np.array([3.0]))


def test_bundle_linearization_matrix_shape():
    """linearization_matrix returns A of shape (n_cuts, dim+1) and b of shape (n_cuts,)."""
    from pygeoinf.convex_optimisation import Bundle, Cut

    space = EuclideanSpace(3)
    bundle = Bundle()
    for i in range(3):
        x = np.random.randn(3)
        g = np.random.randn(3)
        bundle.add_cut(Cut(x=x, f=float(i), g=g, iteration=i))
    stability_center = np.zeros(3)
    A, b = bundle.linearization_matrix(stability_center, space)
    assert A.shape == (3, 4), f"Expected (3, 4), got {A.shape}"
    assert b.shape == (3,), f"Expected (3,), got {b.shape}"


def test_bundle_linearization_matrix_values():
    """
    1D case: one cut at x=0 with f=1, g=2, stability_center=0.
    Constraint: g*λ - t ≤ g*x_j - f = 2*0 - 1 = -1.
    So A[0] = [2, -1] and b[0] = -1.
    """
    from pygeoinf.convex_optimisation import Bundle, Cut

    space = EuclideanSpace(1)
    bundle = Bundle()
    x_j = np.array([0.0])
    g_j = np.array([2.0])
    cut = Cut(x=x_j, f=1.0, g=g_j, iteration=0)
    bundle.add_cut(cut)

    stability_center = np.array([0.0])
    A, b = bundle.linearization_matrix(stability_center, space)

    np.testing.assert_allclose(A[0], [2.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(b[0], -1.0, atol=1e-12)


def test_bundle_compress_keeps_recent():
    """After adding 5 cuts and compressing to max_size=2, only 2 cuts remain (the last 2)."""
    from pygeoinf.convex_optimisation import Bundle, Cut

    bundle = Bundle()
    for i in range(5):
        x = np.array([float(i)])
        g = np.array([1.0])
        bundle.add_cut(Cut(x=x, f=float(i), g=g, iteration=i))
    bundle.compress(max_size=2)
    assert len(bundle) == 2
    # The last 2 cuts have x = [3.0] and x = [4.0]
    cuts = bundle._cuts  # access internal list for verification
    np.testing.assert_array_equal(cuts[0].x, np.array([3.0]))
    np.testing.assert_array_equal(cuts[1].x, np.array([4.0]))


# ---------------------------------------------------------------------------
# SciPyQPSolver tests
# ---------------------------------------------------------------------------


def test_scipy_qp_solver_simple():
    """
    Solve: min 0.5*(x0^2 + x1^2) - x0 - x1
    s.t.  x0 + x1 <= 2
          x0 >= 0
          x1 >= 0

    Constraints in l <= Ax <= u form:
      A = [[1, 1],   l = [-inf, 0, 0]   u = [2, inf, inf]
           [1, 0],
           [0, 1]]

    Optimal: x* = [1, 1], obj = -1.
    """
    from pygeoinf.convex_optimisation import SciPyQPSolver

    solver = SciPyQPSolver()
    P = np.eye(2)
    q = np.array([-1.0, -1.0])
    A = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    l = np.array([-np.inf, 0.0, 0.0])
    u = np.array([2.0, np.inf, np.inf])

    result = solver.solve(P=P, q=q, A=A, l=l, u=u)

    assert result.status == "solved", f"Expected 'solved', got '{result.status}'"
    np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(result.obj, -1.0, atol=1e-5)


def test_scipy_qp_solver_infeasible():
    """
    Infeasible problem: min x^2 s.t. x >= 1 AND x <= -1.
    In l <= Ax <= u form with A = [[1], [1]]:
      row 0: 1 <= x <= inf  (x >= 1)
      row 1: -inf <= x <= -1  (x <= -1)
    """
    from pygeoinf.convex_optimisation import SciPyQPSolver

    solver = SciPyQPSolver()
    P = np.array([[2.0]])
    q = np.array([0.0])
    A = np.array([[1.0], [1.0]])
    l = np.array([1.0, -np.inf])
    u = np.array([np.inf, -1.0])

    result = solver.solve(P=P, q=q, A=A, l=l, u=u)
    assert result.status != "solved", f"Expected failure, got '{result.status}'"


# ---------------------------------------------------------------------------
# BundleResult structural test
# ---------------------------------------------------------------------------


def test_bundle_result_fields():
    """BundleResult can be constructed and fields are accessible."""
    from pygeoinf.convex_optimisation import BundleResult

    space = EuclideanSpace(2)
    x = np.array([1.0, 2.0])
    result = BundleResult(
        x_best=x,
        f_best=0.5,
        f_low=0.3,
        gap=0.2,
        converged=True,
        num_iterations=10,
        num_serious_steps=3,
        function_values=[1.0, 0.8, 0.5],
    )
    np.testing.assert_array_equal(result.x_best, x)
    assert result.f_best == pytest.approx(0.5)
    assert result.f_low == pytest.approx(0.3)
    assert result.gap == pytest.approx(0.2)
    assert result.converged is True
    assert result.num_iterations == 10
    assert result.num_serious_steps == 3
    assert result.iterates is None
