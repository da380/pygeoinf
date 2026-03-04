import numpy as np
import pytest
from pygeoinf.convex_optimisation import SciPyQPSolver, QPResult, best_available_qp_solver

np.random.seed(42)

# Shared 2D QP:
# min 0.5*(x0^2 + x1^2) - x0 - x1
# s.t. x0 + x1 <= 2 (upper), x0 >= 0 (lower), x1 >= 0 (lower)
# Optimal: x* = [1, 1], obj = -1
P = np.eye(2)
q = np.array([-1.0, -1.0])
A = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
l = np.array([-np.inf, 0.0, 0.0])
u = np.array([2.0, np.inf, np.inf])
x_true = np.array([1.0, 1.0])


def test_scipy_qp_known_solution():
    solver = SciPyQPSolver()
    result = solver.solve(P, q, A, l, u)
    assert result.status == 'solved'
    np.testing.assert_allclose(result.x, x_true, atol=1e-5)


def test_osqp_qp_known_solution():
    osqp = pytest.importorskip('osqp')
    from pygeoinf.convex_optimisation import OSQPQPSolver
    solver = OSQPQPSolver()
    result = solver.solve(P, q, A, l, u)
    assert result.status == 'solved'
    np.testing.assert_allclose(result.x, x_true, atol=1e-4)


def test_clarabel_qp_known_solution():
    clarabel = pytest.importorskip('clarabel')
    from pygeoinf.convex_optimisation import ClarabelQPSolver
    solver = ClarabelQPSolver()
    result = solver.solve(P, q, A, l, u)
    assert 'solved' in result.status.lower() or result.status == 'solved'
    np.testing.assert_allclose(result.x, x_true, atol=1e-4)


def test_osqp_warm_start():
    osqp = pytest.importorskip('osqp')
    from pygeoinf.convex_optimisation import OSQPQPSolver
    solver = OSQPQPSolver()
    r1 = solver.solve(P, q, A, l, u)
    r2 = solver.solve(P, q, A, l, u, x0=r1.x)
    # Both should give same answer
    np.testing.assert_allclose(r2.x, x_true, atol=1e-4)


def test_best_available_returns_solver():
    solver = best_available_qp_solver()
    result = solver.solve(P, q, A, l, u)
    assert result.status == 'solved' or 'solved' in result.status.lower()
    np.testing.assert_allclose(result.x, x_true, atol=1e-3)


def test_backends_agree():
    """All available backends give the same solution."""
    from pygeoinf.convex_optimisation import SciPyQPSolver
    results = [SciPyQPSolver().solve(P, q, A, l, u)]
    try:
        from pygeoinf.convex_optimisation import OSQPQPSolver
        results.append(OSQPQPSolver().solve(P, q, A, l, u))
    except ImportError:
        pass
    try:
        from pygeoinf.convex_optimisation import ClarabelQPSolver
        results.append(ClarabelQPSolver().solve(P, q, A, l, u))
    except ImportError:
        pass
    for r in results:
        np.testing.assert_allclose(r.x, x_true, atol=1e-3)
