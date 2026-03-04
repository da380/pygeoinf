"""
Tests for solve_support_values in convex_optimisation.py.

Covers:
- Single-direction call
- Multiple directions sequential
- Warm-start vs cold-start agreement
- Parallel agrees with sequential (requires joblib)
- Return types
"""

import numpy as np
import pytest

from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.convex_analysis import BallSupportFunction
from pygeoinf.backus_gilbert import DualMasterCostFunction
from pygeoinf.convex_optimisation import (
    ProximalBundleMethod,
    BundleResult,
    solve_support_values,
)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_cost_and_lambda0(n_prop: int = 2):
    """Return (cost, lambda0) for a small DualMasterCostFunction.

    Parameters:
        n_prop: Dimension of the property space (default 2 so we have
            multiple distinct directions to evaluate).
    """
    rng = np.random.default_rng(42)
    n_data = 4
    n_model = 5

    data_space = EuclideanSpace(n_data)
    model_space = EuclideanSpace(n_model)
    prop_space = EuclideanSpace(n_prop)

    G_matrix = rng.standard_normal((n_data, n_model))
    T_matrix = rng.standard_normal((n_prop, n_model))
    G = LinearOperator.from_matrix(model_space, data_space, G_matrix)
    T = LinearOperator.from_matrix(model_space, prop_space, T_matrix)

    model_prior = BallSupportFunction(model_space, model_space.zero, 1.0)
    data_error = BallSupportFunction(data_space, data_space.zero, 0.5)

    d_tilde = rng.standard_normal(n_data)
    q0 = np.array([1.0, 0.0]) if n_prop == 2 else rng.standard_normal(n_prop)

    cost = DualMasterCostFunction(
        data_space, prop_space, model_space,
        G, T, model_prior, data_error, d_tilde, q0
    )
    lambda0 = data_space.zero
    return cost, lambda0


def _make_solver(cost):
    return ProximalBundleMethod(cost, tolerance=1e-3, max_iterations=100)


# Canonical set of 2-D directions used across tests.
_DIRECTIONS = [
    np.array([1.0, 0.0]),
    np.array([0.0, 1.0]),
    np.array([-1.0, 0.0]),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_single_direction():
    """solve_support_values for a single direction returns array of shape (1,)."""
    cost, lambda0 = _make_cost_and_lambda0()
    solver = _make_solver(cost)

    qs = [np.array([1.0, 0.0])]
    values, lambdas, diagnostics = solve_support_values(
        cost, qs, solver, lambda0
    )

    assert values.shape == (1,), f"Expected shape (1,), got {values.shape}"
    assert np.isfinite(values[0]), "Support value should be finite"

    # Verify consistency with a direct solve on the same direction.
    cost.set_direction(qs[0])
    direct = solver.solve(lambda0)
    np.testing.assert_allclose(
        values[0], direct.f_best, atol=1e-2,
        err_msg="Single-direction value should match direct solve"
    )


def test_multiple_directions_sequential():
    """solve_support_values returns (p,) array with all-finite values for p directions."""
    cost, lambda0 = _make_cost_and_lambda0()
    solver = _make_solver(cost)

    values, lambdas, diagnostics = solve_support_values(
        cost, _DIRECTIONS, solver, lambda0
    )

    assert values.shape == (3,), f"Expected shape (3,), got {values.shape}"
    assert all(np.isfinite(v) for v in values), "All support values should be finite"
    assert len(lambdas) == 3, "Should return one lambda per direction"
    assert len(diagnostics) == 3, "Should return one diagnostic per direction"

    # Cross-check each direction against an independent solve.
    for i, q in enumerate(_DIRECTIONS):
        cost.set_direction(q)
        direct = solver.solve(lambda0)
        np.testing.assert_allclose(
            values[i], direct.f_best, atol=5e-2,
            err_msg=f"Direction {i}: value {values[i]:.4f} vs direct {direct.f_best:.4f}"
        )


def test_warm_start_vs_cold_start_agreement():
    """Warm-start and cold-start should produce the same support values (atol=1e-2)."""
    cost, lambda0 = _make_cost_and_lambda0()

    solver_warm = _make_solver(cost)
    solver_cold = _make_solver(cost)

    vals_warm, _, _ = solve_support_values(
        cost, _DIRECTIONS, solver_warm, lambda0, warm_start=True
    )
    vals_cold, _, _ = solve_support_values(
        cost, _DIRECTIONS, solver_cold, lambda0, warm_start=False
    )

    np.testing.assert_allclose(
        vals_warm, vals_cold, atol=1e-2,
        err_msg="Warm-start and cold-start values should agree within atol=1e-2"
    )


def test_parallel_agrees_with_sequential():
    """Parallel (n_jobs=2) results should agree with sequential (atol=1e-2)."""
    joblib = pytest.importorskip("joblib")  # skip if not installed

    cost, lambda0 = _make_cost_and_lambda0()
    solver_seq = _make_solver(cost)
    solver_par = _make_solver(cost)

    vals_seq, _, _ = solve_support_values(
        cost, _DIRECTIONS, solver_seq, lambda0, n_jobs=1
    )
    vals_par, _, _ = solve_support_values(
        cost, _DIRECTIONS, solver_par, lambda0, n_jobs=2
    )

    np.testing.assert_allclose(
        vals_par, vals_seq, atol=1e-2,
        err_msg="Parallel values should agree with sequential values"
    )


def test_returns_correct_types():
    """solve_support_values must return (np.ndarray, list, list[BundleResult])."""
    cost, lambda0 = _make_cost_and_lambda0()
    solver = _make_solver(cost)

    values, lambdas, diagnostics = solve_support_values(
        cost, _DIRECTIONS, solver, lambda0
    )

    assert isinstance(values, np.ndarray), "values should be np.ndarray"
    assert isinstance(lambdas, list), "lambdas should be a list"
    assert isinstance(diagnostics, list), "diagnostics should be a list"
    assert all(isinstance(d, BundleResult) for d in diagnostics), (
        "Each diagnostic should be a BundleResult"
    )
