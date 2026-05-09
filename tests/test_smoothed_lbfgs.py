"""
Tests for SmoothedDualMaster and SmoothedLBFGSSolver in convex_optimisation.py.

Covers:
- Gradient consistency via central finite differences for BallSupportFunction
- Gradient consistency via central finite differences for EllipsoidSupportFunction
- Basic convergence of SmoothedLBFGSSolver on ball priors
- Agreement with ProximalBundleMethod solution (rtol=1e-2)
- NotImplementedError raised for unsupported SupportFunction types
"""

from __future__ import annotations

import numpy as np
import pytest

from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.convex_analysis import (
    BallSupportFunction,
    EllipsoidSupportFunction,
    SupportFunction,
)
from pygeoinf.backus_gilbert import DualMasterCostFunction
from pygeoinf.convex_optimisation import (
    SmoothedDualMaster,
    SmoothedLBFGSSolver,
    ProximalBundleMethod,
    BundleResult,
)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Shared dimensions
# ---------------------------------------------------------------------------
N_DATA = 4
N_MODEL = 5
N_PROP = 1


# ---------------------------------------------------------------------------
# Helper: unsupported SupportFunction for error tests
# ---------------------------------------------------------------------------

class _UnsupportedSupport(SupportFunction):
    """A SupportFunction subclass not handled by SmoothedDualMaster."""

    def _mapping(self, q: object) -> float:  # type: ignore[override]
        return 0.0


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_operators(rng):
    """Return (data_space, model_space, prop_space, G, T)."""
    data_space = EuclideanSpace(N_DATA)
    model_space = EuclideanSpace(N_MODEL)
    prop_space = EuclideanSpace(N_PROP)

    G_matrix = rng.standard_normal((N_DATA, N_MODEL))
    T_matrix = rng.standard_normal((N_PROP, N_MODEL))
    G = LinearOperator.from_matrix(model_space, data_space, G_matrix)
    T = LinearOperator.from_matrix(model_space, prop_space, T_matrix)
    return data_space, model_space, prop_space, G, T


def _make_ball_cost():
    """Return (cost, data_space) with BallSupportFunction priors."""
    rng = np.random.default_rng(42)
    data_space, model_space, prop_space, G, T = _make_operators(rng)

    model_prior = BallSupportFunction(model_space, model_space.zero, 1.0)
    data_error = BallSupportFunction(data_space, data_space.zero, 0.5)

    d_tilde = rng.standard_normal(N_DATA)
    q = rng.standard_normal(N_PROP)

    cost = DualMasterCostFunction(
        data_space, prop_space, model_space,
        G, T, model_prior, data_error, d_tilde, q,
    )
    return cost, data_space


def _make_ellipsoid_cost():
    """Return (cost, data_space) with EllipsoidSupportFunction priors.

    Uses diagonal A = 2*I in both model and data spaces so that
    A^{-1} = 0.5*I and A^{-1/2} = I/sqrt(2) are easy to construct.
    """
    rng = np.random.default_rng(42)
    data_space, model_space, prop_space, G, T = _make_operators(rng)

    # Model space: A = 2I
    A_model_mat = 2.0 * np.eye(N_MODEL)
    A_model = LinearOperator.from_matrix(model_space, model_space, A_model_mat)
    A_model_inv_mat = 0.5 * np.eye(N_MODEL)
    A_model_inv = LinearOperator.from_matrix(model_space, model_space, A_model_inv_mat)
    A_model_inv_sqrt_mat = (1.0 / np.sqrt(2.0)) * np.eye(N_MODEL)
    A_model_inv_sqrt = LinearOperator.from_matrix(
        model_space, model_space, A_model_inv_sqrt_mat
    )

    # Data space: A = 2I
    A_data_mat = 2.0 * np.eye(N_DATA)
    A_data = LinearOperator.from_matrix(data_space, data_space, A_data_mat)
    A_data_inv_mat = 0.5 * np.eye(N_DATA)
    A_data_inv = LinearOperator.from_matrix(data_space, data_space, A_data_inv_mat)
    A_data_inv_sqrt_mat = (1.0 / np.sqrt(2.0)) * np.eye(N_DATA)
    A_data_inv_sqrt = LinearOperator.from_matrix(
        data_space, data_space, A_data_inv_sqrt_mat
    )

    model_prior = EllipsoidSupportFunction(
        model_space, model_space.zero, 1.0,
        A_model,
        inverse_operator=A_model_inv,
        inverse_sqrt_operator=A_model_inv_sqrt,
    )
    data_error = EllipsoidSupportFunction(
        data_space, data_space.zero, 0.5,
        A_data,
        inverse_operator=A_data_inv,
        inverse_sqrt_operator=A_data_inv_sqrt,
    )

    d_tilde = rng.standard_normal(N_DATA)
    q = rng.standard_normal(N_PROP)

    cost = DualMasterCostFunction(
        data_space, prop_space, model_space,
        G, T, model_prior, data_error, d_tilde, q,
    )
    return cost, data_space


def _fd_gradient(smoothed: SmoothedDualMaster, lam, data_space, eps_fd: float = 1e-5):
    """Central finite-difference gradient of smoothed objective at lam."""
    comps = data_space.to_components(lam)
    n = comps.shape[0]
    fd_grad = np.zeros(n)
    for i in range(n):
        step = np.zeros(n)
        step[i] = eps_fd
        lam_p = data_space.from_components(comps + step)
        lam_m = data_space.from_components(comps - step)
        fd_grad[i] = (smoothed(lam_p) - smoothed(lam_m)) / (2.0 * eps_fd)
    return fd_grad


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_smoothed_ball_gradient_consistency():
    """Analytic gradient of SmoothedDualMaster (ball priors) matches central FD, rtol=1e-3."""
    cost, data_space = _make_ball_cost()
    rng = np.random.default_rng(7)
    lam = data_space.from_components(rng.standard_normal(N_DATA))
    epsilon = 1e-2

    smoothed = SmoothedDualMaster(cost, epsilon)

    analytic_grad_vec = smoothed.gradient(lam)
    analytic_grad = data_space.to_components(analytic_grad_vec)
    fd_grad = _fd_gradient(smoothed, lam, data_space)

    np.testing.assert_allclose(
        analytic_grad, fd_grad, rtol=1e-3, atol=1e-7,
        err_msg="Analytic ball gradient should match finite-difference gradient",
    )


def test_smoothed_ellipsoid_gradient_consistency():
    """Analytic gradient of SmoothedDualMaster (ellipsoid priors) matches central FD, rtol=1e-3."""
    cost, data_space = _make_ellipsoid_cost()
    rng = np.random.default_rng(7)
    lam = data_space.from_components(rng.standard_normal(N_DATA))
    epsilon = 1e-2

    smoothed = SmoothedDualMaster(cost, epsilon)

    analytic_grad_vec = smoothed.gradient(lam)
    analytic_grad = data_space.to_components(analytic_grad_vec)
    fd_grad = _fd_gradient(smoothed, lam, data_space)

    np.testing.assert_allclose(
        analytic_grad, fd_grad, rtol=1e-3, atol=1e-7,
        err_msg="Analytic ellipsoid gradient should match finite-difference gradient",
    )


def test_smoothed_lbfgs_converges_ball():
    """SmoothedLBFGSSolver should decrease the objective and return finite f_best."""
    cost, data_space = _make_ball_cost()
    lam0 = data_space.zero
    f_initial = cost(lam0)

    solver = SmoothedLBFGSSolver(cost, epsilon0=1e-2, n_levels=4, tolerance=1e-6)
    result = solver.solve(lam0)

    assert isinstance(result, BundleResult), "Should return BundleResult"
    assert np.isfinite(result.f_best), "f_best should be finite"
    assert result.f_best <= f_initial + 1e-6, (
        f"Solver should decrease f: f_best={result.f_best:.6f}, f_initial={f_initial:.6f}"
    )
    assert result.num_iterations > 0, "Should perform at least one iteration"


def test_smoothed_lbfgs_agrees_with_proximal_bundle():
    """SmoothedLBFGSSolver and ProximalBundleMethod should agree on f_best, rtol=1e-2."""
    cost, data_space = _make_ball_cost()
    lam0 = data_space.zero

    # Run proximal bundle method
    bundle_solver = ProximalBundleMethod(cost, tolerance=1e-4, max_iterations=300)
    bundle_result = bundle_solver.solve(lam0)

    # Run smoothed L-BFGS-B
    lbfgs_solver = SmoothedLBFGSSolver(
        cost, epsilon0=1e-2, n_levels=5, tolerance=1e-6, max_iter_per_level=500
    )
    lbfgs_result = lbfgs_solver.solve(lam0)

    np.testing.assert_allclose(
        lbfgs_result.f_best, bundle_result.f_best, rtol=1e-2,
        err_msg=(
            f"L-BFGS-B f_best={lbfgs_result.f_best:.6f} should agree with "
            f"bundle f_best={bundle_result.f_best:.6f} within rtol=1e-2"
        ),
    )


def test_smoothed_raises_for_unsupported_support():
    """SmoothedDualMaster should raise NotImplementedError for unsupported SupportFunction."""
    rng = np.random.default_rng(42)
    data_space, model_space, prop_space, G, T = _make_operators(rng)

    # Use a supported type for one and unsupported for the other
    model_prior = _UnsupportedSupport(model_space)
    data_error = BallSupportFunction(data_space, data_space.zero, 0.5)

    d_tilde = rng.standard_normal(N_DATA)
    q = rng.standard_normal(N_PROP)

    cost = DualMasterCostFunction(
        data_space, prop_space, model_space,
        G, T, model_prior, data_error, d_tilde, q,
    )

    smoothed = SmoothedDualMaster(cost, 1e-2)
    lam = data_space.from_components(rng.standard_normal(N_DATA))

    with pytest.raises(NotImplementedError):
        smoothed(lam)
