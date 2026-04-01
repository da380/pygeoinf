"""
Tests for ProximalBundleMethod in convex_optimisation.py.

Covers:
- Smooth quadratic minimisation (1-D)
- Non-smooth absolute value minimisation (1-D)
- Gap certificate at convergence
- Serious-step counting
- Integration with DualMasterCostFunction
"""

import numpy as np
import pytest

from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.convex_analysis import BallSupportFunction
from pygeoinf.backus_gilbert import DualMasterCostFunction
from pygeoinf.nonlinear_forms import NonLinearForm
from pygeoinf.convex_optimisation import ProximalBundleMethod, BundleResult

np.random.seed(42)

# ---------------------------------------------------------------------------
# Helper: small DualMasterCostFunction fixture
# ---------------------------------------------------------------------------

def _make_dual_master_cost():
    """Return a small DualMasterCostFunction for tests."""
    rng = np.random.default_rng(42)
    n_data = 4
    n_model = 5
    n_prop = 1

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
    q = rng.standard_normal(n_prop)

    cost = DualMasterCostFunction(
        data_space, prop_space, model_space,
        G, T, model_prior, data_error, d_tilde, q
    )
    return cost, data_space


# ---------------------------------------------------------------------------
# Simple oracle helpers
# ---------------------------------------------------------------------------

def _quadratic_oracle():
    """f(λ) = λ² + 2λ, minimiser at λ* = -1, f* = -1."""
    domain = EuclideanSpace(1)
    def f(x):
        return float(x[0] ** 2 + 2.0 * x[0])
    def g(x):
        return np.array([2.0 * x[0] + 2.0])
    return NonLinearForm(domain, f, subgradient=g), domain


def _abs_oracle():
    """f(λ) = |λ - 0.5|, minimiser at λ* = 0.5, f* = 0."""
    domain = EuclideanSpace(1)
    def f(x):
        return float(abs(x[0] - 0.5))
    def g(x):
        v = x[0] - 0.5
        return np.array([1.0 if v > 0 else (-1.0 if v < 0 else 0.0)])
    return NonLinearForm(domain, f, subgradient=g), domain


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_proximal_bundle_quadratic_1d():
    """Bundle method should minimise f(λ)=λ²+2λ to within atol=1e-3."""
    oracle, domain = _quadratic_oracle()
    solver = ProximalBundleMethod(
        oracle,
        tolerance=1e-5,
        max_iterations=300,
    )
    x0 = domain.from_components(np.array([2.0]))
    result = solver.solve(x0)

    np.testing.assert_allclose(
        domain.to_components(result.x_best),
        np.array([-1.0]),
        atol=1e-3,
        err_msg="Minimiser should be at λ* = -1",
    )
    np.testing.assert_allclose(
        result.f_best,
        -1.0,
        atol=1e-3,
        err_msg="Minimum value should be f* = -1",
    )


def test_proximal_bundle_convex_nonsmooth_1d():
    """Bundle method should minimise f(λ)=|λ-0.5| to within atol=1e-3."""
    oracle, domain = _abs_oracle()
    solver = ProximalBundleMethod(
        oracle,
        tolerance=1e-5,
        max_iterations=300,
    )
    x0 = domain.from_components(np.array([3.0]))
    result = solver.solve(x0)

    np.testing.assert_allclose(
        domain.to_components(result.x_best),
        np.array([0.5]),
        atol=1e-3,
        err_msg="Minimiser of |λ-0.5| should be at λ* = 0.5",
    )


def test_proximal_bundle_gap_certificate():
    """At convergence the optimality gap should satisfy gap <= tolerance."""
    oracle, domain = _quadratic_oracle()
    tol = 1e-4
    solver = ProximalBundleMethod(
        oracle,
        tolerance=tol,
        max_iterations=300,
    )
    x0 = domain.from_components(np.array([2.0]))
    result = solver.solve(x0)

    assert result.converged, "Should have converged on smooth quadratic"
    assert result.gap <= tol * 10, (
        f"Gap {result.gap} should be at most 10 * tolerance = {10 * tol}"
    )


def test_proximal_bundle_serious_steps():
    """The smooth quadratic should require at least one serious step."""
    oracle, domain = _quadratic_oracle()
    solver = ProximalBundleMethod(
        oracle,
        tolerance=1e-4,
        max_iterations=300,
    )
    x0 = domain.from_components(np.array([2.0]))
    result = solver.solve(x0)

    assert result.num_serious_steps > 0, (
        "Smooth quadratic should trigger at least one serious step"
    )


def test_proximal_bundle_dual_master():
    """ProximalBundleMethod should decrease f when applied to DualMasterCostFunction."""
    cost, data_space = _make_dual_master_cost()
    x0 = data_space.zero
    f_initial = cost(x0)

    solver = ProximalBundleMethod(
        cost,
        tolerance=1e-4,
        max_iterations=200,
    )
    result = solver.solve(x0)

    assert isinstance(result, BundleResult), "Should return BundleResult"
    assert result.f_best <= f_initial + 1e-10, (
        f"Bundle should not increase f: f_best={result.f_best}, f_initial={f_initial}"
    )


def test_proximal_bundle_dual_master_gap():
    """ProximalBundleMethod on DualMasterCostFunction: gap <= 10 * tolerance."""
    cost, data_space = _make_dual_master_cost()
    x0 = data_space.zero
    tol = 1e-4

    solver = ProximalBundleMethod(
        cost,
        tolerance=tol,
        max_iterations=200,
    )
    result = solver.solve(x0)

    assert result.gap <= tol * 10, (
        f"Gap {result.gap:.2e} exceeds 10 * tolerance = {10 * tol:.2e}"
    )
