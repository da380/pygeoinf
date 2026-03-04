"""
Tests for ChambollePockSolver and solve_primal_feasibility in convex_optimisation.py.

Covers:
- Feasibility of the solution: ||G*m + v - d_tilde|| < tol
- Primal variable m lies within ball B
- Primal variable v lies within ball V
- Return type is ChambollePockResult with finite attributes
- solve_primal_feasibility values match ProximalBundleMethod (loose tolerance)
"""

import numpy as np
import pytest

from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.convex_analysis import BallSupportFunction
from pygeoinf.backus_gilbert import DualMasterCostFunction
from pygeoinf.convex_optimisation import (
    ChambollePockSolver,
    ChambollePockResult,
    solve_primal_feasibility,
    ProximalBundleMethod,
    solve_support_values,
)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def _make_fixture():
    """Return a small, feasible Chambolle-Pock problem setup."""
    rng = np.random.default_rng(42)
    n_data = 4
    n_model = 5
    n_prop = 2

    data_space = EuclideanSpace(n_data)
    model_space = EuclideanSpace(n_model)
    prop_space = EuclideanSpace(n_prop)

    G_matrix = rng.standard_normal((n_data, n_model))
    T_matrix = rng.standard_normal((n_prop, n_model))
    G = LinearOperator.from_matrix(model_space, data_space, G_matrix)
    T = LinearOperator.from_matrix(model_space, prop_space, T_matrix)

    model_prior = BallSupportFunction(model_space, model_space.zero, 1.0)
    data_error = BallSupportFunction(data_space, data_space.zero, 0.5)

    # Construct a feasible d_tilde = G m_feas + v_feas
    m_feas = rng.standard_normal(n_model) * 0.5   # ||m_feas|| < 1  -> in B
    v_feas = rng.standard_normal(n_data) * 0.2    # ||v_feas|| < 0.5 -> in V
    d_tilde = G_matrix @ m_feas + v_feas

    cp_solver = ChambollePockSolver(
        model_prior,
        data_error,
        G,
        d_tilde,
        max_iterations=3000,
        tolerance=1e-4,
    )

    return dict(
        data_space=data_space,
        model_space=model_space,
        prop_space=prop_space,
        G=G,
        T=T,
        model_prior=model_prior,
        data_error=data_error,
        d_tilde=d_tilde,
        cp_solver=cp_solver,
        G_matrix=G_matrix,
        T_matrix=T_matrix,
    )


def _make_cost(fx, seed=0):
    """Return a DualMasterCostFunction for the fixture."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(fx["prop_space"].dim)
    q_vec = fx["prop_space"].from_components(q)
    cost = DualMasterCostFunction(
        fx["data_space"],
        fx["prop_space"],
        fx["model_space"],
        fx["G"],
        fx["T"],
        fx["model_prior"],
        fx["data_error"],
        fx["d_tilde"],
        q_vec,
    )
    return cost


def _random_direction(fx, seed=1):
    """Return a random direction c = T*q in model space."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(fx["prop_space"].dim)
    q_vec = fx["prop_space"].from_components(q)
    return fx["T"].adjoint(q_vec)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_chambolle_pock_feasibility():
    """||G*m + v - d_tilde|| <= tolerance * 10 after solve."""
    fx = _make_fixture()
    c = _random_direction(fx, seed=1)
    result = fx["cp_solver"].solve(c)

    data_space = fx["data_space"]
    G = fx["G"]

    Gm = G(result.m)
    residual = data_space.subtract(data_space.add(Gm, result.v), fx["d_tilde"])
    feas = data_space.norm(residual)

    assert feas <= fx["cp_solver"]._tolerance * 10, (
        f"Feasibility residual {feas:.2e} exceeds 10x tolerance "
        f"{fx['cp_solver']._tolerance:.2e}"
    )


def test_chambolle_pock_m_in_B():
    """Optimal m lies within the ball B (up to small numerical tolerance)."""
    fx = _make_fixture()
    c = _random_direction(fx, seed=2)
    result = fx["cp_solver"].solve(c)

    model_space = fx["model_space"]
    center = fx["model_prior"]._center
    radius = fx["model_prior"]._radius

    diff = model_space.subtract(result.m, center)
    dist = model_space.norm(diff)
    assert dist <= radius + 1e-3, (
        f"||m - center|| = {dist:.6f}, radius = {radius}"
    )


def test_chambolle_pock_v_in_V():
    """Optimal v lies within the ball V (up to small numerical tolerance)."""
    fx = _make_fixture()
    c = _random_direction(fx, seed=3)
    result = fx["cp_solver"].solve(c)

    data_space = fx["data_space"]
    center = fx["data_error"]._center
    radius = fx["data_error"]._radius

    diff = data_space.subtract(result.v, center)
    dist = data_space.norm(diff)
    assert dist <= radius + 1e-3, (
        f"||v - center|| = {dist:.6f}, radius = {radius}"
    )


def test_chambolle_pock_returns_result():
    """Result is a ChambollePockResult with finite, valid attributes."""
    fx = _make_fixture()
    c = _random_direction(fx, seed=4)
    result = fx["cp_solver"].solve(c)

    assert isinstance(result, ChambollePockResult)
    assert np.isfinite(result.primal_dual_gap)
    assert isinstance(result.converged, bool)
    assert isinstance(result.num_iterations, int)
    assert result.num_iterations > 0
    assert result.m is not None
    assert result.v is not None
    assert result.mu is not None
    assert np.all(np.isfinite(result.m))
    assert np.all(np.isfinite(result.v))
    assert np.all(np.isfinite(result.mu))


def test_solve_primal_feasibility_multiple_directions():
    """solve_primal_feasibility values match ProximalBundleMethod (rtol=0.15)."""
    fx = _make_fixture()
    rng = np.random.default_rng(5)
    cost = _make_cost(fx, seed=99)

    # Build 3 directions in prop_space
    Q = rng.standard_normal((3, fx["prop_space"].dim))
    qs = [fx["prop_space"].from_components(Q[i]) for i in range(3)]

    # Chambolle-Pock values
    cp_values = solve_primal_feasibility(cost, qs, fx["cp_solver"])

    # Bundle method reference
    lambda0 = fx["data_space"].zero
    solver = ProximalBundleMethod(cost, tolerance=1e-4, max_iterations=500)
    bundle_values, _, _ = solve_support_values(cost, qs, solver, lambda0)

    np.testing.assert_allclose(
        cp_values,
        bundle_values,
        rtol=0.15,
        atol=1e-2,
        err_msg=(
            f"Chambolle-Pock values {cp_values} differ from bundle values "
            f"{bundle_values} beyond rtol=0.15, atol=1e-2"
        ),
    )
