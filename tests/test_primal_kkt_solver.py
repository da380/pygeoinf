"""Tests for PrimalKKTSolver and KKTResult in convex_optimisation.py.

Covers:
- KKTResult has the required attributes (m, multipliers, converged, num_iterations)
- Primal variable m lies within the model prior ball
- G @ m satisfies the data constraint (‖Gm - d̃‖ ≤ r)
- Support value ⟨c, m⟩ agrees with ChambollePockSolver (loose tolerance)
- Warm-start multipliers (_lambda_prev, _mu_prev) are updated after a solve
  where both constraints are active
- Ellipsoid prior: (m - u0)^T A (m - u0) ≤ η²
- PrimalKKTSolver and KKTResult are importable from the top-level pygeoinf package
"""

from __future__ import annotations

import numpy as np
import pytest

from pygeoinf.hilbert_space import EuclideanSpace, MassWeightedHilbertSpace
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.convex_analysis import BallSupportFunction, EllipsoidSupportFunction
from pygeoinf.convex_optimisation import (
    PrimalKKTSolver,
    KKTResult,
    ChambollePockSolver,
)

np.random.seed(42)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ball_fixture(seed: int = 42, n_model: int = 5, n_data: int = 3):
    """Small ball-prior + ball-data-error fixture with a feasible d_tilde."""
    rng = np.random.default_rng(seed)
    ms = EuclideanSpace(n_model)
    ds = EuclideanSpace(n_data)

    G_mat = rng.standard_normal((n_data, n_model))
    G = LinearOperator.from_matrix(ms, ds, G_mat)

    # Construct a feasible d_tilde = G m_feas + v_feas
    m_feas = rng.standard_normal(n_model) * 0.4   # ‖m_feas‖ < 1.0 (inside B)
    v_feas = rng.standard_normal(n_data) * 0.15   # ‖v_feas‖ < 0.5 (inside V)
    d_tilde_comps = G_mat @ m_feas + v_feas
    d_tilde = ds.from_components(d_tilde_comps)

    B = BallSupportFunction(ms, ms.zero, 1.0)    # prior ball
    V = BallSupportFunction(ds, ds.zero, 0.5)    # data error ball

    return dict(
        ms=ms, ds=ds,
        G=G, G_mat=G_mat,
        d_tilde=d_tilde, d_tilde_comps=d_tilde_comps,
        B=B, V=V,
        prior_radius=1.0, data_radius=0.5,
    )


def _make_tight_fixture(seed: int = 0, n_model: int = 6, n_data: int = 4):
    """Fixture with tight data ball, forcing both KKT constraints to be active.

    Uses d_tilde = 0 so u=0 is trivially feasible.  The data ball has
    radius 0.1, so the prior-ball support point c/‖c‖ (which has
    ‖G c/‖c‖‖ ~ O(1) for random G) violates the data constraint and
    Branch 2 (2D fsolve) is triggered.
    """
    rng = np.random.default_rng(seed)
    ms = EuclideanSpace(n_model)
    ds = EuclideanSpace(n_data)

    G_mat = rng.standard_normal((n_data, n_model))
    G = LinearOperator.from_matrix(ms, ds, G_mat)

    # Trivially feasible: u=0, v=0, d_tilde=0
    d_tilde = ds.zero
    d_tilde_comps = np.zeros(n_data)

    B = BallSupportFunction(ms, ms.zero, 1.0)
    V = BallSupportFunction(ds, ds.zero, 0.1)   # tight data ball

    # Random unit direction (‖G c_normalized‖ >> 0.1 w.h.p.)
    c_comps = rng.standard_normal(n_model)
    c_comps /= np.linalg.norm(c_comps)
    c = ms.from_components(c_comps)

    return dict(
        ms=ms, ds=ds,
        G=G, G_mat=G_mat,
        d_tilde=d_tilde, d_tilde_comps=d_tilde_comps,
        B=B, V=V,
        prior_radius=1.0, data_radius=0.1,
        c=c, c_comps=c_comps,
    )


def _make_mass_weighted_tight_fixture(
    seed: int = 123, n_model: int = 5, n_data: int = 3
):
    """Tight feasible fixture on a non-Euclidean Hilbert space.

    This reproduces the case where primal components and dual components differ,
    which is the regression surface for the PrimalKKTSolver KKT RHS.
    """
    rng = np.random.default_rng(seed)
    base_ms = EuclideanSpace(n_model)
    ds = EuclideanSpace(n_data)

    mass_diag = 1.0 + rng.uniform(0.2, 2.0, size=n_model)
    mass_op = LinearOperator.self_adjoint_from_matrix(base_ms, np.diag(mass_diag))
    inv_mass_op = LinearOperator.self_adjoint_from_matrix(
        base_ms, np.diag(1.0 / mass_diag)
    )
    ms = MassWeightedHilbertSpace(base_ms, mass_op, inv_mass_op)

    G_base_mat = rng.standard_normal((n_data, n_model))
    G_base = LinearOperator.from_matrix(base_ms, ds, G_base_mat)
    G = LinearOperator.from_formal_adjoint(ms, ds, G_base)

    d_tilde = ds.zero
    d_tilde_comps = np.zeros(n_data)

    B = BallSupportFunction(ms, ms.zero, 1.0)
    V = BallSupportFunction(ds, ds.zero, 0.15)

    for _ in range(20):
        c_comps = rng.standard_normal(n_model)
        c = ms.from_components(c_comps)
        u_ball = B.support_point(c)
        residual = np.linalg.norm(ds.to_components(G(u_ball)) - d_tilde_comps)
        if residual > 1.2 * V._radius:
            return dict(
                ms=ms,
                ds=ds,
                G=G,
                d_tilde=d_tilde,
                B=B,
                V=V,
                c=c,
            )

    raise RuntimeError("Could not generate a both-active mass-weighted fixture")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestImportable:
    def test_importable_from_pygeoinf(self):
        """PrimalKKTSolver and KKTResult must be importable from top-level pygeoinf."""
        from pygeoinf import PrimalKKTSolver as PKS, KKTResult as KR
        assert PKS is not None
        assert KR is not None


class TestKKTResult:
    def test_attributes(self):
        """KKTResult should have m, multipliers, converged, num_iterations."""
        ms = EuclideanSpace(2)
        r = KKTResult(
            m=ms.zero,
            multipliers=(1.0, 0.5),
            converged=True,
            num_iterations=3,
        )
        assert hasattr(r, "m")
        assert hasattr(r, "multipliers")
        assert hasattr(r, "converged")
        assert hasattr(r, "num_iterations")
        assert r.multipliers == (1.0, 0.5)
        assert r.converged is True
        assert r.num_iterations == 3

    def test_m_attribute_accessible(self):
        """result.m must work (used by solve_primal_feasibility duck-typing)."""
        ms = EuclideanSpace(3)
        m = ms.zero
        r = KKTResult(m=m, multipliers=(0.5, 0.0), converged=False, num_iterations=1)
        np.testing.assert_array_equal(r.m, ms.zero)


class TestPrimalBallFeasibility:
    """Solver output must lie within the prior ball."""

    def test_prior_ball_satisfied_random_directions(self):
        fx = _make_ball_fixture()
        solver = PrimalKKTSolver(fx["B"], fx["V"], fx["G"], fx["d_tilde"])
        rng = np.random.default_rng(10)
        for _ in range(5):
            c_comps = rng.standard_normal(fx["ms"].dim)
            c = fx["ms"].from_components(c_comps)
            result = solver.solve(c)
            m_comps = fx["ms"].to_components(result.m)
            norm_m = np.linalg.norm(m_comps)
            np.testing.assert_array_less(
                norm_m, fx["prior_radius"] + 1e-5,
                err_msg=f"‖m*‖={norm_m:.6f} > η={fx['prior_radius']}"
            )

    def test_prior_ball_tight_data(self):
        """Both-active branch: prior ball still respected."""
        fx = _make_tight_fixture()
        solver = PrimalKKTSolver(fx["B"], fx["V"], fx["G"], fx["d_tilde"])
        result = solver.solve(fx["c"])
        m_comps = fx["ms"].to_components(result.m)
        norm_m = np.linalg.norm(m_comps)
        np.testing.assert_array_less(norm_m, fx["prior_radius"] + 1e-5)


class TestDataBallFeasibility:
    """G @ m* must lie within d̃ + data error ball."""

    def test_data_ball_satisfied_random_directions(self):
        fx = _make_ball_fixture()
        solver = PrimalKKTSolver(fx["B"], fx["V"], fx["G"], fx["d_tilde"])
        rng = np.random.default_rng(11)
        for _ in range(5):
            c_comps = rng.standard_normal(fx["ms"].dim)
            c = fx["ms"].from_components(c_comps)
            result = solver.solve(c)
            m_comps = fx["ms"].to_components(result.m)
            residual = fx["G_mat"] @ m_comps - fx["d_tilde_comps"]
            norm_res = np.linalg.norm(residual)
            np.testing.assert_array_less(
                norm_res, fx["data_radius"] + 1e-5,
                err_msg=f"‖Gm*-d̃‖={norm_res:.6f} > r={fx['data_radius']}"
            )

    def test_data_ball_tight(self):
        """Both-active branch: data constraint still respected."""
        fx = _make_tight_fixture()
        solver = PrimalKKTSolver(fx["B"], fx["V"], fx["G"], fx["d_tilde"])
        result = solver.solve(fx["c"])
        m_comps = fx["ms"].to_components(result.m)
        residual = fx["G_mat"] @ m_comps - fx["d_tilde_comps"]
        norm_res = np.linalg.norm(residual)
        np.testing.assert_array_less(norm_res, fx["data_radius"] + 1e-5)


class TestSupportValueAgreement:
    """⟨c, m*⟩ from PrimalKKTSolver must agree with ChambollePockSolver."""

    def test_agreement_small_problem(self):
        fx = _make_ball_fixture(seed=42, n_model=5, n_data=3)
        kkt_solver = PrimalKKTSolver(fx["B"], fx["V"], fx["G"], fx["d_tilde"])
        cp_solver = ChambollePockSolver(
            fx["B"], fx["V"], fx["G"], fx["d_tilde"],
            max_iterations=5000, tolerance=1e-5,
        )
        rng = np.random.default_rng(7)
        for _ in range(3):
            c_comps = rng.standard_normal(fx["ms"].dim)
            c_comps /= np.linalg.norm(c_comps)
            c = fx["ms"].from_components(c_comps)

            kkt_result = kkt_solver.solve(c)
            cp_result = cp_solver.solve(c)

            kkt_val = fx["ms"].inner_product(c, kkt_result.m)
            cp_val = fx["ms"].inner_product(c, cp_result.m)

            np.testing.assert_allclose(
                kkt_val, cp_val, rtol=1e-2, atol=1e-3,
                err_msg=f"KKT val={kkt_val:.6f} vs CP val={cp_val:.6f}"
            )

    def test_tight_data_agreement(self):
        """Both-active branch: support value still matches CP."""
        fx = _make_tight_fixture(seed=5, n_model=4, n_data=3)
        kkt_solver = PrimalKKTSolver(fx["B"], fx["V"], fx["G"], fx["d_tilde"])
        cp_solver = ChambollePockSolver(
            fx["B"], fx["V"], fx["G"], fx["d_tilde"],
            max_iterations=5000, tolerance=1e-5,
        )
        kkt_result = kkt_solver.solve(fx["c"])
        cp_result = cp_solver.solve(fx["c"])

        kkt_val = fx["ms"].inner_product(fx["c"], kkt_result.m)
        cp_val = fx["ms"].inner_product(fx["c"], cp_result.m)

        np.testing.assert_allclose(
            kkt_val, cp_val, rtol=1e-2, atol=1e-3,
            err_msg=f"KKT val={kkt_val:.6f} vs CP val={cp_val:.6f}"
        )

    def test_mass_weighted_ball_agreement(self):
        """Non-Euclidean ball prior should still match ChambollePock."""
        fx = _make_mass_weighted_tight_fixture()
        kkt_solver = PrimalKKTSolver(fx["B"], fx["V"], fx["G"], fx["d_tilde"])
        cp_solver = ChambollePockSolver(
            fx["B"], fx["V"], fx["G"], fx["d_tilde"],
            max_iterations=5000, tolerance=1e-5,
        )

        kkt_result = kkt_solver.solve(fx["c"])
        cp_result = cp_solver.solve(fx["c"])

        kkt_val = fx["ms"].inner_product(fx["c"], kkt_result.m)
        cp_val = fx["ms"].inner_product(fx["c"], cp_result.m)

        np.testing.assert_allclose(
            kkt_val, cp_val, rtol=1e-4, atol=1e-5,
            err_msg=f"KKT val={kkt_val:.6f} vs CP val={cp_val:.6f}"
        )


class TestWarmStart:
    """Warm-start multipliers (_lambda_prev, _mu_prev) should be updated."""

    def test_warm_start_updated_after_both_active(self):
        """When both constraints become active, the warm-start state changes."""
        fx = _make_tight_fixture(seed=77, n_model=6, n_data=4)
        solver = PrimalKKTSolver(fx["B"], fx["V"], fx["G"], fx["d_tilde"])

        lam_init = solver._lambda_prev
        mu_init = solver._mu_prev

        solver.solve(fx["c"])  # both-active case

        # At least one multiplier should have changed
        changed = (
            solver._lambda_prev != lam_init
            or solver._mu_prev != mu_init
        )
        assert changed, (
            f"Warm-start state unchanged: λ={solver._lambda_prev}, μ={solver._mu_prev}"
        )
        assert solver._has_warm_start is True

    def test_initial_warm_start_values(self):
        """Initial warm-start should be positive λ, zero μ (unconstrained start)."""
        fx = _make_ball_fixture()
        solver = PrimalKKTSolver(fx["B"], fx["V"], fx["G"], fx["d_tilde"])
        assert solver._lambda_prev > 0, "_lambda_prev should be positive at init"
        assert solver._mu_prev == 0.0, "_mu_prev should be 0 at init"
        assert solver._has_warm_start is False


class TestEllipsoidPrior:
    """PrimalKKTSolver works when B is an EllipsoidSupportFunction."""

    def test_ellipsoid_prior_feasible(self):
        """(m* - u0)^T A (m* - u0) ≤ η²."""
        rng = np.random.default_rng(0)
        n = 4
        ms = EuclideanSpace(n)
        ds = EuclideanSpace(3)

        G_mat = rng.standard_normal((3, n))
        G = LinearOperator.from_matrix(ms, ds, G_mat)

        # Diagonal ellipsoid: A = diag(2, 1, 3, 0.5)
        diag_A = np.array([2.0, 1.0, 3.0, 0.5])
        A_mat = np.diag(diag_A)
        A_inv_mat = np.diag(1.0 / diag_A)
        A_inv_sqrt_mat = np.diag(1.0 / np.sqrt(diag_A))

        A_op = LinearOperator.from_matrix(ms, ms, A_mat)
        A_inv_op = LinearOperator.from_matrix(ms, ms, A_inv_mat)
        A_inv_sqrt_op = LinearOperator.from_matrix(ms, ms, A_inv_sqrt_mat)

        u0 = ms.zero
        eta = 1.0
        B = EllipsoidSupportFunction(ms, u0, eta, A_op, A_inv_op, A_inv_sqrt_op)

        d_comps = rng.standard_normal(3) * 0.3
        d_tilde = ds.from_components(d_comps)
        V = BallSupportFunction(ds, ds.zero, 0.5)

        solver = PrimalKKTSolver(B, V, G, d_tilde)

        for seed in range(4):
            rng2 = np.random.default_rng(100 + seed)
            c_comps = rng2.standard_normal(n)
            c = ms.from_components(c_comps)
            result = solver.solve(c)
            m_comps = ms.to_components(result.m)

            # Check ellipsoid constraint: m^T A m ≤ η² (u0=0)
            ellipsoid_val = m_comps @ A_mat @ m_comps
            np.testing.assert_array_less(
                ellipsoid_val, eta ** 2 + 1e-5,
                err_msg=f"Ellipsoid constraint violated: {ellipsoid_val:.6f} > {eta**2}"
            )

    def test_ellipsoid_support_value_agreement(self):
        """Ellipsoid-prior support value should match ChambollePockSolver (ball case only)."""
        # Use identity matrix for ellipsoid → reduces to ball; CP handles both cases
        rng = np.random.default_rng(22)
        n = 4
        ms = EuclideanSpace(n)
        ds = EuclideanSpace(3)

        G_mat = rng.standard_normal((3, n))
        G = LinearOperator.from_matrix(ms, ds, G_mat)

        # Identity ellipsoid = ball
        A_mat = np.eye(n)
        A_op = LinearOperator.from_matrix(ms, ms, A_mat)
        A_inv_op = LinearOperator.from_matrix(ms, ms, A_mat)
        A_inv_sqrt_op = LinearOperator.from_matrix(ms, ms, A_mat)

        u0 = ms.zero
        eta = 1.0
        B_ellip = EllipsoidSupportFunction(ms, u0, eta, A_op, A_inv_op, A_inv_sqrt_op)
        B_ball = BallSupportFunction(ms, ms.zero, eta)
        V = BallSupportFunction(ds, ds.zero, 0.5)

        m_feas = rng.standard_normal(n) * 0.3
        d_tilde_comps = G_mat @ m_feas + rng.standard_normal(3) * 0.1
        d_tilde = ds.from_components(d_tilde_comps)

        kkt_ellip = PrimalKKTSolver(B_ellip, V, G, d_tilde)
        kkt_ball = PrimalKKTSolver(B_ball, V, G, d_tilde)

        c_comps = rng.standard_normal(n)
        c_comps /= np.linalg.norm(c_comps)
        c = ms.from_components(c_comps)

        val_ellip = ms.inner_product(c, kkt_ellip.solve(c).m)
        val_ball = ms.inner_product(c, kkt_ball.solve(c).m)

        np.testing.assert_allclose(
            val_ellip, val_ball, rtol=1e-6, atol=1e-6,
            err_msg="Identity ellipsoid should give same result as ball"
        )
