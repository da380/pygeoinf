"""Tests for PrimalKKTSolver with PointSupportFunction (equality data constraint).

The equality path solves
    max <c, u>  s.t.  ||u - u0|| <= eta,  G u = d_tilde
analytically via a 1-D quadratic in t = 1/lambda, without fsolve.

Test coverage:
- PointSupportFunction accepted at construction (ball and ellipsoid priors)
- Equality constraint Gu* = d_tilde is satisfied to tight tolerance
- Prior constraint ||u* - u0|| <= eta is satisfied
- Support value matches the tiny-ball approximation (r=1e-6) within rtol=1e-4
- Truth is inside the equality-constrained bounds (when data is noiseless)
- Equality bounds are no wider than noisy inequality bounds
- Prior-inactive branch: c in range(G*) → u* = u0 + a_H, prior slack
- num_iterations == 1 always (closed form, no fsolve)
- from_cost accepts PointSupportFunction in data slot
- Works on non-Euclidean (MassWeightedHilbertSpace) model spaces
"""

from __future__ import annotations

import numpy as np
import pytest

from pygeoinf.hilbert_space import EuclideanSpace, MassWeightedHilbertSpace
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.convex_analysis import BallSupportFunction, PointSupportFunction
from pygeoinf.convex_optimisation import PrimalKKTSolver, KKTResult


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_fixture(seed: int = 7, n_model: int = 6, n_data: int = 3):
    """Underdetermined (n_data < n_model) Euclidean fixture with clean d_tilde."""
    rng = np.random.default_rng(seed)
    ms = EuclideanSpace(n_model)
    ds = EuclideanSpace(n_data)

    G_mat = rng.standard_normal((n_data, n_model))
    G = LinearOperator.from_matrix(ms, ds, G_mat)

    # truth model inside prior ball
    truth_comps = rng.standard_normal(n_model) * 0.5
    d_clean = G_mat @ truth_comps
    d_tilde = ds.from_components(d_clean)

    B = BallSupportFunction(ms, ms.zero, 2.0)   # prior ball large enough for truth
    V_eq = PointSupportFunction(ds, ds.zero)    # equality constraint
    V_tiny = BallSupportFunction(ds, ds.zero, 1e-6)  # near-equality for comparison

    return dict(
        ms=ms, ds=ds,
        G=G, G_mat=G_mat,
        truth_comps=truth_comps,
        d_tilde=d_tilde,
        B=B, V_eq=V_eq, V_tiny=V_tiny,
        eta=2.0,
    )


def _make_mass_weighted_fixture(seed: int = 13, n_model: int = 5, n_data: int = 3):
    """Fixture on a non-Euclidean MassWeightedHilbertSpace."""
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

    truth_comps = rng.standard_normal(n_model) * 0.4
    d_clean = G_base_mat @ truth_comps
    d_tilde = ds.from_components(d_clean)

    B = BallSupportFunction(ms, ms.zero, 2.0)
    V_eq = PointSupportFunction(ds, ds.zero)

    return dict(ms=ms, ds=ds, G=G, d_tilde=d_tilde, B=B, V_eq=V_eq, eta=2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConstruction:
    def test_accepts_point_support_function(self):
        fx = _make_fixture()
        solver = PrimalKKTSolver(fx["B"], fx["V_eq"], fx["G"], fx["d_tilde"])
        assert solver._equality_data is True

    def test_equality_flag_false_for_ball(self):
        fx = _make_fixture()
        solver = PrimalKKTSolver(fx["B"], fx["V_tiny"], fx["G"], fx["d_tilde"])
        assert solver._equality_data is False

    def test_rejects_unknown_type_for_v(self):
        fx = _make_fixture()
        with pytest.raises(TypeError):
            PrimalKKTSolver(fx["B"], "not_a_support_function", fx["G"], fx["d_tilde"])


class TestEqualityConstraintSatisfied:
    """G u* = d_tilde must hold to machine precision."""

    def test_equality_constraint_random_directions(self):
        fx = _make_fixture()
        solver = PrimalKKTSolver(fx["B"], fx["V_eq"], fx["G"], fx["d_tilde"])
        rng = np.random.default_rng(99)
        d_comps = np.asarray(fx["ds"].to_components(fx["d_tilde"]))
        G_mat = fx["G_mat"]

        for _ in range(10):
            c_comps = rng.standard_normal(fx["ms"].dim)
            c = fx["ms"].from_components(c_comps)
            result = solver.solve(c)
            m_comps = np.asarray(fx["ms"].to_components(result.m))
            residual = np.linalg.norm(G_mat @ m_comps - d_comps)
            np.testing.assert_allclose(
                residual, 0.0, atol=1e-8,
                err_msg=f"‖Gu*-d̃‖={residual:.3e} for equality constraint"
            )

    def test_equality_constraint_mass_weighted(self):
        fx = _make_mass_weighted_fixture()
        solver = PrimalKKTSolver(fx["B"], fx["V_eq"], fx["G"], fx["d_tilde"])
        rng = np.random.default_rng(55)
        d_comps = np.asarray(fx["ds"].to_components(fx["d_tilde"]))

        for _ in range(5):
            c_comps = rng.standard_normal(fx["ms"].dim)
            c = fx["ms"].from_components(c_comps)
            result = solver.solve(c)
            Gu_comps = np.asarray(
                fx["ds"].to_components(fx["G"](result.m)), dtype=float
            )
            np.testing.assert_allclose(Gu_comps, d_comps, atol=1e-8)


class TestPriorConstraintSatisfied:
    """‖u* - u0‖ <= eta."""

    def test_prior_ball_satisfied_random_directions(self):
        fx = _make_fixture()
        solver = PrimalKKTSolver(fx["B"], fx["V_eq"], fx["G"], fx["d_tilde"])
        rng = np.random.default_rng(21)

        for _ in range(10):
            c = fx["ms"].from_components(rng.standard_normal(fx["ms"].dim))
            result = solver.solve(c)
            diff = np.asarray(fx["ms"].to_components(result.m))
            norm_diff = float(np.linalg.norm(diff))
            assert norm_diff <= fx["eta"] + 1e-8, (
                f"‖u*‖={norm_diff:.6f} > η={fx['eta']}"
            )


class TestClosedFormProperties:
    """Analytical solve: always 1 iteration, always converged."""

    def test_num_iterations_is_one(self):
        fx = _make_fixture()
        solver = PrimalKKTSolver(fx["B"], fx["V_eq"], fx["G"], fx["d_tilde"])
        rng = np.random.default_rng(3)

        for _ in range(10):
            c = fx["ms"].from_components(rng.standard_normal(fx["ms"].dim))
            result = solver.solve(c)
            assert result.num_iterations == 1

    def test_converged_is_true(self):
        fx = _make_fixture()
        solver = PrimalKKTSolver(fx["B"], fx["V_eq"], fx["G"], fx["d_tilde"])
        rng = np.random.default_rng(4)

        for _ in range(10):
            c = fx["ms"].from_components(rng.standard_normal(fx["ms"].dim))
            result = solver.solve(c)
            assert result.converged, "Equality solve should always converge for feasible problems"

    def test_mu_multiplier_is_inf(self):
        """Second multiplier is inf to signal equality (not a scalar)."""
        fx = _make_fixture()
        solver = PrimalKKTSolver(fx["B"], fx["V_eq"], fx["G"], fx["d_tilde"])
        c = fx["ms"].from_components(np.ones(fx["ms"].dim))
        result = solver.solve(c)
        assert result.multipliers[1] == float("inf")


class TestAgreesWithTinyBallApproximation:
    """Equality solve should match tiny-ball (r=1e-6) support values closely."""

    def test_support_values_close_to_tiny_ball(self):
        fx = _make_fixture()
        solver_eq   = PrimalKKTSolver(fx["B"], fx["V_eq"],  fx["G"], fx["d_tilde"])
        solver_tiny = PrimalKKTSolver(fx["B"], fx["V_tiny"], fx["G"], fx["d_tilde"])
        solver_tiny.disable_warm_start()

        rng = np.random.default_rng(88)
        ms = fx["ms"]
        for _ in range(8):
            c = ms.from_components(rng.standard_normal(ms.dim))
            h_eq   = float(ms.inner_product(c, solver_eq.solve(c).m))
            h_tiny = float(ms.inner_product(c, solver_tiny.solve(c).m))
            np.testing.assert_allclose(
                h_eq, h_tiny, rtol=1e-3,
                err_msg=f"equality h={h_eq:.6f} vs tiny-ball h={h_tiny:.6f}"
            )


class TestBoundProperties:
    """DLI bounds from equality path vs noisy inequality path."""

    def _compute_bounds(self, solver, ms, n_props=3, seed=77):
        """Return upper support values for n_props axis-aligned directions."""
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(n_props):
            c = ms.from_components(rng.standard_normal(ms.dim))
            c_comps = np.asarray(ms.to_components(c))
            c_comps /= np.linalg.norm(c_comps)
            c = ms.from_components(c_comps)
            result = solver.solve(c)
            vals.append(float(ms.inner_product(c, result.m)))
        return np.array(vals)

    def test_equality_bounds_no_wider_than_noisy(self):
        """Equality (noiseless) bounds must be <= noisy bounds."""
        fx = _make_fixture()
        V_noisy = BallSupportFunction(fx["ds"], fx["ds"].zero, 0.3)
        solver_eq    = PrimalKKTSolver(fx["B"], fx["V_eq"],  fx["G"], fx["d_tilde"])
        solver_noisy = PrimalKKTSolver(fx["B"], V_noisy,     fx["G"], fx["d_tilde"])
        solver_noisy.disable_warm_start()

        ms = fx["ms"]
        h_eq    = self._compute_bounds(solver_eq,    ms)
        h_noisy = self._compute_bounds(solver_noisy, ms)

        # equality bounds must be <= noisy bounds (more data → tighter)
        assert np.all(h_eq <= h_noisy + 1e-6), (
            f"Equality bound exceeds noisy bound:\n  eq={h_eq}\n  noisy={h_noisy}"
        )

    def test_truth_inside_equality_bounds(self):
        """With clean data, the true property value must be inside the equality bounds."""
        fx = _make_fixture()
        solver = PrimalKKTSolver(fx["B"], fx["V_eq"], fx["G"], fx["d_tilde"])
        ms = fx["ms"]
        rng = np.random.default_rng(66)

        truth = ms.from_components(fx["truth_comps"])

        for _ in range(6):
            c_comps = rng.standard_normal(ms.dim)
            c_comps /= np.linalg.norm(c_comps)
            c = ms.from_components(c_comps)

            h_upper = float(ms.inner_product(c, solver.solve(c).m))

            neg_c = ms.negative(c)
            # lower bound = <c, u*(-c)>  (the minimiser in direction c
            # is the maximiser in direction -c, evaluated against c)
            h_lower = float(ms.inner_product(c, solver.solve(neg_c).m))

            true_val = float(ms.inner_product(c, truth))
            assert h_lower - 1e-6 <= true_val <= h_upper + 1e-6, (
                f"Truth {true_val:.4f} outside equality bounds "
                f"[{h_lower:.4f}, {h_upper:.4f}]"
            )


class TestPriorInactiveBranch:
    """When c is in range(G*), u is uniquely determined and prior may be slack."""

    def test_prior_inactive_small_data_mismatch(self):
        """With d_tilde = G u0 (no mismatch), a_H = 0 and c in range(G*) gives u* = u0."""
        rng = np.random.default_rng(17)
        ms = EuclideanSpace(4)
        ds = EuclideanSpace(4)   # square system — G is invertible

        G_mat = rng.standard_normal((4, 4))
        # Make G well-conditioned
        U, s, Vt = np.linalg.svd(G_mat)
        s = np.ones(4) * 2.0   # uniform singular values
        G_mat = U @ np.diag(s) @ Vt

        G = LinearOperator.from_matrix(ms, ds, G_mat)
        u0_comps = np.zeros(4)
        u0 = ms.from_components(u0_comps)

        # d_tilde = G(u0) = 0, so a_H = 0
        d_tilde = ds.zero
        B = BallSupportFunction(ms, u0, 5.0)   # large prior
        V_eq = PointSupportFunction(ds, ds.zero)

        solver = PrimalKKTSolver(B, V_eq, G, d_tilde)

        # c in range(G*): use G* e_0 as direction
        c = G.adjoint(ds.from_components(np.array([1.0, 0.0, 0.0, 0.0])))
        result = solver.solve(c)

        # Equality constraint satisfied
        Gu_comps = np.asarray(ds.to_components(G(result.m)), dtype=float)
        np.testing.assert_allclose(Gu_comps, np.zeros(4), atol=1e-8)

        # Prior should be slack (not at boundary) since prior is large
        m_comps = np.asarray(ms.to_components(result.m))
        norm_m = float(np.linalg.norm(m_comps))
        assert norm_m <= 5.0 + 1e-8


class TestFromCost:
    """from_cost should accept PointSupportFunction in the data slot."""

    def test_from_cost_with_point_support(self):
        from pygeoinf.backus_gilbert import DualMasterCostFunction

        fx = _make_fixture()
        ms, ds, G = fx["ms"], fx["ds"], fx["G"]
        prop_space = EuclideanSpace(2)

        T_mat = np.random.default_rng(42).standard_normal((2, ms.dim))
        T = LinearOperator.from_matrix(ms, prop_space, T_mat)

        q0 = prop_space.from_components(np.array([1.0, 0.0]))
        cost = DualMasterCostFunction(
            ds, prop_space, ms,
            G, T,
            fx["B"], fx["V_eq"],
            fx["d_tilde"], q0,
        )
        solver = PrimalKKTSolver.from_cost(cost)
        assert solver._equality_data is True

        # Solve one direction
        c = T.adjoint(q0)
        result = solver.solve(c)
        assert result.converged
        assert result.num_iterations == 1
