"""Tests for PolyhedralApproximation and DirectionSampler.

Covers:
- DirectionSampler: shape, unit norms, simplex non-degeneracy.
- PolyhedralApproximation: initialization, constraint counts,
  deduplication, as_polyhedral_set(), containment of true property value.
"""

import numpy as np
import pytest

from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.convex_analysis import BallSupportFunction
from pygeoinf.backus_gilbert import DualMasterCostFunction
from pygeoinf.convex_optimisation import SubgradientDescent
from pygeoinf.subsets import PolyhedralSet
from pygeoinf.polyhedral_approximation import DirectionSampler, PolyhedralApproximation


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _make_problem(seed=42):
    """Build a small DLI problem in R^3 model / R^2 data / R^2 property space.

    Returns:
        (cost, solver, property_space, true_prop_vec)
        where true_prop_vec is T @ m_truth and is guaranteed feasible.
    """
    rng = np.random.default_rng(seed)

    n_model, n_data, n_prop = 4, 3, 2
    model_space = EuclideanSpace(n_model)
    data_space = EuclideanSpace(n_data)
    prop_space = EuclideanSpace(n_prop)

    G_mat = rng.standard_normal((n_data, n_model))
    T_mat = rng.standard_normal((n_prop, n_model))
    G = LinearOperator.from_matrix(model_space, data_space, G_mat)
    T = LinearOperator.from_matrix(model_space, prop_space, T_mat)

    # Truth: unit model, small noise on data
    m_truth = rng.standard_normal(n_model)
    m_truth /= np.linalg.norm(m_truth) * 2.0   # ||m*|| = 0.5 < R_model
    d_noisefree = G_mat @ m_truth
    noise = rng.standard_normal(n_data) * 0.05   # small noise
    d_obs = d_noisefree + noise

    R_model = 1.0   # prior ball radius — strictly larger than ||m*||
    R_data = 0.3    # data ball radius — strictly larger than ||noise||
    model_ball = BallSupportFunction(model_space, model_space.zero, R_model)
    data_ball = BallSupportFunction(data_space, data_space.zero, R_data)

    true_prop = T_mat @ m_truth

    q0 = prop_space.from_components(np.array([1.0, 0.0]))
    cost = DualMasterCostFunction(
        data_space, prop_space, model_space,
        G, T, model_ball, data_ball,
        d_obs, q0,
    )
    solver = SubgradientDescent(cost, step_size=1e-2, max_iterations=600)

    return cost, solver, prop_space, true_prop


# ===========================================================================
# DirectionSampler
# ===========================================================================

class TestDirectionSamplerBox:

    def test_shape(self):
        D = DirectionSampler.box(4)
        assert D.shape == (8, 4)

    def test_unit_norms(self):
        D = DirectionSampler.box(5)
        norms = np.linalg.norm(D, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_contains_pos_and_neg_identity(self):
        D = DirectionSampler.box(3)
        for sign in (1, -1):
            for i in range(3):
                e = sign * np.eye(3)[i]
                assert any(np.allclose(row, e) for row in D)


class TestDirectionSamplerSimplex:

    def test_shape(self):
        D = DirectionSampler.simplex(4, random_state=0)
        assert D.shape == (5, 4)

    def test_unit_norms(self):
        D = DirectionSampler.simplex(5, random_state=0)
        norms = np.linalg.norm(D, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_non_degenerate(self):
        """First n rows must be linearly independent."""
        n = 4
        D = DirectionSampler.simplex(n, random_state=7)
        vol = np.abs(np.linalg.det(D[:n]))
        assert vol > 1e-3


class TestDirectionSamplerRandomUniform:

    def test_shape(self):
        D = DirectionSampler.random_uniform(4, 20, random_state=0)
        assert D.shape == (20, 4)

    def test_unit_norms(self):
        D = DirectionSampler.random_uniform(3, 50, random_state=1)
        norms = np.linalg.norm(D, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_reproducible_with_seed(self):
        D1 = DirectionSampler.random_uniform(3, 10, random_state=99)
        D2 = DirectionSampler.random_uniform(3, 10, random_state=99)
        np.testing.assert_array_equal(D1, D2)


class TestDirectionSamplerGet:

    def test_box(self):
        D = DirectionSampler.get("box", 3)
        assert D.shape == (6, 3)

    def test_simplex(self):
        D = DirectionSampler.get("simplex", 3, random_state=0)
        assert D.shape == (4, 3)

    def test_random_uniform(self):
        D = DirectionSampler.get("random_uniform", 3, n_new=15, random_state=0)
        assert D.shape == (15, 3)

    def test_random_uniform_requires_n_new(self):
        with pytest.raises(ValueError, match="n_new is required"):
            DirectionSampler.get("random_uniform", 3)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            DirectionSampler.get("bogus", 3)


# ===========================================================================
# PolyhedralApproximation
# ===========================================================================

class TestPolyhedralApproximationInit:

    def test_initialize_box_constraint_count(self):
        cost, solver, prop_space, _ = _make_problem()
        approx = PolyhedralApproximation(prop_space, cost, solver)
        approx.initialize("box")
        # box yields 2 * n_prop directions
        assert approx.n_constraints == 2 * prop_space.dim

    def test_initialize_simplex_constraint_count(self):
        cost, solver, prop_space, _ = _make_problem()
        approx = PolyhedralApproximation(prop_space, cost, solver)
        approx.initialize("simplex")
        assert approx.n_constraints == prop_space.dim + 1

    def test_initialize_random_uniform_raises(self):
        cost, solver, prop_space, _ = _make_problem()
        approx = PolyhedralApproximation(prop_space, cost, solver)
        with pytest.raises(ValueError, match="refine"):
            approx.initialize("random_uniform")

    def test_empty_raises_on_as_polyhedral_set(self):
        cost, solver, prop_space, _ = _make_problem()
        approx = PolyhedralApproximation(prop_space, cost, solver)
        with pytest.raises(RuntimeError, match="No directions"):
            approx.as_polyhedral_set()


class TestPolyhedralApproximationDeduplication:

    def test_duplicate_directions_not_resolved_twice(self):
        cost, solver, prop_space, _ = _make_problem()
        approx = PolyhedralApproximation(prop_space, cost, solver)
        dirs = DirectionSampler.box(prop_space.dim)

        approx.add_directions(dirs)
        n_after_first = approx.n_constraints

        # Adding the same directions again must not grow the cache
        approx.add_directions(dirs)
        assert approx.n_constraints == n_after_first

    def test_unnormalized_same_direction_deduplicated(self):
        cost, solver, prop_space, _ = _make_problem()
        approx = PolyhedralApproximation(prop_space, cost, solver)

        q = np.array([1.0, 0.0])
        approx.add_directions(q.reshape(1, -1))
        n1 = approx.n_constraints

        # Same direction at different scale must be a duplicate
        approx.add_directions((3.7 * q).reshape(1, -1))
        assert approx.n_constraints == n1

    def test_zero_vector_raises(self):
        cost, solver, prop_space, _ = _make_problem()
        approx = PolyhedralApproximation(prop_space, cost, solver)
        with pytest.raises(ValueError, match="zero vector"):
            approx.add_directions(np.zeros((1, prop_space.dim)))


class TestPolyhedralApproximationRefine:

    def test_refine_adds_constraints(self):
        cost, solver, prop_space, _ = _make_problem()
        approx = PolyhedralApproximation(prop_space, cost, solver)
        approx.initialize("box")
        n_box = approx.n_constraints

        approx.refine("random_uniform", n_new=10, random_state=0)
        assert approx.n_constraints >= n_box

    def test_cumulative_refine(self):
        cost, solver, prop_space, _ = _make_problem()
        approx = PolyhedralApproximation(prop_space, cost, solver)
        approx.initialize("box")
        approx.refine("random_uniform", n_new=8, random_state=1)
        approx.refine("simplex", random_state=2)
        # Shouldn't crash; constraint count should grow monotonically
        assert approx.n_constraints >= 2 * prop_space.dim


class TestPolyhedralApproximationPolyhedralSet:

    def test_returns_polyhedral_set(self):
        cost, solver, prop_space, _ = _make_problem()
        approx = PolyhedralApproximation(prop_space, cost, solver)
        approx.initialize("box")
        poly = approx.as_polyhedral_set()
        assert isinstance(poly, PolyhedralSet)

    def test_polyhedral_set_correct_domain(self):
        cost, solver, prop_space, _ = _make_problem()
        approx = PolyhedralApproximation(prop_space, cost, solver)
        approx.initialize("box")
        poly = approx.as_polyhedral_set()
        assert poly.domain is prop_space

    def test_polyhedral_set_half_space_count(self):
        cost, solver, prop_space, _ = _make_problem()
        approx = PolyhedralApproximation(prop_space, cost, solver)
        approx.initialize("box")
        n_dirs = approx.n_constraints
        poly = approx.as_polyhedral_set()
        assert len(poly.half_spaces) == n_dirs


class TestPolyhedralApproximationCorrectness:
    """The true property value must lie inside the polyhedral approximation."""

    def test_true_property_inside_box_approximation(self):
        """Box bounds must enclose the true property value.

        The DLI guarantees that h_U(q) >= <q, p*> for any feasible p*,
        so the box (4 half-spaces in 2D) must contain p*.
        """
        cost, solver, prop_space, true_prop = _make_problem(seed=7)
        approx = PolyhedralApproximation(prop_space, cost, solver)
        approx.initialize("box")
        poly = approx.as_polyhedral_set()

        p = prop_space.from_components(true_prop)
        assert poly.is_element(p, rtol=1e-2), (
            f"True property {true_prop} not inside box approximation. "
            "DLI feasibility guarantee violated."
        )

    def test_true_property_inside_refined_approximation(self):
        """Refinement must not remove the true property from the feasible set."""
        cost, solver, prop_space, true_prop = _make_problem(seed=13)
        approx = PolyhedralApproximation(prop_space, cost, solver)
        approx.initialize("box")
        approx.refine("random_uniform", n_new=20, random_state=5)

        poly = approx.as_polyhedral_set()
        p = prop_space.from_components(true_prop)
        assert poly.is_element(p, rtol=1e-2), (
            f"True property {true_prop} not inside refined approximation."
        )

    def test_bounds_are_finite(self):
        """All cached support values must be finite for a bounded prior."""
        cost, solver, prop_space, _ = _make_problem()
        approx = PolyhedralApproximation(prop_space, cost, solver)
        approx.initialize("box")
        approx.refine("random_uniform", n_new=5, random_state=0)

        for q_key, h_val in approx._cache.items():
            assert np.isfinite(h_val), f"Non-finite bound for direction {q_key}"
