"""
Tests for Phase 1 support-function constructors:
  - CallableSupportFunction
  - PointSupportFunction
  - SupportFunction.callable(...)
  - SupportFunction.point(...)

TDD: tests were written before the implementation.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.convex_analysis import (
    SupportFunction,
    BallSupportFunction,
    EllipsoidSupportFunction,
    CallableSupportFunction,
    PointSupportFunction,
)

np.random.seed(0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def space_2d():
    return EuclideanSpace(2)


@pytest.fixture
def space_3d():
    return EuclideanSpace(3)


# ---------------------------------------------------------------------------
# CallableSupportFunction – value evaluation
# ---------------------------------------------------------------------------

class TestCallableSupportFunction:
    """Tests for CallableSupportFunction."""

    def test_evaluate_matches_callable(self, space_2d):
        """h(q) returns the value produced by the wrapped callable."""
        # Use L1 norm as a support function (unit L-inf ball in R^2)
        def fn(q):
            return float(np.sum(np.abs(q)))

        h = CallableSupportFunction(space_2d, fn)
        q = np.array([1.0, -2.0])
        assert_allclose(h(q), 3.0, rtol=1e-12)

    def test_evaluate_another_direction(self, space_2d):
        """h(q) is correct for several query directions."""
        def fn(q):
            return float(np.sum(np.abs(q)))

        h = CallableSupportFunction(space_2d, fn)
        for q in [np.array([0.0, 0.0]), np.array([3.0, 4.0]), np.array([-1.0, -1.0])]:
            assert_allclose(h(q), float(np.sum(np.abs(q))), rtol=1e-12)

    def test_primal_domain_stored(self, space_2d):
        """primal_domain is the space passed at construction."""
        def fn(q):
            return 0.0

        h = CallableSupportFunction(space_2d, fn)
        assert h.primal_domain is space_2d

    # ------------------------------------------------------------------
    # Without support_point callable: support_point returns None,
    # subgradient raises NotImplementedError
    # ------------------------------------------------------------------

    def test_support_point_none_when_no_fn(self, space_2d):
        """support_point returns None when no callable is provided."""
        def fn(q):
            return float(np.linalg.norm(q))

        h = CallableSupportFunction(space_2d, fn)
        assert h.support_point(np.array([1.0, 0.0])) is None

    def test_subgradient_raises_when_no_fn(self, space_2d):
        """subgradient raises NotImplementedError when support_point is unavailable."""
        def fn(q):
            return float(np.linalg.norm(q))

        h = CallableSupportFunction(space_2d, fn)
        with pytest.raises(NotImplementedError):
            h.subgradient(np.array([1.0, 0.0]))

    # ------------------------------------------------------------------
    # With support_point callable: support_point delegates to it,
    # and subgradient returns it
    # ------------------------------------------------------------------

    def test_support_point_delegates_to_callable(self, space_2d):
        """support_point calls the user-supplied callback and returns its result."""
        # L2-ball support: h(q) = ||q||, x*(q) = q / ||q||
        def h_fn(q):
            return float(np.linalg.norm(q))

        def sp_fn(q):
            n = np.linalg.norm(q)
            return q / n if n > 1e-14 else np.zeros_like(q)

        h = CallableSupportFunction(space_2d, h_fn, support_point_fn=sp_fn)
        q = np.array([3.0, 4.0])
        expected = q / np.linalg.norm(q)
        sp = h.support_point(q)
        assert sp is not None
        assert_allclose(sp, expected, rtol=1e-12)

    def test_subgradient_uses_support_point_callable(self, space_2d):
        """subgradient delegates to support_point when a callable is provided."""
        def h_fn(q):
            return float(np.linalg.norm(q))

        def sp_fn(q):
            n = np.linalg.norm(q)
            return q / n if n > 1e-14 else np.zeros_like(q)

        h = CallableSupportFunction(space_2d, h_fn, support_point_fn=sp_fn)
        q = np.array([1.0, 0.0])
        assert_allclose(h.subgradient(q), np.array([1.0, 0.0]), rtol=1e-12)

    def test_is_instance_of_support_function(self, space_2d):
        """CallableSupportFunction is a subclass of SupportFunction."""
        h = CallableSupportFunction(space_2d, lambda q: 0.0)
        assert isinstance(h, SupportFunction)


# ---------------------------------------------------------------------------
# PointSupportFunction – singleton-set support h(q) = <q, p>
# ---------------------------------------------------------------------------

class TestPointSupportFunction:
    """Tests for PointSupportFunction."""

    def test_evaluate_inner_product(self, space_2d):
        """h(q) = <q, p> for a fixed point p."""
        p = np.array([2.0, 3.0])
        h = PointSupportFunction(space_2d, p)
        q = np.array([1.0, -1.0])
        expected = float(np.dot(q, p))
        assert_allclose(h(q), expected, rtol=1e-12)

    def test_evaluate_zero_query(self, space_2d):
        """h(0) = 0 for any point p."""
        p = np.array([5.0, -7.0])
        h = PointSupportFunction(space_2d, p)
        assert_allclose(h(space_2d.zero), 0.0, atol=1e-14)

    def test_evaluate_3d(self, space_3d):
        """Correct value in higher dimension."""
        p = np.array([1.0, 2.0, 3.0])
        h = PointSupportFunction(space_3d, p)
        q = np.array([4.0, 5.0, 6.0])
        assert_allclose(h(q), float(np.dot(q, p)), rtol=1e-12)

    def test_primal_domain_stored(self, space_2d):
        """primal_domain matches the space passed at construction."""
        p = np.array([1.0, 0.0])
        h = PointSupportFunction(space_2d, p)
        assert h.primal_domain is space_2d

    def test_support_point_is_fixed_point(self, space_2d):
        """support_point always returns p, regardless of q."""
        p = np.array([2.0, -1.0])
        h = PointSupportFunction(space_2d, p)
        for q in [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([3.0, 4.0])]:
            assert_allclose(h.support_point(q), p, rtol=1e-12)

    def test_subgradient_is_fixed_point(self, space_2d):
        """subgradient returns p at every query direction."""
        p = np.array([2.0, -1.0])
        h = PointSupportFunction(space_2d, p)
        q = np.array([1.5, -2.5])
        assert_allclose(h.subgradient(q), p, rtol=1e-12)

    def test_is_instance_of_support_function(self, space_2d):
        """PointSupportFunction is a subclass of SupportFunction."""
        h = PointSupportFunction(space_2d, np.array([0.0, 0.0]))
        assert isinstance(h, SupportFunction)

    def test_evaluate_linearity_in_q(self, space_2d):
        """h(alpha * q) = alpha * h(q) for alpha > 0 (positive homogeneity)."""
        p = np.array([1.0, 2.0])
        h = PointSupportFunction(space_2d, p)
        q = np.array([3.0, -1.0])
        alpha = 2.5
        assert_allclose(h(alpha * q), alpha * h(q), rtol=1e-12)


# ---------------------------------------------------------------------------
# Convenience constructors on SupportFunction
# ---------------------------------------------------------------------------

class TestSupportFunctionConvenienceConstructors:
    """Tests for SupportFunction.callable() and SupportFunction.point()."""

    def test_callable_returns_callable_support_function(self, space_2d):
        """SupportFunction.callable(...) returns a CallableSupportFunction."""
        def fn(q):
            return float(np.linalg.norm(q))

        h = SupportFunction.callable(space_2d, fn)
        assert isinstance(h, CallableSupportFunction)

    def test_callable_value_matches_fn(self, space_2d):
        """SupportFunction.callable: h(q) agrees with the supplied function."""
        def fn(q):
            return float(np.max(np.abs(q)))

        h = SupportFunction.callable(space_2d, fn)
        q = np.array([3.0, -4.0])
        assert_allclose(h(q), 4.0, rtol=1e-12)

    def test_callable_with_support_point(self, space_2d):
        """SupportFunction.callable: optional support_point callable is wired up."""
        def fn(q):
            return float(np.linalg.norm(q))

        def sp_fn(q):
            n = np.linalg.norm(q)
            return q / n if n > 1e-14 else np.zeros_like(q)

        h = SupportFunction.callable(space_2d, fn, support_point=sp_fn)
        q = np.array([0.0, 1.0])
        sp = h.support_point(q)
        assert sp is not None
        assert_allclose(sp, np.array([0.0, 1.0]), rtol=1e-12)

    def test_callable_no_support_point_returns_none(self, space_2d):
        """SupportFunction.callable without support_point callback: returns None."""
        h = SupportFunction.callable(space_2d, lambda q: 0.0)
        assert h.support_point(np.array([1.0, 0.0])) is None

    def test_point_returns_point_support_function(self, space_2d):
        """SupportFunction.point(...) returns a PointSupportFunction."""
        p = np.array([1.0, 2.0])
        h = SupportFunction.point(space_2d, p)
        assert isinstance(h, PointSupportFunction)

    def test_point_value_is_inner_product(self, space_2d):
        """SupportFunction.point: h(q) = <q, p>."""
        p = np.array([1.0, 2.0])
        h = SupportFunction.point(space_2d, p)
        q = np.array([3.0, -1.0])
        assert_allclose(h(q), float(np.dot(q, p)), rtol=1e-12)

    def test_point_subgradient_is_p(self, space_2d):
        """SupportFunction.point: subgradient returns the fixed point p."""
        p = np.array([1.0, 2.0])
        h = SupportFunction.point(space_2d, p)
        q = np.array([3.0, -1.0])
        assert_allclose(h.subgradient(q), p, rtol=1e-12)

    def test_callable_subgradient_matches_support_point(self, space_2d):
        """SupportFunction.callable: subgradient(q) equals support_point(q)."""
        def fn(q):
            return float(np.linalg.norm(q))

        def sp_fn(q):
            n = np.linalg.norm(q)
            return q / n if n > 1e-14 else np.zeros_like(q)

        h = SupportFunction.callable(space_2d, fn, support_point=sp_fn)
        q = np.array([3.0, 4.0])
        sp = h.support_point(q)
        assert sp is not None
        sg = h.subgradient(q)
        assert_allclose(sg, sp, rtol=1e-12)


# ---------------------------------------------------------------------------
# Phase 2: value_and_support_point fused API
# ---------------------------------------------------------------------------


class TestValueAndSupportPoint:
    """Tests for SupportFunction.value_and_support_point(q).

    TDD: written before the fused implementation. All tests below must be
    green after Phase 2 implementation.
    """

    # ------------------------------------------------------------------
    # Default implementation (SupportFunction base class)
    # ------------------------------------------------------------------

    def test_default_callable_value_consistent(self, space_2d):
        """Default: value equals h(q) for a CallableSupportFunction."""
        h = CallableSupportFunction(
            space_2d,
            lambda q: float(np.linalg.norm(q)),
            support_point_fn=lambda q: q / np.linalg.norm(q),
        )
        q = np.array([3.0, 4.0])
        val, _ = h.value_and_support_point(q)
        assert_allclose(val, h(q), rtol=1e-12)

    def test_default_callable_point_consistent(self, space_2d):
        """Default: point equals support_point(q) for a CallableSupportFunction."""
        def sp_fn(q):
            return q / np.linalg.norm(q)

        h = CallableSupportFunction(
            space_2d,
            lambda q: float(np.linalg.norm(q)),
            support_point_fn=sp_fn,
        )
        q = np.array([3.0, 4.0])
        _, pt = h.value_and_support_point(q)
        assert pt is not None
        assert_allclose(pt, h.support_point(q), rtol=1e-12)

    def test_default_no_support_point_returns_none(self, space_2d):
        """Default: when support_point returns None, second element is None."""
        h = CallableSupportFunction(
            space_2d, lambda q: float(np.linalg.norm(q))  # no support_point_fn
        )
        q = np.array([1.0, 2.0])
        val, pt = h.value_and_support_point(q)
        assert_allclose(val, h(q), rtol=1e-12)
        assert pt is None

    # ------------------------------------------------------------------
    # BallSupportFunction overridden implementation
    # ------------------------------------------------------------------

    def test_ball_value_matches_standalone(self, space_2d):
        """BallSupportFunction.value_and_support_point: value equals h(q)."""
        center = np.array([1.0, 2.0])
        h = BallSupportFunction(space_2d, center, 3.0)
        q = np.array([4.0, -3.0])
        val, _ = h.value_and_support_point(q)
        assert_allclose(val, h(q), rtol=1e-12)

    def test_ball_point_matches_standalone(self, space_2d):
        """BallSupportFunction.value_and_support_point: point equals support_point(q)."""
        center = np.array([1.0, 2.0])
        h = BallSupportFunction(space_2d, center, 3.0)
        q = np.array([4.0, -3.0])
        _, pt = h.value_and_support_point(q)
        expected = h.support_point(q)
        assert pt is not None
        assert expected is not None
        assert_allclose(pt, expected, rtol=1e-12)

    def test_ball_value_formula_multiple_directions(self, space_2d):
        """BallSupportFunction: value = <q,c> + r||q|| for several directions."""
        center = np.array([0.5, -0.5])
        r = 2.0
        h = BallSupportFunction(space_2d, center, r)
        rng = np.random.default_rng(42)
        for _ in range(5):
            q = rng.standard_normal(2)
            val, _ = h.value_and_support_point(q)
            expected = float(np.dot(q, center)) + r * float(np.linalg.norm(q))
            assert_allclose(val, expected, rtol=1e-12)

    def test_ball_near_zero_q_value_correct(self, space_2d):
        """BallSupportFunction: q≈0 gives value=0 (center_term=0, norm=0)."""
        center = np.array([2.0, -1.0])
        h = BallSupportFunction(space_2d, center, 0.5)
        q = np.zeros(2)
        val, pt = h.value_and_support_point(q)
        assert_allclose(val, 0.0, atol=1e-14)
        assert pt is not None
        assert_allclose(pt, center, rtol=1e-12)

    def test_ball_near_zero_q_point_is_center(self, space_2d):
        """BallSupportFunction: q≈0 subcase — support_point is center."""
        center = np.array([1.0, 3.0])
        h = BallSupportFunction(space_2d, center, 1.5)
        q = np.full(2, 1e-20)  # effectively zero
        _, pt = h.value_and_support_point(q)
        assert_allclose(pt, center, rtol=1e-12)

    # ------------------------------------------------------------------
    # EllipsoidSupportFunction overridden implementation
    # ------------------------------------------------------------------

    @pytest.fixture
    def ellipsoid_h(self, space_2d):
        """EllipsoidSupportFunction with A = diag(4, 9), center=[1,-1], r=2."""
        center = np.array([1.0, -1.0])
        radius = 2.0
        A_diag = np.array([4.0, 9.0])
        A = LinearOperator.from_matrix(space_2d, space_2d, np.diag(A_diag))
        A_inv = LinearOperator.from_matrix(space_2d, space_2d, np.diag(1.0 / A_diag))
        A_inv_sqrt = LinearOperator.from_matrix(
            space_2d, space_2d, np.diag(1.0 / np.sqrt(A_diag))
        )
        return EllipsoidSupportFunction(space_2d, center, radius, A, A_inv, A_inv_sqrt)

    def test_ellipsoid_value_matches_standalone(self, ellipsoid_h):
        """EllipsoidSupportFunction: value_and_support_point value equals h(q)."""
        q = np.array([3.0, 2.0])
        val, _ = ellipsoid_h.value_and_support_point(q)
        assert_allclose(val, ellipsoid_h(q), rtol=1e-10)

    def test_ellipsoid_point_matches_standalone(self, ellipsoid_h):
        """EllipsoidSupportFunction: point equals support_point(q)."""
        q = np.array([3.0, 2.0])
        _, pt = ellipsoid_h.value_and_support_point(q)
        expected = ellipsoid_h.support_point(q)
        assert pt is not None
        assert expected is not None
        assert_allclose(pt, expected, rtol=1e-10)

    def test_ellipsoid_multiple_directions(self, ellipsoid_h):
        """EllipsoidSupportFunction: value and point consistent for 5 random q."""
        rng = np.random.default_rng(7)
        for _ in range(5):
            q = rng.standard_normal(2)
            val, pt = ellipsoid_h.value_and_support_point(q)
            assert_allclose(val, ellipsoid_h(q), rtol=1e-10)
            assert pt is not None
            assert_allclose(pt, ellipsoid_h.support_point(q), rtol=1e-10)

    def test_ellipsoid_no_inverse_support_point_none(self, space_2d):
        """EllipsoidSupportFunction without A_inv: value still returned, point None."""
        center = np.array([0.0, 0.0])
        radius = 1.0
        A_diag = np.array([4.0, 9.0])
        A = LinearOperator.from_matrix(space_2d, space_2d, np.diag(A_diag))
        A_inv_sqrt = LinearOperator.from_matrix(
            space_2d, space_2d, np.diag(1.0 / np.sqrt(A_diag))
        )
        # No inverse_operator – support_point returns None
        h = EllipsoidSupportFunction(
            space_2d, center, radius, A, inverse_sqrt_operator=A_inv_sqrt
        )
        q = np.array([1.0, 1.0])
        val, pt = h.value_and_support_point(q)
        assert pt is None
        assert_allclose(val, h(q), rtol=1e-10)

    def test_ellipsoid_zero_q_value_and_point(self, ellipsoid_h):
        """EllipsoidSupportFunction: q=0 → value=<q,c>=0, point=center (norm_term<1e-14 branch)."""
        q = np.zeros(2)
        val, pt = ellipsoid_h.value_and_support_point(q)
        # center_term = <q, c> = 0 since q is the zero vector
        assert_allclose(val, 0.0, atol=1e-14)
        assert pt is not None
        # The near-zero branch returns center = [1, -1] (from fixture)
        assert_allclose(pt, np.array([1.0, -1.0]), rtol=1e-12)

    def test_ellipsoid_very_small_q_returns_center(self, ellipsoid_h):
        """EllipsoidSupportFunction: q with tiny norm → point = center (branch check)."""
        q = np.full(2, 1e-20)  # norm_term << 1e-14 after A_inv application
        _, pt = ellipsoid_h.value_and_support_point(q)
        assert pt is not None
        assert_allclose(pt, np.array([1.0, -1.0]), rtol=1e-12)
