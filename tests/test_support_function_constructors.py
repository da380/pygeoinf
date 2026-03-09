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
from pygeoinf.convex_analysis import (
    SupportFunction,
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
