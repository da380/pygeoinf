"""
Tests for Phase 2 support-function algebra:
  - LinearImageSupportFunction  (h_{A(C)}(q) = h_C(A* q))
  - MinkowskiSumSupportFunction  (h_{C+D}(q) = h_C(q) + h_D(q))
  - ScaledSupportFunction        (h_{alpha C}(q) = alpha h_C(q))
  - SupportFunction.image(A)
  - SupportFunction.translate(p)
  - SupportFunction.scale(alpha)
  - SupportFunction.__add__ / __mul__ / __rmul__

TDD: tests were written before the implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.convex_analysis import (
    SupportFunction,
    BallSupportFunction,
    CallableSupportFunction,
    PointSupportFunction,
    LinearImageSupportFunction,
    MinkowskiSumSupportFunction,
    ScaledSupportFunction,
)

np.random.seed(42)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def R2():
    return EuclideanSpace(2)


@pytest.fixture
def R3():
    return EuclideanSpace(3)


@pytest.fixture
def R4():
    return EuclideanSpace(4)


@pytest.fixture
def ball_R2(R2):
    """Unit ball centred at origin in R^2."""
    return BallSupportFunction(R2, R2.zero, 1.0)


@pytest.fixture
def point_sf_R2(R2):
    """Support function for singleton {p} with p = [1, 2] in R^2."""
    p = np.array([1.0, 2.0])
    return PointSupportFunction(R2, p)


@pytest.fixture
def A_R2_to_R3(R2, R3):
    """A fixed 3x2 linear operator from R^2 to R^3."""
    mat = np.array([[1., 0.],
                    [0., 1.],
                    [1., 1.]])
    return LinearOperator.from_matrix(R2, R3, mat)


@pytest.fixture
def A_R3_to_R2(R2, R3):
    """Transpose of the above: 2x3 linear operator from R^3 to R^2."""
    mat = np.array([[1., 0., 1.],
                    [0., 1., 1.]])
    return LinearOperator.from_matrix(R3, R2, mat)


# ---------------------------------------------------------------------------
# LinearImageSupportFunction
# ---------------------------------------------------------------------------

class TestLinearImageSupportFunction:
    """h_{A(C)}(q) = h_C(A^* q)."""

    def test_value_matches_adjoint_pullback(self, R2, R3, ball_R2, A_R2_to_R3):
        """h_{A(C)}(q) equals h_C(A^* q) for a unit-ball support function."""
        h_image = LinearImageSupportFunction(ball_R2, A_R2_to_R3)

        rng = np.random.default_rng(0)
        for _ in range(10):
            q = rng.standard_normal(3)
            expected = ball_R2(A_R2_to_R3.adjoint(q))
            assert_allclose(h_image(q), expected, rtol=1e-12)

    def test_primal_domain_is_operator_codomain(self, R3, ball_R2, A_R2_to_R3):
        """primal_domain of image support equals operator codomain."""
        h_image = LinearImageSupportFunction(ball_R2, A_R2_to_R3)
        assert h_image.primal_domain is R3

    def test_returns_support_function_instance(self, ball_R2, A_R2_to_R3):
        """LinearImageSupportFunction is a SupportFunction."""
        h_image = LinearImageSupportFunction(ball_R2, A_R2_to_R3)
        assert isinstance(h_image, SupportFunction)

    def test_domain_mismatch_raises(self, R3, R2, A_R2_to_R3):
        """Raises ValueError when operator.domain != base.primal_domain."""
        ball_R3 = BallSupportFunction(R3, R3.zero, 1.0)
        # A_R2_to_R3.domain is R2, but ball_R3 lives in R3 — mismatch
        with pytest.raises(ValueError, match="domain"):
            LinearImageSupportFunction(ball_R3, A_R2_to_R3)

    def test_support_point_is_none_by_default(self, ball_R2, A_R2_to_R3):
        """Phase 3: support_point now propagates from base."""
        h_image = LinearImageSupportFunction(ball_R2, A_R2_to_R3)
        q = np.array([1.0, 0.0, 0.0])
        # Phase 3: BallSupportFunction has support_point, so h_image should too
        assert h_image.support_point(q) is not None

    def test_image_convenience_method(self, R3, ball_R2, A_R2_to_R3):
        """SupportFunction.image(A) returns a LinearImageSupportFunction on A.codomain."""
        h_image = ball_R2.image(A_R2_to_R3)
        assert isinstance(h_image, LinearImageSupportFunction)
        assert h_image.primal_domain is R3
        q = np.array([1.0, 0.0, -1.0])
        assert_allclose(h_image(q), ball_R2(A_R2_to_R3.adjoint(q)), rtol=1e-12)

    def test_image_method_domain_mismatch_raises(self, R3, A_R2_to_R3):
        """image() raises when operator.domain != self.primal_domain."""
        ball_R3 = BallSupportFunction(R3, R3.zero, 1.0)
        with pytest.raises(ValueError, match="domain"):
            ball_R3.image(A_R2_to_R3)

    def test_identity_operator_unchanged(self, R2, ball_R2):
        """Applying the identity operator produces the same values as the original."""
        eye = LinearOperator.from_matrix(R2, R2, np.eye(2))
        h_image = ball_R2.image(eye)
        rng = np.random.default_rng(1)
        for _ in range(8):
            q = rng.standard_normal(2)
            assert_allclose(h_image(q), ball_R2(q), rtol=1e-12)


# ---------------------------------------------------------------------------
# MinkowskiSumSupportFunction
# ---------------------------------------------------------------------------

class TestMinkowskiSumSupportFunction:
    """h_{C+D}(q) = h_C(q) + h_D(q)."""

    def test_value_is_sum_of_individual_supports(self, R2, ball_R2, point_sf_R2):
        """h_{C+D}(q) equals h_C(q) + h_D(q)."""
        h_sum = MinkowskiSumSupportFunction(ball_R2, point_sf_R2)
        rng = np.random.default_rng(2)
        for _ in range(10):
            q = rng.standard_normal(2)
            expected = ball_R2(q) + point_sf_R2(q)
            assert_allclose(h_sum(q), expected, rtol=1e-12)

    def test_primal_domain_preserved(self, R2, ball_R2, point_sf_R2):
        """primal_domain of Minkowski sum equals the shared space."""
        h_sum = MinkowskiSumSupportFunction(ball_R2, point_sf_R2)
        assert h_sum.primal_domain is R2

    def test_returns_support_function_instance(self, ball_R2, point_sf_R2):
        """MinkowskiSumSupportFunction is a SupportFunction."""
        h_sum = MinkowskiSumSupportFunction(ball_R2, point_sf_R2)
        assert isinstance(h_sum, SupportFunction)

    def test_domain_mismatch_raises(self, R2, R3, ball_R2):
        """Raises ValueError when primal_domain of two summands differ."""
        ball_R3 = BallSupportFunction(R3, R3.zero, 1.0)
        with pytest.raises(ValueError, match="primal_domain"):
            MinkowskiSumSupportFunction(ball_R2, ball_R3)

    def test_add_operator_returns_minkowski_sum(self, R2, ball_R2, point_sf_R2):
        """h1 + h2 returns a MinkowskiSumSupportFunction, not a plain NonLinearForm."""
        h_sum = ball_R2 + point_sf_R2
        assert isinstance(h_sum, MinkowskiSumSupportFunction)
        rng = np.random.default_rng(3)
        for _ in range(8):
            q = rng.standard_normal(2)
            assert_allclose(h_sum(q), ball_R2(q) + point_sf_R2(q), rtol=1e-12)

    def test_support_point_is_none_by_default(self, ball_R2, point_sf_R2):
        """Phase 3: support_point now propagates from both operands."""
        h_sum = MinkowskiSumSupportFunction(ball_R2, point_sf_R2)
        q = np.array([1.0, 0.0])
        # Phase 3: Both operands have support_point, so h_sum should too
        assert h_sum.support_point(q) is not None

    def test_symmetry(self, R2, ball_R2, point_sf_R2):
        """h_C + h_D == h_D + h_C (same values)."""
        h1 = MinkowskiSumSupportFunction(ball_R2, point_sf_R2)
        h2 = MinkowskiSumSupportFunction(point_sf_R2, ball_R2)
        rng = np.random.default_rng(4)
        for _ in range(8):
            q = rng.standard_normal(2)
            assert_allclose(h1(q), h2(q), rtol=1e-12)


# ---------------------------------------------------------------------------
# ScaledSupportFunction
# ---------------------------------------------------------------------------

class TestScaledSupportFunction:
    """h_{alpha C}(q) = alpha h_C(q)."""

    def test_positive_scaling(self, R2, ball_R2):
        """h_{alpha C}(q) = alpha h_C(q) for alpha > 0."""
        alpha = 3.5
        h_scaled = ScaledSupportFunction(ball_R2, alpha)
        rng = np.random.default_rng(5)
        for _ in range(10):
            q = rng.standard_normal(2)
            assert_allclose(h_scaled(q), alpha * ball_R2(q), rtol=1e-12)

    def test_unit_scaling(self, R2, ball_R2):
        """Scaling by 1.0 leaves values unchanged."""
        h_scaled = ScaledSupportFunction(ball_R2, 1.0)
        rng = np.random.default_rng(6)
        for _ in range(8):
            q = rng.standard_normal(2)
            assert_allclose(h_scaled(q), ball_R2(q), rtol=1e-12)

    def test_zero_scaling_returns_zero(self, R2, ball_R2):
        """Scaling by 0 produces h = 0 everywhere (support of singleton {0})."""
        h_zero = ScaledSupportFunction(ball_R2, 0.0)
        rng = np.random.default_rng(7)
        for _ in range(8):
            q = rng.standard_normal(2)
            assert_allclose(h_zero(q), 0.0, atol=1e-14)

    def test_zero_scaling_on_unbounded_support_is_zero(self, R2):
        """alpha=0 on a support that returns +inf must give 0.0, not nan.

        Uses a CallableSupportFunction that mimics the half-space {x: x[0] <= 1}
        (returns +inf for directions not in the nonneg span of [1,0]).
        Without the short-circuit, 0 * float('inf') == nan.
        """
        def half_space_like(q):
            # sigma_{H}(q) = q[0] if q[1]==0 and q[0]>=0, else +inf
            if q[1] != 0.0 or q[0] < 0.0:
                return float("inf")
            return float(q[0])

        h_unbounded = CallableSupportFunction(R2, half_space_like)
        h_zero = ScaledSupportFunction(h_unbounded, 0.0)

        # direction where base is finite
        q_finite = np.array([2.0, 0.0])
        assert_allclose(h_zero(q_finite), 0.0, atol=1e-14)
        # direction where base returns +inf — must be 0.0, not nan
        q_inf = np.array([0.0, 1.0])
        result = h_zero(q_inf)
        assert result == 0.0, f"Expected 0.0, got {result} (0*inf nan-bug not fixed)"

    def test_negative_scaling_raises(self, R2, ball_R2):
        """Negative alpha raises ValueError."""
        with pytest.raises(ValueError, match="[Nn]on-?negative|alpha"):
            ScaledSupportFunction(ball_R2, -1.0)

    def test_primal_domain_preserved(self, R2, ball_R2):
        """primal_domain is unchanged after scaling."""
        h_scaled = ScaledSupportFunction(ball_R2, 2.0)
        assert h_scaled.primal_domain is R2

    def test_returns_support_function_instance(self, ball_R2):
        """ScaledSupportFunction is a SupportFunction."""
        h_scaled = ScaledSupportFunction(ball_R2, 2.0)
        assert isinstance(h_scaled, SupportFunction)

    def test_support_point_is_none_by_default(self, ball_R2):
        """Phase 3: support_point now propagates from base."""
        h_scaled = ScaledSupportFunction(ball_R2, 2.0)
        q = np.array([1.0, 0.0])
        # Phase 3: BallSupportFunction has support_point, so h_scaled should too
        assert h_scaled.support_point(q) is not None

    def test_mul_operator(self, R2, ball_R2):
        """h * alpha returns ScaledSupportFunction (not plain NonLinearForm)."""
        h_scaled = ball_R2 * 4.0
        assert isinstance(h_scaled, ScaledSupportFunction)
        rng = np.random.default_rng(8)
        for _ in range(8):
            q = rng.standard_normal(2)
            assert_allclose(h_scaled(q), 4.0 * ball_R2(q), rtol=1e-12)

    def test_rmul_operator(self, R2, ball_R2):
        """alpha * h returns ScaledSupportFunction (not plain NonLinearForm)."""
        h_scaled = 4.0 * ball_R2
        assert isinstance(h_scaled, ScaledSupportFunction)
        rng = np.random.default_rng(9)
        for _ in range(8):
            q = rng.standard_normal(2)
            assert_allclose(h_scaled(q), 4.0 * ball_R2(q), rtol=1e-12)

    def test_mul_negative_raises(self, ball_R2):
        """h * (-1) raises ValueError."""
        with pytest.raises(ValueError, match="[Nn]on-?negative|alpha"):
            _ = ball_R2 * (-1.0)

    def test_rmul_negative_raises(self, ball_R2):
        """(-1) * h raises ValueError."""
        with pytest.raises(ValueError, match="[Nn]on-?negative|alpha"):
            _ = (-1.0) * ball_R2


# ---------------------------------------------------------------------------
# SupportFunction.translate
# ---------------------------------------------------------------------------

class TestTranslate:
    """h_{C + p}(q) = h_C(q) + <q, p>."""

    def test_value_matches_formula(self, R2, ball_R2):
        """h_{C+p}(q) = h_C(q) + <q, p>."""
        p = np.array([1.5, -0.5])
        h_trans = ball_R2.translate(p)
        rng = np.random.default_rng(10)
        for _ in range(10):
            q = rng.standard_normal(2)
            expected = ball_R2(q) + R2.inner_product(q, p)
            assert_allclose(h_trans(q), expected, rtol=1e-12)

    def test_returns_support_function_instance(self, R2, ball_R2):
        """translate() returns a SupportFunction."""
        p = np.array([0.0, 1.0])
        h_trans = ball_R2.translate(p)
        assert isinstance(h_trans, SupportFunction)

    def test_translate_returns_minkowski_sum(self, R2, ball_R2):
        """translate() produces a MinkowskiSumSupportFunction."""
        p = np.array([1.0, -1.0])
        h_trans = ball_R2.translate(p)
        assert isinstance(h_trans, MinkowskiSumSupportFunction)

    def test_translate_zero_no_change(self, R2, ball_R2):
        """Translating by zero vector leaves values unchanged."""
        p = R2.zero
        h_trans = ball_R2.translate(p)
        rng = np.random.default_rng(11)
        for _ in range(8):
            q = rng.standard_normal(2)
            assert_allclose(h_trans(q), ball_R2(q), rtol=1e-12)


# ---------------------------------------------------------------------------
# SupportFunction.scale convenience method
# ---------------------------------------------------------------------------

class TestScaleMethod:
    """scale(alpha) convenience method."""

    def test_scale_returns_scaled_support_function(self, R2, ball_R2):
        """scale(alpha) returns a ScaledSupportFunction."""
        h = ball_R2.scale(2.0)
        assert isinstance(h, ScaledSupportFunction)

    def test_scale_value_matches(self, R2, ball_R2):
        """scale(alpha) produces correct values."""
        alpha = 2.5
        h = ball_R2.scale(alpha)
        rng = np.random.default_rng(12)
        for _ in range(8):
            q = rng.standard_normal(2)
            assert_allclose(h(q), alpha * ball_R2(q), rtol=1e-12)

    def test_scale_negative_raises(self, ball_R2):
        """scale(-1) raises ValueError."""
        with pytest.raises(ValueError):
            ball_R2.scale(-1.0)


# ---------------------------------------------------------------------------
# Chained algebra
# ---------------------------------------------------------------------------

class TestChainedAlgebra:
    """Verify multi-step compositions produce correct values."""

    def test_scaled_then_translated(self, R2, ball_R2):
        """(alpha * h) + h_p = alpha*h_C(q) + <q, p>."""
        alpha = 2.0
        p = np.array([1.0, 0.5])
        h_p = PointSupportFunction(R2, p)

        h_composed = ball_R2.scale(alpha) + h_p
        rng = np.random.default_rng(13)
        for _ in range(10):
            q = rng.standard_normal(2)
            expected = alpha * ball_R2(q) + R2.inner_product(q, p)
            assert_allclose(h_composed(q), expected, rtol=1e-12)

    def test_image_then_scaled(self, R2, R3, ball_R2, A_R2_to_R3):
        """(beta * h.image(A))(q) = beta * h_C(A^* q)."""
        beta = 3.0
        h_image = ball_R2.image(A_R2_to_R3)
        h_composed = beta * h_image
        rng = np.random.default_rng(14)
        for _ in range(10):
            q = rng.standard_normal(3)
            expected = beta * ball_R2(A_R2_to_R3.adjoint(q))
            assert_allclose(h_composed(q), expected, rtol=1e-12)

    def test_add_non_support_function_raises(self, R2, ball_R2):
        """Adding a plain NonLinearForm (not a SupportFunction) raises TypeError."""
        from pygeoinf.nonlinear_forms import NonLinearForm
        plain_form = NonLinearForm(R2, lambda q: 0.0)
        with pytest.raises(TypeError):
            _ = ball_R2 + plain_form

    def test_mul_non_scalar_raises(self, ball_R2):
        """Multiplying by a non-scalar raises TypeError."""
        with pytest.raises((TypeError, ValueError)):
            _ = ball_R2 * "oops"


# ---------------------------------------------------------------------------
# Phase 3: Support-point propagation through algebraic wrappers
# ---------------------------------------------------------------------------


class TestLinearImageSupportPointPropagation:
    """Phase 3: LinearImageSupportFunction.support_point propagates base support points."""

    def test_support_point_propagates_from_ball(self, R2, R3, ball_R2, A_R2_to_R3):
        """LinearImageSupportFunction.support_point returns A(x_base*(A^* q)) when available."""
        h_image = LinearImageSupportFunction(ball_R2, A_R2_to_R3)

        q = np.array([1.0, 0.0, 0.0])
        x_star = h_image.support_point(q)

        # ball_R2 has a support point, so x_star should not be None
        assert x_star is not None

        # The support point should be in R^3 (the codomain of A)
        assert x_star.shape == (3,)

        # Verify: x_star should be A(x_C^*(A^*(q)))
        adj_q = A_R2_to_R3.adjoint(q)
        x_base = ball_R2.support_point(adj_q)
        expected = A_R2_to_R3(x_base)  # Apply A to the base support point
        assert_allclose(x_star, expected, rtol=1e-12)

    def test_support_point_returns_none_for_callable_without_fn(self, R2, R3, A_R2_to_R3):
        """LinearImageSupportFunction.support_point returns None when base has no support_point."""
        # Create a callable support function without a support_point callback
        h_callable = CallableSupportFunction(R2, lambda q: float(np.linalg.norm(q)))
        h_image = LinearImageSupportFunction(h_callable, A_R2_to_R3)

        q = np.array([1.0, 0.0, 0.0])
        assert h_image.support_point(q) is None

    def test_support_point_propagates_from_point_sf(self, R2, R3, A_R2_to_R3):
        """LinearImageSupportFunction.support_point works with PointSupportFunction base."""
        p = np.array([1.0, 2.0])
        h_point = PointSupportFunction(R2, p)
        h_image = LinearImageSupportFunction(h_point, A_R2_to_R3)

        # For a singleton {p}, the support point is always A(p) (in the codomain)
        q = np.array([3.0, -1.0, 0.5])
        x_star = h_image.support_point(q)
        assert x_star is not None
        expected = A_R2_to_R3(p)  # Apply A to p
        assert_allclose(x_star, expected, rtol=1e-12)

    def test_subgradient_works_when_support_point_available(self, R2, R3, ball_R2, A_R2_to_R3):
        """subgradient(q) works through linearimage when support_point is available."""
        h_image = LinearImageSupportFunction(ball_R2, A_R2_to_R3)

        q = np.array([1.0, 0.0, 0.0])
        # subgradient should delegate to support_point and return a result
        grad = h_image.subgradient(q)
        assert grad is not None
        assert grad.shape == (3,)  # Result is in R^3 (codomain of A)

    def test_subgradient_raises_when_support_point_unavailable(self, R2, R3, A_R2_to_R3):
        """subgradient(q) raises NotImplementedError when support_point is None."""
        h_callable = CallableSupportFunction(R2, lambda q: float(np.linalg.norm(q)))
        h_image = LinearImageSupportFunction(h_callable, A_R2_to_R3)

        q = np.array([1.0, 0.0, 0.0])
        with pytest.raises(NotImplementedError, match="Support point"):
            h_image.subgradient(q)


class TestMinkowskiSumSupportPointPropagation:
    """Phase 3: MinkowskiSumSupportFunction.support_point adds support points when both are available."""

    def test_support_point_sum_both_available(self, R2, ball_R2, point_sf_R2):
        """MinkowskiSumSupportFunction.support_point = left.support_point + right.support_point."""
        h_sum = MinkowskiSumSupportFunction(ball_R2, point_sf_R2)

        q = np.array([1.0, 0.0])
        x_sum = h_sum.support_point(q)

        # Both have support points, so x_sum should not be None
        assert x_sum is not None

        # x_sum should equal the sum of individual support points
        x_ball = ball_R2.support_point(q)
        x_point = point_sf_R2.support_point(q)
        expected = R2.add(x_ball, x_point)
        assert_allclose(x_sum, expected, rtol=1e-12)

    def test_support_point_returns_none_left_unavailable(self, R2, ball_R2):
        """MinkowskiSumSupportFunction.support_point returns None if left has no support_point."""
        h_callable = CallableSupportFunction(R2, lambda q: float(np.linalg.norm(q)))
        h_sum = MinkowskiSumSupportFunction(h_callable, ball_R2)

        q = np.array([1.0, 0.0])
        assert h_sum.support_point(q) is None

    def test_support_point_returns_none_right_unavailable(self, R2, ball_R2):
        """MinkowskiSumSupportFunction.support_point returns None if right has no support_point."""
        h_callable = CallableSupportFunction(R2, lambda q: float(np.linalg.norm(q)))
        h_sum = MinkowskiSumSupportFunction(ball_R2, h_callable)

        q = np.array([1.0, 0.0])
        assert h_sum.support_point(q) is None

    def test_support_point_returns_none_both_unavailable(self, R2):
        """MinkowskiSumSupportFunction.support_point returns None if both have no support_point."""
        h1 = CallableSupportFunction(R2, lambda q: float(np.linalg.norm(q)))
        h2 = CallableSupportFunction(R2, lambda q: 2.0 * float(np.linalg.norm(q)))
        h_sum = MinkowskiSumSupportFunction(h1, h2)

        q = np.array([1.0, 0.0])
        assert h_sum.support_point(q) is None

    def test_subgradient_works_both_available(self, R2, ball_R2, point_sf_R2):
        """subgradient works when both operands have support_point."""
        h_sum = MinkowskiSumSupportFunction(ball_R2, point_sf_R2)

        q = np.array([1.0, 0.0])
        grad = h_sum.subgradient(q)
        assert grad is not None
        assert grad.shape == (2,)

    def test_subgradient_raises_when_either_unavailable(self, R2, ball_R2):
        """subgradient raises NotImplementedError when either operand has no support_point."""
        h_callable = CallableSupportFunction(R2, lambda q: float(np.linalg.norm(q)))
        h_sum = MinkowskiSumSupportFunction(ball_R2, h_callable)

        q = np.array([1.0, 0.0])
        with pytest.raises(NotImplementedError, match="Support point"):
            h_sum.subgradient(q)

    def test_support_point_with_minkowski_operator(self, R2, ball_R2, point_sf_R2):
        """Support point propagates through + operator composition."""
        h_sum = ball_R2 + point_sf_R2

        q = np.array([1.0, 0.0])
        x_sum = h_sum.support_point(q)
        assert x_sum is not None

        x_ball = ball_R2.support_point(q)
        x_point = point_sf_R2.support_point(q)
        expected = R2.add(x_ball, x_point)
        assert_allclose(x_sum, expected, rtol=1e-12)


class TestScaledSupportPointPropagation:
    """Phase 3: ScaledSupportFunction.support_point scales base support points."""

    def test_support_point_positive_scaling(self, R2, ball_R2):
        """ScaledSupportFunction.support_point = alpha * base.support_point for alpha > 0."""
        alpha = 2.5
        h_scaled = ScaledSupportFunction(ball_R2, alpha)

        q = np.array([1.0, 0.0])
        x_scaled = h_scaled.support_point(q)

        assert x_scaled is not None

        # x_scaled should equal alpha times the base support point
        x_base = ball_R2.support_point(q)
        expected = R2.multiply(alpha, x_base)
        assert_allclose(x_scaled, expected, rtol=1e-12)

    def test_support_point_zero_scaling_returns_zero(self, R2, ball_R2):
        """ScaledSupportFunction.support_point returns zero vector for alpha=0."""
        h_zero = ScaledSupportFunction(ball_R2, 0.0)

        q = np.array([1.0, 0.0])
        x_zero = h_zero.support_point(q)

        # The zero set has only the zero point as a support point everywhere
        assert x_zero is not None
        assert_allclose(x_zero, R2.zero, atol=1e-14)

    def test_support_point_zero_scaling_for_all_directions(self, R2, ball_R2):
        """ScaledSupportFunction with alpha=0 returns zero for any direction."""
        h_zero = ScaledSupportFunction(ball_R2, 0.0)

        rng = np.random.default_rng(20)
        for _ in range(10):
            q = rng.standard_normal(2)
            x_zero = h_zero.support_point(q)
            assert x_zero is not None
            assert_allclose(x_zero, R2.zero, atol=1e-14)

    def test_support_point_unit_scaling(self, R2, ball_R2):
        """ScaledSupportFunction.support_point with alpha=1 equals base.support_point."""
        h_unit = ScaledSupportFunction(ball_R2, 1.0)

        q = np.array([1.0, 0.0])
        x_unit = h_unit.support_point(q)
        x_base = ball_R2.support_point(q)
        assert_allclose(x_unit, x_base, rtol=1e-12)

    def test_support_point_returns_none_no_base_support_point(self, R2):
        """ScaledSupportFunction.support_point returns None if base has no support_point."""
        h_callable = CallableSupportFunction(R2, lambda q: float(np.linalg.norm(q)))
        h_scaled = ScaledSupportFunction(h_callable, 2.0)

        q = np.array([1.0, 0.0])
        assert h_scaled.support_point(q) is None

    def test_subgradient_works_positive_scaling(self, R2, ball_R2):
        """subgradient works for positively scaled support functions."""
        h_scaled = ScaledSupportFunction(ball_R2, 3.0)

        q = np.array([1.0, 0.0])
        grad = h_scaled.subgradient(q)
        assert grad is not None
        assert grad.shape == (2,)

    def test_subgradient_works_zero_scaling(self, R2, ball_R2):
        """subgradient returns the zero vector for alpha=0 (support_point is available as zero)."""
        h_zero = ScaledSupportFunction(ball_R2, 0.0)

        q = np.array([1.0, 0.0])
        grad = h_zero.subgradient(q)
        assert grad is not None
        assert_allclose(grad, R2.zero, atol=1e-14)

    def test_subgradient_raises_when_base_unavailable(self, R2):
        """subgradient raises NotImplementedError when base has no support_point."""
        h_callable = CallableSupportFunction(R2, lambda q: float(np.linalg.norm(q)))
        h_scaled = ScaledSupportFunction(h_callable, 2.0)

        q = np.array([1.0, 0.0])
        with pytest.raises(NotImplementedError, match="Support point"):
            h_scaled.subgradient(q)

    def test_support_point_with_mul_operator(self, R2, ball_R2):
        """Support point propagates through * operator composition."""
        h_scaled = ball_R2 * 2.0

        q = np.array([1.0, 0.0])
        x_scaled = h_scaled.support_point(q)
        assert x_scaled is not None

        x_base = ball_R2.support_point(q)
        expected = R2.multiply(2.0, x_base)
        assert_allclose(x_scaled, expected, rtol=1e-12)

    def test_support_point_with_rmul_operator(self, R2, ball_R2):
        """Support point propagates through rmul operator composition."""
        h_scaled = 2.0 * ball_R2

        q = np.array([1.0, 0.0])
        x_scaled = h_scaled.support_point(q)
        assert x_scaled is not None

        x_base = ball_R2.support_point(q)
        expected = R2.multiply(2.0, x_base)
        assert_allclose(x_scaled, expected, rtol=1e-12)


class TestComplexSupportPointPropagation:
    """Phase 3: Support point propagation through complex nested compositions."""

    def test_nested_minkowski_scaled(self, R2, ball_R2, point_sf_R2):
        """Support point propagates through (h1 + h2) * alpha."""
        alpha = 2.0
        h_composed = (ball_R2 + point_sf_R2) * alpha
        # h_composed is ScaledSupportFunction(MinkowskiSumSupportFunction(...), alpha)

        q = np.array([1.0, 0.0])
        x_composed = h_composed.support_point(q)

        # Should be alpha * (x_ball + x_point)
        assert x_composed is not None

        x_ball = ball_R2.support_point(q)
        x_point = point_sf_R2.support_point(q)
        expected = R2.multiply(alpha, R2.add(x_ball, x_point))
        assert_allclose(x_composed, expected, rtol=1e-12)

    def test_scaled_then_add_point(self, R2, ball_R2):
        """Support point propagates through (alpha * h) + h_p."""
        alpha = 2.0
        p = np.array([0.5, -0.5])
        h_point = PointSupportFunction(R2, p)
        h_composed = (ball_R2 * alpha) + h_point

        q = np.array([1.0, 0.0])
        x_composed = h_composed.support_point(q)

        # Should be (alpha * x_ball) + x_point
        assert x_composed is not None

        x_ball = ball_R2.support_point(q)
        x_point = h_point.support_point(q)
        expected = R2.add(R2.multiply(alpha, x_ball), x_point)
        assert_allclose(x_composed, expected, rtol=1e-12)

    def test_support_point_propagates_through_image(self, R2, R3, ball_R2, A_R2_to_R3):
        """Support point propagates through linear image composition."""
        h_image = ball_R2.image(A_R2_to_R3)
        p_codomain = np.array([1.0, 0.0, -1.0])
        h_point_codomain = PointSupportFunction(R3, p_codomain)
        h_composed = h_image + h_point_codomain

        # h_image.primal_domain = R3, h_point_codomain.primal_domain = R3 ✓
        q = np.array([1.0, 0.0, 0.0])
        x_composed = h_composed.support_point(q)

        assert x_composed is not None
        x_image = h_image.support_point(q)
        x_point = h_point_codomain.support_point(q)
        expected = R3.add(x_image, x_point)
        assert_allclose(x_composed, expected, rtol=1e-12)

    def test_support_point_returns_none_partial_composition(self, R2, ball_R2):
        """Support point returns None when one branch of composition has no support_point."""
        h_callable = CallableSupportFunction(R2, lambda q: float(np.linalg.norm(q)))
        h_composed = (ball_R2 * 2.0) + h_callable

        q = np.array([1.0, 0.0])
        # MinkowskiSum has left support point but right does not → returns None
        assert h_composed.support_point(q) is None
