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
        """Phase 2: support_point returns None (Phase 3 will propagate)."""
        h_image = LinearImageSupportFunction(ball_R2, A_R2_to_R3)
        q = np.array([1.0, 0.0, 0.0])
        assert h_image.support_point(q) is None

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
        """Phase 2: support_point returns None."""
        h_sum = MinkowskiSumSupportFunction(ball_R2, point_sf_R2)
        q = np.array([1.0, 0.0])
        assert h_sum.support_point(q) is None

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
        """Phase 2: support_point returns None."""
        h_scaled = ScaledSupportFunction(ball_R2, 2.0)
        q = np.array([1.0, 0.0])
        assert h_scaled.support_point(q) is None

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
