"""
Tests for the NonLinearForm class.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from pygeoinf.hilbert_space import HilbertSpace, EuclideanSpace, Vector
from pygeoinf.symmetric_space.circle import Sobolev as CircleSobolev
from pygeoinf.nonlinear_forms import NonLinearForm


# Use a smaller set of spaces for brevity, or the full set from the other test file.
space_implementations = [
    EuclideanSpace(10),
    CircleSobolev(16, 1.0, 0.1),
]


@pytest.fixture(params=space_implementations)
def space(request) -> HilbertSpace:
    """Provides parametrized HilbertSpace instances for the tests."""
    return request.param


# =============================================================================
# Unified Test Suite for NonLinearForm
# =============================================================================


class TestNonLinearForm:
    """
    A suite of tests for the NonLinearForm class using a simple quadratic form.
    """

    @pytest.fixture
    def x(self, space: HilbertSpace) -> Vector:
        """Provides a random vector from the parametrized space."""
        return space.random()

    @pytest.fixture
    def quadratic_form(self, space: HilbertSpace) -> NonLinearForm:
        """
        Provides a simple, verifiable NonLinearForm instance: f(x) = x · x.

        The derivatives of this form are:
        - Gradient: ∇f(x) = 2x
        - Hessian:  H(x) = 2I (where I is the identity operator)
        """
        # Mapping: f(x) = x · x
        # mapping = lambda v: np.dot(space.to_components(v), space.to_components(v))
        mapping = lambda x: space.inner_product(x, x)

        # Gradient: ∇f(x) = 2x
        gradient = lambda x: space.multiply(2, x)

        # Hessian: H(x) = 2 * Identity
        hessian = lambda v: space.identity_operator() * 2.0

        return NonLinearForm(space, mapping, gradient=gradient, hessian=hessian)

    def test_initialization(self, quadratic_form: NonLinearForm):
        """Tests that a NonLinearForm object is created successfully."""
        assert quadratic_form is not None

    def test_call_action(
        self, space: HilbertSpace, quadratic_form: NonLinearForm, x: Vector
    ):
        """Tests that the form's action on a vector is correct."""
        expected_value = space.inner_product(x, x)
        actual_value = quadratic_form(x)
        assert np.isclose(actual_value, expected_value)

    def test_gradient(
        self, space: HilbertSpace, quadratic_form: NonLinearForm, x: Vector
    ):
        """Tests that the gradient of the form is computed correctly."""
        grad_vector = quadratic_form.gradient(x)

        # Expected is 2*x
        expected_components = 2.0 * space.to_components(x)
        assert_allclose(space.to_components(grad_vector), expected_components)

    def test_hessian(
        self, space: HilbertSpace, quadratic_form: NonLinearForm, x: Vector
    ):
        """Tests that the Hessian of the form is computed correctly."""
        hessian_operator = quadratic_form.hessian(x)

        # Expected operator is 2 * Identity, so H(x) should be 2*x
        output_vector = hessian_operator(x)
        expected_components = 2.0 * space.to_components(x)
        assert_allclose(space.to_components(output_vector), expected_components)

    def test_negation(self, quadratic_form: NonLinearForm, x: Vector):
        """Tests the negation of a form and its derivatives."""
        neg_form = -quadratic_form

        assert np.isclose(neg_form(x), -quadratic_form(x))

        # Test gradient of negated form
        neg_grad = neg_form.gradient(x)
        orig_grad = quadratic_form.gradient(x)
        assert_allclose(
            neg_form.domain.to_components(neg_grad),
            -neg_form.domain.to_components(orig_grad),
        )

    def test_scalar_multiplication(self, quadratic_form: NonLinearForm, x: Vector):
        """Tests scalar multiplication of a form and its derivatives."""
        scalar = 3.5
        scaled_form = scalar * quadratic_form

        assert np.isclose(scaled_form(x), scalar * quadratic_form(x))

        # Test gradient of scaled form
        scaled_grad = scaled_form.gradient(x)
        orig_grad = quadratic_form.gradient(x)
        assert_allclose(
            scaled_form.domain.to_components(scaled_grad),
            scalar * scaled_form.domain.to_components(orig_grad),
        )

    def test_addition(self, quadratic_form: NonLinearForm, x: Vector):
        """Tests the addition of two non-linear forms."""
        form_sum = quadratic_form + quadratic_form

        assert np.isclose(form_sum(x), 2 * quadratic_form(x))

        # Test gradient of the sum
        sum_grad = form_sum.gradient(x)
        orig_grad = quadratic_form.gradient(x)
        assert_allclose(
            form_sum.domain.to_components(sum_grad),
            2 * form_sum.domain.to_components(orig_grad),
        )

    def test_unimplemented_derivatives_raise_error(
        self, space: HilbertSpace, x: Vector
    ):
        """Tests that calling for a gradient/Hessian raises an error if not provided."""
        form_no_derivs = NonLinearForm(space, lambda v: 1.0)

        with pytest.raises(NotImplementedError):
            form_no_derivs.gradient(x)

        with pytest.raises(NotImplementedError):
            form_no_derivs.hessian(x)
