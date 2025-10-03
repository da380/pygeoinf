"""
Tests for the LinearForm class.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from pygeoinf.hilbert_space import EuclideanSpace, HilbertSpace, Vector
from pygeoinf.symmetric_space.circle import Sobolev as CircleSobolev
from pygeoinf.symmetric_space.sphere import Sobolev as SphereSobolev
from pygeoinf.linear_forms import LinearForm


# =============================================================================
# Parametrized Fixtures
# =============================================================================

# Define the different Hilbert space instances we want to test against.
space_implementations = [
    EuclideanSpace(dim=10),
    CircleSobolev(16, 1.0, 0.1),
    SphereSobolev(16, 1.0, 0.1),
]


@pytest.fixture(params=space_implementations)
def space(request) -> HilbertSpace:
    """Provides parametrized HilbertSpace instances for the tests."""
    return request.param


# =============================================================================
# Unified Test Suite
# =============================================================================


class TestLinearForm:
    """
    A unified suite of tests for the LinearForm class that runs against
    multiple different HilbertSpace implementations.
    """

    @pytest.fixture
    def x(self, space: HilbertSpace) -> Vector:
        """Provides a random vector from the parametrized space."""
        return space.random()

    @pytest.fixture
    def components(self, space: HilbertSpace) -> np.ndarray:
        """Provides a random component vector for a linear form."""
        return np.random.randn(space.dim)

    @pytest.fixture
    def form_from_components(
        self, space: HilbertSpace, components: np.ndarray
    ) -> LinearForm:
        """Provides a LinearForm instance created from a component vector."""
        return LinearForm(space, components=components)

    @pytest.fixture
    def form_from_mapping(
        self, space: HilbertSpace, components: np.ndarray
    ) -> LinearForm:
        """Provides a LinearForm instance created from a mapping."""
        mapping = lambda vec: np.dot(components, space.to_components(vec))
        return LinearForm(space, mapping=mapping)

    def test_initialization(
        self, form_from_components: LinearForm, form_from_mapping: LinearForm
    ):
        """Tests that LinearForm objects are created successfully."""
        assert form_from_components is not None
        assert form_from_mapping is not None
        assert form_from_components.domain == form_from_mapping.domain

    def test_call_action(
        self,
        space: HilbertSpace,
        form_from_components: LinearForm,
        x: Vector,
        components: np.ndarray,
    ):
        """Tests that the form's action on a vector is correct."""
        expected_value = np.dot(components, space.to_components(x))
        actual_value = form_from_components(x)
        assert np.isclose(actual_value, expected_value)

    def test_component_property(
        self, form_from_components: LinearForm, components: np.ndarray
    ):
        """Tests that the .components property returns the correct vector."""
        assert_allclose(form_from_components.components, components)

    def test_component_computation(
        self, form_from_mapping: LinearForm, components: np.ndarray
    ):
        """Tests that components are computed correctly for a form defined by a mapping."""
        computed_components = form_from_mapping.components
        assert_allclose(computed_components, components)

    def test_parallel_component_computation(
        self, space: HilbertSpace, components: np.ndarray
    ):
        """Tests that components are computed correctly using the parallel backend."""
        mapping = lambda vec: np.dot(components, space.to_components(vec))
        form = LinearForm(space, mapping=mapping, parallel=True, n_jobs=-1)
        computed_components = form.components
        assert_allclose(computed_components, components)

    def test_addition(
        self, form_from_components: LinearForm, form_from_mapping: LinearForm, x: Vector
    ):
        """Tests the addition of two linear forms: (f1 + f2)(x) = f1(x) + f2(x)."""
        f1 = form_from_components
        f2 = form_from_mapping
        sum_form = f1 + f2
        assert np.isclose(sum_form(x), f1(x) + f2(x))
        assert_allclose(sum_form.components, f1.components + f2.components)

    def test_subtraction(
        self, form_from_components: LinearForm, form_from_mapping: LinearForm, x: Vector
    ):
        """Tests subtraction: (f1 - f2)(x) = f1(x) - f2(x)."""
        f1 = form_from_components
        f2 = form_from_mapping
        diff_form = f1 - f2
        assert np.isclose(diff_form(x), f1(x) - f2(x))
        assert_allclose(diff_form.components, f1.components - f2.components)

    def test_scalar_multiplication(self, form_from_components: LinearForm, x: Vector):
        """Tests scalar multiplication: (a * f)(x) = a * f(x)."""
        scalar = 2.5
        scaled_form = scalar * form_from_components
        assert np.isclose(scaled_form(x), scalar * form_from_components(x))
        assert_allclose(
            scaled_form.components, scalar * form_from_components.components
        )

    def test_negation(self, form_from_components: LinearForm, x: Vector):
        """Tests negation: (-f)(x) = -f(x)."""
        neg_form = -form_from_components
        assert np.isclose(neg_form(x), -form_from_components(x))
        assert_allclose(neg_form.components, -form_from_components.components)

    def test_iadd(
        self, form_from_components: LinearForm, form_from_mapping: LinearForm
    ):
        """Tests in-place addition (+=)."""
        f1 = form_from_components.copy()
        f2 = form_from_mapping

        expected_components = f1.components + f2.components

        # The operation should return self
        result = f1.__iadd__(f2)

        assert result is f1
        assert_allclose(f1.components, expected_components)

    def test_imul(self, form_from_components: LinearForm):
        """Tests in-place scalar multiplication (*=)."""
        f1 = form_from_components.copy()
        scalar = -1.5

        expected_components = f1.components * scalar

        # The operation should return self
        result = f1.__imul__(scalar)

        assert result is f1
        assert_allclose(f1.components, expected_components)

    def test_copy(self, form_from_components: LinearForm):
        """Tests that the copy method creates an independent instance."""
        f1 = form_from_components
        f2 = f1.copy()

        # They should be equal in value but not the same object
        assert f1 is not f2
        assert f1.domain == f2.domain
        assert_allclose(f1.components, f2.components)

        # Modify the copy and ensure the original is unchanged
        f2 *= 2.0
        assert not np.allclose(f1.components, f2.components)

    def test_domain_mismatch_raises_error(self, form_from_components: LinearForm):
        """Tests that operating on forms with different domains raises a ValueError."""
        # Create a form on a different space
        different_space = EuclideanSpace(form_from_components.domain.dim + 1)
        form_on_different_domain = LinearForm(
            different_space, components=np.random.randn(different_space.dim)
        )

        with pytest.raises(ValueError):
            _ = form_from_components + form_on_different_domain

        with pytest.raises(ValueError):
            form_from_components += form_on_different_domain

    def test_as_linear_operator(self, form_from_components: LinearForm, x: Vector):
        """Tests the conversion to a LinearOperator."""
        form = form_from_components
        operator = form.as_linear_operator

        # The operator's codomain should be a 1D Euclidean space
        assert isinstance(operator.codomain, EuclideanSpace)
        assert operator.codomain.dim == 1

        # The action of the operator should match the action of the form
        form_value = form(x)
        operator_value = operator(x)

        assert isinstance(operator_value, np.ndarray)
        assert operator_value.shape == (1,)
        assert np.isclose(form_value, operator_value[0])

    def test_gradient_is_constant(
        self, space: HilbertSpace, form_from_components: LinearForm, x: Vector
    ):
        """Tests that the gradient of a linear form is a constant vector."""
        # The gradient of a linear form f(x) = c · x is the constant vector c.
        # The implementation returns this vector via space.from_dual(form).
        grad_vector = form_from_components.gradient(x)
        form_as_vector = space.from_dual(form_from_components)

        # The components of the gradient should match the components of the form.
        assert_allclose(
            space.to_components(form_as_vector), space.to_components(grad_vector)
        )

    def test_hessian_is_zero_operator(
        self, space: HilbertSpace, form_from_components: LinearForm, x: Vector
    ):
        """Tests that the Hessian of a linear form is the zero operator."""
        # The Hessian (second derivative) of a linear function is zero.
        hessian_operator = form_from_components.hessian(x)

        # Applying the zero operator to any vector should yield the zero vector.
        zero_vector = hessian_operator(x)
        expected_zero_components = np.zeros(space.dim)

        assert_allclose(space.to_components(zero_vector), expected_zero_components)

    def test_addition_with_nonlinear_form(
        self, space: HilbertSpace, form_from_components: LinearForm, x: Vector
    ):
        """Tests adding a LinearForm to a general NonLinearForm."""
        from pygeoinf.nonlinear_forms import NonLinearForm

        # Create a simple quadratic form: f(x) = x · x
        quadratic_mapping = lambda v: np.dot(
            space.to_components(v), space.to_components(v)
        )
        nonlinear_form = NonLinearForm(space, quadratic_mapping)

        linear_form = form_from_components

        # The result should be a general NonLinearForm, not a LinearForm
        sum_form = linear_form + nonlinear_form
        assert isinstance(sum_form, NonLinearForm)
        assert not isinstance(sum_form, LinearForm)

        # Verify the action of the resulting form
        expected = linear_form(x) + nonlinear_form(x)
        assert np.isclose(sum_form(x), expected)

    def test_scalar_division(self, form_from_components: LinearForm, x: Vector):
        """Tests scalar division: (f / a)(x) = f(x) / a."""
        scalar = -3.0
        divided_form = form_from_components / scalar
        assert np.isclose(divided_form(x), form_from_components(x) / scalar)
        assert_allclose(
            divided_form.components, form_from_components.components / scalar
        )
