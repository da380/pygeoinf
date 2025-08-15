"""
Tests for the LinearForm class.
"""
import pytest
import numpy as np
from typing import  Union
from pygeoinf.hilbert_space import EuclideanSpace, HilbertSpace
from pygeoinf.symmetric_space.circle import Sobolev as CircleSobolev
from pygeoinf.symmetric_space.line import Sobolev as LineSobolev
from pygeoinf.symmetric_space.sphere import Sobolev as SphereSobolev
from pygeoinf.forms import LinearForm



from pygeoinf.hilbert_space import T_vec


# =============================================================================
# Parametrized Fixtures
# =============================================================================

# Define the different Hilbert space instances we want to test against.
space_implementations = [
    EuclideanSpace(dim=10),
    CircleSobolev(16, 1.0, 0.1),
    LineSobolev(16, 1.0, 0.1),
    SphereSobolev(16, 1.0, 0.1),
]

# Use pytest.mark.parametrize to create a fixture that will run tests
# for each of the space implementations defined above.
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
    def x(self, space: HilbertSpace) -> T_vec:
        """Provides a random vector from the parametrized space."""
        return space.random()

    @pytest.fixture
    def components(self, space: HilbertSpace) -> np.ndarray:
        """Provides a random component vector for a linear form."""
        return np.random.randn(space.dim)

    @pytest.fixture
    def form_from_components(self, space: HilbertSpace, components: np.ndarray) -> LinearForm:
        """Provides a LinearForm instance created from a component vector."""
        return LinearForm(space, components=components)

    @pytest.fixture
    def form_from_mapping(self, space: HilbertSpace, components: np.ndarray) -> LinearForm:
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
        x: T_vec,
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
        assert np.allclose(form_from_components.components, components)

    def test_component_computation(
        self, form_from_mapping: LinearForm, components: np.ndarray
    ):
        """Tests that components are computed correctly for a form defined by a mapping."""
        computed_components = form_from_mapping.components
        assert np.allclose(computed_components, components)

    def test_addition(
        self, form_from_components: LinearForm, form_from_mapping: LinearForm, x: T_vec
    ):
        """Tests the addition of two linear forms: (f1 + f2)(x) = f1(x) + f2(x)."""
        f1 = form_from_components
        f2 = form_from_mapping
        sum_form = f1 + f2
        assert np.isclose(sum_form(x), f1(x) + f2(x))
        assert np.allclose(sum_form.components, f1.components + f2.components)

    def test_subtraction(
        self, form_from_components: LinearForm, form_from_mapping: LinearForm, x: T_vec
    ):
        """Tests subtraction: (f1 - f2)(x) = f1(x) - f2(x)."""
        f1 = form_from_components
        f2 = form_from_mapping
        diff_form = f1 - f2
        assert np.isclose(diff_form(x), f1(x) - f2(x))
        assert np.allclose(diff_form.components, f1.components - f2.components)

    def test_scalar_multiplication(self, form_from_components: LinearForm, x: T_vec):
        """Tests scalar multiplication: (a * f)(x) = a * f(x)."""
        scalar = 2.5
        scaled_form = scalar * form_from_components
        assert np.isclose(scaled_form(x), scalar * form_from_components(x))
        assert np.allclose(scaled_form.components, scalar * form_from_components.components)

    def test_negation(self, form_from_components: LinearForm, x: T_vec):
        """Tests negation: (-f)(x) = -f(x)."""
        neg_form = -form_from_components
        assert np.isclose(neg_form(x), -form_from_components(x))
        assert np.allclose(neg_form.components, -form_from_components.components)
