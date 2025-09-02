"""
Tests for the interval Lebesgue space implementation.

This file uses the abstract HilbertSpaceChecks class to verify that the
interval Lebesgue space properly adheres to the core pygeoinf HilbertSpace
standards and mathematical axioms.
"""

import pytest
import numpy as np
from pygeoinf.interval.lebesgue_space import Lebesgue
from pygeoinf.interval.interval_domain import IntervalDomain
from pygeoinf.interval import function_providers as fp
from pygeoinf.interval.providers import BasisProvider
from pygeoinf.hilbert_space import HilbertSpace
from .checks.hilbert_space import HilbertSpaceChecks


@pytest.fixture(scope="module")
def interval_domain() -> IntervalDomain:
    """A simple interval domain [0, 1]."""
    return IntervalDomain(0, 1)


@pytest.fixture(scope="module")
def lebesgue_space_callables(interval_domain: IntervalDomain) -> Lebesgue:
    """
    A Lebesgue space created with direct callable functions.
    This tests the immediate setup case.
    """
    # Use simple polynomial basis: 1, x, x^2, x^3, x^4
    callables = [
        lambda x: np.ones_like(x),           # constant
        lambda x: x,                         # linear
        lambda x: x**2,                      # quadratic
        lambda x: x**3,                      # cubic
        lambda x: x**4,                      # quartic
    ]
    return Lebesgue(5, interval_domain, basis=callables)


@pytest.fixture(scope="module")
def lebesgue_space_provider(interval_domain: IntervalDomain) -> Lebesgue:
    """
    A Lebesgue space created using the typical BasisProvider workflow.
    This tests the basis='none' -> set_basis_provider pattern.
    """
    # Create baseless space
    space = Lebesgue(6, interval_domain, basis='none')

    # Create and set BasisProvider
    f_provider = fp.SineFunctionProvider(space)
    basis_provider = BasisProvider(space, f_provider, basis_type='sine')
    space.set_basis_provider(basis_provider)

    return space


class TestLebesgueSpaceCallables(HilbertSpaceChecks):
    """
    Runs the standard suite of Hilbert space checks on a Lebesgue space
    created with direct callable functions.
    """

    @pytest.fixture
    def space(self, lebesgue_space_callables: Lebesgue) -> HilbertSpace:
        """Provides the callable-based Lebesgue space to the test suite."""
        return lebesgue_space_callables


class TestLebesgueSpaceProvider(HilbertSpaceChecks):
    """
    Runs the standard suite of Hilbert space checks on a Lebesgue space
    created using the BasisProvider workflow.
    """

    @pytest.fixture
    def space(self, lebesgue_space_provider: Lebesgue) -> HilbertSpace:
        """Provides the BasisProvider-based Lebesgue space to the test suite."""
        return lebesgue_space_provider


class TestLebesgueSpaceSpecific:
    """
    Additional tests specific to the Lebesgue space implementation.
    These tests check functionality beyond the standard HilbertSpace interface.
    """

    def test_baseless_space_creation(self, interval_domain: IntervalDomain):
        """Test that baseless spaces can be created and configured."""
        space = Lebesgue(5, interval_domain, basis='none')
        assert space._basis_type == 'none'
        assert space.basis_provider is None
        assert not space._use_basis_provider

        # Should not be able to access basis functions
        with pytest.raises(RuntimeError, match="No basis functions available"):
            space.basis_functions

        with pytest.raises(RuntimeError, match="No basis functions available"):
            space.get_basis_function(0)

    def test_invalid_basis_types(self, interval_domain: IntervalDomain):
        """Test that invalid basis types raise appropriate errors."""
        # Valid string basis types should work now
        space = Lebesgue(5, interval_domain, basis='fourier')
        assert space._basis_type == 'fourier'
        assert space.dim == 5

        # Invalid string basis type should raise ValueError
        with pytest.raises(ValueError, match="Unsupported basis type"):
            Lebesgue(5, interval_domain, basis='invalid_basis')

        # Wrong type entirely
        with pytest.raises(TypeError, match="must be a string or list"):
            Lebesgue(5, interval_domain, basis=42)

    def test_callable_dimension_mismatch(self, interval_domain: IntervalDomain):
        """Test that callable list must match dimension."""
        callables = [lambda x: 1, lambda x: x]  # Only 2 functions
        with pytest.raises(ValueError, match="must match dimension"):
            Lebesgue(5, interval_domain, basis=callables)  # Expects 5

    def test_set_basis_provider_workflow(self, interval_domain: IntervalDomain):
        """Test the complete basis='none' -> set_basis_provider workflow."""
        # Start with baseless space
        space = Lebesgue(4, interval_domain, basis='none')
        assert space._basis_type == 'none'

        # Create and set provider
        f_provider = fp.SineFunctionProvider(space)
        basis_provider = BasisProvider(space, f_provider, basis_type='sine')
        space.set_basis_provider(basis_provider)

        # Now should work
        assert space._basis_type == 'sine'
        assert space._use_basis_provider
        assert space.basis_provider is not None
        assert len(space.basis_functions) == 4

        # Should be able to get individual functions
        func = space.get_basis_function(0)
        assert func is not None

        # Should be able to evaluate functions
        value = func.evaluate(0.5)
        assert isinstance(value, (int, float, np.number))

    def test_function_evaluation_consistency(self, lebesgue_space_callables: Lebesgue):
        """Test that Function objects evaluate consistently."""
        space = lebesgue_space_callables

        # Test polynomial basis evaluations
        x_test = 0.5

        # φ₀(x) = 1
        func0 = space.get_basis_function(0)
        assert np.isclose(func0.evaluate(x_test), 1.0)

        # φ₁(x) = x
        func1 = space.get_basis_function(1)
        assert np.isclose(func1.evaluate(x_test), x_test)

        # φ₂(x) = x²
        func2 = space.get_basis_function(2)
        assert np.isclose(func2.evaluate(x_test), x_test**2)

    def test_space_equality(self, interval_domain: IntervalDomain):
        """Test space equality comparison."""
        # Same domain and dimension should be equal
        space1 = Lebesgue(5, interval_domain, basis='none')
        space2 = Lebesgue(5, interval_domain, basis='none')
        assert space1 == space2

        # Different dimensions should not be equal
        space3 = Lebesgue(6, interval_domain, basis='none')
        assert space1 != space3

        # Different domains should not be equal
        domain2 = IntervalDomain(0, 2)
        space4 = Lebesgue(5, domain2, basis='none')
        assert space1 != space4
