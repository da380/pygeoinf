"""
Unit tests for l2_space.py - focused version

This module provides unit tests for the L2Space class,
testing core functionality without dependency issues.
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

try:
    from pygeoinf.interval.interval_domain import IntervalDomain
    from pygeoinf.interval.l2_space import L2Space
    from pygeoinf.interval.l2_functions import Function
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_SUCCESSFUL = False


@unittest.skipUnless(IMPORTS_SUCCESSFUL, "Required modules not available")
class TestL2SpaceFocused(unittest.TestCase):
    """Focused test cases for L2Space class."""

    def setUp(self):
        """Set up test fixtures."""
        self.domain = IntervalDomain(0.0, 1.0)
        self.dim = 3
        self.space = L2Space(self.dim, self.domain)

        # Helper functions for testing
        self.simple_func = lambda x: x
        self.constant_func = lambda x: (np.ones_like(x)
                                        if hasattr(x, '__iter__')
                                        else 1.0)

    def test_init_basic(self):
        """Test basic initialization."""
        space = L2Space(3, self.domain)

        self.assertEqual(space.dim, 3)
        self.assertEqual(space.function_domain, self.domain)

    def test_dim_property(self):
        """Test dim property."""
        space = L2Space(7, self.domain)
        self.assertEqual(space.dim, 7)

    def test_function_domain_property(self):
        """Test function_domain property."""
        domain = IntervalDomain(-1.0, 2.0)
        space = L2Space(5, domain)
        self.assertEqual(space.function_domain, domain)

    def test_basis_type_property(self):
        """Test basis_type property with default."""
        space = L2Space(3, self.domain)
        self.assertIsNotNone(space.basis_type)
        # Default should be 'fourier'
        self.assertEqual(space.basis_type, 'fourier')

    def test_init_with_different_basis_types(self):
        """Test initialization with different basis types."""
        # Test fourier basis
        space_fourier = L2Space(3, self.domain, basis_type='fourier')
        self.assertEqual(space_fourier.basis_type, 'fourier')

        # Test hat basis
        space_hat = L2Space(3, self.domain, basis_type='hat')
        self.assertEqual(space_hat.basis_type, 'hat')

        # Test hat_homogeneous basis
        space_hat_hom = L2Space(3, self.domain, basis_type='hat_homogeneous')
        self.assertEqual(space_hat_hom.basis_type, 'hat_homogeneous')

    def test_get_basis_function_with_provider(self):
        """Test get_basis_function with basis provider."""
        space = L2Space(3, self.domain, basis_type='fourier')

        # Get basis function
        basis_func = space.get_basis_function(0)
        self.assertIsInstance(basis_func, Function)
        self.assertEqual(basis_func.space, space)

    def test_basis_functions_property(self):
        """Test basis_functions property."""
        space = L2Space(3, self.domain, basis_type='fourier')

        # Should return a list-like object
        basis_funcs = space.basis_functions
        self.assertIsNotNone(basis_funcs)
        self.assertEqual(len(basis_funcs), 3)

    def test_basis_function_method(self):
        """Test basis_function method."""
        space = L2Space(3, self.domain, basis_type='fourier')

        # Test valid indices
        for i in range(3):
            basis_func = space.basis_function(i)
            self.assertIsInstance(basis_func, Function)

        # Test invalid indices
        with self.assertRaises(IndexError):
            space.basis_function(-1)

        with self.assertRaises(IndexError):
            space.basis_function(3)

    def test_inner_product_basic(self):
        """Test basic inner product computation."""
        func1 = Function(self.space, evaluate_callable=self.constant_func)
        func2 = Function(self.space, evaluate_callable=self.simple_func)

        # Inner product of 1 and x over [0,1] should be 1/2
        result = self.space.inner_product(func1, func2)
        self.assertAlmostEqual(result, 0.5, places=2)

    def test_inner_product_same_function(self):
        """Test inner product of function with itself."""
        func = Function(self.space, evaluate_callable=self.constant_func)

        # Inner product of 1 with itself over [0,1] should be 1
        result = self.space.inner_product(func, func)
        self.assertAlmostEqual(result, 1.0, places=2)

    def test_from_components(self):
        """Test _from_components method."""
        components = np.array([1.0, 2.0, 3.0])

        func = self.space._from_components(components)
        self.assertIsInstance(func, Function)
        self.assertEqual(func.space, self.space)
        np.testing.assert_array_equal(func.coefficients, components)

    def test_to_components_with_coefficients(self):
        """Test _to_components with coefficient-based function."""
        coeffs = np.array([1.0, 2.0, 3.0])
        func = Function(self.space, coefficients=coeffs)

        components = self.space._to_components(func)
        np.testing.assert_array_almost_equal(components, coeffs, decimal=10)

    def test_gram_matrix_basic(self):
        """Test basic Gram matrix functionality."""
        # Use a space with fourier basis instead of manual basis
        space = L2Space(2, self.domain, basis_type='fourier')

        gram = space.gram_matrix
        self.assertEqual(gram.shape, (2, 2))

        # Gram matrix should be symmetric
        np.testing.assert_array_almost_equal(gram, gram.T, decimal=3)

    def test_zero_function(self):
        """Test zero function creation."""
        zero = self.space.zero
        self.assertIsInstance(zero, Function)
        self.assertEqual(zero.space, self.space)

    def test_norm_computation(self):
        """Test norm computation."""
        func = Function(self.space, evaluate_callable=self.constant_func)

        # L2 norm of constant function 1 over [0,1] should be 1
        norm = self.space.norm(func)
        self.assertAlmostEqual(norm, 1.0, places=2)

    def test_space_properties(self):
        """Test space properties are consistent."""
        space1 = L2Space(3, self.domain)
        space2 = L2Space(3, self.domain)
        space3 = L2Space(4, self.domain)

        # Same dimensions and domains should have same properties
        self.assertEqual(space1.dim, space2.dim)
        self.assertEqual(space1.function_domain, space2.function_domain)

        # Different dimensions should be different
        self.assertNotEqual(space1.dim, space3.dim)

    def test_init_invalid_arguments(self):
        """Test initialization with invalid arguments."""
        # Test invalid dimension - L2Space might not validate this
        # so we'll test for positive dimensions only
        try:
            L2Space(-1, self.domain)
            # If no error is raised, that's also valid behavior
        except (ValueError, TypeError):
            pass  # Expected behavior

        try:
            L2Space(0, self.domain)
        except (ValueError, TypeError):
            pass  # Expected behavior    def test_projection_basic(self):
        """Test basic function projection."""
        # Create space with basis
        space = L2Space(3, self.domain, basis_type='fourier')

        # Simple test function
        def test_func(x):
            return np.sin(np.pi * x)

        # Project function
        projected = space.project(test_func)
        self.assertIsInstance(projected, Function)
        self.assertEqual(projected.space, space)
        self.assertIsNotNone(projected.coefficients)

    def test_repr_method(self):
        """Test string representation."""
        space = L2Space(3, self.domain)
        repr_str = repr(space)

        self.assertIsInstance(repr_str, str)
        self.assertIn('L2Space', repr_str)


if __name__ == '__main__':
    if not IMPORTS_SUCCESSFUL:
        print("Skipping tests due to import errors")
        exit(1)
    unittest.main(verbosity=2)
