"""
Unit tests for l2_space.py

This module provides comprehensive unit tests for the L2Space class,
testing all methods, properties, edge cases, and error conditions.
"""

import unittest
import numpy as np
from unittest.mock import Mock
import sys
import os

# Add the parent directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from pygeoinf.interval.l2_space import L2Space
from pygeoinf.interval.interval_domain import IntervalDomain
from pygeoinf.interval.l2_functions import Function


class TestL2Space(unittest.TestCase):
    """Test cases for L2Space class."""

    def setUp(self):
        """Set up test fixtures."""
        self.domain = IntervalDomain(0.0, 1.0)
        self.dim = 5
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
        self.assertIsNone(space._gram_matrix)

    def test_init_with_fourier_basis(self):
        """Test initialization with Fourier basis."""
        space = L2Space(4, self.domain, basis_type='fourier')

        self.assertEqual(space.dim, 4)
        self.assertEqual(space._basis_type, 'fourier')
        self.assertIsNotNone(space._basis_provider)

    def test_init_with_hat_basis(self):
        """Test initialization with hat basis."""
        space = L2Space(3, self.domain, basis_type='hat')

        self.assertEqual(space.dim, 3)
        self.assertEqual(space._basis_type, 'hat')
        self.assertIsNotNone(space._basis_provider)

    def test_init_with_hat_homogeneous_basis(self):
        """Test initialization with hat homogeneous basis."""
        space = L2Space(4, self.domain, basis_type='hat_homogeneous')

        self.assertEqual(space.dim, 4)
        self.assertEqual(space._basis_type, 'hat_homogeneous')
        self.assertIsNotNone(space._basis_provider)

    def test_init_with_basis_callables(self):
        """Test initialization with explicit basis callables."""
        basis_funcs = [
            lambda x: (np.ones_like(x) if hasattr(x, '__iter__') else 1.0),
            lambda x: x,
            lambda x: x**2
        ]
        space = L2Space(3, self.domain, basis_callables=basis_funcs)

        self.assertEqual(space.dim, 3)
        self.assertIsNotNone(space._manual_basis_functions)
        self.assertEqual(len(space._manual_basis_functions), 3)

    def test_init_with_basis_provider(self):
        """Test initialization with custom basis provider."""
        # Create a mock basis provider
        mock_provider = Mock()
        space = L2Space(3, self.domain, basis_provider=mock_provider)

        self.assertEqual(space.dim, 3)
        self.assertEqual(space._basis_provider, mock_provider)

    def test_dim_property(self):
        """Test dim property."""
        space = L2Space(7, self.domain)
        self.assertEqual(space.dim, 7)

    def test_function_domain_property(self):
        """Test function_domain property."""
        domain = IntervalDomain(-1.0, 2.0)
        space = L2Space(3, domain)
        self.assertEqual(space.function_domain, domain)

    def test_get_basis_function_with_provider(self):
        """Test get_basis_function with basis provider."""
        space = L2Space(3, self.domain, basis_type='fourier')

        # Get basis function
        basis_func = space.get_basis_function(0)
        self.assertIsInstance(basis_func, Function)
        self.assertEqual(basis_func.space, space)

    def test_get_basis_function_with_manual_basis(self):
        """Test get_basis_function with manual basis functions."""
        basis_funcs = [
            lambda x: (np.ones_like(x) if hasattr(x, '__iter__') else 1.0),
            lambda x: x
        ]
        space = L2Space(2, self.domain, basis_callables=basis_funcs)

        # Test getting basis functions
        basis_func_0 = space.get_basis_function(0)
        basis_func_1 = space.get_basis_function(1)

        self.assertIsInstance(basis_func_0, Function)
        self.assertIsInstance(basis_func_1, Function)

    def test_get_basis_function_error(self):
        """Test get_basis_function raises error when no basis available."""
        # Create space without basis
        space = L2Space.__new__(L2Space)
        space._dim = 3
        space._function_domain = self.domain
        space._basis_functions = None
        space._basis_provider = None

        with self.assertRaises(RuntimeError):
            space.get_basis_function(0)

    def test_basis_type_property(self):
        """Test basis_type property."""
        space = L2Space(3, self.domain, basis_type='fourier')
        self.assertEqual(space.basis_type, 'fourier')

    def test_inner_product_basic(self):
        """Test basic inner product computation."""
        func1 = Function(self.space, evaluate_callable=self.constant_func)
        func2 = Function(self.space, evaluate_callable=self.simple_func)

        # Inner product of 1 and x over [0,1] should be 1/2
        result = self.space.inner_product(func1, func2)
        self.assertAlmostEqual(result, 0.5, places=3)

    def test_inner_product_same_function(self):
        """Test inner product of function with itself."""
        func = Function(self.space, evaluate_callable=self.constant_func)

        # Inner product of 1 with itself over [0,1] should be 1
        result = self.space.inner_product(func, func)
        self.assertAlmostEqual(result, 1.0, places=3)

    def test_inner_product_with_coefficients(self):
        """Test inner product with coefficient-based functions."""
        # Create space with basis
        space = L2Space(3, self.domain, basis_type='fourier')

        func1 = Function(space, coefficients=np.array([1.0, 0.0, 0.0]))
        func2 = Function(space, coefficients=np.array([0.0, 1.0, 0.0]))

        # Different basis functions should have small inner product
        result = space.inner_product(func1, func2)
        self.assertAlmostEqual(result, 0.0, places=1)

    def test_to_components_with_coefficients(self):
        """Test _to_components with coefficient-based function."""
        coeffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        func = Function(self.space, coefficients=coeffs)

        components = self.space._to_components(func)
        np.testing.assert_array_equal(components, coeffs)

    def test_to_components_with_callable(self):
        """Test _to_components with callable function."""
        # Create space with basis for projection
        space = L2Space(3, self.domain, basis_type='fourier')
        func = Function(space, evaluate_callable=self.constant_func)

        components = space._to_components(func)
        self.assertIsInstance(components, np.ndarray)
        self.assertEqual(len(components), 3)

    def test_from_components(self):
        """Test _from_components method."""
        components = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        func = self.space._from_components(components)
        self.assertIsInstance(func, Function)
        self.assertEqual(func.space, self.space)
        np.testing.assert_array_equal(func.coefficients, components)

    def test_default_to_dual(self):
        """Test _default_to_dual method."""
        coeffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        func = Function(self.space, coefficients=coeffs)

        # For L2Space, to_dual should return the same coefficients
        dual_func = self.space._default_to_dual(func)
        self.assertIsInstance(dual_func, Function)
        np.testing.assert_array_equal(dual_func.coefficients,
                                      func.coefficients)

    def test_default_from_dual(self):
        """Test _default_from_dual method."""
        coeffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        func = Function(self.space, coefficients=coeffs)

        # For L2Space, from_dual should return the same coefficients
        primal_func = self.space._default_from_dual(func)
        self.assertIsInstance(primal_func, Function)
        np.testing.assert_array_equal(primal_func.coefficients,
                                      func.coefficients)

    def test_copy_method(self):
        """Test _copy method."""
        coeffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        func = Function(self.space, coefficients=coeffs)

        copied_func = self.space._copy(func)
        self.assertIsInstance(copied_func, Function)
        self.assertIsNot(copied_func, func)
        np.testing.assert_array_equal(copied_func.coefficients,
                                      func.coefficients)

    def test_gram_matrix_computation(self):
        """Test Gram matrix computation."""
        # Create space with explicit basis
        basis_funcs = [
            lambda x: (np.ones_like(x) if hasattr(x, '__iter__') else 1.0),
            lambda x: x
        ]
        space = L2Space(2, self.domain, basis_callables=basis_funcs)

        gram = space.gram_matrix
        self.assertEqual(gram.shape, (2, 2))

        # Gram matrix should be symmetric
        np.testing.assert_array_almost_equal(gram, gram.T)

        # Gram matrix should be positive definite
        eigenvals = np.linalg.eigvals(gram)
        self.assertTrue(np.all(eigenvals > 0))

    def test_gram_matrix_caching(self):
        """Test that Gram matrix is cached."""
        basis_funcs = [
            lambda x: (np.ones_like(x) if hasattr(x, '__iter__') else 1.0)
        ]
        space = L2Space(1, self.domain, basis_callables=basis_funcs)

        # First call should compute and cache
        gram1 = space.gram_matrix
        self.assertIsNotNone(space._gram_matrix)

        # Second call should return cached version
        gram2 = space.gram_matrix
        self.assertIs(gram1, gram2)

    def test_norm_computation(self):
        """Test norm computation."""
        func = Function(self.space, evaluate_callable=self.constant_func)

        # L2 norm of constant function 1 over [0,1] should be 1
        norm = self.space.norm(func)
        self.assertAlmostEqual(norm, 1.0, places=3)

    def test_distance_computation(self):
        """Test distance computation."""
        func1 = Function(self.space, evaluate_callable=self.constant_func)
        const2_func = lambda x: (2 * np.ones_like(x)
                                 if hasattr(x, '__iter__') else 2.0)
        func2 = Function(self.space, evaluate_callable=const2_func)

        # Distance between constant functions 1 and 2 should be 1
        distance = self.space.distance(func1, func2)
        self.assertAlmostEqual(distance, 1.0, places=3)

    def test_zero_function(self):
        """Test zero function creation."""
        zero = self.space.zero()
        self.assertIsInstance(zero, Function)
        self.assertEqual(zero.space, self.space)

        # Zero function should have zero coefficients
        if zero.coefficients is not None:
            np.testing.assert_array_equal(zero.coefficients,
                                          np.zeros(self.dim))

    def test_space_equality(self):
        """Test space equality comparison."""
        space1 = L2Space(3, self.domain)
        space2 = L2Space(3, self.domain)
        space3 = L2Space(4, self.domain)
        space4 = L2Space(3, IntervalDomain(0.0, 2.0))

        self.assertEqual(space1, space2)
        self.assertNotEqual(space1, space3)
        self.assertNotEqual(space1, space4)

    def test_projection_basic(self):
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

    def test_projection_consistency(self):
        """Test that projection is consistent."""
        # Create space with basis
        space = L2Space(3, self.domain, basis_type='fourier')
        func = Function(space, evaluate_callable=self.simple_func)

        # Project function to coefficients and back
        coeffs = space._to_components(func)
        reconstructed = space._from_components(coeffs)

        # Check that reconstruction preserves essential properties
        self.assertEqual(reconstructed.space, space)
        self.assertIsNotNone(reconstructed.coefficients)

    def test_init_invalid_arguments(self):
        """Test initialization with invalid arguments."""
        # Test invalid dimension
        with self.assertRaises((ValueError, TypeError)):
            L2Space(-1, self.domain)

        with self.assertRaises((ValueError, TypeError)):
            L2Space(0, self.domain)

    def test_init_with_multiple_basis_specifications(self):
        """Test initialization with conflicting basis specifications."""
        basis_funcs = [lambda x: x]

        # Should be able to specify multiple basis types
        # The exact behavior depends on implementation
        try:
            space = L2Space(
                1, self.domain,
                basis_type='fourier',
                basis_callables=basis_funcs
            )
            # If this succeeds, check that one of the basis types is used
            self.assertTrue(
                space._basis_type == 'fourier' or
                space._manual_basis_functions is not None
            )
        except (ValueError, TypeError):
            # If it raises an error, that's also acceptable behavior
            pass

    def test_edge_cases_with_small_domain(self):
        """Test edge cases with very small domains."""
        small_domain = IntervalDomain(0.0, 1e-10)
        space = L2Space(2, small_domain)

        self.assertEqual(space.function_domain, small_domain)
        self.assertEqual(space.dim, 2)

    def test_get_basis_function_out_of_bounds(self):
        """Test get_basis_function with out of bounds index."""
        space = L2Space(3, self.domain, basis_type='fourier')

        # This might raise an IndexError depending on implementation
        try:
            result = space.get_basis_function(10)
            # If it doesn't raise an error, result should be a Function
            if result is not None:
                self.assertIsInstance(result, Function)
        except (IndexError, KeyError):
            # IndexError or KeyError is acceptable behavior
            pass

    def test_inner_product_with_zero_functions(self):
        """Test inner product with zero functions."""
        zero1 = self.space.zero()
        zero2 = self.space.zero()
        func = Function(self.space, evaluate_callable=self.constant_func)

        # Inner product of zero functions should be zero
        self.assertAlmostEqual(self.space.inner_product(zero1, zero2),
                               0.0, places=10)
        self.assertAlmostEqual(self.space.inner_product(zero1, func),
                               0.0, places=10)
        self.assertAlmostEqual(self.space.inner_product(func, zero1),
                               0.0, places=10)

    def test_repr_method(self):
        """Test string representation."""
        space = L2Space(3, self.domain)
        repr_str = repr(space)

        self.assertIsInstance(repr_str, str)
        self.assertIn('L2Space', repr_str)
        self.assertIn('3', repr_str)  # dimension
        self.assertIn('[0.0, 1.0]', repr_str)  # domain

    def test_basis_functions_property(self):
        """Test basis_functions property."""
        space = L2Space(3, self.domain, basis_type='fourier')

        # Should return a list-like object
        basis_funcs = space.basis_functions
        self.assertIsNotNone(basis_funcs)
        self.assertEqual(len(basis_funcs), 3)

        # All elements should be Functions
        for i in range(3):
            self.assertIsInstance(basis_funcs[i], Function)

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

    def test_basis_provider_property(self):
        """Test basis_provider property."""
        space = L2Space(3, self.domain, basis_type='fourier')
        provider = space.basis_provider
        self.assertIsNotNone(provider)

    def test_basis_provider_property_error(self):
        """Test basis_provider property when no provider available."""
        # Create space with manual basis functions
        basis_funcs = [lambda x: x]
        space = L2Space(1, self.domain, basis_callables=basis_funcs)

        with self.assertRaises(RuntimeError):
            _ = space.basis_provider


if __name__ == '__main__':
    unittest.main()
