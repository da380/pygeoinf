"""
Unit tests for l2_functions.py

This module provides comprehensive unit tests for the Function class,
testing all methods, properties, edge cases, and error conditions.
"""

import unittest
import numpy as np
from unittest.mock import patch
import sys
import os

# Add the parent directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from pygeoinf.interval.functions import Function
from pygeoinf.interval.interval_domain import IntervalDomain
from pygeoinf.interval import L2Space


class TestFunction(unittest.TestCase):
    """Test cases for Function class."""

    def setUp(self):
        """Set up test fixtures."""
        self.domain = IntervalDomain(0.0, 1.0)
        self.space = L2Space(5, self.domain)

        # Test functions
        self.coeffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        def simple_func(x):
            return x**2

        def constant_func(x):
            return np.ones_like(x) if hasattr(x, '__len__') else 1.0

        self.simple_func = simple_func
        self.constant_func = constant_func

    def test_init_with_coefficients(self):
        """Test initialization with coefficient representation."""
        func = Function(self.space, coefficients=self.coeffs)

        np.testing.assert_array_equal(func.coefficients, self.coeffs)
        self.assertEqual(func.space, self.space)
        self.assertIsNone(func.evaluate_callable)
        self.assertIsNone(func.support)

    def test_init_with_callable(self):
        """Test initialization with callable representation."""
        func = Function(self.space, evaluate_callable=self.simple_func)

        self.assertIsNone(func.coefficients)
        self.assertEqual(func.space, self.space)
        self.assertEqual(func.evaluate_callable, self.simple_func)
        self.assertIsNone(func.support)

    def test_init_with_name(self):
        """Test initialization with custom name."""
        func = Function(
            self.space,
            evaluate_callable=self.simple_func,
            name="test_function"
        )

        self.assertEqual(func.name, "test_function")

    def test_init_with_single_support(self):
        """Test initialization with single interval support."""
        support = (0.2, 0.8)
        func = Function(
            self.space,
            evaluate_callable=self.simple_func,
            support=support
        )

        self.assertEqual(func.support, [support])
        self.assertTrue(func.has_compact_support)

    def test_init_with_multiple_support(self):
        """Test initialization with multiple interval support."""
        support = [(0.1, 0.3), (0.6, 0.9)]
        func = Function(
            self.space,
            evaluate_callable=self.simple_func,
            support=support
        )

        self.assertEqual(func.support, support)
        self.assertTrue(func.has_compact_support)

    def test_init_invalid_arguments(self):
        """Test initialization with invalid arguments."""
        # No representation provided
        with self.assertRaises(ValueError):
            Function(self.space)

        # Both representations provided
        with self.assertRaises(ValueError):
            Function(
                self.space,
                coefficients=self.coeffs,
                evaluate_callable=self.simple_func
            )

    def test_init_invalid_support(self):
        """Test initialization with invalid support specification."""
        # Invalid tuple
        with self.assertRaises(ValueError):
            Function(
                self.space,
                evaluate_callable=self.simple_func,
                support=(0.8, 0.2)  # a >= b
            )

        # Invalid list element
        with self.assertRaises(ValueError):
            Function(
                self.space,
                evaluate_callable=self.simple_func,
                support=[(0.1, 0.3), (0.2, 0.4)]  # overlapping
            )

        # Support outside domain
        with self.assertRaises(ValueError):
            Function(
                self.space,
                evaluate_callable=self.simple_func,
                support=(1.5, 2.0)  # outside [0, 1]
            )

    def test_function_domain_property(self):
        """Test function_domain property."""
        func = Function(self.space, coefficients=self.coeffs)
        self.assertEqual(func.function_domain, self.domain)

    def test_has_compact_support_property(self):
        """Test has_compact_support property."""
        # Function without compact support
        func1 = Function(self.space, coefficients=self.coeffs)
        self.assertFalse(func1.has_compact_support)

        # Function with compact support
        func2 = Function(
            self.space,
            evaluate_callable=self.simple_func,
            support=(0.2, 0.8)
        )
        self.assertTrue(func2.has_compact_support)

    def test_union_supports_static_method(self):
        """Test _union_supports static method."""
        # Both None
        result = Function._union_supports(None, None)
        self.assertIsNone(result)

        # One None
        support1 = [(0.1, 0.3)]
        result = Function._union_supports(support1, None)
        self.assertEqual(result, support1)

        result = Function._union_supports(None, support1)
        self.assertEqual(result, support1)

        # Disjoint intervals
        support2 = [(0.6, 0.9)]
        result = Function._union_supports(support1, support2)
        expected = [(0.1, 0.3), (0.6, 0.9)]
        self.assertEqual(result, expected)

        # Overlapping intervals
        support3 = [(0.2, 0.7)]
        result = Function._union_supports(support1, support3)
        expected = [(0.1, 0.7)]
        self.assertEqual(result, expected)

    def test_intersect_supports_static_method(self):
        """Test _intersect_supports static method."""
        # Both None
        result = Function._intersect_supports(None, None)
        self.assertIsNone(result)

        # One None
        support1 = [(0.1, 0.3)]
        result = Function._intersect_supports(support1, None)
        self.assertIsNone(result)

        # Disjoint intervals
        support2 = [(0.6, 0.9)]
        result = Function._intersect_supports(support1, support2)
        self.assertEqual(result, [])

        # Overlapping intervals
        support3 = [(0.2, 0.7)]
        result = Function._intersect_supports(support1, support3)
        expected = [(0.2, 0.3)]
        self.assertEqual(result, expected)

    def test_is_zero_at(self):
        """Test is_zero_at method."""
        # Function without compact support
        func1 = Function(self.space, evaluate_callable=self.simple_func)
        self.assertFalse(func1._is_zero_at(0.5))

        # Function with compact support
        func2 = Function(
            self.space,
            evaluate_callable=self.simple_func,
            support=(0.2, 0.8)
        )
        self.assertTrue(func2._is_zero_at(0.1))  # outside support
        self.assertFalse(func2._is_zero_at(0.5))  # inside support

    def test_evaluate_with_callable(self):
        """Test evaluate method with callable representation."""
        func = Function(self.space, evaluate_callable=self.simple_func)

        # Single point
        result = func.evaluate(0.5)
        self.assertEqual(result, 0.25)

        # Array of points
        x = np.array([0.0, 0.5, 1.0])
        result = func.evaluate(x)
        expected = np.array([0.0, 0.25, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_evaluate_with_compact_support(self):
        """Test evaluate method with compact support."""
        func = Function(
            self.space,
            evaluate_callable=self.simple_func,
            support=(0.2, 0.8)
        )

        # Points outside support should be zero
        self.assertEqual(func.evaluate(0.1), 0.0)
        self.assertEqual(func.evaluate(0.9), 0.0)

        # Points inside support should use callable
        self.assertEqual(func.evaluate(0.5), 0.25)

    def test_evaluate_with_coefficients(self):
        """Test evaluate method with coefficient representation."""
        func = Function(self.space, coefficients=self.coeffs)

        # This should call _evaluate_from_coefficients
        result = func.evaluate(0.5)
        self.assertIsInstance(result, (float, np.floating))

    def test_call_method(self):
        """Test __call__ method (should behave like evaluate)."""
        func = Function(self.space, evaluate_callable=self.simple_func)

        # Should be equivalent to evaluate
        self.assertEqual(func(0.5), func.evaluate(0.5))

    def test_integrate_method(self):
        """Test integrate method."""
        func = Function(self.space, evaluate_callable=self.constant_func)

        # Integrate constant function over [0, 1] should give 1
        result = func.integrate()
        self.assertAlmostEqual(result, 1.0, places=3)

    def test_integrate_with_compact_support(self):
        """Test integrate method with compact support."""
        func = Function(
            self.space,
            evaluate_callable=self.constant_func,
            support=(0.2, 0.8)
        )

        # Integrate constant function over [0.2, 0.8] should give 0.6
        result = func.integrate()
        self.assertAlmostEqual(result, 0.6, places=3)

    def test_plot_method(self):
        """Test plot method."""
        func = Function(self.space, evaluate_callable=self.simple_func)

        with patch('matplotlib.pyplot.show') as mock_show:
            with patch('matplotlib.pyplot.plot') as mock_plot:
                func.plot()
                mock_plot.assert_called_once()

    def test_addition_with_function(self):
        """Test addition with another function."""
        func1 = Function(self.space, evaluate_callable=self.simple_func)
        func2 = Function(self.space, evaluate_callable=self.constant_func)

        result = func1 + func2

        self.assertIsInstance(result, Function)
        self.assertEqual(result.space, self.space)

        # Test evaluation
        x = 0.5
        expected = self.simple_func(x) + self.constant_func(x)
        self.assertAlmostEqual(result.evaluate(x), expected)

    def test_addition_with_scalar(self):
        """Test addition with scalar."""
        func = Function(self.space, evaluate_callable=self.simple_func)
        scalar = 2.0

        result = func + scalar

        self.assertIsInstance(result, Function)

        # Test evaluation
        x = 0.5
        expected = self.simple_func(x) + scalar
        self.assertAlmostEqual(result.evaluate(x), expected)

    def test_right_addition_with_scalar(self):
        """Test right addition with scalar."""
        func = Function(self.space, evaluate_callable=self.simple_func)
        scalar = 2.0

        result = scalar + func

        self.assertIsInstance(result, Function)

        # Test evaluation
        x = 0.5
        expected = scalar + self.simple_func(x)
        self.assertAlmostEqual(result.evaluate(x), expected)

    def test_subtraction_with_function(self):
        """Test subtraction with another function."""
        func1 = Function(self.space, evaluate_callable=self.simple_func)
        func2 = Function(self.space, evaluate_callable=self.constant_func)

        result = func1 - func2

        self.assertIsInstance(result, Function)

        # Test evaluation
        x = 0.5
        expected = self.simple_func(x) - self.constant_func(x)
        self.assertAlmostEqual(result.evaluate(x), expected)

    def test_subtraction_with_scalar(self):
        """Test subtraction with scalar."""
        func = Function(self.space, evaluate_callable=self.simple_func)
        scalar = 1.0

        result = func - scalar

        self.assertIsInstance(result, Function)

        # Test evaluation
        x = 0.5
        expected = self.simple_func(x) - scalar
        self.assertAlmostEqual(result.evaluate(x), expected)

    def test_multiplication_with_function(self):
        """Test multiplication with another function."""
        func1 = Function(self.space, evaluate_callable=self.simple_func)
        func2 = Function(self.space, evaluate_callable=self.constant_func)

        result = func1 * func2

        self.assertIsInstance(result, Function)

        # Test evaluation
        x = 0.5
        expected = self.simple_func(x) * self.constant_func(x)
        self.assertAlmostEqual(result.evaluate(x), expected)

    def test_multiplication_with_scalar(self):
        """Test multiplication with scalar."""
        func = Function(self.space, evaluate_callable=self.simple_func)
        scalar = 3.0

        result = func * scalar

        self.assertIsInstance(result, Function)

        # Test evaluation
        x = 0.5
        expected = self.simple_func(x) * scalar
        self.assertAlmostEqual(result.evaluate(x), expected)

    def test_right_multiplication_with_scalar(self):
        """Test right multiplication with scalar."""
        func = Function(self.space, evaluate_callable=self.simple_func)
        scalar = 3.0

        result = scalar * func

        self.assertIsInstance(result, Function)

        # Test evaluation
        x = 0.5
        expected = scalar * self.simple_func(x)
        self.assertAlmostEqual(result.evaluate(x), expected)

    def test_multiplication_with_zero(self):
        """Test multiplication with zero scalar."""
        func = Function(self.space, evaluate_callable=self.simple_func)

        result = func * 0

        self.assertIsInstance(result, Function)

        # Should be zero everywhere
        self.assertEqual(result.evaluate(0.5), 0.0)

    def test_repr_method(self):
        """Test string representation."""
        func = Function(
            self.space,
            evaluate_callable=self.simple_func,
            name="test_func"
        )

        repr_str = repr(func)
        self.assertIn("test_func", repr_str)

    def test_copy_method(self):
        """Test copy method."""
        func = Function(
            self.space,
            evaluate_callable=self.simple_func,
            name="original"
        )

        func_copy = func.copy()

        self.assertIsNot(func, func_copy)
        self.assertEqual(func.space, func_copy.space)
        self.assertEqual(func.name, func_copy.name)
        self.assertEqual(func.evaluate_callable, func_copy.evaluate_callable)

    def test_copy_with_coefficients(self):
        """Test copy method with coefficient representation."""
        func = Function(self.space, coefficients=self.coeffs, name="original")

        func_copy = func.copy()

        # Should have independent coefficient arrays
        np.testing.assert_array_equal(func.coefficients, func_copy.coefficients)
        self.assertIsNot(func.coefficients, func_copy.coefficients)

    def test_support_validation_edge_cases(self):
        """Test edge cases in support validation."""
        # Empty list (should be allowed)
        func = Function(
            self.space,
            evaluate_callable=self.simple_func,
            support=[]
        )
        self.assertEqual(func.support, [])

        # Single point interval (should fail)
        with self.assertRaises(ValueError):
            Function(
                self.space,
                evaluate_callable=self.simple_func,
                support=(0.5, 0.5)
            )

    def test_arithmetic_with_incompatible_spaces(self):
        """Test arithmetic operations with functions from different spaces."""
        other_domain = IntervalDomain(1.0, 2.0)
        other_space = L2Space(3, other_domain)

        func1 = Function(self.space, evaluate_callable=self.simple_func)
        func2 = Function(other_space, evaluate_callable=self.constant_func)

        # Should raise error for incompatible spaces
        with self.assertRaises(ValueError):
            func1 + func2

    def test_edge_cases_with_numpy_scalars(self):
        """Test edge cases with numpy scalar types."""
        func = Function(self.space, evaluate_callable=self.simple_func)

        # Test with numpy scalar
        np_scalar = np.float64(2.0)
        result = func * np_scalar

        self.assertIsInstance(result, Function)
        self.assertAlmostEqual(result.evaluate(0.5), 0.5)


if __name__ == '__main__':
    unittest.main()
