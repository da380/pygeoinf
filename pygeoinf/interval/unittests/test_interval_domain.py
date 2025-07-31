"""
Unit tests for interval_domain.py

This module provides comprehensive unit tests for the IntervalDomain class,
testing all methods, properties, edge cases, and error conditions.
"""

import unittest
import numpy as np
from unittest.mock import patch
import sys
import os

# Add the parent directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from interval_domain import IntervalDomain


class TestIntervalDomain(unittest.TestCase):
    """Test cases for IntervalDomain class."""

    def setUp(self):
        """Set up test fixtures."""
        self.interval_closed = IntervalDomain(0.0, 1.0, boundary_type='closed')
        self.interval_open = IntervalDomain(-1.0, 2.0, boundary_type='open')
        self.interval_left_open = IntervalDomain(
            0.5, 3.5, boundary_type='left_open'
        )
        self.interval_right_open = IntervalDomain(
            -2.0, -0.5, boundary_type='right_open'
        )
        self.interval_named = IntervalDomain(0.0, 1.0, name="unit_interval")

    def test_init_valid_intervals(self):
        """Test initialization with valid parameters."""
        # Test closed interval
        domain = IntervalDomain(0.0, 1.0)
        self.assertEqual(domain.a, 0.0)
        self.assertEqual(domain.b, 1.0)
        self.assertEqual(domain.boundary_type, 'closed')
        self.assertEqual(domain.name, "[0.0, 1.0]")

        # Test with custom name
        domain_named = IntervalDomain(0.0, 1.0, name="test")
        self.assertEqual(domain_named.name, "test")

        # Test different boundary types
        for boundary_type in ['closed', 'open', 'left_open', 'right_open']:
            domain = IntervalDomain(0.0, 1.0, boundary_type=boundary_type)
            self.assertEqual(domain.boundary_type, boundary_type)

    def test_init_invalid_intervals(self):
        """Test initialization with invalid parameters."""
        # Test a >= b
        with self.assertRaises(ValueError):
            IntervalDomain(1.0, 0.0)

        with self.assertRaises(ValueError):
            IntervalDomain(1.0, 1.0)

        # Test negative intervals (should work)
        domain = IntervalDomain(-2.0, -1.0)
        self.assertEqual(domain.a, -2.0)
        self.assertEqual(domain.b, -1.0)

    def test_length_property(self):
        """Test length property (Lebesgue measure)."""
        self.assertEqual(self.interval_closed.length, 1.0)
        self.assertEqual(self.interval_open.length, 3.0)
        self.assertEqual(self.interval_left_open.length, 3.0)
        self.assertEqual(self.interval_right_open.length, 1.5)

    def test_center_property(self):
        """Test center property."""
        self.assertEqual(self.interval_closed.center, 0.5)
        self.assertEqual(self.interval_open.center, 0.5)
        self.assertEqual(self.interval_left_open.center, 2.0)
        self.assertEqual(self.interval_right_open.center, -1.25)

    def test_radius_property(self):
        """Test radius property."""
        self.assertEqual(self.interval_closed.radius, 0.5)
        self.assertEqual(self.interval_open.radius, 1.5)
        self.assertEqual(self.interval_left_open.radius, 1.5)
        self.assertEqual(self.interval_right_open.radius, 0.75)

    def test_contains_scalar(self):
        """Test contains method with scalar values."""
        # Closed interval [0, 1]
        self.assertTrue(self.interval_closed.contains(0.0))
        self.assertTrue(self.interval_closed.contains(0.5))
        self.assertTrue(self.interval_closed.contains(1.0))
        self.assertFalse(self.interval_closed.contains(-0.1))
        self.assertFalse(self.interval_closed.contains(1.1))

        # Open interval (-1, 2)
        self.assertTrue(self.interval_open.contains(0.0))
        self.assertTrue(self.interval_open.contains(1.5))
        self.assertFalse(self.interval_open.contains(-1.0))
        self.assertFalse(self.interval_open.contains(2.0))
        self.assertFalse(self.interval_open.contains(-1.1))
        self.assertFalse(self.interval_open.contains(2.1))

        # Left open interval (0.5, 3.5]
        self.assertTrue(self.interval_left_open.contains(1.0))
        self.assertTrue(self.interval_left_open.contains(3.5))
        self.assertFalse(self.interval_left_open.contains(0.5))
        self.assertFalse(self.interval_left_open.contains(3.6))

        # Right open interval [-2, -0.5)
        self.assertTrue(self.interval_right_open.contains(-2.0))
        self.assertTrue(self.interval_right_open.contains(-1.0))
        self.assertFalse(self.interval_right_open.contains(-0.5))
        self.assertFalse(self.interval_right_open.contains(-2.1))

    def test_contains_array(self):
        """Test contains method with numpy arrays."""
        x = np.array([-0.5, 0.0, 0.5, 1.0, 1.5])

        # Closed interval [0, 1]
        expected_closed = np.array([False, True, True, True, False])
        np.testing.assert_array_equal(
            self.interval_closed.contains(x), expected_closed
        )

        # Open interval (-1, 2)
        expected_open = np.array([True, True, True, True, True])
        np.testing.assert_array_equal(
            self.interval_open.contains(x), expected_open
        )

    def test_contains_invalid_boundary_type(self):
        """Test contains method with invalid boundary type."""
        # Create an interval with invalid boundary type by direct assignment
        interval = IntervalDomain(0.0, 1.0)
        interval.boundary_type = 'invalid'

        with self.assertRaises(ValueError):
            interval.contains(0.5)

    def test_interior(self):
        """Test interior method."""
        interior = self.interval_closed.interior()
        self.assertEqual(interior.a, 0.0)
        self.assertEqual(interior.b, 1.0)
        self.assertEqual(interior.boundary_type, 'open')

        # Test idempotency
        interior_of_interior = interior.interior()
        self.assertEqual(interior_of_interior.boundary_type, 'open')

    def test_closure(self):
        """Test closure method."""
        closure = self.interval_open.closure()
        self.assertEqual(closure.a, -1.0)
        self.assertEqual(closure.b, 2.0)
        self.assertEqual(closure.boundary_type, 'closed')

        # Test idempotency
        closure_of_closure = closure.closure()
        self.assertEqual(closure_of_closure.boundary_type, 'closed')

    def test_boundary_points(self):
        """Test boundary_points method."""
        points = self.interval_closed.boundary_points()
        self.assertEqual(points, (0.0, 1.0))

        points = self.interval_open.boundary_points()
        self.assertEqual(points, (-1.0, 2.0))

    def test_uniform_mesh_closed(self):
        """Test uniform_mesh method for closed intervals."""
        mesh = self.interval_closed.uniform_mesh(5)
        expected = np.linspace(0.0, 1.0, 5)
        np.testing.assert_array_almost_equal(mesh, expected)

    def test_uniform_mesh_open(self):
        """Test uniform_mesh method for open intervals."""
        mesh = self.interval_open.uniform_mesh(3)
        # Should exclude both endpoints
        expected = np.linspace(-1.0, 2.0, 5)[1:-1]  # Remove first and last
        np.testing.assert_array_almost_equal(mesh, expected)

    def test_uniform_mesh_left_open(self):
        """Test uniform_mesh method for left-open intervals."""
        mesh = self.interval_left_open.uniform_mesh(3)
        # Should exclude left endpoint, include right endpoint
        expected = np.linspace(0.5, 3.5, 4)[1:]  # Remove first
        np.testing.assert_array_almost_equal(mesh, expected)

    def test_uniform_mesh_right_open(self):
        """Test uniform_mesh method for right-open intervals."""
        mesh = self.interval_right_open.uniform_mesh(3)
        # Should include left endpoint, exclude right endpoint
        expected = np.linspace(-2.0, -0.5, 3, endpoint=False)
        np.testing.assert_array_almost_equal(mesh, expected)

    def test_uniform_mesh_invalid_boundary_type(self):
        """Test uniform_mesh method with invalid boundary type."""
        interval = IntervalDomain(0.0, 1.0)
        interval.boundary_type = 'invalid'

        with self.assertRaises(ValueError):
            interval.uniform_mesh(5)

    def test_adaptive_mesh(self):
        """Test adaptive_mesh method."""
        # Simple function for testing
        def f(x):
            return x**2

        mesh = self.interval_closed.adaptive_mesh(f, tol=1e-6, max_points=100)

        # Check that all points are in the domain
        self.assertTrue(np.all(self.interval_closed.contains(mesh)))

        # Check that we get a reasonable number of points
        self.assertLessEqual(len(mesh), 100)
        self.assertGreater(len(mesh), 0)

    def test_random_points_closed(self):
        """Test random_points method for closed intervals."""
        np.random.seed(42)  # For reproducibility
        points = self.interval_closed.random_points(100)

        # Check all points are in domain
        self.assertTrue(np.all(self.interval_closed.contains(points)))
        self.assertEqual(len(points), 100)

        # Check bounds
        self.assertTrue(np.all(points >= 0.0))
        self.assertTrue(np.all(points <= 1.0))

    def test_random_points_open(self):
        """Test random_points method for open intervals."""
        np.random.seed(42)
        points = self.interval_open.random_points(100)

        # Check all points are in domain
        self.assertTrue(np.all(self.interval_open.contains(points)))
        self.assertEqual(len(points), 100)

        # Check bounds (should not include exact endpoints)
        self.assertTrue(np.all(points > -1.0))
        self.assertTrue(np.all(points < 2.0))

    def test_random_points_with_seed(self):
        """Test random_points method with seed parameter."""
        points1 = self.interval_closed.random_points(10, seed=42)
        points2 = self.interval_closed.random_points(10, seed=42)

        np.testing.assert_array_almost_equal(points1, points2)

    def test_integrate_full_domain(self):
        """Test integrate method over full domain."""
        # Simple function: f(x) = x
        def f(x):
            return x

        # Analytical result: ∫₀¹ x dx = 1/2
        result = self.interval_closed.integrate(
            f, method='trapz', n_points=1000
        )
        self.assertAlmostEqual(result, 0.5, places=3)

    def test_integrate_single_subinterval(self):
        """Test integrate method over single subinterval."""
        def f(x):
            return x**2

        # Integrate over [0.25, 0.75]
        result = self.interval_closed.integrate(
            f, method='trapz', support=(0.25, 0.75), n_points=1000
        )

        # Analytical result: ∫₀.₂₅^₀.₇₅ x² dx = [x³/3]₀.₂₅^₀.₇₅
        analytical = (0.75**3 - 0.25**3) / 3
        self.assertAlmostEqual(result, analytical, places=3)

    def test_integrate_multiple_subintervals(self):
        """Test integrate method over multiple subintervals."""
        def f(x):  # Constant function
            return 1.0

        # Integrate over [0.1, 0.3] ∪ [0.6, 0.9]
        result = self.interval_closed.integrate(
            f, method='simpson', support=[(0.1, 0.3), (0.6, 0.9)],
            n_points=1000
        )

        # Analytical result: (0.3 - 0.1) + (0.9 - 0.6) = 0.2 + 0.3 = 0.5
        self.assertAlmostEqual(result, 0.5, places=6)

    def test_integrate_invalid_subinterval(self):
        """Test integrate method with invalid subinterval."""
        def f(x):
            return x

        # Subinterval outside domain
        with self.assertRaises(ValueError):
            self.interval_closed.integrate(f, support=(1.5, 2.0))

        # Invalid subinterval (a >= b)
        with self.assertRaises(ValueError):
            self.interval_closed.integrate(f, support=(0.5, 0.3))

    @patch('scipy.integrate.quad')
    def test_integrate_adaptive_method(self, mock_quad):
        """Test integrate method with adaptive method."""
        mock_quad.return_value = (0.5, 1e-8)  # (result, error)

        def f(x):
            return x
        result = self.interval_closed.integrate(f, method='adaptive')

        self.assertEqual(result, 0.5)
        mock_quad.assert_called_once()

    def test_integrate_adaptive_method_no_scipy(self):
        """Test integrate adaptive method when scipy is not available."""
        with patch.dict('sys.modules', {'scipy.integrate': None}):
            def f(x):
                return x
            with self.assertRaises(ImportError):
                self.interval_closed.integrate(f, method='adaptive')

    def test_integrate_gauss_method(self):
        """Test integrate method with Gauss-Legendre quadrature."""
        def f(x):
            return x**2

        try:
            result = self.interval_closed.integrate(f, method='gauss', n=50)
            # Analytical result: ∫₀¹ x² dx = 1/3
            self.assertAlmostEqual(result, 1.0/3.0, places=6)
        except ImportError:
            # Skip if scipy not available
            self.skipTest("scipy not available for Gauss-Legendre quadrature")

    def test_integrate_gauss_method_no_scipy(self):
        """Test Gauss-Legendre integration when scipy is not available."""
        with patch.dict('sys.modules', {'scipy.special': None}):
            def f(x):
                return x
            with self.assertRaises(ImportError):
                self.interval_closed.integrate(f, method='gauss')

    def test_integrate_simpson_method(self):
        """Test integrate method with Simpson's rule."""
        def f(x):
            return x**3

        result = self.interval_closed.integrate(
            f, method='simpson', n_points=1001
        )
        # Analytical result: ∫₀¹ x³ dx = 1/4
        self.assertAlmostEqual(result, 0.25, places=3)

    def test_integrate_invalid_method(self):
        """Test integrate method with invalid method."""
        def f(x):
            return x

        with self.assertRaises(ValueError):
            self.interval_closed.integrate(f, method='invalid_method')

    def test_point_evaluation_functional(self):
        """Test point_evaluation_functional method."""
        # Point inside domain
        x = 0.5
        delta_x = self.interval_closed.point_evaluation_functional(x)

        # Test the functional
        def f(y):
            return y**2
        result = delta_x(f)
        self.assertEqual(result, 0.25)

    def test_point_evaluation_functional_outside_domain(self):
        """Test point_evaluation_functional with point outside domain."""
        with self.assertRaises(ValueError):
            self.interval_closed.point_evaluation_functional(1.5)

    def test_restriction_to_subinterval(self):
        """Test restriction_to_subinterval method."""
        subinterval = self.interval_closed.restriction_to_subinterval(0.2, 0.8)

        self.assertEqual(subinterval.a, 0.2)
        self.assertEqual(subinterval.b, 0.8)
        self.assertEqual(subinterval.boundary_type, 'closed')

    def test_restriction_to_subinterval_invalid(self):
        """Test restriction_to_subinterval with invalid subinterval."""
        # Subinterval outside domain
        with self.assertRaises(ValueError):
            self.interval_closed.restriction_to_subinterval(1.5, 2.0)

        # Invalid subinterval (c >= d)
        with self.assertRaises(ValueError):
            self.interval_closed.restriction_to_subinterval(0.8, 0.2)

        # Subinterval partially outside
        with self.assertRaises(ValueError):
            self.interval_closed.restriction_to_subinterval(-0.5, 0.5)

    def test_repr(self):
        """Test string representation."""
        # Closed interval
        self.assertEqual(repr(self.interval_closed), "[0.0, 1.0]")

        # Open interval
        self.assertEqual(repr(self.interval_open), "(-1.0, 2.0)")

        # Left open interval
        self.assertEqual(repr(self.interval_left_open), "(0.5, 3.5]")

        # Right open interval
        self.assertEqual(repr(self.interval_right_open), "[-2.0, -0.5)")

    def test_equality(self):
        """Test equality comparison."""
        # Same intervals
        interval1 = IntervalDomain(0.0, 1.0, boundary_type='closed')
        interval2 = IntervalDomain(0.0, 1.0, boundary_type='closed')
        self.assertEqual(interval1, interval2)

        # Different bounds
        interval3 = IntervalDomain(0.0, 2.0, boundary_type='closed')
        self.assertNotEqual(interval1, interval3)

        # Different boundary types
        interval4 = IntervalDomain(0.0, 1.0, boundary_type='open')
        self.assertNotEqual(interval1, interval4)

        # Different type
        self.assertNotEqual(interval1, "not an interval")

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Very small interval
        tiny_interval = IntervalDomain(0.0, 1e-10)
        self.assertAlmostEqual(tiny_interval.length, 1e-10)

        # Large interval
        large_interval = IntervalDomain(-1e6, 1e6)
        self.assertEqual(large_interval.length, 2e6)

        # Negative intervals
        neg_interval = IntervalDomain(-5.0, -2.0)
        self.assertEqual(neg_interval.center, -3.5)


if __name__ == '__main__':
    unittest.main()
