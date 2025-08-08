"""
Complete unit tests for interval_domain.py

This module provides comprehensive unit tests for the IntervalDomain class,
testing all methods, properties, edge cases, and error conditions.
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

try:
    from pygeoinf.interval.interval_domain import IntervalDomain
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_SUCCESSFUL = False


@unittest.skipUnless(IMPORTS_SUCCESSFUL, "Required modules not available")
class TestIntervalDomain(unittest.TestCase):
    """Complete test cases for IntervalDomain class."""

    def setUp(self):
        """Set up test fixtures."""
        self.domain_closed = IntervalDomain(0.0, 1.0, boundary_type='closed')
        self.domain_open = IntervalDomain(-1.0, 2.0, boundary_type='open')
        self.domain_left_open = IntervalDomain(1.0, 3.0, boundary_type='left_open')
        self.domain_right_open = IntervalDomain(-2.0, 0.0, boundary_type='right_open')

    # === INITIALIZATION TESTS ===

    def test_init_basic_closed(self):
        """Test basic initialization with closed interval."""
        domain = IntervalDomain(0.0, 1.0)

        self.assertEqual(domain.a, 0.0)
        self.assertEqual(domain.b, 1.0)
        self.assertEqual(domain.boundary_type, 'closed')  # default
        self.assertEqual(domain.name, '[0.0, 1.0]')

    def test_init_with_different_boundary_types(self):
        """Test initialization with different boundary types."""
        boundary_types = ['closed', 'open', 'left_open', 'right_open']

        for boundary_type in boundary_types:
            domain = IntervalDomain(-1.0, 2.0, boundary_type=boundary_type)
            self.assertEqual(domain.boundary_type, boundary_type)
            self.assertEqual(domain.a, -1.0)
            self.assertEqual(domain.b, 2.0)

    def test_init_with_custom_name(self):
        """Test initialization with custom name."""
        domain = IntervalDomain(0.0, 1.0, name="Unit Interval")

        self.assertEqual(domain.name, "Unit Interval")
        self.assertEqual(domain.a, 0.0)
        self.assertEqual(domain.b, 1.0)

    def test_init_invalid_interval_equal_endpoints(self):
        """Test initialization with equal endpoints raises error."""
        with self.assertRaises(ValueError):
            IntervalDomain(1.0, 1.0)

    def test_init_invalid_interval_reversed_endpoints(self):
        """Test initialization with reversed endpoints raises error."""
        with self.assertRaises(ValueError):
            IntervalDomain(2.0, 1.0)

    def test_init_with_floats_and_ints(self):
        """Test initialization with different numeric types."""
        # Integer endpoints
        domain1 = IntervalDomain(0, 5)
        self.assertIsInstance(domain1.a, float)
        self.assertIsInstance(domain1.b, float)

        # Mixed types
        domain2 = IntervalDomain(1, 2.5)
        self.assertEqual(domain2.a, 1.0)
        self.assertEqual(domain2.b, 2.5)

    def test_init_different_intervals(self):
        """Test initialization with various intervals."""
        intervals = [
            (-10.0, 10.0),
            (1e-6, 1e6),
            (-1e-3, 1e-3),
            (0.0, np.pi),
            (-np.e, np.e)
        ]

        for a, b in intervals:
            domain = IntervalDomain(a, b)
            self.assertEqual(domain.a, a)
            self.assertEqual(domain.b, b)

    # === PROPERTY TESTS ===

    def test_length_property(self):
        """Test length property computation."""
        test_cases = [
            ((0.0, 1.0), 1.0),
            ((-1.0, 2.0), 3.0),
            ((1.5, 3.5), 2.0),
            ((-10.0, -5.0), 5.0),
            ((0.0, np.pi), np.pi)
        ]

        for (a, b), expected_length in test_cases:
            domain = IntervalDomain(a, b)
            self.assertAlmostEqual(domain.length, expected_length, places=10)

    def test_center_property(self):
        """Test center property computation."""
        test_cases = [
            ((0.0, 1.0), 0.5),
            ((-1.0, 1.0), 0.0),
            ((2.0, 6.0), 4.0),
            ((-10.0, -5.0), -7.5),
            ((1.0, 3.0), 2.0)
        ]

        for (a, b), expected_center in test_cases:
            domain = IntervalDomain(a, b)
            self.assertAlmostEqual(domain.center, expected_center, places=10)

    def test_radius_property(self):
        """Test radius property computation."""
        test_cases = [
            ((0.0, 1.0), 0.5),
            ((-1.0, 3.0), 2.0),
            ((2.0, 8.0), 3.0),
            ((-5.0, -1.0), 2.0),
            ((0.0, 2.0), 1.0)
        ]

        for (a, b), expected_radius in test_cases:
            domain = IntervalDomain(a, b)
            self.assertAlmostEqual(domain.radius, expected_radius, places=10)

    # === CONTAINS METHOD TESTS ===

    def test_contains_closed_interval(self):
        """Test contains method for closed intervals."""
        domain = IntervalDomain(0.0, 1.0, boundary_type='closed')

        # Interior points
        self.assertTrue(domain.contains(0.5))
        self.assertTrue(domain.contains(0.25))
        self.assertTrue(domain.contains(0.75))

        # Boundary points
        self.assertTrue(domain.contains(0.0))
        self.assertTrue(domain.contains(1.0))

        # Exterior points
        self.assertFalse(domain.contains(-0.1))
        self.assertFalse(domain.contains(1.1))

    def test_contains_open_interval(self):
        """Test contains method for open intervals."""
        domain = IntervalDomain(0.0, 1.0, boundary_type='open')

        # Interior points
        self.assertTrue(domain.contains(0.5))
        self.assertTrue(domain.contains(0.25))
        self.assertTrue(domain.contains(0.75))

        # Boundary points (should be excluded)
        self.assertFalse(domain.contains(0.0))
        self.assertFalse(domain.contains(1.0))

        # Exterior points
        self.assertFalse(domain.contains(-0.1))
        self.assertFalse(domain.contains(1.1))

    def test_contains_left_open_interval(self):
        """Test contains method for left-open intervals."""
        domain = IntervalDomain(0.0, 1.0, boundary_type='left_open')

        # Interior points
        self.assertTrue(domain.contains(0.5))

        # Boundary points
        self.assertFalse(domain.contains(0.0))  # Left endpoint excluded
        self.assertTrue(domain.contains(1.0))   # Right endpoint included

        # Exterior points
        self.assertFalse(domain.contains(-0.1))
        self.assertFalse(domain.contains(1.1))

    def test_contains_right_open_interval(self):
        """Test contains method for right-open intervals."""
        domain = IntervalDomain(0.0, 1.0, boundary_type='right_open')

        # Interior points
        self.assertTrue(domain.contains(0.5))

        # Boundary points
        self.assertTrue(domain.contains(0.0))   # Left endpoint included
        self.assertFalse(domain.contains(1.0))  # Right endpoint excluded

        # Exterior points
        self.assertFalse(domain.contains(-0.1))
        self.assertFalse(domain.contains(1.1))

    def test_contains_array_input(self):
        """Test contains method with array input."""
        domain = IntervalDomain(0.0, 1.0, boundary_type='closed')

        x = np.array([-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
        expected = np.array([False, True, True, True, True, True, False])

        result = domain.contains(x)
        np.testing.assert_array_equal(result, expected)

    def test_contains_edge_cases(self):
        """Test contains method with edge cases."""
        domain = IntervalDomain(0.0, 1.0, boundary_type='closed')

        # Very close to boundaries
        self.assertTrue(domain.contains(1e-15))
        self.assertTrue(domain.contains(1.0 - 1e-15))

        # Exactly at numerical precision
        self.assertTrue(domain.contains(0.0 + np.finfo(float).eps))
        self.assertTrue(domain.contains(1.0 - np.finfo(float).eps))

    def test_contains_invalid_boundary_type(self):
        """Test contains method with invalid boundary type."""
        domain = IntervalDomain(0.0, 1.0)
        domain.boundary_type = 'invalid'

        with self.assertRaises(ValueError):
            domain.contains(0.5)

    # === TOPOLOGICAL METHODS TESTS ===

    def test_interior_method(self):
        """Test interior method."""
        test_domains = [
            self.domain_closed,
            self.domain_open,
            self.domain_left_open,
            self.domain_right_open
        ]

        for domain in test_domains:
            interior = domain.interior()
            self.assertEqual(interior.a, domain.a)
            self.assertEqual(interior.b, domain.b)
            self.assertEqual(interior.boundary_type, 'open')

    def test_closure_method(self):
        """Test closure method."""
        test_domains = [
            self.domain_closed,
            self.domain_open,
            self.domain_left_open,
            self.domain_right_open
        ]

        for domain in test_domains:
            closure = domain.closure()
            self.assertEqual(closure.a, domain.a)
            self.assertEqual(closure.b, domain.b)
            self.assertEqual(closure.boundary_type, 'closed')

    def test_boundary_points_method(self):
        """Test boundary_points method."""
        domain = IntervalDomain(-2.5, 3.7)

        boundary = domain.boundary_points()
        self.assertEqual(boundary, (-2.5, 3.7))
        self.assertIsInstance(boundary, tuple)

    # === MESH GENERATION TESTS ===

    def test_uniform_mesh_closed(self):
        """Test uniform mesh generation for closed intervals."""
        domain = IntervalDomain(0.0, 1.0, boundary_type='closed')

        mesh = domain.uniform_mesh(5)
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        np.testing.assert_array_almost_equal(mesh, expected, decimal=10)
        self.assertEqual(len(mesh), 5)

    def test_uniform_mesh_open(self):
        """Test uniform mesh generation for open intervals."""
        domain = IntervalDomain(0.0, 1.0, boundary_type='open')

        mesh = domain.uniform_mesh(3)
        # Should exclude both endpoints
        expected = np.array([0.25, 0.5, 0.75])

        np.testing.assert_array_almost_equal(mesh, expected, decimal=10)
        self.assertEqual(len(mesh), 3)

    def test_uniform_mesh_left_open(self):
        """Test uniform mesh generation for left-open intervals."""
        domain = IntervalDomain(0.0, 1.0, boundary_type='left_open')

        mesh = domain.uniform_mesh(4)
        # Should exclude left endpoint, include right endpoint
        expected = np.array([0.25, 0.5, 0.75, 1.0])

        np.testing.assert_array_almost_equal(mesh, expected, decimal=10)
        self.assertEqual(len(mesh), 4)

    def test_uniform_mesh_right_open(self):
        """Test uniform mesh generation for right-open intervals."""
        domain = IntervalDomain(0.0, 1.0, boundary_type='right_open')

        mesh = domain.uniform_mesh(4)
        # Should include left endpoint, exclude right endpoint
        expected = np.array([0.0, 0.25, 0.5, 0.75])

        np.testing.assert_array_almost_equal(mesh, expected, decimal=10)
        self.assertEqual(len(mesh), 4)

    def test_uniform_mesh_different_intervals(self):
        """Test uniform mesh generation for different intervals."""
        domain = IntervalDomain(-2.0, 3.0, boundary_type='closed')

        mesh = domain.uniform_mesh(6)
        expected = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

        np.testing.assert_array_almost_equal(mesh, expected, decimal=10)

    def test_uniform_mesh_single_point(self):
        """Test uniform mesh generation with n=1."""
        domain = IntervalDomain(0.0, 1.0, boundary_type='closed')

        mesh = domain.uniform_mesh(1)
        self.assertEqual(len(mesh), 1)
        # For n=1, numpy.linspace returns the start point
        self.assertAlmostEqual(mesh[0], 0.0, places=10)

    def test_uniform_mesh_invalid_boundary_type(self):
        """Test uniform mesh with invalid boundary type."""
        domain = IntervalDomain(0.0, 1.0)
        domain.boundary_type = 'invalid'

        with self.assertRaises(ValueError):
            domain.uniform_mesh(5)

    # === ADAPTIVE MESH TESTS ===

    def test_adaptive_mesh_basic(self):
        """Test basic adaptive mesh generation."""
        domain = IntervalDomain(0.0, 1.0, boundary_type='closed')

        def simple_func(x):
            return x**2

        mesh = domain.adaptive_mesh(simple_func, max_points=10)

        self.assertIsInstance(mesh, np.ndarray)
        self.assertGreater(len(mesh), 0)
        self.assertLessEqual(len(mesh), 10)

    def test_adaptive_mesh_different_functions(self):
        """Test adaptive mesh with different functions."""
        domain = IntervalDomain(-1.0, 1.0, boundary_type='closed')

        functions = [
            lambda x: np.ones_like(x),
            lambda x: x**2,
            lambda x: np.sin(x),
            lambda x: np.exp(x)
        ]

        for func in functions:
            mesh = domain.adaptive_mesh(func, max_points=20)
            self.assertIsInstance(mesh, np.ndarray)
            self.assertGreater(len(mesh), 0)

    def test_adaptive_mesh_tolerance_parameter(self):
        """Test adaptive mesh with different tolerance values."""
        domain = IntervalDomain(0.0, 1.0, boundary_type='closed')

        def test_func(x):
            return x**3

        tolerances = [1e-3, 1e-6, 1e-9]
        for tol in tolerances:
            mesh = domain.adaptive_mesh(test_func, tol=tol, max_points=50)
            self.assertIsInstance(mesh, np.ndarray)

    # === RANDOM POINTS TESTS ===

    def test_random_points_basic(self):
        """Test basic random point generation."""
        domain = IntervalDomain(0.0, 1.0, boundary_type='closed')

        points = domain.random_points(100, seed=42)

        self.assertEqual(len(points), 100)
        self.assertTrue(np.all(points >= 0.0))
        self.assertTrue(np.all(points <= 1.0))
        self.assertTrue(np.all(domain.contains(points)))

    def test_random_points_reproducibility(self):
        """Test random point generation reproducibility with seed."""
        domain = IntervalDomain(-1.0, 2.0, boundary_type='closed')

        points1 = domain.random_points(50, seed=123)
        points2 = domain.random_points(50, seed=123)

        np.testing.assert_array_equal(points1, points2)

    def test_random_points_open_interval(self):
        """Test random point generation for open intervals."""
        domain = IntervalDomain(0.0, 1.0, boundary_type='open')

        points = domain.random_points(1000, seed=42)

        # Points should be strictly inside the interval
        self.assertTrue(np.all(points > 0.0))
        self.assertTrue(np.all(points < 1.0))
        self.assertTrue(np.all(domain.contains(points)))

    def test_random_points_different_sizes(self):
        """Test random point generation with different sizes."""
        domain = IntervalDomain(-2.0, 3.0, boundary_type='closed')

        sizes = [1, 10, 100, 1000]
        for n in sizes:
            points = domain.random_points(n, seed=42)
            self.assertEqual(len(points), n)
            self.assertTrue(np.all(domain.contains(points)))

    def test_random_points_different_domains(self):
        """Test random point generation for different domains."""
        domains = [
            IntervalDomain(-10.0, 10.0),
            IntervalDomain(0.0, 1e-3),
            IntervalDomain(1e3, 1e6),
            IntervalDomain(-np.pi, np.pi)
        ]

        for domain in domains:
            points = domain.random_points(50, seed=42)
            self.assertEqual(len(points), 50)
            self.assertTrue(np.all(domain.contains(points)))

    # === INTEGRATION TESTS ===

    def test_integrate_simple_function(self):
        """Test integration of simple functions."""
        domain = IntervalDomain(0.0, 1.0, boundary_type='closed')

        # Constant function: ∫₀¹ 1 dx = 1
        constant_func = lambda x: 1.0
        result = domain.integrate(constant_func, method='trapz', n_points=100)
        self.assertAlmostEqual(result, 1.0, places=2)

        # Linear function: ∫₀¹ x dx = 1/2
        linear_func = lambda x: x
        result = domain.integrate(linear_func, method='trapz', n_points=100)
        self.assertAlmostEqual(result, 0.5, places=2)

    def test_integrate_quadratic_function(self):
        """Test integration of quadratic function."""
        domain = IntervalDomain(0.0, 2.0, boundary_type='closed')

        # Quadratic function: ∫₀² x² dx = 8/3
        quadratic_func = lambda x: x**2
        result = domain.integrate(quadratic_func, method='trapz', n_points=1000)
        expected = 8.0 / 3.0
        self.assertAlmostEqual(result, expected, places=1)

    def test_integrate_different_methods(self):
        """Test integration with different methods."""
        domain = IntervalDomain(0.0, 1.0, boundary_type='closed')

        def test_func(x):
            return x**2

        methods = ['trapz', 'simpson']
        results = []

        for method in methods:
            try:
                result = domain.integrate(test_func, method=method, n_points=100)
                results.append(result)
                self.assertIsInstance(result, float)
            except ImportError:
                # Method may not be available
                pass

        # Results should be similar (all approximating 1/3)
        if len(results) > 1:
            for result in results:
                self.assertAlmostEqual(result, 1.0/3.0, places=1)

    def test_integrate_subinterval(self):
        """Test integration over subinterval."""
        domain = IntervalDomain(0.0, 2.0, boundary_type='closed')

        def test_func(x):
            return x

        # Integrate over [0.5, 1.5]: ∫₀.₅¹.⁵ x dx = (1.5² - 0.5²)/2 = 1
        result = domain.integrate(test_func, support=(0.5, 1.5), method='trapz')
        expected = (1.5**2 - 0.5**2) / 2
        self.assertAlmostEqual(result, expected, places=2)

    def test_integrate_multiple_subintervals(self):
        """Test integration over multiple subintervals."""
        domain = IntervalDomain(0.0, 3.0, boundary_type='closed')

        def test_func(x):
            return 1.0  # Constant function

        # Integrate over [0.5, 1.0] ∪ [1.5, 2.0] ∪ [2.5, 3.0]
        # Total length = 0.5 + 0.5 + 0.5 = 1.5
        subintervals = [(0.5, 1.0), (1.5, 2.0), (2.5, 3.0)]
        result = domain.integrate(test_func, support=subintervals, method='trapz')
        self.assertAlmostEqual(result, 1.5, places=2)

    def test_integrate_invalid_subinterval(self):
        """Test integration with invalid subinterval."""
        domain = IntervalDomain(0.0, 1.0, boundary_type='closed')

        def test_func(x):
            return x

        # Subinterval outside domain
        with self.assertRaises(ValueError):
            domain.integrate(test_func, support=(-0.5, 0.5))

        # Subinterval partially outside domain
        with self.assertRaises(ValueError):
            domain.integrate(test_func, support=(0.5, 1.5))

    @unittest.skipIf(True, "Adaptive integration requires scipy")
    def test_integrate_adaptive(self):
        """Test adaptive integration if scipy is available."""
        domain = IntervalDomain(0.0, 1.0, boundary_type='closed')

        def test_func(x):
            return x**3

        try:
            result = domain.integrate(test_func, method='adaptive')
            expected = 1.0 / 4.0  # ∫₀¹ x³ dx = 1/4
            self.assertAlmostEqual(result, expected, places=5)
        except ImportError:
            self.skipTest("scipy not available for adaptive integration")

    def test_integrate_unknown_method(self):
        """Test integration with unknown method."""
        domain = IntervalDomain(0.0, 1.0, boundary_type='closed')

        def test_func(x):
            return x

        with self.assertRaises(ValueError):
            domain.integrate(test_func, method='unknown_method')

    # === FUNCTIONAL ANALYSIS TESTS ===

    def test_point_evaluation_functional(self):
        """Test point evaluation functional creation."""
        domain = IntervalDomain(0.0, 1.0, boundary_type='closed')

        # Test functional at midpoint
        delta_half = domain.point_evaluation_functional(0.5)

        # Test with simple function
        test_func = lambda x: x**2
        result = delta_half(test_func)
        self.assertAlmostEqual(result, 0.25, places=10)

    def test_point_evaluation_functional_boundary(self):
        """Test point evaluation functional at boundaries."""
        domain = IntervalDomain(0.0, 1.0, boundary_type='closed')

        # Test at left boundary
        delta_left = domain.point_evaluation_functional(0.0)
        test_func = lambda x: x + 1
        self.assertAlmostEqual(delta_left(test_func), 1.0, places=10)

        # Test at right boundary
        delta_right = domain.point_evaluation_functional(1.0)
        self.assertAlmostEqual(delta_right(test_func), 2.0, places=10)

    def test_point_evaluation_functional_invalid_point(self):
        """Test point evaluation functional with invalid point."""
        domain = IntervalDomain(0.0, 1.0, boundary_type='open')

        # Point outside domain
        with self.assertRaises(ValueError):
            domain.point_evaluation_functional(1.5)

        # Boundary point for open interval
        with self.assertRaises(ValueError):
            domain.point_evaluation_functional(0.0)

    def test_restriction_to_subinterval(self):
        """Test restriction to subinterval."""
        domain = IntervalDomain(0.0, 2.0, boundary_type='closed')

        subdomain = domain.restriction_to_subinterval(0.5, 1.5)

        self.assertEqual(subdomain.a, 0.5)
        self.assertEqual(subdomain.b, 1.5)
        self.assertEqual(subdomain.boundary_type, domain.boundary_type)

    def test_restriction_invalid_subinterval(self):
        """Test restriction with invalid subinterval."""
        domain = IntervalDomain(0.0, 1.0, boundary_type='closed')

        # Subinterval outside domain
        with self.assertRaises(ValueError):
            domain.restriction_to_subinterval(-0.5, 0.5)

        # Reversed endpoints
        with self.assertRaises(ValueError):
            domain.restriction_to_subinterval(0.8, 0.3)

        # Right endpoint outside domain
        with self.assertRaises(ValueError):
            domain.restriction_to_subinterval(0.5, 1.5)

    # === STRING REPRESENTATION TESTS ===

    def test_repr_closed_interval(self):
        """Test string representation for closed intervals."""
        domain = IntervalDomain(0.0, 1.0, boundary_type='closed')
        repr_str = repr(domain)

        self.assertEqual(repr_str, '[0.0, 1.0]')

    def test_repr_open_interval(self):
        """Test string representation for open intervals."""
        domain = IntervalDomain(-1.0, 2.0, boundary_type='open')
        repr_str = repr(domain)

        self.assertEqual(repr_str, '(-1.0, 2.0)')

    def test_repr_left_open_interval(self):
        """Test string representation for left-open intervals."""
        domain = IntervalDomain(1.0, 3.0, boundary_type='left_open')
        repr_str = repr(domain)

        self.assertEqual(repr_str, '(1.0, 3.0]')

    def test_repr_right_open_interval(self):
        """Test string representation for right-open intervals."""
        domain = IntervalDomain(-2.0, 0.0, boundary_type='right_open')
        repr_str = repr(domain)

        self.assertEqual(repr_str, '[-2.0, 0.0)')

    # === EQUALITY TESTS ===

    def test_equality_same_domains(self):
        """Test equality for identical domains."""
        domain1 = IntervalDomain(0.0, 1.0, boundary_type='closed')
        domain2 = IntervalDomain(0.0, 1.0, boundary_type='closed')

        self.assertEqual(domain1, domain2)

    def test_equality_different_endpoints(self):
        """Test equality for domains with different endpoints."""
        domain1 = IntervalDomain(0.0, 1.0, boundary_type='closed')
        domain2 = IntervalDomain(0.0, 2.0, boundary_type='closed')

        self.assertNotEqual(domain1, domain2)

    def test_equality_different_boundary_types(self):
        """Test equality for domains with different boundary types."""
        domain1 = IntervalDomain(0.0, 1.0, boundary_type='closed')
        domain2 = IntervalDomain(0.0, 1.0, boundary_type='open')

        self.assertNotEqual(domain1, domain2)

    def test_equality_with_non_domain(self):
        """Test equality with non-IntervalDomain objects."""
        domain = IntervalDomain(0.0, 1.0, boundary_type='closed')

        self.assertNotEqual(domain, "not a domain")
        self.assertNotEqual(domain, 123)
        self.assertNotEqual(domain, [0.0, 1.0])

    # === EDGE CASES AND ERROR HANDLING ===

    def test_very_small_intervals(self):
        """Test with very small intervals."""
        domain = IntervalDomain(0.0, 1e-12, boundary_type='closed')

        self.assertAlmostEqual(domain.length, 1e-12, places=15)
        self.assertAlmostEqual(domain.center, 5e-13, places=15)
        self.assertTrue(domain.contains(5e-13))

    def test_very_large_intervals(self):
        """Test with very large intervals."""
        domain = IntervalDomain(-1e6, 1e6, boundary_type='closed')

        self.assertAlmostEqual(domain.length, 2e6, places=5)
        self.assertAlmostEqual(domain.center, 0.0, places=10)
        self.assertTrue(domain.contains(0.0))

    def test_negative_intervals(self):
        """Test with intervals in negative range."""
        domain = IntervalDomain(-10.0, -1.0, boundary_type='closed')

        self.assertEqual(domain.length, 9.0)
        self.assertEqual(domain.center, -5.5)
        self.assertTrue(domain.contains(-5.0))
        self.assertFalse(domain.contains(0.0))

    def test_floating_point_precision(self):
        """Test with floating point precision issues."""
        # Create interval with values that might have precision issues
        a = 0.1 + 0.2  # This might not be exactly 0.3
        b = 1.0
        domain = IntervalDomain(a, b, boundary_type='closed')

        self.assertIsInstance(domain.a, float)
        self.assertIsInstance(domain.b, float)
        self.assertGreater(domain.length, 0.0)

    def test_special_float_values(self):
        """Test behavior with special float values."""
        # Test with very small positive numbers
        domain = IntervalDomain(np.finfo(float).eps, 1.0, boundary_type='closed')
        self.assertGreater(domain.length, 0.0)

        # Test with numbers close to floating point limits
        large_val = 1e10
        domain_large = IntervalDomain(-large_val, large_val, boundary_type='closed')
        self.assertEqual(domain_large.length, 2 * large_val)


if __name__ == '__main__':
    if not IMPORTS_SUCCESSFUL:
        print("Skipping tests due to import errors")
        exit(1)
    unittest.main(verbosity=2)
