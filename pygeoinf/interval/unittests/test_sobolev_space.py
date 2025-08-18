"""
Unit tests for Sobolev space implementation.

Tests the Sobolev class which provides H^s spaces on intervals with
spectral inner products based on eigenvalues of differential operators.
"""

import unittest
import numpy as np
import numpy.testing as npt
import sys
import os

# Add the parent directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from pygeoinf.interval.sobolev_space import Sobolev
from pygeoinf.interval.interval_domain import IntervalDomain
from pygeoinf.interval.boundary_conditions import BoundaryConditions
from pygeoinf.interval.functions import Function


class TestSobolevSpaceInitialization(unittest.TestCase):
    """Test Sobolev space initialization with different parameters."""

    def setUp(self):
        """Set up test fixtures."""
        self.domain = IntervalDomain(0, 1)
        self.dim = 5
        self.order = 1.5

    def test_init_with_basis_type(self):
        """Test initialization with basis_type parameter."""
        space = Sobolev(
            self.dim, self.domain, self.order,
            basis_type='fourier'
        )

        self.assertEqual(space.dim, self.dim)
        self.assertEqual(space.order, self.order)
        self.assertEqual(space.function_domain, self.domain)
        self.assertIsNotNone(space._spectrum_provider)

    def test_init_with_basis_callables_and_eigenvalues(self):
        """Test initialization with custom basis functions and eigenvalues."""
        # Create simple polynomial basis functions
        basis_callables = [
            lambda x: np.ones_like(x),  # constant
            lambda x: x,  # linear
            lambda x: x**2,  # quadratic
        ]
        eigenvalues = np.array([0.0, 1.0, 4.0])

        space = Sobolev(
            3, self.domain, self.order,
            basis_callables=basis_callables,
            eigenvalues=eigenvalues
        )

        self.assertEqual(space.dim, 3)
        npt.assert_array_equal(space.eigenvalues, eigenvalues)

    def test_init_with_boundary_conditions(self):
        """Test initialization with boundary conditions."""
        bc = BoundaryConditions('dirichlet', left=0.0, right=0.0)

        space = Sobolev(
            self.dim, self.domain, self.order,
            basis_type='fourier',
            boundary_conditions=bc
        )

        self.assertEqual(space.boundary_conditions, bc)

    def test_init_invalid_no_basis_specified(self):
        """Test that initialization fails when no basis is specified."""
        with self.assertRaises(ValueError) as context:
            Sobolev(self.dim, self.domain, self.order)

        self.assertIn("exactly one of", str(context.exception))

    def test_init_invalid_multiple_basis_specified(self):
        """Test initialization fails when multiple basis options given."""
        basis_callables = [lambda x: np.ones_like(x)]
        eigenvalues = np.array([0.0])

        with self.assertRaises(ValueError) as context:
            Sobolev(
                self.dim, self.domain, self.order,
                basis_type='fourier',
                basis_callables=basis_callables,
                eigenvalues=eigenvalues
            )

        self.assertIn("exactly one of", str(context.exception))

    def test_init_basis_callables_without_eigenvalues(self):
        """Test that basis_callables requires eigenvalues."""
        basis_callables = [lambda x: np.ones_like(x)]

        with self.assertRaises(ValueError) as context:
            Sobolev(
                self.dim, self.domain, self.order,
                basis_callables=basis_callables
            )

        self.assertIn("eigenvalues must also be provided",
                      str(context.exception))

    def test_init_eigenvalues_wrong_length(self):
        """Test that eigenvalues must match dimension."""
        basis_callables = [lambda x: np.ones_like(x)]
        eigenvalues = np.array([0.0, 1.0])  # Wrong length

        with self.assertRaises(ValueError) as context:
            Sobolev(
                1, self.domain, self.order,
                basis_callables=basis_callables,
                eigenvalues=eigenvalues
            )

        self.assertIn("eigenvalues length", str(context.exception))


class TestSobolevSpaceProperties(unittest.TestCase):
    """Test Sobolev space properties and metadata."""

    def setUp(self):
        """Set up test fixtures."""
        self.domain = IntervalDomain(0, 2*np.pi)
        self.space = Sobolev(
            5, self.domain, 1.0,
            basis_type='fourier'
        )

    def test_order_property(self):
        """Test that order property returns correct value."""
        self.assertEqual(self.space.order, 1.0)

    def test_boundary_conditions_property(self):
        """Test boundary conditions property."""
        # Test without boundary conditions
        self.assertIsNone(self.space.boundary_conditions)

        # Test with boundary conditions
        bc = BoundaryConditions('periodic')
        space_with_bc = Sobolev(
            3, self.domain, 1.0,
            basis_type='fourier',
            boundary_conditions=bc
        )
        self.assertEqual(space_with_bc.boundary_conditions, bc)

    def test_eigenvalues_property(self):
        """Test eigenvalues property."""
        eigenvalues = self.space.eigenvalues
        self.assertIsInstance(eigenvalues, np.ndarray)
        self.assertEqual(len(eigenvalues), self.space.dim)

    def test_operator_property(self):
        """Test operator property returns correct metadata."""
        operator_info = self.space.operator

        self.assertIsInstance(operator_info, dict)
        self.assertEqual(operator_info['type'], 'negative_laplacian')
        self.assertEqual(operator_info['symbol'], '-Δ')
        self.assertIn('boundary_conditions', operator_info)
        self.assertIn('domain', operator_info)
        self.assertIn('description', operator_info)
        self.assertIn('eigenfunction_basis', operator_info)

    def test_operator_with_different_boundary_conditions(self):
        """Test operator description with different boundary conditions."""
        bc_types = ['periodic', 'dirichlet', 'neumann']

        for bc_type in bc_types:
            bc = BoundaryConditions(bc_type)
            space = Sobolev(
                3, self.domain, 1.0,
                basis_type='fourier',
                boundary_conditions=bc
            )

            operator_info = space.operator
            self.assertEqual(operator_info['boundary_conditions'], bc_type)
            self.assertIn(bc_type, operator_info['description'])


class TestSobolevSpaceInnerProduct(unittest.TestCase):
    """Test Sobolev space inner product calculations."""

    def setUp(self):
        """Set up test fixtures."""
        self.domain = IntervalDomain(0, 2*np.pi)
        self.space = Sobolev(
            3, self.domain, 1.0,
            basis_type='fourier'
        )

    def test_spectral_inner_product_with_basis_functions(self):
        """Test spectral inner product between basis functions."""
        # Get two basis functions
        phi_0 = self.space.get_basis_function(0)
        phi_1 = self.space.get_basis_function(1)

        # Inner product should depend on eigenvalues and Sobolev order
        result = self.space.inner_product(phi_0, phi_1)
        self.assertIsInstance(result, float)

    def test_spectral_inner_product_self_positive(self):
        """Test that inner product of function with itself is positive."""
        phi_0 = self.space.get_basis_function(0)
        result = self.space.inner_product(phi_0, phi_0)
        self.assertGreater(result, 0)

    def test_spectral_inner_product_linearity(self):
        """Test linearity of inner product in first argument."""
        phi_0 = self.space.get_basis_function(0)
        phi_1 = self.space.get_basis_function(1)

        # Create linear combination
        coeffs = np.array([2.0, 3.0, 0.0])
        f = self.space.from_components(coeffs)

        # Test linearity: <a*phi_0 + b*phi_1, phi_0> =
        # a*<phi_0,phi_0> + b*<phi_1,phi_0>
        result1 = self.space.inner_product(f, phi_0)
        result2 = (2.0 * self.space.inner_product(phi_0, phi_0) +
                   3.0 * self.space.inner_product(phi_1, phi_0))

        self.assertAlmostEqual(result1, result2, places=10)

    def test_spectral_inner_product_invalid_types(self):
        """Test that inner product requires Function instances."""
        phi_0 = self.space.get_basis_function(0)

        with self.assertRaises(TypeError):
            self.space.inner_product(phi_0, "not a function")

        with self.assertRaises(TypeError):
            self.space.inner_product(3.14, phi_0)


class TestSobolevSpaceGramMatrix(unittest.TestCase):
    """Test Gram matrix computation for Sobolev spaces."""

    def setUp(self):
        """Set up test fixtures."""
        self.domain = IntervalDomain(0, 2*np.pi)
        self.space = Sobolev(
            3, self.domain, 1.0,
            basis_type='fourier'
        )

    def test_gram_matrix_shape(self):
        """Test that Gram matrix has correct shape."""
        gram = self.space.gram_matrix
        self.assertEqual(gram.shape, (self.space.dim, self.space.dim))

    def test_gram_matrix_symmetric(self):
        """Test that Gram matrix is symmetric."""
        gram = self.space.gram_matrix
        npt.assert_array_almost_equal(gram, gram.T)

    def test_gram_matrix_positive_definite(self):
        """Test that Gram matrix is positive definite."""
        gram = self.space.gram_matrix
        eigenvals = np.linalg.eigvals(gram)
        self.assertTrue(np.all(eigenvals > 0))

    def test_gram_matrix_caching(self):
        """Test that Gram matrix is computed only once (cached)."""
        # First access computes the matrix
        gram1 = self.space.gram_matrix
        # Second access should return cached version
        gram2 = self.space.gram_matrix

        # Should be the exact same object (not just equal values)
        self.assertIs(gram1, gram2)


class TestSobolevSpaceComponentTransforms(unittest.TestCase):
    """Test conversion between functions and coefficient representations."""

    def setUp(self):
        """Set up test fixtures."""
        self.domain = IntervalDomain(0, 2*np.pi)
        self.space = Sobolev(
            3, self.domain, 1.0,
            basis_type='fourier'
        )

    def test_to_components_and_from_components_roundtrip(self):
        """Test that to_components and from_components are inverses."""
        # Create a function from known coefficients
        original_coeffs = np.array([1.0, 2.0, -0.5])
        f = self.space.from_components(original_coeffs)

        # Convert back to coefficients
        recovered_coeffs = self.space.to_components(f)

        # Should recover original coefficients
        npt.assert_array_almost_equal(original_coeffs, recovered_coeffs)

    def test_from_components_creates_function(self):
        """Test that from_components creates valid Function instance."""
        coeffs = np.array([1.0, 0.0, -1.0])
        f = self.space.from_components(coeffs)

        self.assertIsInstance(f, Function)
        self.assertEqual(f.space, self.space)

    def test_from_components_wrong_length(self):
        """Test that from_components rejects wrong coefficient length."""
        wrong_coeffs = np.array([1.0, 2.0])  # Too short

        with self.assertRaises(ValueError):
            self.space.from_components(wrong_coeffs)

    def test_to_components_uses_sobolev_inner_product(self):
        """Test that to_components uses Sobolev (not L2) inner products."""
        # Create a basis function
        phi_0 = self.space.get_basis_function(0)

        # Convert to coefficients
        coeffs = self.space.to_components(phi_0)

        # The coefficient should reflect Sobolev inner product weights
        # For basis functions, this involves the Gram matrix
        expected_coeffs = np.linalg.solve(
            self.space.gram_matrix,
            np.array([1.0, 0.0, 0.0])  # RHS from Sobolev inner products
        )

        npt.assert_array_almost_equal(coeffs, expected_coeffs)


class TestSobolevSpaceWithCustomBasis(unittest.TestCase):
    """Test Sobolev space with custom basis functions and eigenvalues."""

    def setUp(self):
        """Set up test fixtures with custom basis."""
        self.domain = IntervalDomain(0, 1)

        # Simple polynomial basis: 1, x, x^2
        self.basis_callables = [
            lambda x: np.ones_like(x),
            lambda x: x,
            lambda x: x**2
        ]

        # Corresponding eigenvalues (made up for testing)
        self.eigenvalues = np.array([0.0, 1.0, 4.0])

        self.space = Sobolev(
            3, self.domain, 1.0,
            basis_callables=self.basis_callables,
            eigenvalues=self.eigenvalues
        )

    def test_eigenvalues_stored_correctly(self):
        """Test that custom eigenvalues are stored correctly."""
        npt.assert_array_equal(self.space.eigenvalues, self.eigenvalues)

    def test_basis_functions_accessible(self):
        """Test that custom basis functions are accessible."""
        phi_0 = self.space.get_basis_function(0)
        phi_1 = self.space.get_basis_function(1)

        self.assertIsInstance(phi_0, Function)
        self.assertIsInstance(phi_1, Function)

    def test_spectral_inner_product_with_custom_eigenvalues(self):
        """Test spectral inner product uses custom eigenvalues."""
        phi_0 = self.space.get_basis_function(0)  # eigenvalue = 0
        phi_1 = self.space.get_basis_function(1)  # eigenvalue = 1

        # For order s=1.0:
        # Weight for phi_0: (1 + 0)^1 = 1
        # Weight for phi_1: (1 + 1)^1 = 2

        # Self inner products should reflect these weights
        result_0 = self.space.inner_product(phi_0, phi_0)
        result_1 = self.space.inner_product(phi_1, phi_1)

        self.assertGreater(result_0, 0)
        self.assertGreater(result_1, 0)
        # phi_1 should have higher norm due to larger eigenvalue
        # (exact relationship depends on L2 inner products of basis functions)


class TestSobolevSpaceAutomorphism(unittest.TestCase):
    """Test automorphism functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.domain = IntervalDomain(0, 2*np.pi)
        self.space = Sobolev(
            3, self.domain, 1.0,
            basis_type='fourier'
        )

    def test_automorphism_creation(self):
        """Test that automorphism can be created."""
        # Create a simple scaling automorphism
        f = lambda k: 2.0 if k == 0 else 1.0
        auto = self.space.automorphism(f)

        # Should return a LinearOperator
        from pygeoinf.operators import LinearOperator
        self.assertIsInstance(auto, LinearOperator)

    def test_automorphism_application(self):
        """Test applying automorphism to a function."""
        # Create scaling automorphism
        f = lambda k: 2.0 if k == 0 else 1.0
        auto = self.space.automorphism(f)

        # Create test function
        coeffs = np.array([1.0, 1.0, 1.0])
        test_func = self.space.from_components(coeffs)

        # Apply automorphism
        result = auto(test_func)

        self.assertIsInstance(result, Function)
        self.assertEqual(result.space, self.space)


class TestSobolevSpaceEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.domain = IntervalDomain(0, 1)

    def test_zero_order_sobolev_space(self):
        """Test Sobolev space with order 0 (should be L2-like)."""
        space = Sobolev(
            3, self.domain, 0.0,
            basis_type='fourier'
        )

        self.assertEqual(space.order, 0.0)

        # Order 0 should give weights (1 + λ_k)^0 = 1 for all k
        phi_0 = space.get_basis_function(0)
        phi_1 = space.get_basis_function(1)

        # Inner products should exist and be positive
        self.assertGreater(space.inner_product(phi_0, phi_0), 0)
        self.assertGreater(space.inner_product(phi_1, phi_1), 0)

    def test_negative_order_sobolev_space(self):
        """Test Sobolev space with negative order."""
        space = Sobolev(
            3, self.domain, -0.5,
            basis_type='fourier'
        )

        self.assertEqual(space.order, -0.5)

        # Should still work mathematically
        phi_0 = space.get_basis_function(0)
        self.assertGreater(space.inner_product(phi_0, phi_0), 0)

    def test_very_small_dimension(self):
        """Test Sobolev space with dimension 1."""
        space = Sobolev(
            1, self.domain, 1.0,
            basis_type='fourier'
        )

        self.assertEqual(space.dim, 1)

        # Should still work
        phi_0 = space.get_basis_function(0)
        self.assertGreater(space.inner_product(phi_0, phi_0), 0)


if __name__ == '__main__':
    unittest.main()