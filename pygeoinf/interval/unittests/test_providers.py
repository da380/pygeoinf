"""
Comprehensive unit tests for providers.py

This module provides detailed tests for basis and spectrum providers,
eigenvalue providers, and factory functions.
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

try:
    from pygeoinf.interval.providers import (
        EigenvalueProvider, BasisProvider, SpectrumProvider,
        FourierEigenvalueProvider, ZeroEigenvalueProvider,
        CustomEigenvalueProvider, LaplacianEigenvalueProvider,
        create_basis_provider, create_spectrum_provider,
        create_laplacian_spectrum_provider
    )
    from pygeoinf.interval.function_providers import (
        SineFunctionProvider, CosineFunctionProvider, FourierFunctionProvider
    )
    from pygeoinf.interval.boundary_conditions import BoundaryConditions
    from pygeoinf.interval.interval_domain import IntervalDomain
    from pygeoinf.interval import L2Space
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_SUCCESSFUL = False


@unittest.skipUnless(IMPORTS_SUCCESSFUL, "Required modules not available")
class TestProviders(unittest.TestCase):
    """Test cases for provider classes."""

    def setUp(self):
        """Set up test fixtures."""
        self.domain = IntervalDomain(0.0, 1.0)
        self.space = L2Space(10, self.domain)
        self.dirichlet_bc = BoundaryConditions.dirichlet(0.0, 0.0)
        self.neumann_bc = BoundaryConditions.neumann(0.0, 0.0)

    # === EIGENVALUE PROVIDER TESTS ===

    def test_fourier_eigenvalue_provider_basic(self):
        """Test basic Fourier eigenvalue provider functionality."""
        provider = FourierEigenvalueProvider(1.0)  # Unit domain length

        # Index 0: constant term (eigenvalue 0)
        self.assertEqual(provider.get_eigenvalue(0), 0.0)

        # Index 1: first cosine mode
        eigenval_1 = provider.get_eigenvalue(1)
        expected_1 = (2 * 1 * np.pi / 1.0) ** 2  # k=1 for index 1
        self.assertAlmostEqual(eigenval_1, expected_1, places=10)

        # Index 2: first sine mode (same eigenvalue as cosine)
        eigenval_2 = provider.get_eigenvalue(2)
        self.assertAlmostEqual(eigenval_2, expected_1, places=10)

    def test_fourier_eigenvalue_provider_different_domain(self):
        """Test Fourier eigenvalue provider with different domain length."""
        length = 2.0
        provider = FourierEigenvalueProvider(length)

        eigenval_1 = provider.get_eigenvalue(1)
        expected_1 = (2 * 1 * np.pi / length) ** 2
        self.assertAlmostEqual(eigenval_1, expected_1, places=10)

    def test_fourier_eigenvalue_provider_get_eigenvalues(self):
        """Test getting multiple eigenvalues at once."""
        provider = FourierEigenvalueProvider(1.0)

        eigenvals = provider.get_eigenvalues(5)
        self.assertEqual(len(eigenvals), 5)
        self.assertEqual(eigenvals[0], 0.0)  # Constant term

    def test_zero_eigenvalue_provider(self):
        """Test zero eigenvalue provider."""
        provider = ZeroEigenvalueProvider()

        for i in range(10):
            self.assertEqual(provider.get_eigenvalue(i), 0.0)

    def test_custom_eigenvalue_provider(self):
        """Test custom eigenvalue provider with user-specified values."""
        eigenvals = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
        provider = CustomEigenvalueProvider(eigenvals)

        for i, expected in enumerate(eigenvals):
            self.assertEqual(provider.get_eigenvalue(i), expected)

    def test_custom_eigenvalue_provider_out_of_range(self):
        """Test custom eigenvalue provider raises error for out of range."""
        eigenvals = np.array([1.0, 2.0, 3.0])
        provider = CustomEigenvalueProvider(eigenvals)

        with self.assertRaises(IndexError):
            provider.get_eigenvalue(5)

    # === LAPLACIAN EIGENVALUE PROVIDER TESTS ===

    def test_laplacian_eigenvalue_provider_dirichlet(self):
        """Test Laplacian eigenvalue provider with Dirichlet BC."""
        provider = LaplacianEigenvalueProvider(
            self.domain, self.dirichlet_bc, inverse=False
        )

        # For Dirichlet: λₖ = (kπ/L)² where k = index + 1
        eigenval_0 = provider.get_eigenvalue(0)  # k=1
        expected_0 = (1 * np.pi / 1.0) ** 2
        self.assertAlmostEqual(eigenval_0, expected_0, places=10)

        eigenval_1 = provider.get_eigenvalue(1)  # k=2
        expected_1 = (2 * np.pi / 1.0) ** 2
        self.assertAlmostEqual(eigenval_1, expected_1, places=10)

    def test_laplacian_eigenvalue_provider_neumann(self):
        """Test Laplacian eigenvalue provider with Neumann BC."""
        provider = LaplacianEigenvalueProvider(
            self.domain, self.neumann_bc, inverse=False
        )

        # For Neumann: index 0 has eigenvalue 0 (constant mode)
        self.assertEqual(provider.get_eigenvalue(0), 0.0)

        # index 1: λ₁ = (π/L)²
        eigenval_1 = provider.get_eigenvalue(1)
        expected_1 = (1 * np.pi / 1.0) ** 2
        self.assertAlmostEqual(eigenval_1, expected_1, places=10)

    def test_laplacian_eigenvalue_provider_inverse(self):
        """Test Laplacian eigenvalue provider with inverse=True."""
        provider = LaplacianEigenvalueProvider(
            self.domain, self.dirichlet_bc, inverse=True
        )

        # For inverse Laplacian with Dirichlet: 1/λₖ = 1/(kπ/L)²
        eigenval_0 = provider.get_eigenvalue(0)  # k=1
        expected_0 = 1.0 / ((1 * np.pi / 1.0) ** 2)
        self.assertAlmostEqual(eigenval_0, expected_0, places=10)

    def test_laplacian_eigenvalue_provider_inverse_neumann_zero(self):
        """Test inverse Laplacian with Neumann BC handles zero eigenvalue."""
        provider = LaplacianEigenvalueProvider(
            self.domain, self.neumann_bc, inverse=True
        )

        # For constant mode with eigenvalue 0, inverse should be infinity
        eigenval_0 = provider.get_eigenvalue(0)
        self.assertEqual(eigenval_0, float('inf'))

    def test_laplacian_eigenvalue_provider_periodic(self):
        """Test Laplacian eigenvalue provider with periodic BC."""
        periodic_bc = BoundaryConditions.periodic()
        provider = LaplacianEigenvalueProvider(
            self.domain, periodic_bc, inverse=False
        )

        # Index 0: constant mode (eigenvalue 0)
        self.assertEqual(provider.get_eigenvalue(0), 0.0)

        # Index 1,2: both have same eigenvalue (2π/L)²
        eigenval_1 = provider.get_eigenvalue(1)
        eigenval_2 = provider.get_eigenvalue(2)
        expected = (2 * 1 * np.pi / 1.0) ** 2
        self.assertAlmostEqual(eigenval_1, expected, places=10)
        self.assertAlmostEqual(eigenval_2, expected, places=10)

    def test_laplacian_eigenvalue_provider_unsupported_bc(self):
        """Test Laplacian provider raises error for unsupported BC."""
        # Create an invalid boundary condition by modifying type
        invalid_bc = BoundaryConditions.dirichlet()
        invalid_bc.type = 'invalid'

        provider = LaplacianEigenvalueProvider(
            self.domain, invalid_bc, inverse=False
        )

        with self.assertRaises(ValueError):
            provider.get_eigenvalue(0)

    # === BASIS PROVIDER TESTS ===

    def test_basis_provider_basic(self):
        """Test basic basis provider functionality."""
        func_provider = SineFunctionProvider(self.space)
        provider = BasisProvider(self.space, func_provider)

        self.assertEqual(provider.space, self.space)
        self.assertEqual(len(provider), self.space.dim)

    def test_basis_provider_get_basis_function(self):
        """Test basis provider get_basis_function method."""
        func_provider = SineFunctionProvider(self.space)
        provider = BasisProvider(self.space, func_provider)

        func = provider.get_basis_function(0)
        self.assertIsNotNone(func)
        self.assertEqual(func.space, self.space)

    def test_basis_provider_indexing(self):
        """Test basis provider indexing syntax."""
        func_provider = SineFunctionProvider(self.space)
        provider = BasisProvider(self.space, func_provider)

        func = provider[0]
        self.assertIsNotNone(func)

    def test_basis_provider_iteration(self):
        """Test basis provider iteration."""
        func_provider = SineFunctionProvider(self.space)
        provider = BasisProvider(self.space, func_provider)

        functions = list(provider)
        self.assertEqual(len(functions), self.space.dim)

    def test_basis_provider_get_all_basis_functions(self):
        """Test basis provider get_all_basis_functions method."""
        func_provider = SineFunctionProvider(self.space)
        provider = BasisProvider(self.space, func_provider)

        functions = provider.get_all_basis_functions()
        self.assertEqual(len(functions), self.space.dim)

    def test_basis_provider_out_of_range(self):
        """Test basis provider raises error for out of range indices."""
        func_provider = SineFunctionProvider(self.space)
        provider = BasisProvider(self.space, func_provider)

        with self.assertRaises(IndexError):
            provider.get_basis_function(self.space.dim)

        with self.assertRaises(IndexError):
            provider.get_basis_function(-1)

    def test_basis_provider_caching(self):
        """Test basis provider caches functions."""
        func_provider = SineFunctionProvider(self.space)
        provider = BasisProvider(self.space, func_provider)

        func1 = provider.get_basis_function(0)
        func2 = provider.get_basis_function(0)

        self.assertIs(func1, func2)

    # === SPECTRUM PROVIDER TESTS ===

    def test_spectrum_provider_basic(self):
        """Test basic spectrum provider functionality."""
        func_provider = SineFunctionProvider(self.space)
        eigenval_provider = LaplacianEigenvalueProvider(
            self.domain, self.dirichlet_bc, inverse=False
        )
        provider = SpectrumProvider(self.space, func_provider, eigenval_provider)

        self.assertEqual(provider.space, self.space)

    def test_spectrum_provider_get_eigenvalue(self):
        """Test spectrum provider get_eigenvalue method."""
        func_provider = SineFunctionProvider(self.space)
        eigenval_provider = LaplacianEigenvalueProvider(
            self.domain, self.dirichlet_bc, inverse=False
        )
        provider = SpectrumProvider(self.space, func_provider, eigenval_provider)

        eigenval = provider.get_eigenvalue(0)
        expected = (1 * np.pi / 1.0) ** 2  # First Dirichlet eigenvalue
        self.assertAlmostEqual(eigenval, expected, places=10)

    def test_spectrum_provider_get_eigenfunction(self):
        """Test spectrum provider get_eigenfunction method."""
        func_provider = SineFunctionProvider(self.space)
        eigenval_provider = LaplacianEigenvalueProvider(
            self.domain, self.dirichlet_bc, inverse=False
        )
        provider = SpectrumProvider(self.space, func_provider, eigenval_provider)

        func = provider.get_eigenfunction(0)
        self.assertIsNotNone(func)

    def test_spectrum_provider_get_all_eigenvalues(self):
        """Test spectrum provider get_all_eigenvalues method."""
        func_provider = SineFunctionProvider(self.space)
        eigenval_provider = LaplacianEigenvalueProvider(
            self.domain, self.dirichlet_bc, inverse=False
        )
        provider = SpectrumProvider(self.space, func_provider, eigenval_provider)

        eigenvals = provider.get_all_eigenvalues(5)
        self.assertEqual(len(eigenvals), 5)

    def test_spectrum_provider_eigenvalue_caching(self):
        """Test spectrum provider caches eigenvalues."""
        func_provider = SineFunctionProvider(self.space)
        eigenval_provider = LaplacianEigenvalueProvider(
            self.domain, self.dirichlet_bc, inverse=False
        )
        provider = SpectrumProvider(self.space, func_provider, eigenval_provider)

        eigenval1 = provider.get_eigenvalue(0)
        eigenval2 = provider.get_eigenvalue(0)

        self.assertEqual(eigenval1, eigenval2)

    def test_spectrum_provider_out_of_range(self):
        """Test spectrum provider raises error for out of range indices."""
        func_provider = SineFunctionProvider(self.space)
        eigenval_provider = LaplacianEigenvalueProvider(
            self.domain, self.dirichlet_bc, inverse=False
        )
        provider = SpectrumProvider(self.space, func_provider, eigenval_provider)

        with self.assertRaises(IndexError):
            provider.get_eigenvalue(self.space.dim)

    # === FACTORY FUNCTION TESTS ===

    def test_create_basis_provider_fourier(self):
        """Test creating Fourier basis provider."""
        provider = create_basis_provider(self.space, 'fourier')

        self.assertIsInstance(provider, BasisProvider)
        self.assertEqual(provider.space, self.space)

    def test_create_basis_provider_hat(self):
        """Test creating hat basis provider."""
        provider = create_basis_provider(self.space, 'hat')

        self.assertIsInstance(provider, BasisProvider)

    def test_create_basis_provider_hat_homogeneous(self):
        """Test creating homogeneous hat basis provider."""
        provider = create_basis_provider(self.space, 'hat_homogeneous')

        self.assertIsInstance(provider, BasisProvider)

    def test_create_basis_provider_unsupported(self):
        """Test creating basis provider with unsupported type."""
        with self.assertRaises(ValueError):
            create_basis_provider(self.space, 'unsupported')

    def test_create_spectrum_provider(self):
        """Test creating spectrum provider."""
        func_provider = SineFunctionProvider(self.space)
        eigenval_provider = ZeroEigenvalueProvider()

        provider = create_spectrum_provider(
            self.space, func_provider, eigenval_provider
        )

        self.assertIsInstance(provider, SpectrumProvider)

    def test_create_laplacian_spectrum_provider_dirichlet(self):
        """Test creating Laplacian spectrum provider with Dirichlet BC."""
        provider = create_laplacian_spectrum_provider(
            self.space, self.dirichlet_bc, inverse=False
        )

        self.assertIsInstance(provider, SpectrumProvider)

    def test_create_laplacian_spectrum_provider_neumann(self):
        """Test creating Laplacian spectrum provider with Neumann BC."""
        provider = create_laplacian_spectrum_provider(
            self.space, self.neumann_bc, inverse=False
        )

        self.assertIsInstance(provider, SpectrumProvider)

    def test_create_laplacian_spectrum_provider_periodic(self):
        """Test creating Laplacian spectrum provider with periodic BC."""
        periodic_bc = BoundaryConditions.periodic()
        provider = create_laplacian_spectrum_provider(
            self.space, periodic_bc, inverse=False
        )

        self.assertIsInstance(provider, SpectrumProvider)

    def test_create_laplacian_spectrum_provider_inverse(self):
        """Test creating inverse Laplacian spectrum provider."""
        provider = create_laplacian_spectrum_provider(
            self.space, self.dirichlet_bc, inverse=True
        )

        self.assertIsInstance(provider, SpectrumProvider)

    def test_create_laplacian_spectrum_provider_unsupported_bc(self):
        """Test error for unsupported boundary conditions."""
        invalid_bc = BoundaryConditions.dirichlet()
        invalid_bc.type = 'invalid'

        with self.assertRaises(ValueError):
            create_laplacian_spectrum_provider(
                self.space, invalid_bc, inverse=False
            )

    # === EDGE CASES AND ERROR HANDLING ===

    def test_eigenvalue_provider_get_eigenvalues_zero(self):
        """Test getting zero eigenvalues."""
        provider = ZeroEigenvalueProvider()

        eigenvals = provider.get_eigenvalues(0)
        self.assertEqual(len(eigenvals), 0)

    def test_spectrum_provider_all_eigenvalues_default(self):
        """Test spectrum provider get_all_eigenvalues with default n."""
        func_provider = SineFunctionProvider(self.space)
        eigenval_provider = ZeroEigenvalueProvider()
        provider = SpectrumProvider(self.space, func_provider, eigenval_provider)

        eigenvals = provider.get_all_eigenvalues()  # Default n=space.dim
        self.assertEqual(len(eigenvals), self.space.dim)

    def test_laplacian_eigenvalue_provider_caching(self):
        """Test Laplacian eigenvalue provider caches results."""
        provider = LaplacianEigenvalueProvider(
            self.domain, self.dirichlet_bc, inverse=False
        )

        eigenval1 = provider.get_eigenvalue(0)
        eigenval2 = provider.get_eigenvalue(0)

        self.assertEqual(eigenval1, eigenval2)

    def test_different_domain_lengths(self):
        """Test providers work with different domain lengths."""
        domain2 = IntervalDomain(-2.0, 3.0)  # Length = 5
        space2 = L2Space(5, domain2)

        provider = LaplacianEigenvalueProvider(
            domain2, self.dirichlet_bc, inverse=False
        )

        eigenval = provider.get_eigenvalue(0)
        expected = (1 * np.pi / 5.0) ** 2
        self.assertAlmostEqual(eigenval, expected, places=10)


if __name__ == '__main__':
    if not IMPORTS_SUCCESSFUL:
        print("Skipping tests due to import errors")
        exit(1)
    unittest.main(verbosity=2)
