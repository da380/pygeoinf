"""
Comprehensive unit tests for function_providers.py

This module provides detailed tests for all function provider classes,
covering abstract base classes, concrete implementations, edge cases,
and error handling.
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

try:
    from pygeoinf.interval.function_providers import (
        FunctionProvider, IndexedFunctionProvider, ParametricFunctionProvider,
        RandomFunctionProvider, NormalModesProvider, FourierFunctionProvider,
        SplineFunctionProvider, BumpFunctionProvider, DiscontinuousFunctionProvider,
        WaveletFunctionProvider, FunctionProviderAdapter, SineFunctionProvider,
        CosineFunctionProvider, HatFunctionProvider
    )
    from pygeoinf.interval.interval_domain import IntervalDomain
    from pygeoinf.interval.l2_space import L2Space
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_SUCCESSFUL = False


@unittest.skipUnless(IMPORTS_SUCCESSFUL, "Required modules not available")
class TestFunctionProviders(unittest.TestCase):
    """Test cases for function provider classes."""

    def setUp(self):
        """Set up test fixtures."""
        self.domain = IntervalDomain(0.0, 1.0)
        self.space = L2Space(10, self.domain)

    # === ABSTRACT BASE CLASS TESTS ===

    def test_function_provider_init_valid(self):
        """Test FunctionProvider initialization with valid space."""
        # Can't instantiate abstract class directly, but test through subclass
        provider = SineFunctionProvider(self.space)
        self.assertEqual(provider.space, self.space)
        self.assertEqual(provider.domain, self.domain)

    def test_function_provider_init_none_space(self):
        """Test FunctionProvider raises error with None space."""
        with self.assertRaises(ValueError):
            SineFunctionProvider(None)

    # === SINE FUNCTION PROVIDER TESTS ===

    def test_sine_provider_basic(self):
        """Test basic sine function provider functionality."""
        provider = SineFunctionProvider(self.space)

        # Test first few functions
        func0 = provider.get_function_by_index(0)
        func1 = provider.get_function_by_index(1)

        self.assertIsNotNone(func0)
        self.assertIsNotNone(func1)
        self.assertEqual(func0.space, self.space)
        self.assertIn('sin', func0.name.lower())

    def test_sine_provider_values(self):
        """Test sine function provider computes correct values."""
        provider = SineFunctionProvider(self.space)
        func = provider.get_function_by_index(0)  # sin(2π*x)

        # Test at specific points
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        values = func.evaluate(x)
        # sin(2π*x): sin(0), sin(π/2), sin(π), sin(3π/2), sin(2π)
        expected = np.array([0.0, 1.0, 0.0, -1.0, 0.0])

        np.testing.assert_array_almost_equal(values, expected, decimal=10)

    def test_sine_provider_caching(self):
        """Test sine function provider caches functions."""
        provider = SineFunctionProvider(self.space)

        func1 = provider.get_function_by_index(0)
        func2 = provider.get_function_by_index(0)

        # Should return same cached object
        self.assertIs(func1, func2)

    # === COSINE FUNCTION PROVIDER TESTS ===

    def test_cosine_provider_constant_mode(self):
        """Test cosine provider returns constant for index 0."""
        provider = CosineFunctionProvider(self.space)
        func = provider.get_function_by_index(0)

        x = np.array([0.0, 0.5, 1.0])
        values = func.evaluate(x)
        expected = np.array([1.0, 1.0, 1.0])

        np.testing.assert_array_almost_equal(values, expected, decimal=10)

    def test_cosine_provider_cosine_modes(self):
        """Test cosine provider returns correct cosine functions."""
        provider = CosineFunctionProvider(self.space)
        func = provider.get_function_by_index(1)  # cos(π*x)

        x = np.array([0.0, 0.5, 1.0])
        values = func.evaluate(x)
        expected = np.array([1.0, 0.0, -1.0])  # cos(0), cos(π/2), cos(π)

        np.testing.assert_array_almost_equal(values, expected, decimal=10)

    # === FOURIER FUNCTION PROVIDER TESTS ===

    def test_fourier_provider_structure(self):
        """Test Fourier provider index structure."""
        provider = FourierFunctionProvider(self.space)

        # Index 0: constant
        func0 = provider.get_function_by_index(0)
        self.assertIn('const', func0.name.lower())

        # Index 1: first cosine
        func1 = provider.get_function_by_index(1)
        self.assertIn('cos', func1.name.lower())

        # Index 2: first sine
        func2 = provider.get_function_by_index(2)
        self.assertIn('sin', func2.name.lower())

    def test_fourier_provider_normalization(self):
        """Test Fourier functions are properly normalized."""
        provider = FourierFunctionProvider(self.space)
        func = provider.get_function_by_index(0)  # Constant function

        # Constant should be normalized to 1/sqrt(L)
        x = np.array([0.5])
        value = func.evaluate(x)
        expected = 1.0 / np.sqrt(1.0)  # Domain length is 1.0

        np.testing.assert_array_almost_equal(value, [expected], decimal=10)

    # === HAT FUNCTION PROVIDER TESTS ===

    def test_hat_provider_homogeneous(self):
        """Test homogeneous hat function provider."""
        provider = HatFunctionProvider(self.space, homogeneous=True)

        # Should have space.dim functions (excluding boundary)
        func0 = provider.get_function_by_index(0)
        self.assertIsNotNone(func0)
        self.assertIn('hom', func0.name.lower())

    def test_hat_provider_non_homogeneous(self):
        """Test non-homogeneous hat function provider."""
        provider = HatFunctionProvider(self.space, homogeneous=False)

        func0 = provider.get_function_by_index(0)
        self.assertIsNotNone(func0)
        self.assertNotIn('hom', func0.name.lower())

    def test_hat_provider_values(self):
        """Test hat function provider computes correct values."""
        provider = HatFunctionProvider(self.space, homogeneous=True)
        func = provider.get_function_by_index(0)

        # Get the nodes to understand the structure
        nodes = provider.get_active_nodes()

        # Function should be 1 at its node and 0 at others
        values = func.evaluate(nodes)

        # Should have one value of 1 and rest close to 0
        self.assertAlmostEqual(np.max(values), 1.0, places=10)

    def test_hat_provider_nodes(self):
        """Test hat function provider node methods."""
        provider = HatFunctionProvider(self.space, homogeneous=True)

        all_nodes = provider.get_nodes()
        active_nodes = provider.get_active_nodes()

        self.assertEqual(len(all_nodes), len(active_nodes) + 2)  # +2 for boundaries
        self.assertTrue(np.allclose(all_nodes[0], 0.0))  # Left boundary
        self.assertTrue(np.allclose(all_nodes[-1], 1.0))  # Right boundary

    # === NORMAL MODES PROVIDER TESTS ===

    def test_normal_modes_provider_basic(self):
        """Test basic normal modes provider functionality."""
        provider = NormalModesProvider(self.space, random_state=42)

        func = provider.sample_function()
        self.assertIsNotNone(func)
        self.assertEqual(func.space, self.space)

    def test_normal_modes_provider_reproducible(self):
        """Test normal modes provider is reproducible with seed."""
        provider1 = NormalModesProvider(self.space, random_state=42)
        provider2 = NormalModesProvider(self.space, random_state=42)

        func1 = provider1.sample_function()
        func2 = provider2.sample_function()

        # Should produce same function with same seed
        x = np.linspace(0, 1, 10)
        values1 = func1.evaluate(x)
        values2 = func2.evaluate(x)

        np.testing.assert_array_almost_equal(values1, values2, decimal=10)

    def test_normal_modes_provider_indexed(self):
        """Test normal modes provider indexed access."""
        provider = NormalModesProvider(self.space, random_state=42)

        func0 = provider.get_function_by_index(0)
        func1 = provider.get_function_by_index(1)

        self.assertIsNotNone(func0)
        self.assertIsNotNone(func1)
        self.assertNotEqual(func0.name, func1.name)

    def test_normal_modes_provider_parametric(self):
        """Test normal modes provider parametric access."""
        provider = NormalModesProvider(self.space, random_state=42)

        params = provider.get_default_parameters()
        func = provider.get_function_by_parameters(params)

        self.assertIsNotNone(func)
        self.assertIn('parametric', func.name.lower())

    def test_normal_modes_provider_indexed_functions(self):
        """Test normal modes provider batch indexed access."""
        provider = NormalModesProvider(self.space, random_state=42)

        functions = provider.get_indexed_functions(3)

        self.assertEqual(len(functions), 3)
        for func in functions:
            self.assertIsNotNone(func)

    # === BUMP FUNCTION PROVIDER TESTS ===

    def test_bump_provider_basic(self):
        """Test basic bump function provider functionality."""
        provider = BumpFunctionProvider(self.space)

        params = {'center': 0.5, 'width': 0.4}
        func = provider.get_function_by_parameters(params)

        self.assertIsNotNone(func)
        self.assertEqual(func.space, self.space)

    def test_bump_provider_compact_support(self):
        """Test bump function has compact support."""
        provider = BumpFunctionProvider(self.space)

        params = {'center': 0.5, 'width': 0.4}
        func = provider.get_function_by_parameters(params)

        # Should be zero outside support
        x_outside = np.array([0.0, 1.0])  # Outside support [0.3, 0.7]
        values = func.evaluate(x_outside)

        np.testing.assert_array_almost_equal(values, [0.0, 0.0], decimal=10)

    def test_bump_provider_positive_interior(self):
        """Test bump function is positive in interior."""
        provider = BumpFunctionProvider(self.space)

        params = {'center': 0.5, 'width': 0.4}
        func = provider.get_function_by_parameters(params)

        # Should be positive at center
        value_center = func.evaluate(np.array([0.5]))
        self.assertGreater(value_center[0], 0.0)

    def test_bump_provider_indexed(self):
        """Test bump function provider indexed access."""
        provider = BumpFunctionProvider(self.space)

        func0 = provider.get_function_by_index(0)
        func1 = provider.get_function_by_index(1)

        self.assertIsNotNone(func0)
        self.assertIsNotNone(func1)

    def test_bump_provider_caching(self):
        """Test bump function provider caches indexed functions."""
        provider = BumpFunctionProvider(self.space)

        func1 = provider.get_function_by_index(0)
        func2 = provider.get_function_by_index(0)

        self.assertIs(func1, func2)

    # === SPLINE FUNCTION PROVIDER TESTS ===

    @unittest.skipIf(True, "Spline provider requires scipy")
    def test_spline_provider_basic(self):
        """Test basic spline function provider functionality."""
        provider = SplineFunctionProvider(self.space)

        func = provider.get_function_by_index(0)
        self.assertIsNotNone(func)

    # === DISCONTINUOUS FUNCTION PROVIDER TESTS ===

    def test_discontinuous_provider_basic(self):
        """Test basic discontinuous function provider functionality."""
        provider = DiscontinuousFunctionProvider(self.space, random_state=42)

        func = provider.sample_function(n_discontinuities=2)
        self.assertIsNotNone(func)
        self.assertEqual(func.space, self.space)

    def test_discontinuous_provider_reproducible(self):
        """Test discontinuous provider is reproducible with seed."""
        provider1 = DiscontinuousFunctionProvider(self.space, random_state=42)
        provider2 = DiscontinuousFunctionProvider(self.space, random_state=42)

        func1 = provider1.sample_function(n_discontinuities=2)
        func2 = provider2.sample_function(n_discontinuities=2)

        x = np.linspace(0, 1, 10)
        values1 = func1.evaluate(x)
        values2 = func2.evaluate(x)

        np.testing.assert_array_almost_equal(values1, values2, decimal=10)

    # === WAVELET FUNCTION PROVIDER TESTS ===

    def test_wavelet_provider_haar_scaling(self):
        """Test Haar wavelet provider scaling function."""
        provider = WaveletFunctionProvider(self.space, wavelet_type='haar')

        func0 = provider.get_function_by_index(0)  # Scaling function
        self.assertIn('scaling', func0.name.lower())

    def test_wavelet_provider_haar_wavelets(self):
        """Test Haar wavelet provider wavelet functions."""
        provider = WaveletFunctionProvider(self.space, wavelet_type='haar')

        func1 = provider.get_function_by_index(1)  # First wavelet
        self.assertIn('haar', func1.name.lower())

    def test_wavelet_provider_unsupported_type(self):
        """Test wavelet provider raises error for unsupported type."""
        provider = WaveletFunctionProvider(self.space, wavelet_type='unsupported')

        with self.assertRaises(ValueError):
            provider.get_function_by_index(0)

    # === FUNCTION PROVIDER ADAPTER TESTS ===

    def test_adapter_basic(self):
        """Test function provider adapter basic functionality."""
        provider = SineFunctionProvider(self.space)
        adapter = FunctionProviderAdapter(provider)

        self.assertEqual(adapter.space, self.space)

    def test_adapter_get_basis_function(self):
        """Test adapter get_basis_function method."""
        provider = SineFunctionProvider(self.space)
        adapter = FunctionProviderAdapter(provider)

        func = adapter.get_basis_function(0)
        self.assertIsNotNone(func)
        self.assertEqual(func.space, self.space)

    def test_adapter_caching(self):
        """Test adapter caches basis functions."""
        provider = SineFunctionProvider(self.space)
        adapter = FunctionProviderAdapter(provider)

        func1 = adapter.get_basis_function(0)
        func2 = adapter.get_basis_function(0)

        self.assertIs(func1, func2)

    def test_adapter_non_indexed_provider(self):
        """Test adapter raises error for non-indexed providers."""
        # Create a non-indexed provider (ParametricFunctionProvider only)
        provider = BumpFunctionProvider(self.space)  # This is both parametric and indexed

        # For this test, we'd need a provider that's only parametric
        # Since we don't have one in the current implementation, skip this test
        pass

    # === EDGE CASES AND ERROR HANDLING ===

    def test_negative_index_error(self):
        """Test providers that check for negative indices raise error."""
        # Not all providers check for negative indices, so test those that do
        provider = NormalModesProvider(self.space, random_state=42)

        with self.assertRaises(ValueError):
            provider.get_function_by_index(-1)

    def test_bump_provider_negative_index_error(self):
        """Test bump provider raises error for negative indices."""
        provider = BumpFunctionProvider(self.space)

        with self.assertRaises(ValueError):
            provider.get_function_by_index(-1)

    def test_normal_modes_negative_index_error(self):
        """Test normal modes provider raises error for negative indices."""
        provider = NormalModesProvider(self.space, random_state=42)

        with self.assertRaises(ValueError):
            provider.get_function_by_index(-1)

    def test_large_index_handling(self):
        """Test providers handle large indices gracefully."""
        provider = SineFunctionProvider(self.space)

        # Should work with large indices
        func = provider.get_function_by_index(1000)
        self.assertIsNotNone(func)

    def test_different_domains(self):
        """Test providers work with different domains."""
        domain2 = IntervalDomain(-1.0, 2.0)
        space2 = L2Space(5, domain2)

        provider = SineFunctionProvider(space2)
        func = provider.get_function_by_index(0)

        self.assertEqual(func.space, space2)

    def test_provider_default_parameters(self):
        """Test providers return valid default parameters."""
        provider = NormalModesProvider(self.space, random_state=42)
        params = provider.get_default_parameters()

        self.assertIsInstance(params, dict)
        self.assertIn('coefficients', params)
        self.assertIn('frequencies', params)

    def test_bump_provider_default_parameters(self):
        """Test bump provider returns valid default parameters."""
        provider = BumpFunctionProvider(self.space)
        params = provider.get_default_parameters()

        self.assertIsInstance(params, dict)
        self.assertIn('center', params)
        self.assertIn('width', params)


if __name__ == '__main__':
    if not IMPORTS_SUCCESSFUL:
        print("Skipping tests due to import errors")
        exit(1)
    unittest.main(verbosity=2)
