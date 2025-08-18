"""
Comprehensive unit tests for sola_operator.py

This module provides detailed tests for the SOLAOperator class,
covering initialization, projection, reconstruction, and caching.
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

try:
    from pygeoinf.interval.operators import SOLAOperator
    from pygeoinf.interval.function_providers import (
        NormalModesProvider, SineFunctionProvider, BumpFunctionProvider
    )
    from pygeoinf.interval.interval_domain import IntervalDomain
    from pygeoinf.interval.l2_space import L2Space
    from pygeoinf.hilbert_space import EuclideanSpace
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_SUCCESSFUL = False


@unittest.skipUnless(IMPORTS_SUCCESSFUL, "Required modules not available")
class TestSOLAOperator(unittest.TestCase):
    """Test cases for SOLAOperator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.domain = IntervalDomain(0.0, 1.0)
        self.space = L2Space(10, self.domain)
        self.euclidean_space = EuclideanSpace(5)

    # === INITIALIZATION TESTS ===

    def test_sola_operator_default_init(self):
        """Test SOLA operator initialization with default provider."""
        operator = SOLAOperator(
            self.space, self.euclidean_space, random_state=42
        )

        self.assertEqual(operator.domain, self.space)
        self.assertEqual(operator.codomain, self.euclidean_space)
        self.assertEqual(operator.N_d, 5)
        self.assertIsInstance(operator.function_provider, NormalModesProvider)

    def test_sola_operator_custom_provider_init(self):
        """Test SOLA operator initialization with custom provider."""
        custom_provider = SineFunctionProvider(self.space)
        operator = SOLAOperator(
            self.space, self.euclidean_space,
            function_provider=custom_provider
        )

        self.assertEqual(operator.function_provider, custom_provider)

    def test_sola_operator_caching_enabled(self):
        """Test SOLA operator initialization with caching enabled."""
        operator = SOLAOperator(
            self.space, self.euclidean_space,
            random_state=42, cache_functions=True
        )

        self.assertTrue(operator.cache_functions)
        self.assertIsNotNone(operator._function_cache)

    def test_sola_operator_caching_disabled(self):
        """Test SOLA operator initialization with caching disabled."""
        operator = SOLAOperator(
            self.space, self.euclidean_space,
            random_state=42, cache_functions=False
        )

        self.assertFalse(operator.cache_functions)
        self.assertIsNone(operator._function_cache)

    # === PROJECTION FUNCTION ACCESS TESTS ===

    def testget_kernel_indexed_provider(self):
        """Test getting projection function from indexed provider."""
        custom_provider = SineFunctionProvider(self.space)
        operator = SOLAOperator(
            self.space, self.euclidean_space,
            function_provider=custom_provider
        )

        func = operator.get_kernel(0)
        self.assertIsNotNone(func)
        self.assertEqual(func.space, self.space)

    def testget_kernel_caching(self):
        """Test projection function caching."""
        operator = SOLAOperator(
            self.space, self.euclidean_space,
            random_state=42, cache_functions=True
        )

        func1 = operator.get_kernel(0)
        func2 = operator.get_kernel(0)

        self.assertIs(func1, func2)

    def testget_kernel_no_caching(self):
        """Test projection function without caching."""
        operator = SOLAOperator(
            self.space, self.euclidean_space,
            random_state=42, cache_functions=False
        )

        func1 = operator.get_kernel(0)
        func2 = operator.get_kernel(0)

        # Without caching, functions should be newly generated each time
        # but should have same properties
        self.assertEqual(func1.space, func2.space)

    # === PROJECTION TESTS ===

    def test_project_function_basic(self):
        """Test basic function projection."""
        operator = SOLAOperator(
            self.space, self.euclidean_space, random_state=42
        )

        # Create a simple test function
        def test_func_callable(x):
            return np.sin(2 * np.pi * x)

        from pygeoinf.interval.functions import Function
        test_func = Function(self.space, evaluate_callable=test_func_callable)

        # Project the function
        data = operator._project_function(test_func)

        self.assertEqual(len(data), self.euclidean_space.dim)
        self.assertIsInstance(data, np.ndarray)

    def test_project_function_sine_provider(self):
        """Test function projection with sine provider."""
        custom_provider = SineFunctionProvider(self.space)
        operator = SOLAOperator(
            self.space, self.euclidean_space,
            function_provider=custom_provider
        )

        # Create first sine function as test
        test_func = custom_provider.get_function_by_index(0)

        # Project should give non-zero coefficient for first component
        data = operator._project_function(test_func)

        # First coefficient should be largest due to orthogonality
        self.assertGreater(abs(data[0]), 1e-10)

    def test_project_function_zero_function(self):
        """Test projecting zero function."""
        operator = SOLAOperator(
            self.space, self.euclidean_space, random_state=42
        )

        zero_func = self.space.zero
        data = operator._project_function(zero_func)

        # All coefficients should be close to zero
        np.testing.assert_array_almost_equal(data, np.zeros(5), decimal=10)

    # === RECONSTRUCTION TESTS ===

    def test_reconstruct_function_basic(self):
        """Test basic function reconstruction."""
        operator = SOLAOperator(
            self.space, self.euclidean_space, random_state=42
        )

        # Create test data
        data = np.array([1.0, 0.5, -0.3, 0.0, 0.2])

        # Reconstruct function
        reconstructed = operator._reconstruct_function(data)

        self.assertEqual(reconstructed.space, self.space)

    def test_reconstruct_function_zero_data(self):
        """Test reconstructing from zero data."""
        operator = SOLAOperator(
            self.space, self.euclidean_space, random_state=42
        )

        data = np.zeros(self.euclidean_space.dim)
        reconstructed = operator._reconstruct_function(data)

        # Should be close to zero function
        x_test = np.linspace(0, 1, 10)
        values = reconstructed.evaluate(x_test)
        np.testing.assert_array_almost_equal(values, np.zeros(10), decimal=10)

    def test_reconstruct_function_unit_data(self):
        """Test reconstructing from unit vector data."""
        operator = SOLAOperator(
            self.space, self.euclidean_space, random_state=42
        )

        # Unit vector in first component
        data = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        reconstructed = operator._reconstruct_function(data)

        # Should be equal to first projection function
        first_proj_func = operator.get_kernel(0)

        x_test = np.linspace(0, 1, 10)
        reconstructed_values = reconstructed.evaluate(x_test)
        expected_values = first_proj_func.evaluate(x_test)

        np.testing.assert_array_almost_equal(
            reconstructed_values, expected_values, decimal=10
        )

    def test_reconstruct_function_sparse_data(self):
        """Test reconstruction skips very small coefficients."""
        operator = SOLAOperator(
            self.space, self.euclidean_space, random_state=42
        )

        # Data with one large and one tiny coefficient
        data = np.array([1.0, 1e-16, 0.0, 0.0, 0.0])
        reconstructed = operator._reconstruct_function(data)

        # Should effectively be just the first component
        first_proj_func = operator.get_kernel(0)

        x_test = np.linspace(0, 1, 10)
        reconstructed_values = reconstructed.evaluate(x_test)
        expected_values = first_proj_func.evaluate(x_test)

        np.testing.assert_array_almost_equal(
            reconstructed_values, expected_values, decimal=10
        )

    # === ROUND-TRIP TESTS ===

    def test_project_reconstruct_consistency(self):
        """Test that projection followed by reconstruction is consistent."""
        custom_provider = SineFunctionProvider(self.space)
        operator = SOLAOperator(
            self.space, self.euclidean_space,
            function_provider=custom_provider
        )

        # Use one of the projection functions as test
        original_func = custom_provider.get_function_by_index(0)

        # Project and reconstruct
        data = operator._project_function(original_func)
        reconstructed = operator._reconstruct_function(data)

        # Should be similar (but not exact due to truncation)
        x_test = np.linspace(0, 1, 10)
        original_values = original_func.evaluate(x_test)
        reconstructed_values = reconstructed.evaluate(x_test)

        # First coefficient should dominate, so should be fairly close
        correlation = np.corrcoef(original_values, reconstructed_values)[0, 1]
        self.assertGreater(correlation, 0.8)

    # === UTILITY METHOD TESTS ===

    def testget_kernels(self):
        """Test getting all projection functions."""
        operator = SOLAOperator(
            self.space, self.euclidean_space, random_state=42
        )

        functions = operator.get_kernels()

        self.assertEqual(len(functions), self.euclidean_space.dim)
        for func in functions:
            self.assertEqual(func.space, self.space)

    def test_evaluate_kernels(self):
        """Test evaluating all projection functions at points."""
        operator = SOLAOperator(
            self.space, self.euclidean_space, random_state=42
        )

        x = np.linspace(0, 1, 20)
        values = operator.evaluate_kernel(x)

        self.assertEqual(values.shape, (self.euclidean_space.dim, len(x)))

    def test_compute_gram_matrix(self):
        """Test computing Gram matrix of projection functions."""
        operator = SOLAOperator(
            self.space, self.euclidean_space, random_state=42
        )

        gram = operator.compute_gram_matrix()

        self.assertEqual(gram.shape, (self.euclidean_space.dim, self.euclidean_space.dim))

        # Should be symmetric
        np.testing.assert_array_almost_equal(gram, gram.T, decimal=10)

        # Diagonal should be positive (norms squared)
        self.assertTrue(np.all(np.diag(gram) > 0))

    def test_compute_gram_matrix_orthogonal_functions(self):
        """Test Gram matrix with orthogonal functions."""
        # Use sine functions which are orthogonal
        custom_provider = SineFunctionProvider(self.space)
        euclidean_space_small = EuclideanSpace(3)  # Use fewer functions
        operator = SOLAOperator(
            self.space, euclidean_space_small,
            function_provider=custom_provider
        )

        gram = operator.compute_gram_matrix()

        # Off-diagonal elements should be small (orthogonal)
        off_diagonal_max = np.max(np.abs(gram - np.diag(np.diag(gram))))
        self.assertLess(off_diagonal_max, 1e-10)

    # === CACHE MANAGEMENT TESTS ===

    def test_clear_cache(self):
        """Test clearing function cache."""
        operator = SOLAOperator(
            self.space, self.euclidean_space,
            random_state=42, cache_functions=True
        )

        # Access a function to populate cache
        operator.get_kernel(0)
        self.assertEqual(len(operator._function_cache), 1)

        # Clear cache
        operator.clear_cache()
        self.assertEqual(len(operator._function_cache), 0)

    def test_clear_cache_no_caching(self):
        """Test clearing cache when caching is disabled."""
        operator = SOLAOperator(
            self.space, self.euclidean_space,
            random_state=42, cache_functions=False
        )

        # Should not raise error
        operator.clear_cache()

    def test_get_cache_info_with_caching(self):
        """Test getting cache info when caching is enabled."""
        operator = SOLAOperator(
            self.space, self.euclidean_space,
            random_state=42, cache_functions=True
        )

        # Initially empty
        info = operator.get_cache_info()
        self.assertTrue(info["caching_enabled"])
        self.assertEqual(info["cached_functions"], 0)

        # Access some functions
        operator.get_kernel(0)
        operator.get_kernel(1)

        info = operator.get_cache_info()
        self.assertEqual(info["cached_functions"], 2)
        self.assertAlmostEqual(info["cache_coverage"], 2/5, places=10)

    def test_get_cache_info_no_caching(self):
        """Test getting cache info when caching is disabled."""
        operator = SOLAOperator(
            self.space, self.euclidean_space,
            random_state=42, cache_functions=False
        )

        info = operator.get_cache_info()
        self.assertFalse(info["caching_enabled"])

    # === STRING REPRESENTATION TESTS ===

    def test_str_representation(self):
        """Test string representation of SOLA operator."""
        operator = SOLAOperator(
            self.space, self.euclidean_space, random_state=42
        )

        str_repr = str(operator)

        self.assertIn("SOLAOperator", str_repr)
        self.assertIn("NormalModesProvider", str_repr)
        self.assertIn(str(self.euclidean_space.dim), str_repr)
        self.assertIn(str(self.space.dim), str_repr)

    # === EDGE CASES AND ERROR HANDLING ===

    def test_different_euclidean_dimensions(self):
        """Test SOLA operator with different output dimensions."""
        dimensions = [1, 3, 10, 20]

        for dim in dimensions:
            euclidean_space = EuclideanSpace(dim)
            operator = SOLAOperator(
                self.space, euclidean_space, random_state=42
            )

            self.assertEqual(operator.N_d, dim)

    def test_project_reconstruct_different_providers(self):
        """Test projection/reconstruction with different providers."""
        providers = [
            SineFunctionProvider(self.space),
            BumpFunctionProvider(self.space),
        ]

        for provider in providers:
            operator = SOLAOperator(
                self.space, self.euclidean_space,
                function_provider=provider
            )

            # Test basic projection
            zero_func = self.space.zero
            data = operator._project_function(zero_func)

            self.assertEqual(len(data), self.euclidean_space.dim)

    def test_reproducibility_with_seed(self):
        """Test operator reproducibility with random seed."""
        operator1 = SOLAOperator(
            self.space, self.euclidean_space, random_state=42
        )
        operator2 = SOLAOperator(
            self.space, self.euclidean_space, random_state=42
        )

        # Should produce same projection functions
        func1 = operator1.get_kernel(0)
        func2 = operator2.get_kernel(0)

        x_test = np.linspace(0, 1, 10)
        values1 = func1.evaluate(x_test)
        values2 = func2.evaluate(x_test)

        np.testing.assert_array_almost_equal(values1, values2, decimal=10)

    def test_large_output_dimension(self):
        """Test SOLA operator with large output dimension."""
        large_euclidean_space = EuclideanSpace(100)
        operator = SOLAOperator(
            self.space, large_euclidean_space, random_state=42
        )

        self.assertEqual(operator.N_d, 100)

        # Should still work for basic operations
        zero_func = self.space.zero
        data = operator._project_function(zero_func)
        self.assertEqual(len(data), 100)


if __name__ == '__main__':
    if not IMPORTS_SUCCESSFUL:
        print("Skipping tests due to import errors")
        exit(1)
    unittest.main(verbosity=2)
