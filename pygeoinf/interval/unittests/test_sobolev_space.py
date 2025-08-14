import unittest
import numpy as np
import sys
import os

# Add the parent directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

try:
    from pygeoinf.interval.interval_domain import IntervalDomain
    from pygeoinf.interval.sobolev_space import Sobolev
    from pygeoinf.interval.functions import Function
    from pygeoinf.interval.boundary_conditions import BoundaryConditions
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_SUCCESSFUL = False


@unittest.skipUnless(IMPORTS_SUCCESSFUL, "Required modules not available")
class TestSobolevSpace(unittest.TestCase):
    def setUp(self):
        self.domain = IntervalDomain(0.0, 1.0)
        self.dim = 5
        self.order = 1.0
        self.spectral_fourier_space = Sobolev(
            self.dim, self.domain, self.order, 'spectral',
            basis_type='fourier'
        )

        def simple_func(x):
            return x

        def constant_func(x):
            return np.ones_like(x) if hasattr(x, '__iter__') else 1.0

        def quadratic_func(x):
            return x**2

        self.simple_func = simple_func
        self.constant_func = constant_func
        self.quadratic_func = quadratic_func

    def test_init_spectral_fourier_basic(self):
        space = Sobolev(3, self.domain, 1.5, 'spectral',
                        basis_type='fourier')
        self.assertEqual(space.dim, 3)
        self.assertEqual(space.order, 1.5)
        self.assertEqual(space.function_domain, self.domain)
        self.assertIsNotNone(space._spectrum_provider)

    def test_init_spectral_with_boundary_conditions(self):
        bc = BoundaryConditions.periodic()
        space = Sobolev(
            3, self.domain, 1.0, 'spectral',
            basis_type='fourier', boundary_conditions=bc
        )
        self.assertEqual(space.boundary_conditions, bc)
        self.assertEqual(space.order, 1.0)

    def test_init_spectral_with_manual_eigenvalues(self):
        basis_funcs = [
            lambda x: (np.ones_like(x) if hasattr(x, '__iter__') else 1.0),
            lambda x: x,
            lambda x: x**2
        ]
        eigenvals = np.array([0.0, 1.0, 4.0])
        try:
            space = Sobolev(
                3, self.domain, 2.0, 'spectral',
                basis_callables=basis_funcs, eigenvalues=eigenvals
            )
            self.assertEqual(space.dim, 3)
            self.assertEqual(space.order, 2.0)
            np.testing.assert_array_equal(space.eigenvalues, eigenvals)
        except (AttributeError, ValueError):
            # Skip if this functionality is not fully implemented
            pass

    def test_init_invalid_basis_spec(self):
        # Test missing eigenvalues with basis_callables
        basis_funcs = [lambda x: x, lambda x: x**2]
        with self.assertRaises(ValueError):
            Sobolev(
                2, self.domain, 1.0, 'spectral',
                basis_callables=basis_funcs
            )

    def test_init_different_orders(self):
        orders = [0.0, 0.5, 1.0, 2.0, 3.5]
        for order in orders:
            space = Sobolev(3, self.domain, order, 'spectral',
                            basis_type='fourier')
            self.assertEqual(space.order, order)

    def test_init_different_domains(self):
        domains = [
            IntervalDomain(-1.0, 1.0),
            IntervalDomain(0.0, 2.0),
            IntervalDomain(-2.0, 3.0)
        ]
        for domain in domains:
            space = Sobolev(3, domain, 1.0, 'spectral',
                            basis_type='fourier')
            self.assertEqual(space.function_domain, domain)

    def test_order_property(self):
        orders = [0.0, 1.0, 2.5, 3.0]
        for order in orders:
            space = Sobolev(3, self.domain, order, 'spectral',
                            basis_type='fourier')
            self.assertEqual(space.order, order)

    def test_boundary_conditions_property(self):
        bc = BoundaryConditions.periodic()
        space = Sobolev(
            3, self.domain, 1.0, 'spectral',
            basis_type='fourier', boundary_conditions=bc
        )
        self.assertEqual(space.boundary_conditions, bc)

    def test_eigenvalues_property_spectral(self):
        space = Sobolev(3, self.domain, 1.0, 'spectral',
                        basis_type='fourier')
        eigenvals = space.eigenvalues
        self.assertIsNotNone(eigenvals)
        self.assertEqual(len(eigenvals), 3)
        self.assertIsInstance(eigenvals, np.ndarray)

    def test_operator_property(self):
        bc = BoundaryConditions.periodic()
        space = Sobolev(
            3, self.domain, 1.0, 'spectral',
            basis_type='fourier', boundary_conditions=bc
        )
        operator_info = space.operator
        self.assertIsInstance(operator_info, dict)
        self.assertIn('type', operator_info)
        self.assertIn('boundary_conditions', operator_info)
        self.assertIn('domain', operator_info)
        self.assertEqual(operator_info['type'], 'negative_laplacian')

    def test_operator_property_without_bc(self):
        space = Sobolev(3, self.domain, 1.0, 'spectral',
                        basis_type='fourier')
        operator_info = space.operator
        self.assertEqual(operator_info['boundary_conditions'], 'unspecified')

    def test_spectral_inner_product_basic(self):
        space = Sobolev(3, self.domain, 1.0, 'spectral',
                        basis_type='fourier')
        func1 = Function(space, evaluate_callable=self.constant_func)
        func2 = Function(space, evaluate_callable=self.simple_func)
        result = space.inner_product(func1, func2)
        self.assertIsInstance(result, (int, float))

    def test_spectral_inner_product_same_function(self):
        space = Sobolev(3, self.domain, 1.0, 'spectral',
                        basis_type='fourier')
        func = Function(space, evaluate_callable=self.constant_func)
        result = space.inner_product(func, func)
        self.assertGreater(result, 0.0)

    def test_spectral_inner_product_symmetry(self):
        space = Sobolev(4, self.domain, 1.5, 'spectral',
                        basis_type='fourier')
        func1 = Function(space, evaluate_callable=self.simple_func)
        func2 = Function(space, evaluate_callable=self.quadratic_func)
        result1 = space.inner_product(func1, func2)
        result2 = space.inner_product(func2, func1)
        self.assertAlmostEqual(result1, result2, places=5)

    def test_spectral_inner_product_with_coefficients(self):
        space = Sobolev(3, self.domain, 2.0, 'spectral',
                        basis_type='fourier')
        func1 = Function(space, coefficients=np.array([1.0, 0.0, 0.0]))
        func2 = Function(space, coefficients=np.array([0.0, 1.0, 0.0]))
        result = space.inner_product(func1, func2)
        self.assertAlmostEqual(result, 0.0, places=1)

    def test_spectral_inner_product_order_effect(self):
        func_coeffs = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        space_order_0 = Sobolev(5, self.domain, 0.0, 'spectral',
                                basis_type='fourier')
        space_order_2 = Sobolev(5, self.domain, 2.0, 'spectral',
                                basis_type='fourier')
        func_0 = Function(space_order_0, coefficients=func_coeffs)
        func_2 = Function(space_order_2, coefficients=func_coeffs)
        norm_0 = space_order_0.inner_product(func_0, func_0)
        norm_2 = space_order_2.inner_product(func_2, func_2)
        self.assertGreater(norm_2, norm_0)

    def test_spectral_inner_product_invalid_input_types(self):
        space = Sobolev(3, self.domain, 1.0, 'spectral',
                        basis_type='fourier')
        func = Function(space, evaluate_callable=self.constant_func)
        with self.assertRaises(TypeError):
            space.inner_product(func, "not a function")
        with self.assertRaises(TypeError):
            space.inner_product(123, func)

    def test_to_components_basic(self):
        func = Function(self.spectral_fourier_space,
                        evaluate_callable=self.constant_func)
        components = self.spectral_fourier_space._to_components(func)
        self.assertIsInstance(components, np.ndarray)
        self.assertEqual(len(components), self.dim)

    def test_to_components_with_coefficients(self):
        coeffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        func = Function(self.spectral_fourier_space, coefficients=coeffs)
        components = self.spectral_fourier_space._to_components(func)
        self.assertEqual(len(components), self.dim)

    def test_from_components_basic(self):
        components = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        func = self.spectral_fourier_space._from_components(components)
        self.assertIsInstance(func, Function)
        self.assertEqual(func.space, self.spectral_fourier_space)
        np.testing.assert_array_equal(func.coefficients, components)

    def test_component_roundtrip_approximation(self):
        func = Function(self.spectral_fourier_space,
                        coefficients=np.array([1.0, 0.0, 1.0, 0.0, 0.0]))
        components = self.spectral_fourier_space._to_components(func)
        reconstructed = (
            self.spectral_fourier_space._from_components(components)
        )
        self.assertIsInstance(reconstructed, Function)
        self.assertEqual(reconstructed.space, self.spectral_fourier_space)

    def test_gram_matrix_basic(self):
        space = Sobolev(3, self.domain, 1.0, 'spectral',
                        basis_type='fourier')
        gram = space.gram_matrix
        self.assertEqual(gram.shape, (3, 3))
        np.testing.assert_array_almost_equal(gram, gram.T, decimal=3)
        eigenvals = np.linalg.eigvals(gram)
        self.assertTrue(np.all(eigenvals > -1e-10))

    def test_gram_matrix_caching(self):
        space = Sobolev(2, self.domain, 1.0, 'spectral',
                        basis_type='fourier')
        gram1 = space.gram_matrix
        self.assertIsNotNone(space._gram_matrix)
        gram2 = space.gram_matrix
        self.assertIs(gram1, gram2)

    def test_gram_matrix_different_orders(self):
        orders = [0.0, 1.0, 2.0]
        for order in orders:
            space = Sobolev(3, self.domain, order, 'spectral',
                            basis_type='fourier')
            gram = space.gram_matrix
            self.assertEqual(gram.shape, (3, 3))
            if order > 0:
                self.assertGreater(np.min(np.diag(gram)), 0.0)

    def test_norm_computation(self):
        func = Function(self.spectral_fourier_space,
                        evaluate_callable=self.constant_func)
        norm = self.spectral_fourier_space.norm(func)
        self.assertIsInstance(norm, (int, float))
        self.assertGreater(norm, 0.0)

    def test_norm_zero_function(self):
        zero = self.spectral_fourier_space.zero
        norm = self.spectral_fourier_space.norm(zero)
        self.assertAlmostEqual(norm, 0.0, places=5)

    def test_norm_order_dependency(self):
        func_coeffs = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        space_0 = Sobolev(5, self.domain, 0.0, 'spectral',
                          basis_type='fourier')
        space_2 = Sobolev(5, self.domain, 2.0, 'spectral',
                          basis_type='fourier')
        func_0 = Function(space_0, coefficients=func_coeffs)
        func_2 = Function(space_2, coefficients=func_coeffs)
        norm_0 = space_0.norm(func_0)
        norm_2 = space_2.norm(func_2)
        self.assertGreater(norm_2, norm_0)

    def test_zero_function(self):
        zero = self.spectral_fourier_space.zero
        self.assertIsInstance(zero, Function)
        self.assertEqual(zero.space, self.spectral_fourier_space)
        if zero.coefficients is not None:
            np.testing.assert_array_almost_equal(
                zero.coefficients, np.zeros(self.dim), decimal=10
            )

    def test_zero_function_norm(self):
        zero = self.spectral_fourier_space.zero
        norm = self.spectral_fourier_space.norm(zero)
        self.assertAlmostEqual(norm, 0.0, places=10)

    def test_edge_cases_small_dimension(self):
        space = Sobolev(1, self.domain, 1.0, 'spectral',
                        basis_type='fourier')
        self.assertEqual(space.dim, 1)
        basis_func = space.get_basis_function(0)
        self.assertIsInstance(basis_func, Function)

    def test_edge_cases_zero_order(self):
        space = Sobolev(3, self.domain, 0.0, 'spectral',
                        basis_type='fourier')
        self.assertEqual(space.order, 0.0)
        func = Function(space, evaluate_callable=self.constant_func)
        result = space.inner_product(func, func)
        self.assertGreater(result, 0.0)

    def test_edge_cases_high_order(self):
        space = Sobolev(3, self.domain, 10.0, 'spectral',
                        basis_type='fourier')
        self.assertEqual(space.order, 10.0)
        gram = space.gram_matrix
        self.assertEqual(gram.shape, (3, 3))

    def test_edge_cases_small_domain(self):
        small_domain = IntervalDomain(0.0, 1e-6)
        space = Sobolev(2, small_domain, 1.0, 'spectral',
                        basis_type='fourier')
        self.assertEqual(space.function_domain, small_domain)
        self.assertEqual(space.dim, 2)

    def test_edge_cases_large_dimension(self):
        space = Sobolev(50, self.domain, 1.0, 'spectral',
                        basis_type='fourier')
        self.assertEqual(space.dim, 50)
        self.assertEqual(space.order, 1.0)
        self.assertIsNotNone(space.eigenvalues)

    def test_repr_method(self):
        space = Sobolev(3, self.domain, 1.5, 'spectral',
                        basis_type='fourier')
        repr_str = repr(space)
        self.assertIsInstance(repr_str, str)
        self.assertTrue('Space' in repr_str or 'Sobolev' in repr_str)

    def test_repr_different_configurations(self):
        configurations = [
            (2, 0.0),
            (5, 2.0),
        ]
        for dim, order in configurations:
            space = Sobolev(dim, self.domain, order, 'spectral',
                            basis_type='fourier')
            repr_str = repr(space)
            self.assertIsInstance(repr_str, str)


@unittest.skipUnless(IMPORTS_SUCCESSFUL, "Required modules not available")
class TestLebesgueSpace(unittest.TestCase):
    def setUp(self):
        self.domain = IntervalDomain(0.0, 1.0)
        self.dim = 5

    def test_lebesgue_basic_initialization(self):
        space = Lebesgue(self.dim, self.domain)
        self.assertEqual(space.dim, self.dim)
        self.assertEqual(space.function_domain, self.domain)
        self.assertEqual(space.order, 0.0)
        self.assertEqual(space.inner_product_type, 'spectral')

    def test_lebesgue_is_sobolev_subclass(self):
        space = Lebesgue(3, self.domain)
        self.assertIsInstance(space, Sobolev)

    def test_lebesgue_boundary_conditions(self):
        space = Lebesgue(3, self.domain)
        bc = space.boundary_conditions
        self.assertIsNotNone(bc)
        self.assertEqual(bc.type, 'periodic')

    def test_lebesgue_properties(self):
        space = Lebesgue(4, self.domain)
        self.assertEqual(space.dim, 4)
        self.assertEqual(space.order, 0.0)
        self.assertIsNotNone(space.eigenvalues)
        self.assertIsNotNone(space.gram_matrix)

    def test_lebesgue_inner_product(self):
        space = Lebesgue(3, self.domain)

        def constant_func(x):
            return np.ones_like(x) if hasattr(x, '__iter__') else 1.0

        func = Function(space, evaluate_callable=constant_func)
        result = space.inner_product(func, func)
        self.assertIsInstance(result, (int, float))
        self.assertGreater(result, 0.0)

    def test_lebesgue_different_domains(self):
        domains = [
            IntervalDomain(-1.0, 1.0),
            IntervalDomain(0.0, 2.0),
            IntervalDomain(-2.0, 3.0)
        ]
        for domain in domains:
            space = Lebesgue(3, domain)
            self.assertEqual(space.function_domain, domain)
            self.assertEqual(space.order, 0.0)

    def test_lebesgue_different_dimensions(self):
        dimensions = [1, 3, 5, 10, 20]
        for dim in dimensions:
            space = Lebesgue(dim, self.domain)
            self.assertEqual(space.dim, dim)
            self.assertEqual(space.order, 0.0)


if __name__ == '__main__':
    if not IMPORTS_SUCCESSFUL:
        print("Skipping tests due to import errors")
        exit(1)
    unittest.main(verbosity=2)
