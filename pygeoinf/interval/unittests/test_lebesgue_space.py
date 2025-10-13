"""
Comprehensive unit tests for lebesgue_space.py

Covers creation modes, basis management, projections, metric properties,
linear operations with coefficients, integration settings, equality,
and projection behavior.
"""

import unittest
import numpy as np
import sys
import os

# Ensure repo root is on path (mirror style of other unittests here)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

try:
    from pygeoinf.interval.lebesgue_space import Lebesgue
    from pygeoinf.interval.interval_domain import IntervalDomain
    from pygeoinf.interval.providers import CustomBasisProvider
    from pygeoinf.interval.function_providers import (
        SineFunctionProvider,
    )
    from pygeoinf.interval.functions import Function
    IMPORTS_SUCCESSFUL = True
except Exception as e:  # pragma: no cover - allow running partial test suites
    print(f"Import error in test_lebesgue_space: {e}")
    IMPORTS_SUCCESSFUL = False


@unittest.skipUnless(IMPORTS_SUCCESSFUL, "Required modules not available")
class TestLebesgueCreationAndBasis(unittest.TestCase):
    def setUp(self):
        self.domain = IntervalDomain(0.0, 1.0)

    def test_create_baseless_space(self):
        space = Lebesgue(5, self.domain, basis='none')
        self.assertEqual(space.dim, 5)
        # Accessing basis should raise
        with self.assertRaises(RuntimeError):
            _ = space.basis_functions
        with self.assertRaises(RuntimeError):
            _ = space.get_basis_function(0)

    def test_create_with_string_basis(self):
        for basis in ['fourier', 'sine', 'cosine', 'hat']:
            space = Lebesgue(6, self.domain, basis=basis)
            self.assertEqual(space.dim, 6)
            # Should have a provider and basis available lazily
            funcs = space.basis_functions
            self.assertEqual(len(funcs), 6)
            # Each is a Function in this space
            self.assertTrue(all(isinstance(f, Function) for f in funcs))

    def test_create_with_callables(self):
        callables = [
            lambda x: np.ones_like(x),
            lambda x: x,
            lambda x: x**2,
            lambda x: x**3,
        ]
        space = Lebesgue(4, self.domain, basis=callables)
        self.assertEqual(len(space.basis_functions), 4)
        # Should retrieve the same functions
        f2 = space.get_basis_function(2)
        self.assertIsInstance(f2, Function)

    def test_invalid_basis_inputs(self):
        # Mismatch count
        with self.assertRaises(ValueError):
            Lebesgue(3, self.domain, basis=[lambda x: 1, lambda x: x])
        # Unsupported string
        with self.assertRaises(ValueError):
            Lebesgue(5, self.domain, basis='unsupported')
        # Wrong type
        with self.assertRaises(TypeError):
            Lebesgue(5, self.domain, basis=42)  # type: ignore[arg-type]

    def test_set_basis_provider_workflow(self):
        space = Lebesgue(5, self.domain, basis='none')
        # Build a provider (use sine on [0,1], orthonormalized in provider)
        fprov = SineFunctionProvider(space)
        provider = CustomBasisProvider(
            space,
            functions_provider=fprov,
            orthonormal=True,
            basis_type='sine',
        )
        space.set_basis_provider(provider)
        self.assertEqual(len(space.basis_functions), 5)
        # get_basis_function should work now
        g0 = space.get_basis_function(0)
        self.assertIsInstance(g0, Function)


@unittest.skipUnless(IMPORTS_SUCCESSFUL, "Required modules not available")
class TestLebesgueMetricAndProjection(unittest.TestCase):
    def setUp(self):
        self.domain = IntervalDomain(0.0, 1.0)

    def test_metric_is_symmetric_spd(self):
        space = Lebesgue(6, self.domain, basis='fourier')
        G = space.metric
        # Symmetry
        self.assertTrue(np.allclose(G, G.T, atol=1e-10))
        # Positive definiteness (eigenvalues positive)
        w = np.linalg.eigvalsh(G)
        self.assertTrue(np.all(w > 1e-12))

    def test_metric_identity_like_for_orthonormal_basis(self):
        # Fourier provider in this repo normalizes functions on [0,1]
        space = Lebesgue(8, self.domain, basis='fourier')
        G = space.metric
        self.assertTrue(np.allclose(G, np.eye(space.dim), atol=5e-2))

    def test_to_components_and_from_components(self):
        space = Lebesgue(5, self.domain, basis='fourier')
    # Pick a basis function and try to recover its coefficients
        phi2 = space.get_basis_function(2)
        # Build a fresh Function by evaluate_callable so projection is used
        f = Function(space, evaluate_callable=lambda x: phi2.evaluate(x))
        c = space.to_components(f)
        # Should be close to the unit vector e2
        e = np.zeros(space.dim)
        e[2] = 1.0
        self.assertTrue(np.allclose(c, e, atol=1e-1))
        # Reconstruct
        f_rec = space.from_components(c)
        xs = np.linspace(0, 1, 200)
        self.assertTrue(
            np.allclose(
                f_rec.evaluate(xs),
                phi2.evaluate(xs),
                atol=1e-1,
            )
        )

    def test_project_function(self):
        space = Lebesgue(7, self.domain, basis='fourier')
        # External function not necessarily in span
        g = Function(space, evaluate_callable=lambda x: np.exp(x))
        p = space.project(g)
        # p should live in this space and be consistent with its coefficients
        self.assertIs(p.space, space)
        c = space.to_components(p)
        p2 = space.from_components(c)
        xs = np.linspace(0, 1, 100)
        self.assertTrue(
            np.allclose(p.evaluate(xs), p2.evaluate(xs), atol=1e-10)
        )


@unittest.skipUnless(IMPORTS_SUCCESSFUL, "Required modules not available")
class TestLebesgueLinearOpsWithCoefficients(unittest.TestCase):
    def setUp(self):
        self.domain = IntervalDomain(0.0, 1.0)
        self.space = Lebesgue(4, self.domain, basis='fourier')

    def test_multiply_and_add_coeff_based(self):
        # Build two Functions using explicit coefficients
        x = self.space.from_components(np.array([1.0, 2.0, 3.0, 4.0]))
        y = self.space.from_components(np.array([-1.0, 0.5, 0.0, 1.0]))

        ax = self.space.multiply(2.0, x)
        self.assertIsNotNone(ax.coefficients)
        self.assertTrue(
            np.allclose(np.asarray(ax.coefficients), [2, 4, 6, 8])
        )

        s = self.space.add(x, y)
        self.assertIsNotNone(s.coefficients)
        self.assertTrue(
            np.allclose(np.asarray(s.coefficients), [0.0, 2.5, 3.0, 5.0])
        )

    def test_ax_in_place(self):
        z = self.space.from_components(np.array([1.0, -2.0, 3.0, -4.0]))
        self.space.ax(3.0, z)
        self.assertIsNotNone(z.coefficients)
        self.assertTrue(
            np.allclose(np.asarray(z.coefficients), [3.0, -6.0, 9.0, -12.0])
        )

    def test_axpy_in_place(self):
        x = self.space.from_components(np.array([1.0, 1.0, 1.0, 1.0]))
        y = self.space.from_components(np.array([0.0, 2.0, 4.0, 8.0]))
        self.space.axpy(2.0, x, y)
        self.assertIsNotNone(y.coefficients)
        self.assertTrue(
            np.allclose(np.asarray(y.coefficients), [2.0, 4.0, 6.0, 10.0])
        )

    def test_ax_errors_without_coefficients(self):
        # Function defined via callable has no coefficients
        f = Function(self.space, evaluate_callable=lambda x: np.sin(2*np.pi*x))
        with self.assertRaises(ValueError):
            self.space.ax(2.0, f)
        with self.assertRaises(ValueError):
            self.space.axpy(2.0, f, f)


@unittest.skipUnless(IMPORTS_SUCCESSFUL, "Required modules not available")
class TestLebesgueIntegrationAndEquality(unittest.TestCase):
    def setUp(self):
        self.domain = IntervalDomain(0.0, 1.0)

    def test_integration_settings_validation(self):
        space = Lebesgue(5, self.domain, basis='fourier')
        # Invalid method type
        with self.assertRaises(TypeError):
            space.integration_method = 123  # type: ignore[assignment]
        # Unsupported method string
        with self.assertRaises(ValueError):
            space.integration_method = 'unsupported'
        # Valid change should clear caches and set
        space.integration_method = 'trapz'
        self.assertEqual(space.integration_method, 'trapz')

        # npoints validation
        with self.assertRaises(TypeError):
            space.integration_npoints = 12.5  # type: ignore[assignment]
        with self.assertRaises(ValueError):
            space.integration_npoints = 0
        space.integration_npoints = 500
        self.assertEqual(space.integration_npoints, 500)

    def test_space_equality(self):
        d1 = IntervalDomain(0.0, 1.0)
        d2 = IntervalDomain(0.0, 2.0)
        s1 = Lebesgue(5, d1, basis='fourier')
        s2 = Lebesgue(5, d1, basis='sine')
        s3 = Lebesgue(6, d1, basis='fourier')
        s4 = Lebesgue(5, d2, basis='fourier')
    # Same dim and same domain => equal, regardless of basis type per
    # __eq__
        self.assertEqual(s1, s2)
        # Different dim
        self.assertNotEqual(s1, s3)
        # Different domain
        self.assertNotEqual(s1, s4)

    def test_zero_function(self):
        space = Lebesgue(3, self.domain, basis='fourier')
        z = space.zero
        xs = np.linspace(0, 1, 10)
        self.assertTrue(np.allclose(z.evaluate(xs), 0.0))


if __name__ == '__main__':
    unittest.main(verbosity=2)
