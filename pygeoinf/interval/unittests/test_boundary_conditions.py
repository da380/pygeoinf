"""
Comprehensive unit tests for boundary_conditions.py

This module provides detailed tests for the BoundaryConditions class,
covering all boundary condition types, properties, methods, edge cases, and error handling.
"""

import unittest

from pygeoinf.interval.boundary_conditions import BoundaryConditions

class TestBoundaryConditions(unittest.TestCase):
    """Test cases for BoundaryConditions class."""

    # === INITIALIZATION TESTS ===
    def test_init_dirichlet_defaults(self):
        bc = BoundaryConditions('dirichlet')
        self.assertEqual(bc.type, 'dirichlet')
        self.assertEqual(bc.get_parameter('left'), 0.0)
        self.assertEqual(bc.get_parameter('right'), 0.0)
        self.assertTrue(bc.is_homogeneous)

    def test_init_dirichlet_custom(self):
        bc = BoundaryConditions('dirichlet', left=1.5, right=-2.0)
        self.assertEqual(bc.get_parameter('left'), 1.5)
        self.assertEqual(bc.get_parameter('right'), -2.0)
        self.assertFalse(bc.is_homogeneous)

    def test_init_neumann_defaults(self):
        bc = BoundaryConditions('neumann')
        self.assertEqual(bc.type, 'neumann')
        self.assertEqual(bc.get_parameter('left'), 0.0)
        self.assertEqual(bc.get_parameter('right'), 0.0)
        self.assertTrue(bc.is_homogeneous)

    def test_init_neumann_custom(self):
        bc = BoundaryConditions('neumann', left=2.5, right=-1.0)
        self.assertEqual(bc.get_parameter('left'), 2.5)
        self.assertEqual(bc.get_parameter('right'), -1.0)
        self.assertFalse(bc.is_homogeneous)

    def test_init_robin_valid(self):
        bc = BoundaryConditions('robin', left_alpha=1, left_beta=2, left_value=3,
                               right_alpha=4, right_beta=5, right_value=6)
        self.assertEqual(bc.type, 'robin')
        self.assertEqual(bc.get_parameter('left_alpha'), 1)
        self.assertEqual(bc.get_parameter('right_value'), 6)

    def test_init_robin_missing_param(self):
        with self.assertRaises(ValueError):
            BoundaryConditions('robin', left_alpha=1, left_beta=2, left_value=3,
                              right_alpha=4, right_beta=5)  # missing right_value

    def test_init_periodic(self):
        bc = BoundaryConditions('periodic')
        self.assertEqual(bc.type, 'periodic')
        self.assertTrue(bc.is_homogeneous)

    def test_invalid_type(self):
        with self.assertRaises(ValueError):
            BoundaryConditions('invalid_type')

    # === CLASSMETHOD FACTORY TESTS ===
    def test_dirichlet_factory(self):
        bc = BoundaryConditions.dirichlet(1.0, -1.0)
        self.assertEqual(bc.type, 'dirichlet')
        self.assertEqual(bc.get_parameter('left'), 1.0)
        self.assertEqual(bc.get_parameter('right'), -1.0)

    def test_neumann_factory(self):
        bc = BoundaryConditions.neumann(2.0, -2.0)
        self.assertEqual(bc.type, 'neumann')
        self.assertEqual(bc.get_parameter('left'), 2.0)
        self.assertEqual(bc.get_parameter('right'), -2.0)

    def test_robin_factory(self):
        bc = BoundaryConditions.robin(1, 2, 3, 4, 5, 6)
        self.assertEqual(bc.type, 'robin')
        self.assertEqual(bc.get_parameter('left_alpha'), 1)
        self.assertEqual(bc.get_parameter('right_value'), 6)

    def test_periodic_factory(self):
        bc = BoundaryConditions.periodic()
        self.assertEqual(bc.type, 'periodic')
        self.assertTrue(bc.is_homogeneous)

    # === PROPERTY AND METHOD TESTS ===
    def test_is_homogeneous_dirichlet(self):
        bc = BoundaryConditions('dirichlet', left=0, right=0)
        self.assertTrue(bc.is_homogeneous)
        bc2 = BoundaryConditions('dirichlet', left=1, right=0)
        self.assertFalse(bc2.is_homogeneous)

    def test_is_homogeneous_neumann(self):
        bc = BoundaryConditions('neumann', left=0, right=0)
        self.assertTrue(bc.is_homogeneous)
        bc2 = BoundaryConditions('neumann', left=0, right=2)
        self.assertFalse(bc2.is_homogeneous)

    def test_is_homogeneous_periodic(self):
        bc = BoundaryConditions('periodic')
        self.assertTrue(bc.is_homogeneous)

    def test_get_parameter_default(self):
        bc = BoundaryConditions('dirichlet')
        self.assertEqual(bc.get_parameter('nonexistent', 123), 123)

    # === STRING REPRESENTATION TESTS ===
    def test_str_repr(self):
        bc = BoundaryConditions('dirichlet', left=1, right=2)
        self.assertIn('dirichlet', str(bc))
        self.assertIn('left=1', str(bc))
        self.assertIn('right=2', str(bc))
        self.assertIn('dirichlet', repr(bc))
        self.assertIn('left', repr(bc))

    def test_str_periodic(self):
        bc = BoundaryConditions('periodic')
        self.assertEqual(str(bc), 'periodic')

    # === EQUALITY TESTS ===
    def test_equality(self):
        bc1 = BoundaryConditions('dirichlet', left=1, right=2)
        bc2 = BoundaryConditions('dirichlet', left=1, right=2)
        bc3 = BoundaryConditions('dirichlet', left=0, right=0)
        self.assertEqual(bc1, bc2)
        self.assertNotEqual(bc1, bc3)
        self.assertNotEqual(bc1, 'not a bc')

    # === EDGE CASES ===
    def test_dirichlet_float_precision(self):
        bc = BoundaryConditions('dirichlet', left=0.1+0.2, right=0.3)
        self.assertAlmostEqual(bc.get_parameter('left'), 0.3, places=7)
        self.assertAlmostEqual(bc.get_parameter('right'), 0.3, places=7)

    def test_robin_all_zero(self):
        bc = BoundaryConditions.robin(0, 0, 0, 0, 0, 0)
        self.assertEqual(bc.type, 'robin')
        self.assertEqual(bc.get_parameter('left_alpha'), 0)
        self.assertEqual(bc.get_parameter('right_value'), 0)

if __name__ == '__main__':
    unittest.main(verbosity=2)
