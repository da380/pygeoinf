"""
Unit tests for FEM solver.

Tests GeneralFEMSolver class including the analytical stiffness matrix
optimization for hat functions.
"""

import unittest
import numpy as np

from pygeoinf.interval.interval_domain import IntervalDomain
from pygeoinf.interval.boundary_conditions import BoundaryConditions
from pygeoinf.interval.l2_space import L2Space
from pygeoinf.interval.fem_solvers import GeneralFEMSolver
from pygeoinf.interval.functions import Function


class TestGeneralFEMSolver(unittest.TestCase):
    """Test GeneralFEMSolver class."""

    def setUp(self):
        """Set up test fixtures."""
        self.domain = IntervalDomain(0.0, 1.0)
        self.bc_homogeneous = BoundaryConditions('dirichlet', left=0.0, right=0.0)
        self.bc_non_homogeneous = BoundaryConditions('dirichlet', left=1.0, right=0.0)
        self.n_dofs = 5

    def test_analytical_computation_detection_hat_homogeneous(self):
        """Test analytical computation is detected for hat_homogeneous basis."""
        l2_space = L2Space(self.n_dofs, self.domain, basis_type='hat_homogeneous')
        solver = GeneralFEMSolver(l2_space, self.bc_homogeneous)
        solver.setup()

        self.assertTrue(solver._can_use_analytical)

    def test_analytical_computation_detection_fourier(self):
        """Test analytical computation is NOT used for Fourier basis."""
        l2_space = L2Space(self.n_dofs, self.domain, basis_type='fourier')
        solver = GeneralFEMSolver(l2_space, self.bc_homogeneous)
        solver.setup()

        self.assertFalse(solver._can_use_analytical)

    def test_analytical_computation_detection_non_homogeneous(self):
        """Test analytical computation is NOT used for non-homogeneous BCs."""
        l2_space = L2Space(self.n_dofs, self.domain, basis_type='hat_homogeneous')
        solver = GeneralFEMSolver(l2_space, self.bc_non_homogeneous)
        solver.setup()

        self.assertFalse(solver._can_use_analytical)

    def test_analytical_vs_numerical_methods_match(self):
        """Test analytical and numerical methods produce same results."""
        # Create GeneralFEMSolver with hat functions (uses analytical)
        l2_space = L2Space(self.n_dofs, self.domain, basis_type='hat_homogeneous')
        analytical_solver = GeneralFEMSolver(l2_space, self.bc_homogeneous)
        analytical_solver.setup()

        # Force numerical computation by temporarily setting _can_use_analytical
        numerical_solver = GeneralFEMSolver(l2_space, self.bc_homogeneous)
        numerical_solver._can_use_analytical = False
        numerical_solver._stiffness_matrix = (
            numerical_solver._assemble_stiffness_matrix_numerical()
        )
        numerical_solver._is_setup = True

        # Get stiffness matrices
        K_analytical = analytical_solver.get_stiffness_matrix()
        K_numerical = numerical_solver.get_stiffness_matrix()

        # They should match closely (allowing for numerical errors)
        np.testing.assert_allclose(K_analytical, K_numerical, rtol=1e-10)

    def test_analytical_stiffness_matrix_structure(self):
        """Test analytical stiffness matrix has correct tridiagonal structure."""
        l2_space = L2Space(self.n_dofs, self.domain, basis_type='hat_homogeneous')
        solver = GeneralFEMSolver(l2_space, self.bc_homogeneous)
        solver.setup()

        K = solver.get_stiffness_matrix()

        # Should be tridiagonal
        self.assertEqual(K.shape, (self.n_dofs, self.n_dofs))

        # Check structure: diagonal should be 2/h, off-diagonal should be -1/h
        h = (self.domain.b - self.domain.a) / (self.n_dofs + 1)
        expected_diag = 2.0 / h
        expected_off_diag = -1.0 / h

        # Check diagonal entries
        for i in range(self.n_dofs):
            self.assertAlmostEqual(K[i, i], expected_diag, places=10)

        # Check off-diagonal entries
        for i in range(self.n_dofs - 1):
            self.assertAlmostEqual(K[i, i+1], expected_off_diag, places=10)
            self.assertAlmostEqual(K[i+1, i], expected_off_diag, places=10)

        # Check that distant entries are zero
        if self.n_dofs > 2:
            self.assertEqual(K[0, 2], 0.0)
            self.assertEqual(K[2, 0], 0.0)

    def test_setup_required_before_use(self):
        """Test that setup() must be called before using solver."""
        l2_space = L2Space(self.n_dofs, self.domain, basis_type='hat_homogeneous')
        solver = GeneralFEMSolver(l2_space, self.bc_homogeneous)

        # Should raise error before setup
        with self.assertRaises(RuntimeError):
            solver.get_stiffness_matrix()

    def test_solve_poisson_basic(self):
        """Test basic Poisson equation solving."""
        l2_space = L2Space(self.n_dofs, self.domain, basis_type='hat_homogeneous')
        solver = GeneralFEMSolver(l2_space, self.bc_homogeneous)
        solver.setup()

        # Create a simple RHS function: f(x) = 1
        rhs = Function(l2_space, evaluate_callable=lambda x: np.ones_like(x), name="f(x)=1")

        # Solve
        coeffs = solver.solve_poisson(rhs)

        # Check that we get reasonable coefficients
        self.assertEqual(len(coeffs), self.n_dofs)
        self.assertTrue(np.all(np.isfinite(coeffs)))

    def test_solution_to_function_conversion(self):
        """Test converting solution coefficients to Function."""
        l2_space = L2Space(self.n_dofs, self.domain, basis_type='hat_homogeneous')
        solver = GeneralFEMSolver(l2_space, self.bc_homogeneous)
        solver.setup()

        # Create dummy coefficients
        coeffs = np.ones(self.n_dofs)

        # Convert to function
        sol_func = solver.solution_to_function(coeffs)

        # Check it's a valid Function
        self.assertIsInstance(sol_func, Function)

        # Check it can be evaluated
        x_test = np.linspace(0.1, 0.9, 10)
        values = sol_func(x_test)
        self.assertEqual(len(values), len(x_test))
        self.assertTrue(np.all(np.isfinite(values)))


class TestGeneralFEMSolverNumericalMethod(unittest.TestCase):
    """Test GeneralFEMSolver with numerical stiffness matrix assembly."""

    def setUp(self):
        """Set up test fixtures."""
        self.domain = IntervalDomain(0.0, 1.0)
        self.bc = BoundaryConditions('dirichlet', left=0.0, right=0.0)
        self.n_dofs = 3

    def test_fourier_basis_numerical_assembly(self):
        """Test numerical assembly works with Fourier basis."""
        # Use Fourier basis to force numerical assembly
        l2_space = L2Space(self.n_dofs, self.domain, basis_type='fourier')
        solver = GeneralFEMSolver(l2_space, self.bc)
        solver.setup()

        # Should use numerical method
        self.assertFalse(solver._can_use_analytical)

        # Should have valid stiffness matrix
        K = solver.get_stiffness_matrix()
        self.assertEqual(K.shape, (self.n_dofs, self.n_dofs))
        self.assertTrue(np.all(np.isfinite(K)))

    def test_sine_basis_numerical_assembly(self):
        """Test numerical assembly works with sine basis."""
        # Use sine basis
        l2_space = L2Space(self.n_dofs, self.domain, basis_type='sine')
        solver = GeneralFEMSolver(l2_space, self.bc)
        solver.setup()

        # Should have valid stiffness matrix
        K = solver.get_stiffness_matrix()
        self.assertEqual(K.shape, (self.n_dofs, self.n_dofs))
        self.assertTrue(np.all(np.isfinite(K)))

    def test_stiffness_matrix_properties(self):
        """Test that the stiffness matrix has expected properties."""
        l2_space = L2Space(self.n_dofs, self.domain, basis_type='sine')
        solver = GeneralFEMSolver(l2_space, self.bc)
        solver.setup()

        # Should use numerical method
        self.assertFalse(solver._can_use_analytical)

        K = solver.get_stiffness_matrix()

        # Should be symmetric
        np.testing.assert_allclose(K, K.T, rtol=1e-12)

        # Should be positive definite (all eigenvalues > 0)
        eigenvals = np.linalg.eigvals(K)
        self.assertTrue(np.all(eigenvals > 1e-10))

    def test_gradient_operator_integration_functionality(self):
        """Test that the gradient operator and Function integration work together."""
        l2_space = L2Space(3, self.domain, basis_type='sine')
        solver = GeneralFEMSolver(l2_space, self.bc)
        solver.setup()

        # Should use gradient operator method
        self.assertFalse(solver._can_use_analytical)

        # Should produce a valid stiffness matrix
        K = solver.get_stiffness_matrix()

        self.assertEqual(K.shape, (3, 3))
        self.assertTrue(np.allclose(K, K.T))  # Symmetric
        self.assertFalse(np.any(np.isnan(K)))  # No NaN values
        self.assertFalse(np.any(np.isinf(K)))  # No infinite values


if __name__ == '__main__':
    unittest.main()
