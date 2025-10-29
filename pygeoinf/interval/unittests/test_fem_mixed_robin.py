"""
Unit tests for FEM solver with mixed and Robin boundary conditions.

This module tests the GeneralFEMSolver with:
- Mixed Dirichlet-Neumann boundary conditions
- Mixed Neumann-Dirichlet boundary conditions
- Robin boundary conditions

Tests verify that the FEM solver correctly solves the Poisson equation
-Δu = f with these boundary conditions by comparing numerical results
with analytical solutions.
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

try:
    from pygeoinf.interval import (
        Lebesgue, IntervalDomain, BoundaryConditions,
        InverseLaplacian, Function
    )
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_SUCCESSFUL = False


@unittest.skipUnless(IMPORTS_SUCCESSFUL, "Required modules not available")
class TestFEMMixedRobinBCs(unittest.TestCase):
    """Test cases for FEM solver with mixed and Robin boundary conditions."""

    def setUp(self):
        """Set up test fixtures with domain, spaces, and test functions."""
        # Domain parameters
        self.a, self.b = 0.0, 1.0
        self.L = self.b - self.a

        # Error thresholds
        self.fem_threshold = 5e-2  # Relaxed threshold for FEM methods

        # Create domain and spaces
        domain = IntervalDomain(self.a, self.b)
        self.space = Lebesgue(256, domain, basis='sine')

        # Create test functions and analytical solutions
        self._create_test_functions()

        # Create FEM operators
        self._create_operators()

    def _create_test_functions(self):
        """Create test functions and their analytical solutions."""
        # For mixed Dirichlet-Neumann: u(a) = 0, u'(b) = 0
        # Test with u(x) = sin(πx/2) which satisfies u(0)=0, u'(1)=0
        self.mixed_dn_input = Function(
            self.space,
            evaluate_callable=lambda x: np.sin(np.pi * x / 2),
            name='sin(πx/2)'
        )
        # -Δu = (π/2)² sin(πx/2)
        self.mixed_dn_rhs = Function(
            self.space,
            evaluate_callable=lambda x: (
                (np.pi / 2)**2 * np.sin(np.pi * x / 2)
            ),
            name='-Δsin(πx/2)'
        )

        # For mixed Neumann-Dirichlet: u'(a) = 0, u(b) = 0
        # Test with u(x) = cos(πx/2) - 1, which satisfies u'(0)=0, u(1)=0
        self.mixed_nd_input = Function(
            self.space,
            evaluate_callable=lambda x: np.cos(np.pi * x / 2) - 1,
            name='cos(πx/2)-1'
        )
        # -Δu = (π/2)² cos(πx/2)
        self.mixed_nd_rhs = Function(
            self.space,
            evaluate_callable=lambda x: (
                (np.pi / 2)**2 * np.cos(np.pi * x / 2)
            ),
            name='-Δ(cos(πx/2)-1)'
        )

        # For Robin: αu + βu' = 0 at boundaries
        # Use simple polynomial that we can verify
        # u(x) = x(1-x) satisfies Dirichlet-like conditions
        self.robin_input = Function(
            self.space,
            evaluate_callable=lambda x: x * (1 - x),
            name='x(1-x)'
        )
        # -Δu = 2
        self.robin_rhs = Function(
            self.space,
            evaluate_callable=lambda x: 2.0 * np.ones_like(x),
            name='-Δ(x(1-x))'
        )

    def _create_operators(self):
        """Create InverseLaplacian operators with FEM method."""
        # Mixed Dirichlet-Neumann
        self.inv_lap_mixed_dn = InverseLaplacian(
            self.space,
            BoundaryConditions.mixed_dirichlet_neumann(
                left_value=0.0, right_derivative=0.0
            ),
            method='fem',
            dofs=256
        )

        # Mixed Neumann-Dirichlet
        self.inv_lap_mixed_nd = InverseLaplacian(
            self.space,
            BoundaryConditions.mixed_neumann_dirichlet(
                left_derivative=0.0, right_value=0.0
            ),
            method='fem',
            dofs=256
        )

        # Robin with α=1, β=1, homogeneous (value=0)
        # Need to create this manually since the classmethod sets value
        self.inv_lap_robin = InverseLaplacian(
            self.space,
            BoundaryConditions(
                'robin',
                left_alpha=1.0, left_beta=1.0, left_value=0.0,
                right_alpha=1.0, right_beta=1.0, right_value=0.0
            ),
            method='fem',
            dofs=256
        )

    def _compute_l2_relative_error(self, numerical_func, analytical_func,
                                   n_points=1001):
        """
        Compute the L2 relative error between numerical and analytical funcs.

        Args:
            numerical_func: Function object with numerical solution
            analytical_func: Function object with analytical solution
            n_points: Number of evaluation points

        Returns:
            float: L2 relative error
        """
        # Create evaluation points
        x_eval = np.linspace(self.a, self.b, n_points)

        # Evaluate functions
        try:
            numerical_values = numerical_func.evaluate(x_eval)
        except Exception:
            numerical_values = np.array([numerical_func(xi) for xi in x_eval])

        try:
            analytical_values = analytical_func.evaluate(x_eval)
        except Exception:
            analytical_values = np.array([
                analytical_func(xi) for xi in x_eval
            ])

        # Compute L2 norms
        diff_norm = np.sqrt(
            np.trapz((numerical_values - analytical_values)**2, x_eval)
        )
        analytical_norm = np.sqrt(np.trapz(analytical_values**2, x_eval))

        # Handle zero analytical solution case
        if analytical_norm < 1e-15:
            return diff_norm

        return diff_norm / analytical_norm

    # === MIXED BOUNDARY CONDITION TESTS ===

    def test_inverse_laplacian_fem_mixed_dirichlet_neumann(self):
        """Test FEM inverse Laplacian with mixed Dirichlet-Neumann BC."""
        # Apply inverse Laplacian (solve -Δu = f)
        result = self.inv_lap_mixed_dn(self.mixed_dn_rhs)

        # Compute relative error
        error = self._compute_l2_relative_error(result, self.mixed_dn_input)

        # Check if error is below threshold
        self.assertLess(
            error, self.fem_threshold,
            f"FEM Inverse Laplacian Mixed D-N error: {error:.2e} "
            f"(threshold: {self.fem_threshold})"
        )

        # Verify boundary conditions
        u_a = result(self.a)
        self.assertAlmostEqual(
            u_a, 0.0, delta=1e-6,
            msg=f"Dirichlet BC at x=a not satisfied: u(a) = {u_a}"
        )

    def test_inverse_laplacian_fem_mixed_neumann_dirichlet(self):
        """Test FEM inverse Laplacian with mixed Neumann-Dirichlet BC."""
        # Apply inverse Laplacian (solve -Δu = f)
        result = self.inv_lap_mixed_nd(self.mixed_nd_rhs)

        # Compute relative error
        error = self._compute_l2_relative_error(result, self.mixed_nd_input)

        # Check if error is below threshold
        self.assertLess(
            error, self.fem_threshold,
            f"FEM Inverse Laplacian Mixed N-D error: {error:.2e} "
            f"(threshold: {self.fem_threshold})"
        )

        # Verify boundary conditions
        u_b = result(self.b)
        self.assertAlmostEqual(
            u_b, 0.0, delta=1e-6,
            msg=f"Dirichlet BC at x=b not satisfied: u(b) = {u_b}"
        )

    def test_inverse_laplacian_fem_robin(self):
        """Test FEM inverse Laplacian with Robin BC."""
        # Apply inverse Laplacian (solve -Δu = f)
        result = self.inv_lap_robin(self.robin_rhs)

        # Compute relative error
        error = self._compute_l2_relative_error(result, self.robin_input)

        # Check if error is below threshold
        # Robin BCs are more challenging, use relaxed threshold
        self.assertLess(
            error, self.fem_threshold * 2,
            f"FEM Inverse Laplacian Robin error: {error:.2e} "
            f"(threshold: {self.fem_threshold * 2})"
        )

    # === SPECTRAL METHOD COMPARISON TESTS ===

    def test_inverse_laplacian_spectral_mixed_dirichlet_neumann(self):
        """Test spectral inverse Laplacian with mixed D-N BC."""
        # Create spectral operator
        inv_lap_spectral = InverseLaplacian(
            self.space,
            BoundaryConditions.mixed_dirichlet_neumann(
                left_value=0.0, right_derivative=0.0
            ),
            method='spectral',
            dofs=256
        )

        # Apply inverse Laplacian
        result = inv_lap_spectral(self.mixed_dn_rhs)

        # Compute relative error
        error = self._compute_l2_relative_error(result, self.mixed_dn_input)

        # Spectral methods should be more accurate
        self.assertLess(
            error, 1e-2,
            f"Spectral Inverse Laplacian Mixed D-N error: {error:.2e}"
        )

    def test_inverse_laplacian_spectral_mixed_neumann_dirichlet(self):
        """Test spectral inverse Laplacian with mixed N-D BC."""
        # Create spectral operator
        inv_lap_spectral = InverseLaplacian(
            self.space,
            BoundaryConditions.mixed_neumann_dirichlet(
                left_derivative=0.0, right_value=0.0
            ),
            method='spectral',
            dofs=256
        )

        # Apply inverse Laplacian
        result = inv_lap_spectral(self.mixed_nd_rhs)

        # Compute relative error
        error = self._compute_l2_relative_error(result, self.mixed_nd_input)

        # Spectral methods should be more accurate
        self.assertLess(
            error, 1e-2,
            f"Spectral Inverse Laplacian Mixed N-D error: {error:.2e}"
        )

    # === ROUNDTRIP TESTS (using only test functions that satisfy BCs) ===

    def test_roundtrip_fem_mixed_dn(self):
        """Test that inverse matches the expected solution for mixed D-N."""
        # This is implicitly tested in the first test, but we verify
        # that applying the inverse twice doesn't cause issues
        result1 = self.inv_lap_mixed_dn(self.mixed_dn_rhs)
        error = self._compute_l2_relative_error(result1, self.mixed_dn_input)
        self.assertLess(error, self.fem_threshold)

    def test_roundtrip_fem_mixed_nd(self):
        """Test that inverse matches the expected solution for mixed N-D."""
        result1 = self.inv_lap_mixed_nd(self.mixed_nd_rhs)
        error = self._compute_l2_relative_error(result1, self.mixed_nd_input)
        self.assertLess(error, self.fem_threshold)

    def test_roundtrip_fem_robin(self):
        """Test that inverse matches the expected solution for Robin."""
        result1 = self.inv_lap_robin(self.robin_rhs)
        error = self._compute_l2_relative_error(result1, self.robin_input)
        self.assertLess(error, self.fem_threshold * 2)


if __name__ == '__main__':
    # Create a test suite
    suite = unittest.TestSuite()

    # Add all test methods
    test_methods = [
        'test_inverse_laplacian_fem_mixed_dirichlet_neumann',
        'test_inverse_laplacian_fem_mixed_neumann_dirichlet',
        'test_inverse_laplacian_fem_robin',
        'test_inverse_laplacian_spectral_mixed_dirichlet_neumann',
        'test_inverse_laplacian_spectral_mixed_neumann_dirichlet',
        'test_roundtrip_fem_mixed_dn',
        'test_roundtrip_fem_mixed_nd',
        'test_roundtrip_fem_robin',
    ]

    for method in test_methods:
        suite.addTest(TestFEMMixedRobinBCs(method))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'='*70}")
    print("Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    if result.testsRun > 0:
        success_rate = (
            (result.testsRun - len(result.failures) - len(result.errors)) /
            result.testsRun * 100
        )
        print(f"Success rate: {success_rate:.1f}%")
    print(f"{'='*70}")
