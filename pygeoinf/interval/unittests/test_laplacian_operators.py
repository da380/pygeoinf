"""
Comprehensive unit tests for Laplacian and InverseLaplacian operators

This module provides detailed tests for both spectral and non-spectral methods
of the Laplacian and InverseLaplacian operators, comparing numerical results
with analytical solutions using L2 norm relative error.

Tests include:
- Dirichlet boundary conditions
- Neumann boundary conditions
- Periodic boundary conditions
- Both spectral and finite difference/finite element methods
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
        Laplacian, InverseLaplacian, Function
    )
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_SUCCESSFUL = False


@unittest.skipUnless(IMPORTS_SUCCESSFUL, "Required modules not available")
class TestLaplacianOperators(unittest.TestCase):
    """Test cases for Laplacian and InverseLaplacian operators."""

    def setUp(self):
        """Set up test fixtures with domain, spaces, and test functions."""
        # Domain parameters
        self.a, self.b = 0.0, 1.0
        self.L = self.b - self.a

        # Error threshold for tests
        self.strict_threshold = 1e-8   # For trigonometric functions
        self.moderate_threshold = 1e-1  # For polynomial/mixed functions
        self.loose_threshold = 1e-1    # For FD/FEM methods



        # Spaces per boundary condition family (use natural bases)
        domain = IntervalDomain(self.a, self.b)
        self.space_dirichlet = Lebesgue(256, domain, basis='sine')
        self.space_neumann = Lebesgue(256, domain, basis='cosine')
        self.space_periodic = Lebesgue(257, domain, basis='fourier')

        # Create test functions and their analytical Laplacians
        self._create_dirichlet_test_functions()
        self._create_neumann_test_functions()
        self._create_periodic_test_functions()

        # Create operators for different methods
        self._create_operators()

    def _create_dirichlet_test_functions(self):
        """Create Dirichlet test functions and their analytical Laplacians."""
        # Input functions
        self.dirichlet_inputs = [
            Function(
                self.space_dirichlet,
                evaluate_callable=lambda x: (x - self.a) * (self.b - x),
                name='D1_bump'
            ),
            Function(
                self.space_dirichlet,
                evaluate_callable=lambda x: np.sin(
                    np.pi * (x - self.a) / self.L
                ),
                name='D2_sin_pi'
            ),
            Function(
                self.space_dirichlet,
                evaluate_callable=lambda x: np.sin(
                    2*np.pi * (x - self.a) / self.L
                ),
                name='D3_sin_2pi'
            ),
            Function(
                self.space_dirichlet,
                evaluate_callable=lambda x: (
                    (x-self.a)**2 * (self.b-x)**2
                ),
                name='D4_poly'
            ),
            Function(
                self.space_dirichlet,
                evaluate_callable=lambda x: (
                    5*np.sin(np.pi*(x-self.a)/self.L) -
                    7*np.sin(3*np.pi*(x-self.a)/self.L)
                ),
                name='D5_mix'
            )
        ]

        # Analytical Laplacians
        self.dirichlet_expected = [
            Function(
                self.space_dirichlet,
                evaluate_callable=lambda x: 2.0 * np.ones_like(x),
                name='D1_L'
            ),
            Function(
                self.space_dirichlet,
                evaluate_callable=lambda x: (
                    (np.pi / self.L)**2 * np.sin(
                        np.pi * (x - self.a) / self.L
                    )
                ),
                name='D2_L'
            ),
            Function(
                self.space_dirichlet,
                evaluate_callable=lambda x: (
                    (2 * np.pi / self.L)**2 * np.sin(
                        2 * np.pi * (x - self.a) / self.L
                    )
                ),
                name='D3_L'
            ),
            Function(
                self.space_dirichlet,
                evaluate_callable=lambda x: (
                    -2*(x-self.a)**2 + 8*(x-self.a)*(self.b-x) -
                    2*(self.b-x)**2
                ),
                name='D4_L'
            ),
            Function(
                self.space_dirichlet,
                evaluate_callable=lambda x: (
                    5*(np.pi / self.L)**2 * np.sin(
                        np.pi * (x - self.a) / self.L
                    ) -
                    7*(3 * np.pi / self.L)**2 * np.sin(
                        3 * np.pi * (x - self.a) / self.L
                    )
                ),
                name='D5_L'
            )
        ]

    def _create_neumann_test_functions(self):
        """Create Neumann test functions and their analytical Laplacians."""
        # Input functions
        self.neumann_inputs = [
            Function(
                self.space_neumann,
                evaluate_callable=lambda x: np.ones_like(x),
                name='N1_const'
            ),
            Function(
                self.space_neumann,
                evaluate_callable=lambda x: np.cos(
                    np.pi*(x-self.a)/self.L
                ),
                name='N2_cos_pi'
            ),
            Function(
                self.space_neumann,
                evaluate_callable=lambda x: np.cos(
                    2*np.pi*(x-self.a)/self.L
                ),
                name='N3_cos_2pi'
            ),
            Function(
                self.space_neumann,
                evaluate_callable=lambda x: (
                    3.0*np.cos(np.pi*(x-self.a)/self.L) -
                    np.cos(3*np.pi*(x-self.a)/self.L)
                ),
                name='N4_mix'
            )
        ]

        # Analytical Laplacians
        self.neumann_expected = [
            Function(
                self.space_neumann,
                evaluate_callable=lambda x: np.zeros_like(x),
                name='N1_L'
            ),  # constant -> Laplacian 0
            Function(
                self.space_neumann,
                evaluate_callable=lambda x: (
                    (np.pi / self.L)**2 * np.cos(
                        np.pi * (x - self.a) / self.L
                    )
                ),
                name='N2_L'
            ),
            Function(
                self.space_neumann,
                evaluate_callable=lambda x: (
                    (2 * np.pi / self.L)**2 * np.cos(
                        2 * np.pi * (x - self.a) / self.L
                    )
                ),
                name='N3_L'
            ),
            Function(
                self.space_neumann,
                evaluate_callable=lambda x: (
                    3*(np.pi / self.L)**2 * np.cos(
                        np.pi * (x - self.a) / self.L
                    ) -
                    (3 * np.pi / self.L)**2 * np.cos(
                        3 * np.pi * (x - self.a) / self.L
                    )
                ),
                name='N4_L'
            )
        ]

    def _create_periodic_test_functions(self):
        """Create periodic test functions and their analytical Laplacians."""
        # Input functions
        self.periodic_inputs = [
            Function(
                self.space_periodic,
                evaluate_callable=lambda x: np.ones_like(x),
                name='P1_const'
            ),
            Function(
                self.space_periodic,
                evaluate_callable=lambda x: np.sin(
                    2*np.pi*(x-self.a)/self.L
                ),
                name='P2_sin_2pi'
            ),
            Function(
                self.space_periodic,
                evaluate_callable=lambda x: np.cos(
                    2*np.pi*(x-self.a)/self.L
                ),
                name='P3_cos_2pi'
            ),
            Function(
                self.space_periodic,
                evaluate_callable=lambda x: (
                    np.sin(4*np.pi*(x-self.a)/self.L) +
                    2*np.cos(6*np.pi*(x-self.a)/self.L)
                ),
                name='P4_combo'
            )
        ]

        # Analytical Laplacians
        self.periodic_expected = [
            Function(
                self.space_periodic,
                evaluate_callable=lambda x: np.zeros_like(x),
                name='P1_L'
            ),
            Function(
                self.space_periodic,
                evaluate_callable=lambda x: (
                    (2 * np.pi / self.L)**2 * np.sin(
                        2 * np.pi * (x - self.a) / self.L
                    )
                ),
                name='P2_L'
            ),
            Function(
                self.space_periodic,
                evaluate_callable=lambda x: (
                    (2 * np.pi / self.L)**2 * np.cos(
                        2 * np.pi * (x - self.a) / self.L
                    )
                ),
                name='P3_L'
            ),
            Function(
                self.space_periodic,
                evaluate_callable=lambda x: (
                    (4 * np.pi / self.L)**2 * np.sin(
                        4 * np.pi * (x - self.a) / self.L
                    ) +
                    2 * (6 * np.pi / self.L)**2 * np.cos(
                        6 * np.pi * (x - self.a) / self.L
                    )
                ),
                name='P4_L'
            )
        ]

    def _create_operators(self):
        """Create Laplacian and InverseLaplacian operators."""
        # Spectral Laplacian operators
        self.laplacian_spectral_dirichlet = Laplacian(
            self.space_dirichlet, BoundaryConditions('dirichlet'),
            method='spectral', dofs=512
        )
        self.laplacian_spectral_neumann = Laplacian(
            self.space_neumann, BoundaryConditions('neumann'),
            method='spectral', dofs=512
        )
        self.laplacian_spectral_periodic = Laplacian(
            self.space_periodic, BoundaryConditions('periodic'),
            method='spectral', dofs=512
        )

        # Finite difference Laplacian operators
        self.laplacian_fd_dirichlet = Laplacian(
            self.space_dirichlet, BoundaryConditions('dirichlet'),
            method='fd', dofs=512
        )
        self.laplacian_fd_neumann = Laplacian(
            self.space_neumann, BoundaryConditions('neumann'),
            method='fd', dofs=512
        )
        self.laplacian_fd_periodic = Laplacian(
            self.space_periodic, BoundaryConditions('periodic'),
            method='fd', dofs=512
        )

        # Spectral Inverse Laplacian operators
        self.inv_laplacian_spectral_dirichlet = InverseLaplacian(
            self.space_dirichlet, BoundaryConditions('dirichlet'),
            method='spectral', dofs=512
        )
        self.inv_laplacian_spectral_neumann = InverseLaplacian(
            self.space_neumann, BoundaryConditions('neumann'),
            method='spectral', dofs=512
        )
        self.inv_laplacian_spectral_periodic = InverseLaplacian(
            self.space_periodic, BoundaryConditions('periodic'),
            method='spectral', dofs=512
        )

        # Finite element Inverse Laplacian operators
        self.inv_laplacian_fem_dirichlet = InverseLaplacian(
            self.space_dirichlet, BoundaryConditions('dirichlet'),
            method='fem', dofs=512
        )
        self.inv_laplacian_fem_neumann = InverseLaplacian(
            self.space_neumann, BoundaryConditions('neumann'),
            method='fem', dofs=512
        )
        self.inv_laplacian_fem_periodic = InverseLaplacian(
            self.space_periodic, BoundaryConditions('periodic'),
            method='fem', dofs=512
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
            analytical_values = np.array([analytical_func(xi)
                                         for xi in x_eval])

        # Compute L2 norms
        diff_norm = np.sqrt(
            np.trapezoid((numerical_values - analytical_values)**2, x_eval)
        )
        analytical_norm = np.sqrt(np.trapezoid(analytical_values**2, x_eval))

        # Handle zero analytical solution case
        if analytical_norm < 1e-15:
            return diff_norm

        return diff_norm / analytical_norm

    # === SPECTRAL LAPLACIAN TESTS ===

    def test_laplacian_spectral_dirichlet(self):
        """Test spectral Laplacian with Dirichlet boundary conditions."""
        for input_func, expected_func in zip(self.dirichlet_inputs,
                                             self.dirichlet_expected):
            with self.subTest(function=input_func.name):
                # Apply Laplacian
                result = self.laplacian_spectral_dirichlet(input_func)

                # Compute relative error
                error = self._compute_l2_relative_error(result, expected_func)

                # Use stricter threshold for trigonometric functions
                func_name = input_func.name or "unknown"
                if ('sin' in func_name or 'cos' in func_name or
                        func_name == 'N1_const'):
                    threshold = self.strict_threshold
                else:
                    threshold = self.moderate_threshold

                # Check if error is below threshold
                self.assertLess(
                    error, threshold,
                    f"Spectral Laplacian Dirichlet error for "
                    f"{input_func.name}: {error:.2e} (threshold: {threshold})"
                )

    def test_laplacian_spectral_neumann(self):
        """Test spectral Laplacian with Neumann boundary conditions."""
        for input_func, expected_func in zip(self.neumann_inputs,
                                             self.neumann_expected):
            with self.subTest(function=input_func.name):
                # Apply Laplacian
                result = self.laplacian_spectral_neumann(input_func)

                # Compute relative error
                error = self._compute_l2_relative_error(result, expected_func)

                # Check if error is below threshold
                self.assertLess(
                    error, self.moderate_threshold,
                    f"Spectral Laplacian Neumann error for "
                    f"{input_func.name}: {error:.2e}"
                )

    def test_laplacian_spectral_periodic(self):
        """Test spectral Laplacian with periodic boundary conditions."""
        for input_func, expected_func in zip(self.periodic_inputs,
                                             self.periodic_expected):
            with self.subTest(function=input_func.name):
                # Apply Laplacian
                result = self.laplacian_spectral_periodic(input_func)

                # Compute relative error
                error = self._compute_l2_relative_error(result, expected_func)

                # Check if error is below threshold
                self.assertLess(
                    error, self.moderate_threshold,
                    f"Spectral Laplacian Periodic error for "
                    f"{input_func.name}: {error:.2e}"
                )

    # === FINITE DIFFERENCE LAPLACIAN TESTS ===

    def test_laplacian_fd_dirichlet(self):
        """Test finite difference Laplacian with Dirichlet BC."""
        for input_func, expected_func in zip(self.dirichlet_inputs,
                                             self.dirichlet_expected):
            with self.subTest(function=input_func.name):
                # Apply Laplacian
                result = self.laplacian_fd_dirichlet(input_func)

                # Compute relative error
                error = self._compute_l2_relative_error(result, expected_func)

                # Check if error is below threshold (looser for FD methods)
                self.assertLess(
                    error, self.loose_threshold,
                    f"FD Laplacian Dirichlet error for "
                    f"{input_func.name}: {error:.2e}"
                )

    def test_laplacian_fd_neumann(self):
        """Test finite difference Laplacian with Neumann BC."""
        for input_func, expected_func in zip(self.neumann_inputs,
                                             self.neumann_expected):
            with self.subTest(function=input_func.name):
                # Apply Laplacian
                result = self.laplacian_fd_neumann(input_func)

                # Compute relative error
                error = self._compute_l2_relative_error(result, expected_func)

                # Check if error is below threshold
                self.assertLess(
                    error, self.loose_threshold,
                    f"FD Laplacian Neumann error for "
                    f"{input_func.name}: {error:.2e}"
                )

    def test_laplacian_fd_periodic(self):
        """Test finite difference Laplacian with periodic BC."""
        for input_func, expected_func in zip(self.periodic_inputs,
                                             self.periodic_expected):
            with self.subTest(function=input_func.name):
                # Apply Laplacian
                result = self.laplacian_fd_periodic(input_func)

                # Compute relative error
                error = self._compute_l2_relative_error(result, expected_func)

                # Check if error is below threshold
                self.assertLess(
                    error, self.loose_threshold,
                    f"FD Laplacian Periodic error for "
                    f"{input_func.name}: {error:.2e}"
                )

    # === SPECTRAL INVERSE LAPLACIAN TESTS ===

    def test_inverse_laplacian_spectral_dirichlet(self):
        """Test spectral inverse Laplacian with Dirichlet BC."""
        for input_func, expected_func in zip(self.dirichlet_expected,
                                             self.dirichlet_inputs):
            with self.subTest(function=expected_func.name):
                # Apply inverse Laplacian
                result = self.inv_laplacian_spectral_dirichlet(input_func)

                # Compute relative error
                error = self._compute_l2_relative_error(result, expected_func)

                # Check if error is below threshold
                self.assertLess(
                    error, self.moderate_threshold,
                    f"Spectral Inverse Laplacian Dirichlet error for "
                    f"{expected_func.name}: {error:.2e}"
                )

    def test_inverse_laplacian_spectral_neumann(self):
        """Test spectral inverse Laplacian with Neumann BC."""
        # Skip constant function for Neumann (not invertible)
        for i, (input_func, expected_func) in enumerate(
                zip(self.neumann_expected[1:], self.neumann_inputs[1:])):
            with self.subTest(function=expected_func.name):
                # Apply inverse Laplacian
                result = self.inv_laplacian_spectral_neumann(input_func)

                # Compute relative error
                error = self._compute_l2_relative_error(result, expected_func)

                # Check if error is below threshold
                self.assertLess(
                    error, self.moderate_threshold,
                    f"Spectral Inverse Laplacian Neumann error for "
                    f"{expected_func.name}: {error:.2e}"
                )

    def test_inverse_laplacian_spectral_periodic(self):
        """Test spectral inverse Laplacian with periodic BC."""
        # Skip constant function for periodic (not invertible)
        for input_func, expected_func in zip(self.periodic_expected[1:],
                                             self.periodic_inputs[1:]):
            with self.subTest(function=expected_func.name):
                # Apply inverse Laplacian
                result = self.inv_laplacian_spectral_periodic(input_func)

                # Compute relative error
                error = self._compute_l2_relative_error(result, expected_func)

                # Check if error is below threshold
                self.assertLess(
                    error, self.moderate_threshold,
                    f"Spectral Inverse Laplacian Periodic error for "
                    f"{expected_func.name}: {error:.2e}"
                )

    # === FINITE ELEMENT INVERSE LAPLACIAN TESTS ===

    def test_inverse_laplacian_fem_dirichlet(self):
        """Test finite element inverse Laplacian with Dirichlet BC."""
        for input_func, expected_func in zip(self.dirichlet_expected,
                                             self.dirichlet_inputs):
            with self.subTest(function=expected_func.name):
                # Apply inverse Laplacian
                result = self.inv_laplacian_fem_dirichlet(input_func)

                # Compute relative error
                error = self._compute_l2_relative_error(result, expected_func)

                # Check if error is below threshold (looser for FEM methods)
                self.assertLess(
                    error, self.loose_threshold,
                    f"FEM Inverse Laplacian Dirichlet error for "
                    f"{expected_func.name}: {error:.2e}"
                )

    def test_inverse_laplacian_fem_neumann(self):
        """Test finite element inverse Laplacian with Neumann BC."""
        # Skip constant function for Neumann (not invertible)
        for input_func, expected_func in zip(self.neumann_expected[1:],
                                             self.neumann_inputs[1:]):
            with self.subTest(function=expected_func.name):
                # Apply inverse Laplacian
                result = self.inv_laplacian_fem_neumann(input_func)

                # Compute relative error
                error = self._compute_l2_relative_error(result, expected_func)

                # Check if error is below threshold
                self.assertLess(
                    error, self.loose_threshold,
                    f"FEM Inverse Laplacian Neumann error for "
                    f"{expected_func.name}: {error:.2e}"
                )

    def test_inverse_laplacian_fem_periodic(self):
        """Test finite element inverse Laplacian with periodic BC."""
        # Skip constant function for periodic (not invertible)
        for input_func, expected_func in zip(self.periodic_expected[1:],
                                             self.periodic_inputs[1:]):
            with self.subTest(function=expected_func.name):
                # Apply inverse Laplacian
                result = self.inv_laplacian_fem_periodic(input_func)

                # Compute relative error
                error = self._compute_l2_relative_error(result, expected_func)

                # Check if error is below threshold
                self.assertLess(
                    error, self.loose_threshold,
                    f"FEM Inverse Laplacian Periodic error for "
                    f"{expected_func.name}: {error:.2e}"
                )

    # === ROUND-TRIP TESTS (Laplacian then Inverse Laplacian) ===

    def test_roundtrip_spectral_dirichlet(self):
        """Test L->L^-1 roundtrip (spectral, Dirichlet)."""
        for input_func in self.dirichlet_inputs:
            with self.subTest(function=input_func.name):
                # Apply Laplacian then inverse
                laplacian_result = self.laplacian_spectral_dirichlet(
                    input_func
                )
                roundtrip_result = self.inv_laplacian_spectral_dirichlet(
                    laplacian_result
                )

                # Compute relative error
                error = self._compute_l2_relative_error(
                    roundtrip_result, input_func
                )

                # Check if error is below threshold
                self.assertLess(
                    error, self.moderate_threshold,
                    f"Spectral roundtrip Dirichlet error for "
                    f"{input_func.name}: {error:.2e}"
                )

    def test_roundtrip_spectral_neumann(self):
        """Test Laplacian followed by inverse Laplacian (spectral) Neumann."""
        # Skip constant function for Neumann (not invertible)
        for input_func in self.neumann_inputs[1:]:
            with self.subTest(function=input_func.name):
                # Apply Laplacian then inverse
                laplacian_result = self.laplacian_spectral_neumann(input_func)
                roundtrip_result = self.inv_laplacian_spectral_neumann(
                    laplacian_result
                )

                # Compute relative error
                error = self._compute_l2_relative_error(
                    roundtrip_result, input_func
                )

                # Check if error is below threshold
                self.assertLess(
                    error, self.moderate_threshold,
                    f"Spectral roundtrip Neumann error for "
                    f"{input_func.name}: {error:.2e}"
                )

    def test_roundtrip_spectral_periodic(self):
        """Test Laplacian followed by inverse Laplacian (spectral) periodic."""
        # Skip constant function for periodic (not invertible)
        for input_func in self.periodic_inputs[1:]:
            with self.subTest(function=input_func.name):
                # Apply Laplacian then inverse
                laplacian_result = self.laplacian_spectral_periodic(input_func)
                roundtrip_result = self.inv_laplacian_spectral_periodic(
                    laplacian_result
                )

                # Compute relative error
                error = self._compute_l2_relative_error(
                    roundtrip_result, input_func
                )

                # Check if error is below threshold
                self.assertLess(
                    error, self.moderate_threshold,
                    f"Spectral roundtrip Periodic error for "
                    f"{input_func.name}: {error:.2e}"
                )


if __name__ == '__main__':
    # Create a test suite
    suite = unittest.TestSuite()

    # Add all test methods
    test_methods = [
        'test_laplacian_spectral_dirichlet',
        'test_laplacian_spectral_neumann',
        'test_laplacian_spectral_periodic',
        'test_laplacian_fd_dirichlet',
        'test_laplacian_fd_neumann',
        'test_laplacian_fd_periodic',
        'test_inverse_laplacian_spectral_dirichlet',
        'test_inverse_laplacian_spectral_neumann',
        'test_inverse_laplacian_spectral_periodic',
        'test_inverse_laplacian_fem_dirichlet',
        'test_inverse_laplacian_fem_neumann',
        'test_inverse_laplacian_fem_periodic',
        'test_roundtrip_spectral_dirichlet',
        'test_roundtrip_spectral_neumann',
        'test_roundtrip_spectral_periodic'
    ]

    for method in test_methods:
        suite.addTest(TestLaplacianOperators(method))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'='*70}")
    print("Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    success_rate = (result.testsRun - len(result.failures) -
                    len(result.errors))/result.testsRun*100
    print(f"Success rate: {success_rate:.1f}%")
    print(f"{'='*70}")
