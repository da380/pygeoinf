"""
Laplacian Inverse Operator with Multiple FEM Backend Support

This module provides the LaplacianInverseOperator that can use different
FEM solvers as backends (DOLFINx or native Python implementation).
"""

import numpy as np
from typing import Callable, Union
from .fem_solvers import create_fem_solver
from .interval_domain import BoundaryConditions
from pygeoinf.hilbert_space import LinearOperator
from pygeoinf.interval.l2_functions import L2Function


class LaplacianInverseOperator(LinearOperator):
    """
    Inverse Laplacian operator that acts as a covariance operator.

    This operator solves -Δu = f with homogeneous boundary conditions,
    providing a self-adjoint operator suitable for Gaussian measures.

    Supports multiple FEM backends:
    - 'dolfinx': Uses DOLFINx library (requires installation)
    - 'native': Pure Python implementation (no external dependencies)
    """

    def __init__(self, domain, mesh_resolution: int = 100,
                 boundary_conditions: Union[str, BoundaryConditions] = "dirichlet",
                 solver_type: str = "auto"):
        """
        Initialize the Laplacian inverse operator.

        Args:
            domain: The L2Space or SobolevSpace where functions live
            mesh_resolution: Number of FEM elements
            boundary_conditions: BoundaryConditions object or str for backward compatibility
            solver_type: 'dolfinx', 'native', or 'auto'
        """
        self._domain = domain
        self._mesh_resolution = mesh_resolution

        # Convert string to BoundaryConditions object if needed
        if isinstance(boundary_conditions, str):
            if boundary_conditions == 'dirichlet':
                self._boundary_conditions = BoundaryConditions.dirichlet()
            elif boundary_conditions == 'neumann':
                self._boundary_conditions = BoundaryConditions.neumann()
            elif boundary_conditions == 'periodic':
                self._boundary_conditions = BoundaryConditions.periodic()
            else:
                raise ValueError(f"Unknown boundary condition string: {boundary_conditions}")
        elif isinstance(boundary_conditions, BoundaryConditions):
            self._boundary_conditions = boundary_conditions
        else:
            raise ValueError(f"boundary_conditions must be BoundaryConditions or str, got {type(boundary_conditions)}")

        self._interval = (domain.domain.a, domain.domain.b)

        # Choose solver type
        if solver_type == "auto":
            # Try DOLFINx first, fall back to native
            try:
                from .fem_solvers import DOLFINX_AVAILABLE
                if DOLFINX_AVAILABLE:
                    # Try to create DOLFINx solver to test compatibility
                    try:
                        test_solver = create_fem_solver(
                            "dolfinx", self._interval, 10, "dirichlet"
                        )
                        test_solver.setup()
                        self._solver_type = "dolfinx"
                    except Exception:
                        # DOLFINx available but has compatibility issues
                        self._solver_type = "native"
                        print("DOLFINx found but has compatibility issues, "
                              "using native solver")
                else:
                    self._solver_type = "native"
            except ImportError:
                self._solver_type = "native"
        else:
            self._solver_type = solver_type

        # Create and setup FEM solver
        self._fem_solver = create_fem_solver(
            self._solver_type,
            self._interval,
            mesh_resolution,
            self._boundary_conditions
        )
        self._fem_solver.setup()

        print(f"LaplacianInverseOperator initialized with {self._solver_type} "
              f"solver, {boundary_conditions} BCs")

        # Initialize LinearOperator with self-adjoint mapping
        # The operator solves -Δu = f, so it's the inverse of -Δ
        super().__init__(
            domain, domain,
            self._solve_laplacian,
            adjoint_mapping=self._solve_laplacian  # Self-adjoint
        )

    def _solve_laplacian(self, f: L2Function) -> L2Function:
        """
        Apply the inverse Laplacian operator to function f.

        Args:
            f: L2Function or SobolevFunction to apply operator to

        Returns:
            L2Function representing (-Δ)⁻¹f
        """
        # Convert hilbert space function to evaluation function
        def rhs_function(x):
            return f.evaluate(x)

        # Solve PDE using FEM solver
        solution_values = self._fem_solver.solve_poisson(rhs_function)

        # Convert back to hilbert space function
        return self._fem_to_hilbert_function(solution_values)

    def _fem_to_hilbert_function(self, fem_solution):
        """Convert FEM solution to hilbert space function."""

        # Create interpolation function
        def evaluate_func(x):
            return self._fem_solver.interpolate_solution(fem_solution, x)

        # Create L2Function with interpolated solution
        return L2Function(
            self._domain,
            evaluate_callable=evaluate_func,
            name=f"Laplacian⁻¹ solution ({self._solver_type})"
        )

    @property
    def solver_type(self) -> str:
        """Get the current solver type."""
        return self._solver_type

    @property
    def boundary_conditions(self) -> BoundaryConditions:
        """Get the boundary conditions object."""
        return self._boundary_conditions

    @property
    def mesh_resolution(self) -> int:
        """Get the mesh resolution."""
        return self._mesh_resolution

    def get_solver_info(self) -> dict:
        """Get information about the current solver."""
        return {
            'solver_type': self._solver_type,
            'boundary_conditions': self._boundary_conditions,
            'mesh_resolution': self._mesh_resolution,
            'interval': self._interval,
            'fem_coordinates': self._fem_solver.get_coordinates()
        }

    def change_solver(self, new_solver_type: str):
        """
        Change the FEM solver backend.

        Args:
            new_solver_type: 'dolfinx' or 'native'
        """
        if new_solver_type == self._solver_type:
            print(f"Already using {new_solver_type} solver")
            return

        # Create new solver
        self._solver_type = new_solver_type
        self._fem_solver = create_fem_solver(
            self._solver_type,
            self._interval,
            self._mesh_resolution,
            self._boundary_conditions
        )
        self._fem_solver.setup()

        print(f"Switched to {self._solver_type} solver")

    def benchmark_solvers(self,
                          test_function: Callable[[np.ndarray], np.ndarray],
                          num_runs: int = 5) -> dict:
        """
        Benchmark different solvers on a test function.

        Args:
            test_function: Function to use for testing
            num_runs: Number of runs for timing

        Returns:
            Dictionary with timing results
        """
        import time
        from .l2_functions import L2Function

        results = {}
        original_solver = self._solver_type

        # Test each available solver
        solvers_to_test = ['native']
        try:
            from .fem_solvers import DOLFINX_AVAILABLE
            if DOLFINX_AVAILABLE:
                solvers_to_test.append('dolfinx')
        except ImportError:
            pass

        test_func = L2Function(
            self._domain,
            evaluate_callable=test_function,
            name="Benchmark test function"
        )

        for solver in solvers_to_test:
            try:
                self.change_solver(solver)

                # Warm up
                _ = self(test_func)

                # Time multiple runs
                times = []
                for _ in range(num_runs):
                    start = time.time()
                    _ = self(test_func)
                    end = time.time()
                    times.append(end - start)

                results[solver] = {
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times)
                }

            except Exception as e:
                results[solver] = {'error': str(e)}

        # Restore original solver
        self.change_solver(original_solver)

        return results
