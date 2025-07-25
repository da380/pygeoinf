"""
Laplacian Inverse Operator with Multiple FEM Backend Support

This module provides the LaplacianInverseOperator that can use different
FEM solvers as backends (DOLFINx or native Python implementation).
"""

from .fem_solvers import create_fem_solver
from .sobolev_space import Sobolev
from .boundary_conditions import BoundaryConditions
from pygeoinf.hilbert_space import LinearOperator
from .l2_functions import L2Function


class LaplacianInverseOperator(LinearOperator):
    """
    Inverse Laplacian operator that acts as a covariance operator.

    This operator solves -Δu = f with homogeneous boundary conditions,
    providing a self-adjoint operator suitable for Gaussian measures.

    Supports multiple FEM backends:
    - 'dolfinx': Uses DOLFINx library (requires installation)
    - 'native': Pure Python implementation (no external dependencies)
    """

    def __init__(
        self,
        domain: Sobolev,
        /,
        *,
        dofs: int = 100,
        solver_type: str = "auto"
    ):
        """
        Initialize the Laplacian inverse operator.

        Args:
            domain: The Sobolev space where functions live (positional-only)
            dofs: Number of degrees of freedom (default: 100)
            solver_type: 'dolfinx', 'native', or 'auto'
        """
        # Check that domain is a Sobolev space
        if not isinstance(domain, Sobolev):
            raise TypeError(
                f"domain must be a Sobolev space, got {type(domain)}"
            )

        self._domain = domain
        self._dofs = dofs

        # Get boundary conditions from the Sobolev space
        self._boundary_conditions = domain.boundary_conditions

        # Get function domain from domain (L2Space and SobolevSpace)
        self._function_domain = domain.function_domain
        self._interval = (self._function_domain.a, self._function_domain.b)

        # Choose solver type
        if solver_type == "auto":
            # Try DOLFINx first, fall back to native
            try:
                from .fem_solvers import DOLFINX_AVAILABLE
                if DOLFINX_AVAILABLE:
                    # Try to create DOLFINx solver to test compatibility
                    try:
                        test_bc = BoundaryConditions('dirichlet')
                        test_solver = create_fem_solver(
                            "dolfinx", self._function_domain, 10, test_bc
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
            self._function_domain,
            dofs,
            self._boundary_conditions
        )
        self._fem_solver.setup()

        print(f"LaplacianInverseOperator initialized with {self._solver_type} "
              f"solver, {self._boundary_conditions} BCs")

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

        # Solve PDE using FEM solver with L2Function
        solution_values = self._fem_solver.solve_poisson(f)

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
    def dofs(self) -> int:
        """Get the mesh resolution."""
        return self._dofs

    def get_solver_info(self) -> dict:
        """Get information about the current solver."""
        return {
            'solver_type': self._solver_type,
            'boundary_conditions': self._boundary_conditions,
            'dofs': self._dofs,
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
            self._function_domain,
            self._dofs,
            self._boundary_conditions
        )
        self._fem_solver.setup()

        print(f"Switched to {self._solver_type} solver")
