"""
Laplacian Inverse Operator with Multiple FEM Backend Support

This module provides the LaplacianInverseOperator that can use different
FEM solvers as backends (DOLFINx or native Python implementation).
"""

import numpy as np
from .fem_solvers import create_fem_solver
from .l2_space import L2Space
from .sobolev_space import Sobolev
from .boundary_conditions import BoundaryConditions
from pygeoinf.hilbert_space import LinearOperator
from .l2_functions import Function
from .providers import create_laplacian_spectrum_provider


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
        domain: L2Space,
        boundary_conditions: BoundaryConditions,
        /,
        *,
        dofs: int = 100,
        solver_type: str = "auto"
    ):
        """
        Initialize the Laplacian inverse operator.

        The operator maps from L² space to Sobolev space:
        (-Δ)⁻¹: L² → H^s with specified boundary conditions

        Args:
            domain: The L2 space (domain of the operator)
            boundary_conditions: Boundary conditions for the operator
            sobolev_order: Order of the Sobolev space (codomain), default 1.0
            dofs: Number of degrees of freedom (default: 100)
            solver_type: 'dolfinx', 'native', or 'auto'
        """
        # Check that domain is an L2 space
        if not isinstance(domain, L2Space):
            raise TypeError(
                f"domain must be an L2 space, got {type(domain)}"
            )

        self._domain = domain
        self._dofs = dofs
        self._boundary_conditions = boundary_conditions

        # Create the Sobolev space as codomain
        self._codomain = Sobolev(
            domain.dim,
            domain.function_domain,
            2.0,
            'spectral',
            basis_type='fourier',  # Use Fourier basis for spectral approach
            boundary_conditions=boundary_conditions
        )

        # Get function domain from domain (L2Space and SobolevSpace)
        self._function_domain = domain.function_domain
        self._interval = (self._function_domain.a, self._function_domain.b)

        # Initialize spectrum provider for lazy eigenvalue computation
        self._spectrum_provider = create_laplacian_spectrum_provider(
            domain, self._boundary_conditions, inverse=True
        )

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

        # Initialize LinearOperator with L2 domain and Sobolev codomain
        # The operator maps (-Δ)⁻¹: L² → H^s
        super().__init__(
            domain, self._codomain,
            self._solve_laplacian,
            adjoint_mapping=None  # Adjoint maps H^s → L²
        )

    def _solve_laplacian(self, f: Function) -> Function:
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

        # Create Function with interpolated solution
        return Function(
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

    @property
    def spectrum_provider(self):
        """Get the spectrum provider for eigenvalue computations."""
        return self._spectrum_provider

    def get_eigenvalue(self, index: int) -> float:
        """
        Get eigenvalue of the inverse Laplacian operator.

        Args:
            index: Index of the eigenvalue (0 to dim-1)

        Returns:
            float: Eigenvalue λₖ of (-Δ)^(-1)
        """
        return self._spectrum_provider.get_eigenvalue(index)

    def get_eigenfunction(self, index: int) -> Function:
        """
        Get eigenfunction of the inverse Laplacian operator.

        Args:
            index: Index of the eigenfunction (0 to dim-1)

        Returns:
            Function: Eigenfunction φₖ of (-Δ)^(-1)
        """
        return self._spectrum_provider.get_basis_function(index)

    def get_all_eigenvalues(self, n: int | None = None) -> np.ndarray:
        """
        Get all eigenvalues of the inverse Laplacian operator.
        Args:
            n: Number of eigenvalues to compute (default: space.dim)
        Returns:
            np.ndarray: Array of all eigenvalues
        """
        if n is None:
            n = self._domain.dim
        return self._spectrum_provider.get_all_eigenvalues(n)

    def get_solver_info(self) -> dict:
        """Get information about the current solver."""
        return {
            'solver_type': self._solver_type,
            'boundary_conditions': self._boundary_conditions,
            'dofs': self._dofs,
            'interval': self._interval,
            'fem_coordinates': self._fem_solver.get_coordinates(),
            'spectrum_available': True,
            'eigenvalue_range': (
                self.get_eigenvalue(0) if self._domain.dim > 0 else None,
                (self.get_eigenvalue(self._domain.dim - 1)
                 if self._domain.dim > 0 else None)
            )
        }

    def change_solver(self, new_solver_type: str):
        """
        Change the FEM solver backend.

        Args:
            new_solver_type: 'dolfinx' or 'native'

        Note:
            This does not affect the spectrum computation, which is
            analytical and independent of the FEM solver.
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
