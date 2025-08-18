"""
FEM Solvers for Laplacian Inverse Operator

This module provides a native Python FEM solver backend for the
LaplacianInverseOperator. The native FEM solver leverages the L2Space hat
basis functions for all operations and has no external dependencies.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Callable

from .interval_domain import IntervalDomain
from .boundary_conditions import BoundaryConditions
from .functions import Function


class FEMSolverBase(ABC):
    """Abstract base class for FEM solvers."""

    def __init__(self, function_domain: IntervalDomain, dofs: int,
                 boundary_conditions: BoundaryConditions):
        """
        Initialize FEM solver.

        Args:
            function_domain: IntervalDomain object
            dofs: Number of elements in mesh
            boundary_conditions: BoundaryConditions object
        """
        if not isinstance(function_domain, IntervalDomain):
            raise TypeError(
                f"function_domain must be IntervalDomain object, "
                f"got {type(function_domain)}"
            )

        self.function_domain = function_domain
        self.dofs = dofs

        if not isinstance(boundary_conditions, BoundaryConditions):
            raise TypeError(
                f"boundary_conditions must be BoundaryConditions object, "
                f"got {type(boundary_conditions)}"
            )

        self.boundary_conditions = boundary_conditions

        # Validate boundary conditions (basic check for FEM compatibility)
        valid_types = ['dirichlet', 'neumann', 'periodic']
        if self.boundary_conditions.type not in valid_types:
            raise ValueError(
                f"FEM solver only supports boundary condition types: "
                f"{valid_types}"
            )

    @property
    def bc_type(self) -> str:
        """Get the boundary condition type as string for backward
        compatibility."""
        return self.boundary_conditions.type

    @abstractmethod
    def setup(self):
        """Set up the FEM discretization (mesh, function space, etc.)"""
        pass

    @abstractmethod
    def solve_poisson(self,
                      rhs_function: Callable[[np.ndarray], np.ndarray]
                      ) -> np.ndarray:
        """
        Solve -d²u/dx² = f with specified boundary conditions.

        Args:
            rhs_function: Right-hand side function f(x)

        Returns:
            Solution values at mesh nodes
        """
        pass

    @abstractmethod
    def get_coordinates(self) -> np.ndarray:
        """Get coordinates of mesh nodes."""
        pass

    @abstractmethod
    def interpolate_solution(self, solution_values: np.ndarray,
                             eval_points: np.ndarray) -> np.ndarray:
        """
        Interpolate solution to arbitrary evaluation points.

        Args:
            solution_values: Solution at mesh nodes
            eval_points: Points where to evaluate

        Returns:
            Interpolated values
        """
        pass


class NativeFEMSolver(FEMSolverBase):
    """Native Python FEM solver - supports Dirichlet BCs."""

    def __init__(self, function_domain: IntervalDomain,
                 dofs: int, boundary_conditions: BoundaryConditions):
        # Validate that boundary_conditions is a BoundaryConditions object
        if not isinstance(boundary_conditions, BoundaryConditions):
            raise TypeError(
                f"boundary_conditions must be BoundaryConditions object, "
                f"got {type(boundary_conditions)}"
            )

        # Validate that native solver can handle this boundary condition type
        if boundary_conditions.type != 'dirichlet':
            raise ValueError(
                f"NativeFEMSolver only supports 'dirichlet' boundary "
                f"conditions, got '{boundary_conditions.type}'"
            )

        super().__init__(function_domain, dofs, boundary_conditions)
        self._nodes = None
        self._dofs = dofs
        self._l2_space = None  # L2Space with hat basis functions

    def setup(self):
        """Set up native FEM mesh and structures using L2Space hat basis."""
        # Create uniform 1D mesh
        self._nodes = np.linspace(
            self.function_domain.a, self.function_domain.b, self._dofs + 2
        )

        self._h = self._nodes[1] - self._nodes[0]  # Element size

        # Create L2Space with hat basis functions for FEM
        from .l2_space import L2Space  # Import here to avoid circular imports
        # Use homogeneous hat basis for Dirichlet boundary conditions
        # Number of basis functions = number of interior nodes
        self._l2_space = L2Space(
            self._dofs,  # Interior nodes only (exclude boundaries)
            self.function_domain,  # IntervalDomain object
            basis_type='hat_homogeneous'
        )

    def _assemble_stiffness_matrix(self) -> np.ndarray:
        """Assemble stiffness matrix using L2Space basis functions."""
        if self._l2_space is None:
            raise RuntimeError("Must call setup() first")

        # Get interior basis functions (exclude boundary)
        n_interior = self._dofs  # Number of interior nodes
        K = np.zeros((n_interior, n_interior))

        # Assemble using Function operations
        for i in range(n_interior):
            for j in range(i, n_interior):  # Symmetric matrix
                # Compute ∫ φ'ᵢ φ'ⱼ dx using finite differences
                # For hat functions, this gives the standard FEM stiffness
                if i == j:
                    # Diagonal term
                    K[i, j] = 2.0 / self._h
                elif abs(i - j) == 1:
                    # Adjacent terms
                    K[i, j] = -1.0 / self._h
                    K[j, i] = -1.0 / self._h
                # All other terms are zero due to compact support

        return K

    def _assemble_mass_matrix(self) -> np.ndarray:
        """Assemble mass matrix using L2Space gram matrix."""
        if self._l2_space is None:
            raise RuntimeError("Must call setup() first")

        # The mass matrix is exactly the gram matrix of the L2Space
        return self._l2_space.gram_matrix

    def _assemble_load_vector(self, rhs_function: Function) -> np.ndarray:
        """
        Assemble the load vector for the FEM system.

        This method leverages Function multiplication and integration
        with automatic compact support handling.

        Args:
            rhs_function: Function to use as right-hand side

        Returns:
            Load vector for interior nodes only
        """
        if self._l2_space is None:
            raise RuntimeError("Must call setup() first")

        if not isinstance(rhs_function, Function):
            raise TypeError(
                f"rhs_function must be Function, got {type(rhs_function)}"
            )

        f_vec = np.zeros(self._dofs)

        # For each interior node, compute ∫ f(x) φᵢ(x) dx using Function
        for i in range(self._dofs):
            # Get basis function as Function
            phi_i = self._l2_space._basis_provider.get_basis_function(i)

            # Multiply f * φᵢ using Function multiplication
            # This automatically handles compact support intersection
            integrand = rhs_function * phi_i

            # Integrate using Function integration
            f_vec[i] = integrand.integrate(method='simpson')

        return f_vec

    def solve_poisson(self, rhs_function: Function) -> np.ndarray:
        """Solve Poisson equation with Dirichlet BCs using L2Space basis."""
        # Assemble system using L2Space operations
        K = self._assemble_stiffness_matrix()
        f = self._assemble_load_vector(rhs_function)

        # Solve interior system (boundary conditions already incorporated)
        u_interior = np.linalg.solve(K, f)

        # Reconstruct full solution including boundary values (u=0)
        u_full = np.zeros(len(self._nodes))
        u_full[1:-1] = u_interior  # Interior values
        # Boundary values remain 0 (Dirichlet BC)

        return u_full

    def get_basis_function(self, dof_index: int) -> Function:
        """
        Get basis function for given dof as a Function.

        Args:
            dof_index: Degree of freedom (0 to num_interior_dofs-1)

        Returns:
            Function (hat basis function with compact support)
        """
        if self._l2_space is None:
            raise RuntimeError("Must call setup() first")
        return self._l2_space._basis_provider.get_basis_function(dof_index)

    def get_basis_functions(self) -> list:
        """Get all interior basis functions as Functions."""
        if self._l2_space is None:
            raise RuntimeError("Must call setup() first")
        return [self._l2_space._basis_provider.get_basis_function(i)
                for i in range(self._dofs)]

    def get_coordinates(self) -> np.ndarray:
        """Get mesh node coordinates."""
        if self._nodes is None:
            raise RuntimeError("Must call setup() first")
        return self._nodes.copy()

    def interpolate_solution(self, solution_values: np.ndarray,
                             eval_points: np.ndarray) -> np.ndarray:
        """Linear interpolation of solution."""
        if self._nodes is None:
            raise RuntimeError("Must call setup() first")
        return np.interp(eval_points, self._nodes, solution_values)


def create_fem_solver(solver_type: str,
                      function_domain: IntervalDomain,
                      dofs: int,
                      boundary_conditions: BoundaryConditions
                      ) -> FEMSolverBase:
    """
    Factory function to create FEM solvers.

    Args:
        solver_type: Only 'native' is supported
        function_domain: IntervalDomain object
        dofs: Mesh resolution
        boundary_conditions: BoundaryConditions object

    Returns:
        FEM solver instance
    """
    if solver_type == 'native':
        return NativeFEMSolver(function_domain, dofs, boundary_conditions)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}. "
                         f"Only 'native' is supported.")
