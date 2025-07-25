"""
FEM Solvers for Laplacian Inverse Operator

This module provides different FEM solver backends for the LaplacianInverseOperator.
Users can choose between:
1. DOLFINx-based solver (requires DOLFINx installation)
2. Custom native Python FEM solver (no external dependencies)

The native FEM solver leverages the L2Space hat basis functions for all operations.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Callable
import warnings

from .interval_domain import IntervalDomain
from .boundary_conditions import BoundaryConditions
from .l2_functions import L2Function

try:
    import dolfinx
    import dolfinx.fem
    import dolfinx.fem.petsc
    import ufl
    from petsc4py import PETSc
    DOLFINX_AVAILABLE = True
except ImportError:
    DOLFINX_AVAILABLE = False


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
                f"FEM solver only supports boundary condition types: {valid_types}"
            )

    @property
    def bc_type(self) -> str:
        """Get the boundary condition type as string for backward compatibility."""
        return self.boundary_conditions.type

    @abstractmethod
    def setup(self):
        """Set up the FEM discretization (mesh, function space, etc.)"""
        pass

    @abstractmethod
    def solve_poisson(self, rhs_function: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
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


class DOLFINxSolver(FEMSolverBase):
    """DOLFINx-based FEM solver."""

    def __init__(self, function_domain: IntervalDomain, dofs: int,
                 boundary_conditions: BoundaryConditions):
        if not DOLFINX_AVAILABLE:
            raise ImportError(
                "DOLFINx is not available. Install with: "
                "conda install -c conda-forge fenics-dolfinx"
            )

        super().__init__(function_domain, dofs, boundary_conditions)
        self._mesh = None
        self._V = None
        self._bcs = None
        self._periodic_dofs = None

    def setup(self):
        """Set up DOLFINx mesh and function space."""
        from dolfinx import mesh

        # Create 1D interval mesh
        self._mesh = mesh.create_interval(
            comm=PETSc.COMM_WORLD,
            nx=self.dofs,
            points=np.array([self.function_domain.a, self.function_domain.b])
        )

        # Create function space (P1 Lagrange elements)
        self._V = dolfinx.fem.FunctionSpace(self._mesh, ("Lagrange", 1))

        # Set up boundary conditions
        self._setup_boundary_conditions()

    def _setup_boundary_conditions(self):
        """Set up boundary conditions."""
        self._bcs = []

        if self.bc_type == 'dirichlet':
            # Homogeneous Dirichlet: u = 0 on boundary
            def boundary_all(x):
                return np.logical_or(
                    np.isclose(x[0], self.function_domain.a),
                    np.isclose(x[0], self.function_domain.b)
                )

            boundary_dofs = dolfinx.fem.locate_dofs_geometrical(self._V, boundary_all)
            bc_dirichlet = dolfinx.fem.dirichletbc(
                PETSc.ScalarType(0), boundary_dofs, self._V
            )
            self._bcs = [bc_dirichlet]

        elif self.bc_type == 'neumann':
            # Natural boundary conditions (no explicit BCs needed)
            self._bcs = []

        elif self.bc_type == 'periodic':
            # Set up periodic constraint
            self._setup_periodic_bcs()

    def _setup_periodic_bcs(self):
        """Set up periodic boundary conditions."""
        # Get DOF coordinates and find boundary DOFs
        dof_coords = self._V.tabulate_dof_coordinates()[:, 0]

        left_dofs = []
        right_dofs = []

        for i, coord in enumerate(dof_coords):
            if np.isclose(coord, self.function_domain.a):
                left_dofs.append(i)
            elif np.isclose(coord, self.function_domain.b):
                right_dofs.append(i)

        if len(left_dofs) != 1 or len(right_dofs) != 1:
            raise ValueError("Unexpected number of boundary DOFs for periodic BC")

        self._periodic_dofs = (left_dofs[0], right_dofs[0])
        # Note: Periodic constraint will be applied in the solver

    def solve_poisson(self, rhs_function: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Solve Poisson equation with DOLFINx."""
        # Create trial and test functions
        u = ufl.TrialFunction(self._V)
        v = ufl.TestFunction(self._V)

        # Create RHS function
        f = dolfinx.fem.Function(self._V)
        coords = self._V.tabulate_dof_coordinates()[:, 0]
        f.x.array[:] = rhs_function(coords)

        # Bilinear form: a(u,v) = ∫ ∇u·∇v dx
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

        # Linear form: L(v) = ∫ f·v dx
        L = ufl.inner(f, v) * ufl.dx

        if self.bc_type == 'periodic':
            # Handle periodic BCs with custom matrix modification
            return self._solve_periodic(a, L, f)
        else:
            # Standard solve
            return self._solve_standard(a, L, f)

    def _solve_standard(self, a, L, f):
        """Standard DOLFINx solve."""
        if self.bc_type == 'neumann':
            # Singular system - use iterative solver
            petsc_options = {
                "ksp_type": "cg",
                "pc_type": "jacobi",
                "ksp_rtol": 1e-10,
                "ksp_max_it": 1000
            }

            # Check solvability condition
            dx = ufl.dx
            volume = dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(dolfinx.fem.Constant(self._mesh, 1.0) * dx)
            )
            rhs_integral = dolfinx.fem.assemble_scalar(dolfinx.fem.form(f * dx))

            if abs(rhs_integral) > 1e-12:
                mean_rhs = rhs_integral / volume
                f.x.array[:] -= mean_rhs
        else:
            # Regular system - direct solver
            petsc_options = {
                "ksp_type": "preonly",
                "pc_type": "lu"
            }

        # Solve
        problem = dolfinx.fem.petsc.LinearProblem(
            a, L, bcs=self._bcs, petsc_options=petsc_options
        )
        solution_func = problem.solve()

        # Enforce zero mean for Neumann
        if self.bc_type == 'neumann':
            dx = ufl.dx
            volume = dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(dolfinx.fem.Constant(self._mesh, 1.0) * dx)
            )
            mean_sol = dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(solution_func * dx)
            ) / volume
            solution_func.x.array[:] -= mean_sol

        return solution_func.x.array.copy()

    def _solve_periodic(self, a, L, f):
        """Solve with periodic boundary conditions."""
        # Assemble system manually to modify for periodicity
        a_form = dolfinx.fem.form(a)
        L_form = dolfinx.fem.form(L)

        A = dolfinx.fem.petsc.assemble_matrix(a_form)
        A.assemble()
        b = dolfinx.fem.petsc.assemble_vector(L_form)
        b.assemble()

        # Apply periodic constraint: u(left) = u(right)
        left_dof, right_dof = self._periodic_dofs

        # Modify matrix: replace right DOF equation with constraint
        A.zeroRows([right_dof], diag=1.0)
        A.setValue(right_dof, right_dof, 1.0)
        A.setValue(right_dof, left_dof, -1.0)
        b.setValue(right_dof, 0.0)

        A.assemble()
        b.assemble()

        # Check solvability and solve
        dx = ufl.dx
        volume = dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(dolfinx.fem.Constant(self._mesh, 1.0) * dx)
        )
        rhs_integral = dolfinx.fem.assemble_scalar(dolfinx.fem.form(f * dx))

        if abs(rhs_integral) > 1e-12:
            mean_rhs = rhs_integral / volume
            b_array = b.getArray()
            b_array[:] -= mean_rhs * volume / len(b_array)
            b.restoreArray(b_array)

        # Solve system
        u = b.duplicate()
        ksp = PETSc.KSP().create()
        ksp.setOperators(A)
        ksp.setType("cg")
        ksp.getPC().setType("jacobi")
        ksp.setTolerances(rtol=1e-10, max_it=1000)
        ksp.solve(b, u)

        solution = u.getArray().copy()

        # Enforce zero mean
        mean_sol = np.mean(solution)
        solution -= mean_sol

        return solution

    def get_coordinates(self) -> np.ndarray:
        """Get mesh node coordinates."""
        return self._V.tabulate_dof_coordinates()[:, 0]

    def interpolate_solution(self, solution_values: np.ndarray,
                           eval_points: np.ndarray) -> np.ndarray:
        """Interpolate using DOLFINx function."""
        # Create function from solution values
        solution_func = dolfinx.fem.Function(self._V)
        solution_func.x.array[:] = solution_values

        # Evaluate at points
        points = np.column_stack([eval_points, np.zeros_like(eval_points),
                                 np.zeros_like(eval_points)])
        return solution_func.eval(points, self._mesh.geometry.x)


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

        # Assemble using L2Function operations
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

    def _assemble_load_vector(self, rhs_function: L2Function) -> np.ndarray:
        """
        Assemble load vector using L2Space basis functions.

        This method leverages L2Function multiplication and integration
        with automatic compact support handling.

        Args:
            rhs_function: L2Function to use as right-hand side

        Returns:
            Load vector for interior nodes only
        """
        if self._l2_space is None:
            raise RuntimeError("Must call setup() first")

        if not isinstance(rhs_function, L2Function):
            raise TypeError(
                f"rhs_function must be L2Function, got {type(rhs_function)}"
            )

        f_vec = np.zeros(self._dofs)

        # For each interior node, compute ∫ f(x) φᵢ(x) dx using L2Function
        for i in range(self._dofs):
            # Get basis function as L2Function
            phi_i = self._l2_space._basis_provider.get_basis_function(i)

            # Multiply f * φᵢ using L2Function multiplication
            # This automatically handles compact support intersection
            integrand = rhs_function * phi_i

            # Integrate using L2Function integration
            f_vec[i] = integrand.integrate(method='simpson')

        return f_vec

    def solve_poisson(self, rhs_function: L2Function) -> np.ndarray:
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

    def get_basis_function(self, dof_index: int) -> L2Function:
        """
        Get basis function for given dof as an L2Function.

        Args:
            dof_index: Degree of freedom (0 to num_interior_dofs-1)

        Returns:
            L2Function (hat basis function with compact support)
        """
        if self._l2_space is None:
            raise RuntimeError("Must call setup() first")
        return self._l2_space._basis_provider.get_basis_function(dof_index)

    def get_basis_functions(self) -> list:
        """Get all interior basis functions as L2Functions."""
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
        solver_type: 'dolfinx' or 'native'
        function_domain: IntervalDomain object
        dofs: Mesh resolution
        boundary_conditions: BoundaryConditions object

    Returns:
        FEM solver instance
    """
    if solver_type == 'dolfinx':
        return DOLFINxSolver(function_domain, dofs, boundary_conditions)
    elif solver_type == 'native':
        return NativeFEMSolver(function_domain, dofs, boundary_conditions)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")
