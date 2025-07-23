"""
FEM Solvers for Laplacian Inverse Operator

This module provides different FEM solver backends for the LaplacianInverseOperator.
Users can choose between:
1. DOLFINx-based solver (requires DOLFINx installation)
2. Custom native Python FEM solver (no external dependencies)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Callable
import warnings

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

    def __init__(self, interval: Tuple[float, float], mesh_resolution: int,
                 boundary_conditions: str):
        """
        Initialize FEM solver.

        Args:
            interval: Domain interval [a, b]
            mesh_resolution: Number of elements in mesh
            boundary_conditions: Type of BCs ('dirichlet', 'neumann', 'periodic')
        """
        self.interval = interval
        self.mesh_resolution = mesh_resolution
        self.boundary_conditions = boundary_conditions

        # Validate boundary conditions
        valid_bcs = ['dirichlet', 'neumann', 'periodic']
        if boundary_conditions not in valid_bcs:
            raise ValueError(f"boundary_conditions must be one of {valid_bcs}")

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

    def __init__(self, interval: Tuple[float, float], mesh_resolution: int,
                 boundary_conditions: str):
        if not DOLFINX_AVAILABLE:
            raise ImportError(
                "DOLFINx is not available. Install with: "
                "conda install -c conda-forge fenics-dolfinx"
            )

        super().__init__(interval, mesh_resolution, boundary_conditions)
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
            nx=self.mesh_resolution,
            points=np.array([self.interval[0], self.interval[1]])
        )

        # Create function space (P1 Lagrange elements)
        self._V = dolfinx.fem.FunctionSpace(self._mesh, ("Lagrange", 1))

        # Set up boundary conditions
        self._setup_boundary_conditions()

    def _setup_boundary_conditions(self):
        """Set up boundary conditions."""
        self._bcs = []

        if self.boundary_conditions == 'dirichlet':
            # Homogeneous Dirichlet: u = 0 on boundary
            def boundary_all(x):
                return np.logical_or(
                    np.isclose(x[0], self.interval[0]),
                    np.isclose(x[0], self.interval[1])
                )

            boundary_dofs = dolfinx.fem.locate_dofs_geometrical(self._V, boundary_all)
            bc_dirichlet = dolfinx.fem.dirichletbc(
                PETSc.ScalarType(0), boundary_dofs, self._V
            )
            self._bcs = [bc_dirichlet]

        elif self.boundary_conditions == 'neumann':
            # Natural boundary conditions (no explicit BCs needed)
            self._bcs = []

        elif self.boundary_conditions == 'periodic':
            # Set up periodic constraint
            self._setup_periodic_bcs()

    def _setup_periodic_bcs(self):
        """Set up periodic boundary conditions."""
        # Get DOF coordinates and find boundary DOFs
        dof_coords = self._V.tabulate_dof_coordinates()[:, 0]

        left_dofs = []
        right_dofs = []

        for i, coord in enumerate(dof_coords):
            if np.isclose(coord, self.interval[0]):
                left_dofs.append(i)
            elif np.isclose(coord, self.interval[1]):
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

        if self.boundary_conditions == 'periodic':
            # Handle periodic BCs with custom matrix modification
            return self._solve_periodic(a, L, f)
        else:
            # Standard solve
            return self._solve_standard(a, L, f)

    def _solve_standard(self, a, L, f):
        """Standard DOLFINx solve."""
        if self.boundary_conditions == 'neumann':
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
        if self.boundary_conditions == 'neumann':
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
    """Native Python FEM solver - no external dependencies."""

    def __init__(self, interval: Tuple[float, float], mesh_resolution: int,
                 boundary_conditions: str):
        super().__init__(interval, mesh_resolution, boundary_conditions)
        self._nodes = None
        self._elements = None
        self._boundary_nodes = None

    def setup(self):
        """Set up native FEM mesh and structures."""
        # Create uniform 1D mesh
        self._nodes = np.linspace(self.interval[0], self.interval[1],
                                 self.mesh_resolution + 1)

        # Create elements (pairs of consecutive nodes)
        self._elements = np.column_stack([
            np.arange(self.mesh_resolution),
            np.arange(1, self.mesh_resolution + 1)
        ])

        # Identify boundary nodes
        self._boundary_nodes = {
            'left': 0,
            'right': self.mesh_resolution
        }

    def _assemble_stiffness_matrix(self) -> np.ndarray:
        """Assemble global stiffness matrix."""
        n_nodes = len(self._nodes)
        K = np.zeros((n_nodes, n_nodes))

        for elem in self._elements:
            i, j = elem[0], elem[1]
            h = self._nodes[j] - self._nodes[i]  # Element length

            # Local stiffness matrix for 1D Laplacian
            k_local = (1.0 / h) * np.array([[1, -1], [-1, 1]])

            # Add to global matrix
            K[i, i] += k_local[0, 0]
            K[i, j] += k_local[0, 1]
            K[j, i] += k_local[1, 0]
            K[j, j] += k_local[1, 1]

        return K

    def _assemble_mass_matrix(self) -> np.ndarray:
        """Assemble mass matrix for RHS integration."""
        n_nodes = len(self._nodes)
        M = np.zeros((n_nodes, n_nodes))

        for elem in self._elements:
            i, j = elem[0], elem[1]
            h = self._nodes[j] - self._nodes[i]

            # Local mass matrix
            m_local = (h / 6.0) * np.array([[2, 1], [1, 2]])

            # Add to global matrix
            M[i, i] += m_local[0, 0]
            M[i, j] += m_local[0, 1]
            M[j, i] += m_local[1, 0]
            M[j, j] += m_local[1, 1]

        return M

    def _assemble_rhs(self, rhs_function: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Assemble right-hand side vector."""
        n_nodes = len(self._nodes)
        f_vec = np.zeros(n_nodes)

        # Evaluate RHS function at nodes
        f_vals = rhs_function(self._nodes)

        # Simple lumped mass approach (sufficient for uniform mesh)
        for i in range(n_nodes):
            if i == 0:  # Left boundary
                h = self._nodes[1] - self._nodes[0]
                f_vec[i] = f_vals[i] * h / 2.0
            elif i == n_nodes - 1:  # Right boundary
                h = self._nodes[i] - self._nodes[i-1]
                f_vec[i] = f_vals[i] * h / 2.0
            else:  # Interior nodes
                h_left = self._nodes[i] - self._nodes[i-1]
                h_right = self._nodes[i+1] - self._nodes[i]
                f_vec[i] = f_vals[i] * (h_left + h_right) / 2.0

        return f_vec

    def solve_poisson(self, rhs_function: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Solve Poisson equation with native FEM."""
        # Assemble system
        K = self._assemble_stiffness_matrix()
        f = self._assemble_rhs(rhs_function)

        if self.boundary_conditions == 'dirichlet':
            return self._solve_dirichlet(K, f)
        elif self.boundary_conditions == 'neumann':
            return self._solve_neumann(K, f)
        elif self.boundary_conditions == 'periodic':
            return self._solve_periodic(K, f)
        else:
            raise ValueError(f"Unsupported boundary condition: {self.boundary_conditions}")

    def _solve_dirichlet(self, K: np.ndarray, f: np.ndarray) -> np.ndarray:
        """Solve with homogeneous Dirichlet BCs."""
        # Apply BCs: u = 0 at boundaries
        left_idx = self._boundary_nodes['left']
        right_idx = self._boundary_nodes['right']

        # Modify system
        K[left_idx, :] = 0
        K[left_idx, left_idx] = 1
        f[left_idx] = 0

        K[right_idx, :] = 0
        K[right_idx, right_idx] = 1
        f[right_idx] = 0

        # Solve
        return np.linalg.solve(K, f)

    def _solve_neumann(self, K: np.ndarray, f: np.ndarray) -> np.ndarray:
        """Solve with Neumann BCs (natural BCs)."""
        # Check solvability condition
        total_rhs = np.sum(f)
        if abs(total_rhs) > 1e-12:
            # Project to solvable subspace
            f = f - total_rhs / len(f)

        # Solve singular system with constraint
        n = len(f)

        # Add constraint equation: ∑u_i = 0
        K_aug = np.block([[K, np.ones((n, 1))],
                          [np.ones((1, n)), np.zeros((1, 1))]])
        f_aug = np.append(f, 0)

        # Solve augmented system
        solution_aug = np.linalg.solve(K_aug, f_aug)

        return solution_aug[:n]

    def _solve_periodic(self, K: np.ndarray, f: np.ndarray) -> np.ndarray:
        """Solve with periodic BCs."""
        left_idx = self._boundary_nodes['left']
        right_idx = self._boundary_nodes['right']

        # Apply periodic constraint: u_left = u_right
        # Replace right equation with constraint equation
        K[right_idx, :] = 0
        K[right_idx, left_idx] = -1
        K[right_idx, right_idx] = 1
        f[right_idx] = 0

        # Check solvability condition
        total_rhs = np.sum(f)
        if abs(total_rhs) > 1e-12:
            f = f - total_rhs / len(f)

        # Solve with constraint
        n = len(f)
        K_aug = np.block([[K, np.ones((n, 1))],
                          [np.ones((1, n)), np.zeros((1, 1))]])
        f_aug = np.append(f, 0)

        solution_aug = np.linalg.solve(K_aug, f_aug)

        return solution_aug[:n]

    def get_coordinates(self) -> np.ndarray:
        """Get mesh node coordinates."""
        return self._nodes.copy()

    def interpolate_solution(self, solution_values: np.ndarray,
                           eval_points: np.ndarray) -> np.ndarray:
        """Linear interpolation of solution."""
        return np.interp(eval_points, self._nodes, solution_values)


def create_fem_solver(solver_type: str, interval: Tuple[float, float],
                      mesh_resolution: int, boundary_conditions: str) -> FEMSolverBase:
    """
    Factory function to create FEM solvers.

    Args:
        solver_type: 'dolfinx' or 'native'
        interval: Domain interval
        mesh_resolution: Mesh resolution
        boundary_conditions: Boundary condition type

    Returns:
        FEM solver instance
    """
    if solver_type == 'dolfinx':
        return DOLFINxSolver(interval, mesh_resolution, boundary_conditions)
    elif solver_type == 'native':
        return NativeFEMSolver(interval, mesh_resolution, boundary_conditions)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")
