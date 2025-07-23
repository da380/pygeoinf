"""
FEM Solvers for Laplacian Inverse Operator

This module provides different FEM solver backends for the LaplacianInverseOperator.
Users can choose between:
1. DOLFINx-based solver (requires DOLFINx installation)
2. Custom native Python FEM solver (no external dependencies)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Callable, Union
import warnings

from .interval_domain import IntervalDomain
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


class FEMBasisFunction(L2Function):
    """
    A single FEM basis function as an L2Function with compact support.

    Represents a piecewise linear basis function φᵢ with support on
    at most two adjacent elements.
    """

    def __init__(self, space, node_index: int, nodes: np.ndarray,
                 element_size: float):
        """
        Initialize a FEM basis function.

        Args:
            space: L2Space this function belongs to
            node_index: Index of the node (0 to len(nodes)-1)
            nodes: Array of all mesh nodes
            element_size: Mesh element size h
        """
        self.node_index = node_index
        self.nodes = nodes
        self.element_size = element_size

        # Determine compact support
        support_elements = []

        # Left element (if exists)
        if node_index > 0:
            support_elements.append((node_index - 1, node_index))

        # Right element (if exists)
        if node_index < len(nodes) - 1:
            support_elements.append((node_index, node_index + 1))

        # Compact support is union of support elements
        if support_elements:
            left_nodes = [elem[0] for elem in support_elements]
            right_nodes = [elem[1] for elem in support_elements]
            support_a = nodes[min(left_nodes)]
            support_b = nodes[max(right_nodes)]
            support = (support_a, support_b)
        else:
            # Single node (shouldn't happen in practice)
            support = (nodes[node_index], nodes[node_index])

        # Create the basis function callable
        def basis_callable(x):
            x_array = np.asarray(x)
            is_scalar = x_array.ndim == 0
            if is_scalar:
                x_array = x_array.reshape(1)

            result = np.zeros_like(x_array, dtype=float)

            # Evaluate on each element in support
            for left_idx, right_idx in support_elements:
                x_left = nodes[left_idx]
                x_right = nodes[right_idx]

                # Find points in this element
                in_element = (x_array >= x_left) & (x_array <= x_right)

                if np.any(in_element):
                    x_elem = x_array[in_element]

                    if left_idx == node_index:
                        # Decreasing from 1 to 0
                        result[in_element] = (x_right - x_elem) / element_size
                    elif right_idx == node_index:
                        # Increasing from 0 to 1
                        result[in_element] = (x_elem - x_left) / element_size

            return result.item() if is_scalar else result

        super().__init__(
            space,
            evaluate_callable=basis_callable,
            name=f'φ_{node_index}',
            support=support
        )


class LazyFEMBasisProvider:
    """
    Lazy provider for FEM basis functions.

    Creates basis functions on demand and caches them.
    """

    def __init__(self, space, nodes: np.ndarray, element_size: float):
        """
        Initialize the basis provider.

        Args:
            space: L2Space for the basis functions
            nodes: Array of mesh nodes
            element_size: Mesh element size h
        """
        self.space = space
        self.nodes = nodes
        self.element_size = element_size
        self._cache = {}

    def get_basis_function(self, node_index: int) -> FEMBasisFunction:
        """
        Get basis function for given node index.

        Args:
            node_index: Index of the node

        Returns:
            FEMBasisFunction for that node
        """
        if node_index not in self._cache:
            if not (0 <= node_index < len(self.nodes)):
                raise ValueError(
                    f"Node index {node_index} out of range [0, {len(self.nodes)-1}]"
                )
            self._cache[node_index] = FEMBasisFunction(
                self.space, node_index, self.nodes, self.element_size
            )
        return self._cache[node_index]

    def __getitem__(self, node_index: int) -> FEMBasisFunction:
        """Allow indexing syntax: provider[i]."""
        return self.get_basis_function(node_index)

    def get_interior_basis_functions(self) -> list:
        """Get all interior (non-boundary) basis functions."""
        return [self.get_basis_function(i) for i in range(1, len(self.nodes) - 1)]


class FEMSolverBase(ABC):
    """Abstract base class for FEM solvers."""

    def __init__(self, interval: Tuple[float, float], dof: int,
                 boundary_conditions: str):
        """
        Initialize FEM solver.

        Args:
            interval: Domain interval [a, b]
            dof: Number of elements in mesh
            boundary_conditions: Type of BCs ('dirichlet', 'neumann', 'periodic')
        """
        self.interval = interval
        self.dof = dof
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

    def __init__(self, interval: Tuple[float, float], dof: int,
                 boundary_conditions: str):
        if not DOLFINX_AVAILABLE:
            raise ImportError(
                "DOLFINx is not available. Install with: "
                "conda install -c conda-forge fenics-dolfinx"
            )

        super().__init__(interval, dof, boundary_conditions)
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
            nx=self.dof,
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
    """Native Python FEM solver - only supports Dirichlet BCs."""

    def __init__(self, interval: Union[Tuple[float, float], IntervalDomain],
                 dof: int, boundary_conditions: str):
        if boundary_conditions != 'dirichlet':
            raise ValueError(
                "NativeFEMSolver only supports 'dirichlet' boundary conditions"
            )

        # Convert interval to IntervalDomain if needed
        if isinstance(interval, tuple):
            self.domain = IntervalDomain(interval[0], interval[1])
            interval_tuple = interval
        else:
            self.domain = interval
            interval_tuple = (interval.a, interval.b)

        super().__init__(interval_tuple, dof, boundary_conditions)
        self._nodes = None
        self._elements = None
        self._boundary_nodes = None
        self._dof = dof
        self._basis_provider = None  # Lazy basis function provider

    def setup(self):
        """Set up native FEM mesh and structures."""
        # Create uniform 1D mesh
        self._nodes = np.linspace(self.domain.a, self.domain.b,
                                  self._dof + 2)

        self._h = self._nodes[1] - self._nodes[0]  # Element size

        # Create elements (pairs of consecutive nodes)
        self._elements = np.column_stack([
            np.arange(self._dof),
            np.arange(1, self._dof + 1)
        ])

        # Identify boundary nodes
        n_nodes = len(self._nodes)
        self._boundary_nodes = {
            'left': 0,
            'right': n_nodes - 1  # Last node index
        }

        # Create lazy basis function provider
        # We need an L2Space for the basis functions
        from .l2_space import L2Space  # Import here to avoid circular imports
        # For FEM, we use as many basis functions as we have nodes
        basis_space = L2Space(
            len(self._nodes),
            interval=(self.domain.a, self.domain.b)
        )
        self._basis_provider = LazyFEMBasisProvider(
            basis_space, self._nodes, self._h
        )

    def _assemble_stiffness_matrix(self) -> np.ndarray:
        """Assemble global stiffness matrix for all nodes."""
        n_nodes = len(self._nodes)
        K = np.zeros((n_nodes, n_nodes))

        # Assemble element contributions
        for i in range(n_nodes - 1):  # Each element
            # Element stiffness matrix (2x2)
            K_elem = (1.0 / self._h) * np.array([[1, -1], [-1, 1]])

            # Assemble into global matrix
            K[i:i+2, i:i+2] += K_elem

        return K

    def _assemble_load_vector_l2(self, rhs_function: Union[Callable, L2Function]) -> np.ndarray:
        """
        Assemble load vector using L2Function operations.

        This method leverages L2Function multiplication and integration
        with automatic compact support handling.

        Args:
            rhs_function: RHS function (callable or L2Function)

        Returns:
            Load vector
        """
        if self._basis_provider is None:
            raise RuntimeError("Must call setup() first")

        n_nodes = len(self._nodes)
        f_vec = np.zeros(n_nodes)

        # Convert RHS to L2Function if needed
        if not isinstance(rhs_function, L2Function):
            # Create L2Function from callable
            from .l2_space import L2Space
            # Create a space for the RHS function (can be different dimension)
            rhs_space = L2Space(1, interval=(self.domain.a, self.domain.b))
            if callable(rhs_function):
                rhs_l2 = L2Function(rhs_space, evaluate_callable=rhs_function)
            else:
                raise TypeError("rhs_function must be callable or L2Function")
        else:
            rhs_l2 = rhs_function

        # For each node, compute ∫ f(x) φᵢ(x) dx using L2Function operations
        for i in range(n_nodes):
            # Get basis function as L2Function
            phi_i = self._basis_provider.get_basis_function(i)

            # Multiply f * φᵢ using L2Function multiplication
            # This automatically handles compact support intersection
            integrand = rhs_l2 * phi_i

            # Integrate using L2Function integration (leverages compact support)
            f_vec[i] = integrand.integrate(method='simpson')

        return f_vec

    def solve_poisson(self, rhs_function: Union[Callable[[np.ndarray],
                                                     np.ndarray], L2Function]) -> np.ndarray:
        """Solve Poisson equation with Dirichlet BCs using L2Function operations."""
        # Assemble system using L2Function operations
        K = self._assemble_stiffness_matrix()
        f = self._assemble_load_vector_l2(rhs_function)

        return self._solve_dirichlet(K, f)

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

    def get_basis_function(self, node_index: int) -> FEMBasisFunction:
        """
        Get basis function for given node index as an L2Function.

        Args:
            node_index: Index of the node (0 to num_nodes-1)

        Returns:
            FEMBasisFunction (L2Function with compact support)
        """
        if self._basis_provider is None:
            raise RuntimeError("Must call setup() first")
        return self._basis_provider.get_basis_function(node_index)

    def get_interior_basis_functions(self) -> list:
        """Get all interior (non-boundary) basis functions as L2Functions."""
        if self._basis_provider is None:
            raise RuntimeError("Must call setup() first")
        return self._basis_provider.get_interior_basis_functions()

    def visualize_basis_functions_l2(self, indices: Optional[list] = None,
                                   n_plot_points: int = 1000):
        """
        Visualize FEM basis functions using their L2Function plot methods.

        Args:
            indices: Which basis functions to plot (default: first few)
            n_plot_points: Number of points for smooth plotting
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_style("whitegrid")
        except ImportError:
            raise ImportError("matplotlib and seaborn required for plotting")

        if self._basis_provider is None:
            raise RuntimeError("Must call setup() before visualizing")

        if indices is None:
            # Show first few interior basis functions
            interior_funcs = self.get_interior_basis_functions()
            indices = list(range(min(5, len(interior_funcs))))

        plt.figure(figsize=(12, 8))

        for i, idx in enumerate(indices):
            if idx >= len(self._nodes):
                continue

            # Get basis function as L2Function
            phi = self.get_basis_function(idx)

            # Use L2Function's plot method but customize
            ax = phi.plot(n_points=n_plot_points, use_seaborn=True)

            # Highlight compact support
            if phi.has_compact_support:
                support_a, support_b = phi.support
                plt.axvspan(support_a, support_b, alpha=0.1,
                           color=plt.gca().lines[-1].get_color())

        # Mark all nodes
        plt.scatter(self._nodes, np.zeros_like(self._nodes),
                   color='red', s=60, zorder=5, alpha=0.8,
                   label='Mesh nodes')

        plt.xlabel('x')
        plt.ylabel('φᵢ(x)')
        plt.title('FEM Basis Functions (L2Functions with Compact Support)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def visualize_basis_functions(self, indices: Optional[list] = None,
                                 n_plot_points: int = 1000):
        """
        Visualize FEM basis functions using domain's mesh.

        Args:
            indices: Which basis functions to plot (default: first few)
            n_plot_points: Number of points for smooth plotting
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_style("whitegrid")
        except ImportError:
            raise ImportError("matplotlib and seaborn required for plotting")

        if self._nodes is None:
            raise RuntimeError("Must call setup() before visualizing")

        if indices is None:
            # Show first few basis functions
            indices = list(range(min(5, self._dof)))

        # Use domain's uniform mesh for plotting
        x_plot = self.domain.uniform_mesh(n_plot_points)

        plt.figure(figsize=(10, 6))

        for idx in indices:
            if idx >= self._dof:
                continue

            node_idx = idx + 1  # Skip left boundary
            y_plot = np.zeros_like(x_plot)

            # Evaluate basis function at plot points
            for i, x in enumerate(x_plot):
                if not self.domain.contains(x):
                    continue

                # Find which element contains x
                for elem_idx in range(len(self._nodes) - 1):
                    x_left = self._nodes[elem_idx]
                    x_right = self._nodes[elem_idx + 1]

                    if x_left <= x <= x_right:
                        # Check if basis function is active on this element
                        if elem_idx == node_idx - 1:
                            # Right side of basis function (decreasing)
                            y_plot[i] = (x_right - x) / self._h
                        elif elem_idx == node_idx:
                            # Left side of basis function (increasing)
                            y_plot[i] = (x - x_left) / self._h
                        break

            plt.plot(x_plot, y_plot, label=f'φ_{idx}', linewidth=2)

        # Mark nodes
        plt.scatter(self._nodes, np.zeros_like(self._nodes),
                   color='red', s=50, zorder=5, alpha=0.7)

        plt.xlabel('x')
        plt.ylabel('φᵢ(x)')
        plt.title('FEM Basis Functions (Linear Elements)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def get_coordinates(self) -> np.ndarray:
        """Get mesh node coordinates."""
        return self._nodes.copy()

    def interpolate_solution(self, solution_values: np.ndarray,
                           eval_points: np.ndarray) -> np.ndarray:
        """Linear interpolation of solution."""
        return np.interp(eval_points, self._nodes, solution_values)


def create_fem_solver(solver_type: str,
                      interval: Union[Tuple[float, float], IntervalDomain],
                      dof: int, boundary_conditions: str) -> FEMSolverBase:
    """
    Factory function to create FEM solvers.

    Args:
        solver_type: 'dolfinx' or 'native'
        interval: Domain interval (tuple or IntervalDomain)
        dof: Mesh resolution
        boundary_conditions: Boundary condition type

    Returns:
        FEM solver instance
    """
    # Convert IntervalDomain to tuple for DOLFINx solver compatibility
    if isinstance(interval, IntervalDomain):
        interval_tuple = (interval.a, interval.b)
    else:
        interval_tuple = interval

    if solver_type == 'dolfinx':
        return DOLFINxSolver(interval_tuple, dof, boundary_conditions)
    elif solver_type == 'native':
        return NativeFEMSolver(interval, dof, boundary_conditions)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")
