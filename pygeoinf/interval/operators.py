"""
Differential operators on interval domains.

This module provides differential operators for interval domains with various
solution methods:

- LaplacianOperator: -Δ with spectral/finite difference methods
- LaplacianInverseOperator: (-Δ)^(-1) with FEM solvers
- GradientOperator: ∇ with finite difference and automatic differentiation

The operators implement proper function space mappings and support
boundary conditions through the underlying function spaces.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, Literal, List, Callable
import warnings

from pygeoinf.hilbert_space import EuclideanSpace
from .l2_space import L2Space
from .sobolev_space import Sobolev
from .boundary_conditions import BoundaryConditions
from pygeoinf.operators import LinearOperator
from .functions import Function
from .providers import create_laplacian_spectrum_provider
# FEM import only needed for LaplacianInverseOperator
from pygeoinf.interval.fem_solvers import GeneralFEMSolver
from pygeoinf.interval.function_providers import (
    FunctionProvider, IndexedFunctionProvider, NormalModesProvider
)


class DifferentialOperator(LinearOperator, ABC):
    """
    Abstract base class for differential operators on interval domains.

    Provides common functionality for discretization method selection
    and error handling.
    """

    def __init__(self, domain, codomain, boundary_conditions,
                 method='spectral'):
        self.boundary_conditions = boundary_conditions
        self.method = method
        self._validate_method()
        super().__init__(domain, codomain, self._apply)

    def _validate_method(self):
        """Validate the chosen discretization method."""
        valid_methods = ['spectral', 'finite_difference']
        if self.method not in valid_methods:
            raise ValueError(f"Unknown method '{self.method}'. "
                             f"Valid methods: {valid_methods}")

    @abstractmethod
    def _apply(self, f: Function) -> Function:
        """Apply the operator to a function."""
        pass


class GradientOperator(DifferentialOperator):
    """
    The gradient operator (d/dx) on interval domains.

    In 1D, the gradient is simply the first derivative. This operator
    provides multiple methods for computing derivatives:

    - 'finite_difference': Numerical differentiation using finite differences
    - 'automatic': Automatic differentiation using JAX (if available)

    The operator maps between function spaces:
    - H^s → H^(s-1) (reduces regularity by one order)
    """

    def __init__(
        self,
        domain: Union[L2Space, Sobolev],
        /,
        *,
        method: Literal[
            'finite_difference', 'automatic'
        ] = 'finite_difference',
        fd_order: int = 2,
        fd_step: Optional[float] = None,
        boundary_treatment: str = 'one_sided'
    ):
        """
        Initialize the gradient operator.

        Args:
            domain: Function space (L2Space or SobolevSpace)
            method: Differentiation method
                ('finite_difference', 'automatic')
            fd_order: Order of finite difference stencil (2, 4, 6)
                for FD method
            fd_step: Step size for finite differences (auto-computed if None)
            boundary_treatment: How to handle boundaries for FD
                ('one_sided', 'extrapolate')
        """
        self._domain = domain
        self._fd_order = fd_order
        self._fd_step = fd_step
        self._boundary_treatment = boundary_treatment

        # Determine codomain based on method and domain
        if isinstance(domain, Sobolev):
            # Ensure domain has sufficient regularity for gradient
            if domain.order < 1.0:
                warnings.warn(
                    f"Gradient requires domain with Sobolev order ≥ 1, "
                    f"got {domain.order}. Consider using H¹ space."
                )

            # Create codomain with one order less regularity
            target_order = domain.order - 1.0
            if hasattr(domain, '_basis_type'):
                basis_type = domain._basis_type
            else:
                basis_type = 'fourier'

            codomain = Sobolev(
                domain.dim, domain.function_domain,
                target_order,
                basis_type=basis_type
            )
        else:
            # For L2 domain (H⁰), gradient maps to H⁻¹
            codomain = Sobolev(
                domain.dim, domain.function_domain,
                -1.0,
                basis_type='fourier'
            )

        # No boundary conditions needed for gradient (unlike Laplacian)
        super().__init__(domain, codomain, None, method)

        # Initialize method-specific components
        self._setup_method()

    def _validate_method(self):
        """Override to validate gradient-specific methods."""
        valid_methods = ['finite_difference', 'automatic']
        if self.method not in valid_methods:
            raise ValueError(f"Unknown method '{self.method}'. "
                             f"Valid methods: {valid_methods}")

    def _setup_method(self):
        """Setup method-specific components."""
        if self.method == 'finite_difference':
            self._setup_finite_difference()
        elif self.method == 'automatic':
            self._setup_automatic_differentiation()

    def _setup_finite_difference(self):
        """Setup finite difference method."""
        # Determine step size if not provided
        if self._fd_step is None:
            a, b = self._domain.function_domain.a, \
                   self._domain.function_domain.b
            # Use a fraction of the domain size
            self._fd_step = (b - a) / 1000

        print(f"GradientOperator (finite difference, order {self._fd_order}) "
              f"initialized with step size {self._fd_step:.2e}")

    def _setup_automatic_differentiation(self):
        """Setup automatic differentiation method."""
        try:
            import jax  # noqa: F401
            import jax.numpy as jnp  # noqa: F401
            from jax import grad  # noqa: F401
            self._jax = jax
            self._jnp = jnp
            self._grad = grad
            print("GradientOperator (automatic differentiation) "
                  "initialized with JAX")
        except ImportError:
            raise ImportError(
                "JAX is required for automatic differentiation. "
                "Install with: pip install jax jaxlib"
            )

    def _apply(self, f: Function) -> Function:
        """Apply the gradient operator to function f."""
        if self.method == 'finite_difference':
            return self._apply_finite_difference(f)
        elif self.method == 'automatic':
            return self._apply_automatic(f)
        else:
            raise ValueError(f"Unknown method '{self.method}'")

    def _apply_finite_difference(self, f: Function) -> Function:
        """Apply gradient using finite difference method."""

        def gradient_func(x):
            """Compute gradient at points x using finite differences."""
            # Handle scalar input
            if np.isscalar(x):
                x = np.array([x])
                scalar_input = True
            else:
                scalar_input = False

            df_dx = np.zeros_like(x)
            h = self._fd_step

            for i, xi in enumerate(x):
                # Check boundaries
                a, b = (self._domain.function_domain.a,
                        self._domain.function_domain.b)

                if self._boundary_treatment == 'one_sided':
                    # Use one-sided differences at boundaries
                    if xi <= a + h:
                        # Forward difference at left boundary
                        if self._fd_order == 2:
                            df_dx[i] = (
                                -3*f(xi) + 4*f(xi + h) - f(xi + 2*h)
                            ) / (2*h)
                        else:  # Higher order forward
                            df_dx[i] = (
                                -f(xi + 2*h) + 8*f(xi + h) - 8*f(xi) + f(xi)
                            ) / (12*h)
                    elif xi >= b - h:
                        # Backward difference at right boundary
                        if self._fd_order == 2:
                            df_dx[i] = (
                                3*f(xi) - 4*f(xi - h) + f(xi - 2*h)
                            ) / (2*h)
                        else:  # Higher order backward
                            df_dx[i] = (
                                f(xi - 2*h) - 8*f(xi - h) + 8*f(xi) - f(xi)
                            ) / (12*h)
                    else:
                        # Central difference in interior
                        if self._fd_order == 2:
                            df_dx[i] = (f(xi + h) - f(xi - h)) / (2*h)
                        elif self._fd_order == 4:
                            df_dx[i] = (
                                -f(xi + 2*h) + 8*f(xi + h) - 8*f(xi - h) +
                                f(xi - 2*h)
                            ) / (12*h)
                        else:
                            # Default to 2nd order
                            df_dx[i] = (f(xi + h) - f(xi - h)) / (2*h)
                else:
                    # Simple central difference everywhere (boundary issues)
                    if self._fd_order == 2:
                        df_dx[i] = (f(xi + h) - f(xi - h)) / (2*h)
                    elif self._fd_order == 4:
                        df_dx[i] = (
                            -f(xi + 2*h) + 8*f(xi + h) - 8*f(xi - h) +
                            f(xi - 2*h)
                        ) / (12*h)
                    else:
                        df_dx[i] = (f(xi + h) - f(xi - h)) / (2*h)

            return df_dx[0] if scalar_input else df_dx

        return Function(
            self.codomain,
            evaluate_callable=gradient_func,
            name=f"∇({f.name})" if hasattr(f, 'name') else "∇f"
        )

    def _apply_automatic(self, f: Function) -> Function:
        """Apply gradient using automatic differentiation."""

        def gradient_func(x):
            """Compute gradient using JAX automatic differentiation."""
            if not hasattr(f, 'evaluate_callable'):
                raise ValueError(
                    "Function must have evaluate_callable for "
                    "automatic differentiation"
                )

            # Create a JAX-native version of the function
            # This requires the user's function to be JAX-compatible
            original_func = f.evaluate_callable

            def jax_compatible_func(xi):
                """JAX-compatible wrapper that works with JAX arrays."""
                try:
                    # Try to call the function with JAX arrays directly
                    return original_func(xi)
                except Exception:
                    # If that fails, the function may not be JAX-compatible
                    # Convert to numpy, compute, then back to JAX
                    if hasattr(xi, 'shape'):  # It's an array-like
                        xi_np = np.asarray(xi)
                        result_np = original_func(xi_np)
                        return self._jnp.asarray(result_np)
                    else:  # It's a scalar
                        xi_float = float(xi)
                        result_float = original_func(xi_float)
                        return self._jnp.asarray(result_float)

            # Get the gradient function
            grad_jax_func = self._grad(jax_compatible_func)

            # Handle scalar vs array input
            if np.isscalar(x):
                x_jax = self._jnp.array(float(x))
                result_jax = grad_jax_func(x_jax)
                return float(result_jax)
            else:
                # For array input, compute gradient at each point individually
                # This avoids vectorization issues with JAX tracing
                results = []
                for xi in x:
                    x_jax = self._jnp.array(float(xi))
                    result_jax = grad_jax_func(x_jax)
                    results.append(float(result_jax))
                return np.array(results)

        return Function(
            self.codomain,
            evaluate_callable=gradient_func,
            name=f"∇({f.name})" if hasattr(f, 'name') else "∇f"
        )

    @property
    def fd_step(self) -> Optional[float]:
        """Get the finite difference step size."""
        return self._fd_step

    @property
    def fd_order(self) -> int:
        """Get the finite difference order."""
        return self._fd_order


class LaplacianOperator(DifferentialOperator):
    """
    The negative Laplacian operator (-Δ) on interval domains.

    This operator computes -d²u/dx² with specified boundary conditions.
    Multiple discretization methods are available:

    - 'spectral': Uses analytical eigendecomposition with
                  Fourier/sine/cosine basis
    - 'finite_difference': Centered finite differences

    The operator maps between function spaces based on the method:
    - Spectral: L² → L² (preserves regularity via eigenfunction expansion)
    - FD: H^s → H^(s-2) (requires domain with sufficient regularity)
    """

    def __init__(
        self,
        domain: Union[L2Space, Sobolev],
        boundary_conditions: BoundaryConditions,
        /,
        *,
        method: Literal['spectral', 'finite_difference'] = 'spectral',
        dofs: Optional[int] = None,
        fd_order: int = 2
    ):
        """
        Initialize the negative Laplacian operator.

        Args:
            domain: Function space (L2Space or SobolevSpace)
            boundary_conditions: Boundary conditions for the operator
            method: Discretization method ('spectral', 'finite_difference')
            dofs: Number of degrees of freedom for FD methods
                  (default: domain.dim)
            fd_order: Order of finite difference stencil (2, 4, 6)
        """
        self._domain = domain
        self._dofs = dofs or domain.dim
        self._fd_order = fd_order

        # Determine codomain based on method
        if method == 'spectral':
            # Spectral methods preserve the space (with proper basis)
            codomain = domain
        else:
            # FD/FE methods: Laplacian maps H^s → H^(s-2)
            # For Laplacian, domain should be at least H^2
            if isinstance(domain, Sobolev):
                # Ensure domain has sufficient regularity for Laplacian
                if domain.order < 2.0:
                    warnings.warn(
                        f"Laplacian requires domain with Sobolev order ≥ 2, "
                        f"got {domain.order}. Consider using H² space."
                    )

                # Create codomain with two orders less regularity
                target_order = domain.order - 2.0
                # Get basis type if available, default to fourier
                if hasattr(domain, '_basis_type'):
                    basis_type = domain._basis_type
                else:
                    basis_type = 'fourier'
                codomain = Sobolev(
                    domain.dim, domain.function_domain,
                    target_order, domain._inner_product_type,
                    basis_type=basis_type,
                    boundary_conditions=boundary_conditions
                )
            else:
                # For L2 domain (H⁰), Laplacian maps to H⁻²
                # Create H⁻² space
                codomain = Sobolev(
                    domain.dim, domain.function_domain,
                    -2.0, 'spectral',  # Default to spectral for negative order
                    basis_type='fourier',
                    boundary_conditions=boundary_conditions
                )

        super().__init__(domain, codomain, boundary_conditions, method)

        # Initialize method-specific components
        self._setup_method()

    def _setup_method(self):
        """Setup method-specific components."""
        if self.method == 'spectral':
            self._setup_spectral()
        elif self.method == 'finite_difference':
            self._setup_finite_difference()

    def _setup_spectral(self):
        """Setup spectral method using eigenfunction expansion."""
        # Create spectrum provider for eigenvalues/eigenfunctions
        self._spectrum_provider = create_laplacian_spectrum_provider(
            self._domain, self.boundary_conditions, inverse=False
        )

        # For spectral methods, we use the analytical eigendecomposition
        print(f"LaplacianOperator (spectral) initialized with "
              f"{self.boundary_conditions} boundary conditions")

    def _setup_finite_difference(self):
        """Setup finite difference discretization."""
        # Create finite difference grid
        a, b = self._domain.function_domain.a, self._domain.function_domain.b
        self._x_grid = np.linspace(a, b, self._dofs)
        self._dx = (b - a) / (self._dofs - 1)

        # Create finite difference matrix
        self._fd_matrix = self._create_fd_matrix()

        print(f"LaplacianOperator (finite difference, order {self._fd_order}) "
              f"initialized with {self._dofs} grid points")

    def _create_fd_matrix(self):
        """Create finite difference matrix for the negative Laplacian."""
        n = self._dofs
        dx2 = self._dx**2

        if self._fd_order == 2:
            # Second-order centered differences: [-1, 2, -1] / dx²
            matrix = np.zeros((n, n))

            # Interior points
            for i in range(1, n-1):
                matrix[i, i-1] = -1.0 / dx2
                matrix[i, i] = 2.0 / dx2
                matrix[i, i+1] = -1.0 / dx2

            # Apply boundary conditions
            self._apply_fd_boundary_conditions(matrix)

        elif self._fd_order == 4:
            # Fourth-order centered differences: [1, -8, 12, -8, 1] / 12dx²
            matrix = np.zeros((n, n))

            # Interior points (need at least 2 points from boundary)
            for i in range(2, n-2):
                matrix[i, i-2] = 1.0 / (12 * dx2)
                matrix[i, i-1] = -8.0 / (12 * dx2)
                matrix[i, i] = 12.0 / (12 * dx2)
                matrix[i, i+1] = -8.0 / (12 * dx2)
                matrix[i, i+2] = 1.0 / (12 * dx2)

            # Near-boundary points use second-order
            for i in [1, n-2]:
                matrix[i, i-1] = -1.0 / dx2
                matrix[i, i] = 2.0 / dx2
                matrix[i, i+1] = -1.0 / dx2

            self._apply_fd_boundary_conditions(matrix)

        else:
            raise ValueError(
                f"Finite difference order {self._fd_order} not implemented"
            )

        return matrix

    def _apply_fd_boundary_conditions(self, matrix):
        """Apply boundary conditions to finite difference matrix."""
        if self.boundary_conditions.type == 'dirichlet':
            # Dirichlet: u = 0 at boundaries
            matrix[0, :] = 0
            matrix[0, 0] = 1
            matrix[-1, :] = 0
            matrix[-1, -1] = 1

        elif self.boundary_conditions.type == 'neumann':
            # Neumann: du/dx = 0 at boundaries (second-order)
            # Left boundary: u_0 = u_1
            matrix[0, :] = 0
            matrix[0, 0] = -1
            matrix[0, 1] = 1

            # Right boundary: u_n = u_{n-1}
            matrix[-1, :] = 0
            matrix[-1, -1] = -1
            matrix[-1, -2] = 1

        elif self.boundary_conditions.type == 'periodic':
            # Periodic: handle wraparound
            # This is more complex and requires special treatment
            warnings.warn(
                "Periodic boundary conditions for FD not fully implemented"
            )

        else:
            raise ValueError(
                f"Boundary condition {self.boundary_conditions.type} "
                f"not supported for finite differences"
            )

    def _apply(self, f: Function) -> Function:
        """Apply the negative Laplacian operator to function f."""
        if self.method == 'spectral':
            return self._apply_spectral(f)
        elif self.method == 'finite_difference':
            return self._apply_finite_difference(f)
        else:
            raise ValueError(f"Unknown method '{self.method}'")

    def _apply_spectral(self, f: Function) -> Function:
        """Apply Laplacian using spectral method (eigenfunction expansion)."""
        # Project f onto eigenfunctions and apply eigenvalues
        if not hasattr(self._domain, 'to_components'):
            raise ValueError(
                "Spectral method requires domain with "
                "coefficient representation"
            )

        # Get coefficients of f in the eigenfunction basis
        coeffs = self._domain.to_components(f)

        # Apply eigenvalues (multiply by λᵢ for -Δ)
        laplacian_coeffs = np.zeros_like(coeffs)
        for i in range(len(coeffs)):
            eigenvalue = self._spectrum_provider.get_eigenvalue(i)
            laplacian_coeffs[i] = eigenvalue * coeffs[i]

        # Reconstruct function in the codomain
        return self.codomain.from_components(laplacian_coeffs)

    def _apply_finite_difference(self, f: Function) -> Function:
        """Apply Laplacian using finite difference method."""
        # Evaluate function on grid
        f_values = f(self._x_grid)

        # Apply finite difference matrix
        laplacian_values = self._fd_matrix @ f_values

        # Create result function by interpolation
        def laplacian_func(x):
            return np.interp(x, self._x_grid, laplacian_values)

        return Function(self.codomain, evaluate_callable=laplacian_func)


class LaplacianInverseOperator(LinearOperator):
    """
    Inverse Laplacian operator that acts as a covariance operator.

    This operator solves -Δu = f with homogeneous boundary conditions,
    providing a self-adjoint operator suitable for Gaussian measures.

    Uses native Python FEM implementation (no external dependencies).
    """

    def __init__(
        self,
        domain: L2Space,
        boundary_conditions: BoundaryConditions,
        /,
        *,
        alpha: float = 1.0,
        dofs: int = 100,
        fem_type: str = "hat"
    ):
        """
        Initialize the Laplacian inverse operator.

        The operator maps from L² space to Sobolev space:
        (-Δ)⁻¹: L² → H^s with specified boundary conditions

        Args:
            domain: The L2 space (domain of the operator)
            boundary_conditions: Boundary conditions for the operator
            alpha: Scaling factor for the operator (default: 1.0)
            dofs: Number of degrees of freedom (default: 100)
            fem_type: FEM implementation to use (deprecated):
                - "hat": Uses hat function basis (default)
                - "general": Uses domain's basis functions
                Note: Both options now use GeneralFEMSolver internally.
        """
        # Check that domain is an L2 space
        if not isinstance(domain, L2Space):
            raise TypeError(
                f"domain must be an L2 space, got {type(domain)}"
            )

        self._domain = domain
        self._dofs = dofs
        self._boundary_conditions = boundary_conditions
        self._fem_type = fem_type

        # Validate fem_type
        if fem_type not in ["hat", "general"]:
            raise ValueError(
                f"fem_type must be 'hat' or 'general', got '{fem_type}'"
            )

        # Create the Sobolev space as codomain
        # IN TESTING NOW!!!!
        """ self._codomain = Sobolev(
            domain.dim,
            domain.function_domain,
            2.0,
            'spectral',
            basis_type='fourier',  # Use Fourier basis for spectral approach
            boundary_conditions=boundary_conditions
        ) """
        self._codomain = domain  # FOR TESTING ONLY

        # Get function domain from domain (L2Space and SobolevSpace)
        self._function_domain = domain.function_domain

        # Store scaling factor
        if not isinstance(alpha, (int, float)):
            raise TypeError(f"alpha must be a number, got {type(alpha)}")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self._alpha = alpha

        # Initialize spectrum provider for lazy eigenvalue computation
        self._spectrum_provider = create_laplacian_spectrum_provider(
            domain, self._boundary_conditions, inverse=True
        )

        # Create and setup FEM solver
        if self._fem_type == "hat":
            # Create L2Space with hat functions for optimized performance
            fem_l2_space = L2Space(
                dofs,
                self._function_domain,
                basis_type='hat_homogeneous'
            )
            self._fem_solver = GeneralFEMSolver(
                fem_l2_space,
                self._boundary_conditions
            )
            solver_description = "hat function FEM solver"
        elif self._fem_type == "general":
            # Use general implementation with domain's basis functions
            self._fem_solver = GeneralFEMSolver(
                domain,  # Pass the L2Space directly
                self._boundary_conditions
            )
            solver_description = "general FEM solver"

        self._fem_solver.setup()

        print(
            f"LaplacianInverseOperator initialized with {solver_description}, "
            f"{self._boundary_conditions} BCs"
        )

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
        f = self._alpha * f  # Scale input by alpha

        # Solve PDE using GeneralFEMSolver
        # Both hat and general types now use GeneralFEMSolver
        solution_coeffs = self._fem_solver.solve_poisson(f)
        return self._fem_solver.solution_to_function(solution_coeffs)

    @property
    def boundary_conditions(self) -> BoundaryConditions:
        """Get the boundary conditions object."""
        return self._boundary_conditions

    @property
    def fem_type(self) -> str:
        """Get the FEM implementation type."""
        return self._fem_type

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
        return self._alpha * self._spectrum_provider.get_eigenvalue(index)

    def get_eigenfunction(self, index: int) -> Function:
        """
        Get eigenfunction of the inverse Laplacian operator.

        Args:
            index: Index of the eigenfunction (0 to dim-1)

        Returns:
            Function: Eigenfunction φₖ of (-Δ)^(-1)
        """
        return self._spectrum_provider.get_basis_function(index)

    from typing import Optional

    def get_all_eigenvalues(self, n: Optional[int] = None) -> np.ndarray:
        """
        Get all eigenvalues of the inverse Laplacian operator.
        Args:
            n: Number of eigenvalues to compute (default: space.dim)
        Returns:
            np.ndarray: Array of all eigenvalues
        """
        if n is None:
            n = self._domain.dim
        return self._alpha * self._spectrum_provider.get_all_eigenvalues(n)

    def get_solver_info(self) -> dict:
        """Get information about the current solver."""
        return {
            'fem_type': self._fem_type,
            'boundary_conditions': self._boundary_conditions,
            'dofs': self._dofs,
            'fem_coordinates': self._fem_solver.get_coordinates(),
            'spectrum_available': True,
            'eigenvalue_range': (
                self.get_eigenvalue(0) if self._domain.dim > 0 else None,
                (self.get_eigenvalue(self._domain.dim - 1)
                 if self._domain.dim > 0 else None)
            )
        }


class SOLAOperator(LinearOperator):
    """
    SOLA operator that projects functions onto a basis.

    This operator takes a function from an L2 space and computes inner products
    against a set of kernel functions, resulting in a vector in the specified
    Euclidean space.

    The operator maps: L2Space -> EuclideanSpace

    The kernel functions can be provided in three ways:
    1. Via a FunctionProvider (original functionality)
    2. Via a list of Function objects
    3. Via a list of callables (automatically converted to Function objects)

    Examples:
        # Using a function provider
        >>> provider = NormalModesProvider(l2_space)
        >>> sola_op = SOLAOperator(l2_space, euclidean_space,
        ...                        function_provider=provider)

        # Using direct callables
        >>> kernels = [lambda x: np.sin(x), lambda x: np.cos(x)]
        >>> sola_op = SOLAOperator(l2_space, euclidean_space,
        ...                        functions=kernels)

        # Using Function objects
        >>> func1 = Function(l2_space, evaluate_callable=lambda x: x**2)
        >>> func2 = Function(l2_space, evaluate_callable=lambda x: x**3)
        >>> sola_op = SOLAOperator(l2_space, euclidean_space,
        ...                        functions=[func1, func2])
    """

    def __init__(self, domain, codomain: EuclideanSpace,
                 function_provider: Optional[FunctionProvider] = None,
                 functions: Optional[List[Union[Function, Callable]]] = None,
                 random_state: Optional[int] = None,
                 cache_functions: bool = False,
                 integration_method: str = 'simpson',
                 n_points: int = 1000):
        """
        Initialize the SOLA operator.

        Args:
            domain: L2Space instance (the function space)
            codomain: EuclideanSpace instance that defines the output dimension
            function_provider: Provider for generating kernels.
                              If None and functions is None, creates a default
                              NormalModesProvider.
            functions: List of Function instances or callables to use as
                      kernels. If provided, takes precedence over
                      function_provider. Callables will be converted to
                      Function instances.
            random_state: Random seed for reproducible function generation
            cache_functions: If True, cache kernels after first
                           access for faster repeated operations
            integration_method: Method for numerical integration
                ('simpson', 'trapezoid')
            n_points: Number of points for numerical integration
        """
        self.N_d = codomain.dim
        self.cache_functions = cache_functions
        self._function_cache = {} if cache_functions else None
        self._integration_method = integration_method
        self._n_points = n_points

        # Check for mutually exclusive arguments
        if function_provider is not None and functions is not None:
            raise ValueError(
                "Cannot specify both function_provider and functions. "
                "Please provide only one."
            )

        # Handle direct function list
        if functions is not None:
            if len(functions) != self.N_d:
                raise ValueError(
                    f"Number of functions ({len(functions)}) must match "
                    f"codomain dimension ({self.N_d})"
                )

            # Convert callables to Function instances if needed
            self._functions = []
            for i, func in enumerate(functions):
                if callable(func) and not isinstance(func, Function):
                    # Convert callable to Function instance
                    self._functions.append(
                        Function(domain, evaluate_callable=func)
                    )
                elif isinstance(func, Function):
                    self._functions.append(func)
                else:
                    raise TypeError(
                        f"Function at index {i} must be either a Function "
                        f"instance or a callable, got {type(func)}"
                    )

            self.function_provider = None
            self._use_direct_functions = True
        else:
            # Create or use provided FunctionProvider
            if function_provider is None:
                self.function_provider = NormalModesProvider(
                    domain,
                    random_state=random_state,
                    n_modes_range=(2, 5),
                    coeff_range=(-1.0, 1.0),
                    freq_range=(0.5, 5.0),
                    gaussian_width_percent_range=(20.0, 40.0)
                )
            else:
                # Store the function provider for lazy evaluation
                self.function_provider = function_provider

            self._functions = None
            self._use_direct_functions = False

        # Define the mapping function
        def mapping(func):
            """Project function onto the kernel basis."""
            return self._project_function(func)

        # Define the adjoint mapping
        def adjoint_mapping(data):
            """Reconstruct function from data."""
            return self._reconstruct_function(data)

        super().__init__(
            domain,
            codomain,
            mapping,
            adjoint_mapping=adjoint_mapping
        )

    def get_kernel(self, index: int):
        """
        Lazily get the i-th kernel with optional caching.

        Args:
            index: Index of the kernel to retrieve

        Returns:
            Function: The i-th kernel
        """
        # If using direct functions, simply return the indexed function
        if self._use_direct_functions:
            if (self.cache_functions and self._function_cache is not None
                    and index in self._function_cache):
                return self._function_cache[index]

            assert self._functions is not None  # For type checker
            function = self._functions[index]

            # Cache the function if caching is enabled
            if self.cache_functions and self._function_cache is not None:
                self._function_cache[index] = function

            return function

        # Otherwise use function provider logic
        # Check cache first if caching is enabled
        if (self.cache_functions and self._function_cache is not None
                and index in self._function_cache):
            return self._function_cache[index]

        # Generate the function using the appropriate provider method
        assert self.function_provider is not None  # For type checker
        if isinstance(self.function_provider, IndexedFunctionProvider):
            # Use the indexed access for providers that support it
            function = self.function_provider.get_function_by_index(index)
        elif hasattr(self.function_provider, 'sample_function'):
            # Fall back to sampling for providers that support it
            function = getattr(self.function_provider, 'sample_function')()
        else:
            raise ValueError(
                f"Function provider {type(self.function_provider)} does not "
                "support either indexed access or sampling"
            )

        # Cache the function if caching is enabled
        if self.cache_functions and self._function_cache is not None:
            self._function_cache[index] = function

        return function

    def _project_function(self, func):
        """
        Project a function onto the kernels using lazy evaluation.

        Args:
            func: Function from the domain space

        Returns:
            numpy.ndarray: Vector of data in R^{N_d}
        """
        data = np.zeros(self.N_d)

        for i in range(self.N_d):
            # Lazily get the i-th kernel
            proj_func = self.get_kernel(i)
            # Compute inner product with the i-th kernel
            data[i] = self.domain.inner_product(
                func,
                proj_func,
                method=self._integration_method,
                n_points=self._n_points
            )

        return data

    def _reconstruct_function(self, data):
        """
        Reconstruct a function from data using lazy evaluation.

        Args:
            data: numpy.ndarray of data in R^{N_d}

        Returns:
            Function: Reconstructed function in the domain space
        """
        # Start with zero function
        reconstructed = self.domain.zero

        # Add weighted kernels
        for i, coeff in enumerate(data):
            if abs(coeff) > 1e-14:  # Avoid numerical noise
                # Lazily get the i-th kernel
                proj_func = self.get_kernel(i)
                reconstructed += coeff * proj_func

        return reconstructed

    def get_kernels(self):
        """
        Get the list of kernels used by this operator.
        Note: This materializes all functions and may be expensive.

        Returns:
            list: List of kernels used for projection
        """
        return [self.get_kernel(i) for i in range(self.N_d)]

    def evaluate_kernel(self, x):
        """
        Evaluate all kernel functions at given points using lazy
        evaluation.

        Args:
            x: numpy.ndarray of evaluation points

        Returns:
            numpy.ndarray: Matrix of shape (N_d, len(x)) with function values
        """
        values = np.zeros((self.N_d, len(x)))

        for i in range(self.N_d):
            proj_func = self.get_kernel(i)
            values[i, :] = proj_func.evaluate(x)

        return values

    def compute_gram_matrix(self):
        """
        Compute the Gram matrix of the kernels using lazy
        evaluation.

        Returns:
            numpy.ndarray: N_d x N_d matrix of inner products between
                          kernels
        """
        gram = np.zeros((self.N_d, self.N_d))

        for i in range(self.N_d):
            proj_func_i = self.get_kernel(i)
            for j in range(self.N_d):
                proj_func_j = self.get_kernel(j)
                gram[i, j] = self.domain.inner_product(
                    proj_func_i,
                    proj_func_j,
                    method=self._integration_method,
                    n_points=self._n_points
                )

        return gram

    def clear_cache(self):
        """Clear the function cache if caching is enabled."""
        if self.cache_functions and self._function_cache is not None:
            self._function_cache.clear()

    def get_cache_info(self):
        """
        Get information about the function cache.

        Returns:
            dict: Cache statistics including size and hit rate
        """
        if not self.cache_functions:
            return {"caching_enabled": False}

        assert self._function_cache is not None  # For type checker
        return {
            "caching_enabled": True,
            "cached_functions": len(self._function_cache),
            "total_functions": self.N_d,
            "cache_coverage": len(self._function_cache) / self.N_d
        }

    def __str__(self):
        """String representation of the SOLA operator."""
        if self._use_direct_functions:
            return (f"SOLAOperator: {self.domain} -> {self.codomain}\n"
                    f"  Uses {self.N_d} direct function kernels\n"
                    f"  Domain dimension: {self.domain.dim}\n"
                    f"  Codomain dimension: {self.codomain.dim}")
        else:
            provider_type = type(self.function_provider).__name__
            return (f"SOLAOperator: {self.domain} -> {self.codomain}\n"
                    f"  Uses {self.N_d} kernels "
                    f"from {provider_type}\n"
                    f"  Domain dimension: {self.domain.dim}\n"
                    f"  Codomain dimension: {self.codomain.dim}")
