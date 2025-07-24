"""
L² spaces on interval domains.

This module provides L² Hilbert spaces on intervals as the foundation
for more specialized function spaces like Sobolev spaces.
"""

import numpy as np
import math

from pygeoinf.hilbert_space import HilbertSpace
from pygeoinf.hilbert_space import LinearForm
from pygeoinf.interval.l2_functions import L2Function
from pygeoinf.interval.interval_domain import (
    BoundaryConditions, IntervalDomain
)


class LazyL2BasisProvider:
    """
    Lazy provider for L2 basis functions.

    Creates basis functions on demand and caches them to avoid
    memory issues with high-dimensional spaces.
    """

    def __init__(self, space, basis_type: str):
        """
        Initialize the lazy basis provider.

        Args:
            space: L2Space that owns this provider
            basis_type: Type of basis functions
                ('fourier', 'hat', 'hat_homogeneous')
            dim: Dimension of the space
        """
        self.space = space
        self.basis_type = basis_type
        self._cache = {}

    def get_basis_function(self, index: int):
        """
        Get basis function for given index.

        Args:
            index: Index of the basis function (0 to dim-1)

        Returns:
            L2Function for that index
        """
        if not (0 <= index < self.space.dim):
            raise IndexError(
                f"Basis index {index} out of range [0, {self.space.dim})"
            )

        if index not in self._cache:
            self._cache[index] = self._create_basis_function(index)
        return self._cache[index]

    def _create_basis_function(self, index: int):
        """Create a single basis function for the given index."""
        if self.basis_type == 'fourier':
            return self._create_fourier_basis_function(index)
        elif self.basis_type == 'hat':
            return self._create_full_hat_basis_function(index)
        elif self.basis_type == 'hat_homogeneous':
            return self._create_homogeneous_hat_basis_function(index)
        else:
            raise ValueError(f"Unsupported basis type: {self.basis_type}")

    def _create_fourier_basis_function(self, index: int):
        """Create a single Fourier basis function."""
        bc = self.space.boundary_conditions
        domain = self.space.domain
        length = domain.b - domain.a

        if bc is None or bc.type == 'periodic':
            # Periodic boundary conditions: full Fourier basis
            normalization_factor = math.sqrt(2 / length)

            if index == 0:
                # Constant term
                def constant_func(x):
                    return (normalization_factor *
                            np.ones_like(x) / np.sqrt(2))
                return L2Function(
                    self.space,
                    evaluate_callable=constant_func,
                    name='constant'
                )
            else:
                # For index > 0, we alternate between cosine and sine
                k = (index + 1) // 2  # Frequency index
                freq = 2 * k * math.pi / length

                if index % 2 == 1:  # Odd indices are cosine
                    def cosine_func(x):
                        return (normalization_factor *
                                np.cos(freq * (x - domain.a)))
                    return L2Function(
                        self.space,
                        evaluate_callable=cosine_func,
                        name=f'cos_{k}'
                    )
                else:  # Even indices > 0 are sine
                    def sine_func(x):
                        return (normalization_factor *
                                np.sin(freq * (x - domain.a)))
                    return L2Function(
                        self.space,
                        evaluate_callable=sine_func,
                        name=f'sin_{k}'
                    )
        else:
            raise NotImplementedError(
                f"Boundary condition type '{bc.type}' "
                "not implemented yet"
            )

    def _create_homogeneous_hat_basis_function(self, index: int):
        """Create a single homogeneous hat (piecewise linear) basis function.

        These are interior hat functions that vanish at the boundaries,
        suitable for homogeneous Dirichlet boundary conditions.
        """
        domain = self.space.domain

        # Create uniform mesh for hat functions
        # For dim basis functions, we need dim+2 nodes (including boundaries)
        nodes = np.linspace(domain.a, domain.b, self.space.dim + 2)
        element_size = nodes[1] - nodes[0]

        # Hat function φᵢ has support on [nodes[i], nodes[i+2]]
        # and has value 1 at nodes[i+1]
        node_index = index + 1  # Interior node index
        support = (nodes[index], nodes[index + 2])

        def hat_func(x):
            x_array = np.asarray(x)
            is_scalar = x_array.ndim == 0
            if is_scalar:
                x_array = x_array.reshape(1)

            result = np.zeros_like(x_array, dtype=float)

            # Left element: increasing from 0 to 1
            in_left_element = (
                (x_array >= support[0]) &
                (x_array <= nodes[node_index])
            )
            if np.any(in_left_element):
                x_left = x_array[in_left_element]
                result[in_left_element] = (
                    (x_left - support[0]) / element_size
                )

            # Right element: decreasing from 1 to 0
            in_right_element = (
                (x_array > nodes[node_index]) &
                (x_array <= support[1])
            )
            if np.any(in_right_element):
                x_right = x_array[in_right_element]
                result[in_right_element] = (
                    (support[1] - x_right) / element_size
                )

            return result.item() if is_scalar else result

        return L2Function(
            self.space,
            evaluate_callable=hat_func,
            name=f'φ_{index}',
            support=support
        )

    def _create_full_hat_basis_function(self, index: int):
        """Create a single full hat (piecewise linear) basis function.

        These include boundary functions and interior functions,
        forming a complete basis without boundary conditions.
        """
        domain = self.space.domain

        # For full hat functions, we have dim nodes from a to b
        nodes = np.linspace(domain.a, domain.b, self.space.dim)

        if self.space.dim == 1:
            # Special case: single node (constant function)
            def constant_hat_func(x):
                x_array = np.asarray(x)
                return np.ones_like(x_array, dtype=float)

            return L2Function(
                self.space,
                evaluate_callable=constant_hat_func,
                name=f'φ_{index}',
                support=(domain.a, domain.b)
            )

        element_size = nodes[1] - nodes[0]
        node_x = nodes[index]

        # Determine support based on position
        if index == 0:
            # Left boundary: half-hat from left boundary to first interior
            support = (domain.a, nodes[1])

            def left_boundary_hat_func(x):
                x_array = np.asarray(x)
                is_scalar = x_array.ndim == 0
                if is_scalar:
                    x_array = x_array.reshape(1)

                result = np.zeros_like(x_array, dtype=float)

                # Decreasing from 1 to 0
                in_support = (x_array >= support[0]) & (x_array <= support[1])
                if np.any(in_support):
                    x_support = x_array[in_support]
                    result[in_support] = (nodes[1] - x_support) / element_size

                return result.item() if is_scalar else result

            hat_func = left_boundary_hat_func

        elif index == self.space.dim - 1:
            # Right boundary: half-hat from last interior to right boundary
            support = (nodes[-2], domain.b)

            def right_boundary_hat_func(x):
                x_array = np.asarray(x)
                is_scalar = x_array.ndim == 0
                if is_scalar:
                    x_array = x_array.reshape(1)

                result = np.zeros_like(x_array, dtype=float)

                # Increasing from 0 to 1
                in_support = (x_array >= support[0]) & (x_array <= support[1])
                if np.any(in_support):
                    x_support = x_array[in_support]
                    result[in_support] = (x_support - nodes[-2]) / element_size

                return result.item() if is_scalar else result

            hat_func = right_boundary_hat_func

        else:
            # Interior: full triangle hat
            support = (nodes[index-1], nodes[index+1])

            def interior_hat_func(x):
                x_array = np.asarray(x)
                is_scalar = x_array.ndim == 0
                if is_scalar:
                    x_array = x_array.reshape(1)

                result = np.zeros_like(x_array, dtype=float)

                # Left element: increasing from 0 to 1
                in_left_element = (
                    (x_array >= support[0]) &
                    (x_array <= node_x)
                )
                if np.any(in_left_element):
                    x_left = x_array[in_left_element]
                    result[in_left_element] = (
                        (x_left - support[0]) / element_size
                    )

                # Right element: decreasing from 1 to 0
                in_right_element = (
                    (x_array > node_x) &
                    (x_array <= support[1])
                )
                if np.any(in_right_element):
                    x_right = x_array[in_right_element]
                    result[in_right_element] = (
                        (support[1] - x_right) / element_size
                    )

                return result.item() if is_scalar else result

            hat_func = interior_hat_func

        return L2Function(
            self.space,
            evaluate_callable=hat_func,
            name=f'φ_{index}',
            support=support
        )

    def __getitem__(self, index: int):
        """Allow indexing syntax: provider[i]."""
        return self.get_basis_function(index)

    def __len__(self):
        """Return the dimension of the space."""
        return self.space.dim

    def __iter__(self):
        """Allow iteration over basis functions."""
        for i in range(self.space.dim):
            yield self.get_basis_function(i)


class L2Space(HilbertSpace):
    """
    L² Hilbert space on an interval [a,b] with inner product
    ⟨u,v⟩ = ∫_a^b u(x)v(x) dx.

    This class provides the foundation for Sobolev spaces and manages:
    - L² inner product and norm via integration
    - Basis function creation and management (Fourier, etc.)
    - Function evaluation and coefficient transformations
    - Domain operations on intervals

    This serves as the base class for SobolevSpace.
    """

    def __init__(
        self,
        dim: int,
        /,
        *,
        basis_functions: list = None,
        basis_type: str = 'fourier',
        domain: IntervalDomain,
    ):
        """
        Args:
            dim (int): Dimension of the space.
            basis_functions (list, optional): Custom list of basis functions.
                If provided, basis_type is ignored.
            basis_type (str): Type of basis functions
                ('fourier', 'hat', 'hat_homogeneous').
                Only used if basis_functions is None.
            domain (IntervalDomain): Domain object with optional boundary
                conditions. If boundary conditions are not specified in the
                domain, defaults will be applied based on basis_type:
                periodic for Fourier, dirichlet for hat, and homogeneous
                dirichlet for hat_homogeneous.
        """
        self._dim = dim
        self._domain = domain

        # Determine basis type from either explicit type or existing functions
        if basis_functions is not None:
            self._basis_type = 'custom'
            # Validate dimension
            if len(basis_functions) != dim:
                raise ValueError(
                    f"basis_functions length ({len(basis_functions)}) "
                    f"must match dim ({dim})"
                )
        else:
            self._basis_type = basis_type

        # Store boundary conditions with validation and conversion
        boundary_conditions = domain.boundary_conditions
        if boundary_conditions is None:
            if basis_type == 'fourier' or self._basis_type == 'fourier':
                self._boundary_conditions = BoundaryConditions.periodic()
            elif basis_type == 'hat' or self._basis_type == 'hat':
                self._boundary_conditions = BoundaryConditions.dirichlet()
            elif (basis_type == 'hat_homogeneous' or
                  self._basis_type == 'hat_homogeneous'):
                self._boundary_conditions = (
                    BoundaryConditions.dirichlet(0.0, 0.0)
                )
            else:
                self._boundary_conditions = None
        elif isinstance(boundary_conditions, BoundaryConditions) or (
            hasattr(boundary_conditions, '__class__') and
            boundary_conditions.__class__.__name__ == 'BoundaryConditions'
        ):
            self._boundary_conditions = boundary_conditions
        else:
            raise ValueError(
                "domain.boundary_conditions must be BoundaryConditions "
                "object or None"
            )

        # Validate boundary condition and basis type compatibility
        self._validate_basis_boundary_compatibility()

        # Create or store basis functions
        if basis_functions is not None:
            self._basis_functions = basis_functions
            self._basis_provider = None  # No lazy provider needed
        else:
            # Create lazy basis provider instead of all functions upfront
            self._basis_provider = LazyL2BasisProvider(
                self, basis_type
            )
            self._basis_functions = None  # Will be created on demand

        # Initialize Gram matrix as None - computed lazily when needed
        self._gram_matrix = None

        # Initialize the parent HilbertSpace with L² inner product
        super().__init__(
            dim,
            self._to_components,
            self._from_components,
            self.inner_product,
            self._default_to_dual,
            self._default_from_dual,
            copy=self._copy,
        )

    def _validate_basis_boundary_compatibility(self):
        """Validate that basis type and boundary conditions are compatible."""
        if self._boundary_conditions is None:
            return

        bc_type = self._boundary_conditions.type

        # Define compatibility rules
        if self._basis_type == 'hat_homogeneous':
            is_homogeneous_dirichlet = (
                bc_type == 'dirichlet' and
                self._boundary_conditions.is_homogeneous
            )
            if not is_homogeneous_dirichlet:
                raise ValueError(
                    f"Basis type 'hat_homogeneous' requires "
                    f"homogeneous Dirichlet boundary conditions, "
                    f"got '{bc_type}'"
                )

        # Additional validation rules can be added here
        valid_combinations = {
            'fourier': ['periodic'],
            'hat': ['dirichlet', 'neumann'],
            'hat_homogeneous': ['dirichlet']  # Only homogeneous dirichlet
        }

        if self._basis_type in valid_combinations:
            valid_bcs = valid_combinations[self._basis_type]
            if bc_type not in valid_bcs:
                raise ValueError(
                    f"Basis type '{self._basis_type}' is not compatible "
                    f"with boundary condition '{bc_type}'. "
                    f"Valid boundary conditions: {valid_bcs}"
                )

            # Special check for hat_homogeneous: must be homogeneous dirichlet
            is_hat_homogeneous_special_case = (
                self._basis_type == 'hat_homogeneous' and
                bc_type == 'dirichlet' and
                not self._boundary_conditions.is_homogeneous
            )
            if is_hat_homogeneous_special_case:
                raise ValueError(
                    "Basis type 'hat_homogeneous' requires "
                    "homogeneous Dirichlet boundary conditions"
                )

    @property
    def dim(self):
        """Return the dimension of the space."""
        return self._dim

    @property
    def domain(self):
        """Return the IntervalDomain object for this space."""
        return self._domain

    @property
    def interval(self):
        """Return the interval endpoints from domain."""
        return (self._domain.a, self._domain.b)

    def get_basis_function(self, index: int):
        """Get basis function by index, works with both lazy and explicit."""
        if self._basis_functions is not None:
            return self._basis_functions[index]
        elif self._basis_provider is not None:
            return self._basis_provider.get_basis_function(index)
        else:
            raise RuntimeError(
                "Neither explicit nor lazy basis functions available"
            )

    @property
    def basis_functions(self):
        """Property to access basis functions with consistent interface."""
        if self._basis_functions is not None:
            return self._basis_functions
        else:
            # Return the lazy provider which supports indexing and iteration
            # This provides a consistent interface whether using explicit
            # or lazy functions
            return self._basis_provider

    def basis_vector(self, i):
        """Return the ith basis function directly."""
        if i < 0 or i >= self.dim:
            raise IndexError(f"Basis index {i} out of range [0, {self.dim})")

        return self.get_basis_function(i)

    @property
    def boundary_conditions(self):
        """Boundary conditions for this space."""
        return self._boundary_conditions

    @property
    def gram_matrix(self):
        """The Gram matrix of basis functions."""
        if self._gram_matrix is None:
            self._compute_gram_matrix()
        return self._gram_matrix

    @property
    def basis_type(self):
        """The type of basis functions used."""
        return self._basis_type

    def inner_product(self, u, v):
        """
        L² inner product: ⟨u,v⟩_L² = ∫_a^b u(x)v(x) dx

        Args:
            u, v: Functions in this L² space

        Returns:
            float: L² inner product

        For L² functions, we compute ⟨u,v⟩_L² = ∫_a^b u(x)v(x) dx through
        numerical integration, not pointwise evaluation (which is not
        mathematically well-defined for general L² functions).
        """
        # For L² functions, we need to be careful about pointwise operations
        # In practice, we work with smooth approximations
        product = u * v
        return product.integrate()

    def _compute_gram_matrix(self):
        """
        Compute the Gram matrix of the basis functions using L2 inner products.
        """
        n = self.dim
        self._gram_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):  # Only compute upper triangle
                # Get basis functions - works with both lazy and explicit
                basis_i = self.get_basis_function(i)
                basis_j = self.get_basis_function(j)

                inner_prod = self.inner_product(basis_i, basis_j)
                self._gram_matrix[i, j] = inner_prod
                self._gram_matrix[j, i] = inner_prod  # Symmetric matrix

    def project(self, f):
        """
        Project a function onto this L2 space.

        Args:
            f: Function to project (callable or L2Function)

        Returns:
            L2Function: The projection of f onto this space
        """
        if callable(f):
            # Create L2Function from callable
            func = L2Function(self, evaluate_callable=f)
        else:
            func = f

        # Compute coefficients via L2 inner products
        coeffs = self._to_components(func)
        return self._from_components(coeffs)

    def _to_components(self, u):
        """
        Convert a function to coefficients using inner products with basis
        functions.
        """
        # Compute right-hand side: b_i = <u, φ_i>_L²
        rhs = np.zeros(self.dim)
        for k in range(self.dim):
            basis_func = self.get_basis_function(k)
            rhs[k] = self.inner_product(u, basis_func)

        # Solve the linear system: G * c = rhs
        gram = self.gram_matrix
        if gram is None:
            raise ValueError("Gram matrix not computed")
        coeffs = np.linalg.solve(gram, rhs)
        return coeffs

    def _from_components(self, coeff):
        """
        Convert coefficients to a function using linear combination of
        basis functions.
        """
        coeff = np.asarray(coeff)
        if len(coeff) != self.dim:
            raise ValueError(f"Coefficients must have length {self.dim}")

        # Create L2Function directly with coefficients
        return L2Function(self, coefficients=coeff)

    # Default dual space mappings
    def _default_to_dual(self, u: L2Function):
        """Default mapping to dual space using Gram matrix."""
        return LinearForm(self, mapping=lambda v: self.inner_product(u, v))

    def _default_from_dual(self, up: LinearForm):
        """Default mapping from dual space using inverse Gram matrix."""
        dual_components = np.zeros(self.dim)
        for i in range(self.dim):
            basis_func = self.get_basis_function(i)
            dual_components[i] = up(basis_func)

        gram = self.gram_matrix
        if gram is None:
            raise ValueError("Gram matrix not computed")
        components = np.linalg.solve(gram, dual_components)
        return L2Function(
            self,
            coefficients=components,
        )

    def _copy(self, x):
        """Custom copy implementation for L2Functions."""
        return L2Function(
            self,
            coefficients=self.to_components(x).copy(),
            name=getattr(x, 'name', None)
        )
