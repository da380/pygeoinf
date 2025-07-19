"""
L² spaces on interval domains.

This module provides L² Hilbert spaces on intervals as the foundation
for more specialized function spaces like Sobolev spaces.
"""

import numpy as np
import math

from pygeoinf.hilbert_space import HilbertSpace


class L2Space(HilbertSpace):
    """
    L² Hilbert space on an interval [a,b] with inner product ⟨u,v⟩ = ∫_a^b u(x)v(x) dx.

    This class provides the foundation for Sobolev spaces and manages:
    - L² inner product and norm via integration
    - Basis function creation and management (Fourier, etc.)
    - Function evaluation and coefficient transformations
    - Domain operations on intervals

    This serves as the base class for SobolevSpace.
    """

    def __init__(
        self,
        dim,
        basis_type='fourier',
        /,
        *,
        interval=(0, 1),
        boundary_conditions=None,
    ):
        """
        Args:
            dim (int): Dimension of the space.
            basis_type (str): Type of basis functions ('fourier').
            interval (tuple): Interval endpoints (a, b). Default is (0, 1).
            boundary_conditions (dict, optional): Boundary conditions specification.
                If None, defaults to periodic for Fourier basis.
        """
        self._dim = dim
        self._interval = interval
        self._a, self._b = interval
        self._length = self._b - self._a
        self._basis_type = basis_type

        # Store boundary conditions
        if boundary_conditions is None:
            if basis_type == 'fourier':
                self._boundary_conditions = {'type': 'periodic'}
            else:
                self._boundary_conditions = None
        else:
            self._boundary_conditions = boundary_conditions

        # Store IntervalDomain object
        from .interval_domain import IntervalDomain
        self._interval_domain = IntervalDomain(
            self._a, self._b, boundary_type='closed',
            name=f'[{self._a}, {self._b}]'
        )

        # Create basis functions
        self._basis_functions = self._create_basis_functions(basis_type)

        # Compute Gram matrix for L² inner products
        self._compute_gram_matrix()

        # Initialize the parent HilbertSpace with L² inner product
        super().__init__(
            dim,
            self._to_components,
            self._from_components,
            self._l2_inner_product,
            self._default_to_dual,
            self._default_from_dual,
        )

    @property
    def dim(self):
        """Return the dimension of the space."""
        return self._dim

    @property
    def interval_domain(self):
        """Return the IntervalDomain object for this space."""
        return self._interval_domain

    @property
    def interval(self):
        """Return the interval endpoints."""
        return self._interval

    @property
    def basis_functions(self):
        """Property to access basis functions."""
        return self._basis_functions

    @property
    def boundary_conditions(self):
        """Boundary conditions for this space."""
        return self._boundary_conditions

    @property
    def gram_matrix(self):
        """The Gram matrix of basis functions."""
        return self._gram_matrix

    def create_function(self, *, coefficients=None, evaluate_callable=None, name=None):
        """
        Create an L2Function instance in this space.

        Args:
            coefficients: Optional coefficient array
            evaluate_callable: Optional evaluation function
            name: Optional function name

        Returns:
            L2Function: Function in this L² space
        """
        # Import here to avoid circular imports
        from .l2_functions import L2Function
        return L2Function(
            self,
            coefficients=coefficients,
            evaluate_callable=evaluate_callable,
            name=name
        )

    def l2_inner_product(self, u, v):
        """
        L² inner product: ⟨u,v⟩_L² = ∫_a^b u(x)v(x) dx

        Args:
            u, v: Functions in this L² space

        Returns:
            float: L² inner product
        """
        return self._l2_inner_product(u, v)

    def _l2_inner_product(self, u, v):
        """
        Implementation of L² inner product using integration.

        For L² functions, we compute ⟨u,v⟩_L² = ∫_a^b u(x)v(x) dx through
        numerical integration, not pointwise evaluation (which is not
        mathematically well-defined for general L² functions).
        """
        # For L² functions, we need to be careful about pointwise operations
        # In practice, we work with smooth approximations
        product = u * v
        return product.integrate()

    def _create_basis_functions(self, basis_type):
        """
        Create basis functions based on the specified type and boundary conditions.
        """
        if basis_type != 'fourier':
            raise ValueError(f"Only 'fourier' basis is supported. Got: {basis_type}")

        # Import here to avoid circular imports
        from .l2_functions import L2Function
        basis_functions = []
        bc = self._boundary_conditions

        if bc is None or bc.get('type') == 'periodic':
            # Periodic boundary conditions: full Fourier basis
            k = 0
            normalization_factor = math.sqrt(2 / self._length)
            while len(basis_functions) < self.dim:
                freq = 2 * k * math.pi / self._length
                if k == 0:
                    # Constant term
                    def make_constant_func():
                        def constant_func(x):
                            return (normalization_factor *
                                   np.ones_like(x) / np.sqrt(2))
                        return constant_func
                    basis_func = L2Function(
                        self, evaluate_callable=make_constant_func(),
                        name='constant'
                    )
                    basis_functions.append(basis_func)
                else:
                    # Cosine term
                    def make_cosine_func(frequency):
                        def cosine_func(x):
                            return (normalization_factor *
                                   np.cos(frequency * (x - self._a)))
                        return cosine_func
                    basis_func = L2Function(
                        self, evaluate_callable=make_cosine_func(freq),
                        name=f'cos_{k}'
                    )
                    basis_functions.append(basis_func)
                    if len(basis_functions) < self.dim:
                        # Sine term
                        def make_sine_func(frequency):
                            def sine_func(x):
                                return (normalization_factor *
                                       np.sin(frequency * (x - self._a)))
                            return sine_func
                        basis_func = L2Function(
                            self, evaluate_callable=make_sine_func(freq),
                            name=f'sin_{k}'
                        )
                        basis_functions.append(basis_func)
                k += 1
        else:
            raise NotImplementedError(f"Boundary condition type '{bc.get('type')}' not implemented yet")

        return basis_functions

    def _compute_gram_matrix(self):
        """
        Compute the Gram matrix of the basis functions using L2 inner products.
        """
        n = len(self._basis_functions)
        self._gram_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):  # Only compute upper triangle
                inner_prod = self._l2_inner_product(
                    self._basis_functions[i],
                    self._basis_functions[j]
                )
                self._gram_matrix[i, j] = inner_prod
                self._gram_matrix[j, i] = inner_prod  # Symmetric matrix

    def _to_components(self, u):
        """
        Convert a function to coefficients using inner products with basis functions.
        """
        # Compute right-hand side: b_i = <u, φ_i>_L²
        rhs = np.zeros(self.dim)
        for k, basis_func in enumerate(self._basis_functions):
            rhs[k] = self._l2_inner_product(u, basis_func)

        # Solve the linear system: G * c = rhs
        coeffs = np.linalg.solve(self._gram_matrix, rhs)
        return coeffs

    def _from_components(self, coeff):
        """
        Convert coefficients to a function using linear combination of basis functions.
        """
        coeff = np.asarray(coeff)
        if len(coeff) != self.dim:
            raise ValueError(f"Coefficients must have length {self.dim}")

        # Use arithmetic operations on function instances
        result = None
        for k, c in enumerate(coeff):
            if c != 0:  # Skip zero coefficients for efficiency
                term = c * self._basis_functions[k]
                if result is None:
                    result = term
                else:
                    result = result + term

        # Handle the case where all coefficients are zero
        if result is None:
            result = 0.0 * self._basis_functions[0]

        return result

    # Default dual space mappings
    def _default_to_dual(self, u):
        """Default mapping to dual space using Gram matrix."""
        coeff = self._to_components(u)
        dual_coeff = self._gram_matrix @ coeff
        return self.dual.from_components(dual_coeff)

    def _default_from_dual(self, up):
        """Default mapping from dual space using inverse Gram matrix."""
        dual_coeff = self.dual.to_components(up)
        coeff = np.linalg.solve(self._gram_matrix, dual_coeff)
        return self._from_components(coeff)
