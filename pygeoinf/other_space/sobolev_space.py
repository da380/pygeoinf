"""
Sobolev spaces on a segment/interval [a, b].
"""

import numpy as np
from scipy.sparse import diags

from pygeoinf.hilbert_space import (
    LinearOperator,
    EuclideanSpace,
)
from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.other_space.l2_space import L2Space


class Sobolev(L2Space):
    """
    Implementation of the Sobolev space H^s on a segment [a, b].

    This class provides Sobolev spaces on intervals where users specify:
    - The Sobolev order s
    - The basis type (e.g., 'fourier') or custom basis functions
    - Boundary conditions

    The coefficient transformations (to_components/from_components) are
    automatically derived from the basis functions via L2 projections.

    This design is mathematically natural: basis functions are the fundamental
    objects, and coefficients are just projections onto these functions.
    """

    def __init__(
        self,
        dim: int,
        order: int,
        /,
        *,
        basis_functions=None,
        eigenvalues=None,
        basis_type='fourier',
        interval=(0, 1),
        boundary_conditions=None,
    ):
        """
        Args:
            dim (int): Dimension of the space.
            order (float): Sobolev order s.
            basis_functions (list, optional): Custom list of L2Function basis
                functions. If provided, eigenvalues must also be provided.
            eigenvalues (array, optional): Eigenvalues corresponding to the
                basis functions. Required if basis_functions is provided.
            basis_type (str): Type of basis functions ('fourier').
                Only used if basis_functions is None.
            interval (tuple): Interval endpoints (a, b). Default is (0, 1).
            boundary_conditions (dict, optional): Boundary conditions
                specification. If None, defaults to periodic for Fourier.
        """

        self._dim = dim
        self._order = order
        self._interval = interval
        self._a, self._b = interval
        self._length = self._b - self._a

        # Store boundary conditions first (needed for basis function creation)
        if boundary_conditions is None:
            if basis_type == 'fourier':
                self._boundary_conditions = {'type': 'periodic'}
            else:
                self._boundary_conditions = None
        else:
            self._boundary_conditions = boundary_conditions

        # Determine basis type and validate eigenvalues/basis consistency
        if basis_functions is not None:
            self._basis_type = 'custom'
            self._basis_functions = basis_functions

            # Validate dimension
            if len(basis_functions) != dim:
                raise ValueError(
                    f"basis_functions length ({len(basis_functions)}) "
                    f"must match dim ({dim})"
                )

            # For custom basis functions, eigenvalues must be provided
            if eigenvalues is None:
                raise ValueError(
                    "When providing custom basis_functions, you must also "
                    "provide the corresponding eigenvalues for the spectral "
                    "Sobolev inner product"
                )

            # Validate eigenvalues
            eigenvalues = np.asarray(eigenvalues)
            if len(eigenvalues) != dim:
                raise ValueError(
                    f"eigenvalues length ({len(eigenvalues)}) "
                    f"must match dim ({dim})"
                )
            self._eigenvalues = eigenvalues

        else:
            self._basis_type = basis_type

            # Check if user provided eigenvalues without basis functions
            if eigenvalues is not None:
                raise ValueError(
                    "Cannot provide eigenvalues without basis_functions. "
                    "Either provide both or let the system compute them "
                    "automatically from basis_type"
                )

            # Create basis functions and eigenvalues together from basis_type
            result = self._create_basis_and_eigenvalues(basis_type)
            self._basis_functions, self._eigenvalues = result

        # Store IntervalDomain object (needed for integration in Gram matrix)
        from .interval_domain import IntervalDomain
        self._domain = IntervalDomain(
            self._a, self._b, boundary_type='closed',
            name=f'[{self._a}, {self._b}]'
        )

        # Initialize parent L2Space first
        super().__init__(
            dim,
            basis_functions=self._basis_functions,
            interval=interval,
            boundary_conditions=boundary_conditions,
        )

        # Compute Gram matrix for coefficient transformations after parent init
        self._compute_gram_matrix()

    @property
    def order(self):
        """Sobolev order (if available)."""
        return self._order

    @property
    def eigenvalues(self):
        """Eigenvalues of the basis functions (if available)."""
        return self._eigenvalues

    def _to_components(self, u):
        """
        Convert an L2Function to coefficients using projections
        onto basis functions.
        """
        from .l2_functions import L2Function

        if not isinstance(u, L2Function):
            raise TypeError("Expected L2Function instance")

        # Compute right-hand side: b_i = <u, φ_i>_L2
        rhs = np.zeros(self.dim)
        for k, basis_func in enumerate(self._basis_functions):
            rhs[k] = self.inner_product(u, basis_func)

        # Solve the linear system: G * c = rhs
        coeffs = np.linalg.solve(self._gram_matrix, rhs)
        return coeffs

    def _from_components(self, coeff):
        """
        Convert coefficients to an L2Function using linear combination
        of basis functions.
        """
        coeff = np.asarray(coeff)
        if len(coeff) != self.dim:
            raise ValueError(f"Coefficients must have length {self.dim}")

        # Use arithmetic operations on L2Function instances
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

    def inner_product(self, u, v):
        """
        Sobolev inner product H^s using spectral definition.

        For functions in H^s, the inner product is defined as:
        ⟨u,v⟩_H^s = ∑_k (1 + λ_k)^s û_k v̂_k

        where û_k, v̂_k are the L² spectral coefficients:
        û_k = ⟨u, φ_k⟩_L², v̂_k = ⟨v, φ_k⟩_L²

        Args:
            u, v: Functions in this Sobolev space

        Returns:
            float: Sobolev inner product value
        """
        from .l2_functions import L2Function

        if not isinstance(u, L2Function) or not isinstance(v, L2Function):
            raise TypeError(
                "Sobolev inner product requires L2Function instances"
            )

        # Get L² spectral coefficients using L² inner products
        u_coeffs = np.zeros(self.dim)
        v_coeffs = np.zeros(self.dim)

        for k, basis_func in enumerate(self._basis_functions):
            # Compute û_k = ⟨u, φ_k⟩_L² using L2 inner product from parent
            u_coeffs[k] = super().inner_product(u, basis_func)
            # Compute v̂_k = ⟨v, φ_k⟩_L² using L2 inner product from parent
            v_coeffs[k] = super().inner_product(v, basis_func)

        # Apply Sobolev spectral weights: ∑_k (1 + λ_k)^s û_k v̂_k
        result = 0.0
        for k in range(self.dim):
            sobolev_weight = (1.0 + self._eigenvalues[k]) ** self._order
            result += sobolev_weight * u_coeffs[k] * v_coeffs[k]

        return result

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

    @staticmethod
    def _default_spectrum(dim, boundary_conditions, length):
        """
        Return the default spectrum for this Sobolev space.
        This is a placeholder and should be overridden in subclasses.
        """
        import math

        # Compute eigenvalues based on boundary conditions
        eigenvalues = []
        bc_type = boundary_conditions.get('type', 'periodic')

        if bc_type == 'periodic':
            # Full Fourier basis: λ_0 = 0, λ_{2k-1} = λ_{2k} = (kπ/L)^2
            for k in range(dim):
                if k == 0:
                    eigenvalues.append(0.0)  # Constant term
                else:
                    # For both cos and sin terms at frequency k
                    freq_index = 2 * ((k + 1) // 2)
                    eigenval = (freq_index * math.pi / length) ** 2
                    eigenvalues.append(eigenval)

        elif bc_type == 'dirichlet':
            # Sine basis: λ_k = (kπ/L)^2 for k = 1, 2, ...
            for k in range(dim):
                eigenval = ((k + 1) * math.pi / length) ** 2
                eigenvalues.append(eigenval)

        elif bc_type == 'neumann':
            # Cosine basis + constant: λ_0 = 0, λ_k = (kπ/L)^2 for k = 1, 2,
            # ...
            for k in range(dim):
                if k == 0:
                    eigenvalues.append(0.0)  # Constant term
                else:
                    eigenval = (k * math.pi / length) ** 2
                    eigenvalues.append(eigenval)
        else:
            raise ValueError(f"Unknown boundary condition type: {bc_type}")

        return np.array(eigenvalues)

    def _create_basis_and_eigenvalues(self, basis_type):
        """
        Create basis functions and their corresponding eigenvalues together.

        Args:
            basis_type (str): Type of basis functions to create

        Returns:
            tuple: (basis_functions, eigenvalues) where basis_functions is a
                   list of L2Function instances and eigenvalues is a
                   numpy array

        Raises:
            ValueError: If basis_type is not supported
        """
        from .l2_functions import L2Function
        import math

        if basis_type != 'fourier':
            raise ValueError(f"Only 'fourier' basis is supported. "
                             f"Got: {basis_type}")

        # Compute eigenvalues based on boundary conditions
        eigenvalues = []
        basis_functions = []
        bc = self._boundary_conditions
        bc_type = bc.get('type', 'periodic') if bc else 'periodic'

        if bc_type == 'periodic':
            # Full Fourier basis with corresponding eigenvalues
            k = 0
            normalization_factor = math.sqrt(2 / self._length)

            while len(basis_functions) < self.dim:
                freq = 2 * k * math.pi / self._length

                if k == 0:
                    # Constant term: eigenvalue = 0
                    eigenvalues.append(0.0)

                    def make_constant_func():
                        def constant_func(x):
                            ones = np.ones_like(np.asarray(x))
                            result = normalization_factor * ones
                            return result / np.sqrt(2.0)
                        return constant_func
                    basis_func = L2Function(
                        self, evaluate_callable=make_constant_func(),
                        name='constant'
                    )
                    basis_functions.append(basis_func)
                else:
                    # Both cosine and sine terms have same eigenvalue
                    eigenval = (2 * k * math.pi / self._length) ** 2

                    # Cosine term
                    eigenvalues.append(eigenval)
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
                        # Sine term (same eigenvalue)
                        eigenvalues.append(eigenval)
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

        elif bc_type == 'dirichlet':
            # Sine basis: λ_k = (kπ/L)^2 for k = 1, 2, ...
            for k in range(self.dim):
                eigenval = ((k + 1) * math.pi / self._length) ** 2
                eigenvalues.append(eigenval)
                # TODO: Create sine basis functions

        elif bc_type == 'neumann':
            # Cosine basis + constant: λ_0 = 0, λ_k = (kπ/L)^2 for k ≥ 1
            for k in range(self.dim):
                if k == 0:
                    eigenvalues.append(0.0)  # Constant term
                else:
                    eigenval = (k * math.pi / self._length) ** 2
                    eigenvalues.append(eigenval)
                # TODO: Create cosine basis functions

        else:
            raise ValueError(f"Unknown boundary condition type: {bc_type}")

        # For now, only periodic boundary conditions are fully implemented
        if bc_type != 'periodic':
            raise NotImplementedError(
                f"Boundary condition type '{bc_type}' not fully "
                f"implemented yet"
            )

        return basis_functions, np.array(eigenvalues)

    def _compute_gram_matrix(self):
        """
        Compute the Gram matrix of basis functions using L2 inner products.
        """
        n = len(self._basis_functions)
        self._gram_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):  # Only compute upper triangle
                # Use L2 inner product from parent class
                inner_prod = self.inner_product(
                    self._basis_functions[i], self._basis_functions[j]
                )
                self._gram_matrix[i, j] = inner_prod
                self._gram_matrix[j, i] = inner_prod  # Symmetric matrix

    def automorphism(self, f):
        """
        Create an automorphism based on function f.
        This applies f to each mode in coefficient space.
        """
        values = np.fromiter(
            [f(k) for k in range(self.dim)], dtype=float
        )
        matrix = diags(values, 0)

        def mapping(u):
            coeff = self.to_components(u)
            coeff = matrix @ coeff
            return self.from_components(coeff)

        return LinearOperator.formally_self_adjoint(self, mapping)

    def gaussian_measure(self, f, /, *, expectation=None):
        """
        Create a Gaussian measure with covariance given by function f.
        The function f should map mode indices to covariance scaling.
        """
        values = np.fromiter(
            [np.sqrt(f(k)) for k in range(self.dim)],
            dtype=float,
        )
        matrix = diags(values, 0)

        domain = EuclideanSpace(self.dim)
        codomain = self

        def mapping(c):
            coeff = matrix @ c
            return self.from_components(coeff)

        def formal_adjoint(u):
            coeff = self.to_components(u)
            return matrix @ coeff

        covariance_factor = LinearOperator(
            domain, codomain, mapping, formal_adjoint_mapping=formal_adjoint
        )

        return GaussianMeasure(
            covariance_factor=covariance_factor,
            expectation=expectation,
        )


class Lebesgue(Sobolev):
    """
    Implementation of the Lebesgue space L2 on a segment [a, b].

    This is a convenience class that creates an L2 space using
    Fourier basis functions with order=0 (no Sobolev scaling).
    """

    def __init__(
        self,
        dim,
        /,
        *,
        interval=(0, 1),
    ):
        """
        Args:
            dim (int): Dimension of the space.
            interval (tuple): Interval endpoints (a, b). Default is (0, 1).
        """
        # Create L2 space with order=0 (no Sobolev scaling)
        super().__init__(
            dim, 0.0, basis_type='fourier', interval=interval
        )
