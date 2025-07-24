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
from pygeoinf.interval.l2_space import L2Space
from pygeoinf.interval.interval_domain import IntervalDomain


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
        function_domain: 'IntervalDomain',
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
            function_domain (IntervalDomain): Domain object with optional
                boundary conditions.
        """

        self._dim = dim
        self._order = order
        self._function_domain = function_domain

        # For custom basis functions, eigenvalues must be provided
        if basis_functions is not None and eigenvalues is None:
            raise ValueError(
                "When providing custom basis_functions, you must also "
                "provide the corresponding eigenvalues for the spectral "
                "Sobolev inner product"
            )

        # Validate eigenvalues if provided
        if eigenvalues is not None:
            eigenvalues = np.asarray(eigenvalues)
            if len(eigenvalues) != dim:
                raise ValueError(
                    f"eigenvalues length ({len(eigenvalues)}) "
                    f"must match dim ({dim})"
                )
            self._eigenvalues = eigenvalues
        else:
            # Compute eigenvalues from basis type and boundary conditions
            self._eigenvalues = self._compute_eigenvalues(
                dim, basis_type, function_domain
            )

        # Initialize parent L2Space (handles basis creation via lazy provider)
        super().__init__(
            dim,
            basis_functions=basis_functions,
            basis_type=basis_type,
            function_domain=function_domain,
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
        Convert an L2Function to coefficients using Sobolev inner product.

        For Sobolev spaces, we want coefficients c such that:
        u = ∑_k c_k φ_k
        where the coefficients satisfy orthogonality in the Sobolev norm.
        """
        from .l2_functions import L2Function

        if not isinstance(u, L2Function):
            raise TypeError("Expected L2Function instance")

        # First get L² coefficients
        l2_coeffs = np.zeros(self.dim)
        for k in range(self.dim):
            basis_func = self.get_basis_function(k)
            # L² inner product gives us the L² spectral coefficients
            l2_coeffs[k] = super().inner_product(u, basis_func)

        # For Sobolev spaces, the coefficients are related to L² coefficients
        # through the eigenvalue scaling: c_k = û_k / sqrt(1 + λ_k)^s
        # where û_k are the L² coefficients
        sobolev_coeffs = np.zeros(self.dim)
        for k in range(self.dim):
            weight = (1.0 + self._eigenvalues[k]) ** (self._order / 2.0)
            sobolev_coeffs[k] = l2_coeffs[k] / weight

        return sobolev_coeffs

    def _from_components(self, coeff):
        """
        Convert Sobolev coefficients to an L2Function using linear combination
        of basis functions.

        The Sobolev coefficients are scaled by the eigenvalue weights.
        """
        coeff = np.asarray(coeff)
        if len(coeff) != self.dim:
            raise ValueError(f"Coefficients must have length {self.dim}")

        # Convert Sobolev coefficients back to L² coefficients for
        # reconstruction: û_k = c_k * sqrt(1 + λ_k)^s
        l2_coeffs = np.zeros(self.dim)
        for k in range(self.dim):
            weight = (1.0 + self._eigenvalues[k]) ** (self._order / 2.0)
            l2_coeffs[k] = coeff[k] * weight

        # Use arithmetic operations on L2Function instances with L² coeffs
        result = None
        for k, c in enumerate(l2_coeffs):
            if c != 0:  # Skip zero coefficients for efficiency
                basis_func = self.get_basis_function(k)  # Use lazy provider
                term = c * basis_func
                if result is None:
                    result = term
                else:
                    result = result + term

        # Handle the case where all coefficients are zero
        if result is None:
            result = 0.0 * self.get_basis_function(0)

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

        for k in range(self.dim):
            basis_func = self.get_basis_function(k)  # Use lazy basis provider
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
        bc_type = (boundary_conditions.type if boundary_conditions
                   else 'periodic')

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

    def _compute_eigenvalues(self, dim, basis_type, function_domain):
        """
        Compute eigenvalues for the basis functions.

        Args:
            dim (int): Dimension of the space
            basis_type (str): Type of basis functions
            function_domain (IntervalDomain): Domain object

        Returns:
            numpy.ndarray: Array of eigenvalues corresponding to basis
                functions
        """
        import math

        # Get boundary conditions
        from pygeoinf.interval.interval_domain import BoundaryConditions
        boundary_conditions = function_domain.boundary_conditions

        if boundary_conditions is None:
            if basis_type == 'fourier':
                boundary_conditions = BoundaryConditions.periodic()
            else:
                boundary_conditions = None

        # Compute eigenvalues based on boundary conditions and basis type
        eigenvalues = []
        bc_type = (boundary_conditions.type if boundary_conditions
                   else 'periodic')

        if basis_type == 'fourier' and bc_type == 'periodic':
            # Full Fourier basis: λ_0 = 0, λ_{2k-1} = λ_{2k} = (2kπ/L)^2
            k = 0
            while len(eigenvalues) < dim:
                if k == 0:
                    eigenvalues.append(0.0)  # Constant term
                else:
                    # Both cosine and sine terms have same eigenvalue
                    domain_length = function_domain.length
                    eigenval = (2 * k * math.pi / domain_length) ** 2
                    eigenvalues.append(eigenval)  # Cosine term
                    if len(eigenvalues) < dim:
                        eigenvalues.append(eigenval)  # Sine term
                k += 1

        elif basis_type == 'hat' and bc_type == 'dirichlet':
            # Hat functions with Dirichlet boundary conditions
            # Eigenvalues are approximately (kπ/L)^2 for k = 1, 2, ...
            for k in range(dim):
                eigenval = ((k + 1) * math.pi / function_domain.length) ** 2
                eigenvalues.append(eigenval)

        elif basis_type == 'hat_homogeneous' and bc_type == 'dirichlet':
            # Homogeneous hat functions with Dirichlet boundary conditions
            for k in range(dim):
                eigenval = ((k + 1) * math.pi / function_domain.length) ** 2
                eigenvalues.append(eigenval)

        else:
            raise NotImplementedError(
                f"Eigenvalue computation for basis_type='{basis_type}' "
                f"with boundary condition '{bc_type}' not implemented"
            )

        return np.array(eigenvalues[:dim])

    def _compute_gram_matrix(self):
        """
        Compute the Gram matrix of basis functions using L² inner
        products (for numerical stability).
        """
        n = self.dim
        self._gram_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):  # Only compute upper triangle
                # Get basis functions using lazy provider
                basis_i = self.get_basis_function(i)
                basis_j = self.get_basis_function(j)
                # Use L² inner product from parent class for Gram matrix
                inner_prod = super().inner_product(basis_i, basis_j)
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
        function_domain: IntervalDomain,
    ):
        """
        Args:
            dim (int): Dimension of the space.
            function_domain (IntervalDomain): Domain object.
        """
        # Create L2 space with order=0 (no Sobolev scaling)
        super().__init__(
            dim, 0.0, basis_type='fourier', function_domain=function_domain
        )
