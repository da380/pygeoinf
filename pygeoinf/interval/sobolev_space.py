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
from pygeoinf.interval.l2_space import L2Space, LazySpectrumProvider
from pygeoinf.interval.interval_domain import IntervalDomain


class Sobolev(L2Space):
    """
    Implementation of the Sobolev space H^s on a segment [a, b].

    This class provides Sobolev spaces on intervals where users specify:
    - The Sobolev order s
    - The basis type (e.g., 'fourier') or custom basis functions
    - Boundary conditions

    Mathematical Foundation:
    -----------------------
    This implementation uses the SPECTRAL definition of the Sobolev inner
    product:
        ⟨u,v⟩_H^s = ∑_k (1 + λ_k)^s û_k v̂_k

    where λ_k are eigenvalues of the negative Laplacian operator -Δ with
    boundary conditions, and û_k, v̂_k are L² spectral coefficients.

    The Sobolev space is thus intrinsically tied to a self-adjoint operator:
    - Periodic BC: -Δ with periodic boundary conditions
    - Dirichlet BC: -Δ with homogeneous Dirichlet boundary conditions
    - Neumann BC: -Δ with Neumann boundary conditions

    Each choice defines a different operator spectrum, hence different
    Sobolev spaces. When s=0, this reduces to the canonical L² inner product.

    Note: This spectral approach differs from the classical weak derivative
    definition: ⟨u,v⟩_H^s = ∑_{|α|≤s} ⟨D^α u, D^α v⟩_L²

    Design Philosophy:
    -----------------
    Basis functions are the fundamental objects, and coefficients are
    projections onto these functions using the spectral Sobolev inner product.
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
        inner_product_type='spectral',
        spectrum_provider=None,
        operator=None,
    ):
        """
        Args:
            dim (int): Dimension of the space.
            order (float): Sobolev order s.
            basis_functions (list, optional): Custom list of L2Function basis
                functions. If provided, eigenvalues must also be provided.
            eigenvalues (array, optional): Eigenvalues corresponding to the
                basis functions. Required if basis_functions is provided.
                These should be eigenvalues of the self-adjoint operator
                defining the spectral Sobolev inner product.
            basis_type (str): Type of basis functions ('fourier').
                Only used if basis_functions is None.
            function_domain (IntervalDomain): Domain object with optional
                boundary conditions. The boundary conditions determine the
                self-adjoint operator whose spectrum defines this Sobolev
                space.
            inner_product_type (str): Type of inner product to use:
                'spectral' - Uses spectral definition with eigenvalues
                'weak_derivative' - Uses classical weak derivative definition
            spectrum_provider (LazySpectrumProvider, optional): Provider for
                eigenfunctions and eigenvalues. Only used with spectral
                inner product.
            operator (optional): Operator whose spectrum defines the space.
                Only used with spectral inner product.

        Note:
            The eigenvalues λ_k correspond to the spectrum of the negative
            Laplacian operator -Δ with the specified boundary conditions:
            - Periodic: eigenfunctions are Fourier modes
            - Dirichlet: eigenfunctions are sine modes
            - Neumann: eigenfunctions are cosine modes + constant
        """

        self._dim = dim
        self._order = order
        self._function_domain = function_domain
        self._inner_product_type = inner_product_type

        # Validate inner product type
        if inner_product_type not in ['spectral', 'weak_derivative']:
            raise ValueError(
                f"inner_product_type must be 'spectral' or 'weak_derivative', "
                f"got '{inner_product_type}'"
            )

        # For spectral inner product, we need eigenvalue information
        if inner_product_type == 'spectral':
            # For custom basis functions, eigenvalues must be provided
            if basis_functions is not None and eigenvalues is None:
                raise ValueError(
                    "When providing custom basis_functions for spectral "
                    "inner product, you must also provide the corresponding "
                    "eigenvalues"
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
                self._spectrum_provider = spectrum_provider
            else:
                # For spectral inner product, use LazySpectrumProvider
                # which can compute eigenvalues
                if spectrum_provider is None:
                    # Create LazySpectrumProvider for eigenvalue computation
                    self._spectrum_provider = LazySpectrumProvider(
                        self, basis_type, operator
                    )
                else:
                    self._spectrum_provider = spectrum_provider
                self._eigenvalues = None  # Will be computed lazily
        else:
            # For weak derivative inner product, eigenvalues not needed
            self._eigenvalues = None
            self._spectrum_provider = spectrum_provider

        # Store operator for spectral case
        self._operator = operator

        # Initialize parent L2Space (handles basis creation via lazy provider)
        super().__init__(
            dim,
            basis_functions=basis_functions,
            basis_type=basis_type,
            function_domain=function_domain,
        )

        # Override the parent's basis provider if we need spectral information
        if (inner_product_type == 'spectral' and
                basis_functions is None and
                self._spectrum_provider is not None):
            # Replace LazyBasisProvider with LazySpectrumProvider
            self._basis_provider = self._spectrum_provider

        # Don't compute Gram matrix here - will be computed lazily
        self._gram_matrix = None

    @property
    def order(self):
        """Sobolev order (if available)."""
        return self._order

    @property
    def inner_product_type(self):
        """Type of inner product used ('spectral' or 'weak_derivative')."""
        return self._inner_product_type

    @property
    def eigenvalues(self):
        """Eigenvalues of the basis functions (if available)."""
        if self._eigenvalues is not None:
            return self._eigenvalues
        elif (self._inner_product_type == 'spectral' and
              self._spectrum_provider is not None):
            # Get eigenvalues from LazySpectrumProvider
            eigenvals = np.zeros(self.dim)
            for i in range(self.dim):
                eigenvals[i] = self._spectrum_provider.get_eigenvalue(i)
            return eigenvals
        else:
            return None

    @property
    def operator(self):
        """
        Description of the self-adjoint operator defining this Sobolev space.

        Returns:
            dict: Information about the operator whose spectrum defines the
                spectral Sobolev inner product, including operator type,
                boundary conditions, and domain.
        """
        boundary_conditions = self._function_domain.boundary_conditions
        bc_type = (boundary_conditions.type if boundary_conditions
                   else 'periodic')

        return {
            'type': 'negative_laplacian',
            'symbol': '-Δ',
            'boundary_conditions': bc_type,
            'domain': (f'[{self._function_domain.left}, '
                       f'{self._function_domain.right}]'),
            'description': (f'Negative Laplacian operator with {bc_type} '
                            f'boundary conditions'),
            'eigenfunction_basis': self._get_eigenfunction_description(bc_type)
        }

    def _get_eigenfunction_description(self, bc_type):
        """Get description of eigenfunctions for the operator."""
        if bc_type == 'periodic':
            return 'Fourier modes: 1, cos(2πkx/L), sin(2πkx/L)'
        elif bc_type == 'dirichlet':
            return 'Sine modes: sin(πkx/L) for k=1,2,...'
        elif bc_type == 'neumann':
            return 'Cosine modes + constant: 1, cos(πkx/L) for k=1,2,...'
        else:
            return f'Custom basis for {bc_type} boundary conditions'

    @property
    def gram_matrix(self):
        """The Gram matrix of basis functions using Sobolev inner products."""
        if self._gram_matrix is None:
            self._compute_gram_matrix()
        return self._gram_matrix

    def _to_components(self, u):
        """
        Convert a function to coefficients using Sobolev inner products with
        basis functions.
        """
        # Compute right-hand side: b_i = <u, φ_i>_H^s (Sobolev inner product)
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
        Convert coefficients to an L2Function using linear combination
        of basis functions.
        """
        from .l2_functions import L2Function

        coeff = np.asarray(coeff)
        if len(coeff) != self.dim:
            raise ValueError(f"Coefficients must have length {self.dim}")

        # Create L2Function directly with coefficients (same as L2Space)
        return L2Function(self, coefficients=coeff)

    def inner_product(self, u, v):
        """
        Sobolev inner product H^s.

        This method supports two definitions:
        1. Spectral: ⟨u,v⟩_H^s = ∑_k (1 + λ_k)^s û_k v̂_k
        2. Weak derivative: ⟨u,v⟩_H^s = ∑_{|α|≤s} ⟨D^α u, D^α v⟩_L²

        The method used depends on the inner_product_type parameter set
        during initialization.

        Args:
            u, v: Functions in this Sobolev space

        Returns:
            float: Sobolev inner product value
        """
        if self._inner_product_type == 'spectral':
            return self._spectral_inner_product(u, v)
        elif self._inner_product_type == 'weak_derivative':
            return self._weak_derivative_inner_product(u, v)
        else:
            raise ValueError(
                f"Unknown inner product type: {self._inner_product_type}"
            )

    def _spectral_inner_product(self, u, v):
        """
        Spectral definition of Sobolev inner product H^s.

        Mathematical Definition:
        ⟨u,v⟩_H^s = ∑_k (1 + λ_k)^s û_k v̂_k

        where:
        - λ_k are eigenvalues of -Δ (available via self.eigenvalues)
        - û_k, v̂_k are L² spectral coefficients: û_k = ⟨u, φ_k⟩_L²
        - φ_k are eigenfunctions of -Δ (basis functions)
        - s is the Sobolev order (available via self.order)

        Args:
            u, v: Functions in this Sobolev space

        Returns:
            float: Spectral Sobolev inner product value
        """
        from .l2_functions import L2Function

        if not isinstance(u, L2Function) or not isinstance(v, L2Function):
            raise TypeError(
                "Spectral Sobolev inner product requires L2Function instances"
            )

        eigenvalues = self.eigenvalues
        if eigenvalues is None:
            raise ValueError(
                "Eigenvalues not available for spectral inner product"
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
            sobolev_weight = (1.0 + eigenvalues[k]) ** self._order
            result += sobolev_weight * u_coeffs[k] * v_coeffs[k]

        return result

    def _weak_derivative_inner_product(self, u, v):
        """
        Weak derivative definition of Sobolev inner product H^s.

        Mathematical Definition:
        ⟨u,v⟩_H^s = ∑_{|α|≤s} ⟨D^α u, D^α v⟩_L²

        where D^α represents weak derivatives of order α.

        Args:
            u, v: Functions in this Sobolev space

        Returns:
            float: Weak derivative Sobolev inner product value

        Note:
            This is a placeholder implementation. Full implementation
            requires weak derivative computation methods.
        """
        # TODO: Implement weak derivative computation
        # For now, fall back to L² inner product when s=0
        if self._order == 0:
            return super().inner_product(u, v)
        else:
            raise NotImplementedError(
                "Weak derivative inner product not yet implemented for s > 0. "
                "Use inner_product_type='spectral' for now."
            )

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

    def _compute_gram_matrix(self):
        """
        Compute the Gram matrix of basis functions using Sobolev inner
        products.

        This computes G[i,j] = ⟨φ_i, φ_j⟩_H^s where the inner product
        is determined by the inner_product_type (spectral or weak_derivative).
        """
        n = self.dim
        self._gram_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):  # Only compute upper triangle
                # Get basis functions using lazy provider
                basis_i = self.get_basis_function(i)
                basis_j = self.get_basis_function(j)

                # Use Sobolev inner product (dispatches to spectral or weak)
                inner_prod = self.inner_product(basis_i, basis_j)
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
