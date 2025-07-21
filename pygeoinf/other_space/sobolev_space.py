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
        dim,
        order,
        /,
        *,
        basis_type='fourier',
        basis_functions=None,
        interval=(0, 1),
        boundary_conditions=None,
        inner_product=None,
        to_dual=None,
        from_dual=None,
        vector_multiply=None,
    ):
        """
        Args:
            dim (int): Dimension of the space.
            order (float): Sobolev order s.
            basis_type (str): Type of basis functions ('fourier').
                Ignored if basis_functions is provided.
            basis_functions (list, optional): Custom list of basis functions.
                If None, creates basis functions based on basis_type.
            interval (tuple): Interval endpoints (a, b). Default is (0, 1).
            boundary_conditions (dict, optional): Boundary conditions
                specification. If None, defaults to periodic for Fourier.
            inner_product (callable, optional): Custom inner product.
                If None, uses spectral Sobolev inner product.
            to_dual (callable, optional): Custom dual mapping.
            from_dual (callable, optional): Custom dual inverse mapping.
            vector_multiply (callable, optional): Custom pointwise multiplication.
        """

        self._dim = dim
        self._order = order
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

        # Create or store basis functions
        if basis_functions is not None:
            self._basis_functions = basis_functions
            # Validate dimension
            if len(basis_functions) != dim:
                raise ValueError(f"basis_functions length ({len(basis_functions)}) "
                               f"must match dim ({dim})")
        else:
            # Create basis functions from basis_type
            self._basis_functions = self._create_basis_functions(basis_type)

        # Compute eigenvalues for spectral inner product
        if basis_type == 'fourier':
            self._eigenvalues = self._compute_eigenvalues()
        else:
            self._eigenvalues = None

        # Set up inner product
        if inner_product is None and self._eigenvalues is not None:
            inner_product = self._spectral_sobolev_inner_product_factory(
                order, self._eigenvalues
            )
        elif inner_product is None:
            inner_product = self._default_sobolev_inner_product

        # Compute Gram matrix for coefficient transformations
        self._compute_gram_matrix()

        # Initialize the parent L2Space with proper mappings
        L2Space.__init__(
            self,
            dim,
            self._to_components,
            self._from_components,
            inner_product or self._default_sobolev_inner_product,
            to_dual or self._default_to_dual,
            from_dual or self._default_from_dual,
            vector_multiply=vector_multiply
        )

    @property
    def order(self):
        """Sobolev order (if available)."""
        return self._order

    @property
    def boundary_conditions(self):
        """Boundary conditions for this Sobolev space."""
        return self._boundary_conditions

    @property
    def interval_domain(self):
        """Return the IntervalDomain object for this space."""
        return self._interval_domain

    @property
    def dim(self):
        """Return the dimension of the space."""
        return self._dim

    @property
    def basis_functions(self):
        """Property to access basis functions."""
        return self._basis_functions

    @property
    def eigenvalues(self):
        """Eigenvalues of the basis functions (if available)."""
        return self._eigenvalues

    def create_function(self, *, coefficients=None, evaluate_callable=None, name=None):
        """
        Create a SobolevFunction instance in this space.

        Args:
            coefficients: Optional coefficient array
            evaluate_callable: Optional evaluation function
            name: Optional function name

        Returns:
            SobolevFunction: Function in this Sobolev space
        """
        from .sobolev_functions import SobolevFunction
        return SobolevFunction(
            self,
            coefficients=coefficients,
            evaluate_callable=evaluate_callable,
            name=name
        )

    def _compute_eigenvalues(self):
        """Compute eigenvalues for the current basis and boundary conditions."""
        return self._default_spectrum(
            self.dim, self._boundary_conditions, self._length
        )

    def _to_components(self, u):
        """
        Convert a SobolevFunction to coefficients using projections
        onto basis functions.
        """
        from .sobolev_functions import SobolevFunction

        if not isinstance(u, SobolevFunction):
            raise TypeError("Expected SobolevFunction instance")

        # Compute right-hand side: b_i = <u, φ_i>_L2
        rhs = np.zeros(self.dim)
        for k, basis_func in enumerate(self._basis_functions):
            rhs[k] = self._l2_inner_product(u, basis_func)

        # Solve the linear system: G * c = rhs
        coeffs = np.linalg.solve(self._gram_matrix, rhs)
        return coeffs

    def _from_components(self, coeff):
        """
        Convert coefficients to a SobolevFunction using linear combination
        of basis functions.
        """
        coeff = np.asarray(coeff)
        if len(coeff) != self.dim:
            raise ValueError(f"Coefficients must have length {self.dim}")

        # Use arithmetic operations on SobolevFunction instances
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

    def _default_sobolev_inner_product(self, u, v):
        """
        Default Sobolev inner product when spectral method is not available.
        Falls back to L2 inner product with identity scaling.
        """
        # Get coefficients of both functions
        u_coeff = self.to_components(u)
        v_coeff = self.to_components(v)

        # For now, use L2 inner product (identity scaling)
        # In practice, would need finite difference approximation for H^s
        result = 0.0
        for k in range(len(u_coeff)):
            result += u_coeff[k] * v_coeff[k]
        return result

    @staticmethod
    def _spectral_sobolev_inner_product_factory(order, eigenvalues):
        """
        Returns a function that computes the H^s inner product using the
        spectral definition for Laplacian eigenfunction bases.

        For basis functions that are eigenfunctions of the Laplacian operator
        with eigenvalues λ_k, the H^s inner product is:
        ⟨u,v⟩_H^s = ∑_k (1 + λ_k)^s û_k v̂_k

        Args:
            order (float): Sobolev order s
            eigenvalues (array): Eigenvalues corresponding to basis functions

        Returns:
            callable: Inner product function for SobolevFunction instances
        """
        def inner_product(u, v):
            # Get coefficients of both functions
            u_coeff = u.space.to_components(u)
            v_coeff = v.space.to_components(v)

            # Compute spectral inner product: ∑_k (1 + λ_k)^s û_k v̂_k
            result = 0.0
            for k in range(len(u_coeff)):
                sobolev_weight = (1.0 + eigenvalues[k]) ** order
                result += sobolev_weight * u_coeff[k] * v_coeff[k]

            return result
        return inner_product

    def _create_basis_functions(self, basis_type):
        """
        Create basis functions as SobolevFunction instances.

        Args:
            basis_type (str): Type of basis functions to create

        Returns:
            list: List of SobolevFunction instances
        """
        from .sobolev_functions import SobolevFunction
        import math

        if basis_type != 'fourier':
            raise ValueError(f"Only 'fourier' basis is supported. "
                           f"Got: {basis_type}")

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
                    basis_func = SobolevFunction(
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
                    basis_func = SobolevFunction(
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
                        basis_func = SobolevFunction(
                            self, evaluate_callable=make_sine_func(freq),
                            name=f'sin_{k}'
                        )
                        basis_functions.append(basis_func)
                k += 1
        else:
            raise NotImplementedError(f"Boundary condition type "
                                    f"'{bc.get('type')}' not implemented yet")

        return basis_functions

    def _compute_gram_matrix(self):
        """
        Compute the Gram matrix of basis functions using L2 inner products.
        """
        n = len(self._basis_functions)
        self._gram_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):  # Only compute upper triangle
                # Use L2 inner product between basis functions
                inner_prod = self._l2_inner_product(
                    self._basis_functions[i], self._basis_functions[j]
                )
                self._gram_matrix[i, j] = inner_prod
                self._gram_matrix[j, i] = inner_prod  # Symmetric matrix

    def _l2_inner_product(self, u, v):
        """
        Compute L2 inner product between two functions by integration.
        """
        product = u * v
        return product.integrate()

    @staticmethod
    def create_standard_sobolev(
        dim, order, /, *, interval=(0, 1), basis_type='fourier',
        boundary_conditions=None
    ):
        """
        Factory method to create standard Sobolev spaces on intervals using
        eigenfunction bases with spectral inner products.

        This method creates Sobolev spaces with predefined basis functions
        that are eigenfunctions of the Laplacian operator, allowing for
        efficient spectral computation of the Sobolev inner product.

        Args:
            dim (int): Dimension of the approximating space
            order (float): Sobolev order (s in H^s)
            interval (tuple): Interval (a, b) for the domain
            basis_type (str): Type of basis ('fourier' only currently)
            boundary_conditions (dict): Boundary conditions specification.
                Options: {'type': 'periodic'}, {'type': 'dirichlet'},
                        {'type': 'neumann'}
                If None, defaults to {'type': 'periodic'}

        Returns:
            Sobolev: Configured Sobolev space with spectral inner product

        Raises:
            ValueError: If unsupported basis_type or boundary conditions
        """
        if basis_type != 'fourier':
            raise ValueError(f"Only 'fourier' basis is supported. "
                           f"Got: {basis_type}")

        # Set default boundary conditions
        if boundary_conditions is None:
            boundary_conditions = {'type': 'periodic'}

        # Validate boundary conditions
        bc_type = boundary_conditions.get('type', 'periodic')
        if bc_type not in ['periodic', 'dirichlet', 'neumann']:
            raise ValueError(f"Unsupported boundary condition: {bc_type}")

        # Create the Sobolev space - the new constructor handles everything
        return Sobolev(
            dim, order,
            basis_type=basis_type,
            interval=interval,
            boundary_conditions=boundary_conditions
        )

    def random_point(self):
        """Generate a random point in the interval."""
        return np.random.uniform(self._a, self._b)

    def automorphism(self, f):
        """
        Create an automorphism based on function f.
        This applies f to each mode in coefficient space.
        """
        values = np.fromiter(
            [f(k) for k in range(self.dim)], dtype=float
        )
        matrix = diags([values], [0])

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
        matrix = diags([values], [0])

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
            dim, order=0.0, basis_type='fourier', interval=interval
        )
