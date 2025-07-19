"""
Sobolev spaces on a segment/interval [a, b].
"""

import numpy as np
from scipy.sparse import diags

from pygeoinf.hilbert_space import (
    HilbertSpace,
    LinearOperator,
    EuclideanSpace,
)
from pygeoinf.gaussian_measure import GaussianMeasure

class Sobolev(HilbertSpace):
    """
    Implementation of the Sobolev space H^s on a segment [a, b].

    This class provides a flexible framework for Sobolev spaces on intervals,
    allowing users to specify their own basis functions and coefficient
    transformations that align with their specific covariance operators
    for optimal performance in Bayesian inversion.

    The user provides:
    - to_coefficient: Maps function values to coefficient representation
    - from_coefficient: Maps coefficients back to function values
    - sobolev_scaling: Function defining the Sobolev norm scaling

    This design allows the basis choice to be aligned with the covariance
    structure of the specific application.
    """

    def __init__(
        self,
        dim,
        to_coefficient,
        from_coefficient,
        order,
        /,
        *,
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
            to_coefficient (callable): Maps function values to coefficients.
            from_coefficient (callable): Maps coefficients to function values.
            order (float): Sobolev order s.
            interval (tuple): Interval endpoints (a, b). Default is (0, 1).
            boundary_conditions (dict, optional): Boundary conditions specification.
                If None, defaults to periodic for standard spaces.
            inner_product (callable, optional): Custom inner product.
            to_dual (callable, optional): Custom dual mapping.
            from_dual (callable, optional): Custom dual inverse mapping.
            vector_multiply (callable, optional): Custom pointwise
                multiplication.
        """

        self._dim = dim
        self._interval = interval
        self._a, self._b = interval
        self._length = self._b - self._a
        self._to_coefficient = to_coefficient
        self._from_coefficient = from_coefficient
        # Store order if provided (for canonical spaces)
        self._order = order
        # Store boundary conditions
        self._boundary_conditions = boundary_conditions
        # Store IntervalDomain object
        from .interval_domain import IntervalDomain
        self._interval_domain = IntervalDomain(
            self._a, self._b, boundary_type='closed',
            name=f'[{self._a}, {self._b}]'
        )

        # Initialize basis functions storage
        self._basis_functions = None

        # Not all basis functions have eigenvalues
        self._eigenvalues = None

        # Initialize the parent HilbertSpace
        super().__init__(
            dim,
            to_coefficient,  # to_components
            from_coefficient,  # from_components
            inner_product or self._default_inner_product,
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
        """
        Return the dimension of the space.
        """
        return self._dim

    @property
    def basis_functions(self):
        """Property to access basis functions."""
        if self._basis_functions is not None:
            return self._basis_functions
        else:
            raise ValueError(
                "Basis functions not available. Use Sobolev.create_standard_sobolev() to create "
                "a space with basis functions."
            )

    def to_coefficient(self, u):
        """Maps an element to its coefficient representation."""
        return self._to_coefficient(u)

    def from_coefficient(self, coeff):
        """Maps coefficients back to function values."""
        return self._from_coefficient(coeff)

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
            # Cosine basis + constant: λ_0 = 0, λ_k = (kπ/L)^2 for k = 1, 2, ...
            for k in range(dim):
                if k == 0:
                    eigenvalues.append(0.0)  # Constant term
                else:
                    eigenval = (k * math.pi / length) ** 2
                    eigenvalues.append(eigenval)
        else:
            raise ValueError(f"Unknown boundary condition type: {bc_type}")

        return np.array(eigenvalues)

    @staticmethod
    def _spectral_sobolev_inner_product_factory(order, eigenvalues):
        """
        Returns a function that computes the H^s inner product using the spectral
        definition for Laplacian eigenfunction bases.

        For basis functions that are eigenfunctions of the Laplacian operator
        with eigenvalues λ_k, the H^s inner product is:
        ⟨u,v⟩_H^s = ∑_k (1 + λ_k)^s û_k v̂_k

        Args:
            order (float): Sobolev order s
            length (float): Length of the interval
            boundary_conditions (dict): Boundary conditions specification

        Returns:
            callable: Inner product function for SobolevFunction instances
        """

        def inner_product(u, v):
            # Get coefficients of both functions
            u_coeff = u.space.to_coefficient(u)
            v_coeff = v.space.to_coefficient(v)

            # Compute spectral inner product: ∑_k (1 + λ_k)^s û_k v̂_k
            result = 0.0
            for k in range(len(u_coeff)):
                sobolev_weight = (1.0 + eigenvalues[k]) ** order
                result += sobolev_weight * u_coeff[k] * v_coeff[k]

            return result

        return inner_product

    @staticmethod
    def create_standard_sobolev(
        dim, order, /, *, interval=(0, 1), basis_type='fourier',
        boundary_conditions=None
    ):
        """
        create_standard_sobolev to create Sobolev spaces on intervals using eigenfunction bases.

        This create_standard_sobolev creates Sobolev spaces with predefined basis functions that
        are eigenfunctions of the Laplacian operator. Only Fourier-based bases
        are supported as they have well-defined eigenvalues for the spectral
        inner product.

        Args:
            dim (int): Dimension of the approximating space (number of basis functions)
            order (float): Sobolev order (s in H^s)
            interval (tuple): Interval (a, b) for the domain
            basis_type (str): Type of basis ('fourier' only)
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

        # Create placeholder methods - these will be replaced anyway
        def placeholder_to_coeff(u):
            """Placeholder method - will be replaced with basis functions."""
            return u

        def placeholder_from_coeff(coeff):
            """Placeholder method - will be replaced with basis functions."""
            return coeff

        # Create basis functions and eigenvalues
        length = interval[1] - interval[0]
        eigenvalues = Sobolev._default_spectrum(
            dim, boundary_conditions, length
        )

        # Create spectral inner product
        inner_product = Sobolev._spectral_sobolev_inner_product_factory(
            order, eigenvalues
        )

        # Create the Sobolev space with placeholder methods
        space = Sobolev(
            dim, placeholder_to_coeff, placeholder_from_coeff, order,
            interval=interval,
            boundary_conditions=boundary_conditions,
            inner_product=inner_product
        )

        # Create basis functions as SobolevFunction instances
        space._basis_functions = space._create_basis_functions(basis_type)
        space._eigenvalues = eigenvalues
        # Replace coefficient methods with ones that use the basis functions
        space._replace_coefficient_methods_with_basis()

        return space

    def _create_basis_functions(self, basis_type):
        """
        Create basis functions as SobolevFunction instances based on boundary
        conditions.

        Only Fourier-based bases are supported as they correspond to
        Laplacian eigenfunctions with well-defined eigenvalues.
        """
        from .sobolev_functions import SobolevFunction
        import math

        if basis_type != 'fourier':
            raise ValueError(f"Only 'fourier' basis is supported. "
                           f"Got: {basis_type}")

        basis_functions = []
        length = self._length
        bc = self._boundary_conditions

        if bc is None or bc.get('type') == 'periodic':
            # Periodic boundary conditions: full Fourier basis
            k = 0
            normalization_factor = math.sqrt(2 / length)
            while len(basis_functions) < self.dim:
                freq = 2 * k * math.pi / length
                if k == 0:
                    # Constant term
                    def make_constant_func():
                        def constant_func(x):
                            return normalization_factor * np.ones_like(x) / np.sqrt(2)
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
                            return normalization_factor * np.cos(frequency * (x - self._a))
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
                                return normalization_factor * np.sin(frequency * (x - self._a))
                            return sine_func
                        basis_func = SobolevFunction(
                            self, evaluate_callable=make_sine_func(freq),
                            name=f'sin_{k}'
                        )
                        basis_functions.append(basis_func)
                k += 1

        elif bc.get('type') == 'dirichlet':
            # Dirichlet BC: pure sine basis (homogeneous case)
            if bc.get('left', 0) == 0 and bc.get('right', 0) == 0:
                for k in range(1, self.dim + 1):
                    freq = k * math.pi / length
                    def make_sine_func(frequency):
                        def sine_func(x):
                            return np.sin(frequency * (x - self._a))
                        return sine_func
                    basis_func = SobolevFunction(
                        self, evaluate_callable=make_sine_func(freq),
                        name=f'sin_{k}_dirichlet'
                    )
                    basis_functions.append(basis_func)
            else:
                raise NotImplementedError(
                    "Non-homogeneous Dirichlet BC not implemented"
                )

        elif bc.get('type') == 'neumann':
            # Neumann BC: cosine basis + constant (homogeneous case)
            if bc.get('left', 0) == 0 and bc.get('right', 0) == 0:
                # Constant term
                def make_constant_func():
                    def constant_func(x):
                        return np.ones_like(x)
                    return constant_func
                basis_func = SobolevFunction(
                    self, evaluate_callable=make_constant_func(),
                    name='constant_neumann'
                )
                basis_functions.append(basis_func)

                # Cosine terms
                for k in range(1, self.dim):
                    freq = k * math.pi / length

                    def make_cosine_func(frequency):
                        def cosine_func(x):
                            return np.cos(frequency * (x - self._a))
                        return cosine_func
                    basis_func = SobolevFunction(
                        self, evaluate_callable=make_cosine_func(freq),
                        name=f'cos_{k}_neumann'
                    )
                    basis_functions.append(basis_func)
            else:
                raise NotImplementedError(
                    "Non-homogeneous Neumann BC not implemented"
                )
        else:
            raise NotImplementedError(
                f"Fourier basis for BC type '{bc.get('type')}' not implemented"
            )

        return basis_functions

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
            coeff = self.to_coefficient(u)
            coeff = matrix @ coeff
            return self.from_coefficient(coeff)

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
            return self.from_coefficient(coeff)

        def formal_adjoint(u):
            coeff = self.to_coefficient(u)
            return matrix @ coeff

        covariance_factor = LinearOperator(
            domain, codomain, mapping, formal_adjoint_mapping=formal_adjoint
        )

        return GaussianMeasure(
            covariance_factor=covariance_factor,
            expectation=expectation,
        )

    # ================================================================#
    #                       Default methods                          #
    # ================================================================#

    def _default_inner_product(self, u1, u2):
        """Default Sobolev inner product."""
        coeff1 = self.to_coefficient(u1)
        coeff2 = self.to_coefficient(u2)
        return np.dot(self.metric_tensor @ coeff1, coeff2)

    def _default_to_dual(self, u):
        """Default mapping to dual space."""
        coeff = self.to_coefficient(u)
        dual_coeff = self.metric_tensor @ coeff
        return self.dual.from_components(dual_coeff)

    def _default_from_dual(self, up):
        """Default mapping from dual space."""
        dual_coeff = self.dual.to_components(up)
        coeff = self._inverse_metric @ dual_coeff
        return self.from_coefficient(coeff)

    def _replace_coefficient_methods_with_basis(self):
        """
        Replace the coefficient methods with ones that use the basis functions
        and proper inner products.
        """
        # Store original methods for fallback
        self._original_to_coefficient = self._to_coefficient
        self._original_from_coefficient = self._from_coefficient

        # Replace with basis-function-based methods
        self._to_coefficient = self._to_coefficient_with_basis
        self._from_coefficient = self._from_coefficient_with_basis

        # Compute and store the Gram matrix for inner products
        self._compute_gram_matrix()

    def _to_coefficient_with_basis(self, u):
        """
        Convert a SobolevFunction to coefficients using inner products with basis
        functions.

        Args:
            u: A SobolevFunction instance

        Returns:
            np.ndarray: Coefficients in the basis representation
        """
        from .sobolev_functions import SobolevFunction

        if not isinstance(u, SobolevFunction):
            raise TypeError("_to_coefficient_with_basis only accepts SobolevFunction instances")

        # Compute right-hand side: b_i = <u, φ_i>
        rhs = np.zeros(self.dim)
        for k, basis_func in enumerate(self._basis_functions):
            rhs[k] = self._l2_inner_product(u, basis_func)

        # Solve the linear system: G * c = rhs
        # where G is the Gram matrix and c are the coefficients
        coeffs = np.linalg.solve(self._gram_matrix, rhs)
        return coeffs

    def _from_coefficient_with_basis(self, coeff):
        """
        Convert coefficients to a SobolevFunction using linear combination
        of basis functions.

        Args:
            coeff: Array of coefficients

        Returns:
            SobolevFunction: Linear combination of basis functions
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

    def _compute_gram_matrix(self):
        """
        Compute the Gram matrix of the basis functions using L2 inner products.
        This is used for efficient coefficient computations.
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

    def _l2_inner_product(self, u, v):
        """
        Compute the L2 inner product between two SobolevFunctions.

        Args:
            u, v: SobolevFunction instances

        Returns:
            float: L2 inner product <u, v>_L2 = ∫_a^b u(x) v(x) dx
        """
        # Use function multiplication and integrate over the domain
        product = u * v
        return product.integrate()


class Lebesgue(Sobolev):
    """
    Implementation of the Lebesgue space L2 on a segment [a, b].

    This is a convenience class that creates an L2 space using
    a simple identity basis (no transformation).
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

        # Identity transformations for L2 space
        def identity(u):
            return u.copy()

        def l2_scaling(k):
            return 1.0  # No Sobolev scaling for L2

        super().__init__(
            dim, identity, identity, l2_scaling, interval=interval
        )
