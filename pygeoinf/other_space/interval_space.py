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
            sobolev_scaling (callable): Function k -> scaling factor for
                mode k.
            interval (tuple): Interval endpoints (a, b). Default is (0, 1).
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
        # Store IntervalDomain object
        from .interval_domain import IntervalDomain
        self._interval_domain = IntervalDomain(
            self._a, self._b, boundary_type='closed',
            name=f'[{self._a}, {self._b}]'
        )

        # Initialize basis functions storage
        self._basis_functions = None

    @property
    def order(self):
        """Sobolev order (if available)."""
        return self._order

    @property
    def interval_domain(self):
        """Return the IntervalDomain object for this space."""
        return self._interval_domain

    @staticmethod
    def create_standard_sobolev(
        dim, order, /, *, interval=(0, 1), basis_type='fourier'
    ):
        """
        Factory method to create a standard Sobolev space with common
        basis choices and automatic scaling.

        Args:
            dim (int): Dimension of the space.
            order (float): The Sobolev order.
            interval (tuple): Interval endpoints (a, b). Default is (0, 1).
            basis_type (str): Type of basis ('fourier', 'chebyshev').
                'fourier': constant, cos, sin (full trigonometric basis)
                'chebyshev': Chebyshev polynomials of the first kind

        Returns:
            Sobolev: A Sobolev space instance with standard configuration.

        Note:
            This is a convenience method. For custom applications,
            use the main constructor with your own basis functions.
        """

        # Create simple placeholder methods - these will be replaced anyway
        def placeholder_to_coeff(u):
            """Placeholder method - will be replaced with basis functions."""
            # Just return identity for now
            return u

        def placeholder_from_coeff(coeff):
            """Placeholder method - will be replaced with basis functions."""
            # Just return the coefficients as-is
            return coeff

        # Create the Sobolev space with placeholder methods
        space = Sobolev(
            dim, placeholder_to_coeff, placeholder_from_coeff, order,
            interval=interval
        )

        # Create basis functions as SobolevFunction instances
        space._basis_functions = space._create_basis_functions(basis_type)

        # Now replace the coefficient methods with ones that use the basis functions
        space._replace_coefficient_methods_with_basis()

        return space

    def _create_basis_functions(self, basis_type):
        """Create basis functions as SobolevFunction instances."""
        from .sobolev_functions import SobolevFunction

        basis_functions = []
        length = self._length

        if basis_type == 'fourier':
            # True Fourier basis: constant, then cos/sin pairs
            k = 0
            while len(basis_functions) < self.dim:
                freq = k * np.pi / length
                if k == 0:
                    # Constant term (cos(0) = 1)
                    def make_constant_func(frequency):
                        def constant_func(x):
                            return np.ones_like(x)
                        return constant_func
                    basis_func = SobolevFunction(
                        self,
                        evaluate_callable=make_constant_func(freq),
                        name='cos_0(1)'
                    )
                    basis_functions.append(basis_func)
                else:
                    # Cosine term
                    def make_cosine_func(frequency):
                        def cosine_func(x):
                            return np.cos(frequency * (x - self._a))
                        return cosine_func
                    basis_func = SobolevFunction(
                        self,
                        evaluate_callable=make_cosine_func(freq),
                        name=f'cos_{k}({freq:.3f}*(x-{self._a}))'
                    )
                    basis_functions.append(basis_func)
                    if len(basis_functions) < self.dim:
                        # Sine term
                        def make_sine_func(frequency):
                            def sine_func(x):
                                return np.sin(frequency * (x - self._a))
                            return sine_func
                        basis_func = SobolevFunction(
                            self,
                            evaluate_callable=make_sine_func(freq),
                            name=f'sin_{k}({freq:.3f}*(x-{self._a}))'
                        )
                        basis_functions.append(basis_func)
                k += 1

        elif basis_type == 'chebyshev':
            # Create Chebyshev polynomial basis functions
            for k in range(self.dim):
                def make_chebyshev_func(degree):
                    def chebyshev_func(x):
                        # Map x from [a,b] to [-1,1]
                        t = 2 * (x - self._a) / self._length - 1
                        # Evaluate Chebyshev polynomial of first kind
                        if degree == 0:
                            return np.ones_like(t)
                        elif degree == 1:
                            return t
                        else:
                            # Use recurrence relation: T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
                            T_prev_prev = np.ones_like(t)
                            T_prev = t
                            for n in range(2, degree + 1):
                                T_curr = 2 * t * T_prev - T_prev_prev
                                T_prev_prev = T_prev
                                T_prev = T_curr
                            return T_prev
                    return chebyshev_func

                basis_func = SobolevFunction(
                    self,
                    evaluate_callable=make_chebyshev_func(k),
                    name=f'T_{k}'
                )
                basis_functions.append(basis_func)

        return basis_functions

    def get_basis_functions(self):
        """
        Get the basis functions as SobolevFunction instances.

        Returns:
            list: List of SobolevFunction instances representing the basis
        """
        if hasattr(self, '_basis_functions'):
            return self._basis_functions
        else:
            raise ValueError("Basis functions not available. Use create_standard_sobolev to create a space with basis functions.")

    @property
    def basis_functions(self):
        """Property to access basis functions."""
        return self.get_basis_functions()

    @property
    def dim(self):
        """
        Return the dimension of the space.
        """
        return self._dim

    @property
    def interval(self):
        """
        Return the interval endpoints.
        """
        return self._interval

    @property
    def length(self):
        """
        Return the interval length.
        """
        return self._length

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
        return np.dot(self._metric @ coeff1, coeff2)

    def _default_to_dual(self, u):
        """Default mapping to dual space."""
        coeff = self.to_coefficient(u)
        dual_coeff = self._metric @ coeff
        return self.dual.from_components(dual_coeff)

    def _default_from_dual(self, up):
        """Default mapping from dual space."""
        dual_coeff = self.dual.to_components(up)
        coeff = self._inverse_metric @ dual_coeff
        return self.from_coefficient(coeff)

    def to_coefficient(self, u):
        """Maps an element to its coefficient representation."""
        return self._to_coefficient(u)

    def from_coefficient(self, coeff):
        """Maps coefficients back to function values."""
        return self._from_coefficient(coeff)

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
        Convert a function to coefficients using inner products with basis
        functions.

        Args:
            u: A SobolevFunction, callable, or array

        Returns:
            np.ndarray: Coefficients in the basis representation
        """
        from .sobolev_functions import SobolevFunction

        if isinstance(u, SobolevFunction):
            # If u is a SobolevFunction, use the L2 inner product
            # Compute right-hand side: b_i = <u, φ_i>
            rhs = np.zeros(self.dim)
            for k, basis_func in enumerate(self._basis_functions):
                rhs[k] = self._l2_inner_product(u, basis_func)

            # Solve the linear system: G * c = rhs
            # where G is the Gram matrix and c are the coefficients
            coeffs = np.linalg.solve(self._gram_matrix, rhs)
            return coeffs
        elif callable(u):
            # If u is a callable, create a temporary SobolevFunction
            temp_func = SobolevFunction(self, evaluate_callable=u)
            return self._to_coefficient_with_basis(temp_func)
        else:
            # If u is an array, assume it's already function values on grid
            # Fall back to original method
            return self._original_to_coefficient(u)

    def _from_coefficient_with_basis(self, coeff):
        """
        Convert coefficients to a SobolevFunction using linear combination
        of basis.

        Args:
            coeff: Array of coefficients

        Returns:
            SobolevFunction: Linear combination of basis functions
        """
        from .sobolev_functions import SobolevFunction

        coeff = np.asarray(coeff)
        if len(coeff) != self.dim:
            raise ValueError(f"Coefficients must have length {self.dim}")

        # Create a function that evaluates the linear combination
        def linear_combination(x):
            result = 0.0
            for k, c in enumerate(coeff):
                if c != 0:  # Skip zero coefficients for efficiency
                    basis_val = self._basis_functions[k].evaluate(
                        x, check_domain=False
                    )
                    result += c * basis_val
            return result

        # Return as a SobolevFunction
        return SobolevFunction(
            self,
            evaluate_callable=linear_combination,
            coefficients=coeff.copy(),
            name=f"linear_combination_{len(coeff)}_basis_functions"
        )

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
        from scipy.integrate import quad

        def integrand(x):
            u_val = u.evaluate(x, check_domain=False)
            v_val = v.evaluate(x, check_domain=False)
            return u_val * v_val

        # Integrate over the interval
        result, _ = quad(integrand, self._a, self._b)
        return result


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
