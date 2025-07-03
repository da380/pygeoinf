"""
Sobolev spaces on a segment/interval [a, b].
"""

import numpy as np
from scipy.sparse import diags
from scipy.fft import dct, idct, dst, idst

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
        sobolev_scaling,
        /,
        *,
        interval=(0, 1),
        order=None,
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
        self._sobolev_scaling = sobolev_scaling
        # Store order if provided (for canonical spaces)
        self._order = order
        # Store IntervalDomain object
        from .interval_domain import IntervalDomain
        self._interval_domain = IntervalDomain(self._a, self._b, boundary_type='closed', name=f'[{self._a}, {self._b}]')

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
        order, scale, dim, /, *, interval=(0, 1), basis_type='fourier'
    ):
        """
        Factory method to create a standard Sobolev space with common
        basis choices and automatic scaling.

        Args:
            order (float): The Sobolev order.
            scale (float): The Sobolev length-scale.
            dim (int): Dimension of the space.
            interval (tuple): Interval endpoints (a, b). Default is (0, 1).
            basis_type (str): Type of basis ('fourier', 'chebyshev', 'sine').

        Returns:
            Sobolev: A Sobolev space instance with standard configuration.

        Note:
            This is a convenience method. For custom applications,
            use the main constructor with your own basis functions.
        """

        length = interval[1] - interval[0]

        if basis_type == 'fourier':
            # Cosine basis with DCT
            # Grid for function evaluation
            x_grid = np.linspace(interval[0], interval[1], dim)

            def to_coeff(u):
                """Convert SobolevFunction to Fourier coefficients."""
                # Import here to avoid circular imports
                from .sobolev_functions import SobolevFunction

                if isinstance(u, SobolevFunction):
                    # If u is a SobolevFunction, evaluate it on the grid
                    u_vals = u.evaluate(x_grid, check_domain=False)
                elif callable(u):
                    # If u is a callable, evaluate it on the grid
                    u_vals = u(x_grid)
                else:
                    # If u is already an array, use it directly
                    u_vals = np.asarray(u)

                # Ensure we have the right number of points
                if len(u_vals) != dim:
                    raise ValueError(
                        f"Function values must have length {dim}, "
                        f"got {len(u_vals)}"
                    )

                return dct(u_vals, type=2, norm='ortho')

            def from_coeff(coeff):
                """Convert Fourier coefficients back to function values."""
                coeff = np.asarray(coeff)
                if len(coeff) != dim:
                    raise ValueError(
                        f"Coefficients must have length {dim}, "
                        f"got {len(coeff)}"
                    )

                return idct(coeff, type=2, norm='ortho')

            def scaling(k):
                freq = k * np.pi / length
                return (1 + (scale * freq) ** 2) ** order

        elif basis_type == 'sine':
            # Sine basis with DST (zero boundary conditions)
            # Grid excludes endpoints for sine basis
            x_grid = np.linspace(interval[0], interval[1], dim + 2)[1:-1]

            def to_coeff(u):
                """Convert SobolevFunction to sine coefficients."""
                # Import here to avoid circular imports
                from .sobolev_functions import SobolevFunction

                if isinstance(u, SobolevFunction):
                    # If u is a SobolevFunction, evaluate it on the grid
                    u_vals = u.evaluate(x_grid, check_domain=False)
                elif callable(u):
                    # If u is a callable, evaluate it on the grid
                    u_vals = u(x_grid)
                else:
                    # If u is already an array, use it directly
                    u_vals = np.asarray(u)

                # Ensure we have the right number of points
                if len(u_vals) != dim:
                    raise ValueError(
                        f"Function values must have length {dim}, "
                        f"got {len(u_vals)}"
                    )

                return dst(u_vals, type=1, norm='ortho')

            def from_coeff(coeff):
                """Convert sine coefficients back to function values."""
                coeff = np.asarray(coeff)
                if len(coeff) != dim:
                    raise ValueError(
                        f"Coefficients must have length {dim}, "
                        f"got {len(coeff)}"
                    )

                return idst(coeff, type=1, norm='ortho')

            def scaling(k):
                freq = (k + 1) * np.pi / length  # k+1 for sine modes
                return (1 + (scale * freq) ** 2) ** order

        elif basis_type == 'chebyshev':
            # Chebyshev polynomials on [-1, 1], mapped to [a, b]
            # Chebyshev-Gauss-Lobatto points
            cheb_points = np.cos(np.pi * np.arange(dim) / (dim - 1))
            # Map to interval [a, b]
            x_grid = (0.5 * (interval[1] - interval[0]) * cheb_points +
                      0.5 * (interval[1] + interval[0]))

            def to_coeff(u):
                """Convert SobolevFunction to Chebyshev coefficients."""
                # Import here to avoid circular imports
                from .sobolev_functions import SobolevFunction

                if isinstance(u, SobolevFunction):
                    # If u is a SobolevFunction, evaluate on Chebyshev points
                    u_vals = u.evaluate(x_grid, check_domain=False)
                elif callable(u):
                    # If u is a callable, evaluate it on Chebyshev points
                    u_vals = u(x_grid)
                else:
                    # If u is already an array, use it directly
                    u_vals = np.asarray(u)

                # Ensure we have the right number of points
                if len(u_vals) != dim:
                    raise ValueError(
                        f"Function values must have length {dim}, "
                        f"got {len(u_vals)}"
                    )

                # Simple Chebyshev transform (can be improved with proper DCT)
                return dct(u_vals, type=1, norm='ortho')

            def from_coeff(coeff):
                """Convert Chebyshev coefficients back to function values."""
                coeff = np.asarray(coeff)
                if len(coeff) != dim:
                    raise ValueError(
                        f"Coefficients must have length {dim}, "
                        f"got {len(coeff)}"
                    )

                return idct(coeff, type=1, norm='ortho')

            def scaling(k):
                return (1 + (scale * k) ** 2) ** order

        else:
            raise ValueError(f"Unknown basis type: {basis_type}")

        return Sobolev(
            dim, to_coeff, from_coeff, scaling, interval=interval, order=order
        )

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
