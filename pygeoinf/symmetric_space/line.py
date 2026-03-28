from __future__ import annotations
from typing import Callable, Any, Optional, Tuple, List

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from pygeoinf.linear_operators import LinearOperator
from .symmetric_space import AbstractSymmetricLebesgueSpace, SymmetricSobolevSpace
from .circle import Lebesgue as CircleLebesgue
from .circle import Sobolev as CircleSobolev


class Lebesgue(AbstractSymmetricLebesgueSpace):
    """
    Implementation of the Lebesgue space L² on a line of interval.
    In the case of the line, it is assumed that the function has
    support limited to the chosen interval.

    A function is
    represented by its values on an evenly spaced grid. The co-ordinate basis for
    the space is through Fourier expansions. Details of the implementation are
    handled by mapping the function to one defined on a circle.
    """

    def __init__(self, kmax: int, /, *, a: float = 0.0, b: float = 1.0, c: float = 0.1):
        """
        Args:
        kmax: The maximum Fourier degree to be represented.
        a: The left endpoint of the interval. Defaults to 0.0
        b: The right endpoint of the interval. Dafaults to 0.1
        c: The padding distance for the computational domain.
               Defaults to 0.1
        """
        self._a = a
        self._b = b
        self._c = c

        length = b - a + 2 * c
        radius = length / (2 * np.pi)
        self._circle_space = CircleLebesgue(kmax, radius=radius)

        AbstractSymmetricLebesgueSpace.__init__(self, 1, kmax, 2 * kmax, False)

    @property
    def kmax(self) -> int:
        """
        Returns the maximum Fourier degree.
        """
        return self._circle_space.kmax

    @property
    def a(self) -> float:
        """
        Returns the left end point.
        """
        return self._a

    @property
    def b(self) -> float:
        """
        Returns the right end point.
        """
        return self._b

    @property
    def c(self) -> float:
        """
        Returns the padding distance.
        """
        return self._c

    @property
    def circle_space(self):
        """
        Returns the isomorphic space of functions on a circle.
        """
        return self._circle_space

    @classmethod
    def from_covariance(
        cls,
        covariance_function: Callable[[float], float],
        /,
        *,
        a: float = 0.0,
        b: float = 1.0,
        c: float = 0.1,
        rtol: float = 1e-6,
        min_degree: int = 0,
        max_degree: Optional[int] = None,
        power_of_two: bool = False,
    ) -> Lebesgue:
        """
        Creates an instance of the L² space with a Fourier truncation degree (`kmax`)
        automatically chosen to capture the expected energy of functions drawn from
        a specified prior measure.

        This factory method calculates the expected squared norm (energy) of a random field
        whose spectral variances are defined by the provided `covariance_function`. It iteratively
        adds higher frequency modes until the relative contribution of the next degree drops
        below the specified relative tolerance.

        Args:
            covariance_function: A callable mapping a Laplacian eigenvalue to its spectral variance.
            a: The left endpoint of the interval. Defaults to 0.0.
            b: The right endpoint of the interval. Defaults to 1.0.
            c: The padding distance for the computational domain. Defaults to 0.1.
            rtol: The relative tolerance for the energy truncation. Defaults to 1e-6.
            min_degree: The absolute minimum truncation degree to return. Defaults to 0.
            max_degree: An optional safety ceiling for the truncation degree. If convergence
                is not reached by this degree, the search terminates and returns this value.
            power_of_two: If True, rounds the resulting `kmax` up to the nearest
                power of two (useful for FFT optimizations). Defaults to False.

        Returns:
            Lebesgue: A fully instantiated L² space on the line with the optimal `kmax`.

        Raises:
            RuntimeError: If the energy sequence fails to converge within 100,000 iterations
                and no `max_degree` is specified.
        """
        dummy_space = cls(max(1, min_degree), a=a, b=b, c=c)

        optimal_degree = dummy_space.estimate_truncation_degree(
            covariance_function, rtol=rtol, min_degree=min_degree, max_degree=max_degree
        )

        if power_of_two:
            n = int(np.log2(optimal_degree))
            optimal_degree = 2 ** (n + 1)

        return cls(optimal_degree, a=a, b=b, c=c)

    @classmethod
    def from_heat_kernel_prior(
        cls,
        kernel_scale: float,
        /,
        *,
        a: float = 0.0,
        b: float = 1.0,
        c: float = 0.1,
        rtol: float = 1e-6,
        min_degree: int = 0,
        max_degree: Optional[int] = None,
        power_of_two: bool = False,
    ) -> Lebesgue:
        """
        Creates an instance of the L² space on the line, tuned to the expected
        energy of a Heat Kernel prior measure.

        Args:
            kernel_scale: The length-scale parameter of the heat kernel covariance.
            a: The left endpoint of the interval. Defaults to 0.0.
            b: The right endpoint of the interval. Defaults to 1.0.
            c: The padding distance for the computational domain. Defaults to 0.1.
            rtol: The relative tolerance for the energy truncation. Defaults to 1e-6.
            min_degree: The absolute minimum truncation degree to return. Defaults to 0.
            max_degree: An optional safety ceiling for the truncation degree.
            power_of_two: If True, rounds the resulting `kmax` up to the nearest power of two.
        """
        return cls.from_covariance(
            cls.heat_kernel(kernel_scale),
            a=a,
            b=b,
            c=c,
            rtol=rtol,
            min_degree=min_degree,
            max_degree=max_degree,
            power_of_two=power_of_two,
        )

    @classmethod
    def from_sobolev_kernel_prior(
        cls,
        kernel_order: float,
        kernel_scale: float,
        /,
        *,
        a: float = 0.0,
        b: float = 1.0,
        c: float = 0.1,
        rtol: float = 1e-6,
        min_degree: int = 0,
        max_degree: Optional[int] = None,
        power_of_two: bool = False,
    ) -> Lebesgue:
        """
        Creates an instance of the L² space on the line, tuned to the expected
        energy of a Sobolev-type prior measure.

        Args:
            kernel_order: The smoothness order of the Sobolev prior measure.
            kernel_scale: The length-scale parameter of the Sobolev prior measure.
            a: The left endpoint of the interval. Defaults to 0.0.
            b: The right endpoint of the interval. Defaults to 1.0.
            c: The padding distance for the computational domain. Defaults to 0.1.
            rtol: The relative tolerance for the energy truncation. Defaults to 1e-6.
            min_degree: The absolute minimum truncation degree to return. Defaults to 0.
            max_degree: An optional safety ceiling for the truncation degree.
            power_of_two: If True, rounds the resulting `kmax` up to the nearest power of two.
        """
        return cls.from_covariance(
            cls.sobolev_kernel(kernel_order, kernel_scale),
            a=a,
            b=b,
            c=c,
            rtol=rtol,
            min_degree=min_degree,
            max_degree=max_degree,
            power_of_two=power_of_two,
        )

    # ------------------------------------------------------ #
    #                     Public methods                     #
    # ------------------------------------------------------ #

    def points(self) -> np.ndarray:
        """Returns a numpy array of the grid points."""
        return np.fromiter(
            [self.angle_to_point(th) for th in self._circle_space.points()],
            float,
        )

    def project_function(self, f: Callable[[float], float]) -> np.ndarray:
        """
        Returns an element of the space by projecting a given function.

        The function `f` is evaluated at the grid points of the space. To prevent
        spectral ringing (Gibbs phenomenon) from non-periodic functions, the
        function is smoothly tapered to zero within the padding regions
        [a-c, a] and [b, b+c] using a raised cosine window.

        Args:
            f: A function that takes a point (float) and returns a value.
        """
        points = self.points()

        vals = np.fromiter((f(x) for x in points), float)

        mask = np.ones_like(points)

        left_idx = (points < self.a) & (points >= self.a - self._c)
        if np.any(left_idx):
            x_norm_left = (points[left_idx] - (self.a - self._c)) / self._c
            mask[left_idx] = 0.5 * (1 - np.cos(np.pi * x_norm_left))

        right_idx = (points > self.b) & (points <= self.b + self._c)
        if np.any(right_idx):
            x_norm_right = (points[right_idx] - self.b) / self._c
            mask[right_idx] = 0.5 * (1 + np.cos(np.pi * x_norm_right))

        out_of_bounds = (points < self.a - self._c) | (points > self.b + self._c)
        mask[out_of_bounds] = 0.0

        return vals * mask

    def to_coefficients(self, u: np.ndarray) -> np.ndarray:
        """Maps a function vector to its complex Fourier coefficients."""
        return self._circle_space.to_coefficients(u)

    def from_coefficients(self, coeff: np.ndarray) -> np.ndarray:
        """Maps complex Fourier coefficients to a function vector."""
        return self._circle_space.from_coefficients(coeff)

    def point_to_angle(self, x: float) -> float:
        """Maps a point to the corresponding angle."""
        return (x - self.a + self._c) / self._circle_space.radius

    def angle_to_point(self, th: float) -> float:
        """Maps an angle to the corresponding point."""
        return self.a - self._c + self._circle_space.radius * th

    # ------------------------------------------------------ #
    #           Methods for SymmetricHilbertSpace            #
    # ------------------------------------------------------ #

    def integer_to_index(self, i: int) -> int:
        return self._circle_space.integer_to_index(i)

    def index_to_integer(self, k: int) -> int:
        return self._circle_space.index_to_integer(k)

    def laplacian_eigenvalue(self, k: int) -> float:
        return self._circle_space.laplacian_eigenvalue(k)

    def laplacian_eigenvector_squared_norm(self, k: int) -> float:
        return self._circle_space.laplacian_eigenvector_squared_norm(k)

    def laplacian_eigenvectors_at_point(self, x: float) -> np.ndarray:
        th = self.point_to_angle(x)
        return self._circle_space.laplacian_eigenvectors_at_point(th)

    def random_point(self) -> float:
        return np.random.uniform(self.a, self.b)

    def geodesic_distance(self, p1: float, p2: float) -> float:
        return abs(p2 - p1)

    def geodesic_quadrature(
        self, p1: float, p2: float, n_points: int
    ) -> Tuple[List[float], np.ndarray]:

        th1 = self.point_to_angle(p1)
        th2 = self.point_to_angle(p2)
        circle_points, weights = self._circle_space.geodesic_quadrature(
            th1, th2, n_points=n_points
        )

        points = [self.angle_to_point(th) for th in circle_points]

        return points, weights

    def with_degree(self, degree: int) -> Lebesgue:
        return Lebesgue(degree, a=self.a, b=self.b, c=self.c)

    def degree_transfer_operator(self, target_degree: int) -> LinearOperator:
        """
        Returns the transfer operator from this space to one with a different degree.
        """
        codomain = self.with_degree(target_degree)

        circle_transfer = self._circle_space.degree_transfer_operator(target_degree)

        def mapping(u: np.ndarray) -> np.ndarray:
            return circle_transfer(u)

        def adjoint_mapping(v: np.ndarray) -> np.ndarray:
            return circle_transfer.adjoint(v)

        return LinearOperator(self, codomain, mapping, adjoint_mapping=adjoint_mapping)

    def invariant_covariance_function(
        self, spectral_variances: np.ndarray
    ) -> Callable[[np.ndarray], np.ndarray]:
        return self._circle_space.invariant_covariance_function(spectral_variances)

    def degree_multiplicity(self, degree: int) -> int:
        return self._circle_space.degree_multiplicity(degree)

    def representative_index(self, degree: int) -> int:
        return self._circle_space.representative_index(degree)

    # ------------------------------------------------------ #
    #                 Methods for HilbertSpace               #
    # ------------------------------------------------------ #

    def to_components(self, x: np.ndarray) -> np.ndarray:
        return self._circle_space.to_components(x)

    def from_components(self, c: np.ndarray) -> np.ndarray:
        return self._circle_space.from_components(c)

    def is_element(self, x: Any) -> bool:
        return self._circle_space.is_element(x)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Lebesgue):
            return NotImplemented

        return (
            self.kmax == other.kmax
            and self.a == other.a
            and self.b == other.b
            and self._c == other._c
        )

    # ------------------------------------------------------ #
    #                 Methods for HilbertModule              #
    # ------------------------------------------------------ #

    def vector_multiply(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return self._circle_space.vector_multiply(x1, x2)

    def vector_sqrt(self, x: np.ndarray) -> np.ndarray:
        return self._circle_space.vector_sqrt(x)


class Sobolev(SymmetricSobolevSpace):
    """
    Implementation of the Sobolev space Hˢ on a circle.
    """

    def __init__(
        self,
        kmax: int,
        order: float,
        scale: float,
        /,
        *,
        a: float = 0.0,
        b: float = 1.0,
        c: float = None,
        safe: bool = True,
    ):
        """
        Args:
        kmax: The maximum Fourier degree to be represented.
        order: The Sobolev order, controlling the smoothness of functions.
        scale: The Sobolev length-scale.
        a: The left endpoint of the interval. Defaults to 0.0
        b: The right endpoint of the interval. Dafaults to 0.1
        c: The padding distance for the computational domain.
               Defaults to 6*scale
        safe: If true, the class checks for mathematical correctness of operations
                  where possible.
        """

        c = 6 * scale if c is None else c
        lebesgue_space = Lebesgue(kmax, a=a, b=b, c=c)
        SymmetricSobolevSpace.__init__(self, lebesgue_space, order, scale, safe=safe)

    @staticmethod
    def from_sobolev_parameters(
        order: float,
        scale: float,
        /,
        *,
        a: float = 0.0,
        b: float = 1.0,
        c: float = None,
        rtol: float = 1e-6,
        power_of_two: bool = False,
        safe: bool = True,
    ) -> Sobolev:
        """
        Creates an instance with `kmax` chosen based on Sobolev parameters.

        The method estimates the truncation error for the Dirac measure and is
        only applicable for spaces with order > 0.5.

        Args:
            order: The Sobolev order. Must be > 0.5.
            scale: The Sobolev length-scale.
            a: The left endpoint of the interval. Defaults to 0.0
            b: The right endpoint of the interval. Dafaults to 0.1
            c: The padding distance for the computational domain.
               Defaults to 6*scale
            rtol: Relative tolerance used in assessing truncation error.
                Defaults to 1e-8.
            power_of_two: If True, `kmax` is set to the next power of two.
            safe: If true, the class checks for mathematical correctness of operations
                  where possible.

        Returns:
            An instance of the Sobolev class with an appropriate `kmax`.

        Raises:
            ValueError: If order is <= 0.5.
        """
        c = 6 * scale if c is None else c
        length = b - a + 2 * c
        radius = length / (2 * np.pi)
        circle_space = CircleSobolev.from_sobolev_parameters(
            order, scale, radius=radius, rtol=rtol, power_of_two=power_of_two
        )

        return Sobolev(circle_space.kmax, order, scale, a=a, b=b, c=c, safe=safe)

    @classmethod
    def from_covariance(
        cls,
        covariance_function: Callable[[float], float],
        order: float,
        scale: float,
        /,
        *,
        a: float = 0.0,
        b: float = 1.0,
        c: Optional[float] = None,
        rtol: float = 1e-6,
        min_degree: int = 0,
        max_degree: Optional[int] = None,
        power_of_two: bool = False,
        safe: bool = True,
    ) -> Sobolev:
        """
        Creates an instance of the Sobolev space with a Fourier truncation degree (`kmax`)
        automatically chosen to capture the expected energy of functions drawn from
        a specified prior measure.

        This factory method calculates the expected squared norm (energy) of a random field
        whose spectral variances are defined by the provided `covariance_function`, accounting
        for the Sobolev mass-weighting factor. It iteratively adds higher frequency modes
        until the relative contribution of the next degree drops below the specified tolerance.

        Args:
            covariance_function: A callable mapping a Laplacian eigenvalue to its spectral variance.
            order: The Sobolev order, controlling the smoothness of functions.
            scale: The Sobolev length-scale.
            a: The left endpoint of the interval. Defaults to 0.0.
            b: The right endpoint of the interval. Defaults to 1.0.
            c: The padding distance for the computational domain. Defaults to 6 * scale.
            rtol: The relative tolerance for the energy truncation. Defaults to 1e-6.
            min_degree: The absolute minimum truncation degree to return. Defaults to 0.
            max_degree: An optional safety ceiling for the truncation degree. If convergence
                is not reached by this degree, the search terminates and returns this value.
            power_of_two: If True, rounds the resulting `kmax` up to the nearest
                power of two (useful for FFT optimizations). Defaults to False.
            safe: If True, enables mathematical correctness checks during operations.

        Returns:
            Sobolev: A fully instantiated Sobolev space on the line with the optimal `kmax`.

        Raises:
            RuntimeError: If the energy sequence fails to converge within 100,000 iterations
                and no `max_degree` is specified.
        """
        dummy_space = cls(max(1, min_degree), order, scale, a=a, b=b, c=c, safe=safe)

        optimal_degree = dummy_space.estimate_truncation_degree(
            covariance_function, rtol=rtol, min_degree=min_degree, max_degree=max_degree
        )

        if power_of_two:
            n = int(np.log2(optimal_degree))
            optimal_degree = 2 ** (n + 1)

        return cls(optimal_degree, order, scale, a=a, b=b, c=c, safe=safe)

    @classmethod
    def from_heat_kernel_prior(
        cls,
        kernel_scale: float,
        order: float,
        scale: float,
        /,
        *,
        a: float = 0.0,
        b: float = 1.0,
        c: Optional[float] = None,
        rtol: float = 1e-6,
        min_degree: int = 0,
        max_degree: Optional[int] = None,
        power_of_two: bool = False,
        safe: bool = True,
    ) -> Sobolev:
        """
        Creates an instance of the Sobolev space on the line, tuned to the expected
        energy of a Heat Kernel prior measure.

        Args:
            kernel_scale: The length-scale parameter of the heat kernel covariance.
            order: The Sobolev order defining the function space.
            scale: The Sobolev length-scale defining the function space.
            a: The left endpoint of the interval. Defaults to 0.0.
            b: The right endpoint of the interval. Defaults to 1.0.
            c: The padding distance for the computational domain. Defaults to 6 * scale.
            rtol: The relative tolerance for the energy truncation. Defaults to 1e-6.
            min_degree: The absolute minimum truncation degree to return. Defaults to 0.
            max_degree: An optional safety ceiling for the truncation degree.
            power_of_two: If True, rounds the resulting `kmax` up to the nearest power of two.
            safe: If True, enables mathematical correctness checks during operations.
        """
        return cls.from_covariance(
            cls.heat_kernel(kernel_scale),
            order,
            scale,
            a=a,
            b=b,
            c=c,
            rtol=rtol,
            min_degree=min_degree,
            max_degree=max_degree,
            power_of_two=power_of_two,
            safe=safe,
        )

    @classmethod
    def from_sobolev_kernel_prior(
        cls,
        kernel_order: float,
        kernel_scale: float,
        order: float,
        scale: float,
        /,
        *,
        a: float = 0.0,
        b: float = 1.0,
        c: Optional[float] = None,
        rtol: float = 1e-6,
        min_degree: int = 0,
        max_degree: Optional[int] = None,
        power_of_two: bool = False,
        safe: bool = True,
    ) -> Sobolev:
        """
        Creates an instance of the Sobolev space on the line, tuned to the expected
        energy of a Sobolev-type prior measure.

        Args:
            kernel_order: The smoothness order of the Sobolev prior measure.
            kernel_scale: The length-scale parameter of the Sobolev prior measure.
            order: The Sobolev order defining the function space.
            scale: The Sobolev length-scale defining the function space.
            a: The left endpoint of the interval. Defaults to 0.0.
            b: The right endpoint of the interval. Defaults to 1.0.
            c: The padding distance for the computational domain. Defaults to 6 * scale.
            rtol: The relative tolerance for the energy truncation. Defaults to 1e-6.
            min_degree: The absolute minimum truncation degree to return. Defaults to 0.
            max_degree: An optional safety ceiling for the truncation degree.
            power_of_two: If True, rounds the resulting `kmax` up to the nearest power of two.
            safe: If True, enables mathematical correctness checks during operations.
        """
        return cls.from_covariance(
            cls.sobolev_kernel(kernel_order, kernel_scale),
            order,
            scale,
            a=a,
            b=b,
            c=c,
            rtol=rtol,
            min_degree=min_degree,
            max_degree=max_degree,
            power_of_two=power_of_two,
            safe=safe,
        )

    # ---------------------------------------------- #
    #                   Properties                   #
    # -----------------------------------------------#

    @property
    def kmax(self) -> int:
        """
        Returns the maximum Fourier degree.
        """
        return self.underlying_space.kmax

    @property
    def a(self) -> float:
        """
        Returns the left end point.
        """
        return self.underlying_space.a

    @property
    def b(self) -> float:
        """
        Returns the right end point.
        """
        return self.underlying_space.b

    @property
    def c(self) -> float:
        """
        Returns the padding distance.
        """
        return self.underlying_space.c

    @property
    def circle_space(self):
        """
        Returns the isomorphic space of functions on a circle.
        """
        return CircleSobolev(
            self.kmax,
            self.order,
            self.scale,
            radius=self.underlying_space.circle_space.radius,
        )

    @property
    def derivative_operator(self):
        """
        Returns the derivative operator from the space to one with a lower order.
        """
        circle_op = self.circle_space.derivative_operator

        # Refactored: We no longer need to manually pass a, b, and c
        codomain = self.with_order(self.order - 1)

        # Calculate the chain rule scaling factor
        scaling = (2 * np.pi) / self.underlying_space.circle_space.radius

        # Scale the forward and adjoint mappings
        return LinearOperator(
            self,
            codomain,
            lambda u: scaling * circle_op(u),
            adjoint_mapping=lambda u: scaling * circle_op.adjoint(u),
        )

    # ---------------------------------------------- #
    #                 Public methods                 #
    # -----------------------------------------------#

    def with_order(self, order: float) -> Sobolev:
        return Sobolev(self.kmax, order, self.scale, a=self.a, b=self.b, c=self.c)

    def with_degree(self, degree: int) -> Sobolev:
        return Sobolev(degree, self.order, self.scale, a=self.a, b=self.b, c=self.c)

    def points(self) -> np.ndarray:
        """Returns a numpy array of the grid points."""
        return self.underlying_space.points()

    def project_function(self, f: Callable[float, float]) -> np.ndarray:
        """
        Returns an element of the space by projecting a given function.

        The function `f` is evaluated at the grid points of the space.

        Args:
            f: A function that takes an angle (float) and returns a value.
        """
        return self.underlying_space.project_function(f)

    def to_coefficients(self, u: np.ndarray) -> np.ndarray:
        """Maps a function vector to its complex Fourier coefficients."""
        return self.underlying_space.to_coefficients(u)

    def from_coefficients(self, coeff: np.ndarray) -> np.ndarray:
        """Maps complex Fourier coefficients to a function vector."""
        return self.underlying_space.from_coefficients(coeff)

    def point_to_angle(self, x: float) -> float:
        """
        Maps a point to the corresponding angle.
        """
        return self.underlying_space.point_to_angle(x)

    def angle_to_point(self, th: float) -> float:
        """
        Maps an angle to the corresponding point.
        """
        return self.underlying_space.angle_to_point(th)


# ------------------------------------------------- #
#           Associated plotting functions           #
# ------------------------------------------------- #


def plot(
    space: Lebesgue | Sobolev,
    u: np.ndarray,
    ax: Optional[Axes] = None,
    full: Optional[bool] = False,
    **kwargs,
) -> Axes:
    """
    Makes a simple plot of a function on the line/interval.

    Args:
        space: The function space.
        u: The vector representing the function to be plotted.
        ax: An existing Matplotlib Axes object. If None, plots to the current active axes.
        full: If False, limits the x-axis to the interval [a, b]. Defaults to False.
        **kwargs: Keyword arguments forwarded to `ax.plot()`.

    Returns:
        The Matplotlib Axes object.
    """
    if ax is None:
        ax = plt.gca()

    ax.plot(space.points(), u, **kwargs)
    if not full:
        ax.set_xlim(left=space.a, right=space.b)

    return ax


def plot_error_bounds(
    space: Lebesgue | Sobolev,
    u: np.ndarray,
    u_bound: np.ndarray,
    ax: Optional[Axes] = None,
    full: Optional[bool] = False,
    **kwargs,
) -> Axes:
    """
    Plots a function on the line along with its pointwise error bounds.

    Args:
        space: The function space.
        u: A 1D numpy array representing the mean function values.
        u_bound: A 1D numpy array giving the pointwise standard deviations or bounds.
        ax: An existing Matplotlib Axes object. If None, plots to the current active axes.
        full: If False, limits the x-axis to the interval [a, b]. Defaults to False.
        **kwargs: Additional keyword arguments forwarded directly to `ax.fill_between()`.

    Returns:
        The Matplotlib Axes object.
    """
    if ax is None:
        ax = plt.gca()

    ax.fill_between(space.points(), u - u_bound, u + u_bound, **kwargs)
    if not full:
        ax.set_xlim(left=space.a, right=space.b)

    return ax
