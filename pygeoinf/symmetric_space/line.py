from __future__ import annotations
from typing import Callable, Any, Optional, Tuple, List

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
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

        AbstractSymmetricLebesgueSpace.__init__(self, 1, 2 * kmax, False)

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

    # ------------------------------------------------------ #
    #                     Public methods                     #
    # ------------------------------------------------------ #

    def points(self) -> np.ndarray:
        """Returns a numpy array of the grid points."""
        return np.fromiter(
            [self.angle_to_point(th) for th in self._circle_space.angles()],
            float,
        )

    def project_function(self, f: Callable[float, float]) -> np.ndarray:
        """
        Returns an element of the space by projecting a given function.

        The function `f` is evaluated at the grid points of the space.

        Args:
            f: A function that takes an angle (float) and returns a value.
        """
        return np.fromiter((f(x) for x in self.points()), float)

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
    ):
        """
        Args:
        kmax: The maximum Fourier degree to be represented.
        order: The Sobolev order, controlling the smoothness of functions.
        scale: The Sobolev length-scale.
        a: The left endpoint of the interval. Defaults to 0.0
        b: The right endpoint of the interval. Dafaults to 0.1
        c: The padding distance for the computational domain.
               Defaults to 5*scale
        """

        c = 5 * scale if c is None else c
        lebesgue_space = Lebesgue(kmax, a=a, b=b, c=c)
        SymmetricSobolevSpace.__init__(self, lebesgue_space, order, scale)

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
    ) -> "Sobolev":
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
               Defaults to 5*scale
            rtol: Relative tolerance used in assessing truncation error.
                Defaults to 1e-8.
            power_of_two: If True, `kmax` is set to the next power of two.

        Returns:
            An instance of the Sobolev class with an appropriate `kmax`.

        Raises:
            ValueError: If order is <= 0.5.
        """
        c = 5 * scale if c is None else c
        length = b - a + 2 * c
        radius = length / (2 * np.pi)
        circle_space = CircleSobolev.from_sobolev_parameters(
            order, scale, radius=radius, rtol=rtol, power_of_two=power_of_two
        )

        return Sobolev(circle_space.kmax, order, scale, a=a, b=b, c=c)

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
        codomain = Sobolev(
            self.kmax, self.order - 1, self.scale, a=self.a, b=self.b, c=self.c
        )

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


def plot(
    space: Lebesgue | Sobolev,
    u: np.ndarray,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    full: Optional[bool] = False,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """
    Makes a simple plot of a function on the circle.

    Args:
        space: The function space.
        u: The vector representing the function to be plotted.
        fig: An existing Matplotlib Figure object. Defaults to None.
        ax: An existing Matplotlib Axes object. Defaults to None.
        **kwargs: Keyword arguments forwarded to `ax.plot()`.

    Returns:
        A tuple (figure, axes) containing the plot objects.
    """
    figsize = kwargs.pop("figsize", (10, 8))

    if fig is None:
        fig = plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.add_subplot()

    ax.plot(space.points(), u, **kwargs)
    if not full:
        ax.set_xlim(left=space.a, right=space.b)
    return fig, ax


def plot_error_bounds(
    space: Lebesgue | Sobolev,
    u: np.ndarray,
    u_bound: np.ndarray,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    full: Optional[bool] = False,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """
    Plots a function on the circle along with its pointwise error bounds.

    This is particularly useful for visualizing Gaussian measures or Bayesian
    posterior uncertainties over the circular domain.

    Args:
        space: The function space.
        u: A 1D numpy array representing the mean function values.
        u_bound: A 1D numpy array giving the pointwise standard deviations or bounds.
        fig: An existing Matplotlib Figure object. If None, a new figure is created.
        ax: An existing Matplotlib Axes object. If None, a new subplot is added.
        **kwargs: Additional keyword arguments forwarded directly to `ax.fill_between()`
            (e.g., `alpha`, `color`).

    Returns:
        A tuple `(fig, ax)` containing the Matplotlib Figure and Axes objects.
    """
    figsize = kwargs.pop("figsize", (10, 8))

    if fig is None:
        fig = plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.add_subplot()

    ax.fill_between(space.points(), u - u_bound, u + u_bound, **kwargs)
    if not full:
        ax.set_xlim(left=space.a, right=space.b)
    return fig, ax
