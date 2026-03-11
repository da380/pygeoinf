from __future__ import annotations
from typing import Callable, Any, Optional, Tuple, List

import numpy as np
from scipy.fft import rfft, irfft

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from pygeoinf.linear_operators import LinearOperator
from .symmetric_space import AbstractSymmetricLebesgueSpace, SymmetricSobolevSpace
from .circle import Lebesgue as CircleLebesgue


class LineHelper:

    def __init__(self, kmax: int, a: float, b: float, delta: float):
        """
        Args:
            kmax: The maximum Fourier degree to be represented.
            a: The left endpoint of the interval
            b: The right endpoint of the interval
            delta: The padding distance for the computational domain.

        Notes:
            The computational domain used is [a - delta, b + delta] so
            that periodicity of the functions is broken.
        """

        if kmax <= 0:
            raise ValueError("kmax must be non-negative")

        if b <= a:
            raise ValueError("The interval must have b > a")

        if delta < 0:
            raise ValueError("The padding distance must be non-negative")

        self._kmax: int = kmax
        self._a = a
        self._b = b
        self._delta = delta
        self._radius: float = b - a + 2 * delta

        self._fft_factor: float = np.sqrt(2 * np.pi * self._radius) / (2 * self.kmax)
        self._inverse_fft_factor: float = 1.0 / self._fft_factor

    @property
    def kmax(self) -> int:
        """
        The maximum Fourier degree represented in this space.
        """
        return self._kmax

    @property
    def a(self) -> float:
        """
        The left endpoint of the interval.
        """
        return self._a

    @property
    def b(self) -> float:
        """
        The right endpoint of the interval.
        """
        return self._b

    @property
    def delta(self) -> float:
        """
        The padding distance for the computational domain.
        """
        return self._delta

    @property
    def point_spacing(self) -> float:
        """
        The distance between grid points
        """
        return self._radius / (2 * self.kmax)

    @property
    def fft_factor(self) -> float:
        """
        The factor by which the Fourier coefficients are scaled
        in forward transformations.
        """
        return self._fft_factor

    @property
    def inverse_fft_factor(self) -> float:
        """
        The factor by which the Fourier coefficients are scaled
        in inverse transformations.
        """
        return self._inverse_fft_factor

    def points(self) -> np.ndarray:
        """
        Returns a numpy array of the grid points.
        """
        return np.fromiter(
            [
                self.a - self.delta + i * self.point_spacing
                for i in range(2 * self.kmax)
            ],
            float,
        )

    def project_function(self, f: Callable[[float], float]) -> np.ndarray:
        """
        Returns an element of the space by projecting a given function.

        The function `f` is evaluated at the grid points of the space.

        Args:
            f: A function that takes an angle (float) and returns a value.
        """
        return np.fromiter((f(x) for x in self.points()), float)

    def to_coefficients(self, u: np.ndarray) -> np.ndarray:
        """Maps a function vector to its complex Fourier coefficients."""
        return rfft(u) * self.fft_factor

    def from_coefficients(self, coeff: np.ndarray) -> np.ndarray:
        """Maps complex Fourier coefficients to a function vector."""
        return irfft(coeff, n=2 * self.kmax) * self._inverse_fft_factor

    def plot(
        self,
        u: np.ndarray,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """
        Makes a simple plot of a function on the interval.

        Args:
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

        ax.plot(self.points(), u, **kwargs)
        ax.set_xlim(left=self.a, right=self.b)
        return fig, ax

    def plot_error_bounds(
        self,
        u: np.ndarray,
        u_bound: np.ndarray,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """
        Plots a function with pointwise error bounds.

        Args:
            u: The vector representing the mean function.
            u_bound: A vector giving pointwise standard deviations.
            fig: An existing Matplotlib Figure object. Defaults to None.
            ax: An existing Matplotlib Axes object. Defaults to None.
            **kwargs: Keyword arguments forwarded to `ax.fill_between()`.

        Returns:
            A tuple (figure, axes) containing the plot objects.
        """
        figsize = kwargs.pop("figsize", (10, 8))

        if fig is None:
            fig = plt.figure(figsize=figsize)
        if ax is None:
            ax = fig.add_subplot()

        ax.fill_between(self.points(), u - u_bound, u + u_bound, **kwargs)
        return fig, ax

    def _to_angle(self, x: float) -> float:
        """
        Maps a point to the corresponding angle.
        """
        return (
            2 * np.pi * (x - self.a + self.delta) / (self.b - self.a + 2 * self.delta)
        )

    def _from_angle(self, th: float) -> float:
        """
        Maps an angle to the corresponding point.
        """
        return self.a - self.delta + self._radius * th / (2 * np.pi)


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

    def __init__(
        self, kmax: int, /, *, a: float = 0.0, b: float = 1.0, delta: float = 0.1
    ):
        """
        Args:
        kmax: The maximum Fourier degree to be represented.
        a: The left endpoint of the interval. Defaults to 0.0
        b: The right endpoint of the interval. Dafaults to 0.1
        delta: The padding distance for the computational domain.
               Defaults to 0.1
        """
        self._a = a
        self._b = b
        self._delta = delta
        self._circle_space = CircleLebesgue(kmax, radius=b - a + 2 * delta)

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

    # ------------------------------------------------------ #
    #                     Public methods                     #
    # ------------------------------------------------------ #

    def points(self) -> np.ndarray:
        """Returns a numpy array of the grid points."""
        return np.fromiter(
            [self._angle_to_point(th) for th in self._circle_space.angles()],
            float,
        )

    def project_function(self, f: Callable[[float], float]) -> np.ndarray:
        """
        Returns an element of the space by projecting a given function.

        The function `f` is evaluated at the grid points of the space.

        Args:
            f: A function that takes an angle (float) and returns a value.
        """
        return np.fromiter((f(x) for x in self.points()), float)

    def plot(
        self,
        u: np.ndarray,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
        full: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """
        Makes a simple plot of a function on the circle.

        Args:
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

        ax.plot(self.points(), u, **kwargs)
        if not full:
            ax.set_xlim(left=self.a, right=self.b)
        return fig, ax

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
        th = self._point_to_angle(x)
        return self._circle_space.laplacian_eigenvectors_at_point(th)

    def random_point(self) -> float:
        return np.random.uniform(self.a, self.b)

    def geodesic_quadrature(
        self, p1: float, p2: float, n_points: int
    ) -> Tuple[List[float], np.ndarray]:

        th1 = self._point_to_angle(p1)
        th2 = self._point_to_angle(p2)
        circle_points, weights = self._circle_space.geodesic_quadrature(
            th1, th2, n_points=n_points
        )

        points = [self._angle_to_point(th) for th in circle_points]

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
            and self._delta == other._delta
        )

    # ------------------------------------------------------ #
    #                 Methods for HilbertModule              #
    # ------------------------------------------------------ #

    def vector_multiply(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return self._circle_space.vector_multiply(x1, x2)

    def vector_sqrt(self, x: np.ndarray) -> np.ndarray:
        return self._circle_space.vector_sqrt(x)

    # ------------------------------------------------------ #
    #                     Private methods                    #
    # ------------------------------------------------------ #

    def _point_to_angle(self, x: float) -> float:
        """
        Maps a point to the corresponding angle.
        """
        return 2 * np.pi * (x - self.a + self._delta) / self._circle_space.radius

    def _angle_to_point(self, th: float) -> float:
        """
        Maps an angle to the corresponding point.
        """
        return self.a - self._delta + self._circle_space.radius * th / (2 * np.pi)
