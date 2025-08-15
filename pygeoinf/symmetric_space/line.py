"""
Sobolev spaces for functions on a line.
"""

from __future__ import annotations
from typing import Callable, Any, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np


from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pygeoinf.operators import LinearOperator
from pygeoinf.forms import LinearForm
from pygeoinf.gaussian_measure import GaussianMeasure

from pygeoinf.operators import LinearOperator
from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.symmetric_space.symmetric_space import SymmetricSpaceSobolev
from pygeoinf.symmetric_space.circle import Sobolev as CircleSobolev


class Sobolev(SymmetricSpaceSobolev):
    """
    Implementation of the Sobolev space H^s on a finite line interval.

    This class models functions on an interval [x0, x1] by mapping the problem
    to a periodic domain (a circle) with padding. This avoids explicit
    boundary conditions by using smooth tapers.
    """

    def __init__(
        self,
        kmax: int,
        order: float,
        scale: float,
        /,
        *,
        x0: float = 0.0,
        x1: float = 1.0,
    ) -> None:
        """
        Args:
            kmax: The maximum Fourier degree for the underlying circle representation.
            order: The Sobolev order, controlling function smoothness.
            scale: The Sobolev length-scale.
            x0: The left boundary of the interval. Defaults to 0.0.
            x1: The right boundary of the interval. Defaults to 1.0.

        Raises:
            ValueError: If `x0 >= x1` or if `scale <= 0` when `order` is non-zero.
        """

        if x0 >= x1:
            raise ValueError("Invalid interval parameters: x0 must be less than x1.")
        if order != 0 and scale <= 0:
            raise ValueError("Length-scale must be positive for non-L2 spaces.")

        self._kmax: int = kmax
        self._x0: float = x0
        self._x1: float = x1

        # Work out the padding.
        padding_scale: float = 5 * scale if scale > 0 else 0.1 * (x1 - x0)
        number_of_points: int = 2 * kmax
        width: float = x1 - x0
        self._start_index: int = int(
            number_of_points * padding_scale / (width + 2 * padding_scale)
        )
        self._finish_index: int = 2 * kmax - self._start_index + 1
        self._padding_length: float = (
            self._start_index * width / (number_of_points - 2 * self._start_index)
        )

        self._jac: float = (width + 2 * self._padding_length) / (2 * np.pi)
        self._ijac: float = 1.0 / self._jac
        self._sqrt_jac: float = np.sqrt(self._jac)
        self._isqrt_jac: float = 1.0 / self._sqrt_jac

        # Set up the related Sobolev space on the unit circle.
        circle_scale: float = scale * self._ijac
        self._circle_space: CircleSobolev = CircleSobolev(kmax, order, circle_scale)

        super().__init__(
            order,
            scale,
            self._circle_space.dim,
            self._to_components,
            self._from_components,
            self._inner_product,
            self._to_dual,
            self._from_dual,
            vector_multiply=lambda u1, u2: u1 * u2,
        )

    @staticmethod
    def from_sobolev_parameters(
        order: float,
        scale: float,
        /,
        *,
        x0: float = 0.0,
        x1: float = 1.0,
        rtol: float = 1e-8,
        power_of_two: bool = False,
    ) -> "Sobolev":
        """
        Creates an instance with `kmax` selected based on the Sobolev parameters.

        Args:
            order: The Sobolev order.
            scale: The Sobolev length-scale.
            x0: The left boundary of the interval. Defaults to 0.0.
            x1: The right boundary of the interval. Defaults to 1.0.
            rtol: Relative tolerance for truncation error assessment.
            power_of_two: If True, `kmax` is set to the next power of two.

        Returns:
            An instance of the Sobolev class with an appropriate `kmax`.
        """
        if x0 >= x1:
            raise ValueError("Invalid interval parameters")

        circle_scale = scale / (x1 - x0)
        circle_space = CircleSobolev.from_sobolev_parameters(
            order, circle_scale, rtol=rtol, power_of_two=power_of_two
        )
        kmax = circle_space.kmax
        return Sobolev(kmax, order, scale, x0=x0, x1=x1)

    @property
    def kmax(self) -> int:
        """The maximum Fourier degree of the underlying circle representation."""
        return self._kmax

    @property
    def x0(self) -> float:
        """The left boundary point of the interval."""
        return self._x0

    @property
    def x1(self) -> float:
        """The right boundary point of the interval."""
        return self._x1

    @property
    def width(self) -> float:
        """The width of the interval, `x1 - x0`."""
        return self._x1 - self._x0

    @property
    def point_spacing(self) -> float:
        """The spacing between grid points on the interval."""
        return self._circle_space.angle_spacing * self._jac

    def computational_points(self) -> np.ndarray:
        """Returns the grid points on the full computational domain, including padding."""
        return self._x0 - self._padding_length + self._jac * self._circle_space.angles()

    def points(self) -> np.ndarray:
        """Returns the grid points within the primary interval `[x0, x1]`."""
        return self.computational_points()[self._start_index : self._finish_index]

    def project_function(self, f: Callable[[float], float]) -> np.ndarray:
        """
        Returns an element of the space by projecting a given function.

        The function `f` is evaluated at the computational grid points and
        multiplied by a smooth tapering function.

        Args:
            f: A function that takes a position `x` and returns a value.
        """
        return np.fromiter(
            (f(x) * self._taper(x) for x in self.computational_points()), float
        )

    def random_point(self) -> float:
        """Returns a random point within the interval `[x0, x1]`."""
        return np.random.uniform(self._x0, self._x1)

    def plot(
        self,
        u: np.ndarray,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
        /,
        *,
        computational_domain: bool = False,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """
        Makes a simple plot of a function from the space.

        Args:
            u: The vector representing the function to be plotted.
            fig: An existing Matplotlib Figure object. Defaults to None.
            ax: An existing Matplotlib Axes object. Defaults to None.
            computational_domain: If True, plot the whole computational
                domain including the tapered padding. Defaults to False.
            **kwargs: Keyword arguments forwarded to `ax.plot()`.

        Returns:
            A tuple (figure, axes) containing the plot objects.
        """
        figsize = kwargs.pop("figsize", (10, 8))

        if fig is None:
            fig = plt.figure(figsize=figsize)
        if ax is None:
            ax = fig.add_subplot()

        if computational_domain:
            ax.plot(self.computational_points(), u, **kwargs)
        else:
            ax.plot(self.points(), u[self._start_index : self._finish_index], **kwargs)

        return fig, ax

    def plot_pointwise_bounds(
        self,
        u: np.ndarray,
        u_bound: np.ndarray,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
        /,
        *,
        computational_domain: bool = False,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """
        Plots a function with pointwise error bounds.

        Args:
            u: The vector representing the mean function.
            u_bound: A vector giving pointwise standard deviations.
            fig: An existing Matplotlib Figure object. Defaults to None.
            ax: An existing Matplotlib Axes object. Defaults to None.
            computational_domain: If True, plot the whole computational domain.
            **kwargs: Keyword arguments forwarded to `ax.fill_between()`.

        Returns:
            A tuple (figure, axes) containing the plot objects.
        """
        figsize = kwargs.pop("figsize", (10, 8))

        if fig is None:
            fig = plt.figure(figsize=figsize)
        if ax is None:
            ax = fig.add_subplot()

        if computational_domain:
            ax.fill_between(
                self.computational_points(), u - u_bound, u + u_bound, **kwargs
            )
        else:
            ax.fill_between(
                self.points(),
                u[self._start_index : self._finish_index]
                - u_bound[self._start_index : self._finish_index],
                u[self._start_index : self._finish_index]
                + u_bound[self._start_index : self._finish_index],
                **kwargs,
            )

        return fig, ax

    def invariant_automorphism(self, f: Callable[[float], float]) -> "LinearOperator":
        A = self._circle_space.invariant_automorphism(lambda k: f(self._ijac * k))
        return LinearOperator.formally_self_adjoint(self, A)

    def invariant_gaussian_measure(
        self, f: Callable[[float], float], /, *, expectation: Optional[Any] = None
    ) -> "GaussianMeasure":
        mu = self._circle_space.invariant_gaussian_measure(
            lambda k: f(self._ijac * k), expectation=expectation
        )
        covariance = LinearOperator.self_adjoint(self, mu.covariance)
        return GaussianMeasure(
            covariance=covariance, expectation=expectation, sample=mu.sample
        )

    def dirac(self, point: float) -> "LinearForm":
        theta = self._inverse_transformation(point)
        up = self._circle_space.dirac(theta)
        cp = self._circle_space.dual.to_components(up) * self._isqrt_jac
        return self.dual.from_components(cp)

    def __eq__(self, other: object) -> bool:
        """
        Checks for mathematical equality with another Sobolev space on a line.

        Two spaces are considered equal if they are of the same type and have
        the same defining parameters (kmax, order, scale, x0, and x1).
        """
        if not isinstance(other, Sobolev):
            return NotImplemented
    
        return (self.kmax == other.kmax and
                self.order == other.order and
                self.scale == other.scale and
                self.x0 == other.x0 and
                self.x1 == other.x1)


    # =============================================================#
    #                        Private methods                       #
    # =============================================================#

    def _step(self, x: float) -> float:
        if x > 0:
            return np.exp(-1.0 / x)
        else:
            return 0.0

    def _bump_up(self, x: float, x1: float, x2: float) -> float:
        s1 = self._step(x - x1)
        s2 = self._step(x2 - x)
        return s1 / (s1 + s2)

    def _bump_down(self, x: float, x1: float, x2: float) -> float:
        s1 = self._step(x2 - x)
        s2 = self._step(x - x1)
        return s1 / (s1 + s2)

    def _taper(self, x: float) -> float:
        s1 = self._bump_up(x, self._x0 - self._padding_length, self._x0)
        s2 = self._bump_down(x, self._x1, self._x1 + self._padding_length)
        return s1 * s2

    def _transformation(self, th: float) -> float:
        return self._x0 - self._padding_length + self._jac * th

    def _inverse_transformation(self, x: float) -> float:
        return (x - self._x0 + self._padding_length) * self._ijac

    def _to_components(self, u: np.ndarray) -> np.ndarray:
        c = self._circle_space.to_components(u)
        c *= self._sqrt_jac
        return c

    def _from_components(self, c: np.ndarray) -> np.ndarray:        
        u = self._circle_space.from_components(c)
        u *= self._isqrt_jac
        return u

    def _inner_product(self, u1: np.ndarray, u2: np.ndarray) -> float:
        return self._jac * self._circle_space.inner_product(u1, u2)

    def _to_dual(self, u: np.ndarray) -> "LinearForm":
        up = self._circle_space.to_dual(u)
        cp = self._circle_space.dual.to_components(up) * self._sqrt_jac
        return self.dual.from_components(cp)

    def _from_dual(self, up: "LinearForm") -> np.ndarray:
        cp = self.dual.to_components(up)
        vp = self._circle_space.dual.from_components(cp) * self._isqrt_jac
        return self._circle_space.from_dual(vp)


class Lebesgue(Sobolev):
    """
    Implementation of the Lebesgue space L^2 on a line.

    This is a special case of the Sobolev space with order `s=0`.
    """

    def __init__(
        self,
        kmax: int,
        /,
        *,
        x0: float = 0.0,
        x1: float = 1.0,
    ) -> None:
        """
        Args:
            kmax: The maximum Fourier degree for the underlying representation.
            x0: The left boundary of the interval. Defaults to 0.0.
            x1: The right boundary of the interval. Defaults to 1.0.
        """
        super().__init__(kmax, 0.0, 1.0, x0=x0, x1=x1)
