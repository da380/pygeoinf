"""
Lebesgue and Sobolev spaces for functions on a circle.
"""

from __future__ import annotations

from typing import Callable, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, irfft
from scipy.sparse import diags


from matplotlib.figure import Figure
from matplotlib.axes import Axes

from pygeoinf.hilbert_space import (
    HilbertSpace,
    EuclideanSpace,
    MassWeightedHilbertSpace,
)
from pygeoinf.operators import LinearOperator
from pygeoinf.linear_forms import LinearForm
from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.symmetric_space_new.symmetric_space import SymmetricSpaceHelper


class CircleHelper:
    """
    Helper class for function spaces on the circle.
    """

    def __init__(self, kmax: int, radius: float):
        """
        Args:
            kmax: The maximum Fourier degree to be represented.
            radius: Radius of the circle.
        """
        self._kmax: int = kmax
        self._radius: float = radius

    @property
    def kmax(self):
        """The maximum Fourier degree represented in this space."""
        return self._kmax

    @property
    def radius(self) -> float:
        """The radius of the circle."""
        return self._radius

    @property
    def angle_spacing(self) -> float:
        """The angular spacing between grid points."""
        return np.pi / self.kmax

    def random_point(self) -> float:
        """Returns a random angle in the interval [0, 2*pi)."""
        return np.random.uniform(0, 2 * np.pi)

    def angles(self) -> np.ndarray:
        """Returns a numpy array of the grid point angles."""
        return np.fromiter(
            [i * self.angle_spacing for i in range(2 * self.kmax)],
            float,
        )

    def project_function(self, f: Callable[[float], float]) -> np.ndarray:
        """
        Returns an element of the space by projecting a given function.

        The function `f` is evaluated at the grid points of the space.

        Args:
            f: A function that takes an angle (float) and returns a value.
        """
        return np.fromiter((f(theta) for theta in self.angles()), float)

    def plot(
        self,
        u: np.ndarray,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
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

        ax.plot(self.angles(), u, **kwargs)
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

        ax.fill_between(self.angles(), u - u_bound, u + u_bound, **kwargs)
        return fig, ax


class Lebesgue(CircleHelper, HilbertSpace):
    """
    Implementation of the Lebesgue space L^2 on a circle based
    on Fourier expansions.
    """

    def __init__(
        self,
        kmax: int,
        /,
        *,
        radius: float = 1.0,
    ):
        """
        Args:
        kmax: The maximum Fourier degree to be represented.
        radius: Radius of the circle. Defaults to 1.0.
        """

        CircleHelper.__init__(self, kmax, radius)
        HilbertSpace.__init__(
            self,
            2 * kmax,
            self._to_components_impl,
            self._from_components_impl,
            self._to_dual_impl,
            self._from_dual_impl,
            vector_multiply=lambda u1, u2: u1 * u2,
        )

        self._fft_factor: float = np.sqrt(2 * np.pi * radius) / self.dim
        self._inverse_fft_factor: float = 1.0 / self._fft_factor

    def to_coefficient(self, u: np.ndarray) -> np.ndarray:
        """Maps a function vector to its complex Fourier coefficients."""
        return rfft(u) * self._fft_factor

    def from_coefficient(self, coeff: np.ndarray) -> np.ndarray:
        """Maps complex Fourier coefficients to a function vector."""
        return irfft(coeff, n=self.dim) * self._inverse_fft_factor

    def __eq__(self, other: object) -> bool:
        """
        Checks for mathematical equality with another Lebesgue space on a circle.

        Two spaces are considered equal if they are of the same type and have
        the same defining parameters (kmax, order, scale, and radius).
        """
        if not isinstance(other, Lebesgue):
            return NotImplemented

        return self.kmax == other.kmax and self.radius == other.radius

    # ================================================================#
    #                         Private methods                         #
    # ================================================================#

    def _coefficient_to_component(self, coeff: np.ndarray) -> np.ndarray:
        """Packs complex Fourier coefficients into a real component vector."""
        return np.concatenate((coeff.real, coeff.imag[1 : self.kmax]))

    def _component_to_coefficient(self, c: np.ndarray) -> np.ndarray:
        """Unpacks a real component vector into complex Fourier coefficients."""
        coeff_real = c[: self.kmax + 1]
        coeff_imag = np.concatenate([[0], c[self.kmax + 1 :], [0]])
        return coeff_real + 1j * coeff_imag

    def _to_components_impl(self, u: np.ndarray) -> np.ndarray:
        """Converts a function vector to its real component representation."""
        coeff = self.to_coefficient(u)
        return self._coefficient_to_component(coeff)

    def _from_components_impl(self, c: np.ndarray) -> np.ndarray:
        """Converts a real component vector back to a function vector."""
        coeff = self._component_to_coefficient(c)
        return self.from_coefficient(coeff)

    def _to_dual_impl(self, u: np.ndarray) -> "LinearForm":
        """Maps a vector `u` to its dual representation `u*`."""
        cp = self.to_components(u)
        return self.dual.from_components(cp)

    def _from_dual_impl(self, up: "LinearForm") -> np.ndarray:
        """Maps a dual vector `u*` back to its primal representation `u`."""
        c = self.dual.to_components(up)
        return self.from_components(c)


class Sobolev(CircleHelper, MassWeightedHilbertSpace):
    """
    Implementation of the Sobolev space L^2 on a circle based on Fourier expansions.
    """

    def __init__(
        self,
        kmax: int,
        order: float,
        scale: float,
        /,
        *,
        radius: float = 1.0,
    ):
        """
        Args:
        kmax: The maximum Fourier degree to be represented.
        order: The Sobolev order, controlling the smoothness of functions.
        scale: The Sobolev length-scale.
        radius: Radius of the circle. Defaults to 1.0.
        """

        CircleHelper.__init__(self, kmax, radius)

        self._order: float = order
        self._scale: float = scale

        lebesgue = Lebesgue(kmax, radius=radius)

        values = np.zeros(self.kmax + 1)
        values[0] = 1
        for k in range(1, self.kmax + 1):
            values[k] = 2 * self.sobolev_function(k)

        self._metric = diags([values], [0])
        self._inverse_metric = diags([np.reciprocal(values)], [0])

        def mass_operator_mapping(u: np.ndarray) -> np.ndarray:
            coeff = lebesgue.to_coefficient(u)
            coeff = self._metric @ coeff
            return lebesgue.from_coefficient(coeff)

        def inverse_mass_operator_mapping(u: np.ndarray) -> np.ndarray:
            coeff = lebesgue.to_coefficient(u)
            coeff = self._inverse_metric @ coeff
            return lebesgue.from_coefficient(coeff)

        mass_operator = LinearOperator.self_adjoint(lebesgue, mass_operator_mapping)
        inverse_mass_operator = LinearOperator.self_adjoint(
            lebesgue, inverse_mass_operator_mapping
        )

        MassWeightedHilbertSpace.__init__(
            self, lebesgue, mass_operator, inverse_mass_operator
        )

    @property
    def order(self) -> float:
        """The Sobolev order."""
        return self._order

    @property
    def scale(self) -> float:
        """The Sobolev length-scale."""
        return self._scale

    def sobolev_function(self, k: int) -> float:
        """Computes the diagonal entries of the Sobolev metric in Fourier space."""
        return (1 + (self.scale * k / self.radius) ** 2) ** self.order

    def __eq__(self, other: object) -> bool:
        """
        Checks for mathematical equality with another Sobolev space on a circle.

        Two spaces are considered equal if they are of the same type and have
        the same defining parameters (kmax, order, scale, and radius).
        """
        if not isinstance(other, Sobolev):
            return NotImplemented

        return (
            self.kmax == other.kmax
            and self.radius == other.radius
            and self.order == other.order
            and self.scale == other.scale
        )
