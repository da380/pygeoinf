"""
 Sobolev spaces for functions on a circle.
"""

from __future__ import annotations
from typing import Callable, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, irfft
from scipy.sparse import diags


from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pygeoinf.operators import LinearOperator
from pygeoinf.forms import LinearForm
from pygeoinf.gaussian_measure import GaussianMeasure

from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.symmetric_space.symmetric_space import SymmetricSpaceSobolev


class Sobolev(SymmetricSpaceSobolev):
    """
    Implementation of the Sobolev space H^s on a circle.

    Functions on the circle are represented by their values on a grid of
    equally spaced points, and the inner product is defined via their
    Fourier coefficients.
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
            kmax: The maximum Fourier degree to be represented in this space.
            order: The Sobolev order, controlling the smoothness of functions.
            scale: The Sobolev length-scale.
            radius: The radius of the circle. Defaults to 1.0.
        """
        self._kmax: int = kmax
        self._radius: float = radius

        super().__init__(
            order,
            scale,
            2 * kmax,
            self._to_components,
            self._from_components,
            self._inner_product,
            self._to_dual,
            self._from_dual,
            vector_multiply=lambda u1, u2: u1 * u2,
        )

        self._fft_factor: float = np.sqrt(2 * np.pi * radius) / self.dim
        self._inverse_fft_factor: float = 1.0 / self._fft_factor

        values = np.zeros(self.kmax + 1)
        values[0] = 1
        for k in range(1, self.kmax + 1):
            values[k] = 2 * self._sobolev_function(k)

        self._metric = diags([values], [0])
        self._inverse_metric = diags([np.reciprocal(values)], [0])

    @staticmethod
    def from_sobolev_parameters(
        order: float,
        scale: float,
        /,
        *,
        radius: float = 1.0,
        rtol: float = 1e-8,
        power_of_two: bool = False,
    ) -> "Sobolev":
        """
        Creates an instance with `kmax` chosen based on Sobolev parameters.

        The method estimates the truncation error for the Dirac measure and is
        only applicable for spaces with order > 0.5.

        Args:
            order: The Sobolev order. Must be > 0.5.
            scale: The Sobolev length-scale.
            radius: The radius of the circle. Defaults to 1.0.
            rtol: Relative tolerance used in assessing truncation error.
                Defaults to 1e-8.
            power_of_two: If True, `kmax` is set to the next power of two.

        Returns:
            An instance of the Sobolev class with an appropriate `kmax`.

        Raises:
            ValueError: If order is <= 0.5.
        """
        if order <= 0.5:
            raise ValueError("This method is only applicable for orders > 0.5")

        summation = 1.0
        k = 0
        err = 1.0
        while err > rtol:
            k += 1
            term = (1 + (scale * k / radius) ** 2) ** -order
            summation += 2 * term
            err = 2 * term / summation
            if k > 10000:
                raise RuntimeError("Failed to converge on a stable kmax.")

        if power_of_two:
            n = int(np.log2(k))
            k = 2 ** (n + 1)

        return Sobolev(k, order, scale, radius=radius)

    @property
    def kmax(self) -> int:
        """The maximum Fourier degree represented in this space."""
        return self._kmax

    @property
    def radius(self) -> float:
        """The radius of the circle."""
        return self._radius

    @property
    def angle_spacing(self) -> float:
        """The angular spacing between grid points."""
        return 2 * np.pi / self.dim

    def random_point(self) -> float:
        """Returns a random angle in the interval [0, 2*pi)."""
        return np.random.uniform(0, 2 * np.pi)

    def angles(self) -> np.ndarray:
        """Returns a numpy array of the grid point angles."""
        return np.fromiter(
            [i * self.angle_spacing for i in range(self.dim)],
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

    def invariant_automorphism(self, f: Callable[[float], float]) -> "LinearOperator":
        values = np.fromiter(
            (f(k * k / self.radius**2) for k in range(self.kmax + 1)), dtype=float
        )
        matrix = diags([values], [0])

        def mapping(u: np.ndarray) -> np.ndarray:
            coeff = self.to_coefficient(u)
            coeff = matrix @ coeff
            return self.from_coefficient(coeff)

        return LinearOperator.formally_self_adjoint(self, mapping)

    def invariant_gaussian_measure(
        self,
        f: Callable[[float], float],
        /,
        *,
        expectation: Optional[np.ndarray] = None,
    ) -> "GaussianMeasure":
        values = np.fromiter(
            (np.sqrt(f(k * k / self.radius**2)) for k in range(self.kmax + 1)),
            dtype=float,
        )
        matrix = diags([values], [0])

        domain = EuclideanSpace(self.dim)
        codomain = self

        def mapping(c: np.ndarray) -> np.ndarray:
            coeff = self._component_to_coefficient(c)
            coeff = matrix @ coeff
            return self.from_coefficient(coeff)

        def formal_adjoint(u: np.ndarray) -> np.ndarray:
            coeff = self.to_coefficient(u)
            coeff = matrix @ coeff
            return self._coefficient_to_component(coeff)

        covariance_factor = LinearOperator(
            domain, codomain, mapping, formal_adjoint_mapping=formal_adjoint
        )

        return GaussianMeasure(
            covariance_factor=covariance_factor,
            expectation=expectation,
        )

    def dirac(self, point: float) -> "LinearForm":
        coeff = np.zeros(self.kmax + 1, dtype=complex)
        fac = np.exp(-1j * point)
        coeff[0] = 1.0
        for k in range(1, coeff.size):
            coeff[k] = coeff[k - 1] * fac
        coeff *= 1.0 / np.sqrt(2 * np.pi * self.radius)
        coeff[1:] *= 2.0
        cp = self._coefficient_to_component(coeff)
        return LinearForm(self, components=cp)

    def to_coefficient(self, u: np.ndarray) -> np.ndarray:
        """Maps a function vector to its complex Fourier coefficients."""
        return rfft(u) * self._fft_factor

    def from_coefficient(self, coeff: np.ndarray) -> np.ndarray:
        """Maps complex Fourier coefficients to a function vector."""
        return irfft(coeff, n=self.dim) * self._inverse_fft_factor


    def __eq__(self, other: object) -> bool:
        """
        Checks for mathematical equality with another Sobolev space on a circle.

        Two spaces are considered equal if they are of the same type and have
        the same defining parameters (kmax, order, scale, and radius).
        """
        if not isinstance(other, Sobolev):
            return NotImplemented

        return (self.kmax == other.kmax and
                self.order == other.order and
                self.scale == other.scale and
                self.radius == other.radius)

    # ================================================================#
    #                         Private methods                         #
    # ================================================================#

    def _sobolev_function(self, k: int) -> float:
        """Computes the diagonal entries of the Sobolev metric in Fourier space."""
        return (1 + (self.scale * k / self.radius) ** 2) ** self.order

    def _coefficient_to_component(self, coeff: np.ndarray) -> np.ndarray:
        """Packs complex Fourier coefficients into a real component vector."""
        return np.concatenate((coeff.real, coeff.imag[1 : self.kmax]))

    def _component_to_coefficient(self, c: np.ndarray) -> np.ndarray:
        """Unpacks a real component vector into complex Fourier coefficients."""
        coeff_real = c[: self.kmax + 1]
        coeff_imag = np.concatenate([[0], c[self.kmax + 1 :], [0]])
        return coeff_real + 1j * coeff_imag

    def _to_components(self, u: np.ndarray) -> np.ndarray:
        """Converts a function vector to its real component representation."""
        coeff = self.to_coefficient(u)
        return self._coefficient_to_component(coeff)

    def _from_components(self, c: np.ndarray) -> np.ndarray:
        """Converts a real component vector back to a function vector."""
        coeff = self._component_to_coefficient(c)
        return self.from_coefficient(coeff)

    def _inner_product(self, u1: np.ndarray, u2: np.ndarray) -> float:
        """Computes the H^s inner product in the Fourier domain."""
        coeff1 = self.to_coefficient(u1)
        coeff2 = self.to_coefficient(u2)
        return np.real(np.vdot(self._metric @ coeff1, coeff2))

    def _to_dual(self, u: np.ndarray) -> "LinearForm":
        """Maps a vector `u` to its dual representation `u*`."""
        coeff = self.to_coefficient(u)
        cp = self._coefficient_to_component(self._metric @ coeff)
        return self.dual.from_components(cp)

    def _from_dual(self, up: "LinearForm") -> np.ndarray:
        """Maps a dual vector `u*` back to its primal representation `u`."""
        cp = self.dual.to_components(up)
        coeff = self._component_to_coefficient(cp)
        c = self._coefficient_to_component(self._inverse_metric @ coeff)
        return self.from_components(c)


class Lebesgue(Sobolev):
    """
    Implementation of the Lebesgue space L^2 on a circle.

    This is a special case of the Sobolev space with order s=0.
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
        super().__init__(kmax, 0.0, 1.0, radius=radius)
