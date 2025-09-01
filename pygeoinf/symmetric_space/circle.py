"""
Provides concrete implementations of function spaces on the circle (S¹).

This module uses the abstract framework from the symmetric space module to create
fully-featured `Lebesgue` (L²) and `Sobolev` (Hˢ) Hilbert spaces for functions
defined on a circle.

The core representation for a function is a truncated real Fourier series,
and the module provides efficient methods to transform between the spatial domain
(function values on a grid) and the frequency domain (Fourier coefficients)
using the Fast Fourier Transform (FFT). This allows for the construction of
differential operators and rotationally-invariant Gaussian measures on the
circle, which are diagonal in the Fourier basis.

Key Classes
-----------
- `CircleHelper`: A mixin class providing the core geometry, FFT machinery, and
  plotting utilities.
- `Lebesgue`: A concrete implementation of the L²(S¹) space of square-integrable
  functions.
- `Sobolev`: A concrete implementation of the Hˢ(S¹) space of functions with a
  specified degree of smoothness.
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
    HilbertModule,
    MassWeightedHilbertModule,
)
from pygeoinf.operators import LinearOperator
from pygeoinf.linear_forms import LinearForm
from .symmetric_space import (
    AbstractInvariantLebesgueSpace,
    AbstractInvariantSobolevSpace,
)


class CircleHelper:
    """
    A mixin class providing common functionality for function spaces on the circle.

    This helper is not intended to be instantiated directly. It provides the core
    geometry (radius, grid points), the FFT transform machinery, and plotting
    utilities that are shared by both the `Lebesgue` and `Sobolev` space classes.
    """

    def __init__(self, kmax: int, radius: float):
        """
        Args:
            kmax: The maximum Fourier degree to be represented.
            radius: Radius of the circle.
        """
        self._kmax: int = kmax
        self._radius: float = radius

        self._fft_factor: float = np.sqrt(2 * np.pi * radius) / (2 * self.kmax)
        self._inverse_fft_factor: float = 1.0 / self._fft_factor

    def _space(self):
        return self

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

    @property
    def spatial_dimension(self) -> int:
        """The dimension of the space."""
        return 1

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

    def random_point(self) -> float:
        """Returns a random angle in the interval [0, 2*pi)."""
        return np.random.uniform(0, 2 * np.pi)

    def angles(self) -> np.ndarray:
        """Returns a numpy array of the grid point angles."""
        return np.fromiter(
            [i * self.angle_spacing for i in range(2 * self.kmax)],
            float,
        )

    def laplacian_eigenvalue(self, k: int) -> float:
        """
        Returns the k-th eigenvalue of the Laplacian.

        Args:
            k: The index of the eigenvalue to return.
        """
        return (k / self.radius) ** 2

    def trace_of_invariant_automorphism(self, f: Callable[[float], float]) -> float:
        """
        Returns the trace of the automorphism of the form f(Δ) with f a function
        that is well-defined on the spectrum of the Laplacian.

        Args:
            f: A real-valued function that is well-defined on the spectrum
               of the Laplacian.

        Notes:
            The method takes account of the Nyquist theorem for real fast Fourier transforms,
            this meaning that element at k = -kmax is excluded from the trace.
        """
        trace = f(self.laplacian_eigenvalue(0))
        if self.kmax > 0:
            trace += f(self.laplacian_eigenvalue(self.kmax))
        trace += 2 * np.sum(
            [f(self.laplacian_eigenvalue(k)) for k in range(1, self.kmax)]
        )
        return float(trace)

    def project_function(self, f: Callable[[float], float]) -> np.ndarray:
        """
        Returns an element of the space by projecting a given function.

        The function `f` is evaluated at the grid points of the space.

        Args:
            f: A function that takes an angle (float) and returns a value.
        """
        return np.fromiter((f(theta) for theta in self.angles()), float)

    def to_coefficient(self, u: np.ndarray) -> np.ndarray:
        """Maps a function vector to its complex Fourier coefficients."""
        return rfft(u) * self.fft_factor

    def from_coefficient(self, coeff: np.ndarray) -> np.ndarray:
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

    def _coefficient_to_component(self, coeff: np.ndarray) -> np.ndarray:
        """Packs complex Fourier coefficients into a real component vector."""
        # For a real-valued input, the output of rfft (real FFT) has
        # conjugate symmetry. This implies that the imaginary parts of the
        # zero-frequency (k=0) and Nyquist-frequency (k=kmax) components
        # are always zero. We omit them from the component vector to create
        # a minimal, non-redundant representation.
        return np.concatenate((coeff.real, coeff.imag[1 : self.kmax]))

    def _component_to_coefficient(self, c: np.ndarray) -> np.ndarray:
        """Unpacks a real component vector into complex Fourier coefficients."""
        # This is the inverse of `_coefficient_to_component`. It reconstructs
        # the full complex coefficient array that irfft expects. We re-insert
        # the known zeros for the imaginary parts of the zero-frequency (k=0)
        # and Nyquist-frequency (k=kmax) components, which were removed to
        # create the minimal real-valued representation.
        coeff_real = c[: self.kmax + 1]
        coeff_imag = np.concatenate([[0], c[self.kmax + 1 :], [0]])
        return coeff_real + 1j * coeff_imag


class Lebesgue(CircleHelper, HilbertModule, AbstractInvariantLebesgueSpace):
    """
    Implementation of the Lebesgue space L² on a circle.

    This class represents square-integrable functions on a circle. A function is
    represented by its values on an evenly spaced grid. The L² inner product
    is correctly implemented by accounting for the non-orthonormality of the
    real Fourier basis functions.
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

        if kmax < 0:
            raise ValueError("kmax must be non-negative")

        self._dim = 2 * kmax

        CircleHelper.__init__(self, kmax, radius)

        values = np.fromiter(
            [2 if k > 0 else 1 for k in range(self.kmax + 1)], dtype=float
        )
        self._metric = diags([values], [0])
        self._inverse_metric = diags([np.reciprocal(values)], [0])

    @property
    def dim(self) -> int:
        """The dimension of the space."""
        return self._dim

    def to_components(self, u: np.ndarray) -> np.ndarray:
        """Converts a function vector to its real component representation."""
        coeff = self.to_coefficient(u)
        return self._coefficient_to_component(coeff)

    def from_components(self, c: np.ndarray) -> np.ndarray:
        """Converts a real component vector back to a function vector."""
        coeff = self._component_to_coefficient(c)
        return self.from_coefficient(coeff)

    def to_dual(self, u: np.ndarray) -> "LinearForm":
        """Maps a vector `u` to its dual representation `u*`."""
        coeff = self.to_coefficient(u)
        cp = self._coefficient_to_component(self._metric @ coeff)
        return self.dual.from_components(cp)

    def from_dual(self, up: "LinearForm") -> np.ndarray:
        """Maps a dual vector `u*` back to its primal representation `u`."""
        cp = self.dual.to_components(up)
        dual_coeff = self._component_to_coefficient(cp)
        primal_coeff = self._inverse_metric @ dual_coeff
        return self.from_coefficient(primal_coeff)

    def vector_multiply(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        Computes the pointwise product of two vectors.
        """
        return x1 * x2

    def eigenfunction_norms(self) -> np.ndarray:
        """Returns a list of the norms of the eigenfunctions."""
        return np.fromiter(
            [np.sqrt(2) if i > 0 else 1 for i in range(self.dim)],
            dtype=float,
        )

    def __eq__(self, other: object) -> bool:
        """
        Checks for mathematical equality with another Lebesgue space on a circle.

        Two spaces are considered equal if they are of the same type and have
        the same defining parameters (kmax, order, scale, and radius).
        """
        if not isinstance(other, Lebesgue):
            return NotImplemented

        return self.kmax == other.kmax and self.radius == other.radius

    def invariant_automorphism_from_index_function(self, g: Callable[[int], float]):
        """
        Implements an invariant automorphism of the form f(Δ) using Fourier
        expansions on a circle.

        For this method, the function f is given implicitly in terms of a
        function, g, of the eigenvalue indices for the space. Letting k(λ) be
        the index for eigenvalue λ, we then have f(λ) = g(k(λ)).

        Args:
            g: A real-valued function of the eigenvalue index.
        """

        values = np.fromiter(
            (g(k) for k in range(self.kmax + 1)),
            dtype=float,
        )
        matrix = diags([values], [0])

        def mapping(u):
            coeff = self.to_coefficient(u)
            coeff = matrix @ coeff
            return self.from_coefficient(coeff)

        return LinearOperator.self_adjoint(self, mapping)


class Sobolev(
    CircleHelper,
    MassWeightedHilbertModule,
    AbstractInvariantSobolevSpace,
):
    """
    Implementation of the Sobolev space Hˢ on a circle.

    This class represents functions with a specified degree of smoothness. It is
    constructed as a `MassWeightedHilbertModule` over the `Lebesgue` space, where
    the mass operator weights the Fourier coefficients to enforce smoothness. This
    is the primary class for defining smooth, random function fields on the circle.
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
        AbstractInvariantSobolevSpace.__init__(self, order, scale)

        lebesgue = Lebesgue(kmax, radius=radius)

        mass_operator = lebesgue.invariant_automorphism(self.sobolev_function)
        inverse_mass_operator = lebesgue.invariant_automorphism(
            lambda k: 1.0 / self.sobolev_function(k)
        )

        MassWeightedHilbertModule.__init__(
            self, lebesgue, mass_operator, inverse_mass_operator
        )

    @staticmethod
    def from_sobolev_parameters(
        order: float,
        scale: float,
        /,
        *,
        radius: float = 1.0,
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
            if k > 100000:
                raise RuntimeError("Failed to converge on a stable kmax.")

        if power_of_two:
            n = int(np.log2(k))
            k = 2 ** (n + 1)

        return Sobolev(k, order, scale, radius=radius)

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

    def eigenfunction_norms(self) -> np.ndarray:
        """Returns a list of the norms of the eigenfunctions."""
        values = self.underlying_space.eigenfunction_norms()

        i = 0
        for k in range(self.kmax + 1):
            values[i] *= np.sqrt(self.sobolev_function(self.laplacian_eigenvalue(k)))
            i += 1

        for k in range(1, self.kmax):
            values[i] *= np.sqrt(self.sobolev_function(self.laplacian_eigenvalue(k)))
            i += 1

        return values

    def dirac(self, point: float) -> LinearForm:
        """
        Returns the linear functional corresponding to a point evaluation.

        This represents the action of the Dirac delta measure based at the given
        point.

        Args:
            point: The angle for the point at which the measure is based.

        Raises:
            ValueError: If the Sobolev order is less than 1/2.
        """
        if self.order <= 1 / 2:
            raise NotImplementedError(
                "This method is only applicable for orders >= 1/2"
            )

        coeff = np.zeros(self.kmax + 1, dtype=complex)
        fac = np.exp(-1j * point)
        coeff[0] = 1.0
        for k in range(1, coeff.size):
            coeff[k] = coeff[k - 1] * fac
        coeff *= 1.0 / np.sqrt(2 * np.pi * self.radius)
        coeff[1:] *= 2.0
        cp = self._coefficient_to_component(coeff)
        return LinearForm(self, components=cp)
