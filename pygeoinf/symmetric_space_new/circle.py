from __future__ import annotations
from typing import Callable, Any, Optional, Tuple, List

import numpy as np
from scipy.fft import rfft, irfft

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .symmetric_space import AbstractSymmetricLebesgueSpace, SymmetricSobolevSpace


class CircleHelper:
    """
    A mixin class providing common functionality for function spaces on the circle.
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


class Lebesgue(CircleHelper, AbstractSymmetricLebesgueSpace):
    """
    Implementation of the Lebesgue space L² on a circle.

    This class represents square-integrable functions on a circle. A function is
    represented by its values on an evenly spaced grid. The co-ordinate basis for
    the space is through Fourier expansions up.
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

        CircleHelper.__init__(self, kmax, radius)

        AbstractSymmetricLebesgueSpace.__init__(self, 1, 2 * kmax, False)

    # ------------------------------------------------------ #
    #           Methods for SymmetricHilbertSpace            #
    # ------------------------------------------------------ #

    def integer_to_index(self, i: int) -> int:
        """
        Maps 0..dim-1 to a unique Fourier mode index k.

        Mapping logic:
        - i in [0, kmax] -> k = i (Constant and Cosine terms)
        - i in [kmax+1, 2*kmax-1] -> k = -(i - kmax) (Sine terms)
        """
        if i <= self.kmax:
            return i
        return -(i - self.kmax)

    def index_to_integer(self, k: int) -> int:
        """
        Maps a unique Fourier mode index k back to the component integer.

        Inverse mapping logic:
        - k >= 0 -> i = k
        - k < 0  -> i = abs(k) + kmax
        """
        if k >= 0:
            if k > self.kmax:
                raise ValueError(f"Index k={k} exceeds kmax={self.kmax}")
            return k

        i = abs(k) + self.kmax
        if i >= self.dim:
            raise ValueError(f"Index k={k} results in integer {i} out of bounds.")
        return i

    def laplacian_eigenvalue(self, k: int) -> float:
        return (k / self.radius) ** 2

    def laplacian_eigenvector_squared_norm(self, k: int) -> float:
        return 1.0 if k == 0 else 2.0

    def laplacian_eigenvectors_at_point(self, theta: float) -> np.ndarray:
        k_vals = np.arange(self.kmax + 1)
        cos_terms = np.cos(k_vals * theta)
        sin_terms = np.sin(k_vals[1 : self.kmax] * theta)
        return (
            self._metric
            @ np.concatenate([cos_terms, -sin_terms])
            / (np.sqrt(2 * np.pi * self.radius))
        )

    def random_point(self) -> float:
        return np.random.uniform(0, 2 * np.pi)

    def geodesic_quadrature(
        self, p1: float, p2: float, n_points: int
    ) -> Tuple[List[float], np.ndarray]:
        """
        Returns quadrature points and weights for the shortest arc between p1 and p2.

        Args:
            p1: Starting angle in radians.
            p2: Ending angle in radians.
            n_points: Number of quadrature points.

        Returns:
            points: A list of angles (floats) along the shortest arc.
            weights: Integration weights scaled by the arc length.
        """
        diff = (p2 - p1 + np.pi) % (2 * np.pi) - np.pi
        arc_length = np.abs(diff) * self.radius

        x, w = np.polynomial.legendre.leggauss(n_points)

        t = (x + 1) / 2.0
        angles = p1 + t * diff

        scaled_weights = w * (arc_length / 2.0)

        return angles.tolist(), scaled_weights

    # ------------------------------------------------------ #
    #                 Methods for HilbertSpace               #
    # ------------------------------------------------------ #

    def to_components(self, u: np.ndarray) -> np.ndarray:
        """Converts a function vector to its real component representation."""
        coeff = self.to_coefficients(u)
        return self._coefficient_to_component(coeff)

    def from_components(self, c: np.ndarray) -> np.ndarray:
        """Converts a real component vector back to a function vector."""
        coeff = self._component_to_coefficients(c)
        return self.from_coefficients(coeff)

    def is_element(self, u: Any) -> bool:
        """
        Checks if an object is a valid element of the space.
        """
        if not isinstance(u, np.ndarray):
            return False
        if not u.shape == (self.dim,):
            return False
        return True

    def __eq__(self, other: object) -> bool:
        """
        Checks for mathematical equality with another Lebesgue space on a circle.

        Two spaces are considered equal if they are of the same type and have
        the same defining parameters (kmax, order, scale, and radius).
        """
        if not isinstance(other, Lebesgue):
            return NotImplemented

        return self.kmax == other.kmax and self.radius == other.radius

    # ------------------------------------------------------ #
    #                 Methods for HilbertModule              #
    # ------------------------------------------------------ #

    def vector_multiply(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        Computes the pointwise product of two vectors.
        """
        return x1 * x2

    def vector_sqrt(self, u: np.ndarray) -> np.ndarray:
        """
        Returns the pointwise square root of a function.
        """
        return np.sqrt(u)

    # ------------------------------------------------------ #
    #                      Private methods                   #
    # ------------------------------------------------------ #

    def _coefficient_to_component(self, coeff: np.ndarray) -> np.ndarray:
        """Packs complex Fourier coefficients into a real component vector."""
        # For a real-valued input, the output of rfft (real FFT) has
        # conjugate symmetry. This implies that the imaginary parts of the
        # zero-frequency (k=0) and Nyquist-frequency (k=kmax) components
        # are always zero. We omit them from the component vector to create
        # a minimal, non-redundant representation.
        c = np.empty(self.dim, dtype=float)
        c[: self.kmax + 1] = coeff.real
        c[self.kmax + 1 :] = coeff.imag[1 : self.kmax]

        return c

    def _component_to_coefficients(self, c: np.ndarray) -> np.ndarray:
        """Unpacks a real component vector into complex Fourier coefficients."""
        # This is the inverse of `_coefficient_to_component`. It reconstructs
        # the full complex coefficient array that irfft expects. We re-insert
        # the known zeros for the imaginary parts of the zero-frequency (k=0)
        # and Nyquist-frequency (k=kmax) components, which were removed to
        # create the minimal real-valued representation.
        coeff = np.zeros(self.kmax + 1, dtype=complex)
        coeff.real = c[: self.kmax + 1]
        coeff.imag[1 : self.kmax] = c[self.kmax + 1 :]
        return coeff


class Sobolev(CircleHelper, SymmetricSobolevSpace):
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
        radius: float = 1.0,
    ):
        """
        Args:
        kmax: The maximum Fourier degree to be represented.
        order: The Sobolev order, controlling the smoothness of functions.
        scale: The Sobolev length-scale.
        radius: Radius of the circle. Defaults to 1.0.
        """

        if kmax < 0:
            raise ValueError("kmax must be non-negative")

        CircleHelper.__init__(self, kmax, radius)

        lebesgue_space = Lebesgue(kmax, radius=radius)
        SymmetricSobolevSpace.__init__(self, lebesgue_space, order, scale)
