from __future__ import annotations
from typing import Callable, Any, Optional, Tuple, List

import numpy as np
from scipy.fft import rfft, irfft

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from pygeoinf.linear_operators import LinearOperator
from .symmetric_space import AbstractSymmetricLebesgueSpace, SymmetricSobolevSpace


class Lebesgue(AbstractSymmetricLebesgueSpace):
    """
    Implementation of the Lebesgue space L² on a circle.

    This class represents square-integrable functions on a circle. A function is
    represented by its values on an evenly spaced grid. The co-ordinate basis for
    the space is through Fourier expansions.
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

        if kmax <= 0:
            raise ValueError("kmax must be non-negative")

        if radius <= 0:
            raise ValueError("radius must be positive")

        self._kmax: int = kmax
        self._radius: float = radius

        self._fft_factor: float = np.sqrt(2 * np.pi * radius) / (2 * self.kmax)
        self._inverse_fft_factor: float = 1.0 / self._fft_factor

        AbstractSymmetricLebesgueSpace.__init__(self, 1, kmax, 2 * kmax, False)

    @classmethod
    def from_covariance(
        cls,
        covariance_function: Callable[[float], float],
        /,
        *,
        radius: float = 1.0,
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
            radius: The radius of the circle. Defaults to 1.0.
            rtol: The relative tolerance for the energy truncation. Defaults to 1e-6.
            min_degree: The absolute minimum truncation degree to return. Defaults to 0.
            max_degree: An optional safety ceiling for the truncation degree. If convergence
                is not reached by this degree, the search terminates and returns this value.
            power_of_two: If True, rounds the resulting `kmax` up to the nearest
                power of two (useful for FFT optimizations). Defaults to False.

        Returns:
            Lebesgue: A fully instantiated L² space on the circle with the optimal `kmax`.

        Raises:
            RuntimeError: If the energy sequence fails to converge within 100,000 iterations
                and no `max_degree` is specified.
        """
        dummy_space = cls(max(1, min_degree), radius=radius)

        optimal_degree = dummy_space.estimate_truncation_degree(
            covariance_function, rtol=rtol, min_degree=min_degree, max_degree=max_degree
        )

        if power_of_two:
            n = int(np.log2(optimal_degree))
            optimal_degree = 2 ** (n + 1)

        return cls(optimal_degree, radius=radius)

    @classmethod
    def from_heat_kernel_prior(
        cls,
        kernel_scale: float,
        /,
        *,
        radius: float = 1.0,
        rtol: float = 1e-6,
        min_degree: int = 0,
        max_degree: Optional[int] = None,
        power_of_two: bool = False,
    ) -> Lebesgue:
        """
        Creates an instance of the L² space on the circle, tuned to the expected
        energy of a Heat Kernel prior measure.

        Args:
            kernel_scale: The length-scale parameter of the heat kernel covariance.
            radius: The radius of the circle. Defaults to 1.0.
            rtol: The relative tolerance for the energy truncation. Defaults to 1e-6.
            min_degree: The absolute minimum truncation degree to return. Defaults to 0.
            max_degree: An optional safety ceiling for the truncation degree.
            power_of_two: If True, rounds the resulting `kmax` up to the nearest power of two.
        """
        return cls.from_covariance(
            cls.heat_kernel(kernel_scale),
            radius=radius,
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
        radius: float = 1.0,
        rtol: float = 1e-6,
        min_degree: int = 0,
        max_degree: Optional[int] = None,
        power_of_two: bool = False,
    ) -> Lebesgue:
        """
        Creates an instance of the L² space on the circle, tuned to the expected
        energy of a Sobolev-type prior measure.

        Args:
            kernel_order: The smoothness order of the Sobolev prior measure.
            kernel_scale: The length-scale parameter of the Sobolev prior measure.
            radius: The radius of the circle. Defaults to 1.0.
            rtol: The relative tolerance for the energy truncation. Defaults to 1e-6.
            min_degree: The absolute minimum truncation degree to return. Defaults to 0.
            max_degree: An optional safety ceiling for the truncation degree.
            power_of_two: If True, rounds the resulting `kmax` up to the nearest power of two.
        """
        return cls.from_covariance(
            cls.sobolev_kernel(kernel_order, kernel_scale),
            radius=radius,
            rtol=rtol,
            min_degree=min_degree,
            max_degree=max_degree,
            power_of_two=power_of_two,
        )

    # ---------------------------------------------- #
    #                   Properties                   #
    # -----------------------------------------------#

    @property
    def kmax(self) -> int:
        """The maximum Fourier degree represented in this space."""
        return self._kmax

    @property
    def radius(self) -> float:
        """The radius of the circle."""
        return self._radius

    @property
    def point_spacing(self) -> float:
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

    # ---------------------------------------------- #
    #                 Public methods                 #
    # -----------------------------------------------#

    def points(self) -> np.ndarray:
        """Returns a numpy array of the grid point angles."""
        return np.fromiter(
            [i * self.point_spacing for i in range(2 * self.kmax)],
            float,
        )

    def project_function(self, f: Callable[[float], float]) -> np.ndarray:
        """
        Returns an element of the space by projecting a given function.

        The function `f` is evaluated at the grid points of the space.

        Args:
            f: A function that takes an angle (float) and returns a value.
        """
        return np.fromiter((f(theta) for theta in self.points()), float)

    def to_coefficients(self, u: np.ndarray) -> np.ndarray:
        """Maps a function vector to its complex Fourier coefficients."""
        return rfft(u) * self.fft_factor

    def from_coefficients(self, coeff: np.ndarray) -> np.ndarray:
        """Maps complex Fourier coefficients to a function vector."""
        return irfft(coeff, n=2 * self.kmax) * self._inverse_fft_factor

    # ------------------------------------------------------ #
    #           Methods for SymmetricHilbertSpace            #
    # ------------------------------------------------------ #

    def integer_to_index(self, i: int) -> int:
        if i <= self.kmax:
            return i
        return -(i - self.kmax)

    def index_to_integer(self, k: int) -> int:
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

    def geodesic_distance(self, p1: float, p2: float) -> float:
        diff = (p2 - p1 + np.pi) % (2 * np.pi) - np.pi
        return float(np.abs(diff) * self.radius)

    def geodesic_quadrature(
        self, p1: float, p2: float, n_points: int
    ) -> Tuple[List[float], np.ndarray]:

        arc_length = self.geodesic_distance(p1, p2)

        diff = (p2 - p1 + np.pi) % (2 * np.pi) - np.pi

        x, w = np.polynomial.legendre.leggauss(n_points)

        t = (x + 1) / 2.0
        angles = p1 + t * diff

        scaled_weights = w * (arc_length / 2.0)

        return angles.tolist(), scaled_weights

    def with_degree(self, degree: int) -> Lebesgue:
        return Lebesgue(degree, radius=self.radius)

    def degree_transfer_operator(self, target_degree: int) -> LinearOperator:
        """
        Returns the transfer operator from this space to one with a different degree.

        This operator upsamples (by zero-padding Fourier coefficients) or
        downsamples (by truncating Fourier coefficients) the function grid.
        """
        codomain = self.with_degree(target_degree)

        def mapping(u: np.ndarray) -> np.ndarray:
            # 1. Move to the frequency domain
            c_in = self.to_coefficients(u)

            # 2. Pad or truncate
            c_out = np.zeros(target_degree + 1, dtype=complex)
            k_min = min(self.kmax, target_degree)
            c_out[: k_min + 1] = c_in[: k_min + 1]

            # 3. Enforce a strictly real Nyquist frequency when downsampling
            if target_degree < self.kmax:
                c_out[target_degree] = c_out[target_degree].real + 0j

            # 4. Return to the spatial domain
            return codomain.from_coefficients(c_out)

        def adjoint_mapping(v: np.ndarray) -> np.ndarray:
            c_in = codomain.to_coefficients(v)

            c_out = np.zeros(self.kmax + 1, dtype=complex)
            k_min = min(self.kmax, target_degree)
            c_out[: k_min + 1] = c_in[: k_min + 1]

            # The adjoint must mirror the forward map's Nyquist handling perfectly
            if self.kmax < target_degree:
                c_out[self.kmax] = c_out[self.kmax].real + 0j

            return self.from_coefficients(c_out)

        return LinearOperator(self, codomain, mapping, adjoint_mapping=adjoint_mapping)

    def invariant_covariance_function(
        self, spectral_variances: np.ndarray
    ) -> Callable[[np.ndarray], np.ndarray]:

        # Extract the wavenumber variances
        k_variances = np.zeros(self.kmax + 1)
        for k in range(self.kmax + 1):
            idx = self.index_to_integer(k)
            k_variances[k] = spectral_variances[idx]

        # Prepare the Chebyshev coefficients
        coeffs = k_variances.copy()
        coeffs[1:] *= 2.0  # k > 0 modes have multiplicity 2 (cos and sin)[cite: 3]
        coeffs /= 2 * np.pi * self.radius

        def cov_evaluator(distances: np.ndarray) -> np.ndarray:
            theta = distances / self.radius
            # Evaluate the Cosine/Chebyshev series at all distances simultaneously
            return np.polynomial.chebyshev.chebval(np.cos(theta), coeffs)

        return cov_evaluator

    def degree_multiplicity(self, degree: int) -> int:
        return 1 if degree == 0 else 2

    def representative_index(self, degree: int) -> int:
        return degree

    # ------------------------------------------------------ #
    #                 Methods for HilbertSpace               #
    # ------------------------------------------------------ #

    def to_components(self, x: np.ndarray) -> np.ndarray:
        coeff = self.to_coefficients(x)
        return self._coefficient_to_component(coeff)

    def from_components(self, c: np.ndarray) -> np.ndarray:
        coeff = self._component_to_coefficients(c)
        return self.from_coefficients(coeff)

    def is_element(self, x: Any) -> bool:
        if not isinstance(x, np.ndarray):
            return False
        if not x.shape == (self.dim,):
            return False
        return True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Lebesgue):
            return NotImplemented

        return self.kmax == other.kmax and self.radius == other.radius

    # ------------------------------------------------------ #
    #                 Methods for HilbertModule              #
    # ------------------------------------------------------ #

    def vector_multiply(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return x1 * x2

    def vector_sqrt(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(x)

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
        radius: float = 1.0,
        safe: bool = True,
    ):
        """
        Args:
        kmax: The maximum Fourier degree to be represented.
        order: The Sobolev order, controlling the smoothness of functions.
        scale: The Sobolev length-scale.
        radius: Radius of the circle. Defaults to 1.0.
        safe: If true, the class checks for mathematical correctness of operations
                  where possible.
        """

        lebesgue_space = Lebesgue(kmax, radius=radius)
        SymmetricSobolevSpace.__init__(self, lebesgue_space, order, scale, safe=safe)

    @staticmethod
    def from_sobolev_parameters(
        order: float,
        scale: float,
        /,
        *,
        radius: float = 1.0,
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
            radius: The radius of the circle. Defaults to 1.0.
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
        if safe and order <= 0.5:
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

    @classmethod
    def from_covariance(
        cls,
        covariance_function: Callable[[float], float],
        order: float,
        scale: float,
        /,
        *,
        radius: float = 1.0,
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
            radius: The radius of the circle. Defaults to 1.0.
            rtol: The relative tolerance for the energy truncation. Defaults to 1e-6.
            min_degree: The absolute minimum truncation degree to return. Defaults to 0.
            max_degree: An optional safety ceiling for the truncation degree. If convergence
                is not reached by this degree, the search terminates and returns this value.
            power_of_two: If True, rounds the resulting `kmax` up to the nearest
                power of two (useful for FFT optimizations). Defaults to False.
            safe: If True, enables mathematical correctness checks during operations.

        Returns:
            Sobolev: A fully instantiated Sobolev space on the circle with the optimal `kmax`.

        Raises:
            RuntimeError: If the energy sequence fails to converge within 100,000 iterations
                and no `max_degree` is specified.
        """
        dummy_space = cls(max(1, min_degree), order, scale, radius=radius, safe=safe)

        optimal_degree = dummy_space.estimate_truncation_degree(
            covariance_function, rtol=rtol, min_degree=min_degree, max_degree=max_degree
        )

        if power_of_two:
            n = int(np.log2(optimal_degree))
            optimal_degree = 2 ** (n + 1)

        return cls(optimal_degree, order, scale, radius=radius, safe=safe)

    @classmethod
    def from_heat_kernel_prior(
        cls,
        kernel_scale: float,
        order: float,
        scale: float,
        /,
        *,
        radius: float = 1.0,
        rtol: float = 1e-6,
        min_degree: int = 0,
        max_degree: Optional[int] = None,
        power_of_two: bool = False,
        safe: bool = True,
    ) -> Sobolev:
        """
        Creates an instance of the Sobolev space on the circle, tuned to the expected
        energy of a Heat Kernel prior measure.

        Args:
            kernel_scale: The length-scale parameter of the heat kernel covariance.
            order: The Sobolev order defining the function space.
            scale: The Sobolev length-scale defining the function space.
            radius: The radius of the circle. Defaults to 1.0.
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
            radius=radius,
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
        radius: float = 1.0,
        rtol: float = 1e-6,
        min_degree: int = 0,
        max_degree: Optional[int] = None,
        power_of_two: bool = False,
        safe: bool = True,
    ) -> Sobolev:
        """
        Creates an instance of the Sobolev space on the circle, tuned to the expected
        energy of a Sobolev-type prior measure.

        Args:
            kernel_order: The smoothness order of the Sobolev prior measure.
            kernel_scale: The length-scale parameter of the Sobolev prior measure.
            order: The Sobolev order defining the function space.
            scale: The Sobolev length-scale defining the function space.
            radius: The radius of the circle. Defaults to 1.0.
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
            radius=radius,
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
        """The maximum Fourier degree represented in this space."""
        return self.underlying_space.kmax

    @property
    def radius(self) -> float:
        """The radius of the circle."""
        return self.underlying_space.radius

    def points(self) -> np.ndarray:
        """Returns a numpy array of the grid point angles."""
        return self.underlying_space.points()

    @property
    def point_spacing(self) -> float:
        """The angular spacing between grid points."""
        return self.underlying_space.point_spacing

    @property
    def fft_factor(self) -> float:
        """
        The factor by which the Fourier coefficients are scaled
        in forward transformations.
        """
        return self.underlying_space.fft_factor

    @property
    def inverse_fft_factor(self) -> float:
        """
        The factor by which the Fourier coefficients are scaled
        in inverse transformations.
        """
        return self.underlying_space.inverse_fft_factor

    @property
    def derivative_operator(self) -> LinearOperator:
        """
        Returns the derivative operator from the space to one with a lower order.
        """
        codomain = self.with_order(self.order - 1)

        lebesgue_space = self.underlying_space
        k = np.arange(self.kmax + 1)

        def mapping(u):
            coeff = lebesgue_space.to_coefficients(u)
            diff_coeff = 1j * k * coeff
            return lebesgue_space.from_coefficients(diff_coeff)

        op_L2 = LinearOperator(
            lebesgue_space,
            lebesgue_space,
            mapping,
            adjoint_mapping=lambda u: -1 * mapping(u),
        )

        return LinearOperator.from_formal_adjoint(self, codomain, op_L2)

    # ---------------------------------------------- #
    #                 Public methods                 #
    # -----------------------------------------------#

    def with_order(self, order: float) -> Sobolev:
        return Sobolev(self.kmax, order, self.scale, radius=self.radius)

    def with_degree(self, degree: int) -> Sobolev:
        return Sobolev(degree, self.order, self.scale, radius=self.radius)

    def angles(self) -> np.ndarray:
        """Returns a numpy array of the grid point angles."""
        return self.underlying_space.angles()

    def project_function(self, f: Callable[[float], float]) -> np.ndarray:
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


# ------------------------------------------------- #
#           Associated plotting functions           #
# ------------------------------------------------- #


def plot(
    space: Lebesgue | Sobolev,
    u: np.ndarray,
    ax: Optional[Axes] = None,
    **kwargs,
) -> Axes:
    """
    Creates a simple line plot of a function on the circle.

    Args:
        space: The function space.
        u: A 1D numpy array representing the function values (the y-axis).
        ax: An existing Matplotlib Axes object. If None, plots to the current active axes.
        **kwargs: Additional keyword arguments forwarded directly to `ax.plot()`.

    Returns:
        The Matplotlib Axes object.
    """
    if ax is None:
        ax = plt.gca()

    ax.plot(space.points(), u, **kwargs)
    return ax


def plot_error_bounds(
    space: Lebesgue | Sobolev,
    u: np.ndarray,
    u_bound: np.ndarray,
    ax: Optional[Axes] = None,
    **kwargs,
) -> Axes:
    """
    Plots a function on the circle along with its pointwise error bounds.

    This is particularly useful for visualizing Gaussian measures or Bayesian
    posterior uncertainties over the circular domain.

    Args:
        space: The function space.
        u: A 1D numpy array representing the mean function values.
        u_bound: A 1D numpy array giving the pointwise standard deviations or bounds.
        ax: An existing Matplotlib Axes object. If None, plots to the current active axes.
        **kwargs: Additional keyword arguments forwarded directly to `ax.fill_between()`.

    Returns:
        The Matplotlib Axes object.
    """
    if ax is None:
        ax = plt.gca()

    ax.fill_between(space.points(), u - u_bound, u + u_bound, **kwargs)
    return ax
