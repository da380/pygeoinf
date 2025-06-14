"""
Sobolev spaces for functions on a circle. 
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, irfft
from scipy.sparse import diags
from pygeoinf.hilbert_space import (
    HilbertSpace,
    LinearOperator,
    LinearForm,
    EuclideanSpace,
)
from pygeoinf.gaussian_measure import GaussianMeasure


class Sobolev(HilbertSpace):
    """
    Implementation of the Sobolev space H^s on a circle.
    """

    def __init__(
        self,
        kmax,
        exponent,
        scale,
        /,
        *,
        radius=1,
    ):
        """
        Args:
            kmax (float): The maximum Fourier degree.
            exponent (float): Sobolev exponent.
            scale (float): Sobolev length-scale relative to radius.
            radius (float): Radius of the circle. Default is 1.
        """

        self._kmax = kmax
        self._exponent = exponent
        self._scale = scale
        self._radius = radius

        super().__init__(
            2 * kmax,
            self._to_componets,
            self._from_componets,
            self._inner_product,
            self._to_dual,
            self._from_dual,
            vector_multiply=lambda u1, u2: u1 * u2,
        )

        self._fft_factor = np.sqrt(2 * np.pi) / self.dim
        self._inverse_fft_factor = 1 / self._fft_factor

        # indexing information for component vectors
        self._real_start = 0
        self._real_finish = self.dim // 2 + 1
        self._real_k_start = 0
        self._imag_start = self._real_finish
        self._imag_finish = self.dim
        self._imag_k_start = 1

        # Set up sparse matrices for inner product and Riesz mappings
        values = np.zeros(self.kmax + 1)
        values[0] = 1
        for k in range(1, self.kmax + 1):
            values[k] = 2 * self._sobolev_function(k)

        self._metric = diags([values], [0])
        self._inverse_metric = diags([np.reciprocal(values)], [0])

    @staticmethod
    def from_sobolev_parameters(
        exponent, scale, /, *, radius=1, rtol=1e-8, power_of_two=False
    ):
        """
        Returns a instance of the class with the maximum Fourier degree
        chosen based on the Sobolev parameters. The criteria is based on
        convergence of the Dirac representation.
        """

        if exponent <= 0.5:
            raise ValueError("This method is only applicable for exponents > 0.5")

        sum = 1
        k = 0
        err = 1
        while err > rtol:
            k += 1
            term = (1 + (scale * k) ** 2) ** -exponent
            sum += term
            err = term / sum

        if power_of_two:
            n = int(np.log2(k))
            k = 2 ** (n + 1)

        return Sobolev(k, exponent, scale, radius=radius)

    @property
    def kmax(self):
        """
        Return the maximum Fourier degree.
        """
        return self._kmax

    @property
    def radius(self):
        """
        Return the radius.
        """
        return self._radius

    @property
    def exponent(self):
        """
        Return the Sobolev exponent.
        """
        return self._exponent

    @property
    def scale(self):
        """
        Return the relative Sobolev scale.
        """
        return self._scale

    @property
    def angle_spacing(self):
        """
        Return the angle spacing.
        """
        return 2 * np.pi / self.dim

    def angles(self):
        """
        Returns a numpy array of the angles points.
        """
        return np.fromiter(
            [i * self.angle_spacing for i in range(self.dim)],
            float,
        )

    def project_function(self, f):
        """
        Returns an element of the space formed by projecting a given function
        on the sample points.
        """
        return np.fromiter([f(x) for x in self.angles()], float)

    def plot(self, u, fig=None, ax=None, **kwargs):
        """
        Make a simple plot of an element of the space on the computational domain.
        """

        figsize = kwargs.pop("figsize", (10, 8))

        if fig is None:
            fig = plt.figure(figsize=figsize)
        if ax is None:
            ax = fig.add_subplot()

        line = ax.plot(self.angles(), u, **kwargs)

        return fig, ax, line[0]

    def plot_error_bounds(self, u, u_bound, fig=None, ax=None, **kwargs):
        """
        Make a plot of an element of the space bounded above and below by a standard
        deviation curve.
        """

        figsize = kwargs.pop("figsize", (10, 8))

        if fig is None:
            fig = plt.figure(figsize=figsize)
        if ax is None:
            ax = fig.add_subplot()

        obj = ax.fill_between(self.angles(), u - u_bound, u + u_bound, **kwargs)

        return fig, ax, obj

    def invariant_automorphism(self, f):
        """
        Returns a linear operator on the space of the form f(\Delta) with \Delta the Laplacian.
        The resulting operator is only well-defined as a continuous automorphism if f is bounded.
        """

        values = np.fromiter([f(k) for k in range(self.kmax + 1)], dtype=float)
        matrix = diags([values], [0])

        def mapping(u):
            coeff = self._to_coefficient(u)
            coeff = matrix @ coeff
            return self._from_coefficient(coeff)

        return LinearOperator.formally_self_adjoint(self, mapping)

    def invariant_gaussian_measure(self, f, /, *, expectation=None):
        """
        Returns a Gaussian measure on the space whose covariance takes the form f(Delta) with
        Delta the Laplacian. The trace-class condition for the measure to be well-defined implies
        that the sequence {f(k)} must be summable.
        """

        values = np.fromiter([np.sqrt(f(k)) for k in range(self.kmax + 1)], dtype=float)
        matrix = diags([values], [0])

        domain = EuclideanSpace(self.dim)
        codomain = self

        def mapping(c):
            coeff = self._component_to_coefficient(c)
            coeff = matrix @ coeff
            return self._from_coefficient(coeff)

        def formal_adjoint(u):
            coeff = self._to_coefficient(u)
            coeff = matrix @ coeff
            return self._coefficient_to_component(coeff)

        covariance_factor = LinearOperator.from_formal_adjoint(
            domain, codomain, mapping, formal_adjoint
        )

        return GaussianMeasure(
            covariance_factor=covariance_factor,
            expectation=expectation,
        )

    def sobolev_gaussian_measure(
        self, exponent, scale, amplitude, /, *, expectation=None
    ):
        """
        Returns a Gaussian measure with Sobolev covariance. The measure is
        scaled such that the its pointwise standard deviation is equal
        to the optional amplitude (default value 1).
        """
        mu = self.invariant_gaussian_measure(
            lambda k: (1 + (scale * k) ** 2) ** -exponent
        )
        Q = mu.covariance
        th = np.pi
        u = self.dirac_representation(th)
        var = self.inner_product(Q(u), u)
        mu *= amplitude / np.sqrt(var)
        return mu.affine_mapping(translation=expectation)

    def dirac(self, angle):
        """
        Returns a dirac measure as a LinearForm on the space.
        """
        coeff = np.zeros(self.kmax + 1, dtype=complex)
        fac = np.exp(-1j * angle)
        coeff[0] = 1
        for k in range(1, coeff.size):
            coeff[k] = coeff[k - 1] * fac
        coeff *= 1 / np.sqrt(2 * np.pi)
        coeff[1:] *= 2
        cp = self._coefficient_to_component(coeff)
        return LinearForm(self, components=cp)

    def dirac_representation(self, angle):
        """
        Returns the representation of the dirac measure based at x.
        """
        return self.from_dual(self.dirac(angle))

    def point_evaluation_operator(self, angles):
        """
        Returns the point evaluation operator on the space
        at the given list, xs, of points.
        """
        dim = len(angles)
        matrix = np.zeros((dim, self.dim))

        for i, angle in enumerate(angles):
            cp = self.dirac(angle).components
            matrix[i, :] = cp

        return LinearOperator.from_matrix(
            self, EuclideanSpace(dim), matrix, galerkin=True
        )

    def random_angles(self, n):
        """
        Returns n random angles from a uniform distribution.
        """
        return np.random.uniform(0, 2 * np.pi, n)

    # ================================================================#
    #                         Private methods                         #
    # ================================================================#

    def _sobolev_function(self, k):
        return (1 + (self.scale * k) ** 2) ** self.exponent

    def _to_coefficient(self, u):
        return rfft(u) * self._fft_factor

    def _from_coefficient(self, coeff):
        return irfft(coeff, n=self.dim) * self._inverse_fft_factor

    def _coefficient_to_component(self, coeff):
        return np.concatenate([coeff.real, coeff.imag[1 : self.kmax]])

    def _component_to_coefficient(self, c):
        coeff_real = c[: self.kmax + 1]
        coeff_imag = np.concatenate([[0], c[self.kmax + 1 :], [0]])
        return coeff_real + 1j * coeff_imag

    def _sparse_matrix_from_function_of_laplacian(self, f):
        """
        Returns as a sparse matrix acting on components the
        a function of the Laplacian.
        """
        values = np.zeros(self.dim // 2 + 1)
        for k in range(self.dim // 2 + 1):
            values[k] = f(k)
        return diags([values], [0])

    def _to_componets(self, u):
        coeff = self._to_coefficient(u)
        return self._coefficient_to_component(coeff)

    def _from_componets(self, c):
        coeff = self._component_to_coefficient(c)
        return self._from_coefficient(coeff)

    def _inner_product(self, u1, u2):
        coeff1 = self._to_coefficient(u1)
        coeff2 = self._to_coefficient(u2)
        return np.real(np.vdot(self._metric @ coeff1, coeff2))

    def _to_dual(self, u):
        coeff = self._to_coefficient(u)
        cp = self._coefficient_to_component(self._metric @ coeff)
        return self.dual.from_components(cp)

    def _from_dual(self, up):
        cp = self.dual.to_components(up)
        coeff = self._component_to_coefficient(cp)
        c = self._coefficient_to_component(self._inverse_metric @ coeff)
        return self.from_components(c)


class Lebesgue(Sobolev):
    """
    L2 space on a circle. Instance of the Sobolev spcae class when the exponent vanishes.
    """

    def __init__(
        self,
        kmax,
        /,
        *,
        radius=1,
    ):
        super().__init__(kmax, 0, 0, radius=radius)
