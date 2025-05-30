"""
This module attempt at functions on an line using Foutier transforms.
But now using the Fourier coefficients as the coordinates.
"""

from math import ceil
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
    Implementation of the Sobolev space H^s on a real line.
    """

    def __init__(
        self,
        left_boundary_point,
        right_boundary_point,
        sample_point_spacing,
        exponent,
        scale,
        /,
        *,
        power_of_two=False,
    ):
        """
        Args:
            left_boundary_point (float): left end of the line.
            right_boundary_point (float): right end of the line.
            sample_point_spacing (float): the spacing of the sample points.
            exponent (float): Sobolev exponent.
            scale (float): Sobolev scale.
            power_of_two (bool): Make the number of sample points a power of two. Default is false.

        Notes:
            If left_boundary_point > right_boundary_point on input, these values are swapped internally.

        Raises:

            ValueError: If left_boundary_point == right_boundary_point.
        """

        # Work out the number of sample points and their spacing.
        if left_boundary_point == right_boundary_point:
            raise ValueError("Interval has zero length")

        if right_boundary_point > left_boundary_point:
            self._left_boundary_point = left_boundary_point
            self._right_boundary_point = right_boundary_point
        else:
            self._left_boundary_point = right_boundary_point
            self._right_boundary_point = left_boundary_point

        self._number_of_sample_points = ceil(
            (self._right_boundary_point - self._left_boundary_point)
            / sample_point_spacing
        )

        if power_of_two:
            m = int(np.log2(self._number_of_sample_points)) + 1
            self._number_of_sample_points = 2**m

        super().__init__(
            self.number_of_sample_points,
            self._to_componets,
            self._from_componets,
            self._inner_product,
            self._to_dual,
            self._from_dual,
        )

        self._sample_point_spacing = (
            self._right_boundary_point - self._left_boundary_point
        ) / self._number_of_sample_points

        self._exponent = exponent
        self._scale = scale

        self._fft_factor = np.sqrt(
            self.sample_point_spacing / self.number_of_sample_points
        )
        self._inverse_fft_factor = 1 / self._fft_factor

        # indexing information for component vectors
        self._real_start = 0
        self._real_finish = self.dim // 2 + 1
        self._real_k_start = 0
        self._imag_start = self._real_finish
        self._imag_finish = self.dim
        self._imag_k_start = 1

        # Set up sparse matrices for inner product and Riesz mappings
        self._metric = self._sparse_matrix_from_function_of_laplacian(
            lambda k: (2 if k > 1 else 1) * self._sobolev_function(k)
        )

        self._inverse_metric = self._sparse_matrix_from_function_of_laplacian(
            lambda k: (1 / 2 if k > 1 else 1) / self._sobolev_function(k)
        )

        """
        self._riesz_matrix = self._sparse_matrix_from_function_of_laplacian(
            lambda k: 1 / self._sobolev_function(k)
        )

        self._inverse_riesz_matrix = self._sparse_matrix_from_function_of_laplacian(
            self._sobolev_function
        )
        """

    @property
    def left_boundary_point(self):
        """
        Return left boundary point for computational domain.
        """
        return self._left_boundary_point

    @property
    def right_boundary_point(self):
        """
        Return right boundary point for computational domain.
        """
        return self._right_boundary_point

    @property
    def line_width(self):
        """
        Return the width of the line.
        """
        return self.right_boundary_point - self.left_boundary_point

    @property
    def number_of_sample_points(self):
        """
        Return number of sample points.
        """
        return self._number_of_sample_points

    @property
    def sample_point_spacing(self):
        """
        Return the spacing of the sample points.
        """
        return self._sample_point_spacing

    @property
    def exponent(self):
        """
        Return the Sobolev exponent.
        """
        return self._exponent

    @property
    def scale(self):
        """
        Return the Sobolev exponent.
        """
        return self._scale

    def sample_points(self):
        """
        Returns a numpy array of the sample points.
        """
        return np.fromiter(
            [
                self.left_boundary_point + i * self._sample_point_spacing
                for i in range(self.number_of_sample_points)
            ],
            float,
        )

    def project_function(self, f):
        """
        Returns an element of the space formed by projecting a given function
        on the sample points.
        """
        return np.fromiter([f(x) for x in self.sample_points()], float)

    def plot(self, u, *args, **kwargs):
        """
        Make a simple plot of an element of the space on the computational domain.
        Optional arguements are forwarded through to the pylot.plot method.
        """
        plt.plot(self.sample_points(), u, *args, **kwargs)

    def plot_error_bounds(self, ubar, ustd, *args, **kwargs):
        """
        Make a plot of an element of the space bounded above and below by a standard
        deviation curve.
        """
        plt.fill_between(
            self.sample_points(), ubar - ustd, ubar + ustd, *args, **kwargs
        )

    def invariant_automorphism(self, f):
        """
        Returns a linear operator on the space of the form f(\Delta) with \Delta the Laplacian.
        The resulting operator is only well-defined as a continuous automorphism if f is bounded.
        """

        matrix = self._sparse_matrix_from_function_of_laplacian(f)
        return LinearOperator.from_matrix(self, self, matrix, galerkin=True)

    def invariant_measure(self, f):
        """
        Returns a Gaussian measure on the space whose covariance takes the form f(\Delta) with
        \Delta the Laplacian. The trace-class condition for the measure to be well-defined implies
        that the sequence {f(k)} must be summable.
        """

        matrix = self._sparse_matrix_from_function_of_laplacian(lambda k: np.sqrt(f(k)))
        covariance_factor = LinearOperator.from_matrix(
            EuclideanSpace(self.dim), self, matrix, galerkin=True
        )
        return GaussianMeasure(covariance_factor=covariance_factor)

    def sobolev_measure(self, exponent, scale, /, *, amplitude=1):
        """
        Returns a Gaussian measure with Sobolev covariance. The measure is
        scaled such that the its pointwise standard deviation is equal
        to the optional amplitude (default value 1).
        """
        mu = self.invariant_measure(lambda k: (1 + (scale * k) ** 2) ** -exponent)
        x = self.left_boundary_point + 0.5 * self.line_width
        u = self.dirac_representation(x)
        var = self.inner_product(mu.covariance(u), u)
        mu *= amplitude / np.sqrt(var)
        return mu

    def dirac(self, x):
        """
        Returns the dirac measure based at x as a LinearForm on the space.
        """
        # if self.exponent <= 0.5:
        #    raise ValueError("Dirac measure not well-defined on the space")

        coeff = np.zeros(self.dim // 2 + 1, dtype=complex)
        fac = np.exp(-2 * np.pi * 1j * x / self.line_width)
        coeff[0] = 1
        for k in range(1, coeff.size):
            coeff[k] = coeff[k - 1] * fac
        coeff *= 1 / np.sqrt(self.line_width)
        coeff[1:] *= 2
        cp = self._coefficient_to_component(coeff)
        return LinearForm(self, components=cp)

    def dirac_representation(self, x):
        """
        Returns the representation of the dirac measure based at x.
        """
        return self.from_dual(self.dirac(x))

    def point_evaluation_operator(self, xs):
        """
        Returns the point evaluation operator on the space
        at the given list, xs, of points.
        """

        # if self.exponent <= 0.5:
        #    raise ValueError("Point evaluation not well-defined on the space")

        dim = len(xs)
        matrix = np.zeros((dim, self.dim))

        for i, x in enumerate(xs):
            cp = self.dirac(x).components
            matrix[i, :] = cp

        return LinearOperator.from_matrix(
            self, EuclideanSpace(dim), matrix, galerkin=True
        )

    def random_points(self, n, /, *, left=None, right=None):
        """
        Returns n random points in the line drawn from a uniform distribution.
        """
        if left is None:
            left = self.left_boundary_point
        if right is None:
            right = self.right_boundary_point
        return np.random.uniform(left, right, n)

    # ================================================================#
    #                           Private methods                      #
    # ================================================================#

    def _sobolev_function(self, k):
        return (1 + (self.scale * k) ** 2) ** self.exponent

    def _coefficient_to_component(self, coeff):
        return np.concatenate(
            [coeff.real, coeff.imag[1 : self.dim // 2 + self.dim % 2]]
        )

    def _component_to_coefficient(self, c):
        coeff_real = c[: self.dim // 2 + 1]
        coeff_imag = np.concatenate(
            [[0], c[self.dim // 2 + 1 :], np.zeros(1 if self.dim % 2 == 0 else 0)]
        )
        return coeff_real + 1j * coeff_imag

    def _sparse_matrix_from_function_of_laplacian(self, f):
        """
        Returns as a sparse matrix acting on components the
        a function of the Laplacian.
        """
        values = np.zeros(self.dim)
        outer_fac = 2 * np.pi / self.line_width
        k = self._real_k_start
        for i in range(self._real_start, self._real_finish):
            values[i] = f(outer_fac * k)
            k += 1

        k = self._imag_k_start
        for i in range(self._imag_start, self._imag_finish):
            values[i] = f(outer_fac * k)
            k += 1
        return diags([values], [0])

    def _to_componets(self, x):
        coeff = rfft(x) * self._fft_factor
        return self._coefficient_to_component(coeff)

    def _from_componets(self, c):
        coeff = self._component_to_coefficient(c) * self._inverse_fft_factor
        return irfft(coeff, n=self.number_of_sample_points)

    def _inner_product(self, x1, x2):
        c1 = self.to_components(x1)
        c2 = self.to_components(x2)
        return np.dot(self._metric @ c1, c2)

    def _to_dual(self, x):
        c = self.to_components(x)
        cp = self._metric @ c
        return self.dual.from_components(cp)

    def _from_dual(self, xp):
        cp = self.dual.to_components(xp)
        c = self._inverse_metric @ cp
        return self.from_components(c)


class Lebesgue(Sobolev):
    """
    L2 space on a line. Instance of the Sobolev spcae class when the exponent vanishes.
    """

    def __init__(
        self,
        left_boundary_point,
        right_boundary_point,
        sample_point_spacing,
        /,
        *,
        power_of_two=False,
    ):
        super().__init__(
            left_boundary_point,
            right_boundary_point,
            sample_point_spacing,
            0,
            0,
            power_of_two=power_of_two,
        )
