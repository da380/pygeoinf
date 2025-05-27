"""
Module for Lebesgue and Sobolev spaces on the real line based around FFT methods. 
"""

import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.stats import uniform
from scipy.fft import rfft, irfft
from scipy.sparse import diags
from pygeoinf.hilbert import HilbertSpace, LinearForm, LinearOperator, EuclideanSpace


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
            left_boundary_point (float): left end of the interval.
            right_boundary_point (float): right end of the interval.
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

        self._number_of_sample_points = math.ceil(
            (self._right_boundary_point - self._left_boundary_point)
            / sample_point_spacing
        )

        if power_of_two:
            m = int(np.log2(self._number_of_sample_points)) + 1
            self._number_of_sample_points = 2**m

        self._sample_point_spacing = (
            self._right_boundary_point - self._left_boundary_point
        ) / self._number_of_sample_points

        self._exponent = exponent
        self._scale = scale

        # Set up the metric tensor
        values = np.zeros(self.number_of_sample_points // 2 + 1)
        values[0] = 1
        outer_fac = 2 * np.pi * self._scale / self.interval_width
        for k in range(1, values.size):
            values[k] = 2 * (1 + (outer_fac * k) ** 2) ** (self._exponent)
        values *= self.sample_point_spacing / self.number_of_sample_points
        self._metric = diags([values], [0])

        # Set up the matrix needed in the Riesz mapping and its inverse.
        values = np.zeros(self.number_of_sample_points // 2 + 1)
        for k in range(values.size):
            values[k] = (1 + (outer_fac * k) ** 2) ** (self._exponent)
        values *= self.sample_point_spacing
        self._inverse_riesz_matrix = diags([values], [0])
        self._riesz_matrix = diags([np.reciprocal(values)], [0])

        # Initialise the vector space.
        super().__init__(
            self.number_of_sample_points,
            self._to_componets,
            self._from_components,
            self._inner_product,
            self._to_dual,
            self._from_dual,
        )

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
    def interval_width(self):
        """
        Return the width of the interval.
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

    def dirac_measure(self, x):
        """
        Return the Dirac measure at the point x as a linear form on the space.

        ValueError is raised if Sobolev exponent is not > 0.5
        """
        if self._exponent <= 0.5:
            raise ValueError("Dirac measure not well-defined")

        coeff = np.zeros(self.number_of_sample_points // 2 + 1, dtype=complex)
        fac = np.exp(-2 * np.pi * 1j * x / self.interval_width)
        coeff[0] = 1
        for k in range(1, coeff.size):
            coeff[k] = coeff[k - 1] * fac
        cp = irfft(coeff, self.number_of_sample_points)
        return LinearForm(self, components=cp)

    def dirac_representation(self, x):
        """
        Returns the represenation of the Dirac measure at the given point.

        ValueError is raised if Sobolev exponent is not > 0.5
        """
        return self.from_dual(self.dirac_measure(x))

    def point_evaluation_operator(self, points):
        """
        Returns as a linear opearator the mapping from the space
        to a specified set of point values.

        Args:
            points([float]): A list of the points.
        """

        dim = len(points)
        codomain = EuclideanSpace(dim)

        matrix = np.zeros((self.dim, dim))

        for i, x in enumerate(points):
            up = self.dirac_measure(x)
            matrix[:, i] = up.components

        return LinearOperator.from_matrix(self, codomain, matrix.T)

    def derivative_operator(self):
        """
        Returns as a LinearOperator the derative mapping from the space onto
        one whose Sobolev index is one lower.
        """

        codomain = Sobolev(
            self.left_boundary_point,
            self.right_boundary_point,
            self.sample_point_spacing,
            self.exponent - 1,
            self._scale,
        )

        fac = 2 * np.pi * 1j / self.interval_width
        values = np.fromiter(
            [fac * i for i in range(self.number_of_sample_points // 2 + 1)],
            dtype=complex,
        )
        matrix = diags([values], [0])

        def mapping(u):
            coeff = matrix @ rfft(u)
            return irfft(coeff, n=self.number_of_sample_points)

        def adjoint_mapping(u):
            return -1 * mapping(u)

        return LinearOperator(self, codomain, mapping, adjoint_mapping=adjoint_mapping)

    def laplacian_operator(self):
        """
        Returns the Laplacian as a LinearOperator between Sobolev spaces.
        """

        codomain = Sobolev(
            self.left_boundary_point,
            self.right_boundary_point,
            self.sample_point_spacing,
            self.exponent - 2,
            self._scale,
        )

        fac = -((2 * np.pi / self.interval_width) ** 2)
        values = np.fromiter(
            [fac * i for i in range(self.number_of_sample_points // 2 + 1)],
            dtype=complex,
        )
        matrix = diags([values], [0])

        def mapping(u):
            coeff = matrix @ rfft(u)
            return irfft(coeff, n=self.number_of_sample_points)

        return LinearOperator(self, codomain, mapping, adjoint_mapping=mapping)

    def random_points(self, n):
        """
        Returns a list of n random points in the interval drawn from  uniform distribution.
        """
        return np.random.uniform(self.left_boundary_point, self.right_boundary_point, n)

    def plot(self, u):
        """
        Make a simple plot of an element of the space on the computational domain.
        """
        plt.plot(self.sample_points(), self.to_components(u))

    def _to_componets(self, x):
        # Local implementation of to component mapping.
        return x

    def _from_components(self, c):
        # local implementation of from component mapping.
        return c

    def _inner_product(self, x1, x2):
        # Local implementation of inner product.
        coeff1 = rfft(x1)
        coeff2 = rfft(x2)
        return np.real(np.vdot(self._metric @ coeff1, coeff2))

    def _from_dual(self, xp):
        # local implementation of Riesz mapping.
        cp = xp.components
        coeff = self._riesz_matrix @ rfft(cp)
        return irfft(coeff, n=self.number_of_sample_points)

    def _to_dual(self, x):
        # local implementation of inverse Riesz mapping.
        coef = self._inverse_riesz_matrix @ rfft(x)
        cp = irfft(coef, n=self.number_of_sample_points)
        return self.dual.from_components(cp)
