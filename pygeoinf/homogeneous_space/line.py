"""
Sobolev spaces for functions on a line. 
"""

import matplotlib.pyplot as plt
import numpy as np
from pygeoinf.hilbert_space import (
    HilbertSpace,
    LinearOperator,
    EuclideanSpace,
)
from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.homogeneous_space.circle import Sobolev as CicleSobolev


class Sobolev(HilbertSpace):
    """
    Implementation of the Sobolev space H^s on a line.
    """

    def __init__(
        self,
        kmax,
        exponent,
        length_scale,
        /,
        *,
        x0=0,
        x1=1,
    ):
        """
        Args:
            kmax (float): The maximum Fourier degree.
            exponent (float): Sobolev exponent.
            length_scale (float): Sobolev length-scale.
            radius (float): Radius of the circle. Default is 1.
            x0 (float): The left boudary of the interval, Default is 0.
            x1 (float): The right boundary of the interval. Default is 1.
        """

        if x0 >= x1:
            raise ValueError("Invalid interval parameters")

        if exponent != 0 and length_scale <= 0:
            raise ValueError("Length-scale must be positive")

        self._kmax = kmax
        self._x0 = x0
        self._x1 = x1
        self._exponent = exponent
        self._length_scale = length_scale

        # Work out transformation from a unit circle.
        padding_scale = 5 * length_scale if length_scale > 0 else 0.1 * (x1 - x0)
        number_of_points = 2 * kmax
        width = x1 - x0
        self._start_index = int(
            number_of_points * padding_scale / (width + 2 * padding_scale)
        )
        self._finish_index = 2 * kmax - self._start_index + 1
        self._padding_length = (
            self._start_index * width / (number_of_points - 2 * self._start_index)
        )

        self._jac = (width + 2 * self._padding_length) / (2 * np.pi)
        self._ijac = 1 / self._jac
        self._sqrt_jac = np.sqrt(self._jac)
        self._isqrt_jac = 1 / self._sqrt_jac

        # Set up the related Sobolev space on the unit circle.
        circle_length_scale = length_scale * self._ijac
        self._circle_space = CicleSobolev(kmax, exponent, circle_length_scale)

        super().__init__(
            self._circle_space.dim,
            self._to_components,
            self._from_components,
            self._inner_product,
            self._to_dual,
            self._from_dual,
            vector_multiply=lambda u1, u2: u1 * u2,
        )

    @property
    def kmax(self):
        """
        Return the maximum Fourier degree.
        """
        return self._kmax

    @property
    def x0(self):
        """
        Returns the left boundary point.
        """
        return self._x0

    @property
    def x1(self):
        """
        Returns the right boundary point.
        """
        return self._x1

    @property
    def width(self):
        """
        Return the radius.
        """
        return self._x1 - self._x0

    @property
    def exponent(self):
        """
        Return the Sobolev exponent.
        """
        return self._exponent

    @property
    def length_scale(self):
        """
        Return the relative Sobolev length_scale.
        """
        return self._length_scale

    @property
    def point_spacing(self):
        """
        Return the point spacing.
        """
        return self._circle_space.angle_spacing * self._jac

    def computational_points(self):
        """
        Returns a numpy array of the computational points.
        """
        return self._x0 - self._padding_length + self._jac * self._circle_space.angles()

    def points(self):
        """
        Returns a numpy array of the points.
        """
        return self.computational_points()[self._start_index : self._finish_index]

    def project_function(self, f):
        """
        Returns an element of the space formed by projecting a given function.
        """
        return np.fromiter(
            [f(x) * self._taper(x) for x in self.computational_points()], float
        )

    def plot(self, u, fig=None, ax=None, computational_domain=False, **kwargs):
        """
        Make a simple plot of an element of the space on the computational domain.

        Args:
            u (vector): The element of the space to be plotted.
            fig (Figure): An existing Figure object to use. Default is None.
            ax (Axes): An existing Axes object to use. Default is None.
            computatoinal_domain (bool): If True, plot the whole computational
                domain. Default is False.
            kwargs: Keyword arguments forwarded to plot.

        Returns
            Figure: The figure object, either that given or newly created.
            Axes: The axes object, either that given or newly created.
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
        self, u, u_bound, fig=None, ax=None, computational_domain=False, **kwargs
    ):
        """
        Make a plot of an element of the space bounded above and below by a standard
        deviation curve.

        Args:
            u (vector): The element of the space to be plotted.
            u_bounds (vector): A second element giving point-wise bounds.
            fig (Figure): An existing Figure object to use. Default is None.
            ax (Axes): An existing Axes object to use. Default is None.
            kwargs: Keyword arguments forwarded to plot.

        Returns
            Figure: The figure object, either that given or newly created.
            Axes: The axes object, either that given or newly created.
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

    def invariant_automorphism(self, f):
        """
        Returns a linear operator on the space of the form f(Delta) with Delta the Laplacian.
        The resulting operator is only well-defined as a continuous automorphism if f is bounded.
        """
        A = self._circle_space.invariant_automorphism(lambda k: self._ijac * k)
        return LinearOperator.formally_self_adjoint(self, A)

    def invariant_gaussian_measure(self, f, /, *, expectation=None):
        """
        Returns a Gaussian measure on the space whose covariance takes the form f(Delta) with
        Delta the Laplacian. The trace-class condition for the measure to be well-defined implies
        that the sequence {f(k)} must be summable.
        """
        mu = self._circle_space.invariant_gaussian_measure(
            lambda k: self._ijac * k, expectation=expectation
        )
        covariance = LinearOperator.self_adjoint(self, mu.covariance)
        return GaussianMeasure(
            covariance=covariance, expectation=expectation, sample=mu.sample
        )

    def sobolev_gaussian_measure(
        self, exponent, length_scale, amplitude, /, *, expectation=None
    ):
        """
        Returns a Gaussian measure with Sobolev covariance. The measure is
        scaled such that the its pointwise standard deviation is equal
        to the optional amplitude (default value 1).
        """
        mu = self._circle_space.sobolev_gaussian_measure(
            exponent, length_scale * self._ijac, amplitude, expectation=expectation
        )
        covariance = LinearOperator.self_adjoint(self, mu.covariance)
        return GaussianMeasure(
            covariance=covariance, expectation=expectation, sample=mu.sample
        )

    def heat_gaussian_measure(self, length_scale, amplitude, /, *, expectation=None):
        """
        Returns a Gaussian measure with heat kernel covariance. The measure is
        scaled such that the its pointwise standard deviation is equal
        to the optional amplitude (default value 1).
        """
        mu = self._circle_space.heat_gaussian_measure(
            length_scale * self._ijac, amplitude, expectation=expectation
        )
        covariance = LinearOperator.self_adjoint(self, mu.covariance)
        return GaussianMeasure(
            covariance=covariance, expectation=expectation, sample=mu.sample
        )

    def dirac(self, x):
        """
        Returns a dirac measure as a LinearForm on the space.
        """
        theta = self._inverse_transformation(x)
        up = self._circle_space.dirac(theta)
        cp = self._circle_space.dual.to_components(up) * self._isqrt_jac
        return self.dual.from_components(cp)

    def dirac_representation(self, x):
        """
        Returns the representation of the dirac measure based at the point.
        """
        return self.from_dual(self.dirac(x))

    def point_evaluation_operator(self, points):
        """
        Returns the point evaluation operator on the space
        at the given list, xs, of points.
        """
        dim = len(points)
        matrix = np.zeros((dim, self.dim))

        for i, point in enumerate(points):
            cp = self.dirac(point).components
            matrix[i, :] = cp

        return LinearOperator.from_matrix(
            self, EuclideanSpace(dim), matrix, galerkin=True
        )

    # =============================================================#
    #                        Private methods                       #
    # =============================================================#

    def _step(self, x):
        if x > 0:
            return np.exp(-1 / x)
        else:
            return 0

    def _bump_up(self, x, x1, x2):
        s1 = self._step(x - x1)
        s2 = self._step(x2 - x)
        return s1 / (s1 + s2)

    def _bump_down(self, x, x1, x2):
        s1 = self._step(x2 - x)
        s2 = self._step(x - x1)
        return s1 / (s1 + s2)

    def _taper(self, x):
        s1 = self._bump_up(x, self._x0 - self._padding_length, self._x0)
        s2 = self._bump_down(x, self._x1, self._x1 + self._padding_length)
        return s1 * s2

    def _transformation(self, th):
        return self._x0 - self._padding_length + self._jac * th

    def _inverse_transformation(self, x):
        return (x - self._x0 + self._padding_length) * self._ijac

    def _to_components(self, u):
        c = self._circle_space.to_components(u)
        c *= self._sqrt_jac
        return c

    def _from_components(self, c):
        c *= self._isqrt_jac
        u = self._circle_space.from_components(c)
        return u

    def _inner_product(self, u1, u2):
        return self._jac * self._circle_space.inner_product(u1, u2)

    def _to_dual(self, u):
        up = self._circle_space.to_dual(u)
        cp = self._circle_space.dual.to_components(up) * self._sqrt_jac
        return self.dual.from_components(cp)

    def _from_dual(self, up):
        cp = self.dual.to_components(up)
        vp = self._circle_space.dual.from_components(cp) * self._isqrt_jac
        return self._circle_space.from_dual(vp)
