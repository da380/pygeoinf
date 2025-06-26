"""
Module containing the base class for Sobolev spaces defined on homogeneous spaces. 
"""

from abc import ABC, abstractmethod
import numpy as np
from pygeoinf.hilbert_space import HilbertSpace, LinearOperator, EuclideanSpace


class HomogeneousSpaceSobolev(HilbertSpace, ABC):
    """
    Base class for Sobolev spaces of scalar fields on a homogeneous space.
    """

    def __init__(
        self,
        order,
        scale,
        dim,
        to_components,
        from_components,
        inner_product,
        to_dual,
        from_dual,
        /,
        *,
        add=None,
        subtract=None,
        multiply=None,
        axpy=None,
        copy=None,
        vector_multiply=None,
    ):
        """
        Args:
            order (float): Order of the Sobolev space.
            scale (float): Scale for the Sobolev space.
            dim (int): The dimension of the space, or of the
                finite-dimensional approximating space.
            to_components (callable):  A functor that maps vectors
                to their components.
            from_components (callable): A functor that maps components
                to vectors.
            inner_product (callable): A functor the implements the inner
                product on the space.
            to_dual (callable | None): A funcator that maps a vector
                to the cannonically associated dual vector.
            from_dual (callable | None): A functor that maps a dual vector
                to its representation on the space.
            add (callable): Implements vector addition.
            subtract (callable): Implements vector subtraction.
            multiply (callable): Implements scalar multiplication.
            axpy (callable): Implements the mapping y -> a*x + y
            copy (callable): Implements deep copy of a vector.
        """

        self._order = order

        if scale <= 0:
            raise ValueError("Scale must be positive")
        self._scale = scale

        super().__init__(
            dim,
            to_components,
            from_components,
            inner_product,
            to_dual,
            from_dual,
            add=add,
            subtract=subtract,
            multiply=multiply,
            axpy=axpy,
            copy=copy,
            vector_multiply=vector_multiply,
        )

    @property
    def order(self):
        """
        Return the Sobolev order.
        """
        return self._order

    @property
    def scale(self):
        """
        Return the Sobolev scale.
        """
        return self._scale

    @abstractmethod
    def random_point(self):
        """
        Returns a random point within the homogeneous space.
        """

    def random_points(self, n):
        """
        Returns a list of n random points.
        """
        return [self.random_point() for _ in range(n)]

    @abstractmethod
    def dirac(self, point):
        """
        Returns the diract measure based at the given point.
        """

    def dirac_representation(self, point):
        """
        Returns the representation of the Dirac measure based
        at the given point.
        """
        return self.from_dual(self.dirac(point))

    def point_evaluation_operator(self, points):
        """
        Returns as a linear operator the mapping from a function to
        its values at the list of points.
        """
        dim = len(points)
        matrix = np.zeros((dim, self.dim))

        for i, point in enumerate(points):
            cp = self.dirac(point).components
            matrix[i, :] = cp

        return LinearOperator.from_matrix(
            self, EuclideanSpace(dim), matrix, galerkin=True
        )

    @abstractmethod
    def invariant_automorphism(self, f):
        """
        Returns an automorphism of the form f(Delta) with
        Delta the Laplacian.
        """

    @abstractmethod
    def invariant_gaussian_measure(self, f, /, *, expectation=None):
        """
        Returns a Gaussian measure whose covariance takes the form
        f(Delta) with Delta the Laplaceian. An expectation can be
        provided, with the default being zero.
        """

    def _transform_measure(self, amplitude, expectation, mu):
        # Scales an invariant measure to get the right pointwise
        # variance and then adds the expected value.
        Q = mu.covariance
        u = self.dirac_representation(self.random_point())
        var = self.inner_product(Q(u), u)
        mu *= amplitude / np.sqrt(var)
        return mu.affine_mapping(translation=expectation)

    def sobolev_gaussian_measure(self, order, scale, amplitude, /, *, expectation=None):
        """
        Returns an invariant Gaussian measure whose covariance takes the Sobolev form.

        Args:
            order (float): Order parameter. Must be > 1.
            scale (float): Scale paramerer. Must be > 0.
            amplitude (float): Pointwise standard deviation. Must be > 0.
            expectation (vector): Expectation. Default is None.
        """
        mu = self.invariant_gaussian_measure(lambda k: (1 + scale**2 * k) ** -order)
        return self._transform_measure(amplitude, expectation, mu)

    def heat_gaussian_measure(self, scale, amplitude, /, *, expectation=None):
        """
        Returns an invariant Gaussian measure whose covariance takes the heat kernel form.

        Args:
            scale (float): Scale paramerer. Must be > 0.
            amplitude (float): Pointwise standard deviation. Must be > 0.
            expectation (vector): Expectation. Default is None.
        """
        mu = self.invariant_gaussian_measure(lambda k: np.exp(-(scale**2) * k))
        return self._transform_measure(amplitude, expectation, mu)
