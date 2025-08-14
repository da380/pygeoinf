"""
Module containing the base class for Sobolev spaces defined on symmetric spaces.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Any, List, Optional
import numpy as np

from pygeoinf.hilbert_space import HilbertSpace, EuclideanSpace
from pygeoinf.operators import LinearOperator
from pygeoinf.forms import LinearForm
from pygeoinf.gaussian_measure import GaussianMeasure


class SymmetricSpaceSobolev(HilbertSpace, ABC):
    """
    An abstract base class for Sobolev spaces of scalar fields on a symmetric space.

    This class provides a common interface for defining function spaces with spatial
    correlation, typically used in applications like geostatistics and machine
    learning on manifolds.
    """

    def __init__(
        self,
        order: float,
        scale: float,
        dim: int,
        to_components: Callable[[Any], np.ndarray],
        from_components: Callable[[np.ndarray], Any],
        inner_product: Callable[[Any, Any], float],
        to_dual: Callable[[Any], "LinearForm"],
        from_dual: Callable[["LinearForm"], Any],
        /,
        *,
        add: Optional[Callable[[T_vec, T_vec], T_vec]] = None,
        subtract: Optional[Callable[[T_vec, T_vec], T_vec]] = None,
        multiply: Optional[Callable[[float, T_vec], T_vec]] = None,
        ax: Optional[Callable[[float, T_vec], None]] = None,
        axpy: Optional[Callable[[float, T_vec, T_vec], None]] = None,
        copy: Optional[Callable[[T_vec], T_vec]] = None,
        vector_multiply: Optional[Callable[[T_vec, T_vec], T_vec]] = None,
    ) -> None:
        """
        Args:
            order: The order of the Sobolev space, related to its smoothness.
            scale: The characteristic length scale of the Sobolev space.
            dim: The dimension of the space, or of the finite-dimensional
                approximating space.
            to_components: A callable that maps vectors to their component arrays.
            from_components: A callable that maps component arrays to vectors.
            inner_product: A callable that implements the inner product on the space.
            to_dual: A callable that maps a vector to the canonically
                associated dual vector (a LinearForm).
            from_dual: A callable that maps a dual vector back to its
                primal representation in the space.
            add: Custom implementation for vector addition.
            subtract: Custom implementation for vector subtraction.
            multiply: Custom implementation for scalar multiplication.
            axpy: Custom implementation for the mapping `y -> a*x + y`.
            copy: Custom implementation for a deep copy of a vector.
            vector_multiply: Custom implementation for element-wise vector
                multiplication.
        """
        self._order: float = order

        if scale <= 0:
            raise ValueError("Scale must be positive")
        self._scale: float = scale

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
            ax=ax,
            axpy=axpy,
            copy=copy,
            vector_multiply=vector_multiply,
        )

    @property
    def order(self) -> float:
        """The order of the Sobolev space, controlling smoothness."""
        return self._order

    @property
    def scale(self) -> float:
        """The characteristic length scale of the Sobolev space."""
        return self._scale

    @abstractmethod
    def random_point(self) -> Any:
        """Returns a single random point from the underlying symmetric space."""

    def random_points(self, n: int) -> List[Any]:
        """
        Returns a list of `n` random points.

        Args:
            n: The number of random points to generate.
        """
        return [self.random_point() for _ in range(n)]

    @abstractmethod
    def dirac(self, point: Any) -> "LinearForm":
        """
        Returns the linear functional corresponding to a point evaluation.

        This represents the action of the Dirac delta measure based at the given
        point.

        Args:
            point: The point on the symmetric space at which to base the functional.
        """

    def dirac_representation(self, point: Any) -> Any:
        """

        Returns the Riesz representation of the Dirac delta functional.

        This is the vector in the Hilbert space that represents point evaluation
        via the inner product.

        Args:
            point: The point on the symmetric space.
        """
        return self.from_dual(self.dirac(point))

    def point_evaluation_operator(self, points: List[Any]) -> "LinearOperator":
        """
        Returns a linear operator that evaluates a function at a list of points.

        The resulting operator maps a function (a vector in this space) to a
        vector in Euclidean space containing the function's values.

        Args:
            points: A list of points at which to evaluate the functions.
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
    def invariant_automorphism(self, f: Callable[[float], float]) -> "LinearOperator":
        """
        Returns an automorphism of the form `f(Delta)` where `Delta` is the Laplacian.

        This uses functional calculus on the Laplace-Beltrami operator to create
        operators that are invariant to the symmetries of the space.

        Args:
            f: A scalar function to apply to the eigenvalues of the Laplacian.
        """

    @abstractmethod
    def invariant_gaussian_measure(
        self, f: Callable[[float], float], /, *, expectation: Optional[Any] = None
    ) -> "GaussianMeasure":
        """
        Returns a Gaussian measure with a covariance of the form `f(Delta)`.

        The covariance operator is defined via functional calculus on the Laplacian,
        ensuring the resulting measure is statistically invariant under the
        symmetries of the space.

        Args:
            f: The scalar function defining the covariance operator.
            expectation: The mean of the measure. Defaults to zero.
        """

    def _transform_measure(
        self, amplitude: float, expectation: Optional[Any], mu: "GaussianMeasure"
    ) -> "GaussianMeasure":
        """
        Scales an invariant measure to match a target pointwise standard deviation.

        Note:
            This method assumes statistical homogeneity; it calculates the current
            variance at a single random point and scales accordingly. The result
            may vary slightly on each call due to this randomness.
        """
        Q = mu.covariance
        u = self.dirac_representation(self.random_point())
        var = self.inner_product(Q(u), u)
        # Handle the case where variance is zero to avoid division errors
        if var > 0:
            mu *= amplitude / np.sqrt(var)
        return mu.affine_mapping(translation=expectation)

    def sobolev_gaussian_measure(
        self,
        order: float,
        scale: float,
        amplitude: float,
        /,
        *,
        expectation: Optional[Any] = None,
    ) -> "GaussianMeasure":
        """
        Returns an invariant Gaussian measure with a Sobolev-type covariance.

        The covariance operator is `C = (1 + scale^2 * Delta)^-order`, which is
        commonly used to generate spatially correlated random fields.

        Args:
            order: Order parameter for the covariance.
            scale: Scale parameter for the covariance.
            amplitude: The target pointwise standard deviation of the field.
            expectation: The expectation (mean field). Defaults to zero.
        """
        mu = self.invariant_gaussian_measure(lambda k: (1 + scale**2 * k) ** -order)
        return self._transform_measure(amplitude, expectation, mu)

    def heat_gaussian_measure(
        self, scale: float, amplitude: float, /, *, expectation: Optional[Any] = None
    ) -> "GaussianMeasure":
        """
        Returns an invariant Gaussian measure with a heat kernel covariance.

        The covariance operator is `C = exp(-scale^2 * Delta)`, which corresponds
        to the squared-exponential or Gaussian kernel.

        Args:
            scale: Scale parameter for the covariance.
            amplitude: The target pointwise standard deviation of the field.
            expectation: The expectation (mean field). Defaults to zero.
        """
        mu = self.invariant_gaussian_measure(lambda k: np.exp(-(scale**2) * k))
        return self._transform_measure(amplitude, expectation, mu)
