"""
Provides an abstract framework for function spaces on symmetric manifolds.

This module offers a powerful abstract framework for defining Hilbert spaces of
functions on symmetric spaces (like spheres, tori, etc.). The core design
leverages the spectral properties of the Laplace-Beltrami operator (Δ), which
is fundamental to the geometry of these spaces.

By inheriting from these base classes and implementing a few key abstract
methods (like the Laplacian eigenvalues), a concrete class can automatically
gain a rich set of tools for defining invariant operators and probability
measures. This is a cornerstone of fields like spatial statistics and
geometric machine learning.

Key Classes
-----------
AbstractInvariantLebesgueSpace
    An abstract base class for L²-type spaces. It provides methods to construct
    operators that are functions of the Laplacian (`f(Δ)`) and to build
    rotationally-invariant Gaussian measures (e.g., with Sobolev or heat
    kernel covariances).

AbstractInvariantSobolevSpace
    An abstract base class for Sobolev spaces (Hˢ). It extends the Lebesgue
    functionality with features that require higher smoothness, most notably
    point evaluation via Dirac delta functionals, which is essential for
    connecting the abstract function space to discrete data points.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Any, List
import numpy as np
from scipy.sparse import diags

from pygeoinf.hilbert_space import (
    EuclideanSpace,
)
from pygeoinf.operators import LinearOperator
from pygeoinf.linear_forms import LinearForm
from pygeoinf.gaussian_measure import GaussianMeasure


class AbstractInvariantLebesgueSpace(ABC):
    """
    An abstract base class that provides functionality for Lebesgue spaces defined
    over a symmetric space.
    """

    @property
    @abstractmethod
    def spatial_dimension(self):
        """The dimension of the symetric space."""

    @abstractmethod
    def _space(self):
        """The Hilbert space."""

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
    def laplacian_eigenvalue(self, k: int | tuple[int, ...]) -> float:
        """
        Returns the k-th eigenvalue of the Laplacian. Note that the index can
        either be an integer or a tuple of integers.

        Args:
            k: The index of the eigenvalue to return.
        """

    @abstractmethod
    def invariant_automorphism(self, f: Callable[[float], float]) -> LinearOperator:
        """
        Returns an automorphism of the form f(Δ) with f a function
        that is well-defined on the spectrum of the Laplacian, Δ.

        In order to be well-defined, the function must have appropriate
        growth properties. For example, in an L² space we need f to be bounded.
        In Sobolev spaces Hˢ a more complex condition holds depending on the
        Sobolev order. These conditions on the function are not checked.

        Args:
            f: A real-valued function that is well-defined on the spectrum
               of the Laplacian.
        """

    @abstractmethod
    def trace_of_invariant_automorphism(self, f: Callable[[float], float]) -> float:
        """
        Returns the trace of the automorphism of the form f(Δ) with f a function
        that is well-defined on the spectrum of the Laplacian.

        Args:
            f: A real-valued function that is well-defined on the spectrum
               of the Laplacian.
        """

    def invariant_gaussian_measure(
        self,
        f: Callable[[float], float],
    ):
        """
        Returns a Gaussian measure with covariance of the form f(Δ) with f a function
        that is well-defined on the spectrum of the Laplacian.

        To be mathematically well-defined, the operator f(Δ) must be trace-class,
        this imposing a condition on the growth of f that is not checked.

        Args:
            f: A real-valued function that is well-defined on the spectrum
               of the Laplacian, Δ.

        Notes:
            The implementation of this method assumes that the basis for the
            HilbertSpace is comprised of orthogonal eigenvectors for the
            Laplacian. It is not necessary for this basis to be normalised.
        """

        space = self._space()

        values = np.fromiter(
            [1 / space.norm(space.basis_vector(i)) for i in range(space.dim)],
            dtype=float,
        )
        matrix = diags([values], [0])
        inverse_matrix = diags([np.reciprocal(values)], [0])

        def mapping(c: np.ndarray) -> np.ndarray:
            return space.from_components(matrix @ c)

        def adjoint_mapping(u: np.ndarray) -> np.ndarray:
            c = space.to_components(u)
            return inverse_matrix @ c

        component_mapping = LinearOperator(
            EuclideanSpace(space.dim), self, mapping, adjoint_mapping=adjoint_mapping
        )
        sqrt_covariance = self.invariant_automorphism(lambda k: np.sqrt(f(k)))

        covariance_factor = sqrt_covariance @ component_mapping

        return GaussianMeasure(covariance_factor=covariance_factor)

    def norm_scaled_invariant_gaussian_measure(
        self, f: Callable[[float], float], std: float
    ) -> GaussianMeasure:
        """
        Returns a Gaussian measure whose covariance is proportional to f(Δ) with
        f a function that is well-defined on the spectrum of the Laplacian, Δ.

        In order to be well-defined, f(Δ) must be trace class, with this implying
        decay conditions on f whose form depends on the form of the symmetric space.
        These conditions on the function are not checked.

        The measure's covariance is scaled such that the expected value for the
        samples norm is equal to the given standard deviation.

        Args:
            f: A real-valued function that is well-defined on the spectrum
            of the Laplacian.
            std: The desired standard deviation for the norm of samples.
        """
        mu = self.invariant_gaussian_measure(f)
        tr = self.trace_of_invariant_automorphism(f)
        return (std / np.sqrt(tr)) * mu

    def sobolev_kernel_gaussian_measure(self, order: float, scale: float):
        """
        Returns an invariant Gaussian measure with a Sobolev-type covariance
        equal to (1 + scale^2 * Δ)^-order.

        Args:
            order: Order parameter for the covariance.
            scale: Scale parameter for the covariance.
        """
        return self.invariant_gaussian_measure(lambda k: (1 + scale**2 * k) ** -order)

    def norm_scaled_sobolev_kernel_gaussian_measure(
        self, order: float, scale: float, std: float
    ):
        """
        Returns an invariant Gaussian measure with a Sobolev-type covariance
        proportional to (1 + scale^2 * Δ)^-order.

        The measure's covariance is scaled such that the expected value for the
        samples norm is equal to the given standard deviation.

        Args:
            order: Order parameter for the covariance.
            scale: Scale parameter for the covariance.
            std: The desired standard deviation for the norm of samples.
        """
        return self.norm_scaled_invariant_gaussian_measure(
            lambda k: (1 + scale**2 * k) ** -order, std
        )

    def heat_kernel_gaussian_measure(self, scale: float):
        """
        Returns an invariant Gaussian measure with a heat kernel covariance
        equal to exp(-scale^2 * Δ).

        Args:
            scale: Scale parameter for the covariance.
        """
        return self.invariant_gaussian_measure(lambda k: np.exp(-(scale**2) * k))

    def norm_scaled_heat_kernel_gaussian_measure(self, scale: float, std: float):
        """
        Returns an invariant Gaussian measure with a heat kernel covariance
        proportional to exp(-scale^2 * Δ).

        The measure's covariance is scaled such that the expected value for the
        samples norm is equal to the given standard deviation.

        Args:
            scale: Scale parameter for the covariance.
            std: The desired standard deviation for the norm of samples.
        """
        return self.norm_scaled_invariant_gaussian_measure(
            lambda k: np.exp(-(scale**2) * k), std
        )


class AbstractInvariantSobolevSpace(AbstractInvariantLebesgueSpace):
    """
    An abstract base class that builds on AbstractInvariantLebesgueSpace to provide additional functionality
    for Sobolev spaces.
    """

    @property
    @abstractmethod
    def order(self) -> float:
        """The Sobolev order."""

    @property
    @abstractmethod
    def scale(self) -> float:
        """The Sobolev length-scale."""

    @abstractmethod
    def sobolev_function(self, k: float) -> float:
        """
        Implementation of the relevant Sobolev function for the space.
        """

    @abstractmethod
    def dirac(self, point: Any) -> LinearForm:
        """
        Returns the linear functional corresponding to a point evaluation.

        This represents the action of the Dirac delta measure based at the given
        point.

        Args:
            point: The point on the symmetric space at which to base the functional.

        Raises:
            NotImplementedError: If the Sobolev order is less than n/2, with n the spatial dimension.
        """

    def dirac_representation(self, point: Any) -> Any:
        """

        Returns the Riesz representation of the Dirac delta functional.

        This is the vector in the Hilbert space that represents point evaluation
        via the inner product.

        Args:
            point: The point on the symmetric space.

        Raises:
            NotImplementedError: If the Sobolev order is less than n/2, with n the spatial dimension.
        """
        return self._space().from_dual(self.dirac(point))

    def point_evaluation_operator(self, points: List[Any]) -> LinearOperator:
        """
        Returns a linear operator that evaluates a function at a list of points.

        The resulting operator maps a function (a vector in this space) to a
        vector in Euclidean space containing the function's values.

        Args:
            points: A list of points at which to evaluate the functions.

        Raises:
            NotImplementedError: If order <= n/2, where n is the space dimension.
        """
        if self.order <= self.spatial_dimension / 2:
            raise NotImplementedError("Order must be greater than n/2")

        space = self._space()
        dim = len(points)
        matrix = np.zeros((dim, space.dim))

        for i, point in enumerate(points):
            cp = self.dirac(point).components
            matrix[i, :] = cp

        return LinearOperator.from_matrix(
            space, EuclideanSpace(dim), matrix, galerkin=True
        )

    def invariant_automorphism(self, f: Callable[[float], float]):
        """
        Returns an invariant automorphism of the form f(Δ) making use of the equivalent
        operator on the underlying Lebesgue space.

        Args:
            f: A real-valued function that is well-defined on the spectrum
               of the Laplacian, Δ.
        """
        A = self._space().underlying_space.invariant_automorphism(f)
        return LinearOperator.from_formally_self_adjoint(self, A)

    def point_value_scaled_invariant_gaussian_measure(
        self, f: Callable[[float], float], amplitude: float
    ):
        """
        Returns an invariant Gaussian measure with covariance proportional to f(Δ),
        where f must be such that this operator is trace-class.

        The covariance of the operator is scaled such that the standard deviation
        of the point-wise values are equal to the given amplitude.

        Args:
            f: A real-valued function that is well-defined on the spectrum
               of the Laplacian, Δ.
            amplitude: The desired standard deviation for the pointwise values.

        Raises:
            NotImplementedError: If the Sobolev order is less than n/2, with n the spatial dimension.

        Notes:
            This method applies for symmetric spaces an invariant measures. As a result, the
            pointwise variance is the same at all points. Internally, a random point is chosen
            to carry out the normalisation.
        """
        space = self._space()
        point = self.random_point()
        u = self.dirac_representation(point)
        mu = self.invariant_gaussian_measure(f)
        cov = mu.covariance
        var = space.inner_product(cov(u), u)
        return (amplitude / np.sqrt(var)) * mu

    def point_value_scaled_sobolev_kernel_gaussian_measure(
        self, order: float, scale: float, amplitude: float
    ):
        """
        Returns an invariant Gaussian measure with a Sobolev-type covariance
        proportional to (1 + scale^2 * Δ)^-order.

        The covariance of the operator is scaled such that the standard deviation
        of the point-wise values are equal to the given amplitude.

        Args:
            order: Order parameter for the covariance.
            scale: Scale parameter for the covariance.
            amplitude: The desired standard deviation for the pointwise values.
        """
        return self.point_value_scaled_invariant_gaussian_measure(
            lambda k: (1 + scale**2 * k) ** -order, amplitude
        )

    def point_value_scaled_heat_kernel_gaussian_measure(
        self, scale: float, amplitude: float
    ):
        """
        Returns an invariant Gaussian measure with a heat-kernel covariance
        proportional to exp(-scale^2 * Δ).

        The covariance of the operator is scaled such that the standard deviation
        of the point-wise values are equal to the given amplitude.

        Args:
            scale: Scale parameter for the covariance.
            amplitude: The desired standard deviation for the pointwise values.
        """
        return self.point_value_scaled_invariant_gaussian_measure(
            lambda k: np.exp(-(scale**2) * k), amplitude
        )
