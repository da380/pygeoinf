"""
Provides an abstract framework for function spaces on symmetric manifolds.

This module offers a powerful abstract framework for defining Hilbert spaces of
functions on symmetric spaces (like spheres or tori). The core design
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
    statistically isotropic (rotationally-invariant) Gaussian measures.

AbstractInvariantSobolevSpace
    An abstract base class for Sobolev spaces (Hˢ). It extends the Lebesgue
    functionality with features that require higher smoothness, most notably
    point evaluation via Dirac delta functionals, which is essential for

    connecting the abstract function space to discrete data points.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Any, List, Tuple, Optional


import numpy as np
from scipy.sparse import diags

from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.linear_forms import LinearForm
from pygeoinf.gaussian_measure import GaussianMeasure


class AbstractInvariantLebesgueSpace(ABC):
    """
    An abstract base class for L² function spaces on symmetric manifolds.

    This ABC defines the interface for spaces of square-integrable functions.
    It provides a powerful suite of methods for creating invariant
    operators and Gaussian measures by leveraging the spectrum of the
    Laplace-Beltrami operator.
    """

    @property
    @abstractmethod
    def spatial_dimension(self):
        """The dimension of the symetric space."""

    @property
    @abstractmethod
    def dim(self):
        """The dimension of the Hilbert space."""

    @abstractmethod
    def to_components(self, u: Any) -> np.ndarray:
        """Maps a vector `u` to its real component representation."""

    @abstractmethod
    def from_components(self, c: np.ndarray) -> Any:
        """Maps a real component vector back to a vector `u`."""

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
        Returns the eigenvalue of the Laplacian for a given mode index.

        The index `k` can be a single integer (e.g., for a circle) or a
        tuple of integers (e.g., for a sphere or torus), depending on the
        geometry of the space.

        Args:
            k: The index of the eigenvalue to return.
        """

    @abstractmethod
    def eigenfunction_norms(self) -> np.ndarray:
        """Returns a list of the norms of the eigenfunctions."""

    @abstractmethod
    def invariant_automorphism_from_index_function(
        self, g: Callable[[int | tuple[int, ...]], float]
    ) -> LinearOperator:
        """
        Returns an automorphism of the form f(Δ) with f a function
        that is well-defined on the spectrum of the Laplacian, Δ.

        In order to be well-defined, the function must have appropriate
        growth properties. For example, in an L² space we need f to be bounded.
        In Sobolev spaces Hˢ a more complex condition holds depending on the
        Sobolev order. These conditions on the function are not checked.

        For this method, the function f is given implicitly in terms of a
        function, g, of the eigenvalue indices for the space. Letting k(λ) be
        the index for eigenvalue λ, we then have f(λ) = g(k(λ)).

        Args:
            g: A function that takes an eigenvalue index and returns a real value.
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

    @abstractmethod
    def geodesic_quadrature(
        self, p1: Any, p2: Any, n_points: int
    ) -> Tuple[List[Any], np.ndarray]:
        """
        Returns quadrature points and weights for a geodesic between p1 and p2.

        Returns:
            points: List of manifold coordinates.
            weights: Integration weights scaled by the line element.
        """

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

        Notes:
            This method is a convenience wrapper for the more general
            `invariant_automorphism_from_index_function`. It could be
            overriden if computationally advantageous.
        """
        return self.invariant_automorphism_from_index_function(
            lambda k: f(self.laplacian_eigenvalue(k))
        )

    def invariant_gaussian_measure(
        self,
        f: Callable[[float], float],
    ):
        """
        Returns a Gaussian measure with covariance of the form f(Δ).

        The covariance operator of the resulting measure is `C = f(Δ)`, where `f`
        is a function defined on the spectrum of the Laplacian. To be a valid
        covariance, `f` must be non-negative.

        Args:
            f: A real-valued, non-negative function that is well-defined on the
               spectrum of the Laplacian, Δ.

        Notes:
            The implementation assumes the basis for the HilbertSpace consists
            of orthogonal eigenvectors of the Laplacian. The `component_mapping`
            operator used internally handles the mapping from a standard normal
            distribution in R^n to this (potentially non-normalized) eigenbasis.
        """

        values = self.eigenfunction_norms()
        matrix = diags([np.reciprocal(values)], [0])
        inverse_matrix = diags([values], [0])

        def mapping(c: np.ndarray) -> np.ndarray:
            return self.from_components(matrix @ c)

        def adjoint_mapping(u: np.ndarray) -> np.ndarray:
            c = self.to_components(u)
            return inverse_matrix @ c

        component_mapping = LinearOperator(
            EuclideanSpace(self.dim), self, mapping, adjoint_mapping=adjoint_mapping
        )
        sqrt_covariance = self.invariant_automorphism(lambda k: np.sqrt(f(k)))

        covariance_factor = sqrt_covariance @ component_mapping

        return GaussianMeasure(covariance_factor=covariance_factor)

    def norm_scaled_invariant_gaussian_measure(
        self, f: Callable[[float], float], std: float = 1
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
        return self.invariant_gaussian_measure(lambda k: (1 + scale**2 * k) ** (-order))

    def norm_scaled_sobolev_kernel_gaussian_measure(
        self, order: float, scale: float, std: float = 1
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

    def norm_scaled_heat_kernel_gaussian_measure(self, scale: float, std: float = 1):
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
    An ABC for Sobolev spaces (Hˢ) on symmetric manifolds.

    This class extends the Lebesgue space functionality to spaces of functions
    with a specified degree of smoothness (`order`). The primary motivation for
    using a Sobolev space is that for a sufficiently high order, point-wise
    evaluation of a function is a well-defined operation. This is critical for
    linking abstract function fields to discrete data points.
    """

    def __init__(self, order: float, scale: float):
        """
        Args:
            spatial_dimension: The dimension of the space.
            order: The Sobolev order.
            scale: The Sobolev length-scale.
        """

        self._order: float = order
        self._scale: float = scale

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

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """
        Checks for mathematical equality with another Sobolev space.

        Two spaces are considered equal if they are of the same type and have
        the same defining parameters.
        """

    @property
    def order(self) -> float:
        """The Sobolev order."""
        return self._order

    @property
    def scale(self) -> float:
        """The Sobolev length-scale."""
        return self._scale

    def sobolev_function(self, k: float) -> float:
        """
        Implementation of the relevant Sobolev function for the space.
        """
        return (1 + self.scale**2 * k) ** self.order

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
        return self.from_dual(self.dirac(point))

    def point_evaluation_operator(self, points: List[Any]) -> LinearOperator:
        """
        Returns a linear operator that evaluates a function at a list of points.

        The resulting operator maps a function (a vector in this space) to a
        vector in Euclidean space containing the function's values at the
        specified locations. This is the primary mechanism for creating a
        forward operator that links a function field to a set of discrete
        measurements.

        Args:
            points: A list of points at which to evaluate the functions.
        """
        if self.order <= self.spatial_dimension / 2:
            raise NotImplementedError("Order must be greater than n/2")

        dim = len(points)
        matrix = np.zeros((dim, self.dim))

        for i, point in enumerate(points):
            cp = self.dirac(point).components
            matrix[i, :] = cp

        return LinearOperator.from_matrix(
            self, EuclideanSpace(dim), matrix, galerkin=True
        )

    def invariant_automorphism_from_index_function(
        self, g: Callable[[int | tuple[int, ...]], float]
    ):
        """
        Returns an automorphism of the form f(Δ) with f a function
        that is well-defined on the spectrum of the Laplacian, Δ.

        In order to be well-defined, the function must have appropriate
        growth properties. For example, in an L² space we need f to be bounded.
        In Sobolev spaces Hˢ a more complex condition holds depending on the
        Sobolev order. These conditions on the function are not checked.

        For this method, the function f is given implicitly in terms of a
        function, g, of the eigenvalue indices for the space. Letting k(λ) be
        the index for eigenvalue λ, we then have f(λ) = g(k(λ)).

        Args:
            g: A function that takes an eigenvalue index and returns a real value.
        """
        A = self.underlying_space.invariant_automorphism_from_index_function(g)
        return LinearOperator.from_formally_self_adjoint(self, A)

    def point_value_scaled_invariant_gaussian_measure(
        self, f: Callable[[float], float], amplitude: float = 1
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
        point = self.random_point()
        u = self.dirac_representation(point)
        mu = self.invariant_gaussian_measure(f)
        cov = mu.covariance
        var = self.inner_product(cov(u), u)
        return (amplitude / np.sqrt(var)) * mu

    def point_value_scaled_sobolev_kernel_gaussian_measure(
        self, order: float, scale: float, amplitude: float = 1
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
        self, scale: float, amplitude: float = 1
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

    def geodesic_integral(
        self, p1: Any, p2: Any, n_points: Optional[int] = None
    ) -> LinearForm:
        """
        Returns a linear functional representing the line integral of a function
        along a geodesic path.

        This method approximates the integral :math:`\\int_{\\gamma} u(s) ds`, where
        :math:`\\gamma` is the shortest path (geodesic) connecting points `p1` and `p2`.
        The integral is represented as a :class:`LinearForm` in the dual space,
        constructed by summing weighted point evaluations (Dirac measures) along
        the path.

        For Hilbert spaces with a specified :attr:`scale`, the method can
        automatically determine the required quadrature density to resolve the
        smooth features of the space's sensitivity kernels.

        Args:
            p1 (Any): The starting point of the geodesic. The type is manifold-dependent
                (e.g., float for :class:`Circle`, tuple for :class:`Sphere`).
            p2 (Any): The end point of the geodesic.
            n_points (int, optional): The number of Gauss-Legendre quadrature points.
                If None, it is heuristically determined as:
                :math:`n = \\lceil (\\text{arc\\_length} / \\text{scale}) \\times 2 \\rceil`.
                This ensures at least two points per characteristic length-scale,
                providing stable sampling of the sensitivity kernel. Defaults to None.

        Returns:
            LinearForm: A linear functional whose action on a vector `u` computes
                 the approximated line integral.

        Raises:
            NotImplementedError: If the Sobolev order :math:`s` is less than or
                equal to half the spatial dimension :math:`n/2`.
        """
        if self.order <= self.spatial_dimension / 2:
            raise NotImplementedError(
                f"Order {self.order} is too low for point evaluation on a "
                f"{self.spatial_dimension}D manifold."
            )

        # Heuristic quadrature density determination
        if n_points is None:
            # Perform a minimal call to determine the total arc length via weights
            _, temp_weights = self.geodesic_quadrature(p1, p2, n_points=2)
            arc_length = np.sum(temp_weights)

            # Scale-based heuristic (Nyquist-like sampling)
            n_points = int(np.ceil((arc_length / self.scale) * 2.0))
            n_points = max(2, n_points)

        #  Retrieve final manifold-specific points and weights
        points, weights = self.geodesic_quadrature(p1, p2, n_points)

        #  Aggregate weighted components into the dual space representation
        # The components of a LinearForm represent the functional in the dual basis
        total_components = np.zeros(self.dim)
        for pt, weight in zip(points, weights):
            # Accumulate the weighted Riesz representation of each Dirac delta
            total_components += weight * self.dirac(pt).components

        return LinearForm(self, components=total_components)

    def geodesic_integral_representation(
        self, p1: Any, p2: Any, n_points: Optional[int] = None
    ) -> Any:
        """
        Returns the Riesz representation (sensitivity kernel) of the line integral.

        This maps the LinearForm (the integral functional) back into the
        primal Hilbert space. Visualizing this vector reveals the "sensitivity"
        of the line integral to perturbations at different locations in the domain.

        Args:
            p1, p2: Start and end points of the geodesic.
            n_points: Number of quadrature points.
        """
        # Create the functional and map it to a vector in the space
        integral_form = self.geodesic_integral(p1, p2, n_points)
        return self.from_dual(integral_form)

    def path_average_operator(self, paths, n_points=None):
        """
        Constructs a tomographic operator mapping a function field to its
        line integrals along a set of geodesic paths.

        Note: Despite the name, this operator returns the line integral
        (the dual pairing of the function with the path functional) rather
        than a normalized average, unless the user manually scales the forms.
        This corresponds to the 'path average' convention often used in
        seismic and atmospheric tomography.

        Args:
            paths (List[Tuple[Any, Any]]): A list of start and end point pairs
                defining the geodesics.
            n_points (int, optional): The number of quadrature points per path.
                If None, the heuristic based on the Sobolev scale is used.

        Returns:
            LinearOperator: An operator mapping Space -> EuclideanSpace(len(paths)).
                The adjoint of this operator performs the 'back-projection'
                mapping data residuals into the function space.
        """
        # Generate the set of linear functionals representing each path integral
        # The integral logic is handled by the Abstract Geodesic Integral method
        path_forms = [
            self.geodesic_integral(p1, p2, n_points=n_points) for p1, p2 in paths
        ]

        # Convert the list of forms into a single LinearOperator mapping
        return LinearOperator.from_linear_forms(path_forms)

    def random_source_receiver_paths(
        self, n_sources: int, n_receivers: int
    ) -> List[Tuple[Any, Any]]:
        """
        Generates a list of source-receiver pairs by connecting every source to
        every receiver.

        This method uses the existing :meth:`random_points` logic to generate
        coordinates appropriate for the specific symmetric space. For a set
        of S sources and R receivers, this returns a list of S*R paths.

        Args:
            n_sources: The number of random source locations to generate.
            n_receivers: The number of random receiver locations to generate.

        Returns:
            List[Tuple[Any, Any]]: A list of tuples, where each tuple contains
                a (source, receiver) pair.
        """
        # Generate the points using the existing base class method
        sources = self.random_points(n_sources)
        receivers = self.random_points(n_receivers)

        # Create the full-mesh network
        paths = []
        for src in sources:
            for rec in receivers:
                paths.append((src, rec))

        return paths
