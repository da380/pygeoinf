"""
Provides an abstract framework for function spaces on symmetric manifolds.

This module offers an abstract framework for defining Hilbert spaces of
functions on symmetric spaces (like spheres or tori). The core design
leverages the spectral properties of the Laplace-Beltrami operator (Δ).

By inheriting from these base classes and implementing a small number of abstract
methods a concrete class can automatically gain a rich set of tools for defining
invariant operators and probability measures.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Any, List, Optional, TypeAlias, Iterator, Tuple

import numpy as np
from scipy.sparse import diags
from scipy.interpolate import interp1d
import scipy.sparse as sps
import scipy.sparse.linalg as splinalg

from pygeoinf.hilbert_space import (
    HilbertSpace,
    HilbertModule,
    Vector,
    MassWeightedHilbertModule,
    EuclideanSpace,
)
from pygeoinf.linear_operators import LinearOperator, DiagonalSparseMatrixLinearOperator
from pygeoinf.linear_forms import LinearForm
from pygeoinf.gaussian_measure import GaussianMeasure

# Alias for the index for the eigenvalues or eigenfunctions
Index: TypeAlias = int | tuple[int, ...]

# Alias for the truncation degree
Degree: TypeAlias = int | tuple[int, ...]

# Alias for a point within the symmetric manifold
Point: TypeAlias = float | tuple[float, ...]

# Alias for the value type of the fields
Value: TypeAlias = float | np.ndarray


class InvariantLinearAutomorphism(DiagonalSparseMatrixLinearOperator):
    """
    A class for LinearOperators on SymmetricHilbertSpaces
    that are invariant under the symmetry group. Such operators
    can be expressed as a function of the Laplace-Beltrami operator
    and are diagonal within the eigenfunction basis.
    """

    def __init__(
        self,
        domain: SymmetricHilbertSpace,
        eigenvalues: np.ndarray,
    ):
        """
        Args:
            domain: The domain of the operator
            eigenvalues: A vector of the operator's eigenvalues
        """
        diagonals = (eigenvalues.reshape(1, -1), [0])
        super().__init__(domain, domain, diagonals)

    @staticmethod
    def from_index_function(
        domain: SymmetricHilbertSpace, g: Callable[Index, float]
    ) -> InvariantLinearAutomorphism:
        """
        Returns an invariant linear automorphism on a symmetric Hilbert space of the form
        f(Δ) with f a function that is well-defined on the spectrum of the Laplacian, Δ.

        Here the function, f, is expressed implicitly as a function, g, of the
        eigenvalue index.

        Args:
            domain: The domain of the operator
            g: The function expressed in terms of the eigenvalue index
        """
        eigenvalues = np.array([g(index) for index in domain.indices], dtype=float)
        return InvariantLinearAutomorphism(domain, eigenvalues)

    @staticmethod
    def from_function(
        domain: SymmetricHilbertSpace, f: Callable[float, float]
    ) -> InvariantLinearAutomorphism:
        """
        Returns an invariant linear automorphism on a symmetric Hilbert space of the form
        f(Δ) with f a function that is well-defined on the spectrum of the Laplacian, Δ.

        Args:
            domain: The domain of the operator
            f: The function
        """
        return InvariantLinearAutomorphism.from_index_function(
            domain, lambda k: f(domain.laplacian_eigenvalue(k))
        )

    @property
    def eigenvalues(self) -> np.ndarray:
        """Returns the operator's eigenvalues (the diagonal multipliers)."""
        return self.extract_diagonal(galerkin=False)

    @property
    def trace(self) -> float:
        """
        Returns the trace of the automorphism.
        """
        return np.sum(self.eigenvalues)

    @property
    def inverse(self) -> InvariantLinearAutomorphism:
        """
        Returns the inverse of the invariant automorphism.

        Since the operator is diagonal in the spectral basis, the inverse
        is formed by taking the reciprocal of the eigenvalues.
        """
        current_eigenvalues = self.eigenvalues

        if np.any(current_eigenvalues == 0):
            raise ValueError("Cannot invert an operator with zero eigenvalues.")

        return InvariantLinearAutomorphism(
            self.domain, np.reciprocal(current_eigenvalues)
        )

    def __neg__(self) -> InvariantLinearAutomorphism:
        current_diagonal = self.eigenvalues
        return InvariantLinearAutomorphism(self.domain, -current_diagonal)

    def __mul__(self, alpha: float) -> InvariantLinearAutomorphism:
        current_diagonal = self.eigenvalues
        return InvariantLinearAutomorphism(self.domain, alpha * current_diagonal)

    def __rmul__(self, alpha: float) -> InvariantLinearAutomorphism:
        return self * alpha

    def __add__(
        self, other: InvariantLinearAutomorphism | LinearOperator
    ) -> InvariantLinearAutomorphism | LinearOperator:
        if isinstance(other, InvariantLinearAutomorphism):
            if self.domain != other.domain:
                raise ValueError("Domains must match.")

            my_diagonal = self.eigenvalues
            other_diagonal = other.eigenvalues

            return InvariantLinearAutomorphism(
                self.domain, my_diagonal + other_diagonal
            )
        return super().__add__(other)

    def __sub__(
        self, other: InvariantLinearAutomorphism | LinearOperator
    ) -> InvariantLinearAutomorphism | LinearOperator:
        if isinstance(other, InvariantLinearAutomorphism):
            if self.domain != other.domain:
                raise ValueError("Domains must match.")

            my_diagonal = self.eigenvalues
            other_diagonal = other.eigenvalues

            return InvariantLinearAutomorphism(
                self.domain, my_diagonal - other_diagonal
            )
        return super().__sub__(other)

    def __matmul__(
        self, other: InvariantLinearAutomorphism | LinearOperator
    ) -> InvariantLinearAutomorphism | LinearOperator:
        """Composing two invariant operators via element-wise multiplication."""
        if isinstance(other, InvariantLinearAutomorphism):
            if self.codomain != other.domain:
                raise ValueError("Domain/Codomain mismatch.")

            my_diagonal = self.eigenvalues
            other_diagonal = other.eigenvalues

            return InvariantLinearAutomorphism(
                self.domain, my_diagonal * other_diagonal
            )
        return super().__matmul__(other)


class InvariantGaussianMeasure(GaussianMeasure):
    """
    A class for GaussianMeasures on SymmetricHilbertSpaces
    whose covariances are invariant under the symmetry group. The
    covariances can be expressed as a function of the Laplace-Beltrami
    operator and are diagonal within the eigenfunction basis.
    """

    def __init__(
        self,
        domain: SymmetricHilbertSpace,
        spectral_variances: np.ndarray,
        /,
        *,
        expectation: Optional[Vector] = None,
    ):
        """
        Initializes the InvariantGaussianMeasure.

        Args:
            domain: The symmetric space the measure is defined on.
            spectral_variances: A 1D array of variances associated with the
                eigenbasis (i.e., the eigenvalues of the covariance operator).
            expectation: The mean vector. Defaults to zero.
        """
        self._spectral_variances = spectral_variances
        covariance = InvariantLinearAutomorphism(domain, spectral_variances)

        squared_norms = domain.squared_norms
        self._kl_scaling_array = np.sqrt(spectral_variances / squared_norms)

        super().__init__(
            covariance=covariance,
            expectation=expectation,
            sample=self._kl_sample,
        )

    # ---------------------------------------------------------- #
    #                         Constructors                       #
    # ---------------------------------------------------------- #

    @staticmethod
    def from_index_function(
        domain: SymmetricHilbertSpace, g: Callable[Index, float], /, *, expectation=None
    ) -> InvariantGaussianMeasure:
        """
        Returns an invariant Gaussian measure on a symmetric Hilbert space whose
        covariance is of the form f(Δ) with f a function that is well-defined on
        the spectrum of the Laplacian, Δ.

        Here the function, f, is expressed implicitly as a function, g, of the
        eigenvalue index.

        Args:
            domain: The domain of the operator
            g: The function expressed in terms of the eigenvalue index
            expectation: The expected value for the measure. Defaults to None which means zero.
        """

        spectral_variances = np.fromiter(
            (g(k) for k in domain.indices),
            dtype=float,
            count=domain.dim,
        )

        return InvariantGaussianMeasure(
            domain, spectral_variances, expectation=expectation
        )

    @staticmethod
    def from_function(
        domain: SymmetricHilbertSpace, f: Callable[float, float], /, *, expectation=None
    ) -> InvariantGaussianMeasure:
        """
        Returns an invariant Gaussian measure on a symmetric Hilbert space whose
        covariance is of the form f(Δ) with f a function that is well-defined on
        the spectrum of the Laplacian, Δ.

        Args:
            domain: The domain of the operator
            f: The function expressed in terms of the eigenvalue index
            expectation: The expected value for the measure. Defaults to None which means zero.
        """

        return InvariantGaussianMeasure.from_index_function(
            domain, lambda k: f(domain.laplacian_eigenvalue(k)), expectation=expectation
        )

    # ---------------------------------------------------------- #
    #                         Properties                         #
    # ---------------------------------------------------------- #

    @property
    def spectral_variances(self) -> np.ndarray:
        """
        Provides instant access to the exact eigenvalues of the covariance
        operator, useful for log-determinant or KL-divergence calculations.
        """
        return self._spectral_variances

    # ---------------------------------------------------------- #
    #                        Public methods                      #
    # ---------------------------------------------------------- #

    def rescale_norm_variance(self, std: float) -> InvariantGaussianMeasure:
        """
        Returns a new measure whose covariance is scaled such that

        E[||x - E[x]||^2] = std^2.

        The expectation of the measure is unchanged.
        """
        current_trace = self.covariance.trace

        if current_trace <= 0:
            raise ValueError("Trace must be positive to perform rescaling.")

        scale_factor_squared = (std**2) / current_trace
        new_variances = self.spectral_variances * scale_factor_squared

        return InvariantGaussianMeasure(
            self.domain,
            new_variances,
            expectation=self.expectation,
        )

    # ------------------------------------------------------#
    #           Overloads of base class methods             #
    # ------------------------------------------------------#

    def affine_mapping(
        self, /, *, operator: LinearOperator = None, translation: Vector = None
    ) -> GaussianMeasure:
        """
        Transforms the measure under an affine map `y = A(x) + b`.

        If the operator A is an `InvariantLinearAutomorphism` (or None), the
        resulting measure remains invariant. This method intercepts that case
        to return a new, highly optimized `InvariantGaussianMeasure`.
        """

        if operator is None:
            _translation = translation if translation is not None else self.domain.zero
            new_expectation = self.domain.add(self.expectation, _translation)

            return InvariantGaussianMeasure(
                self.domain, self.spectral_variances, expectation=new_expectation
            )

        if isinstance(operator, InvariantLinearAutomorphism):
            new_covariance = self.covariance @ operator @ operator

            _translation = translation if translation is not None else self._domain.zero
            new_expectation = self._domain.add(operator(self.expectation), _translation)

            return InvariantGaussianMeasure(
                self._domain,
                new_covariance.spectral_variances,
                expectation=new_expectation,
            )

        return super().affine_mapping(operator=operator, translation=translation)

    def __neg__(self) -> InvariantGaussianMeasure:
        """Returns the measure with a negated expectation. Covariance is unchanged."""
        return InvariantGaussianMeasure(
            self.domain,
            self.spectral_variances,
            expectation=self.domain.negative(self.expectation),
        )

    def __mul__(self, alpha: float) -> InvariantGaussianMeasure:
        """Scales the measure by a scalar alpha."""
        new_variances = (alpha**2) * self.spectral_variances
        new_expectation = self.domain.multiply(alpha, self.expectation)

        return InvariantGaussianMeasure(
            self.domain,
            new_variances,
            expectation=new_expectation,
        )

    def __rmul__(self, alpha: float) -> InvariantGaussianMeasure:
        return self * alpha

    def __truediv__(self, alpha: float) -> InvariantGaussianMeasure:
        return self * (1.0 / alpha)

    def __add__(self, other: GaussianMeasure) -> GaussianMeasure:
        """Adds two independent Gaussian measures."""
        if self.domain != other.domain:
            raise ValueError("Measures must be defined on the same domain.")

        if isinstance(other, InvariantGaussianMeasure):
            new_variances = self.spectral_variances + other.spectral_variances
            new_expectation = self.domain.add(self.expectation, other.expectation)

            return InvariantGaussianMeasure(
                self.domain,
                new_variances,
                expectation=new_expectation,
            )

        return super().__add__(other)

    def __sub__(self, other: GaussianMeasure) -> GaussianMeasure:
        """Subtracts two independent Gaussian measures."""
        if self.domain != other.domain:
            raise ValueError("Measures must be defined on the same domain.")

        if isinstance(other, InvariantGaussianMeasure):
            new_variances = self.spectral_variances + other.spectral_variances
            new_expectation = self.domain.subtract(self.expectation, other.expectation)

            return InvariantGaussianMeasure(
                self.domain,
                new_variances,
                expectation=new_expectation,
            )

        return super().__sub__(other)

    # ------------------------------------------------------#
    #                    Private methods                    #
    # ------------------------------------------------------#

    def _kl_sample(self) -> Vector:
        """
        Draws a sample using the Karhunen-Loève expansion.
        """
        xi = np.random.randn(self.domain.dim)
        scaled_components = xi * self._kl_scaling_array
        sample_vector = self.domain.from_components(scaled_components)
        return self.domain.add(sample_vector, self.expectation)


class SymmetricHilbertSpace(HilbertSpace, ABC):
    """
    An abstract base class for Hilbert spaces of functions spaces on
    symmetric manifolds.

    The implementation is based on the expansion of elements of the
    space in terms of the eigenfunctions of the Laplace-Beltrami
    operator.

    To inherit from this base class, the user must provide methods
    that provide information on the eigenvalues and eigenfunctions
    of the Laplace-Beltrami operator, including mappings to and
    from the co-ordinate basis. The eigenfunction basis is
    necessarily orthogonal, but need not be normalised.
    """

    def __init__(self, spatial_dim: int, degree: Degree, dim: int, orthonormal: bool):
        """
        Initializes the abstract invariant Lebesgue space.

        Args:
            spatial_dim: The dimension of the symmetric manifold
            degree: The truncation degree.
            dim: The dimension of the space.
            orthonormal: True if the eigenfunction basis is orthonormal.
        """
        self._spatial_dim = spatial_dim
        self._degree = degree
        self._dim = dim
        self._orthonormal = orthonormal

        if self._orthonormal:
            self._metric = None
            self._inverse_metric = None
        else:
            metric_values = np.fromiter(
                (self.laplacian_eigenvector_squared_norm(k) for k in self.indices),
                dtype=float,
                count=self.dim,
            )
            self._metric = diags([metric_values], [0])
            self._inverse_metric = diags([np.reciprocal(metric_values)], [0])

    # ------------------------------------------------------------#
    #                          Properties                        #
    # ------------------------------------------------------------#

    @property
    def spatial_dimension(self) -> int:
        """The dimension of the symetric manifold."""
        return self._spatial_dim

    @property
    def degree(self) -> Degree:
        """The spectral truncation degree of the space."""
        return self._degree

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def orthonormal(self) -> bool:
        """
        True if the eigenfunction basis is normalised
        """
        return self._orthonormal

    @property
    def squared_norms(self) -> np.ndarray:
        """
        Returns a vector of the squared eigenvector norms.
        """
        if self.orthonormal:
            return np.ones(self.dim, dtype=float)
        else:
            return self._metric.diagonal(k=0)

    @property
    def indices(self) -> Iterator[Index]:
        """
        Returns a list of all the eigenvalue indices
        """
        return (self.integer_to_index(i) for i in range(self.dim))

    # ------------------------------------------------------------#
    #                      Abstract methods                       #
    # ------------------------------------------------------------#

    @abstractmethod
    def index_to_integer(self, k: Index) -> int:
        """
        Maps an eigenvalue index to an integer.
        """

    @abstractmethod
    def integer_to_index(self, i: int) -> Index:
        """
        Maps an integer to the eigenvalue index
        """

    @abstractmethod
    def laplacian_eigenvalue(self, k: Index) -> float:
        """
        Returns the eigenvalue of the Laplacian for a given index.

        The index `k` can be a single integer (e.g., for a circle) or a
        tuple of integers (e.g., for a sphere or torus), depending on the
        geometry of the space.

        Args:
            k: The index of the eigenvalue to return.
        """

    @abstractmethod
    def laplacian_eigenvector_squared_norm(self, k: Index) -> float:
        """
        Returns the squared norm of the eigenvalue of the Laplacian for a given index.

        The index `k` can be a single integer (e.g., for a circle) or a
        tuple of integers (e.g., for a sphere or torus), depending on the
        geometry of the space.

        Args:
            k: The index of the eigenvalue to return.
        """

    @abstractmethod
    def laplacian_eigenvectors_at_point(self, x: Point) -> np.ndarray:
        """
        Returns a list of the values of the eigenvectors at a given point

        Args:
            x: The evaluation point.
        """

    @abstractmethod
    def random_point(self) -> Any:
        """Returns a single random point from the underlying symmetric space."""

    @abstractmethod
    def geodesic_distance(self, p1: Any, p2: Any) -> float:
        """
        Returns the shortest distance along the manifold between two points.

        Args:
            p1: The starting point.
            p2: The end point.
        """

    @abstractmethod
    def point_at_distance(self, p1: Point, distance: float) -> Point:
        """
        Returns a point located at exactly the specified geodesic distance from p1.
        Since invariant measures are isotropic, the direction of translation
        is arbitrary.
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

    @abstractmethod
    def with_degree(self, degree: Degree) -> SymmetricHilbertSpace:
        """Returns a new instance of the space with a modified truncation degree."""

    @abstractmethod
    def degree_transfer_operator(self, target_degree: Degree) -> LinearOperator:
        """Returns the transfer operator from this space to one with a different degree."""

    # ------------------------------------------------------------#
    #                        Public methods                       #
    # ------------------------------------------------------------#

    def to_dual(self, x: Vector) -> LinearForm:
        cx = self.to_components(x)
        if self.orthonormal:
            return LinearForm(self, components=cx)
        else:
            cxp = self._metric @ cx
            return LinearForm(self, components=cxp)

    def from_dual(self, xp: LinearForm) -> Vector:
        cxp = xp.components
        if self.orthonormal:
            return self.from_components(cxp)
        else:
            cx = self._inverse_metric @ cxp
            return self.from_components(cx)

    def random_points(self, n: int) -> List[Point]:
        """
        Returns a list of `n` random points.

        Args:
            n: The number of random points to generate.
        """
        return [self.random_point() for _ in range(n)]

    def invariant_automorphism(
        self, f: Callable[float, float]
    ) -> InvariantLinearAutomorphism:
        """
        Returns an automorphism of the form f(Δ) with f a function
        that is well-defined on the spectrum of the Laplacian, Δ.

        Args:
            f: The function
        """
        return InvariantLinearAutomorphism.from_function(self, f)

    def invariant_gaussian_measure(
        self, f: Callable[float, float], /, *, expectation=None
    ) -> InvariantGaussianMeasure:
        """
        Returns a Gaussian measure on the space whose covariance takes the form f(Δ) with f
        a function that is well-defined on the spectrum of the Laplacian, Δ.

        In order for the covariance to be well-defined, the trace of the covariance operator
        must be finite. This condition (in the sense of convergence as the size of the
        approximating space being increased) is not checked.

        Args:
            f: The function
            expectation: The expected value for measure. Defaults to zero

        Returns:
            The measure as an instance of InvariantGaussianMeasure
        """

        return InvariantGaussianMeasure.from_function(self, f, expectation=expectation)

    def norm_scaled_invariant_gaussian_measure(
        self, f: Callable[float, float], /, *, expectation=None, std=1.0
    ) -> InvariantGaussianMeasure:
        """
        Returns a Gaussian measure on the space whose covariance takes the form f(Δ) with f
        a function that is well-defined on the spectrum of the Laplacian, Δ.

        In order for the covariance to be well-defined, the trace of the covariance operator
        must be finite. This condition (in the sense of convergence as the size of the
        approximating space being increased) is not checked.

        The measure's covariance is scaled such that:

        E[||x||^2] = std*std + ||E[x]||^2

        Args:
            f: The function
            expectation: The expected value for measure. Defaults to zero
            std: The desired standard deviation

        Returns:
            The measure as an instance of InvariantGaussianMeasure
        """
        initial_measure = self.invariant_gaussian_measure(f, expectation=expectation)
        return initial_measure.rescale_norm_variance(std)

    def sobolev_kernel_gaussian_measure(
        self, order: float, scale: float, /, *, expectation: Vector = None
    ):
        """
        Returns an invariant Gaussian measure with a Sobolev-type covariance
        equal to (1 + scale^2 * Δ)^-order.

        Args:
            order: Order parameter for the covariance.
            scale: Scale parameter for the covariance.
            expectation: The expected value for measure. Defaults to zero
        """
        return self.invariant_gaussian_measure(
            lambda k: (1 + scale**2 * k) ** (-order), expectation=expectation
        )

    def norm_scaled_sobolev_kernel_gaussian_measure(
        self,
        order: float,
        scale: float,
        /,
        *,
        expectation: Vector = None,
        std: float = 1,
    ):
        """
        Returns an invariant Gaussian measure with a Sobolev-type covariance
        proportional to (1 + scale^2 * Δ)^-order.

        The measure's covariance is scaled such that the expected value for the
        samples norm is equal to the given standard deviation.

        Args:
            order: Order parameter for the covariance.
            scale: Scale parameter for the covariance.
            expectation: The expected value for measure. Defaults to zero
            std: The desired standard deviation
        """
        return self.norm_scaled_invariant_gaussian_measure(
            lambda k: (1 + scale**2 * k) ** -order, expectation=expectation, std=std
        )

    def heat_kernel_gaussian_measure(
        self, scale: float, /, *, expectation: Vector = None
    ):
        """
        Returns an invariant Gaussian measure with a heat kernel covariance
        equal to exp(-scale^2 * Δ).

        Args:
            scale: Scale parameter for the covariance.
            expectation: The expected value for measure. Defaults to zero
        """
        return self.invariant_gaussian_measure(
            lambda k: np.exp(-(scale**2) * k), expectation=expectation
        )

    def norm_scaled_heat_kernel_gaussian_measure(
        self, scale: float, /, *, expectation: Vector = None, std: float = 1
    ):
        """
        Returns an invariant Gaussian measure with a heat kernel covariance
        equal to exp(-scale^2 * Δ).

        Args:
            scale: Scale parameter for the covariance.
            expectation: The expected value for measure. Defaults to zero
            std: The desired standard deviation
        """
        return self.norm_scaled_invariant_gaussian_measure(
            lambda k: np.exp(-(scale**2) * k), expectation=expectation, std=std
        )

    def cluster_points(
        self,
        points: List[Point],
        /,
        *,
        threshold: Optional[float] = None,
        n_clusters: Optional[int] = None,
        linkage_method: str = "complete",
    ) -> List[List[int]]:
        """
        Clusters a list of points into interacting blocks based on their
        geodesic distance.

        This is particularly useful for generating the 'interacting_blocks'
        required by localized preconditioners in Bayesian inversions.

        Args:
            points: A list of points on the symmetric manifold.
            threshold: The maximum geodesic distance between points in a cluster.
            n_clusters: The exact number of clusters to form (alternative to threshold).
            linkage_method: The hierarchical clustering method to use ('complete',
                            'average', 'single', etc.). Defaults to 'complete' to
                            ensure clusters remain geographically compact.

        Returns:
            A list of lists, where each sub-list contains the indices of the
            points belonging to a specific cluster.
        """
        from scipy.cluster.hierarchy import linkage, fcluster

        n = len(points)
        if n == 0:
            return []
        if n == 1:
            return [[0]]

        if threshold is None and n_clusters is None:
            raise ValueError("You must specify either 'threshold' or 'n_clusters'.")

        # 1. Compute pairwise geodesic distances (in SciPy's condensed 1D format)
        # A condensed matrix is required by the linkage function
        distances = np.zeros(n * (n - 1) // 2)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                distances[idx] = self.geodesic_distance(points[i], points[j])
                idx += 1

        # 2. Perform agglomerative hierarchical clustering
        Z = linkage(distances, method=linkage_method)

        # 3. Extract the flat clusters based on the user's criteria
        if n_clusters is not None:
            labels = fcluster(Z, t=n_clusters, criterion="maxclust")
        else:
            labels = fcluster(Z, t=threshold, criterion="distance")

        # 4. Group the point indices by their assigned cluster label
        blocks = {}
        for i, label in enumerate(labels):
            blocks.setdefault(label, []).append(i)

        return list(blocks.values())

    def pairs_within_distance(
        self, points: List[Point], max_distance: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Finds all pairs of points within a given geodesic distance.
        Returns arrays of row indices, column indices, and their geodesic distances.
        """
        n = len(points)
        rows, cols, dists = [], [], []
        for i in range(n):
            for j in range(n):  # Full symmetric sweep
                d = self.geodesic_distance(points[i], points[j])
                if d <= max_distance:
                    rows.append(i)
                    cols.append(j)
                    dists.append(d)
        return np.array(rows), np.array(cols), np.array(dists)


class AbstractSymmetricLebesgueSpace(HilbertModule, SymmetricHilbertSpace, ABC):
    """
    A specialisation for scalar-valued L² function spaces on symmetric manifolds.

    To be instantiated, such a class must provide the following additional methods:

    vector_multiply
    vector_sqrt
    """


class SymmetricSobolevSpace(MassWeightedHilbertModule, SymmetricHilbertSpace):
    """
    The Sobolev space Hˢ constructed over a symmetric Lebesgue space.

    This implementation leverages the mass-weighting framework to ensure that
    the inner product and dual mappings correctly account for the smoothness
    order and scale.
    """

    def __init__(
        self,
        lebesgue_space: AbstractSymmetricLebesgueSpace,
        order: float,
        scale: float,
    ) -> None:
        """
        Args:
            lebesgue_space: The underlying L² space (which provides Δ eigenvalues).
            order: The Sobolev smoothness order (s).
            scale: The Sobolev length-scale (κ).
        """
        self._order = order
        self._scale = scale

        mass_operator = lebesgue_space.invariant_automorphism(self.sobolev_function)
        inverse_mass_operator = mass_operator.inverse

        MassWeightedHilbertModule.__init__(
            self, lebesgue_space, mass_operator, inverse_mass_operator
        )

        SymmetricHilbertSpace.__init__(
            self,
            lebesgue_space.spatial_dimension,
            lebesgue_space.degree,
            lebesgue_space.dim,
            False,
        )

    @property
    def order(self) -> float:
        """The Sobolev order."""
        return self._order

    @property
    def scale(self) -> float:
        """The Sobolev length-scale."""
        return self._scale

    @abstractmethod
    def with_order(self, order: float) -> SymmetricSobolevSpace:
        """
        Returns a new instance of the exact same space but with a modified
        Sobolev order.
        """

    def order_inclusion_operator(self, target_order: float) -> LinearOperator:
        """Returns the inclusion operator from this space to one of a lower order."""
        if target_order > self.order:
            raise ValueError(
                "Target order must be less than or equal to the current order."
            )
        codomain = self.with_order(target_order)
        underlying_identity = self.underlying_space.identity_operator()
        return LinearOperator.from_formal_adjoint(self, codomain, underlying_identity)

    def sobolev_function(self, lambda_val) -> float:
        """
        Returns the value of the Sobolev function associated with the space.
        """
        return (1.0 + (self.scale**2) * lambda_val) ** self.order

    def dirac(self, point: Point) -> LinearForm:
        """
        Returns the Dirac measure at the chosen point.
        """
        if self.order <= self.spatial_dimension / 2:
            raise NotImplementedError("The Dirac measure is not defined on this space.")
        cxp = self.laplacian_eigenvectors_at_point(point)
        return LinearForm(self, components=cxp)

    def dirac_representation(self, point) -> Vector:
        """
        Returns the representation of the Dirac measure at the chosen point.
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
            raise NotImplementedError("Point evaluation is not defined on this space")

        dim = len(points)
        matrix = np.zeros((dim, self.dim))

        for i, point in enumerate(points):
            cp = self.dirac(point).components
            matrix[i, :] = cp

        return LinearOperator.from_matrix(
            self, EuclideanSpace(dim), matrix, galerkin=True
        )

    def point_value_scaled_invariant_gaussian_measure(
        self,
        f: Callable[[float], float],
        /,
        *,
        expectation: Vector = None,
        std: float = 1,
    ):
        """
        Returns an invariant Gaussian measure with covariance proportional to f(Δ),
        where f must be such that this operator is trace-class.

        The covariance of the operator is scaled such that the standard deviation
        of the point-wise values are equal to the given std.

        Args:
            f: A real-valued function that is well-defined on the spectrum
               of the Laplacian, Δ.
            std: The desired standard deviation for the pointwise values.

        Raises:
            NotImplementedError: If the Sobolev order is less than n/2, with n the spatial dimension.

        Notes:
            This method applies for symmetric spaces an invariant measures. As a result, the
            pointwise variance is the same at all points. Internally, a random point is chosen
            to carry out the normalisation.
        """

        unscaled_measure = InvariantGaussianMeasure.from_function(
            self, f, expectation=expectation
        )

        return unscaled_measure.rescale_directional_variance(
            self.dirac_representation(self.random_point()), std
        )

    def point_value_scaled_sobolev_kernel_gaussian_measure(
        self,
        order: float,
        scale: float,
        /,
        *,
        expectation: Vector = None,
        std: float = 1,
    ):
        """
        Returns an invariant Gaussian measure with a Sobolev-type covariance
        proportional to (1 + scale^2 * Δ)^-order.

        The covariance of the operator is scaled such that the standard deviation
        of the point-wise values are equal to the given amplitude.

        Args:
            order: Order parameter for the covariance.
            scale: Scale parameter for the covariance.
            std: The desired standard deviation for the pointwise values.
        """
        return self.point_value_scaled_invariant_gaussian_measure(
            lambda k: (1 + scale**2 * k) ** -order, expectation=expectation, std=std
        )

    def point_value_scaled_heat_kernel_gaussian_measure(
        self, scale: float, /, *, expectation: Vector = None, std: float = 1
    ):
        """
        Returns an invariant Gaussian measure with a heat-kernel covariance
        proportional to exp(-scale^2 * Δ).

        The covariance of the operator is scaled such that the standard deviation
        of the point-wise values are equal to the given amplitude.

        Args:
            scale: Scale parameter for the covariance.
            std: The desired standard deviation for the pointwise values.
        """
        return self.point_value_scaled_invariant_gaussian_measure(
            lambda k: np.exp(-(scale**2) * k), expectation=expectation, std=std
        )

    def geodesic_integral(
        self, p1: Point, p2: Point, /, *, n_points: Optional[int] = None
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
            p1: The starting point of the geodesic. The type is manifold-dependent
                (e.g., float for :class:`Circle`, tuple for :class:`Sphere`).
            p2: The end point of the geodesic.
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

        if n_points is None:
            _, temp_weights = self.geodesic_quadrature(p1, p2, n_points=2)
            arc_length = np.sum(temp_weights)
            n_points = int(np.ceil((arc_length / self.scale) * 2.0))
            n_points = max(2, n_points)

        points, weights = self.geodesic_quadrature(p1, p2, n_points)

        total_components = np.zeros(self.dim)
        for pt, weight in zip(points, weights):
            total_components += weight * self.dirac(pt).components

        return LinearForm(self, components=total_components)

    def geodesic_integral_representation(
        self, p1: Point, p2: Point, /, *, n_points: Optional[int] = None
    ) -> Vector:
        """
        Returns the Riesz representation (sensitivity kernel) of the line integral.

        This maps the LinearForm (the integral functional) back into the
        primal Hilbert space. Visualizing this vector reveals the "sensitivity"
        of the line integral to perturbations at different locations in the domain.

        Args:
            p1, p2: Start and end points of the geodesic.
            n_points: Number of quadrature points.
        """
        integral_form = self.geodesic_integral(p1, p2, n_points=n_points)
        return self.from_dual(integral_form)

    def path_average_operator(self, paths, /, *, n_points=None):
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
        path_forms = [
            self.geodesic_integral(p1, p2, n_points=n_points) for p1, p2 in paths
        ]
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

    def cluster_points(
        self,
        points: List[Point],
        /,
        *,
        threshold: Optional[float] = None,
        n_clusters: Optional[int] = None,
        linkage_method: str = "complete",
    ) -> List[List[int]]:
        """
        Clusters a list of points into interacting blocks based on their
        geodesic distance.

        This is particularly useful for generating the 'interacting_blocks'
        required by localized preconditioners in Bayesian inversions.

        Args:
            points: A list of points on the symmetric manifold.
            threshold: The maximum geodesic distance between points in a cluster.
            n_clusters: The exact number of clusters to form (alternative to threshold).
            linkage_method: The hierarchical clustering method to use ('complete',
                            'average', 'single', etc.). Defaults to 'complete' to
                            ensure clusters remain geographically compact.

        Returns:
            A list of lists, where each sub-list contains the indices of the
            points belonging to a specific cluster.
        """
        return self.underlying_space.cluster_points(
            points,
            threshold=threshold,
            n_clusters=n_clusters,
            linkage_method=linkage_method,
        )

    def distance_localized_preconditioner(
        self,
        prior_measure: InvariantGaussianMeasure,
        points: List[Point],
        data_error_measure: GaussianMeasure,
        max_distance: float,
        /,
        *,
        apply_taper: bool = False,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> LinearOperator:
        """
        Builds a highly specialized, ultra-fast sparse preconditioner for point
        evaluation problems when the prior measure is invariant.

        This exploits the property that the two-point covariance depends solely
        on the geodesic distance. It maps out the 1D covariance function along
        a deterministic path, interpolates it, optionally applies a Gaspari-Cohn taper to
        ensure positive-definiteness, and populates a sparse normal matrix.

        If max_distance is set to 0.0, it bypasses the distance calculations
        and instantly builds a purely diagonal (Jacobi) preconditioner.

        Args:
            prior_measure: The invariant prior Gaussian measure.
            points: The list of observation points.
            data_error_measure: The Gaussian measure describing the data noise.
            max_distance: The geodesic distance beyond which covariance is assumed
                          to be zero (enforces sparsity). Set to 0.0 for a pure diagonal.
            apply_taper: If true, applies Gaspari-Cohn taper.
            parallel: If True, computes the error measure diagonal in parallel.
            n_jobs: Number of CPU cores to use if parallel=True.

        Returns:
            A LinearOperator representing the inverse of the approximated sparse normal matrix.
        """

        n_data = len(points)
        data_space = EuclideanSpace(n_data)

        # =======================================================
        # SPECIAL CASE: Purely Diagonal (Jacobi) Preconditioner
        # =======================================================
        if max_distance <= 0.0:

            p1 = self.random_point()
            rep1 = self.dirac_representation(p1)
            q_rep1 = prior_measure.covariance(rep1)
            prior_var = float(self.inner_product(rep1, q_rep1))

            noise_variance = data_error_measure.covariance.extract_diagonal(
                galerkin=True, parallel=parallel, n_jobs=n_jobs
            )

            inv_diag = 1.0 / (prior_var + noise_variance)

            def apply_diag_preconditioner(x):
                c_vec = data_space.to_components(x)
                return data_space.from_components(c_vec * inv_diag)

            return LinearOperator(
                data_space,
                data_space,
                apply_diag_preconditioner,
                adjoint_mapping=apply_diag_preconditioner,
            )

        # =======================================================
        # STANDARD CASE: Distance-Localized Sparse Preconditioner
        # =======================================================

        n_interp_points = max(20, int(4.0 * max_distance / self.scale))
        exact_distances = np.linspace(0.0, max_distance * 1.1, n_interp_points)

        p1 = self.random_point()
        rep1 = self.dirac_representation(p1)
        q_rep1 = prior_measure.covariance(rep1)

        cov_vals = []
        for d in exact_distances:
            p2 = self.point_at_distance(p1, float(d))
            rep2 = self.dirac_representation(p2)
            cov = self.inner_product(rep2, q_rep1)
            cov_vals.append(cov)

        cov_interpolator = interp1d(
            exact_distances, cov_vals, kind="linear", bounds_error=False, fill_value=0.0
        )

        def gaspari_cohn_vectorized(d_array, c_val):
            """Vectorized Gaspari-Cohn taper for lightning-fast array evaluations."""

            z = d_array / c_val
            taper = np.zeros_like(z)

            mask1 = z <= 1.0
            z1 = z[mask1]
            taper[mask1] = (
                1.0
                - (5.0 / 3.0) * z1**2
                + (5.0 / 8.0) * z1**3
                + (1.0 / 2.0) * z1**4
                - (1.0 / 4.0) * z1**5
            )

            mask2 = (z > 1.0) & (z <= 2.0)
            z2 = z[mask2]
            taper[mask2] = (
                4.0
                - 5.0 * z2
                + (5.0 / 3.0) * z2**2
                + (5.0 / 8.0) * z2**3
                - (1.0 / 2.0) * z2**4
                + (1.0 / 12.0) * z2**5
                - (2.0 / 3.0) / z2
            )

            return taper

        row_indices, col_indices, dists = self.pairs_within_distance(
            points, max_distance
        )

        c_taper = max_distance / 2.0
        raw_vals = cov_interpolator(dists)
        if apply_taper:
            tapers = gaspari_cohn_vectorized(dists, c_taper)
            values = raw_vals * tapers
        else:
            values = raw_vals

        H_sparse = sps.coo_matrix(
            (values, (row_indices, col_indices)), shape=(n_data, n_data)
        )

        noise_variance = data_error_measure.covariance.extract_diagonal(
            galerkin=True, parallel=parallel, n_jobs=n_jobs
        )
        R_sparse = sps.diags(noise_variance)

        H_approx = (H_sparse + R_sparse).tocsc()

        splu_solver = splinalg.splu(H_approx)

        def apply_sparse_preconditioner(x):
            c_vec = data_space.to_components(x)
            c_solved = splu_solver.solve(c_vec)
            return data_space.from_components(c_solved)

        return LinearOperator(
            data_space,
            data_space,
            apply_sparse_preconditioner,
            adjoint_mapping=apply_sparse_preconditioner,
        )

    def degree_transfer_operator(self, target_degree: Degree) -> LinearOperator:
        codomain = self.with_degree(target_degree)
        lebesgue_inclusion = self.underlying_space.degree_transfer_operator(
            target_degree
        )
        return LinearOperator.from_formal_adjoint(self, codomain, lebesgue_inclusion)

    # ------------------------------------------------------- #
    #          Methods defered to the Lebesgue space          #
    # ------------------------------------------------------- #

    def index_to_integer(self, k: Index) -> int:
        return self.underlying_space.index_to_integer(k)

    def integer_to_index(self, i: int) -> Index:
        return self.underlying_space.integer_to_index(i)

    def laplacian_eigenvalue(self, k: Index) -> float:
        return self.underlying_space.laplacian_eigenvalue(k)

    def laplacian_eigenvector_squared_norm(self, k: Index) -> float:
        l2_norm_sq = self.underlying_space.laplacian_eigenvector_squared_norm(k)
        lambda_k = self.laplacian_eigenvalue(k)
        weight = self.sobolev_function(lambda_k)
        return l2_norm_sq * weight

    def laplacian_eigenvectors_at_point(self, x: Point) -> List[Value]:
        return self.underlying_space.laplacian_eigenvectors_at_point(x)

    def random_point(self) -> Point:
        return self.underlying_space.random_point()

    def geodesic_distance(self, p1: Point, p2: Point) -> float:
        return self.underlying_space.geodesic_distance(p1, p2)

    def point_at_distance(self, p1: Point, distance: float) -> Point:
        return self.underlying_space.point_at_distance(p1, distance)

    def geodesic_quadrature(
        self, p1: Any, p2: Any, n_points: int
    ) -> Tuple[List[Any], np.ndarray]:
        return self.underlying_space.geodesic_quadrature(p1, p2, n_points)

    def pairs_within_distance(
        self, points: List[Point], max_distance: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.underlying_space.pairs_within_distance(points, max_distance)

    def __eq__(self, other: object) -> bool:
        """
        Checks for mathematical equality with another Sobolev spaces.
        """
        if not isinstance(other, SymmetricSobolevSpace):
            return NotImplemented

        check1 = self.underlying_space == other.underlying_space
        check2 = self.order == other.order
        check3 = self.scale == other.scale

        return check1 and check2 and check3
