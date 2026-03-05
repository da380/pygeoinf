"""
Provides an abstract framework for function spaces on symmetric manifolds.

This module offers an abstract framework for defining Hilbert spaces of
functions on symmetric spaces (like spheres or tori). The core design
leverages the spectral properties of the Laplace-Beltrami operator (Δ), which
is fundamental to the geometry of these spaces.

By inheriting from these base classes and implementing a small number of abstract
methods (like the Laplacian eigenvalues), a concrete class can automatically
gain a rich set of tools for defining invariant operators and probability
measures.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Any, List, Optional, TypeAlias, Iterator

import numpy as np
from scipy.sparse import diags

from pygeoinf.hilbert_space import HilbertSpace, HilbertModule, Vector
from pygeoinf.linear_operators import LinearOperator, DiagonalSparseMatrixLinearOperator
from pygeoinf.linear_forms import LinearForm
from pygeoinf.gaussian_measure import GaussianMeasure

# Alias for the index for the eigenvalues or eigenfunctions
Index: TypeAlias = int | tuple[int, ...]

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
        super().__init__(domain, domain, diagonals, galerkin=True)

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

    def rescale_total_variance(self, target_std: float) -> InvariantGaussianMeasure:
        """
        Returns a new measure with total variance (Trace) scaled to target_std^2.

        The expectation (mean) of the measure remains unchanged.
        """
        current_trace = self.covariance.trace

        if current_trace <= 0:
            raise ValueError("Trace must be positive to perform rescaling.")

        scale_factor_squared = (target_std**2) / current_trace
        new_variances = self.spectral_variances * scale_factor_squared

        return InvariantGaussianMeasure(
            self.domain,
            new_variances,
            expectation=self.expectation,
        )

    # Inside InvariantGaussianMeasure (symmetric_space.py)

    def rescale_directional_variance(
        self, direction: Vector, target_std: float
    ) -> InvariantGaussianMeasure:
        """
        Rescales the covariance such that Var[<x, direction>] = target_std^2.

        The expectation remains unchanged.
        """

        current_directional_variance = self.domain.inner_product(
            self.covariance(direction), direction
        )

        if current_directional_variance <= 0:
            raise ValueError(
                "Directional variance must be positive to perform rescaling."
            )

        scale_factor_squared = (target_std**2) / current_directional_variance
        new_variances = self.spectral_variances * scale_factor_squared

        return InvariantGaussianMeasure(
            self.domain, new_variances, expectation=self.expectation
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

    def __init__(self, spatial_dim: int, dim: int, orthonormal: bool):
        """
        Initializes the abstract invariant Lebesgue space.

        Args:
            spatial_dim: The dimension of the symmetric manifold
            dim: The dimension of the space.
            orthonormal: True if the eigenfunction basis is orthonormal.
        """
        self._spatial_dim = spatial_dim
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
    def laplacian_eigenvalue(self, k: int | tuple[int, ...]) -> float:
        """
        Returns the eigenvalue of the Laplacian for a given index.

        The index `k` can be a single integer (e.g., for a circle) or a
        tuple of integers (e.g., for a sphere or torus), depending on the
        geometry of the space.

        Args:
            k: The index of the eigenvalue to return.
        """

    @abstractmethod
    def laplacian_eigenvector_squared_norm(self, k: int | tuple[int, ...]) -> float:
        """
        Returns the squared norm of the eigenvalue of the Laplacian for a given index.

        The index `k` can be a single integer (e.g., for a circle) or a
        tuple of integers (e.g., for a sphere or torus), depending on the
        geometry of the space.

        Args:
            k: The index of the eigenvalue to return.
        """

    @abstractmethod
    def laplacian_eigenvectors_at_point(self, x: Point) -> List[Value]:
        """
        Returns a list of the values of the eigenvectors at a given point

        Args:
            x: The evaluation point.
        """

    @abstractmethod
    def random_point(self) -> Any:
        """Returns a single random point from the underlying symmetric space."""

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
            expectation: The expected value for measure. Defaults to None which
                means the zero vector

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
            expectation: The expected value for measure. Defaults to None which
                means the zero vector
            std: The desired standard deviation

        Returns:
            The measure as an instance of InvariantGaussianMeasure
        """

        initial_measure = self.invariant_gaussian_measure(f)
        norm = std / np.sqrt(initial_measure.covariance.trace)
        intermediate_measure = norm * initial_measure
        return intermediate_measure.affine_mapping(translation=expectation)


class SymmetricHilbertModule(SymmetricHilbertSpace, HilbertModule, ABC):
    """
    A specialisation of the SymmetricHilbertSpace class for which
    elements of the space can be multipled. In addition to the
    abstract methods required within the base class, derived
    classed must provide implementations for:

    vector_multiply
    vector_sqrt
    """
