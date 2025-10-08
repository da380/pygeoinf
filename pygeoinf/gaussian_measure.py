"""
Provides a class for representing Gaussian measures on Hilbert spaces.

This module generalizes the concept of a multivariate normal distribution to
the setting of abstract Hilbert spaces. A `GaussianMeasure` is defined by its
expectation (a vector in the space) and its covariance (a self-adjoint,
positive semi-definite `LinearOperator`).

This abstraction is fundamental for Bayesian inference, Gaussian processes, and
data assimilation in function spaces.

Key Features
------------
- Multiple factory methods for creating measures from various inputs (matrices,
  samples, standard deviations).
- A method for drawing random samples from the measure.
- Implementation of the affine transformation rule (`y = A(x) + b`).
- Support for creating low-rank approximations of the measure for efficiency.
- Overloaded arithmetic operators for intuitive combination of measures.
"""

from __future__ import annotations
from typing import Callable, Optional, Any, List, TYPE_CHECKING
import warnings

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import diags
from scipy.stats import multivariate_normal


from .hilbert_space import EuclideanSpace, HilbertModule, Vector

from .linear_operators import (
    LinearOperator,
    DiagonalSparseMatrixLinearOperator,
)

from .direct_sum import (
    BlockDiagonalLinearOperator,
)


# This block is only processed by type checkers, not at runtime.
if TYPE_CHECKING:
    from .hilbert_space import HilbertSpace
    from .typing import Vector


class GaussianMeasure:
    """
    Represents a Gaussian measure on a Hilbert space.

    This class generalizes the multivariate normal distribution to abstract,
    potentially infinite-dimensional, Hilbert spaces. A measure is
    defined by its expectation (mean vector) and its covariance, which is a
    `LinearOperator` on the space.

    It provides a powerful toolkit for probabilistic modeling, especially in
    the context of Bayesian inversion.
    """

    def __init__(
        self,
        /,
        *,
        covariance: LinearOperator = None,
        covariance_factor: LinearOperator = None,
        expectation: Vector = None,
        sample: Callable[[], Vector] = None,
        inverse_covariance: LinearOperator = None,
        inverse_covariance_factor: LinearOperator = None,
    ) -> None:
        """
         Initializes the GaussianMeasure.

        The measure can be defined in several ways, primarily by providing
         either a covariance operator or a covariance factor.

         Args:
             covariance (LinearOperator, optional): A self-adjoint and positive
                 semi-definite linear operator on the domain.
             covariance_factor (LinearOperator, optional): A linear operator L
                 such that the covariance C = L @ L*.
             expectation (vector, optional): The expectation (mean) of the
                 measure. Defaults to the zero vector of the space.
             sample (callable, optional): A function that returns a random
                 sample from the measure. If a `covariance_factor` is given,
                 a default sampler is created.
             inverse_covariance (LinearOperator, optional): The inverse of the
                 covariance operator (the precision operator).
             inverse_covariance_factor (LinearOperator, optional): A factor Li
                 of the inverse covariance, such that C_inv = Li.T @ Li.

         Raises:
             ValueError: If neither `covariance` nor `covariance_factor`
                 is provided.
        """
        if covariance is None and covariance_factor is None:
            raise ValueError(
                "Neither covariance or covariance factor has been provided"
            )

        self._covariance_factor: Optional[LinearOperator] = covariance_factor
        self._covariance: LinearOperator = (
            covariance_factor @ covariance_factor.adjoint
            if covariance is None
            else covariance
        )
        self._domain: HilbertSpace = self._covariance.domain
        self._sample: Optional[Callable[[], Vector]] = (
            sample if covariance_factor is None else self._sample_from_factor
        )
        self._inverse_covariance_factor: Optional[LinearOperator] = (
            inverse_covariance_factor
        )

        if inverse_covariance_factor is not None:
            self._inverse_covariance: Optional[LinearOperator] = (
                inverse_covariance_factor.adjoint @ inverse_covariance_factor
            )
        elif inverse_covariance is not None:
            self._inverse_covariance = inverse_covariance
        else:
            self._inverse_covariance = None

        if expectation is None:
            self._expectation: Vector = self.domain.zero
        else:
            self._expectation = expectation

    @staticmethod
    def from_standard_deviation(
        domain: HilbertSpace,
        standard_deviation: float,
        /,
        *,
        expectation: Vector = None,
    ) -> GaussianMeasure:
        """
        Creates an isotropic Gaussian measure with scaled identity covariance.

        Args:
            domain (HilbertSpace): The Hilbert space for the measure.
            standard_deviation (float): The standard deviation. The covariance
                will be `sigma^2 * I`.
            expectation (vector, optional): The expectation of the measure.
                Defaults to zero.
        """
        covariance_factor = standard_deviation * domain.identity_operator()
        inverse_covariance_factor = (
            1 / standard_deviation
        ) * domain.identity_operator()
        return GaussianMeasure(
            covariance_factor=covariance_factor,
            inverse_covariance_factor=inverse_covariance_factor,
            expectation=expectation,
        )

    @staticmethod
    def from_standard_deviations(
        domain: HilbertSpace,
        standard_deviations: np.ndarray,
        /,
        *,
        expectation: Vector = None,
    ) -> GaussianMeasure:
        """
        Creates a Gaussian measure with a diagonal covariance operator.

        Args:
            domain (HilbertSpace): The Hilbert space for the measure.
            standard_deviations (np.ndarray): A vector of standard deviations
                for each basis direction. The resulting covariance will be
                diagonal in the basis of the space.
            expectation (vector, optional): The expectation of the measure.
                Defaults to zero.
        """

        if standard_deviations.size != domain.dim:
            raise ValueError(
                "Standard deviation vector does not have the correct length"
            )
        euclidean = EuclideanSpace(domain.dim)
        covariance_factor = DiagonalSparseMatrixLinearOperator.from_diagonal_values(
            euclidean, domain, standard_deviations
        )
        return GaussianMeasure(
            covariance_factor=covariance_factor,
            inverse_covariance_factor=covariance_factor.inverse,
            expectation=expectation,
        )

    @staticmethod
    def from_covariance_matrix(
        domain: HilbertSpace,
        covariance_matrix: np.ndarray,
        /,
        *,
        expectation: Vector = None,
        rtol: float = 1e-10,
    ) -> GaussianMeasure:
        """
        Creates a Gaussian measure from a dense covariance matrix.

        The provided matrix is interpreted as the Galerkin representation of
        the covariance operator. This method computes a Cholesky-like
        decomposition of the matrix to create a `covariance_factor`.

        It includes a check to handle numerical precision issues, allowing for
        eigenvalues that are slightly negative within a relative tolerance.

        Args:
            domain: The Hilbert space the measure is defined on.
            covariance_matrix: The dense covariance matrix.
            expectation: The expectation (mean) of the measure.
            rtol: The relative tolerance used to check for negative eigenvalues.
        """

        eigenvalues, U = eigh(covariance_matrix)

        if np.any(eigenvalues < 0):
            max_eig = np.max(np.abs(eigenvalues))
            min_eig = np.min(eigenvalues)

            # Check if the most negative eigenvalue is outside the tolerance
            if min_eig < -rtol * max_eig:
                raise ValueError(
                    "Covariance matrix has significantly negative eigenvalues, "
                    "indicating it is not positive semi-definite."
                )
            else:
                # If negative eigenvalues are within tolerance, warn and correct
                warnings.warn(
                    "Covariance matrix has small negative eigenvalues due to "
                    "numerical error. Clipping them to zero.",
                    UserWarning,
                )
                eigenvalues[eigenvalues < 0] = 0

        values = np.sqrt(eigenvalues)
        D = diags([values], [0])
        # Use pseudo-inverse for singular matrices
        Di = diags([np.reciprocal(values, where=(values != 0))], [0])
        L = U @ D
        Li = Di @ U.T

        covariance_factor = LinearOperator.from_matrix(
            EuclideanSpace(domain.dim), domain, L, galerkin=True
        )
        inverse_covariance_factor = LinearOperator.from_matrix(
            domain, EuclideanSpace(domain.dim), Li, galerkin=False
        )

        return GaussianMeasure(
            covariance_factor=covariance_factor,
            inverse_covariance_factor=inverse_covariance_factor,
            expectation=expectation,
        )

    @staticmethod
    def from_samples(domain: HilbertSpace, samples: List[Vector]) -> GaussianMeasure:
        """
        Estimates a Gaussian measure from a collection of sample vectors.

        The expectation and covariance are estimated using the sample mean
        and sample covariance.

        Args:
            domain (HilbertSpace): The space the measure is defined on.
            samples (list): A list of sample vectors from the domain.
        """

        assert all([domain.is_element(x) for x in samples])
        n = len(samples)
        if n == 0:
            raise ValueError("Cannot estimate measure from zero samples.")

        expectation = domain.sample_expectation(samples)

        if n == 1:
            covariance = domain.zero_operator()

            def sample() -> Vector:
                return expectation

        else:
            offsets = [domain.subtract(x, expectation) for x in samples]
            covariance = LinearOperator.self_adjoint_from_tensor_product(
                domain, offsets
            ) / (n - 1)

            def sample() -> Vector:
                x = domain.copy(expectation)
                randoms = np.random.randn(len(offsets))
                for y, r in zip(offsets, randoms):
                    domain.axpy(r / np.sqrt(n - 1), y, x)
                return x

        return GaussianMeasure(
            covariance=covariance, expectation=expectation, sample=sample
        )

    @staticmethod
    def from_direct_sum(measures: List[GaussianMeasure]) -> GaussianMeasure:
        """
        Constructs a product measure from a list of other measures.

        The resulting measure is defined on the direct sum of the individual
        Hilbert spaces. Its covariance is a block-diagonal operator.

        Args:
            measures (list): A list of `GaussianMeasure` objects.
        """

        expectation = [measure.expectation for measure in measures]
        covariance = BlockDiagonalLinearOperator(
            [measure.covariance for measure in measures]
        )

        inverse_covariance = (
            BlockDiagonalLinearOperator(
                [measure.inverse_covariance for measure in measures]
            )
            if all(measure.inverse_covariance_set for measure in measures)
            else None
        )

        def sample_impl() -> List[Vector]:
            return [measure.sample() for measure in measures]

        sample = (
            sample_impl if all(measure.sample_set for measure in measures) else None
        )

        return GaussianMeasure(
            covariance=covariance,
            expectation=expectation,
            sample=sample,
            inverse_covariance=inverse_covariance,
        )

    @property
    def domain(self) -> HilbertSpace:
        """The Hilbert space the measure is defined on."""
        return self._domain

    @property
    def covariance(self) -> LinearOperator:
        """The covariance operator of the measure."""
        return self._covariance

    @property
    def inverse_covariance_set(self) -> bool:
        """True if the inverse covariance (precision) is available."""
        return self._inverse_covariance is not None

    @property
    def inverse_covariance(self) -> LinearOperator:
        """The inverse covariance (precision) operator."""
        if self._inverse_covariance is None:
            raise AttributeError("Inverse covariance is not set for this measure.")
        return self._inverse_covariance

    @property
    def covariance_factor_set(self) -> bool:
        """True if a covariance factor L (s.t. C=LL*) is available."""
        return self._covariance_factor is not None

    @property
    def covariance_factor(self) -> LinearOperator:
        """The covariance factor L (s.t. C=LL*)."""
        if self._covariance_factor is None:
            raise AttributeError("Covariance factor has not been set.")
        return self._covariance_factor

    @property
    def inverse_covariance_factor_set(self) -> bool:
        """True if an inverse covariance factor is available."""
        return self._inverse_covariance_factor is not None

    @property
    def inverse_covariance_factor(self) -> LinearOperator:
        """The inverse covariance factor."""
        if self._inverse_covariance_factor is None:
            raise AttributeError("Inverse covariance factor has not been set.")
        return self._inverse_covariance_factor

    @property
    def expectation(self) -> Vector:
        """The expectation (mean) of the measure."""
        return self._expectation

    @property
    def sample_set(self) -> bool:
        """True if a method for drawing samples is available."""
        return self._sample is not None

    def sample(self) -> Vector:
        """Returns a single random sample drawn from the measure."""
        if self._sample is None:
            raise NotImplementedError("A sample method is not set for this measure.")
        return self._sample()

    def samples(self, n: int) -> List[Vector]:
        """Returns a list of n random samples from the measure."""
        if n < 1:
            raise ValueError("Number of samples must be a positive integer.")
        return [self.sample() for _ in range(n)]

    def sample_expectation(self, n: int) -> Vector:
        """Estimates the expectation by drawing n samples."""
        if n < 1:
            raise ValueError("Number of samples must be a positive integer.")
        return self.domain.sample_expectation(self.samples(n))

    def sample_pointwise_variance(self, n: int) -> Vector:
        """
        Estimates the pointwise variance by drawing n samples.

        This method is only available if the domain supports vector
        multiplication.
        """
        if not isinstance(self.domain, HilbertModule):
            raise NotImplementedError(
                "Pointwise variance requires vector multiplication on the domain."
            )
        if n < 1:
            raise ValueError("Number of samples must be a positive integer.")

        samples = self.samples(n)
        expectation = self.expectation
        variance = self.domain.zero

        for sample in samples:
            diff = self.domain.subtract(sample, expectation)
            prod = self.domain.vector_multiply(diff, diff)
            self.domain.axpy(1 / n, prod, variance)

        return variance

    def affine_mapping(
        self, /, *, operator: LinearOperator = None, translation: Vector = None
    ) -> GaussianMeasure:
        """
        Transforms the measure under an affine map `y = A(x) + b`.

        If a random variable `x` is distributed according to this Gaussian
        measure, `x ~ N(μ, C)`, this method computes the new Gaussian measure
        for the transformed variable `y`.

        The new measure will have:
        - Expectation: `μ_y = A @ μ + b`
        - Covariance: `C_y = A @ C @ A*`

        Args:
            operator: The linear operator `A` in the transformation.
                Defaults to the identity.
            translation: The translation vector `b`. Defaults to zero.

        Returns:
            The transformed `GaussianMeasure`.
        """
        _operator = (
            operator if operator is not None else self.domain.identity_operator()
        )
        _translation = (
            translation if translation is not None else _operator.codomain.zero
        )

        new_expectation = _operator.codomain.add(
            _operator(self.expectation), _translation
        )

        if self.covariance_factor_set:
            new_covariance_factor = _operator @ self.covariance_factor
            return GaussianMeasure(
                covariance_factor=new_covariance_factor, expectation=new_expectation
            )
        else:
            new_covariance = _operator @ self.covariance @ _operator.adjoint

            def new_sample() -> Vector:
                return _operator.codomain.add(_operator(self.sample()), _translation)

            return GaussianMeasure(
                covariance=new_covariance,
                expectation=new_expectation,
                sample=new_sample if self.sample_set else None,
            )

    def as_multivariate_normal(
        self, /, *, parallel: bool = False, n_jobs: int = -1
    ) -> multivariate_normal:
        """
        Returns the measure as a `scipy.stats.multivariate_normal` object.

        This is only possible if the measure is defined on a EuclideanSpace.

        If the covariance matrix has small negative eigenvalues due to numerical
        precision issues, this method attempts to correct them by setting them
        to zero.

        Args:
            parallel (bool, optional): If `True`, computes the dense covariance
                matrix in parallel. Defaults to `False`.
            n_jobs (int, optional): The number of parallel jobs to use. `-1`
                uses all available cores. Defaults to -1.
        """
        if not isinstance(self.domain, EuclideanSpace):
            raise NotImplementedError(
                "Method only defined for measures on Euclidean space."
            )

        mean_vector = self.expectation

        # Pass the parallelization arguments directly to the matrix creation method
        cov_matrix = self.covariance.matrix(
            dense=True, parallel=parallel, n_jobs=n_jobs
        )

        try:
            # First, try to create the distribution directly.
            return multivariate_normal(
                mean=mean_vector, cov=cov_matrix, allow_singular=True
            )
        except ValueError:
            # If it fails, clean the covariance matrix and try again.
            warnings.warn(
                "Covariance matrix is not positive semi-definite due to "
                "numerical errors. Setting negative eigenvalues to zero.",
                UserWarning,
            )

            eigenvalues, eigenvectors = eigh(cov_matrix)
            eigenvalues[eigenvalues < 0] = 0
            cleaned_cov = eigenvectors @ diags(eigenvalues) @ eigenvectors.T

            return multivariate_normal(
                mean=mean_vector, cov=cleaned_cov, allow_singular=True
            )

    def low_rank_approximation(
        self,
        size_estimate: int,
        /,
        *,
        method: str = "variable",
        max_rank: int = None,
        power: int = 2,
        rtol: float = 1e-4,
        block_size: int = 10,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> GaussianMeasure:
        """
        Constructs a low-rank approximation of the measure.

        The covariance operator is replaced by a low-rank approximation, which
        can be much more efficient for sampling and storage.

        Args:
            size_estimate: For 'fixed' method, the exact target rank. For 'variable'
                       method, this is the initial rank to sample.
            method ({'variable', 'fixed'}): The algorithm to use.
            - 'variable': (Default) Progressively samples to find the rank needed
                          to meet tolerance `rtol`, stopping at `max_rank`.
            - 'fixed': Returns a basis with exactly `size_estimate` columns.
            max_rank: For 'variable' method, a hard limit on the rank. Ignored if
                    method='fixed'. Defaults to min(m, n).
            power: Number of power iterations to improve accuracy.
            rtol: Relative tolerance for the 'variable' method. Ignored if
                method='fixed'.
            block_size: Number of new vectors to sample per iteration in 'variable'
                        method. Ignored if method='fixed'.
            parallel: Whether to use parallel matrix multiplication.
            n_jobs: Number of jobs for parallelism.

        Returns:
            GaussianMeasure: The new, low-rank Gaussian measure.

        Notes:
            Parallel implemention only currently possible with fixed-rank decompositions.
        """
        covariance_factor = self.covariance.random_cholesky(
            size_estimate,
            method=method,
            max_rank=max_rank,
            power=power,
            rtol=rtol,
            block_size=block_size,
            parallel=parallel,
            n_jobs=n_jobs,
        )

        return GaussianMeasure(
            covariance_factor=covariance_factor,
            expectation=self.expectation,
        )

    def two_point_covariance(self, point: Any) -> Vector:
        """
        Computes the two-point covariance function.

        For measures on spaces of functions, this returns the covariance
        between the function value at a fixed `point` and all other points.
        This requires the domain to support point evaluation (a `dirac` method).
        """
        if not hasattr(self.domain, "dirac_representation"):
            raise NotImplementedError(
                "Point evaluation is not defined for this measure's domain."
            )

        u = self.domain.dirac_representation(point)
        cov = self.covariance
        return cov(u)

    def __neg__(self) -> GaussianMeasure:
        """Returns a measure with a negated expectation."""
        if self.covariance_factor_set:
            return GaussianMeasure(
                covariance_factor=self.covariance_factor,
                expectation=self.domain.negative(self.expectation),
            )
        else:
            new_sample = (
                (lambda: self.domain.negative(self.sample()))
                if self.sample_set
                else None
            )
            return GaussianMeasure(
                covariance=self.covariance,
                expectation=self.domain.negative(self.expectation),
                sample=new_sample,
            )

    def __mul__(self, alpha: float) -> GaussianMeasure:
        """Scales the measure by a scalar alpha."""
        if self.covariance_factor_set:
            return GaussianMeasure(
                covariance_factor=alpha * self.covariance_factor,
                expectation=self.domain.multiply(alpha, self.expectation),
            )

        new_sample = (
            (lambda: self.domain.multiply(alpha, self.sample()))
            if self.sample_set
            else None
        )
        return GaussianMeasure(
            covariance=alpha**2 * self.covariance,
            expectation=self.domain.multiply(alpha, self.expectation),
            sample=new_sample,
        )

    def __rmul__(self, alpha: float) -> GaussianMeasure:
        """Scales the measure by a scalar alpha."""
        return self * alpha

    def __truediv__(self, a: float) -> GaussianMeasure:
        """Returns the division of the measure by a scalar."""
        return self * (1.0 / a)

    def __add__(self, other: GaussianMeasure) -> GaussianMeasure:
        """
        Adds two independent Gaussian measures defined on the same domain.
        """
        if self.domain != other.domain:
            raise ValueError("Measures must be defined on the same domain.")

        new_sample = (
            (lambda: self.domain.add(self.sample(), other.sample()))
            if self.sample_set and other.sample_set
            else None
        )
        return GaussianMeasure(
            covariance=self.covariance + other.covariance,
            expectation=self.domain.add(self.expectation, other.expectation),
            sample=new_sample,
        )

    def __sub__(self, other: GaussianMeasure) -> GaussianMeasure:
        """
        Subtracts two independent Gaussian measures on the same domain.
        """
        if self.domain != other.domain:
            raise ValueError("Measures must be defined on the same domain.")

        new_sample = (
            (lambda: self.domain.subtract(self.sample(), other.sample()))
            if self.sample_set and other.sample_set
            else None
        )
        return GaussianMeasure(
            covariance=self.covariance + other.covariance,
            expectation=self.domain.subtract(self.expectation, other.expectation),
            sample=new_sample,
        )

    def _sample_from_factor(self) -> Vector:
        """Default sampling method when a covariance factor is provided."""
        covariance_factor = self.covariance_factor
        # Draw from standard normal in the Euclidean space
        w = np.random.randn(covariance_factor.domain.dim)
        # Map to the Hilbert space
        value = covariance_factor(w)
        # Add the expectation
        return self.domain.add(value, self.expectation)
