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
from typing import Callable, Optional, Any, List, TYPE_CHECKING, Tuple
import warnings

import numpy as np
from scipy.linalg import cholesky, cho_solve
from scipy.linalg import eigh
from scipy.sparse import diags
from scipy.stats import multivariate_normal
from joblib import Parallel, delayed

from .hilbert_space import EuclideanSpace, HilbertModule, Vector

from .linear_operators import (
    LinearOperator,
    DiagonalSparseMatrixLinearOperator,
)

from .affine_operators import AffineOperator

from .direct_sum import (
    BlockDiagonalLinearOperator,
)


# This block is only processed by type checkers, not at runtime.
if TYPE_CHECKING:
    from .hilbert_space import HilbertSpace


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
                 measure. Defaults to the zero vector of the space (stored internally as None).
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

        # Store exactly what is passed (None implies a zero expectation)
        self._expectation: Optional[Vector] = expectation

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

    # ---------------------------------------- #
    #                Constructors              #
    # ---------------------------------------- #

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
        # Optimize expectation aggregation
        if all(measure.has_zero_expectation for measure in measures):
            expectation = None
        else:
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

    # ---------------------------------------- #
    #                 Properties               #
    # ---------------------------------------- #

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
    def has_zero_expectation(self) -> bool:
        """True if the measure has an exactly zero expectation."""
        return self._expectation is None

    @property
    def expectation(self) -> Vector:
        """The expectation (mean) of the measure."""
        if self.has_zero_expectation:
            return self.domain.zero
        return self._expectation

    @property
    def sample_set(self) -> bool:
        """True if a method for drawing samples is available."""
        return self._sample is not None

    # ---------------------------------------- #
    #               Public methods             #
    # ---------------------------------------- #

    def sample(self) -> Vector:
        """Returns a single random sample drawn from the measure."""
        if self._sample is None:
            raise NotImplementedError("A sample method is not set for this measure.")
        return self._sample()

    def samples(
        self, n: int, /, *, parallel: bool = False, n_jobs: int = -1
    ) -> List[Vector]:
        """
        Returns a list of n random samples from the measure.

        Args:
            n: Number of samples to draw.
            parallel: If True, draws samples in parallel.
            n_jobs: Number of CPU cores to use. -1 means all available.
        """
        if n < 1:
            raise ValueError("Number of samples must be a positive integer.")

        if not parallel:
            return [self.sample() for _ in range(n)]

        return Parallel(n_jobs=n_jobs)(delayed(self.sample)() for _ in range(n))

    def sample_expectation(
        self, n: int, /, *, parallel: bool = False, n_jobs: int = -1
    ) -> Vector:
        """
        Estimates the expectation by drawing n samples.

        Args:
            n: Number of samples to draw.
            parallel: If True, draws samples in parallel.
            n_jobs: Number of CPU cores to use. -1 means all available.
        """
        if n < 1:
            raise ValueError("Number of samples must be a positive integer.")
        return self.domain.sample_expectation(
            self.samples(n, parallel=parallel, n_jobs=n_jobs)
        )

    def sample_pointwise_variance(
        self, n: int, /, *, parallel: bool = False, n_jobs: int = -1
    ) -> Vector:
        """
        Estimates the pointwise variance by drawing n samples.

        Args:
            n: Number of samples to draw.
            parallel: If True, draws samples in parallel.
            n_jobs: Number of CPU cores to use. -1 means all available.

        Notes:
            This method is only implemented for measures on HilbertModules
            so that products of vectors can be defined.
        """
        if not isinstance(self.domain, HilbertModule):
            raise NotImplementedError(
                "Pointwise variance requires vector multiplication on the domain."
            )
        if n < 1:
            raise ValueError("Number of samples must be a positive integer.")

        # Draw samples
        samples = self.samples(n, parallel=parallel, n_jobs=n_jobs)

        # Compute variance using vector arithmetic
        variance = self.domain.zero

        if self.has_zero_expectation:
            for sample in samples:
                prod = self.domain.vector_multiply(sample, sample)
                self.domain.axpy(1 / n, prod, variance)
        else:
            expectation = self.expectation
            for sample in samples:
                diff = self.domain.subtract(sample, expectation)
                prod = self.domain.vector_multiply(diff, diff)
                self.domain.axpy(1 / n, prod, variance)

        return variance

    def sample_pointwise_std(
        self, n: int, /, *, parallel: bool = False, n_jobs: int = -1
    ) -> Vector:
        """
        Estimates the pointwise standard deviation by drawing n samples.

        Args:
            n: Number of samples to draw.
            parallel: If True, draws samples in parallel.
            n_jobs: Number of CPU cores to use. -1 means all available.

        Notes:
            This method is only implemented for measures on HilbertModules
            so that products of vectors can be defined.
        """
        variance = self.sample_pointwise_variance(n, parallel=parallel, n_jobs=n_jobs)
        return self.domain.vector_sqrt(variance)

    def deflated_pointwise_variance(
        self,
        rank: int,
        /,
        *,
        size_estimate: int = 0,
        method: str = "variable",
        max_samples: int = None,
        rtol: float = 1e-2,
        block_size: int = 10,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> Vector:
        """
        Estimates the pointwise variance using a deflated Hutchinson's method.
        """
        from .hilbert_space import HilbertModule

        if not isinstance(self.domain, HilbertModule):
            raise NotImplementedError(
                "Pointwise variance requires vector multiplication on the domain."
            )

        if rank < 0 or size_estimate < 0:
            raise ValueError("Rank and size_estimate must be non-negative.")

        space = self.domain
        deterministic_var = space.zero

        # -------------------------------------------------------------
        # 1. Deterministic Low-Rank Variance
        # -------------------------------------------------------------
        if rank > 0:
            if rank == 1:
                raise ValueError("Rank must be greater than 1")

            F = self.low_rank_approximation(rank, method="fixed").covariance_factor

            actual_rank = F.domain.dim

            for i in range(actual_rank):
                e_i = F.domain.basis_vector(i)
                spatial_mode = F(e_i)
                squared_mode = space.vector_multiply(spatial_mode, spatial_mode)
                space.axpy(1.0, squared_mode, deterministic_var)

            residual_cov = self.covariance - (F @ F.adjoint)
        else:
            residual_cov = self.covariance

        # -------------------------------------------------------------
        # 2. Stochastic Residual Variance (Progressive Hutchinson's)
        # -------------------------------------------------------------
        if size_estimate == 0 and method == "fixed":
            return deterministic_var

        if max_samples is None:
            max_samples = space.dim

        num_samples = min(size_estimate, max_samples)

        # Ensure Rademacher noise generates true white noise in the Hilbert space metric
        if hasattr(space, "squared_norms"):
            inv_norms = 1.0 / np.sqrt(space.squared_norms)
        else:
            inv_norms = np.array(
                [
                    1.0 / np.sqrt(space.squared_norm(space.basis_vector(i)))
                    for i in range(space.dim)
                ]
            )

        def _compute_sample_sum(n_samples_to_draw: int) -> Vector:
            """Helper to draw a block of samples and return their spatial sum."""

            def _single_sample():
                # Scale components by 1 / ||e_i|| so Cov(z) = Identity
                c_z = np.random.choice([-1.0, 1.0], size=space.dim) * inv_norms
                z = space.from_components(c_z)
                R_z = residual_cov(z)
                return space.vector_multiply(z, R_z)

            block_sum = space.zero
            if parallel:
                results = Parallel(n_jobs=n_jobs)(
                    delayed(_single_sample)() for _ in range(n_samples_to_draw)
                )
                for res in results:
                    space.axpy(1.0, res, block_sum)
            else:
                for _ in range(n_samples_to_draw):
                    space.axpy(1.0, _single_sample(), block_sum)
            return block_sum

        # Initial batch
        residual_sum = _compute_sample_sum(num_samples)
        residual_est = (
            space.multiply(1.0 / num_samples, residual_sum)
            if num_samples > 0
            else space.zero
        )

        # Progressive sampling
        if method == "variable" and num_samples < max_samples:
            while num_samples < max_samples:
                old_est = space.copy(residual_est)

                samples_to_add = min(block_size, max_samples - num_samples)
                new_sum = _compute_sample_sum(samples_to_add)

                # Update running average
                space.axpy(1.0, new_sum, residual_sum)
                total_samples = num_samples + samples_to_add
                residual_est = space.multiply(1.0 / total_samples, residual_sum)

                # Check convergence using the Hilbert space norm
                norm_new = space.norm(residual_est)
                if norm_new > 0:
                    diff = space.subtract(residual_est, old_est)
                    error = space.norm(diff) / norm_new
                    if error < rtol:
                        break

                num_samples = total_samples

        return space.add(deterministic_var, residual_est)

    def deflated_pointwise_std(
        self,
        rank: int,
        /,
        *,
        size_estimate: int = 0,
        method: str = "variable",
        max_samples: int = None,
        rtol: float = 1e-2,
        block_size: int = 10,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> Vector:
        """
        Estimates the pointwise standard deviation using a deflated Hutchinson's method.

        This method wraps `deflated_pointwise_variance` and returns the pointwise
        square root of the result. It supports both fixed and progressive variable
        sampling strategies for the stochastic residual.

        Args:
            rank: The rank of the deterministic low-rank approximation.
            size_estimate: Initial number of samples for the stochastic residual.
            method: 'variable' to sample until convergence, 'fixed' to stop at size_estimate.
            max_samples: Hard limit on residual samples (defaults to domain dimension).
            rtol: Relative tolerance for the 'variable' method.
            block_size: Number of new samples per iteration in 'variable' method.
            parallel: If True, computes the stochastic residual samples in parallel.
            n_jobs: Number of CPU cores to use. -1 means all available.

        Returns:
            A Vector representing the pointwise standard deviation.

        Raises:
            NotImplementedError: If the domain is not a HilbertModule.
        """
        # The variance method already checks if the domain is a HilbertModule,
        # but we check it here too just to give a clear error message specifically for std.
        from .hilbert_space import HilbertModule

        if not isinstance(self.domain, HilbertModule):
            raise NotImplementedError(
                "Pointwise standard deviation requires vector multiplication on the domain "
                "(the domain must be a HilbertModule)."
            )

        variance = self.deflated_pointwise_variance(
            rank,
            size_estimate=size_estimate,
            method=method,
            max_samples=max_samples,
            rtol=rtol,
            block_size=block_size,
            parallel=parallel,
            n_jobs=n_jobs,
        )

        return self.domain.vector_sqrt(variance)

    def with_dense_covariance(self, parallel: bool = False, n_jobs: int = -1):
        """
        Forms a new Gaussian measure equivalent to the existing one, but
        with its covariance matrix stored in dense form. The dense matrix
        calculation can optionally be parallelised.
        """
        covariance_matrix = self.covariance.matrix(
            dense=True, galerkin=True, parallel=parallel, n_jobs=n_jobs
        )

        # Build the full measure with its dense factors (L and L^-1)
        measure = GaussianMeasure.from_covariance_matrix(
            self.domain, covariance_matrix, expectation=self._expectation
        )

        # Explicitly overwrite the lazy L @ L* composition with the strict dense operator
        measure._covariance = LinearOperator.self_adjoint_from_matrix(
            self.domain, covariance_matrix
        )

        return measure

    def affine_mapping(
        self,
        /,
        *,
        operator: LinearOperator = None,
        translation: Vector = None,
        affine_operator: AffineOperator = None,
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
            affine_operator: An `AffineOperator` instance representing the map.
                If provided, `operator` and `translation` must not be used.

        Returns:
            The transformed `GaussianMeasure`.
        """

        # 1. Handle argument exclusivity and extract the operator/translation
        if affine_operator is not None:
            if operator is not None or translation is not None:
                raise ValueError(
                    "Cannot provide `affine_operator` alongside `operator` "
                    "or `translation`."
                )
            _operator = affine_operator.linear_part
            _translation = affine_operator.translation_part
        else:
            _operator = (
                operator if operator is not None else self.domain.identity_operator()
            )
            # We delay applying codomain.zero unless required by math
            _translation = translation

        # Expectation bypass
        if self.has_zero_expectation:
            new_expectation = (
                _translation  # Remains None (zero) if _translation is None
            )
        else:
            mapped_exp = _operator(self.expectation)
            if _translation is None:
                new_expectation = mapped_exp
            else:
                new_expectation = _operator.codomain.add(mapped_exp, _translation)

        if self.covariance_factor_set:
            new_covariance_factor = _operator @ self.covariance_factor
            return GaussianMeasure(
                covariance_factor=new_covariance_factor, expectation=new_expectation
            )
        else:
            new_covariance = _operator @ self.covariance @ _operator.adjoint

            def new_sample() -> Vector:
                mapped_sample = _operator(self.sample())
                if _translation is None:
                    return mapped_sample
                return _operator.codomain.add(mapped_sample, _translation)

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
            dense=True, galerkin=True, parallel=parallel, n_jobs=n_jobs
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
            max_rank: For 'variable' method, a hard limit on the rank.
            power: Number of power iterations to improve accuracy.
            rtol: Relative tolerance for the 'variable' method.
            block_size: Number of new vectors to sample per iteration.
            parallel: Whether to use parallel matrix multiplication.
            n_jobs: Number of jobs for parallelism.

        Returns:
            GaussianMeasure: The new, low-rank Gaussian measure.
        """
        # Local import to prevent circular dependency with low_rank.py
        from .low_rank import LowRankCholesky

        # We pass measure=None to probe the operator with component-based white noise N(0, I)
        lr_cholesky = LowRankCholesky.from_randomized(
            self.covariance,
            size_estimate,
            measure=None,
            method=method,
            max_rank=max_rank,
            power=power,
            rtol=rtol,
            block_size=block_size,
            parallel=parallel,
            n_jobs=n_jobs,
        )

        # LowRankCholesky provides the L factor we need for the new measure
        return GaussianMeasure(
            covariance_factor=lr_cholesky.l_factor,
            expectation=self._expectation,
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

    def directional_statistics(self, direction: Vector) -> Tuple[float, float]:
        """
        Returns the expectation and variance of the scalar Gaussian <x, direction>.

        Args:
            direction: The vector defining the linear functional.

        Returns:
            A tuple of (mean, variance).
        """
        expectation = (
            0.0
            if self.has_zero_expectation
            else self.domain.inner_product(self.expectation, direction)
        )
        variance = self.domain.inner_product(self.covariance(direction), direction)
        return expectation, variance

    def directional_covariance(self, d1: Vector, d2: Vector) -> float:
        """
            Returns the covariance between <x, d1> and <x, d2>.

        Args:
            d1: The first direction vector.
            d2: The second direction vector.

        Returns:
            The scalar covariance <Q d1, d2>.
        """
        return self.domain.inner_product(self.covariance(d1), d2)

    def directional_variance(self, d: Vector) -> float:
        """
            Returns the variance of <x, d>

        Args:
            d: The direction

        Returns:
            The scalar variance <Q d, d>.
        """
        return self.directional_covariance(d, d)

    def zero_expectation(self) -> GaussianMeasure:
        """
        Returns a new measure with the same covariance, but
        with expectation set to zero.
        """
        if self.covariance_factor_set:
            return GaussianMeasure(
                covariance_factor=self.covariance_factor,
                expectation=None,
            )

        if self.sample_set:
            if self.has_zero_expectation:
                new_sample = self.sample
            else:
                exp = self.expectation

                def new_sample():
                    return self.domain.subtract(self.sample(), exp)

        else:
            new_sample = None

        return GaussianMeasure(
            covariance=self.covariance,
            expectation=None,
            sample=new_sample,
        )

    def rescale_directional_variance(
        self, direction: Vector, std: float
    ) -> GaussianMeasure:
        """
        Returns a new measure where Var[<x, direction>] is scaled to std^2.

        The expectation of the resulting measure is unchanged.
        """
        current_var = self.directional_variance(direction)
        if current_var <= 0:
            raise ValueError("Directional variance must be positive to rescale.")
        norm = std / np.sqrt(current_var)
        shifted_measure = self.zero_expectation()
        scaled_measure = norm * shifted_measure
        return scaled_measure.affine_mapping(translation=self._expectation)

    def kl_divergence(self, other: GaussianMeasure) -> float:
        """
        Computes the exact Kullback-Leibler (KL) divergence D_KL(self || other).

        This computes the divergence of 'self' (P) from the prior/reference
        measure 'other' (Q). It uses the dense Galerkin matrix representations
        of the covariance operators, making it suitable for measures on
        low-dimensional spaces or direct sums thereof.

        Args:
            other: The other GaussianMeasure (usually the prior/approximating measure).

        Returns:
            The KL divergence as a float.

        Raises:
            ValueError: If the measures are not on the same domain, or if
                        the covariances are not positive definite.
        """
        if self.domain != other.domain:
            raise ValueError("Measures must be defined on the same domain.")

        k = self.domain.dim

        # 1. Extract the dense Galerkin matrices
        G_p = self.covariance.matrix(dense=True, galerkin=True)
        G_q = other.covariance.matrix(dense=True, galerkin=True)

        # Cholesky decomposition of Q's covariance matrix for fast solving/determinant
        try:
            L_q = cholesky(G_q, lower=True)
        except np.linalg.LinAlgError:
            raise ValueError(
                "The covariance matrix of the 'other' measure is not positive definite."
            )

        # 2. Trace term: tr(Sigma_Q^-1 Sigma_P)
        X = cho_solve((L_q, True), G_p)
        trace_term = np.trace(X)

        # 3. Log-determinant term: ln(det(Sigma_Q)) - ln(det(Sigma_P))
        log_det_q = 2.0 * np.sum(np.log(np.diag(L_q)))
        sign_p, log_det_p = np.linalg.slogdet(G_p)
        if sign_p <= 0:
            raise ValueError(
                "The covariance matrix of 'self' is not positive definite."
            )
        log_det_term = log_det_q - log_det_p

        # 4. Mahalanobis term: <mu_P - mu_Q, Sigma_Q^-1 (mu_P - mu_Q)>
        if self.has_zero_expectation and other.has_zero_expectation:
            mahalanobis_term = 0.0
        else:
            diff = self.domain.subtract(self.expectation, other.expectation)
            diff_dual = self.domain.to_dual(diff)
            diff_c_prime = self.domain.dual.to_components(diff_dual)
            y = cho_solve((L_q, True), diff_c_prime)
            mahalanobis_term = np.dot(diff_c_prime, y)

        # 5. Assemble the KL divergence
        kl_div = 0.5 * (trace_term + mahalanobis_term - k + log_det_term)
        return float(kl_div)

    # ---------------------------------------- #
    #               Special methods            #
    # ---------------------------------------- #

    def __neg__(self) -> GaussianMeasure:
        """Returns a measure with a negated expectation."""
        new_expectation = (
            None
            if self.has_zero_expectation
            else self.domain.negative(self.expectation)
        )

        if self.covariance_factor_set:
            return GaussianMeasure(
                covariance_factor=self.covariance_factor,
                expectation=new_expectation,
            )
        else:
            new_sample = (
                (lambda: self.domain.negative(self.sample()))
                if self.sample_set
                else None
            )
            return GaussianMeasure(
                covariance=self.covariance,
                expectation=new_expectation,
                sample=new_sample,
            )

    def __mul__(self, alpha: float) -> GaussianMeasure:
        """Scales the measure by a scalar alpha."""
        new_expectation = (
            None
            if self.has_zero_expectation
            else self.domain.multiply(alpha, self.expectation)
        )

        if self.covariance_factor_set:
            return GaussianMeasure(
                covariance_factor=alpha * self.covariance_factor,
                expectation=new_expectation,
            )

        new_sample = (
            (lambda: self.domain.multiply(alpha, self.sample()))
            if self.sample_set
            else None
        )
        return GaussianMeasure(
            covariance=alpha**2 * self.covariance,
            expectation=new_expectation,
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

        if self.has_zero_expectation and other.has_zero_expectation:
            new_expectation = None
        elif self.has_zero_expectation:
            new_expectation = other.expectation
        elif other.has_zero_expectation:
            new_expectation = self.expectation
        else:
            new_expectation = self.domain.add(self.expectation, other.expectation)

        new_sample = (
            (lambda: self.domain.add(self.sample(), other.sample()))
            if self.sample_set and other.sample_set
            else None
        )
        return GaussianMeasure(
            covariance=self.covariance + other.covariance,
            expectation=new_expectation,
            sample=new_sample,
        )

    def __sub__(self, other: GaussianMeasure) -> GaussianMeasure:
        """
        Subtracts two independent Gaussian measures on the same domain.
        """
        if self.domain != other.domain:
            raise ValueError("Measures must be defined on the same domain.")

        if self.has_zero_expectation and other.has_zero_expectation:
            new_expectation = None
        elif self.has_zero_expectation:
            new_expectation = self.domain.negative(other.expectation)
        elif other.has_zero_expectation:
            new_expectation = self.expectation
        else:
            new_expectation = self.domain.subtract(self.expectation, other.expectation)

        new_sample = (
            (lambda: self.domain.subtract(self.sample(), other.sample()))
            if self.sample_set and other.sample_set
            else None
        )
        return GaussianMeasure(
            covariance=self.covariance + other.covariance,
            expectation=new_expectation,
            sample=new_sample,
        )

    # ---------------------------------------- #
    #               Private methods            #
    # ---------------------------------------- #

    def _sample_from_factor(self) -> Vector:
        """Default sampling method when a covariance factor is provided."""
        covariance_factor = self.covariance_factor
        w = covariance_factor.domain.random()
        value = covariance_factor(w)
        if self.has_zero_expectation:
            return value
        return self.domain.add(value, self.expectation)
