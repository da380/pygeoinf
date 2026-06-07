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
from typing import Callable, Optional, Any, List, TYPE_CHECKING, Tuple, Literal
import warnings

import numpy as np
from scipy.linalg import cholesky, cho_solve
from scipy.linalg import eigh
from scipy.sparse import diags
from scipy.stats import chi2, multivariate_normal
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from joblib import Parallel, delayed


from .hilbert_space import (
    EuclideanSpace,
    HilbertModule,
    MassWeightedHilbertModule,
    MassWeightedHilbertSpace,
    Vector,
)

from .linear_operators import (
    LinearOperator,
    DiagonalSparseMatrixLinearOperator,
    SparseMatrixLinearOperator,
)


from .linear_solvers import IterativeLinearSolver, LinearSolver

from .affine_operators import AffineOperator

from .direct_sum import BlockDiagonalLinearOperator, BlockLinearOperator

from .functional_calculus import LanczosOperatorFunction


if TYPE_CHECKING:
    from .hilbert_space import HilbertSpace
    from .low_rank import LowRankEig


class GaussianMeasure:
    """
    Represents a Gaussian measure on a Hilbert space.

    This class generalizes the multivariate normal distribution to abstract,
    potentially infinite-dimensional, Hilbert spaces. A measure is
    defined by its expectation (mean vector) and its covariance, which is a
    `LinearOperator` on the space.
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
            covariance: A self-adjoint and positive semi-definite linear
                operator on the domain.
            covariance_factor: A linear operator L such that the covariance
                C = L @ L*.
            expectation: The expectation (mean) of the measure. Defaults to
                the zero vector of the space (stored internally as None).
            sample: A function that returns a random sample from the measure.
                If a `covariance_factor` is given, a default sampler is created.
            inverse_covariance: The inverse of the covariance operator (the
                precision operator).
            inverse_covariance_factor: A factor Li of the inverse covariance,
                such that C_inv = Li.T @ Li.

        Raises:
            ValueError: If neither `covariance` nor `covariance_factor` is provided.
        """
        if covariance is None and covariance_factor is None:
            raise ValueError(
                "Neither covariance nor covariance_factor has been provided."
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

        self._expectation: Optional[Vector] = expectation

    # ---------------------------------------- #
    #                Constructors              #
    # ---------------------------------------- #

    @staticmethod
    def from_standard_deviation(
        domain: HilbertSpace,
        standard_deviation: float,
        /,
        *,
        expectation: Vector = None,
    ) -> GaussianMeasure:
        """
        Creates an isotropic Gaussian measure with a scaled identity covariance.

        Args:
            domain: The Hilbert space on which the measure is defined.
            standard_deviation: The uniform standard deviation for all dimensions.
            expectation: The mean vector. Defaults to the zero vector.

        Returns:
            A new GaussianMeasure instance.
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
            domain: The Hilbert space on which the measure is defined.
            standard_deviations: A 1D NumPy array representing the diagonal
                entries of the covariance factor.
            expectation: The mean vector. Defaults to the zero vector.

        Returns:
            A new GaussianMeasure instance.

        Raises:
            ValueError: If the size of the array does not match the space dimension.
        """
        if standard_deviations.size != domain.dim:
            raise ValueError(
                "Standard deviation vector does not have the correct length."
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

        Args:
            domain: The Hilbert space on which the measure is defined.
            covariance_matrix: A 2D symmetric positive semi-definite NumPy array.
            expectation: The mean vector. Defaults to the zero vector.
            rtol: Relative tolerance for clipping small negative eigenvalues
                caused by floating-point inaccuracies.

        Returns:
            A new GaussianMeasure instance.

        Raises:
            ValueError: If the matrix has significantly negative eigenvalues.
        """
        eigenvalues, U = eigh(covariance_matrix)

        if np.any(eigenvalues < 0):
            max_eig = np.max(np.abs(eigenvalues))
            min_eig = np.min(eigenvalues)

            if min_eig < -rtol * max_eig:
                raise ValueError(
                    "Covariance matrix has significantly negative eigenvalues, "
                    "indicating it is not positive semi-definite."
                )
            else:
                warnings.warn(
                    "Covariance matrix has small negative eigenvalues due to "
                    "numerical error. Clipping them to zero.",
                    UserWarning,
                )
                eigenvalues[eigenvalues < 0] = 0

        values = np.sqrt(eigenvalues)
        D = diags([values], [0])

        out_array = np.zeros_like(values, dtype=float)
        Di = diags([np.reciprocal(values, out=out_array, where=(values != 0))], [0])
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
    def from_samples(domain: HilbertSpace, samples: List[Vector], /) -> GaussianMeasure:
        """
        Estimates a Gaussian measure from a collection of sample vectors.

        Constructs an empirical mean and an unnormalized sample covariance operator
        using a tensor product expansion.

        Args:
            domain: The Hilbert space the samples belong to.
            samples: A list of sample vectors.

        Returns:
            A new GaussianMeasure instance.

        Raises:
            ValueError: If the list of samples is empty.
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
    def from_direct_sum(measures: List[GaussianMeasure], /) -> GaussianMeasure:
        """
        Constructs a product measure from a list of other measures.

        The resulting measure resides on the direct sum of the input domains,
        with block-diagonal covariance and concatenated expectations.

        Args:
            measures: A list of GaussianMeasure instances.

        Returns:
            A new GaussianMeasure instance defined on the direct sum space.
        """
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
        """True if the inverse covariance (precision) operator is available."""
        return self._inverse_covariance is not None

    @property
    def inverse_covariance(self) -> LinearOperator:
        """The inverse covariance (precision) operator. Raises AttributeError if not set."""
        if self._inverse_covariance is None:
            raise AttributeError("Inverse covariance is not set for this measure.")
        return self._inverse_covariance

    @property
    def covariance_factor_set(self) -> bool:
        """True if a covariance factor L (such that C = L @ L*) is available."""
        return self._covariance_factor is not None

    @property
    def covariance_factor(self) -> LinearOperator:
        """The covariance factor L. Raises AttributeError if not set."""
        if self._covariance_factor is None:
            raise AttributeError("Covariance factor has not been set.")
        return self._covariance_factor

    @property
    def inverse_covariance_factor_set(self) -> bool:
        """True if an inverse covariance factor is available."""
        return self._inverse_covariance_factor is not None

    @property
    def inverse_covariance_factor(self) -> LinearOperator:
        """The inverse covariance factor. Raises AttributeError if not set."""
        if self._inverse_covariance_factor is None:
            raise AttributeError("Inverse covariance factor has not been set.")
        return self._inverse_covariance_factor

    @property
    def has_zero_expectation(self) -> bool:
        """True if the measure is internally stored with an exactly zero expectation."""
        return self._expectation is None

    @property
    def expectation(self) -> Vector:
        """The expectation (mean) vector of the measure."""
        if self.has_zero_expectation:
            return self.domain.zero
        return self._expectation

    @property
    def sample_set(self) -> bool:
        """True if a method for drawing random samples is available."""
        return self._sample is not None

    # ---------------------------------------- #
    #               Public methods             #
    # ---------------------------------------- #

    def sample(self) -> Vector:
        """
        Returns a single random sample drawn from the measure.

        Returns:
            A randomly sampled vector.

        Raises:
            NotImplementedError: If a sample method is not set for this measure.
        """
        if self._sample is None:
            raise NotImplementedError("A sample method is not set for this measure.")
        return self._sample()

    def samples(
        self, n: int, /, *, parallel: bool = False, n_jobs: int = -1
    ) -> List[Vector]:
        """
        Returns a list of `n` independent random samples from the measure.

        Args:
            n: The number of samples to draw.
            parallel: If True, draws samples concurrently.
            n_jobs: The number of CPU cores to use if parallel=True.

        Returns:
            A list of sampled vectors.

        Raises:
            ValueError: If `n` is less than 1.
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
        Estimates the expectation vector by drawing `n` Monte Carlo samples.

        Args:
            n: The number of samples to use for the estimation.
            parallel: If True, draws samples concurrently.
            n_jobs: The number of CPU cores to use if parallel=True.

        Returns:
            The empirical expectation vector.
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
        Estimates the pointwise variance field by drawing `n` Monte Carlo samples.

        Args:
            n: The number of samples to use.
            parallel: If True, draws samples concurrently.
            n_jobs: The number of CPU cores to use if parallel=True.

        Returns:
            A vector representing the pointwise variance field.

        Raises:
            NotImplementedError: If the domain is not a HilbertModule (which
                provides pointwise multiplication).
        """
        if not isinstance(self.domain, HilbertModule):
            raise NotImplementedError(
                "Pointwise variance requires vector multiplication on the domain."
            )
        if n < 1:
            raise ValueError("Number of samples must be a positive integer.")

        samples = self.samples(n, parallel=parallel, n_jobs=n_jobs)
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
        Estimates the pointwise standard deviation field by drawing `n` Monte Carlo samples.

        Args:
            n: The number of samples to use.
            parallel: If True, draws samples concurrently.
            n_jobs: The number of CPU cores to use if parallel=True.

        Returns:
            A vector representing the pointwise standard deviation field.
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
        Estimates the pointwise variance field using a deflated Hutchinson's method.

        This combines a deterministic low-rank extraction (via SVD deflation) with
        a stochastic Hutchinson trace estimator for the residual variance.

        Args:
            rank: The rank of the deterministic SVD deflation.
            size_estimate: The initial number of stochastic residual samples.
            method: 'variable' to sample until `rtol` is met, 'fixed' otherwise.
            max_samples: Hard limit on stochastic residual samples.
            rtol: Relative tolerance for the stochastic residual phase.
            block_size: Number of samples added per check in the 'variable' method.
            parallel: If True, draws stochastic samples in parallel.
            n_jobs: The number of CPU cores to use.

        Returns:
            A vector representing the pointwise variance field.

        Raises:
            NotImplementedError: If the domain is not a HilbertModule.
        """
        if not isinstance(self.domain, HilbertModule):
            raise NotImplementedError(
                "Pointwise variance requires vector multiplication on the domain."
            )

        if rank < 0 or size_estimate < 0:
            raise ValueError("Rank and size_estimate must be non-negative.")

        space = self.domain
        deterministic_var = space.zero

        if rank > 0:
            if rank == 1:
                raise ValueError("Rank must be greater than 1.")

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

        if size_estimate == 0 and method == "fixed":
            return deterministic_var

        if max_samples is None:
            max_samples = space.dim

        num_samples = min(size_estimate, max_samples)

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
            def _single_sample():
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

        residual_sum = _compute_sample_sum(num_samples)
        residual_est = (
            space.multiply(1.0 / num_samples, residual_sum)
            if num_samples > 0
            else space.zero
        )

        if method == "variable" and num_samples < max_samples:
            while num_samples < max_samples:
                old_est = space.copy(residual_est)

                samples_to_add = min(block_size, max_samples - num_samples)
                new_sum = _compute_sample_sum(samples_to_add)

                space.axpy(1.0, new_sum, residual_sum)
                total_samples = num_samples + samples_to_add
                residual_est = space.multiply(1.0 / total_samples, residual_sum)

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
        Estimates the pointwise standard deviation field using a deflated Hutchinson's method.

        Args:
            rank: The rank of the deterministic SVD deflation.
            size_estimate: The initial number of stochastic residual samples.
            method: 'variable' to sample until `rtol` is met, 'fixed' otherwise.
            max_samples: Hard limit on stochastic residual samples.
            rtol: Relative tolerance for the stochastic residual phase.
            block_size: Number of samples added per check in the 'variable' method.
            parallel: If True, draws stochastic samples in parallel.
            n_jobs: The number of CPU cores to use.

        Returns:
            A vector representing the pointwise standard deviation field.
        """
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

    def credible_set(
        self,
        probability: float,
        /,
        *,
        geometry: str = "ellipsoid",
        rank: Optional[int] = None,
        open_set: bool = False,
        theta: Optional[float] = None,
        spectrum=None,
        spectrum_size: Optional[int] = None,
        radius_method: str = "auto",
        quantile_method: str = "auto",
        quantile_tol: float = 1e-2,
        fractional_apply: str = "auto",
        n_samples: int = 10_000,
        lanczos_size_estimate: int = 50,
        lanczos_method: Literal["variable", "fixed"] = "fixed",
        lanczos_max_k: Optional[int] = None,
        lanczos_rtol: float = 1e-3,
        lanczos_atol: float = 1e-8,
        lanczos_check_interval: int = 5,
        spectrum_low_rank_kwargs: Optional[dict] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        r"""
        Return a probability-calibrated Gaussian credible subset.

        Five geometries are supported:

        ``"ellipsoid"`` / ``"mahalanobis"`` / ``"domain"``
            The classical Mahalanobis ellipsoid.

        ``"cameron_martin"`` / ``"cm"`` / ``"ball"`` / ``"norm_ball"``
            The ellipsoid expressed as a unit ball in the Cameron-Martin geometry.

        ``"ambient_ball"`` / ``"ambient"``
            The ambient norm ball $\{m : \|m - m_0\|_H \le r_p\}$.

        ``"weakened_ellipsoid"`` / ``"fractional"``
            The weakened-covariance ellipsoid $\{m : \|C^{-\theta/2}(m-m_0)\|_H \le r_p\}$.

        Args:
            probability: Credible probability $p$, strictly between 0 and 1.
            geometry: Selects the ball/ellipsoid family (see above).
            rank: Chi-square degrees of freedom (legacy modes only).
            open_set: If true, return the open version of the set.
            theta: Fractional exponent in $(0, 1)$, required for weakened_ellipsoid.
            spectrum: Covariance spectrum specification.
            spectrum_size: Truncation length when ``spectrum`` is callable or ``None``.
            radius_method: ``"auto"``, ``"spectral"``, or ``"sampling"``.
            quantile_method: Weighted-chi-square quantile method.
            quantile_tol: Desired relative accuracy of the weighted-chi-square quantile.
            fractional_apply: How to apply $C^{-\theta/2}$ for the weakened ellipsoid.
            n_samples: Monte Carlo sample count for sampling radius.
            lanczos_size_estimate: Initial or fixed Krylov dimension for Lanczos fractional evaluation.
            lanczos_method: 'fixed' or 'variable' dynamic convergence for Lanczos.
            lanczos_max_k: Maximum Krylov dimension if 'variable' is used.
            lanczos_rtol: Relative tolerance for Lanczos convergence.
            lanczos_atol: Absolute tolerance for Lanczos convergence.
            lanczos_check_interval: Number of iterations between Lanczos convergence checks.
            spectrum_low_rank_kwargs: Extra kwargs forwarded to ``LowRankEig.from_randomized``.
            rng: Optional NumPy generator for Monte Carlo paths.

        Returns:
            An Ellipsoid or Ball defining the credible subset.
        """
        if not 0.0 < probability < 1.0:
            raise ValueError("Probability must lie strictly between 0 and 1.")

        geometry_key = geometry.lower().replace("-", "_")

        if geometry_key in {"ellipsoid", "domain", "mahalanobis"}:
            return self._credible_set_chi2_ellipsoid(
                probability, rank=rank, open_set=open_set
            )
        if geometry_key in {"cameron_martin", "cm", "ball", "norm_ball"}:
            return self._credible_set_chi2_cameron_martin(
                probability, rank=rank, open_set=open_set
            )

        if geometry_key in {"ambient_ball", "ambient", "h_ball"}:
            return self._credible_set_ambient_ball(
                probability,
                open_set=open_set,
                spectrum=spectrum,
                spectrum_size=spectrum_size,
                radius_method=radius_method,
                quantile_method=quantile_method,
                quantile_tol=quantile_tol,
                n_samples=n_samples,
                spectrum_low_rank_kwargs=spectrum_low_rank_kwargs,
                rng=rng,
            )
        if geometry_key in {"weakened_ellipsoid", "fractional"}:
            if theta is None:
                raise ValueError("theta is required for geometry='weakened_ellipsoid'.")
            if not 0.0 < theta < 1.0:
                raise ValueError(
                    "theta must lie strictly in (0, 1); the boundary "
                    "theta=1 is the Cameron-Martin norm — use "
                    "geometry='cameron_martin' for the finite-rank "
                    "chi-square version."
                )
            return self._credible_set_weakened_ellipsoid(
                probability,
                theta=theta,
                open_set=open_set,
                spectrum=spectrum,
                spectrum_size=spectrum_size,
                radius_method=radius_method,
                quantile_method=quantile_method,
                quantile_tol=quantile_tol,
                fractional_apply=fractional_apply,
                n_samples=n_samples,
                lanczos_size_estimate=lanczos_size_estimate,
                lanczos_method=lanczos_method,
                lanczos_max_k=lanczos_max_k,
                lanczos_rtol=lanczos_rtol,
                lanczos_atol=lanczos_atol,
                lanczos_check_interval=lanczos_check_interval,
                spectrum_low_rank_kwargs=spectrum_low_rank_kwargs,
                rng=rng,
            )

        raise ValueError(
            "Geometry must be one of 'ellipsoid', 'cameron_martin', "
            "'ambient_ball', or 'weakened_ellipsoid'."
        )

    def ambient_ball(self, probability: float, /, **kwargs):
        """Shortcut for ``credible_set(..., geometry='ambient_ball', ...)``."""
        kwargs.setdefault("geometry", "ambient_ball")
        return self.credible_set(probability, **kwargs)

    def weakened_ellipsoid(self, probability: float, /, *, theta: float, **kwargs):
        """Shortcut for the weakened-covariance ellipsoid mode."""
        kwargs.setdefault("geometry", "weakened_ellipsoid")
        kwargs["theta"] = theta
        return self.credible_set(probability, **kwargs)

    def with_dense_covariance(
        self, /, *, parallel: bool = False, n_jobs: int = -1
    ) -> GaussianMeasure:
        """
        Forms a new Gaussian measure equivalent to the existing one, but
        with its covariance matrix stored explicitly in dense form.

        Args:
            parallel: If True, computes the dense matrix concurrently.
            n_jobs: Number of CPU cores to use if parallel=True.

        Returns:
            A new GaussianMeasure instance backed by a dense matrix.
        """
        covariance_matrix = self.covariance.matrix(
            dense=True, galerkin=True, parallel=parallel, n_jobs=n_jobs
        )

        measure = GaussianMeasure.from_covariance_matrix(
            self.domain, covariance_matrix, expectation=self._expectation
        )

        measure._covariance = LinearOperator.self_adjoint_from_matrix(
            self.domain, covariance_matrix
        )

        return measure

    def with_regularized_inverse(
        self,
        solver: LinearSolver,
        /,
        *,
        damping: float = 0.0,
        preconditioner: Optional[LinearOperator] = None,
    ) -> GaussianMeasure:
        """
        Returns a new GaussianMeasure with a well-defined precision operator
        (inverse covariance) computed via Tikhonov regularization.

        Args:
            solver: The linear solver used to invert the covariance.
            damping: Tikhonov regularization parameter added to the diagonal.
            preconditioner: Optional preconditioner for iterative solvers.

        Returns:
            A new GaussianMeasure instance equipped with an inverse covariance.

        Raises:
            ValueError: If the damping parameter is negative.
        """
        if damping < 0.0:
            raise ValueError("Damping must be non-negative.")

        if damping == 0.0 and self.inverse_covariance_set:
            return self

        if damping > 0.0:
            identity = self.domain.identity_operator()
            regularized_covariance = self.covariance + damping * identity
        else:
            regularized_covariance = self.covariance

        if isinstance(solver, IterativeLinearSolver):
            inverse_covariance = solver(
                regularized_covariance, preconditioner=preconditioner
            )
        else:
            inverse_covariance = solver(regularized_covariance)

        if self.sample_set and damping > 0.0:
            std_dev = np.sqrt(damping)
            noise_measure = GaussianMeasure.from_standard_deviation(
                self.domain, std_dev
            )

            def new_sample() -> Vector:
                return self.domain.add(self.sample(), noise_measure.sample())

        else:
            new_sample = self._sample if self.sample_set else None

        return GaussianMeasure(
            covariance=regularized_covariance,
            expectation=self._expectation,
            sample=new_sample,
            inverse_covariance=inverse_covariance,
        )

    def with_sparse_approximation(
        self,
        /,
        *,
        threshold: float = 1e-3,
        max_nnz: Optional[int] = None,
        diag_rank: int = 0,
        diag_samples: int = 0,
        regularization_fraction: float = 1e-4,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> GaussianMeasure:
        """
        Creates an approximately equivalent measure with a sparse covariance matrix
        and an exactly factorized sparse inverse, built entirely matrix-free.

        Args:
            threshold: Minimum correlation required to keep an off-diagonal element.
            max_nnz: Maximum number of non-zero elements allowed per column.
            diag_rank: Rank of deterministic SVD used to estimate the diagonal.
            diag_samples: Number of stochastic samples used to estimate the diagonal.
            regularization_fraction: Tikhonov regularization applied before sparse inversion.
            parallel: If True, computes the sparse approximations concurrently.
            n_jobs: Number of CPU cores to use if parallel=True.

        Returns:
            A new GaussianMeasure backed by sparse operators.
        """
        from .low_rank import deflated_diagonal

        dim = self.domain.dim

        if diag_rank == 0 and diag_samples == 0:
            diag_vars = self.covariance.extract_diagonal(
                galerkin=True, parallel=parallel, n_jobs=n_jobs
            )
        else:
            diag_vars = deflated_diagonal(
                self.covariance,
                diag_rank,
                diag_samples,
                galerkin=True,
                parallel=parallel,
                n_jobs=n_jobs,
            )

        safe_diag = np.where(np.abs(diag_vars) < 1e-14, 1.0, np.abs(diag_vars))
        scipy_op = self.covariance.matrix(galerkin=True)

        def _process_column(j: int):
            e_j = np.zeros(dim)
            e_j[j] = 1.0

            col_vals = scipy_op @ e_j
            correlations = np.abs(col_vals) / np.sqrt(safe_diag * safe_diag[j])
            valid_indices = np.where(correlations >= threshold)[0]

            if max_nnz is not None and len(valid_indices) > max_nnz:
                k = max_nnz - 1
                if k > 0:
                    corr_masked = correlations.copy()
                    corr_masked[j] = -1.0
                    top_k_indices = np.argpartition(corr_masked, -k)[-k:]
                    rows = np.append(top_k_indices, j)
                else:
                    rows = np.array([j])
            else:
                if j not in valid_indices:
                    rows = np.append(valid_indices, j)
                else:
                    rows = valid_indices

            vals = col_vals[rows]
            return rows.tolist(), [j] * len(rows), vals.tolist()

        if parallel:
            results = Parallel(n_jobs=n_jobs)(
                delayed(_process_column)(j) for j in range(dim)
            )
        else:
            results = [_process_column(j) for j in range(dim)]

        I_global, J_global, V_local = [], [], []
        for rows, cols, vals in results:
            I_global.extend(rows)
            J_global.extend(cols)
            V_local.extend(vals)

        sparse_cov_matrix = sps.coo_matrix(
            (V_local, (I_global, J_global)), shape=(dim, dim)
        ).tocsc()

        sparse_cov_matrix = 0.5 * (sparse_cov_matrix + sparse_cov_matrix.T)

        sparse_cov_op = SparseMatrixLinearOperator.from_matrix(
            self.covariance.domain,
            self.covariance.codomain,
            sparse_cov_matrix,
            galerkin=True,
        )

        try:
            max_eig = spla.eigsh(
                sparse_cov_matrix, k=1, which="LA", return_eigenvectors=False
            )[0]
        except spla.ArpackNoConvergence:
            max_eig = sparse_cov_matrix.diagonal().max()

        regularization_matrix = sps.eye(dim, format="csc") * (
            max_eig * regularization_fraction
        )
        regularized_cov_matrix = sparse_cov_matrix + regularization_matrix

        lu_factor = spla.splu(regularized_cov_matrix)

        def apply_precision(x: Vector) -> Vector:
            cx = self.covariance.codomain.to_dual(x)
            cx_comp = self.covariance.codomain.dual.to_components(cx)
            cy_comp = lu_factor.solve(cx_comp)
            return self.covariance.domain.from_components(cy_comp)

        sparse_inv_op = LinearOperator(
            self.covariance.codomain,
            self.covariance.domain,
            apply_precision,
            adjoint_mapping=apply_precision,
        )

        return GaussianMeasure(
            covariance=sparse_cov_op,
            inverse_covariance=sparse_inv_op,
            expectation=self._expectation,
        )

    def affine_mapping(
        self,
        /,
        *,
        operator: LinearOperator = None,
        translation: Vector = None,
        affine_operator: AffineOperator = None,
        inverse_solver: LinearSolver = None,
        inverse_preconditioner: LinearOperator = None,
    ) -> GaussianMeasure:
        """
        Transforms the measure under an affine map `y = A(x) + b`.

        This method calculates the push-forward measure. It can also construct
        the implied inverse covariance (precision) using a saddle-point (KKT) system.

        Args:
            operator: The linear part of the mapping (A).
            translation: The translation vector (b).
            affine_operator: An AffineOperator instance (cannot be used with
                `operator` or `translation`).
            inverse_solver: A solver used to evaluate the KKT inverse covariance.
            inverse_preconditioner: A preconditioner for the inverse_solver.

        Returns:
            A new GaussianMeasure representing the push-forward distribution.

        Raises:
            ValueError: If mutually exclusive arguments are provided, or if an
                inverse solve is requested but the prior lacks an inverse covariance.
        """
        if operator is None and affine_operator is None:
            if translation is None:
                return self

            new_expectation = (
                translation
                if self.has_zero_expectation
                else self.domain.add(self.expectation, translation)
            )

            if self.sample_set:

                def new_sample() -> Vector:
                    return self.domain.add(self.sample(), translation)

            else:
                new_sample = None

            return GaussianMeasure(
                covariance=self._covariance,
                covariance_factor=self._covariance_factor,
                expectation=new_expectation,
                sample=new_sample,
                inverse_covariance=self._inverse_covariance,
                inverse_covariance_factor=self._inverse_covariance_factor,
            )

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
            _translation = translation

        if self.has_zero_expectation:
            new_expectation = _translation
        else:
            mapped_exp = _operator(self.expectation)
            if _translation is None:
                new_expectation = mapped_exp
            else:
                new_expectation = _operator.codomain.add(mapped_exp, _translation)

        if self.covariance_factor_set:
            new_covariance_factor = _operator @ self.covariance_factor
            new_covariance = None
        else:
            new_covariance_factor = None
            new_covariance = _operator @ self.covariance @ _operator.adjoint

        new_inverse_covariance = None

        if inverse_solver is not None:
            if not self.inverse_covariance_set:
                raise ValueError(
                    "Cannot construct KKT inverse: the prior measure lacks an inverse covariance."
                )

            top_row = [self.inverse_covariance, _operator.adjoint]
            bottom_row = [_operator, _operator.codomain.zero_operator()]
            B = BlockLinearOperator([top_row, bottom_row])

            if inverse_preconditioner is None and isinstance(
                inverse_solver, IterativeLinearSolver
            ):
                schur_op = (
                    new_covariance
                    if new_covariance is not None
                    else (new_covariance_factor @ new_covariance_factor.adjoint)
                )

                bottom_right_block = self._build_jacobi_schur_block(_operator, schur_op)

                inverse_preconditioner = BlockDiagonalLinearOperator(
                    [self.covariance, bottom_right_block]
                )

            B_inv = inverse_solver(B, preconditioner=inverse_preconditioner)

            proj = B_inv.codomain.subspace_projection(1)
            incl = B_inv.domain.subspace_inclusion(1)

            new_inverse_covariance = -1.0 * (proj @ B_inv @ incl)

        if new_covariance_factor is not None:
            return GaussianMeasure(
                covariance_factor=new_covariance_factor,
                expectation=new_expectation,
                inverse_covariance=new_inverse_covariance,
            )
        else:

            def new_sample() -> Vector:
                mapped_sample = _operator(self.sample())
                if _translation is None:
                    return mapped_sample
                return _operator.codomain.add(mapped_sample, _translation)

            return GaussianMeasure(
                covariance=new_covariance,
                expectation=new_expectation,
                sample=new_sample if self.sample_set else None,
                inverse_covariance=new_inverse_covariance,
            )

    def as_multivariate_normal(
        self, /, *, parallel: bool = False, n_jobs: int = -1
    ) -> multivariate_normal:
        """
        Returns the measure as a `scipy.stats.multivariate_normal` object.

        Args:
            parallel: If True, evaluates the dense covariance matrix concurrently.
            n_jobs: The number of CPU cores to use if parallel=True.

        Returns:
            A frozen scipy.stats.multivariate_normal object.

        Raises:
            NotImplementedError: If the measure is not defined on a EuclideanSpace.
        """
        if not isinstance(self.domain, EuclideanSpace):
            raise NotImplementedError(
                "Method only defined for measures on Euclidean space."
            )

        mean_vector = self.expectation
        cov_matrix = self.covariance.matrix(
            dense=True, galerkin=True, parallel=parallel, n_jobs=n_jobs
        )

        try:
            return multivariate_normal(
                mean=mean_vector, cov=cov_matrix, allow_singular=True
            )
        except ValueError:
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

        Uses randomized matrix-free algorithms to factorize the covariance.

        Args:
            size_estimate: Target rank or initial sample size for the algorithm.
            method: 'variable' to sample dynamically, 'fixed' otherwise.
            max_rank: Upper limit on rank for the 'variable' method.
            power: Number of power iterations to enhance spectral decay.
            rtol: Relative tolerance for the 'variable' method.
            block_size: Samples drawn per iteration in the 'variable' method.
            parallel: If True, parallelizes the evaluations.
            n_jobs: Number of CPU cores to use if parallel=True.

        Returns:
            A new GaussianMeasure backed by a LowRankCholesky covariance factor.
        """
        from .low_rank import LowRankCholesky

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

        return GaussianMeasure(
            covariance_factor=lr_cholesky.l_factor,
            expectation=self._expectation,
        )

    def two_point_covariance(self, point: Any, /) -> Vector:
        """
        Computes the two-point covariance function radiating from a specific point.

        Args:
            point: The spatial coordinate to evaluate from.

        Returns:
            The covariance field evaluated at the chosen point.

        Raises:
            NotImplementedError: If the domain lacks a `dirac_representation` method.
        """
        if not hasattr(self.domain, "dirac_representation"):
            raise NotImplementedError(
                "Point evaluation is not defined for this measure's domain."
            )

        u = self.domain.dirac_representation(point)
        cov = self.covariance
        return cov(u)

    def directional_statistics(self, direction: Vector, /) -> Tuple[float, float]:
        """
        Returns the expectation and variance of the scalar Gaussian <x, direction>.

        Args:
            direction: The test vector.

        Returns:
            A tuple containing (expectation, variance).
        """
        expectation = (
            0.0
            if self.has_zero_expectation
            else self.domain.inner_product(self.expectation, direction)
        )
        variance = self.domain.inner_product(self.covariance(direction), direction)
        return expectation, variance

    def directional_covariance(self, d1: Vector, d2: Vector, /) -> float:
        """
        Returns the covariance between the scalar projections <x, d1> and <x, d2>.

        Args:
            d1: The first test vector.
            d2: The second test vector.

        Returns:
            The covariance scalar.
        """
        return self.domain.inner_product(self.covariance(d1), d2)

    def directional_variance(self, d: Vector, /) -> float:
        """
        Returns the variance of the scalar projection <x, d>.

        Args:
            d: The test vector.

        Returns:
            The variance scalar.
        """
        return self.directional_covariance(d, d)

    def zero_expectation(self) -> GaussianMeasure:
        """
        Returns a new measure with the same covariance, but zero expectation.

        Returns:
            A mean-shifted GaussianMeasure.
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
        self, direction: Vector, std: float, /
    ) -> GaussianMeasure:
        """
        Returns a new measure where Var[<x, direction>] is scaled to std^2.

        Args:
            direction: The test vector to scale against.
            std: The target standard deviation for the projection.

        Returns:
            A variance-scaled GaussianMeasure.

        Raises:
            ValueError: If the current directional variance is zero or negative.
        """
        current_var = self.directional_variance(direction)
        if current_var <= 0:
            raise ValueError("Directional variance must be positive to rescale.")
        norm = std / np.sqrt(current_var)
        shifted_measure = self.zero_expectation()
        scaled_measure = norm * shifted_measure
        return scaled_measure.affine_mapping(translation=self._expectation)

    def kl_divergence(
        self,
        other: GaussianMeasure,
        /,
        *,
        method: Literal["dense", "randomized"] = "dense",
        hutchinson_size_estimate: int = 10,
        hutchinson_method: Literal["variable", "fixed"] = "variable",
        max_samples: Optional[int] = None,
        rtol: float = 1e-2,
        block_size: int = 5,
        lanczos_size_estimate: int = 40,
        lanczos_method: Literal["variable", "fixed"] = "variable",
        lanczos_max_k: Optional[int] = None,
        lanczos_rtol: float = 1e-3,
        lanczos_atol: float = 1e-8,
        lanczos_check_interval: int = 5,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> float:
        """
        Computes the exact or approximate Kullback-Leibler (KL) divergence D_KL(self || other).

        This calculates the divergence of 'self' (P) from the prior/reference
        measure 'other' (Q).

        Args:
            other: The reference GaussianMeasure (Q).
            method: 'dense' uses exact dense matrix factorizations (O(N^3)).
                    'randomized' uses matrix-free Stochastic Lanczos Quadrature (SLQ).
            hutchinson_size_estimate: Initial samples for the randomized trace estimator.
            hutchinson_method: 'variable' to sample until `rtol` is met, 'fixed' otherwise.
            max_samples: Hard limit on Hutchinson samples.
            rtol: Relative tolerance for the Hutchinson estimator.
            block_size: Samples added per check in the 'variable' Hutchinson method.
            lanczos_size_estimate: Initial Krylov dimension for fractional evaluations.
            lanczos_method: 'variable' or 'fixed' convergence for Lanczos.
            lanczos_max_k: Maximum Krylov dimension if 'variable' is used.
            lanczos_rtol: Relative tolerance for Lanczos convergence.
            lanczos_atol: Absolute tolerance for Lanczos convergence.
            lanczos_check_interval: Iterations between Lanczos convergence checks.
            parallel: If True, evaluates the stochastic probes concurrently.
            n_jobs: Number of CPU cores to use if parallel=True.

        Returns:
            The calculated KL divergence.

        Raises:
            ValueError: If the measures reside on different domains, or if the 'randomized'
                        method is called without an inverse covariance on the reference measure.
        """
        if self.domain != other.domain:
            raise ValueError("Measures must be defined on the same domain.")

        if method == "dense":
            return self._kl_divergence_dense(other)
        elif method == "randomized":
            return self._kl_divergence_randomized(
                other,
                hutchinson_size_estimate=hutchinson_size_estimate,
                hutchinson_method=hutchinson_method,
                max_samples=max_samples,
                rtol=rtol,
                block_size=block_size,
                lanczos_size_estimate=lanczos_size_estimate,
                lanczos_method=lanczos_method,
                lanczos_max_k=lanczos_max_k,
                lanczos_rtol=lanczos_rtol,
                lanczos_atol=lanczos_atol,
                lanczos_check_interval=lanczos_check_interval,
                parallel=parallel,
                n_jobs=n_jobs,
            )
        else:
            raise ValueError("method must be 'dense' or 'randomized'.")

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
        """Adds two independent Gaussian measures defined on the same domain."""
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
        """Subtracts two independent Gaussian measures on the same domain."""
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

    def _sampling_radius(
        self,
        probability: float,
        gauge_squared,
        *,
        n_samples: int = 10_000,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> float:
        """Empirical p-quantile of R(X)^2 over Monte Carlo draws."""
        if not 0.0 < probability < 1.0:
            raise ValueError("probability must lie strictly between 0 and 1.")
        if not self.sample_set:
            raise ValueError(
                "Sampling-based radius requires a measure equipped with a "
                "sample method."
            )
        samples = self.samples(n_samples, parallel=parallel, n_jobs=n_jobs)
        mean = self.expectation
        gauge_sq_values = np.empty(n_samples, dtype=float)
        for i, x in enumerate(samples):
            diff = self.domain.subtract(x, mean)
            gauge_sq_values[i] = float(gauge_squared(diff))
        return float(np.quantile(gauge_sq_values, probability))

    def _sample_from_factor(self) -> Vector:
        """Default sampling method when a covariance factor is provided."""
        covariance_factor = self.covariance_factor
        w = covariance_factor.domain.random()
        value = covariance_factor(w)
        if self.has_zero_expectation:
            return value
        return self.domain.add(value, self.expectation)

    def _build_jacobi_schur_block(
        self, operator: LinearOperator, schur_op: LinearOperator
    ) -> LinearOperator:
        """Builds the inverse Jacobi (diagonal) approximation of the Schur complement."""
        try:
            n_data = operator.codomain.dim

            if n_data <= 200:
                schur_diag = schur_op.extract_diagonal(galerkin=True)
            else:
                from .low_rank import deflated_diagonal

                schur_diag = deflated_diagonal(
                    schur_op, 10, 40, method="fixed", galerkin=True
                )

            max_var = np.max(schur_diag)
            safe_diag = np.maximum(schur_diag, max_var * 1e-12)

            inv_safe_diag = 1.0 / safe_diag

            from .linear_operators import DiagonalSparseMatrixLinearOperator

            return DiagonalSparseMatrixLinearOperator.from_diagonal_values(
                operator.codomain, operator.codomain, inv_safe_diag, galerkin=True
            )
        except Exception:
            return operator.codomain.identity_operator()

    def _credible_set_chi2_ellipsoid(
        self,
        probability: float,
        *,
        rank: Optional[int],
        open_set: bool,
    ):
        """Builds the classical Mahalanobis ellipsoid."""
        effective_rank = self._resolve_rank(rank)
        radius = float(np.sqrt(chi2.ppf(probability, df=effective_rank)))
        from .subsets import Ellipsoid

        return Ellipsoid(
            self.domain,
            self.expectation,
            radius,
            self.inverse_covariance,
            open_set=open_set,
            inverse_operator=self.covariance,
        )

    def _credible_set_chi2_cameron_martin(
        self,
        probability: float,
        *,
        rank: Optional[int],
        open_set: bool,
    ):
        """Builds the unit ball in Cameron-Martin geometry."""
        effective_rank = self._resolve_rank(rank)
        radius = float(np.sqrt(chi2.ppf(probability, df=effective_rank)))
        from .subsets import Ball

        induced_domain_type = (
            MassWeightedHilbertModule
            if isinstance(self.domain, HilbertModule)
            else MassWeightedHilbertSpace
        )
        induced_domain = induced_domain_type(
            self.domain, self.inverse_covariance, self.covariance
        )
        return Ball(induced_domain, self.expectation, radius, open_set=open_set)

    def _resolve_rank(self, rank: Optional[int]) -> int:
        """Resolves the chi-square degrees of freedom."""
        if rank is None and self.domain.dim < 1:
            raise ValueError(
                "Rank must be provided for domains without a positive "
                "finite dimension, such as basis-free function spaces."
            )
        effective_rank = self.domain.dim if rank is None else rank
        if not isinstance(effective_rank, int) or effective_rank < 1:
            raise ValueError("Rank must be a positive integer.")
        return effective_rank

    def _resolve_spectrum(
        self,
        spectrum,
        spectrum_size: Optional[int],
        spectrum_low_rank_kwargs: Optional[dict],
        rng: Optional[np.random.Generator],
    ):
        """Resolves the covariance spectrum input into eigenvalues and objects."""
        from .low_rank import LowRankEig

        if spectrum is None:
            if spectrum_size is None or spectrum_size < 1:
                raise ValueError(
                    "spectrum_size (>=1) is required when spectrum is None."
                )
            kwargs = dict(spectrum_low_rank_kwargs or {})
            eig = LowRankEig.from_randomized(
                self.covariance,
                spectrum_size,
                measure=self,
                **kwargs,
            )
            return np.asarray(eig.eigenvalues, dtype=float), eig

        if isinstance(spectrum, LowRankEig):
            return np.asarray(spectrum.eigenvalues, dtype=float), spectrum

        if callable(spectrum):
            if spectrum_size is None or spectrum_size < 1:
                raise ValueError(
                    "spectrum_size (>=1) is required when spectrum is a callable."
                )
            eigvals = np.asarray(spectrum(spectrum_size), dtype=float).ravel()
            return eigvals, None

        eigvals = np.asarray(spectrum, dtype=float).ravel()
        if eigvals.ndim != 1:
            raise ValueError(f"spectrum array must be 1-D; got shape {eigvals.shape}.")
        return eigvals, None

    def _resolve_radius_method(self, radius_method: str, has_spectrum: bool) -> str:
        """Resolves the algorithm used to compute the credible radius."""
        if radius_method == "spectral":
            return "spectral"
        if radius_method == "sampling":
            if not self.sample_set:
                raise ValueError(
                    "radius_method='sampling' requires a sampling-equipped "
                    "Gaussian measure."
                )
            return "sampling"
        if radius_method == "auto":
            if has_spectrum:
                return "spectral"
            if self.sample_set:
                return "sampling"
            raise ValueError(
                "Cannot resolve a radius for this measure: pass a spectrum "
                "or set radius_method='sampling' on a sampling-equipped measure."
            )
        raise ValueError("radius_method must be 'auto', 'spectral', or 'sampling'.")

    def _credible_set_ambient_ball(
        self,
        probability: float,
        *,
        open_set: bool,
        spectrum,
        spectrum_size: Optional[int],
        radius_method: str,
        quantile_method: str,
        quantile_tol: float,
        n_samples: int,
        spectrum_low_rank_kwargs: Optional[dict],
        rng: Optional[np.random.Generator],
    ):
        """Builds the ambient norm ball using the specified radius method."""
        from .subsets import Ball
        from .quadratic_form_quantile import weighted_chi2_quantile

        method = self._resolve_radius_method(
            radius_method, has_spectrum=(spectrum is not None)
        )

        if method == "spectral":
            eigvals, _ = self._resolve_spectrum(
                spectrum, spectrum_size, spectrum_low_rank_kwargs, rng
            )
            r_p_sq = weighted_chi2_quantile(
                eigvals, probability, method=quantile_method, tol=quantile_tol
            )
        else:

            def gauge_squared(d):
                return float(self.domain.squared_norm(d))

            r_p_sq = self._sampling_radius(
                probability, gauge_squared, n_samples=n_samples
            )

        r_p = float(np.sqrt(max(r_p_sq, 0.0)))
        return Ball(self.domain, self.expectation, r_p, open_set=open_set)

    def _credible_set_weakened_ellipsoid(
        self,
        probability: float,
        *,
        theta: float,
        open_set: bool,
        spectrum,
        spectrum_size: Optional[int],
        radius_method: str,
        quantile_method: str,
        quantile_tol: float,
        fractional_apply: str,
        n_samples: int,
        lanczos_size_estimate: int,
        lanczos_method: str,
        lanczos_max_k: Optional[int],
        lanczos_rtol: float,
        lanczos_atol: float,
        lanczos_check_interval: int,
        spectrum_low_rank_kwargs: Optional[dict],
        rng: Optional[np.random.Generator],
    ):
        """Builds the weakened-covariance ellipsoid."""
        from .low_rank import LowRankEig
        from .quadratic_form_quantile import weighted_chi2_quantile
        from .subsets import Ellipsoid

        method = self._resolve_radius_method(
            radius_method, has_spectrum=(spectrum is not None)
        )

        if fractional_apply not in ("auto", "lanczos", "low_rank_eig"):
            raise ValueError(
                "fractional_apply must be 'auto', 'lanczos', or 'low_rank_eig'."
            )
        if fractional_apply == "auto":
            if isinstance(spectrum, LowRankEig) or spectrum is None:
                backend = "low_rank_eig"
            else:
                backend = "lanczos"
        else:
            backend = fractional_apply

        eigvals: Optional[np.ndarray] = None
        eig_obj: Optional[LowRankEig] = None
        if method == "spectral" or backend == "low_rank_eig":
            eigvals, eig_obj = self._resolve_spectrum(
                spectrum, spectrum_size, spectrum_low_rank_kwargs, rng
            )

        A, A_inv, A_inv_sqrt = self._build_fractional_operators(
            theta=theta,
            backend=backend,
            eig_obj=eig_obj,
            lanczos_size_estimate=lanczos_size_estimate,
            lanczos_method=lanczos_method,
            lanczos_max_k=lanczos_max_k,
            lanczos_rtol=lanczos_rtol,
            lanczos_atol=lanczos_atol,
            lanczos_check_interval=lanczos_check_interval,
        )

        if method == "spectral":
            assert eigvals is not None
            weights = np.power(eigvals, 1.0 - theta)
            self._maybe_warn_trace_borderline(weights, theta, eigvals.size)
            r_p_sq = weighted_chi2_quantile(
                weights, probability, method=quantile_method, tol=quantile_tol
            )
        else:

            def gauge_squared(d):
                return float(self.domain.inner_product(A(d), d))

            r_p_sq = self._sampling_radius(
                probability, gauge_squared, n_samples=n_samples
            )

        r_p = float(np.sqrt(max(r_p_sq, 0.0)))
        return Ellipsoid(
            self.domain,
            self.expectation,
            r_p,
            A,
            open_set=open_set,
            inverse_operator=A_inv,
            inverse_sqrt_operator=A_inv_sqrt,
        )

    def _build_fractional_operators(
        self,
        *,
        theta: float,
        backend: str,
        eig_obj: LowRankEig,
        lanczos_size_estimate: int,
        lanczos_method: str,
        lanczos_max_k: Optional[int],
        lanczos_rtol: float,
        lanczos_atol: float,
        lanczos_check_interval: int,
    ) -> Tuple[LinearOperator, LinearOperator, LinearOperator]:
        """Builds the fractional metric operators required by the weakened ellipsoid."""
        if backend == "low_rank_eig":
            if eig_obj is None:
                raise ValueError(
                    "fractional_apply='low_rank_eig' requires a "
                    "LowRankEig spectrum or spectrum=None (so it can be "
                    "computed); got an array/callable. Pass "
                    "fractional_apply='lanczos' instead."
                )
            regularization = 1e-12 * float(np.max(np.abs(eig_obj.eigenvalues)))

            return (
                eig_obj.apply_function(
                    lambda x: np.power(x, -theta), regularization=regularization
                ),
                eig_obj.apply_function(
                    lambda x: np.power(x, theta), regularization=regularization
                ),
                eig_obj.apply_function(
                    lambda x: np.power(x, 0.5 * theta), regularization=regularization
                ),
            )

        cov = self.covariance

        def make(power: float) -> LinearOperator:
            return LanczosOperatorFunction(
                cov,
                lambda x: np.power(x, power),
                size_estimate=lanczos_size_estimate,
                method=lanczos_method,
                max_k=lanczos_max_k,
                rtol=lanczos_rtol,
                atol=lanczos_atol,
                check_interval=lanczos_check_interval,
            )

        return make(-theta), make(theta), make(0.5 * theta)

    def _maybe_warn_trace_borderline(
        self,
        weights: np.ndarray,
        theta: float,
        n: int,
    ) -> None:
        """Warns when the (1-theta)-trace condition is numerically borderline."""
        if n < 20:
            return
        tail_share = float(np.sum(weights[n // 2 :])) / max(
            float(np.sum(weights)), 1e-300
        )
        if tail_share > 0.3:
            warnings.warn(
                f"The (1-theta) trace tail of the resolved spectrum is "
                f"{tail_share:.2f} of the total at theta={theta:.3f}; the "
                "trace-class condition may be numerically borderline. "
                "Increase spectrum_size or reduce theta to improve "
                "reliability.",
                UserWarning,
                stacklevel=3,
            )

    def _kl_divergence_dense(self, other: GaussianMeasure) -> float:
        """Original dense matrix implementation of the KL divergence."""
        k = self.domain.dim

        G_p = self.covariance.matrix(dense=True, galerkin=True)
        G_q = other.covariance.matrix(dense=True, galerkin=True)

        try:
            L_q = cholesky(G_q, lower=True)
        except np.linalg.LinAlgError:
            raise ValueError(
                "Covariance matrix of the 'other' measure is not positive definite."
            )

        X = cho_solve((L_q, True), G_p)
        trace_term = np.trace(X)

        log_det_q = 2.0 * np.sum(np.log(np.diag(L_q)))
        sign_p, log_det_p = np.linalg.slogdet(G_p)
        if sign_p <= 0:
            raise ValueError("Covariance matrix of 'self' is not positive definite.")
        log_det_term = log_det_q - log_det_p

        if self.has_zero_expectation and other.has_zero_expectation:
            mahalanobis_term = 0.0
        else:
            diff = self.domain.subtract(self.expectation, other.expectation)
            diff_dual = self.domain.to_dual(diff)
            diff_c_prime = self.domain.dual.to_components(diff_dual)
            y = cho_solve((L_q, True), diff_c_prime)
            mahalanobis_term = np.dot(diff_c_prime, y)

        return float(0.5 * (trace_term + mahalanobis_term - k + log_det_term))

    def _kl_divergence_randomized(
        self,
        other: GaussianMeasure,
        *,
        hutchinson_size_estimate: int,
        hutchinson_method: str,
        max_samples: Optional[int],
        rtol: float,
        block_size: int,
        lanczos_size_estimate: int,
        lanczos_method: str,
        lanczos_max_k: Optional[int],
        lanczos_rtol: float,
        lanczos_atol: float,
        lanczos_check_interval: int,
        parallel: bool,
        n_jobs: int,
    ) -> float:
        """Matrix-free SLQ / Hutchinson estimation of the KL divergence."""
        if not other.inverse_covariance_set:
            raise ValueError(
                "Randomized KL divergence requires the 'other' measure to "
                "have an inverse covariance operator set."
            )

        space = self.domain
        k = space.dim

        if self.has_zero_expectation and other.has_zero_expectation:
            mahalanobis_term = 0.0
        else:
            diff = space.subtract(self.expectation, other.expectation)
            q_inv_diff = other.inverse_covariance(diff)
            mahalanobis_term = space.inner_product(diff, q_inv_diff)

        from .functional_calculus import operator_function_quadratic_form

        if hasattr(space, "squared_norms"):
            inv_norms = 1.0 / np.sqrt(space.squared_norms)
        else:
            inv_norms = np.array(
                [
                    1.0 / np.sqrt(space.squared_norm(space.basis_vector(i)))
                    for i in range(space.dim)
                ]
            )

        def log_func(x: np.ndarray) -> np.ndarray:
            return np.log(np.maximum(x, 1e-15))

        def _single_sample() -> float:
            c_z = np.random.choice([-1.0, 1.0], size=space.dim) * inv_norms
            z = space.from_components(c_z)

            p_z = self.covariance(z)
            q_inv_p_z = other.inverse_covariance(p_z)
            trace_val = space.inner_product(z, q_inv_p_z)

            log_q = operator_function_quadratic_form(
                other.covariance,
                z,
                log_func,
                size_estimate=lanczos_size_estimate,
                method=lanczos_method,
                max_k=lanczos_max_k,
                rtol=lanczos_rtol,
                atol=lanczos_atol,
                check_interval=lanczos_check_interval,
            )

            log_p = operator_function_quadratic_form(
                self.covariance,
                z,
                log_func,
                size_estimate=lanczos_size_estimate,
                method=lanczos_method,
                max_k=lanczos_max_k,
                rtol=lanczos_rtol,
                atol=lanczos_atol,
                check_interval=lanczos_check_interval,
            )

            return trace_val + log_q - log_p

        if max_samples is None:
            max_samples = space.dim

        num_samples = min(hutchinson_size_estimate, max_samples)

        def _compute_sample_sum(n_samples_to_draw: int) -> float:
            if parallel:
                results = Parallel(n_jobs=n_jobs)(
                    delayed(_single_sample)() for _ in range(n_samples_to_draw)
                )
                return sum(results)
            else:
                return sum(_single_sample() for _ in range(n_samples_to_draw))

        total_sum = _compute_sample_sum(num_samples)
        stochastic_term = total_sum / num_samples if num_samples > 0 else 0.0

        if hutchinson_method == "variable" and num_samples < max_samples:
            while num_samples < max_samples:
                old_est = stochastic_term
                samples_to_add = min(block_size, max_samples - num_samples)

                total_sum += _compute_sample_sum(samples_to_add)
                total_samples = num_samples + samples_to_add
                stochastic_term = total_sum / total_samples

                if abs(stochastic_term) > 0:
                    error = abs(stochastic_term - old_est) / abs(stochastic_term)
                    if error < rtol:
                        break

                num_samples = total_samples

        return float(0.5 * (stochastic_term + mahalanobis_term - k))
