"""
Gaussian measures on Hilbert spaces.

A Gaussian is determined by an expectation and a covariance, and stays Gaussian
under an affine map. Both facts are carried structurally here: the pushforward
covariance ``A C A*`` is recognised as positive semidefinite by the
adjoint-palindrome rule of the algebra, with nothing asserted and no
special-casing (DESIGN.md 4.1).

Sampling is where the white-noise correction of DESIGN.md section 9 earns its
keep. Given a factor ``L`` with ``C == L L*``, a sample is ``m + L xi`` where
``xi`` is white noise **on the factor's own domain**. When that domain is the
space itself — the isotropic case, ``L == sigma I`` — the noise must be white
with respect to the space's inner product, not with respect to its components.
v1 draws standard normal components there and so produces covariance
``sigma^2 G`` rather than ``sigma^2 I`` on any mass-weighted space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence

import numpy as np
from numpy.random import Generator

from ..algebra.operators import LinearOperator, require_coordinates
from ..algebra.spaces import CoordinateSpace, EuclideanSpace, HilbertSpace
from ..traits import Traits

if TYPE_CHECKING:  # pragma: no cover
    from ..algebra.direct_sum import DirectSum
    from ..numerics.randomised import Estimate
    from ..geometry.convex import Ellipsoid
from .base import ProbabilityMeasure

__all__ = ["GaussianMeasure"]


_REQUIRED = Traits.SELF_ADJOINT | Traits.POSITIVE_SEMIDEFINITE


def _semidefinite_factors(
    symmetric: np.ndarray, /, *, rtol: float
) -> tuple[np.ndarray, np.ndarray | None]:
    """A square root of a symmetric PSD matrix, and its inverse when it exists.

    Cholesky first, because it is the cheaper factorisation and because
    succeeding is a proof that the matrix is numerically definite. When it
    fails the matrix is singular or has drifted slightly negative, and a
    symmetric eigendecomposition decides which: eigenvalues below
    ``-rtol * max|lambda|`` mean the caller's matrix is not a covariance, and
    smaller negative ones are floating-point noise and are clipped, with a
    warning, exactly as v1 did.

    Args:
        symmetric: the matrix, already symmetrised.
        rtol: how negative an eigenvalue may be, relative to the largest in
            magnitude, before the matrix is refused rather than clipped.

    Returns:
        ``(R, Rinv)`` with ``R R^T == symmetric``. ``Rinv`` is ``None`` when
        the matrix is singular, there being no inverse to report.

    Raises:
        ValueError: if the matrix has a significantly negative eigenvalue.
    """
    import warnings

    from scipy.linalg import solve_triangular

    dimension = symmetric.shape[0]
    try:
        root = np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError:
        pass
    else:
        return root, solve_triangular(root, np.eye(dimension), lower=True)

    eigenvalues, vectors = np.linalg.eigh(symmetric)
    largest = float(np.max(np.abs(eigenvalues))) if dimension else 0.0
    smallest = float(np.min(eigenvalues)) if dimension else 0.0
    if smallest < -rtol * largest:
        raise ValueError(
            f"The covariance matrix has an eigenvalue {smallest:.3e} against a "
            f"largest magnitude {largest:.3e}, so it is not positive "
            f"semidefinite in this representation."
        )
    if smallest < 0.0:
        warnings.warn(
            "The covariance matrix has small negative eigenvalues, which is "
            "what a positive semidefinite matrix assembled in floating point "
            "looks like. Clipping them to zero.",
            UserWarning,
            stacklevel=3,
        )
        eigenvalues = np.clip(eigenvalues, 0.0, None)

    deviations = np.sqrt(eigenvalues)
    root = vectors * deviations
    if dimension and np.min(deviations) > np.sqrt(rtol) * np.max(deviations):
        return root, (vectors / deviations).T
    return root, None


class GaussianMeasure[X](ProbabilityMeasure[X]):
    """A Gaussian measure, defined by any of covariance, factor or precision."""

    def __init__(
        self,
        domain: HilbertSpace[X],
        /,
        *,
        expectation: X | None = None,
        covariance: LinearOperator[X, X] | None = None,
        covariance_factor: LinearOperator | None = None,
        precision: LinearOperator[X, X] | None = None,
        precision_factor: LinearOperator | None = None,
        sample: Callable[[Generator | None], X] | None = None,
    ) -> None:
        """
        Args:
            domain: the space the measure lives on.
            expectation: the mean. Defaults to zero.
            covariance: a self-adjoint positive semidefinite operator. Derived
                from ``covariance_factor`` when not given.
            covariance_factor: ``L`` with ``C == L L*``. Supplying one is what
                makes the measure samplable.
            precision: the inverse covariance. Derived from
                ``precision_factor`` when not given.
            precision_factor: ``Li`` with ``C^-1 == Li* Li``.
            sample: an explicit sampler, for a measure whose factor is not
                available in closed form.
        """
        super().__init__(domain)

        if covariance is None and covariance_factor is not None:
            # L L* -- self-adjoint and positive semidefinite by the palindrome
            # rule, and positive definite when L is invertible. Nothing needs
            # to be claimed here.
            covariance = covariance_factor @ covariance_factor.adjoint

        if precision is None and precision_factor is not None:
            precision = precision_factor.adjoint @ precision_factor

        if covariance is None and precision is None:
            raise ValueError(
                "A Gaussian measure needs a covariance, a covariance factor, "
                "a precision or a precision factor."
            )

        for name, operator in (("covariance", covariance), ("precision", precision)):
            if operator is None:
                continue
            if operator.domain != domain or operator.codomain != domain:
                raise ValueError(f"The {name} must be an operator on {domain!r}.")
            missing = _REQUIRED & ~operator.traits
            if missing:
                raise ValueError(
                    f"The {name} must claim {_REQUIRED!s}; it claims "
                    f"{operator.traits!s} (missing {missing!s}). Attach the "
                    f"traits with with_traits() and verify them with "
                    f"testing.check_traits()."
                )

        if covariance_factor is not None and covariance_factor.codomain != domain:
            raise ValueError("The covariance factor must map into the domain.")

        self._expectation = expectation
        self._covariance = covariance
        self._covariance_factor = covariance_factor
        self._precision = precision
        self._precision_factor = precision_factor
        self._sample_fn = sample
        self._log_normalisation: float | None = None

    # ----------------------------------------------------------------- #
    #                            Constructors                           #
    # ----------------------------------------------------------------- #

    @classmethod
    def from_standard_deviation(
        cls,
        domain: HilbertSpace[X],
        standard_deviation: float,
        /,
        *,
        expectation: X | None = None,
    ) -> GaussianMeasure[X]:
        """An isotropic measure with covariance ``sigma^2 I`` **on the space**.

        The factor's domain is the space itself, so the sample draws white
        noise with respect to the space's inner product. This is the
        construction v1 gets wrong on a mass-weighted space.

        Args:
            domain: the space.
            standard_deviation: one number, this being an isotropic measure.
            expectation: the mean. Zero if omitted.

        Returns:
            The measure, carrying both a factor and a precision factor.

        Raises:
            ValueError: for a non-positive or non-scalar deviation. Use
                from_standard_deviations for one per component.
        """
        if np.ndim(standard_deviation) != 0:
            raise ValueError(
                "from_standard_deviation takes one number, for an isotropic "
                "measure. For a different deviation per component, use "
                "from_standard_deviations."
            )
        if standard_deviation <= 0.0:
            raise ValueError("standard_deviation must be positive.")
        factor = standard_deviation * LinearOperator.identity(domain)
        inverse = (1.0 / standard_deviation) * LinearOperator.identity(domain)
        return cls(
            domain,
            expectation=expectation,
            covariance_factor=factor,
            precision_factor=inverse,
        )

    @classmethod
    def from_standard_deviations(
        cls,
        domain: HilbertSpace[X],
        standard_deviations: np.ndarray,
        /,
        *,
        expectation: X | None = None,
    ) -> GaussianMeasure[X]:
        """A measure with a different standard deviation in each direction.

        The covariance is ``diag(sigma^2)`` **as an operator on the space**,
        which is the generalisation of :meth:`from_standard_deviation` and not
        of v1's method of this name: v1 built its factor as a map from a
        Euclidean coefficient space, making the array a statement about
        components rather than about directions. The two agree on an
        orthonormal space and differ on every other, and the operator reading
        is the one that draws white noise correctly — the defect DESIGN.md
        section 9 exists to record.

        Both a factor and a precision factor are supplied, so the result can be
        sampled and has a density.

        Args:
            domain: the space the measure lives on.
            standard_deviations: one positive number per dimension.
            expectation: the mean. Defaults to zero.

        Returns:
            The measure with that diagonal covariance.

        Raises:
            ValueError: if the array is the wrong length or has an entry that
                is not positive.
        """
        from ..algebra.diagonal import DiagonalLinearOperator

        values = np.asarray(standard_deviations, dtype=float)
        if values.shape != (domain.dim,):
            raise ValueError(
                f"Expected {domain.dim} standard deviations for {domain!r}, "
                f"got an array of shape {values.shape}."
            )
        if np.any(values <= 0.0):
            raise ValueError("Every standard deviation must be positive.")
        return cls(
            domain,
            expectation=expectation,
            covariance_factor=DiagonalLinearOperator(domain, values),
            precision_factor=DiagonalLinearOperator(domain, 1.0 / values),
        )

    @classmethod
    def from_samples(
        cls, domain: HilbertSpace[X], samples: Sequence[X], /
    ) -> GaussianMeasure[X]:
        """The empirical mean and covariance of a set of vectors.

        Coordinate-free: the covariance is a sum of outer products, and its
        factor maps a Euclidean coefficient vector onto the deviations.

        Args:
            domain: the space.
            samples: the draws to estimate from.

        Returns:
            The measure, whose covariance has rank at most one less than the
            number of samples -- a fact about the estimate, not a defect.

        Raises:
            ValueError: for fewer than two samples, there being no covariance
                of one.
        """
        n = len(samples)
        if n < 2:
            raise ValueError("At least two samples are needed for a covariance.")
        mean = domain.mean(samples)
        deviations = [domain.subtract(x, mean) for x in samples]
        scale = 1.0 / np.sqrt(n - 1)

        coefficients = EuclideanSpace(n)

        def factor_value(c: np.ndarray) -> X:
            result = domain.zero()
            for weight, deviation in zip(c, deviations):
                result = domain.axpy(scale * float(weight), deviation, result)
            return result

        def factor_adjoint(x: X) -> np.ndarray:
            return scale * np.array([domain.inner_product(d, x) for d in deviations])

        factor = LinearOperator.from_callables(
            coefficients, domain, factor_value, adjoint=factor_adjoint
        )
        return cls(domain, expectation=mean, covariance_factor=factor)

    @classmethod
    def from_product(
        cls,
        measures: Sequence[GaussianMeasure],
        /,
        *,
        labels: Sequence[str] | None = None,
    ) -> GaussianMeasure:
        """The independent product of Gaussians, on the direct sum of domains.

        The covariance is block diagonal, and the block-diagonal operator gives
        it the right traits by intersecting the blocks': the whole is positive
        definite exactly when every factor is. The precision is block diagonal
        for the same reason, and is built whenever every factor has one.

        That matters beyond the density. The surrogate prior a Woodbury data
        form is built from is a product of invariant measures, each with a
        diagonal precision; without a precision on the product the
        preconditioner falls back to inverting ``Q`` by conjugate gradients on
        every application, which is a solve nested inside a preconditioner
        inside a solve.

        Args:
            measures: the independent factors.
            labels: names for the summands of the direct sum.

        Returns:
            The product measure, on the direct sum of the domains.

        Raises:
            ValueError: if no factors are given, or if some factor has neither
                a covariance nor a covariance factor.
        """
        from ..algebra.direct_sum import BlockDiagonalLinearOperator, DirectSum

        measures = tuple(measures)
        if not measures:
            raise ValueError("A product measure needs at least one factor.")
        domain = DirectSum([m.domain for m in measures], labels=labels)

        covariance = None
        factor = None
        if all(m.covariance_factor is not None for m in measures):
            factor = BlockDiagonalLinearOperator(
                [m.covariance_factor for m in measures]
            )
        elif all(m.covariance is not None for m in measures):
            covariance = BlockDiagonalLinearOperator([m.covariance for m in measures])
        else:
            raise ValueError("Every factor needs a covariance or a covariance factor.")

        precision = None
        precision_factor = None
        if all(m.precision is not None for m in measures):
            precision = BlockDiagonalLinearOperator([m.precision for m in measures])
        elif all(m.precision_factor is not None for m in measures):
            precision_factor = BlockDiagonalLinearOperator(
                [m.precision_factor for m in measures]
            )

        sample = None
        if factor is None and all(m.can_sample for m in measures):

            def sample(rng, _measures=measures):
                return tuple(m.sample(rng=rng) for m in _measures)

        return cls(
            domain,
            expectation=tuple(m.expectation for m in measures),
            covariance=covariance,
            covariance_factor=factor,
            precision=precision,
            precision_factor=precision_factor,
            sample=sample,
        )

    @classmethod
    def from_covariance_matrix(
        cls,
        domain: CoordinateSpace[X],
        matrix: np.ndarray,
        /,
        *,
        form: Literal["galerkin", "components"] = "galerkin",
        expectation: X | None = None,
        rtol: float = 1e-10,
    ) -> GaussianMeasure[X]:
        """From an explicit covariance matrix.

        ``form`` says which representation the array is in, because no trait
        implies it (DESIGN.md 5.3). The Galerkin form is the natural one here:
        a covariance is self-adjoint, so that is the representation in which it
        is symmetric.

        A covariance is required to be positive *semi*definite, so a Cholesky
        factorisation alone is not enough: it refuses every singular
        covariance — a measure supported on a subspace, an empirical
        covariance from fewer samples than dimensions, a pushforward through a
        rank-deficient map — and it refuses a matrix that is semidefinite in
        exact arithmetic but has eigenvalues of size ``-1e-17`` after being
        assembled in floating point. v1 took a symmetric eigendecomposition,
        clipped small negative eigenvalues to zero with a warning, and built
        both factors from it; this does the same. A strictly definite matrix
        still takes the Cholesky route, which is the cheaper one and which
        proves definiteness by succeeding.

        The measure carries a precision factor whenever the covariance is
        nonsingular, so that it has a density — v1 attached one and v2 had
        stopped doing so, leaving :meth:`mahalanobis_squared`,
        :meth:`log_density` and :meth:`grad_log_density` refusing on every
        measure built this way. A *singular* covariance gets none: the measure
        is degenerate, its density with respect to the space's own volume
        measure does not exist, and a pseudo-inverse in the precision slot
        would answer those three methods with a finite number that is not the
        thing they name.

        Cost is cubic in ``domain.dim`` and the result holds two dense
        matrices (three when a precision is attached, the Gram matrix being
        the third).

        Args:
            domain: the space.
            matrix: the covariance, as an array.
            form: which representation *matrix* is in. No default: guessing
                wrong is a silent error of one factor of the Gram matrix.
            expectation: the mean. Zero if omitted.
            rtol: how negative an eigenvalue may be, relative to the largest in
                magnitude, before the matrix is refused rather than clipped.

        Returns:
            The measure.

        Raises:
            ValueError: for an unknown form, a wrongly shaped matrix, or one
                not symmetric positive semidefinite in that representation.
        """
        require_coordinates(domain)
        matrix = np.asarray(matrix, dtype=float)
        expected = (domain.dim, domain.dim)
        if matrix.shape != expected:
            raise ValueError(f"Matrix has shape {matrix.shape}, expected {expected}.")

        if form == "components":
            # The Galerkin matrix of a covariance is ``G C_c``. Build it a
            # column at a time: ``apply_gram`` takes a component vector, and
            # handing it a matrix relies on a broadcasting coincidence that
            # holds only for a diagonal metric (it gives ``C_c G`` otherwise).
            matrix = np.column_stack([domain.apply_gram(col) for col in matrix.T])
        elif form != "galerkin":
            raise ValueError(f"Unknown form {form!r}.")

        symmetric = 0.5 * (matrix + matrix.T)
        root, inverse_root = _semidefinite_factors(symmetric, rtol=rtol)

        # from_matrix(E, X, R, form="galerkin") has component matrix G^-1 R, and
        # (G^-1 R)(G^-1 R)* has Galerkin matrix R R^T == the covariance.
        coefficients = EuclideanSpace(domain.dim)
        factor = LinearOperator.from_matrix(
            coefficients, domain, root, form="galerkin"
        )

        precision = None
        precision_factor = None
        if inverse_root is not None:
            # Li: X -> E with component matrix M has (Li* Li)_c == G^-1 M^T M,
            # and the precision's component matrix is C_c^-1 == C_gal^-1 G. So
            # M^T M must be G C_gal^-1 G, which M == R^-1 G delivers. The Gram
            # matrix appears twice because the precision is a form on the
            # space, not on its components; dropping it is right only for an
            # orthonormal basis, which is v1's version of this line.
            precision_factor = LinearOperator.from_matrix(
                domain,
                coefficients,
                inverse_root @ domain.gram_matrix(),
                form="components",
            )
            # The palindrome rule gives Li* Li only semidefiniteness, but the
            # factorisation earned more than that: it returned an inverse
            # root, which it does only for a covariance that is numerically
            # nonsingular. Definiteness is what a credible set asks for.
            precision = (precision_factor.adjoint @ precision_factor).with_traits(
                Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
            )
        return cls(
            domain,
            expectation=expectation,
            covariance_factor=factor,
            precision=precision,
            precision_factor=precision_factor,
        )

    # ----------------------------------------------------------------- #
    #                              Moments                              #
    # ----------------------------------------------------------------- #

    @property
    def expectation(self) -> X:
        """The mean, which is the zero vector when none was supplied."""
        if self._expectation is None:
            return self._domain.zero()
        return self._expectation

    @property
    def has_zero_expectation(self) -> bool:
        """True when no expectation was supplied, so the mean is zero."""
        return self._expectation is None

    @property
    def covariance(self) -> LinearOperator[X, X] | None:
        """The covariance operator, or None when only a precision is known."""
        return self._covariance

    @property
    def covariance_factor(self) -> LinearOperator | None:
        """The factor ``L`` with ``C == L L*``, or None."""
        return self._covariance_factor

    @property
    def precision(self) -> LinearOperator[X, X] | None:
        """The precision, or None when it was not supplied or derived."""
        return self._precision

    @property
    def precision_factor(self) -> LinearOperator | None:
        """The factor ``Li`` with ``C^-1 == Li* Li``, or None."""
        return self._precision_factor

    # ----------------------------------------------------------------- #
    #                             Sampling                              #
    # ----------------------------------------------------------------- #

    @property
    def can_sample(self) -> bool:
        """True when the measure can be sampled.

        A covariance alone is not enough: sampling needs a factor of it, or an
        explicit sampler.
        """
        return self._sample_fn is not None or self._covariance_factor is not None

    def sample(self, *, rng: Generator | None = None) -> X:
        """One draw, as ``m + L xi`` with ``xi`` white noise on the factor's domain.

        Args:
            rng: the generator.

        Returns:
            A vector of the domain.

        Raises:
            NotImplementedError: if the measure has a covariance but no
                factor and no explicit sampler. A covariance says what the
                spread is; drawing from it needs a square root.
        """
        if self._sample_fn is not None:
            value = self._sample_fn(rng)
        elif self._covariance_factor is not None:
            noise = self._covariance_factor.domain.white_noise(rng=rng)
            value = self._covariance_factor(noise)
        else:
            raise NotImplementedError(
                "This measure has a covariance but no factor, so it cannot be "
                "sampled. Supply covariance_factor, or an explicit sample "
                "callable."
            )
        if self._expectation is None:
            return value
        return self._domain.add(value, self._expectation)

    # ----------------------------------------------------------------- #
    #                             Densities                             #
    # ----------------------------------------------------------------- #

    def _deviation(self, x: X) -> X:
        if self._expectation is None:
            return x
        return self._domain.subtract(x, self._expectation)

    def _diagonal_eigenvalues(self) -> np.ndarray | None:
        """The covariance's spectrum, when it is diagonal in the space's basis."""
        from ..algebra.diagonal import DiagonalLinearOperator

        covariance = self._covariance
        if isinstance(covariance, DiagonalLinearOperator):
            return covariance.eigenvalues
        return None

    def _stochastic_trace(
        self,
        operator: LinearOperator,
        /,
        *,
        samples: int,
        rtol: float | None,
        rng: Generator | None,
        n_jobs: int | None,
    ) -> float:
        """A Hutchinson trace, matrix-free.

        ``random_trace`` draws its probes as white noise *on the space*, so the
        expectation is the trace of the operator -- the component matrix's
        trace -- and not ``tr(G A)``, which is what probes with standard normal
        components would give on any space whose Gram matrix is not the
        identity.

        Args:
            operator: the endomorphism whose trace is wanted.
            samples: how many probes, or the first block when *rtol* is given.
            rtol: stop when the standard error falls to this fraction of the
                estimate, instead of at a fixed count.
            rng: the generator.
            n_jobs: workers for the probes.

        Returns:
            The estimated trace.
        """
        from ..numerics.randomised import random_trace

        return float(
            random_trace(
                operator, samples=samples, rtol=rtol, rng=rng, n_jobs=n_jobs
            ).value
        )

    def hilbert_schmidt_norm(
        self,
        /,
        *,
        method: str = "auto",
        samples: int = 100,
        rtol: float | None = None,
        rng: Generator | None = None,
        n_jobs: int | None = None,
    ) -> float:
        """The Hilbert-Schmidt norm of the covariance, ``sqrt(tr(C* C))``.

        ``"stochastic"`` is a Hutchinson estimate of ``tr(C C)``, which is what
        v1 did and what this docstring has always claimed; it used to form the
        dense component matrix and return the exact answer, quietly, which is
        the opposite of the promise and impossible at the sizes the option
        exists for. The estimator is a trace of ``C^2``, so its relative error
        is worse than a trace of ``C``: ask for more probes here than for
        :meth:`nuclear_norm`, or pass *rtol* and let it decide.

        Args:
            method: ``"dense"`` forms the component matrix, ``"stochastic"``
                estimates the trace with :func:`random_trace`, ``"diagonal"``
                reads the spectrum of a diagonal covariance, and ``"auto"``
                takes the diagonal route when it can and the dense one
                otherwise. ``"auto"`` is always exact; a sampled norm has to be
                asked for by name.
            samples: probes for the stochastic route.
            rtol: draw further blocks of probes until the standard error is
                this fraction of the estimate.
            rng: the generator for the probes.
            n_jobs: workers for the probes.

        Returns:
            The norm.

        Raises:
            ValueError: for an unknown method, or a measure with no covariance.
        """
        eigenvalues = self._diagonal_eigenvalues()
        if eigenvalues is not None and method in ("auto", "diagonal"):
            return float(np.sqrt(np.sum(eigenvalues**2)))
        covariance = self._require_covariance("A Hilbert-Schmidt norm")
        if method == "stochastic":
            # tr(C* C) == tr(C C), the covariance being self-adjoint. The max
            # is against an estimate that has come out slightly negative on a
            # near-singular covariance, where the truth is a very small number.
            squared = self._stochastic_trace(
                covariance @ covariance,
                samples=samples,
                rtol=rtol,
                rng=rng,
                n_jobs=n_jobs,
            )
            return float(np.sqrt(max(squared, 0.0)))
        if method not in ("auto", "dense", "diagonal"):
            raise ValueError(f"Unknown method {method!r}.")
        # tr(C* C) is basis-independent, so it comes from the *component*
        # matrix. The Galerkin one is G C_c, whose trace is a different number.
        matrix = covariance.matrix(form="components")
        return float(np.sqrt(np.sum(matrix * matrix.T)))

    def nuclear_norm(
        self,
        /,
        *,
        method: str = "auto",
        samples: int = 100,
        rtol: float | None = None,
        rng: Generator | None = None,
        n_jobs: int | None = None,
    ) -> float:
        """The trace norm of the covariance, ``tr|C|``.

        For a covariance this is the trace, since it is positive semidefinite —
        the total variance of the measure.

        Args:
            method: as for :meth:`hilbert_schmidt_norm`.
            samples: probes for the stochastic route.
            rtol: target relative standard error for the stochastic route.
            rng: the generator for the probes.
            n_jobs: workers for the probes.

        Returns:
            The norm.

        Raises:
            ValueError: for an unknown method, or a measure with no covariance.
        """
        eigenvalues = self._diagonal_eigenvalues()
        if eigenvalues is not None and method in ("auto", "diagonal"):
            return float(np.sum(np.abs(eigenvalues)))
        covariance = self._require_covariance("A nuclear norm")
        if method == "stochastic":
            return self._stochastic_trace(
                covariance, samples=samples, rtol=rtol, rng=rng, n_jobs=n_jobs
            )
        if method not in ("auto", "dense", "diagonal"):
            raise ValueError(f"Unknown method {method!r}.")
        # A covariance is positive semidefinite, so its trace norm is its
        # trace -- and a trace is the component matrix's, not the Galerkin
        # matrix's, which carries an extra factor of the metric.
        return float(np.trace(covariance.matrix(form="components")))

    def _weighted_squared(self, vector: X, /) -> float:
        """``(C^-1 v, v)``, from the precision if there is one, else densely.

        In Galerkin form the quadratic is ``g^T C_gal^-1 g`` with
        ``g == G c_v`` — the metric appears on both sides because the operator
        being inverted is ``G C_c``, not ``C_c``.
        """
        if self._precision is not None:
            return float(self._domain.inner_product(self._precision(vector), vector))
        components = self._domain.apply_gram(self._domain.to_components(vector))
        return float(components @ np.linalg.solve(self._symmetric_matrix(), components))

    def _require_covariance(self, what: str, /) -> LinearOperator:
        """The covariance, or a message saying why there is not one.

        ``covariance is None`` is a legal state — a measure may be described by
        its precision alone, which is the natural description of a posterior —
        but almost nothing can be done with it, and what came back was a
        ``TypeError`` from an arithmetic operation on ``None`` several frames
        down. This says which method wanted what, and what to do about it.

        Args:
            what: the operation, named for the message.

        Returns:
            The covariance operator.

        Raises:
            ValueError: when the measure has only a precision.
        """
        if self._covariance is None:
            raise ValueError(
                f"{what} needs the covariance, and this measure was given only "
                f"a precision. Supply a covariance alongside it, or invert the "
                f"precision explicitly — the library will not do it silently, "
                f"because that is a solve per application."
            )
        return self._covariance

    def _symmetric_matrix(self) -> np.ndarray:
        """The covariance's Galerkin matrix, symmetrised against round-off."""
        matrix = self._require_covariance("This route").matrix(form="galerkin")
        return 0.5 * (matrix + matrix.T)

    def kl_divergence(
        self,
        other: "GaussianMeasure",
        /,
        *,
        method: str = "auto",
        solver: Any = None,
        samples: int = 100,
        rng: Generator | None = None,
        dense_limit: int = 512,
        **kwargs: Any,
    ) -> float:
        """``D(self || other)`` between two Gaussians on the same space.

        The value alone. :meth:`kl_divergence_estimate` returns it with the
        uncertainty, which is the thing to use when the route is stochastic.

        Args:
            other: the measure to compare against.
            method: ``"dense"``, ``"spectral"``, ``"stochastic"``, or
                ``"auto"``. ``"auto"`` refuses rather than going stochastic
                silently -- an estimate reported as a bare number is the one
                way this quantity misleads.
            solver: how to invert the other's covariance, where needed.
            samples: probes for the stochastic route.
            dense_limit: the dimension below which the dense route is taken.

        Returns:
            The divergence.
        """
        return self.kl_divergence_estimate(
            other,
            method=method,
            solver=solver,
            samples=samples,
            rng=rng,
            dense_limit=dense_limit,
            **kwargs,
        ).value

    def kl_divergence_estimate(
        self,
        other: "GaussianMeasure",
        /,
        *,
        method: str = "auto",
        solver: Any = None,
        samples: int = 100,
        rng: Generator | None = None,
        dense_limit: int = 512,
        **kwargs: Any,
    ) -> "Estimate":
        r"""``D(self || other)``, with the uncertainty of however it was got.

        .. code-block:: text

            2 D == tr(C_o^-1 C_s) + (m_o - m_s, C_o^-1 (m_o - m_s))
                   - dim + log det C_o - log det C_s

        Three routes, and they differ only in how the trace and the two
        determinants are reached.

        ``"spectral"``
            Both covariances diagonal in the space's own basis, so everything
            is a sum over the spectrum: ``O(dim)``, exact, and no factorisation
            at all. Every invariant measure on a symmetric space is of this
            kind, so it is the common case and the one ``"auto"`` takes first.
        ``"dense"``
            Form both matrices and factorise. Exact, and confined to a space
            small enough to hold two of them.
        ``"stochastic"``
            Hutchinson for the trace and stochastic Lanczos for the
            determinants, so nothing is ever formed. The route for a space too
            large to assemble, at the cost of an answer with an error bar.

            **It has to be asked for by name.** On the ill-conditioned spectra
            this library produces it is currently unreliable — measured at
            -88.6 +/- 21.7 for a divergence of zero on a correlated measure of
            dimension 578 — so ``"auto"`` raises rather than reaching for it.

        Args:
            other: the reference measure. The divergence is not symmetric.
            method: ``"auto"``, ``"spectral"``, ``"dense"`` or ``"stochastic"``.
                ``"auto"`` takes the spectral route when both covariances are
                diagonal, then the dense one when the space has coordinates and
                is no larger than *dense_limit*, and raises otherwise rather
                than reaching for the stochastic route unasked.
            solver: how to invert the reference covariance, for the stochastic
                route. Conjugate gradients by default; a factory taking the
                operator is also accepted.
            samples: Hutchinson probes for the trace and each determinant.
            rng: the generator for those probes.
            dense_limit: the dimension above which ``"auto"`` stops forming
                matrices.
            kwargs: passed to
                :func:`~pygeoinf2.numerics.functional_calculus.log_determinant`.

        Returns:
            An :class:`~pygeoinf2.numerics.randomised.Estimate`. The exact
            routes report a standard error of zero.

        Raises:
            ValueError: for an unknown method; if the two measures live on
                different spaces; or, under ``"auto"``, when only the
                stochastic route is available -- which is refused rather than
                taken silently.
        """
        from ..numerics.functional_calculus import log_determinant
        from ..numerics.randomised import Estimate, random_trace
        from ..numerics.solvers import resolve_solver

        if other.domain != self._domain:
            raise ValueError("Both measures must live on the same space.")
        if method not in ("auto", "spectral", "dense", "stochastic"):
            raise ValueError(
                f"The method is 'auto', 'spectral', 'dense' or 'stochastic', "
                f"got {method!r}."
            )
        dimension = self._domain.dim
        shift = self._domain.subtract(other.expectation, self.expectation)

        mine = self._diagonal_eigenvalues()
        theirs = other._diagonal_eigenvalues()
        spectral = mine is not None and theirs is not None
        if method == "spectral" and not spectral:
            raise ValueError(
                "The spectral route needs both covariances diagonal in the "
                "space's own basis, and at least one of these is not."
            )
        if method == "auto":
            if spectral:
                method = "spectral"
            elif isinstance(self._domain, CoordinateSpace) and dimension <= dense_limit:
                method = "dense"
            else:
                raise ValueError(
                    "Neither exact route applies here, and the stochastic one "
                    "is not accurate enough to be chosen on your behalf: on a "
                    "Sobolev spectrum it has returned -88.6 +/- 21.7 for a "
                    "divergence whose true value is zero. Ask for it by name "
                    "with method='stochastic' if an estimate with that error "
                    "bar is what you want; raise dense_limit if the space can "
                    "afford two dense matrices; or give both measures diagonal "
                    "covariances, which is the exact O(dim) route."
                )

        if method == "spectral":
            if np.any(theirs <= 0.0):
                raise ValueError(
                    "The reference measure is singular, so the divergence from "
                    "it is infinite."
                )
            # From the spectrum, not from other._weighted_squared: that falls
            # back to a dense solve whenever the reference has no precision,
            # so the "exact O(dim) route" quietly became an O(n^3) one. The
            # eigenvalues are already in hand, and the operator is diagonal in
            # the space's own basis, so (C^-1 v, v) == (c/lambda) . (G c).
            components = self._domain.to_components(shift)
            quadratic = float(
                np.dot(components / theirs, self._domain.apply_gram(components))
            )
            trace = float(np.sum(mine / theirs))
            logs = float(
                np.sum(np.log(theirs)) - np.sum(np.log(np.clip(mine, 1e-300, None)))
            )
            return Estimate(0.5 * (trace + quadratic - dimension + logs), 0.0, 0)

        if method == "dense":
            mine_matrix = self._symmetric_matrix()
            theirs_matrix = other._symmetric_matrix()
            sign_mine, log_mine = np.linalg.slogdet(mine_matrix)
            sign_theirs, log_theirs = np.linalg.slogdet(theirs_matrix)
            if sign_mine <= 0 or sign_theirs <= 0:
                raise ValueError("Both covariances must be positive definite.")
            trace = float(np.trace(np.linalg.solve(theirs_matrix, mine_matrix)))
            quadratic = other._weighted_squared(shift)
            return Estimate(
                0.5 * (trace + quadratic - dimension + log_theirs - log_mine),
                0.0,
                0,
            )

        # Stochastic: nothing is formed. The trace operator C_o^-1 C_s is not
        # self-adjoint, which costs nothing here -- Hutchinson estimates the
        # trace of any endomorphism, self-adjoint or not.
        definite = Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
        if self._covariance is None or other._covariance is None:
            raise ValueError(
                "The stochastic route needs both covariances as operators, and "
                "at least one of these measures was given only a precision."
            )
        reference = other._covariance.with_traits(definite)
        inverse = (
            other._precision
            if other._precision is not None
            else resolve_solver(solver, reference)(reference)
        )
        trace = random_trace(inverse @ self._covariance, samples=samples, rng=rng)
        quadratic = self._domain.inner_product(inverse(shift), shift)
        log_theirs = log_determinant(
            reference, method="stochastic", samples=samples, rng=rng, **kwargs
        )
        log_mine = log_determinant(
            self._covariance.with_traits(definite),
            method="stochastic",
            samples=samples,
            rng=rng,
            **kwargs,
        )
        value = 0.5 * (
            trace.value + quadratic - dimension + log_theirs.value - log_mine.value
        )
        # Three independent estimates, so the errors add in quadrature and the
        # halving outside carries through.
        error = 0.5 * float(
            np.sqrt(
                trace.standard_error**2
                + log_theirs.standard_error**2
                + log_mine.standard_error**2
            )
        )
        return Estimate(float(value), error, samples)

    def rescale_directional_variance(
        self, direction: X, standard_deviation: float, /
    ) -> "GaussianMeasure[X]":
        """The same measure with ``Var((x, direction))`` set to a given value.

        A whole-measure rescaling, as v1 does it, not a rank-one update: the
        covariance is multiplied by a scalar chosen so that one direction comes
        out right. Every other variance moves by the same factor, which is what
        makes it a recalibration rather than a change of shape.

        Returns:
            The rescaled measure, of the same class.

        Raises:
            ValueError: for a non-positive target variance, or a direction in
                which this measure has none -- there being no factor that
                would give it one.
        """
        current = self.directional_variance(direction)
        if current <= 0.0:
            raise ValueError(
                "The variance along this direction is not positive, so it "
                "cannot be rescaled."
            )
        factor = float(standard_deviation) / np.sqrt(current)
        centred = self.translate(self._domain.negative(self.expectation))
        return (factor * centred).translate(self.expectation)

    def low_rank_approximation(
        self,
        /,
        *,
        rank: int | None = None,
        rng: Generator | None = None,
        **kwargs: Any,
    ) -> "GaussianMeasure[X]":
        """The same expectation, with the covariance truncated to *rank* modes.

        Obtained as a randomised Cholesky factorisation ``C ~ L L*``, so the
        result comes with a covariance *factor* and is therefore samplable even
        when the original was not. Its precision does not survive — a
        rank-deficient covariance has none — which means a low-rank measure can
        stand in for a prior in the data-space formalism and not in the
        model-space one.

        The point is to be cheap rather than to be accurate: this is how a
        surrogate is built when no cheaper physics is available. See
        :meth:`~pygeoinf2.inference.gaussian.LinearGaussianInversion.low_rank_surrogate`.

        Args:
            rank: how many eigenpairs to keep.
            rng: the generator for the probes.
            **kwargs: passed to the randomised routine.

        Returns:
            A measure whose covariance has that rank, and which can be
            sampled -- the factor comes out of the same decomposition.

        Raises:
            ValueError: for a rank above the dimension, or a measure with no
                covariance to approximate.
        """
        from ..numerics.randomised import random_cholesky

        covariance = self._covariance
        if covariance is None:
            raise ValueError(
                "A low-rank approximation needs the covariance, and this "
                "measure was given only a precision."
            )
        factorised = random_cholesky(
            covariance.with_traits(Traits.POSITIVE_SEMIDEFINITE),
            rank=rank,
            rng=rng,
            **kwargs,
        )
        return GaussianMeasure(
            self._domain,
            expectation=self._expectation,
            covariance_factor=factorised.factor,
        )

    def with_regularized_inverse(
        self,
        solver: Any,
        /,
        *,
        damping: float = 0.0,
    ) -> "GaussianMeasure[X]":
        """The same measure, given a precision by inverting ``C + damping I``.

        A covariance with a decaying spectrum is singular in practice long
        before it is in theory, so its precision does not exist and anything
        needing one — a Mahalanobis distance, a log density, a KL divergence —
        is unavailable. Damping supplies one, at the cost of saying that the
        smallest variances are no smaller than ``damping``.

        The covariance itself is left alone. Only the precision is regularised,
        so the two are deliberately *not* inverses of each other and the
        measure says so by construction.

        Args:
            damping: the floor added to the spectrum before inverting. Larger
                is better conditioned and further from the true precision.

        Returns:
            The measure with a regularised precision.

        Raises:
            ValueError: for non-positive damping, or a measure with no
                covariance to regularise the inverse of.
        """
        if damping < 0.0:
            raise ValueError(f"The damping must be non-negative, got {damping}.")
        from ..algebra.operators import LinearOperator
        from ..traits import Traits as _Traits

        operator = self._require_covariance("Regularising the inverse")
        if damping > 0.0:
            operator = operator + damping * LinearOperator.identity(self._domain)
        precision = solver(operator.with_traits(_Traits.POSITIVE_DEFINITE))
        return GaussianMeasure(
            self._domain,
            expectation=self._expectation,
            covariance=self._covariance,
            covariance_factor=self._covariance_factor,
            precision=precision.with_traits(_Traits.POSITIVE_DEFINITE),
            sample=self._sample_fn,
        )

    def with_sparse_approximation(
        self, /, *, threshold: float = 1e-3, form: str = "galerkin"
    ) -> "GaussianMeasure[X]":
        """The same measure with entries below a relative threshold dropped.

        For a covariance with genuinely local correlations, most of the matrix
        is noise-level and storing it is waste. Thresholding is the crudest
        localisation there is and it does not preserve positive definiteness —
        so the result is checked, and refused if it has stopped being a
        covariance rather than being returned as one.

        Args:
            threshold: entries below this fraction of their row's diagonal are
                dropped.
            form: which matrix to threshold.

        Returns:
            The measure with a sparsified covariance.

        Raises:
            ValueError: if the result is no longer positive semidefinite --
                which thresholding can do, and is why it is checked.
        """
        require_coordinates(self._domain)
        if not 0.0 <= threshold < 1.0:
            raise ValueError(f"The threshold lies in [0, 1), got {threshold}.")
        from scipy.sparse import csr_matrix

        from ..algebra.operators import LinearOperator
        from ..traits import Traits as _Traits

        matrix = self._covariance.matrix(form=form)
        matrix = 0.5 * (matrix + matrix.T)
        matrix[np.abs(matrix) < threshold * np.abs(matrix).max()] = 0.0
        if np.linalg.eigvalsh(matrix).min() < -1e-10 * np.abs(matrix).max():
            raise ValueError(
                f"Thresholding at {threshold} left an operator that is no "
                "longer positive semidefinite, so it is not a covariance. Use "
                "a smaller threshold."
            )
        sparse = csr_matrix(matrix)
        covariance = LinearOperator.from_matrix(
            self._domain,
            self._domain,
            sparse,
            form=form,
            traits=_Traits.SELF_ADJOINT | _Traits.POSITIVE_SEMIDEFINITE,
        )
        return GaussianMeasure(
            self._domain, expectation=self._expectation, covariance=covariance
        )

    def credible_set(self, /, *, level: float = 0.95) -> "Ellipsoid":
        """The region carrying a given share of the probability, as a set.

        The **hardening** of DESIGN.md section 18.1: a measure becomes a set at
        a chosen chi-squared level. It is not canonical and it is not
        reversible — the ellipsoid carries no memory of the distribution it
        came from — which is why it is a named step rather than something a
        constructor does quietly.

        Args:
            level: the probability the region carries, in ``(0, 1)``.

        Returns:
            The credible ellipsoid.

        Raises:
            ValueError: for a level outside ``(0, 1)``, or a measure with no
                covariance.
        """
        from scipy.stats import chi2

        from ..geometry.convex import Ellipsoid

        if not 0.0 < level < 1.0:
            raise ValueError(f"A credible level lies in (0, 1), got {level}.")
        threshold = float(chi2.ppf(level, self.domain.dim))
        # The ellipsoid is defined by its precision; the covariance only rides
        # along for whoever wants it. So a measure described by a precision
        # alone has a perfectly good credible set, and used to raise.
        covariance = None if self._covariance is None else self._covariance * threshold
        if self._precision is not None:
            precision = self._precision * (1.0 / threshold)
        else:
            # A measure built from a covariance alone has no precision, and an
            # ellipsoid is defined by one. Inverting densely is the only thing
            # that can be done without being told how, and it is what a caller
            # would otherwise have to write.
            from ..algebra.operators import LinearOperator as _LinearOperator

            require_coordinates(self._domain)
            # The Galerkin matrix of C^-1 is G C_c^-1 == G C_gal^-1 G, not
            # C_gal^-1: inverting the Galerkin matrix gives C_c^-1 G^-1, which
            # is the component matrix of something else entirely. On an
            # orthonormal basis the two coincide, and the resulting credible
            # set covered 46% of its nominal 90% on a weighted one.
            gram = self._domain.gram_matrix()
            galerkin = gram @ np.linalg.solve(covariance.matrix(form="galerkin"), gram)
            precision = _LinearOperator.from_matrix(
                self._domain,
                self._domain,
                0.5 * (galerkin + galerkin.T),
                traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
                form="galerkin",
            )
        return Ellipsoid(
            self.domain,
            precision,
            centre=self.expectation,
            covariance=covariance,
        )

    def ambient_ball(self, /, *, level: float = 0.95, method: str = "auto") -> Any:
        """The smallest ball about the mean carrying a given probability.

        A different hardening from :meth:`credible_set`, and a cruder one: that
        gives the ellipsoid the distribution's own shape, this gives a ball in
        the *space's* norm. The two coincide only for an isotropic measure, and
        the ball is always the larger.

        It is worth having because a norm bound is what a set-theoretic prior
        is, so this is the bridge from a Gaussian belief to one — §18.1's
        conversion, done in the geometry the constraint will be used in.

        The radius is a quantile of ``sum_i lambda_i Z_i^2`` with the
        covariance's eigenvalues as weights, which is not a chi-square unless
        the measure is isotropic.

        Args:
            level: the probability the ball carries, in ``(0, 1)``.
            method: how to invert that weighted chi-squared -- see
                :func:`~pygeoinf2.numerics.quadratic_forms.weighted_chi2_quantile`.

        Returns:
            A ball containing the credible region.
        """
        from ..geometry.convex import Ball
        from ..numerics.quadratic_forms import weighted_chi2_quantile

        require_coordinates(self._domain)
        matrix = self._require_covariance("An ambient ball").matrix(form="components")
        eigenvalues = np.clip(np.linalg.eigvals(matrix).real, 0.0, None)
        radius = np.sqrt(weighted_chi2_quantile(eigenvalues, level, method=method))
        return Ball(self._domain, radius=float(radius), centre=self.expectation)

    def as_multivariate_normal(self) -> Any:
        """The measure as a ``scipy.stats`` object, in components.

        For anything scipy already does — a density, a rank correlation, a
        statistical test. It is the *component* representation, so it is a
        statement about coefficients rather than about fields, and a metric
        that is not the identity does not travel with it.
        """
        from scipy.stats import multivariate_normal

        require_coordinates(self._domain)

        # The covariance *of the components* is G^-1 C_gal G^-1, which is
        # symmetric; the operator's component matrix is not, on a space whose
        # basis is not orthonormal, and scipy rightly refuses it.
        #
        # Applied one column at a time, deliberately. `solve_gram` takes a
        # vector; handing it a matrix and relying on broadcasting gives
        # `M / g` row-wise on a space with a diagonal metric and a genuine
        # `G^-1 M` on one without, so an expression built that way can be
        # right on the first and wrong on the second — which is what this was.
        def divide(matrix: np.ndarray) -> np.ndarray:
            return np.column_stack(
                [self._domain.solve_gram(column) for column in matrix.T]
            )

        galerkin = self._require_covariance("A multivariate normal").matrix(
            form="galerkin"
        )
        components = divide(divide(galerkin).T)
        return multivariate_normal(
            mean=self._domain.to_components(self.expectation),
            cov=0.5 * (components + components.T),
            allow_singular=True,
        )

    def condition(
        self,
        operator: LinearOperator,
        value: X,
        /,
        *,
        noise: "GaussianMeasure | None" = None,
        solver: Any = None,
    ) -> "GaussianMeasure[X]":
        """The measure conditioned on ``A x == value``, or on noisy data.

        The Bayesian update, done here rather than in the inference layer
        because it is a statement about the measure and needs no forward
        problem: give it an operator and an observed value and it returns the
        posterior. With ``noise`` omitted the constraint is exact, and the
        result is supported on an affine subspace.

        **The result can be sampled**, by the Matheron rule: draw ``x`` from
        this measure and ``e`` from the noise, and return
        ``x + K(value - A x - e)``. That has the posterior's mean and, by the
        usual cancellation, exactly its covariance — at the cost of one prior
        draw and one solve, with no factorisation of the posterior covariance.
        It matters because conditioning is how an exact linear constraint — a
        conserved mass, a removed degree — is imposed on a prior, and a prior
        that cannot be sampled cannot generate synthetic data or be checked
        against any of the machinery downstream of it.

        Args:
            operator: the observation operator ``A``.
            value: the observed value.
            noise: the observation error. Omitted means the constraint is
                exact.
            solver: how to invert ``A C A* + R`` on the constraint codomain.
                Conjugate gradients by default; a factory taking the operator
                is also accepted. This used to be a dense ``np.linalg.inv``
                with no way to change it.

        Returns:
            The conditioned measure.

        Raises:
            ValueError: if this measure has no covariance operator.
        """
        from ..numerics.solvers import resolve_solver
        from ..traits import Traits as _Traits

        covariance = self._require_covariance("Conditioning")
        codomain = operator.codomain

        normal = operator @ covariance @ operator.adjoint
        if noise is not None:
            normal = normal + noise.covariance
        normal = normal.with_traits(_Traits.SELF_ADJOINT | _Traits.POSITIVE_DEFINITE)
        inverse = resolve_solver(solver, normal)(normal)
        cross = covariance @ operator.adjoint

        def gain(vector: Any) -> Any:
            return cross(inverse(vector))

        predicted = operator(self.expectation)
        if noise is not None:
            predicted = codomain.add(predicted, noise.expectation)
        shift = gain(codomain.subtract(value, predicted))
        updated = covariance - cross @ inverse @ cross.adjoint

        sample = None
        if self.can_sample and (noise is None or noise.can_sample):

            def sample(rng: Generator | None, _noise=noise) -> Any:
                """A *centred* posterior draw: ``(I - K A) dx - K de``.

                Centred because :meth:`sample` adds the expectation itself, and
                because it is true: the posterior's spread does not depend on
                the data, only its mean does. So this is built from the prior
                and noise *deviations* rather than from the draws.
                """
                drawn = self._domain.subtract(self.sample(rng=rng), self.expectation)
                residual = codomain.negative(operator(drawn))
                if _noise is not None:
                    disturbance = codomain.subtract(
                        _noise.sample(rng=rng), _noise.expectation
                    )
                    residual = codomain.subtract(residual, disturbance)
                return self._domain.add(drawn, gain(residual))

        return GaussianMeasure(
            self._domain,
            expectation=self._domain.add(self.expectation, shift),
            covariance=updated.with_traits(
                _Traits.SELF_ADJOINT | _Traits.POSITIVE_SEMIDEFINITE
            ),
            sample=sample,
        )

    def _require_direct_sum(self, what: str, /) -> "DirectSum":
        """The domain as a direct sum, or a message saying it is not one."""
        from ..algebra.direct_sum import DirectSum

        if not isinstance(self._domain, DirectSum):
            raise ValueError(
                f"{what} needs a measure on a direct sum; this one is on "
                f"{self._domain!r}."
            )
        return self._domain

    def marginal(self, key: int | str, /) -> "GaussianMeasure":
        """The law of one summand, for a measure on a direct sum.

        The joint measure a coupled problem builds is rarely the one it wants
        to look at: the interesting question is usually about one field. This
        is the pushforward under the projection, so it costs nothing and stays
        exact -- the alternative was slicing the covariance's component matrix,
        which is a dense ``dim x dim`` assembly to read one block off.

        Args:
            key: which summand, by position or by label.

        Returns:
            The marginal measure.

        Raises:
            ValueError: if this measure is not on a direct sum.
        """
        from ..algebra.diagonal import DiagonalLinearOperator

        domain = self._require_direct_sum("A marginal")
        covariance = self._covariance
        if covariance is None:
            return self.push_forward(domain.projection(key))

        block = self._block_of(covariance, domain, key, key)
        expectation = (
            None
            if self.has_zero_expectation
            else domain.component(self._expectation, key)
        )
        # A diagonal block carries its own factor -- the square root of its
        # eigenvalues -- so the marginal can still be sampled. The general
        # route cannot say that: if C = L L*, the (i, i) block of C is a sum
        # over the whole i-th row of L, not L_ii L_ii*.
        factor = block.sqrt if isinstance(block, DiagonalLinearOperator) else None
        return GaussianMeasure(
            block.domain,
            covariance=block,
            covariance_factor=factor,
            expectation=expectation,
        )

    def cross_covariance(
        self, first: int | str, second: int | str, /
    ) -> LinearOperator:
        """``Cov(x_i, x_j)`` as an operator, for a measure on a direct sum.

        ``P_i C P_j*``: it maps the *second* summand's space to the first's,
        which is the direction that makes ``(u, Cov(x_i, x_j) v)`` the
        covariance of ``(u, x_i)`` with ``(v, x_j)``. The adjoint is taken
        rather than the inclusion assumed, so this is right whatever metric the
        summands carry.

        Args:
            first: the summand on the left.
            second: the summand on the right.

        Returns:
            The operator.

        Raises:
            ValueError: if this measure is not on a direct sum, or has no
                covariance.
        """
        domain = self._require_direct_sum("A cross-covariance")
        covariance = self._require_covariance("A cross-covariance")
        return self._block_of(covariance, domain, first, second)

    @staticmethod
    def _block_of(
        operator: LinearOperator,
        domain: "DirectSum",
        first: int | str,
        second: int | str,
        /,
    ) -> LinearOperator:
        """One block of an operator on a direct sum.

        Read straight off a block operator where there is one, so that a
        diagonal block stays diagonal and keeps its eigenvalues and its
        traits. Sandwiching it between projections instead is correct but
        gives back a composition, which has forgotten all of that.
        """
        from ..algebra.direct_sum import BlockLinearOperator

        left, right = domain.index(first), domain.index(second)
        if isinstance(operator, BlockLinearOperator):
            return operator.blocks[left][right]
        return domain.projection(first) @ operator @ domain.projection(second).adjoint

    def mahalanobis_squared(self, x: X) -> float:
        """``(x - m, P (x - m))``, the squared Mahalanobis distance.

        Needs a precision, and says so rather than falling back to a dense
        solve. The fallback exists — :meth:`_weighted_squared` has it — but
        reaching it silently would mean a cubic cost incurred by a method that
        looks like a quadratic form. Regularise one into existence with
        :meth:`with_regularized_inverse` if that is what you want.

        Args:
            x: the vector to measure.

        Returns:
            The squared distance.

        Raises:
            ValueError: if the measure has no precision.
        """
        if self._precision is None:
            raise NotImplementedError(
                "This measure has no precision, so no Mahalanobis distance. "
                "Supply precision or precision_factor, or regularise one into "
                "existence with with_regularized_inverse."
            )
        deviation = self._deviation(x)
        return self._domain.inner_product(self._precision(deviation), deviation)

    def log_normalising_constant(self) -> float:
        """``-(n/2) log(2 pi) - (1/2) log det C``, the constant in the density.

        The piece :meth:`log_density` leaves out. It depends on the covariance,
        so it is *not* shared between two measures on the same space, and any
        comparison across measures — a mixture's components, most of all — has
        to put it back. Constant in ``x``, so it is computed once and kept.

        The determinant is the component matrix's, as in
        :func:`~pygeoinf2.numerics.functional_calculus.log_determinant`: that
        is the one belonging to the operator rather than to the metric, and it
        is what makes the density a density with respect to the space's own
        volume measure.

        Returns:
            The additive constant, in nats.

        Raises:
            NotImplementedError: if the measure has no covariance, so there is
                no determinant to take.
        """
        if self._log_normalisation is None:
            self._log_normalisation = self._compute_log_normalisation()
        return self._log_normalisation

    def _compute_log_normalisation(self) -> float:
        """The constant, exactly for a diagonal covariance and otherwise not."""
        from ..numerics.functional_calculus import log_determinant

        gaussian = -0.5 * self._domain.dim * float(np.log(2.0 * np.pi))

        eigenvalues = self._diagonal_eigenvalues()
        if eigenvalues is not None:
            # Exact, and cheap. Taken before any retraiting, so it survives
            # whatever the general route would have done with the claim.
            return gaussian - 0.5 * float(np.sum(np.log(eigenvalues)))

        if self._covariance is None:
            raise NotImplementedError(
                "This measure has no covariance, so no normalising constant. "
                "Supply covariance or covariance_factor."
            )
        definite = self._covariance.with_traits(
            Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
        )
        return gaussian - 0.5 * log_determinant(definite).value

    def log_density(self, x: X) -> float:
        """The log density up to an additive constant.

        The constant is :meth:`log_normalising_constant`, which depends on the
        covariance. Differences of this quantity are meaningful only *within*
        one measure; comparing two measures means adding each one's constant.
        """
        return -0.5 * self.mahalanobis_squared(x)

    def grad_log_density(self, x: X) -> X:
        """``-P (x - m)``, already a vector in the domain.

        No Riesz map is applied here because none is needed: the precision maps
        the space to itself, so its output is a vector. That is the whole
        content of DESIGN.md section 5.6 in its most agreeable form.

        Args:
            x: where to evaluate it.

        Returns:
            The gradient, a vector of the domain.

        Raises:
            ValueError: if the measure has no precision, the gradient being
                ``-P (x - m)``.
        """
        if self._precision is None:
            raise NotImplementedError(
                "This measure has no precision, so no log-density gradient."
            )
        return self._domain.negative(self._precision(self._deviation(x)))

    @property
    def has_log_density(self) -> bool:
        """True when a precision is available, so a density can be evaluated."""
        return self._precision is not None

    @property
    def has_grad_log_density(self) -> bool:
        """True when a precision is available."""
        return self._precision is not None

    # ----------------------------------------------------------------- #
    #                      Structure-preserving algebra                 #
    # ----------------------------------------------------------------- #

    def _rebuild(
        self,
        domain: HilbertSpace,
        /,
        *,
        expectation: object | None,
        covariance: LinearOperator | None,
        covariance_factor: LinearOperator | None,
        precision: LinearOperator | None = None,
        precision_factor: LinearOperator | None = None,
        sample: Callable[[Generator | None], object] | None = None,
    ) -> GaussianMeasure:
        """Build a measure of this class. Subclasses override to stay in theirs.

        The precision is carried because otherwise nothing can carry it: every
        algebraic operation goes through here, so a signature without it means
        no operation whatever can keep a precision, and none did — not even
        translation, which does not change the covariance at all. The cost was
        not only the density (``mu.translate(v).log_density`` raised where
        ``mu.log_density`` worked) but the spectral KL route, which falls back
        to a dense solve when the reference measure has no precision.
        """
        return GaussianMeasure(
            domain,
            expectation=expectation,
            covariance=covariance,
            covariance_factor=covariance_factor,
            precision=precision,
            precision_factor=precision_factor,
            sample=sample,
        )

    @staticmethod
    def _diagonal_inverse(
        covariance: LinearOperator | None,
    ) -> tuple[LinearOperator | None, LinearOperator | None]:
        """A diagonal covariance's factor and precision, when it has them.

        Built from the eigenvalues rather than through
        :attr:`DiagonalLinearOperator.sqrt` and :attr:`inverse`, which gate on
        *traits* that are never deduced on a space whose metric is not
        diagonal. The functional calculus of a diagonalisable operator does not
        depend on the metric, so the eigenvalues are the honest thing to gate
        on.
        """
        from ..algebra.diagonal import DiagonalLinearOperator

        if not isinstance(covariance, DiagonalLinearOperator):
            return None, None
        values = np.asarray(covariance.eigenvalues, dtype=float)
        if np.any(values <= 0.0):
            return None, None
        space = covariance.domain
        return (
            DiagonalLinearOperator(space, np.sqrt(values)),
            DiagonalLinearOperator(space, 1.0 / values),
        )

    def _combine_affine(
        self, operator: LinearOperator, translation: X | None
    ) -> GaussianMeasure[X]:
        """A Gaussian stays Gaussian under an affine map."""
        codomain = operator.codomain
        mean = operator(self.expectation)
        if translation is not None:
            mean = codomain.add(mean, translation)

        factor = (
            None
            if self._covariance_factor is None
            else operator @ self._covariance_factor
        )
        covariance = (
            None
            if factor is not None or self._covariance is None
            else operator @ self._covariance @ operator.adjoint
        )
        # With no factor to map there is still a sampler: push each draw
        # through the operator. Losing samplability under a linear map would
        # be a gratuitous restriction.
        sample = None
        if factor is None and self.can_sample:

            def sample(rng, _operator=operator, _translation=translation):
                mapped = _operator(self.sample(rng=rng))
                if _translation is None:
                    return mapped
                return codomain.add(mapped, _translation)

        precision, precision_factor = self._mapped_precision(operator)
        if (
            covariance is None
            and factor is None
            and precision is None
            and precision_factor is None
        ):
            # A precision-only measure under a map that does not preserve one:
            # nothing describes the result, so fall back rather than build an
            # object whose every method raises.
            return None
        return self._rebuild(
            codomain,
            expectation=mean,
            covariance=covariance,
            covariance_factor=factor,
            precision=precision,
            precision_factor=precision_factor,
            sample=sample,
        )

    def _mapped_precision(
        self, operator: LinearOperator
    ) -> tuple[LinearOperator | None, LinearOperator | None]:
        """The precision of ``A X``, when the map leaves one.

        ``C -> A C A*`` inverts to ``P -> (A^-1)* P A^-1``, so a precision
        survives exactly when ``A`` does. Two cases are worth taking:

        * the identity, which is what a *translation* is built from. A shift
          does not touch the covariance, so dropping the precision there was
          pure loss — and it is the commonest operation of the three.
        * an invertible diagonal operator, which is what masking or rescaling a
          field by a spectral multiplier looks like.

        Anything else gives up, because inverting a general operator is a solve
        and doing one silently inside an algebraic operation is exactly the
        kind of hidden cost the library is trying not to have.
        """
        from ..algebra.diagonal import DiagonalLinearOperator
        from ..algebra.nodes import _Identity

        if isinstance(operator, _Identity):
            return self._precision, self._precision_factor

        if isinstance(operator, DiagonalLinearOperator):
            values = np.asarray(operator.eigenvalues, dtype=float)
            if np.all(values != 0.0):
                inverse = DiagonalLinearOperator(operator.domain, 1.0 / values)
                if self._precision is not None:
                    return inverse.adjoint @ self._precision @ inverse, None
                if self._precision_factor is not None:
                    return None, self._precision_factor @ inverse
        return None, None

    def _combine_add(self, other: object) -> GaussianMeasure[X] | None:
        """The sum of independent Gaussians is Gaussian."""
        if not isinstance(other, GaussianMeasure):
            return None
        if self._covariance is None or other.covariance is None:
            return None

        # The sum of two covariances has no factor in general, but the sum of
        # two samplable measures is samplable: draw from each and add.
        sample = None
        if self.can_sample and other.can_sample:

            def sample(rng, _other=other):
                return self._domain.add(self.sample(rng=rng), _other.sample(rng=rng))

        # Two diagonal covariances add to a diagonal one, and a diagonal
        # covariance hands back its own factor and precision. That turns the
        # sum of two invariant measures -- which is how a prior and a noise
        # model are combined -- back into a measure that can be sampled with
        # one draw rather than two, and whose density can be evaluated at all.
        total = self._covariance + other.covariance
        factor, precision = self._diagonal_inverse(total)
        if factor is not None:
            sample = None  # the factor is the better sampler
        return self._rebuild(
            self._domain,
            expectation=self._domain.add(self.expectation, other.expectation),
            covariance=total,
            covariance_factor=factor,
            precision=precision,
            sample=sample,
        )

    def _combine_scale(self, alpha: float) -> GaussianMeasure[X] | None:
        """``alpha X``: the factor scales by ``alpha`` and the covariance by
        ``alpha^2``.

        Works from whichever of the two the measure has. Requiring a factor
        turned a measure described by a covariance and a precision — which is
        every conditioned or explicitly-built one — into an unspecialised
        pushforward under a scalar multiple.
        """
        if (
            self._covariance_factor is None
            and self._covariance is None
            and self._precision is None
            and self._precision_factor is None
        ):
            return None
        # C -> alpha^2 C, so P -> P / alpha^2 and each factor by |alpha|.
        # A zero scaling is a point mass and has no precision at all.
        precision = None
        precision_factor = None
        if alpha != 0.0:
            if self._precision is not None:
                precision = (1.0 / (alpha * alpha)) * self._precision
            if self._precision_factor is not None:
                precision_factor = (1.0 / abs(alpha)) * self._precision_factor
        factor = (
            None if self._covariance_factor is None else alpha * self._covariance_factor
        )
        covariance = (
            None
            if factor is not None or self._covariance is None
            else (alpha * alpha) * self._covariance
        )
        if covariance is None and factor is None and precision is None:
            # Scaling a precision-only measure by zero.
            return None
        return self._rebuild(
            self._domain,
            expectation=self._domain.scale(alpha, self.expectation),
            covariance=covariance,
            covariance_factor=factor,
            precision=precision,
            precision_factor=precision_factor,
        )

    def __repr__(self) -> str:
        return f"GaussianMeasure({self._domain!r})"
