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
    from ..geometry.convex import Ellipsoid
from .base import ProbabilityMeasure

__all__ = ["GaussianMeasure"]


_REQUIRED = Traits.SELF_ADJOINT | Traits.POSITIVE_SEMIDEFINITE


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
        """
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
    def from_samples(
        cls, domain: HilbertSpace[X], samples: Sequence[X], /
    ) -> GaussianMeasure[X]:
        """The empirical mean and covariance of a set of vectors.

        Coordinate-free: the covariance is a sum of outer products, and its
        factor maps a Euclidean coefficient vector onto the deviations.
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
        definite exactly when every factor is.
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

        sample = None
        if factor is None and all(m.can_sample for m in measures):

            def sample(rng, _measures=measures):
                return tuple(m.sample(rng=rng) for m in _measures)

        return cls(
            domain,
            expectation=tuple(m.expectation for m in measures),
            covariance=covariance,
            covariance_factor=factor,
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
    ) -> GaussianMeasure[X]:
        """From an explicit covariance matrix.

        ``form`` says which representation the array is in, because no trait
        implies it (DESIGN.md 5.3). The Galerkin form is the natural one here:
        a covariance is self-adjoint, so that is the representation in which it
        is symmetric, and the one a Cholesky factorisation wants.
        """
        require_coordinates(domain)
        matrix = np.asarray(matrix, dtype=float)
        expected = (domain.dim, domain.dim)
        if matrix.shape != expected:
            raise ValueError(f"Matrix has shape {matrix.shape}, expected {expected}.")

        if form == "components":
            matrix = domain.apply_gram(matrix.T).T  # to the Galerkin form
        elif form != "galerkin":
            raise ValueError(f"Unknown form {form!r}.")

        symmetric = 0.5 * (matrix + matrix.T)
        root = np.linalg.cholesky(symmetric)
        # from_derivative_matrix(E, X, R) has component matrix G^-1 R, and
        # (G^-1 R)(G^-1 R)* has Galerkin matrix R R^T == the covariance.
        factor = LinearOperator.from_derivative_matrix(
            EuclideanSpace(domain.dim), domain, root
        )
        return cls(domain, expectation=expectation, covariance_factor=factor)

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
        """One draw, as ``m + L xi`` with ``xi`` white noise on the factor's domain."""
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

    def hilbert_schmidt_norm(self, /, *, method: str = "auto") -> float:
        """The Hilbert-Schmidt norm of the covariance, ``sqrt(tr(C* C))``."""
        eigenvalues = self._diagonal_eigenvalues()
        if eigenvalues is not None and method in ("auto", "diagonal"):
            return float(np.sqrt(np.sum(eigenvalues**2)))
        # tr(C* C) is basis-independent, so it comes from the *component*
        # matrix. The Galerkin one is G C_c, whose trace is a different number.
        matrix = self._covariance.matrix(form="components")
        return float(np.sqrt(np.sum(matrix * matrix.T)))

    def nuclear_norm(self, /, *, method: str = "auto") -> float:
        """The trace norm of the covariance, ``tr|C|``.

        For a covariance this is the trace, since it is positive semidefinite —
        the total variance of the measure.
        """
        eigenvalues = self._diagonal_eigenvalues()
        if eigenvalues is not None and method in ("auto", "diagonal"):
            return float(np.sum(np.abs(eigenvalues)))
        # A covariance is positive semidefinite, so its trace norm is its
        # trace -- and a trace is the component matrix's, not the Galerkin
        # matrix's, which carries an extra factor of the metric.
        return float(np.trace(self._covariance.matrix(form="components")))

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

    def _symmetric_matrix(self) -> np.ndarray:
        """The covariance's Galerkin matrix, symmetrised against round-off."""
        matrix = self._covariance.matrix(form="galerkin")
        return 0.5 * (matrix + matrix.T)

    def kl_divergence(
        self, other: "GaussianMeasure", /, *, method: str = "auto"
    ) -> float:
        """``D(self || other)`` between two Gaussians on the same space.

        When both covariances are diagonal in the space's own basis the whole
        thing reduces to sums over the spectrum, which is ``O(dim)`` rather
        than a pair of dense factorisations. That is the case for every
        invariant measure on a symmetric space, so it is the common one.

        Args:
            other: the reference measure.
            method: ``"auto"`` takes the spectral route when it is available,
                ``"dense"`` forces the general one.
        """
        if other.domain != self._domain:
            raise ValueError("Both measures must live on the same space.")
        dimension = self._domain.dim
        shift = self._domain.subtract(other.expectation, self.expectation)

        mine = self._diagonal_eigenvalues()
        theirs = other._diagonal_eigenvalues()
        if method == "auto" and mine is not None and theirs is not None:
            if np.any(theirs <= 0.0):
                raise ValueError(
                    "The reference measure is singular, so the divergence from "
                    "it is infinite."
                )
            components = self._domain.to_components(shift)
            metric = self._domain.apply_gram(components)
            quadratic = float(
                np.sum(components * metric**2 / theirs / metric.clip(min=1e-300))
            )
            quadratic = float(
                other.mahalanobis_squared(other.expectation) * 0.0
            ) + float(self._domain.inner_product(other.precision(shift), shift))
            trace = float(np.sum(mine / theirs))
            logs = float(
                np.sum(np.log(theirs)) - np.sum(np.log(np.clip(mine, 1e-300, None)))
            )
            return 0.5 * (trace + quadratic - dimension + logs)

        mine_matrix = self._symmetric_matrix()
        theirs_matrix = other._symmetric_matrix()
        solved = np.linalg.solve(theirs_matrix, mine_matrix)
        quadratic = other._weighted_squared(shift)
        sign_mine, log_mine = np.linalg.slogdet(mine_matrix)
        sign_theirs, log_theirs = np.linalg.slogdet(theirs_matrix)
        if sign_mine <= 0 or sign_theirs <= 0:
            raise ValueError("Both covariances must be positive definite.")
        return 0.5 * (
            float(np.trace(solved)) + quadratic - dimension + log_theirs - log_mine
        )

    def rescale_directional_variance(
        self, direction: X, standard_deviation: float, /
    ) -> "GaussianMeasure[X]":
        """The same measure with ``Var((x, direction))`` set to a given value.

        A whole-measure rescaling, as v1 does it, not a rank-one update: the
        covariance is multiplied by a scalar chosen so that one direction comes
        out right. Every other variance moves by the same factor, which is what
        makes it a recalibration rather than a change of shape.
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
        """
        if damping < 0.0:
            raise ValueError(f"The damping must be non-negative, got {damping}.")
        from ..algebra.operators import LinearOperator
        from ..traits import Traits as _Traits

        operator = self._covariance
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
        builder = (
            LinearOperator.from_derivative_matrix
            if form == "galerkin"
            else LinearOperator.from_component_matrix
        )
        covariance = builder(
            self._domain,
            self._domain,
            sparse,
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
        """
        from scipy.stats import chi2

        from ..geometry.convex import Ellipsoid

        if not 0.0 < level < 1.0:
            raise ValueError(f"A credible level lies in (0, 1), got {level}.")
        threshold = float(chi2.ppf(level, self.domain.dim))
        return Ellipsoid(
            self.domain,
            self.precision * (1.0 / threshold),
            centre=self.expectation,
            covariance=self.covariance * threshold,
        )

    def mahalanobis_squared(self, x: X) -> float:
        """``(x - m, P (x - m))``, the squared Mahalanobis distance."""
        if self._precision is None:
            raise NotImplementedError(
                "This measure has no precision, so no Mahalanobis distance. "
                "Supply precision or precision_factor."
            )
        deviation = self._deviation(x)
        return self._domain.inner_product(self._precision(deviation), deviation)

    def log_density(self, x: X) -> float:
        """The log density up to an additive constant."""
        return -0.5 * self.mahalanobis_squared(x)

    def grad_log_density(self, x: X) -> X:
        """``-P (x - m)``, already a vector in the domain.

        No Riesz map is applied here because none is needed: the precision maps
        the space to itself, so its output is a vector. That is the whole
        content of DESIGN.md section 5.6 in its most agreeable form.
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
        sample: Callable[[Generator | None], object] | None = None,
    ) -> GaussianMeasure:
        """Build a measure of this class. Subclasses override to stay in theirs."""
        return GaussianMeasure(
            domain,
            expectation=expectation,
            covariance=covariance,
            covariance_factor=covariance_factor,
            sample=sample,
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
            if factor is not None
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

        return self._rebuild(
            codomain,
            expectation=mean,
            covariance=covariance,
            covariance_factor=factor,
            sample=sample,
        )

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

        return self._rebuild(
            self._domain,
            expectation=self._domain.add(self.expectation, other.expectation),
            covariance=self._covariance + other.covariance,
            covariance_factor=None,
            sample=sample,
        )

    def _combine_scale(self, alpha: float) -> GaussianMeasure[X] | None:
        """``alpha X`` scales the factor, and so the covariance by ``alpha^2``."""
        if self._covariance_factor is None:
            return None
        return self._rebuild(
            self._domain,
            expectation=self._domain.scale(alpha, self.expectation),
            covariance=None,
            covariance_factor=alpha * self._covariance_factor,
        )

    def __repr__(self) -> str:
        return f"GaussianMeasure({self._domain!r})"
