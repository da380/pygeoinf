"""
Gaussian measures on Hilbert spaces.

A Gaussian is determined by an expectation and a covariance, and stays Gaussian
under an affine map. Both facts are carried structurally here: the operator
algebra recognises a product of the form ``A C A*`` with ``C`` positive
semidefinite as self-adjoint and positive semidefinite, so the pushforward
covariance carries those traits with nothing asserted.

Given a factor ``L`` with ``C == L L*``, a sample is ``m + L xi`` where ``xi``
is white noise **on the factor's own domain**. When that domain is the space
itself -- the isotropic case, ``L == sigma I`` -- the noise must be white with
respect to the space's inner product, not with respect to its components:
standard normal components would give covariance ``sigma^2 G`` rather than
``sigma^2 I`` on any space whose Gram matrix ``G`` is not the identity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence

import numpy as np
from numpy.random import Generator

from ..algebra.operators import LinearOperator, require_coordinates
from ..algebra.spaces import CoordinateSpace, EuclideanSpace, HilbertSpace
from ..traits import Traits, congruence_traits

if TYPE_CHECKING:  # pragma: no cover
    from ..algebra.direct_sum import DirectSum
    from ..geometry.convex import Ellipsoid
    from ..numerics.randomized import Estimate
from .base import ProbabilityMeasure, _require_density

__all__ = ["GaussianMeasure"]


_REQUIRED = Traits.SELF_ADJOINT | Traits.POSITIVE_SEMIDEFINITE


def _semidefinite_factors(
    symmetric: np.ndarray, /, *, rtol: float
) -> tuple[np.ndarray, np.ndarray | None]:
    """A square root of a symmetric PSD matrix, and its inverse when it exists.

    Cholesky first, because it is the cheaper factorization and because
    succeeding is a proof that the matrix is numerically definite. When it
    fails the matrix is singular or has drifted slightly negative, and a
    symmetric eigendecomposition decides which: eigenvalues below
    ``-rtol * max|lambda|`` mean the caller's matrix is not a covariance, and
    smaller negative ones are floating-point noise and are clipped, with a
    warning.

    Args:
        symmetric: the matrix, already symmetrized.
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


def _kl_kernel(values: np.ndarray) -> np.ndarray:
    """``t - 1 - log t``, the whole of a Gaussian divergence bar the mean shift.

    Non-negative, with a double zero at ``t == 1``: the two properties that
    make a stochastic divergence built on it come out positive and vanish
    identically when the two measures agree.

    Args:
        values: Ritz values from a Lanczos run, all positive in exact
            arithmetic.

    Returns:
        ``g`` at each of them.
    """
    values = np.asarray(values, dtype=float)
    # A Ritz value lies inside the spectrum, so it is positive; rounding can
    # still hand back a zero on a reference that is singular in the direction
    # the probe found. A floor makes that a very large divergence rather than
    # a nan, which is the honest reading of a singular reference.
    safe = np.clip(values, np.finfo(float).tiny, None)
    return safe - 1.0 - np.log(safe)


def _check_round_trip(
    operator: LinearOperator, inverse: LinearOperator, /, *, rtol: float = 1e-4
) -> None:
    """Refuse a solver that does not actually invert the operator.

    A Krylov solver stops on the *residual*, and on an ill-conditioned
    covariance a small residual can hide a large error. Two applications and
    one solve on a white-noise probe detect that, and refuse.

    Args:
        operator: the operator being inverted.
        inverse: the solver's inverse of it.
        rtol: how far ``A^-1 A x`` may sit from ``x``, relatively.

    Raises:
        ValueError: when it sits further than that.
    """
    space = operator.domain
    probe = space.white_noise()
    error = space.norm(space.subtract(inverse(operator(probe)), probe))
    reference = space.norm(probe)
    if error > rtol * reference:
        raise ValueError(
            f"The solver does not invert the reference covariance accurately "
            f"enough for this: a round trip moves a test vector by "
            f"{error / reference:.2e}, against a tolerance of {rtol:.0e}. On "
            f"an ill-conditioned covariance a small residual is not a small "
            f"error. Pass solver= with a preconditioner or a tighter "
            f"tolerance, or supply the reference measure's precision, which "
            f"removes the solve entirely."
        )


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

        if (
            covariance_factor is None
            and covariance is not None
            and covariance.is_identity
        ):
            # The identity is its own factor, so a white measure samples.
            covariance_factor = covariance
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
        self._log_normalization: float | None = None
        self._dense: GaussianMeasure[X] | None = None

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
        unsafe: bool = False,
    ) -> GaussianMeasure[X]:
        """An isotropic measure with covariance ``sigma^2 I`` **on the space**.

        The factor's domain is the space itself, so the sample draws white
        noise with respect to the space's inner product, and the covariance
        is ``sigma^2`` times the identity on the space whatever its metric.

        On a function space (a :class:`~pygeoinf2.algebra.spaces.HilbertModule`)
        this is refused: the identity is not trace class there, so the
        "measure" has no draws in the space, and the truncation only hides
        it. A prior on a function space is a measure of extra regularity on
        it, such as a Sobolev measure of order above half the spatial
        dimension; see :meth:`~pygeoinf2.symmetric_space.base.SymmetricSpace.sobolev_measure`.
        Data errors live on Euclidean spaces, where this is the usual noise.

        Args:
            domain: the space.
            standard_deviation: one number, this being an isotropic measure.
            expectation: the mean. Zero if omitted.
            unsafe: build it on a function space anyway, for a test that
                means it.

        Returns:
            The measure, carrying both a factor and a precision factor.

        Raises:
            TypeError: on a function space without *unsafe*.
            ValueError: for a non-positive or non-scalar deviation. Use
                from_standard_deviations for one per component.
        """
        from ..algebra.spaces import HilbertModule

        if isinstance(domain, HilbertModule) and not unsafe:
            raise TypeError(
                f"An identity covariance on {domain!r} is not a Gaussian "
                "measure on it: the identity is not trace class on a function "
                "space and its draws are not functions of the space. Use a "
                "Sobolev measure of order above half the spatial dimension "
                "(space.sobolev_measure), or unsafe=True for a test."
            )
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
        """A measure with a different standard deviation along each basis vector.

        The covariance factor is the diagonal operator ``diag(sigma)`` **on the
        space**, its factor domain the space itself, which generalizes
        :meth:`from_standard_deviation`; the precision factor is
        ``diag(1/sigma)``, and the covariance is ``diag(sigma^2)``. Both a
        factor and a precision factor are supplied, so the result can be
        sampled and has a density.

        Only on a space whose metric is diagonal in its basis -- every
        symmetric space, every orthonormal one, a weighted coordinate space.
        There ``diag(sigma)`` is self-adjoint and ``Var((x, e_i))`` is
        ``sigma_i^2 ||e_i||^2``, one variance per basis direction as the name
        says. On a space with a non-diagonal metric ``diag(sigma)`` is not
        self-adjoint and ``L L*`` is not diagonal in the components, so the
        construction is refused: give the covariance explicitly with
        :meth:`from_covariance_matrix`, or build an isotropic measure with
        :meth:`from_standard_deviation` and rescale it.

        Args:
            domain: the space the measure lives on.
            standard_deviations: one positive number per dimension.
            expectation: the mean. Defaults to zero.

        Returns:
            The measure with that diagonal covariance.

        Raises:
            ValueError: if the array is the wrong length or has an entry that
                is not positive, or if the space's metric is not diagonal.
        """
        from ..algebra.diagonal import DiagonalLinearOperator

        if not domain.has_diagonal_metric:
            raise ValueError(
                f"from_standard_deviations needs a space whose metric is "
                f"diagonal in its basis, and {type(domain).__name__}'s is not: "
                "diag(sigma) is not self-adjoint there, so its square is not "
                "diag(sigma^2). Give the covariance with "
                "from_covariance_matrix, or start from the isotropic "
                "from_standard_deviation."
            )
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
        domain: DirectSum | None = None,
    ) -> GaussianMeasure:
        """The independent product of Gaussians, on the direct sum of domains.

        The covariance is block diagonal, and the block-diagonal operator gives
        it the right traits by intersecting the blocks': the whole is positive
        definite exactly when every factor is. The precision is block diagonal
        for the same reason, and is built whenever every factor has one.

        The precision matters beyond the density: a Woodbury preconditioner
        built from a product prior inverts it through the precision when
        there is one, and by conjugate gradients at every application when
        there is not.

        Args:
            measures: the independent factors.
            labels: names for the summands of the direct sum.
            domain: the direct sum to put the measure on, when the caller has
                one already, such as a space that is a direct sum with more
                structure than the bare sum of the factors' domains (a
                layered space, with its pointwise product); its summands must
                be the factors' domains, in order. Built from them if
                omitted.

        Returns:
            The product measure, on the direct sum of the domains.

        Raises:
            ValueError: if no factors are given, if some factor has neither
                a covariance nor a covariance factor, or if a given domain's
                summands are not the factors' domains.
        """
        from ..algebra.direct_sum import BlockDiagonalLinearOperator, DirectSum

        measures = tuple(measures)
        if not measures:
            raise ValueError("A product measure needs at least one factor.")
        if domain is None:
            domain = DirectSum([m.domain for m in measures], labels=labels)
        else:
            if labels is not None:
                raise ValueError("Give labels or a domain, which has its own.")
            if not isinstance(domain, DirectSum) or len(domain.subspaces) != len(
                measures
            ):
                raise ValueError(
                    f"The domain must be a direct sum of {len(measures)} summands, "
                    f"got {domain!r}."
                )
            for index, (summand, measure) in enumerate(zip(domain.subspaces, measures)):
                if summand != measure.domain:
                    raise ValueError(
                        f"Summand {index} of the domain is {summand!r}, and the "
                        f"factor there lives on {measure.domain!r}."
                    )

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
        form: Literal["galerkin", "components"],
        expectation: X | None = None,
        rtol: float = 1e-10,
    ) -> GaussianMeasure[X]:
        """From an explicit covariance matrix.

        ``form`` says which representation the array is in: ``"galerkin"``
        (``G C_c``) or ``"components"`` (``C_c``, with ``c_{Cx} == C_c c_x``).
        The Galerkin form is the natural one for a covariance: it is
        self-adjoint, so that is the representation in which it is symmetric.

        A covariance is required to be positive *semi*definite, so a Cholesky
        factorization alone is not enough: it refuses every singular
        covariance -- a measure supported on a subspace, an empirical
        covariance from fewer samples than dimensions, a pushforward through a
        rank-deficient map -- and it refuses a matrix that is semidefinite in
        exact arithmetic but has eigenvalues of size ``-1e-17`` after being
        assembled in floating point. When Cholesky fails, a symmetric
        eigendecomposition is taken instead, small negative eigenvalues are
        clipped to zero with a warning, and both factors are built from it. A
        strictly definite matrix takes the Cholesky route, which is the cheaper
        one and which proves definiteness by succeeding.

        The measure carries a precision and a precision factor whenever the
        covariance is nonsingular, so that :meth:`mahalanobis_squared`,
        :meth:`log_density` and :meth:`grad_log_density` work. A *singular*
        covariance gets neither: the measure is degenerate, its density with
        respect to the space's own volume measure does not exist, and a
        pseudo-inverse in the precision slot would answer those three methods
        with a finite number that is not the thing they name.

        Cost is cubic in ``domain.dim`` and the result holds two dense
        matrices (three when a precision is attached, the Gram matrix being
        the third).

        Args:
            domain: the space.
            matrix: the covariance, as an array.
            form: which representation *matrix* is in. No default: getting
                it wrong is a silent error of one factor of the Gram matrix,
                so the caller names it, as ``LinearOperator.from_matrix``
                requires.
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
        factor = LinearOperator.from_matrix(coefficients, domain, root, form="galerkin")

        precision = None
        precision_factor = None
        if inverse_root is not None:
            # Li: X -> E with component matrix M has (Li* Li)_c == G^-1 M^T M,
            # and the precision's component matrix is C_c^-1 == C_gal^-1 G. So
            # M^T M must be G C_gal^-1 G, which M == R^-1 G delivers. The Gram
            # matrix appears twice because the precision is a form on the
            # space, not on its components; dropping it is right only for an
            # orthonormal basis.
            precision_factor = LinearOperator.from_matrix(
                domain,
                coefficients,
                inverse_root @ domain.gram_matrix(),
                form="components",
            )
            # The palindrome rule gives Li* Li only semidefiniteness, but the
            # factorization earned more than that: it returned an inverse
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
            value = self._draw_through_factor(rng)
        else:
            raise NotImplementedError(
                "This measure has a covariance but no factor, so it cannot be "
                "sampled. Supply covariance_factor, or an explicit sample "
                "callable."
            )
        if self._expectation is None:
            return value
        return self._domain.add(value, self._expectation)

    def _draw_through_factor(self, rng: Generator | None) -> X:
        """``L xi`` for white noise ``xi``, in one transform when ``L`` is diagonal.

        A diagonal factor applies its eigenvalues to components, so ``L xi``
        is ``from_components(lambda * xi_c)`` with ``xi_c`` white noise's
        components, ``N(0, G^-1)``. Going through ``L`` as an operator
        synthesizes ``xi`` onto the grid only to analyze it again and
        synthesize the result: three spectral transforms for one draw.
        Because the short cut is keyed on the factor and not on a measure
        class, a scaled, summed, marginalized or conditioned measure whose
        factor is still diagonal draws in one transform too.
        """
        from ..algebra.diagonal import DiagonalLinearOperator

        factor = self._covariance_factor
        if isinstance(factor, DiagonalLinearOperator):
            space = factor.domain
            components = space.white_noise_components(rng=rng)
            return space.from_components(factor.eigenvalues * components)
        return factor(factor.domain.white_noise(rng=rng))

    # ----------------------------------------------------------------- #
    #                             Densities                             #
    # ----------------------------------------------------------------- #

    def _deviation(self, x: X) -> X:
        if self._expectation is None:
            return x
        return self._domain.subtract(x, self._expectation)

    def _diagonal_eigenvalues(self) -> np.ndarray | None:
        """The covariance's spectrum, when it is diagonal in the space's basis.

        Read off the operator's known scaling of components, which a diagonal
        operator, a scaled identity, their compositions and a block diagonal
        of them on a direct sum all report; such an operator lives only on a
        space whose basis is orthogonal, so its scaling is its spectrum.
        """
        covariance = self._covariance
        if covariance is None:
            return None
        scaling = covariance._known_scaling()
        if scaling is None:
            return None
        return np.asarray(scaling, dtype=float)

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
        from ..numerics.randomized import random_trace

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

        **Exact without the matrix.** ``"auto"`` reads the spectrum of a
        diagonal covariance or the slices of a spectrally block-diagonal one,
        takes ``sum(M * M.T)`` on a covariance that already holds its
        component matrix, and otherwise probes the diagonal of ``C C`` one
        basis vector at a time: two applications per column and linear
        memory. The dense matrix, ``N^2`` memory for a number that needs
        ``N``, is formed only when asked for by name.

        ``"stochastic"`` is a Hutchinson estimate of ``tr(C C)``. The
        estimator is a trace of ``C^2``, so its relative error is worse than a
        trace of ``C``: ask for more probes here than for
        :meth:`nuclear_norm`, or pass *rtol* and let it decide.

        Args:
            method: ``"auto"`` or ``"diagonal"`` is the exact matrix-free
                route above; ``"dense"`` forms the component matrix;
                ``"stochastic"`` estimates the trace with :func:`random_trace`.
            samples: probes for the stochastic route.
            rtol: draw further blocks of probes until the standard error is
                this fraction of the estimate.
            rng: the generator for the probes.
            n_jobs: workers for the probes, or for the diagonal probe.

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
        if method == "dense":
            matrix = covariance.matrix(form="components")
            return float(np.sqrt(np.sum(matrix * matrix.T)))
        slices = self._spectral_slices()
        if slices is not None:
            # tr(C^2) == sum over modes of tr(S_k^2), the blocks being diagonal.
            return float(np.sqrt(np.einsum("kij,kji->", slices, slices)))
        known = covariance.known_matrix(form="components")
        if known is not None:
            return float(np.sqrt(np.sum(known * known.T)))
        space = self._domain
        if not isinstance(space, CoordinateSpace):
            if method == "auto":
                return self.hilbert_schmidt_norm(
                    method="stochastic", samples=samples, rng=rng, n_jobs=n_jobs
                )
            require_coordinates(space)
        # The diagonal of C C, one basis vector at a time: (C C)_ii is the
        # i-th component of C(C(e_i)). Two applications per column and one
        # vector of memory.
        from ..parallel import parallel_map

        def entry(index: int) -> float:
            basis = space.basis_vector(index)
            return float(space.to_components(covariance(covariance(basis)))[index])

        squared = sum(parallel_map(entry, range(space.dim), n_jobs=n_jobs))
        return float(np.sqrt(max(squared, 0.0)))

    def _spectral_slices(self) -> np.ndarray | None:
        """The ``(dim, n, n)`` slices of a spectrally block-diagonal covariance.

        A correlated measure on several fields has one small matrix per mode
        (:class:`~pygeoinf2.symmetric_space.base.SpectralBlockLinearOperator`),
        from which both norms follow in ``O(dim n^2)``. Recognized by shape
        rather than by type, so that this module does not import the
        symmetric spaces.
        """
        slices = getattr(self._covariance, "slices", None)
        if slices is None:
            return None
        slices = np.asarray(slices)
        return slices if slices.ndim == 3 else None

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

        **Exact without the matrix.** ``"auto"`` reads the spectrum of a
        diagonal covariance or the slices of a spectrally block-diagonal one,
        and otherwise sums the component diagonal from :meth:`diagonals`,
        which is free where the operator knows it and one application per
        column where it must be probed, in linear memory. The dense matrix is
        formed only when asked for by name.

        Args:
            method: as for :meth:`hilbert_schmidt_norm`.
            samples: probes for the stochastic route.
            rtol: target relative standard error for the stochastic route.
            rng: the generator for the probes.
            n_jobs: workers for the probes, or for the diagonal probe.

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
        if method == "dense":
            return float(np.trace(covariance.matrix(form="components")))
        slices = self._spectral_slices()
        if slices is not None:
            return float(np.einsum("kii->", slices))
        diagonal = covariance.diagonals(offsets=(0,), form="components", n_jobs=n_jobs)
        return float(np.sum(diagonal[0]))

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
        """The covariance's Galerkin matrix, symmetrized against round-off."""
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
        sample_rtol: float | None = None,
        max_samples: int | None = None,
        n_jobs: int | None = None,
        rng: Generator | None = None,
        dense_limit: int = 4000,
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
            sample_rtol: target relative standard error for that route.
            max_samples: a ceiling on the probes it may draw.
            n_jobs: workers for the probes.
            rng: the generator for the probes.
            dense_limit: the largest dimension at which ``"auto"`` takes the
                dense route. Four thousand, as for
                :func:`~pygeoinf2.numerics.functional_calculus.log_determinant`;
                up to that size the dense route is both exact and faster than
                the stochastic one.
            kwargs: passed to the Lanczos functional calculus.

        Returns:
            The divergence.

        Raises:
            ValueError: as for :meth:`kl_divergence_estimate`.
        """
        return self.kl_divergence_estimate(
            other,
            method=method,
            solver=solver,
            samples=samples,
            sample_rtol=sample_rtol,
            max_samples=max_samples,
            n_jobs=n_jobs,
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
        sample_rtol: float | None = None,
        max_samples: int | None = None,
        n_jobs: int | None = None,
        rng: Generator | None = None,
        dense_limit: int = 4000,
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
            is a sum over the spectrum: ``O(dim)``, exact, and no factorization
            at all. Every invariant measure on a symmetric space is of this
            kind, so it is the common case and the one ``"auto"`` takes first.
        ``"dense"``
            Form both matrices and factorize. Exact, and confined to a space
            small enough to hold two of them.
        ``"stochastic"``
            One Hutchinson estimate of ``tr(g(M))`` with ``M`` similar to
            ``C_o^-1 C_s`` and ``g(t) == t - 1 - log t``, so nothing is ever
            formed. The route for a space too large to assemble, at the cost
            of an answer with an error bar. ``g`` is non-negative, so the
            estimate is too, and it vanishes probe by probe when the two
            measures agree.

            **It has to be asked for by name**, because a divergence reported
            as a bare number, with the error bar dropped, is the one way this
            quantity misleads. ``"auto"`` raises and says so.

        Args:
            other: the reference measure. The divergence is not symmetric.
            method: ``"auto"``, ``"spectral"``, ``"dense"`` or ``"stochastic"``.
                ``"auto"`` takes the spectral route when both covariances are
                diagonal, then the dense one when the space has coordinates and
                is no larger than *dense_limit*, and raises otherwise rather
                than reaching for the stochastic route unasked.
            solver: how to invert the reference covariance, for the stochastic
                route -- needed only when the reference has neither a precision
                nor a precision factor. Conjugate gradients by default; a
                factory taking the operator is also accepted.
            samples: Hutchinson probes.
            sample_rtol: draw further blocks of probes until the standard error
                is this fraction of the estimate.
            max_samples: a ceiling on that.
            n_jobs: workers for the probes.
            rng: the generator for those probes.
            dense_limit: the dimension above which ``"auto"`` stops forming
                matrices. Four thousand, as for
                :func:`~pygeoinf2.numerics.functional_calculus.log_determinant`;
                up to that size the dense route is both exact and faster than
                a hundred probes.
            kwargs: passed to
                :func:`~pygeoinf2.numerics.functional_calculus.operator_function`,
                which is where *max_iterations* and the Lanczos *rtol* live.

        Returns:
            An :class:`~pygeoinf2.numerics.randomized.Estimate`. The exact
            routes report a standard error of zero.

        Raises:
            ValueError: for an unknown method; if the two measures live on
                different spaces; under ``"auto"``, when only the stochastic
                route is available -- which is refused rather than taken
                silently; or if the stochastic route has no square root to
                work with, or cannot invert the reference accurately. An
                estimate whose standard error exceeds it is refused too,
                though the non-negativity of ``g`` makes that unreachable.
        """
        from ..numerics.randomized import Estimate, random_trace

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
                    "is not chosen on your behalf: it returns an estimate with "
                    "an error bar, and a divergence reported as a bare number "
                    "would be read as exact. Ask for it by name with "
                    "method='stochastic' if an estimate is what you want; "
                    "raise dense_limit if the space can afford two dense "
                    "matrices; or give both measures diagonal covariances, "
                    "which is the exact O(dim) route."
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

        # Stochastic: nothing is formed, and the three separately-estimated
        # terms are folded into one. See _kl_similarity for why.
        from ..numerics.functional_calculus import operator_function

        similar, inverse = self._kl_similarity(other, solver=solver)
        estimate = random_trace(
            operator_function(similar, _kl_kernel, **kwargs),
            samples=samples,
            rtol=sample_rtol,
            max_samples=max_samples,
            rng=rng,
            n_jobs=n_jobs,
        )
        quadratic = self._domain.inner_product(inverse(shift), shift)
        value = 0.5 * (estimate.value + quadratic)
        error = 0.5 * estimate.standard_error
        # An estimate whose error bar exceeds it says nothing. This route
        # cannot produce one: every probe
        # returns ``(x, g(M) x) >= 0`` because ``g >= 0``, and for a
        # non-negative sample ``sum x_i^2 <= (sum x_i)^2``, which is exactly
        # the statement that the standard error is at most the mean. The check
        # stays as the tripwire for that property rather than as a filter.
        if error > abs(value):
            raise ValueError(
                f"The estimate is {value:.3g} with a standard error of "
                f"{error:.3g}, which says nothing about the divergence. Draw "
                f"more probes (samples=), or pass sample_rtol= and let the "
                f"estimator decide when to stop."
            )
        return Estimate(float(value), float(error), estimate.samples)

    def _kl_similarity(
        self, other: "GaussianMeasure", /, *, solver: Any
    ) -> tuple[LinearOperator, LinearOperator]:
        r"""An operator with the spectrum of ``C_o^-1 C_s``, and ``C_o^-1``.

        The stochastic divergence is one trace, not three. Estimating
        ``tr(C_o^-1 C_s)`` and the two log-determinants separately and then
        combining them would subtract large, nearly equal, noisy numbers
        whenever the measures are close. With ``M == C_o^-1 C_s``,

        .. code-block:: text

            2 D == tr(M) - dim - log det M + (m_o - m_s, C_o^-1 (m_o - m_s))
                == tr(g(M)) + quadratic,   g(t) == t - 1 - log t

        and ``g`` is non-negative with a double zero at ``t == 1``. So the
        estimate is a trace of a positive operator and cannot come out
        negative; and for two equal measures ``M`` is the identity, ``g(M)``
        is exactly zero, and *every probe returns zero* — the cancellation
        happens inside each probe instead of between three estimates.

        ``M`` itself is not self-adjoint, but it is similar to something that
        is, and a trace only sees the spectrum. Two such:
        ``L_s* C_o^-1 L_s`` from this measure's covariance factor, and
        ``Li_o C_s Li_o*`` from the reference's precision factor. Both have
        the form ``B* A B`` with ``A`` self-adjoint, so the operator algebra
        derives their self-adjointness rather than having it claimed.

        A solve is needed only when the reference has neither a precision nor
        a precision factor. Then the accuracy of the answer is the accuracy of
        that solve, which on an ill-conditioned covariance can be much worse
        than its residual suggests, so the solver is tested on a round trip
        before it is trusted.

        Args:
            other: the reference measure.
            solver: how to invert the reference covariance, if it comes to
                that.

        Returns:
            ``(M', C_o^-1)`` with ``M'`` self-adjoint and similar to
            ``C_o^-1 C_s``.

        Raises:
            ValueError: if neither similarity is available; if the reference
                covariance must be inverted but does not claim positive
                definiteness; or if the solver does not invert it accurately.
        """
        from ..numerics.solvers import resolve_solver

        if other.precision is not None:
            inverse = other.precision
        elif other.precision_factor is not None:
            inverse = other.precision_factor.adjoint @ other.precision_factor
        else:
            reference = other._require_covariance("A stochastic divergence")
            if Traits.POSITIVE_DEFINITE not in reference.traits:
                raise ValueError(
                    "The reference covariance has to be inverted, and it does "
                    "not claim POSITIVE_DEFINITE. Attach the trait with "
                    "with_traits() once testing.check_traits() has confirmed "
                    "it, or supply the measure's precision -- which is the "
                    "cheaper route anyway, since it removes the solve."
                )
            inverse = resolve_solver(solver, reference)(reference)
            _check_round_trip(reference, inverse)

        if self._covariance_factor is not None:
            factor = self._covariance_factor
            return factor.adjoint @ inverse @ factor, inverse
        if other.precision_factor is not None:
            root = other.precision_factor
            covariance = self._require_covariance("A stochastic divergence")
            return root @ covariance @ root.adjoint, inverse
        raise ValueError(
            "The stochastic route needs a square root of one of the two "
            "covariances -- this measure's covariance factor, or the "
            "reference's precision factor -- because the operator whose "
            "spectrum it needs is only self-adjoint when written around one. "
            "Supply covariance_factor here, or precision_factor there, or use "
            "an exact route."
        )

    def rescale_directional_variance(
        self, direction: X, standard_deviation: float, /
    ) -> "GaussianMeasure[X]":
        """The same measure with ``Var((x, direction))`` set to a given value.

        A whole-measure rescaling, not a rank-one update: the covariance is
        multiplied by a scalar chosen so that one direction comes out right.
        Every other variance moves by the same factor, which is what makes it
        a recalibration rather than a change of shape.

        Args:
            direction: the vector whose reading is recalibrated.
            standard_deviation: the standard deviation the reading is to
                have; positive.

        Returns:
            The rescaled measure, of the same class.

        Raises:
            ValueError: for a non-positive target, or for a direction in
                which this measure has no variance, there being no factor
                that would give it one.
        """
        if standard_deviation <= 0.0:
            raise ValueError(
                f"The target standard deviation must be positive, got "
                f"{standard_deviation}."
            )
        current = self.directional_variance(direction)
        if current <= 0.0:
            raise ValueError(
                "The variance along this direction is not positive, so it "
                "cannot be rescaled."
            )
        factor = float(standard_deviation) / np.sqrt(current)
        centered = self.translate(self._domain.negative(self.expectation))
        return (factor * centered).translate(self.expectation)

    def low_rank_approximation(
        self,
        /,
        *,
        rank: int | None = None,
        rng: Generator | None = None,
        **kwargs: Any,
    ) -> "GaussianMeasure[X]":
        """The same expectation, with the covariance truncated to *rank* modes.

        Obtained as a randomized Cholesky factorization ``C ~ L L*``, so the
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
            **kwargs: passed to the randomized routine.

        Returns:
            A measure whose covariance has that rank, and which can be
            sampled -- the factor comes out of the same decomposition.

        Raises:
            ValueError: for a rank above the dimension, or a measure with no
                covariance to approximate.
        """
        from ..numerics.randomized import random_cholesky

        covariance = self._covariance
        if covariance is None:
            raise ValueError(
                "A low-rank approximation needs the covariance, and this "
                "measure was given only a precision."
            )
        factorized = random_cholesky(
            covariance.with_traits(Traits.POSITIVE_SEMIDEFINITE),
            rank=rank,
            rng=rng,
            **kwargs,
        )
        return GaussianMeasure(
            self._domain,
            expectation=self._expectation,
            covariance_factor=factorized.factor,
        )

    def with_dense_covariance(
        self, /, *, n_jobs: int | None = None
    ) -> "GaussianMeasure[X]":
        """The same measure with its covariance assembled as a dense matrix.

        The covariance of a measure built by the algebra -- a pushforward, a
        posterior, a sum -- is a graph of operators, and applying it costs
        whatever that graph costs. Once it is going to be applied many times,
        or once something wants to look at it, assembling it once is cheaper:
        this is what a plotting script does before drawing a variance field.

        The assembly is ``dim`` applications of the covariance, run in
        parallel when *n_jobs* asks. The result carries a matrix-backed
        covariance, and the factor and precision factor that come out of
        factorizing it -- so the returned measure can be sampled and has a
        density even when this one could do neither.

        The dense measure is kept, so a second call is free and a later
        *n_jobs* is ignored.

        Args:
            n_jobs: workers for the ``dim`` applications.

        Returns:
            A measure with the same law and a matrix-backed covariance.

        Raises:
            TypeError: if the space has no coordinates to assemble one in.
            ValueError: if this measure has no covariance operator.
        """
        if self._dense is not None:
            return self._dense

        require_coordinates(self._domain)
        covariance = self._require_covariance("A dense covariance")
        galerkin = covariance.matrix(form="galerkin", n_jobs=n_jobs)
        symmetric = 0.5 * (galerkin + galerkin.T)

        # The factors come from the same construction as
        # from_covariance_matrix, which accepts a semidefinite matrix and
        # attaches a precision when there is one to attach. The covariance
        # itself is then replaced by the matrix rather than left as ``L L*``:
        # one product instead of two, and a matrix the operator algebra can
        # read straight off.
        factored = GaussianMeasure.from_covariance_matrix(
            self._domain, symmetric, form="galerkin", expectation=self._expectation
        )
        dense = LinearOperator.from_matrix(
            self._domain,
            self._domain,
            symmetric,
            traits=_REQUIRED,
            form="galerkin",
        )
        self._dense = self._rebuild(
            self._domain,
            expectation=self._expectation,
            covariance=dense,
            covariance_factor=factored.covariance_factor,
            precision=factored.precision,
            precision_factor=factored.precision_factor,
        )
        return self._dense

    def sparse_covariance(self, /, **options: Any) -> LinearOperator:
        """A sparse operator approximating the covariance, built matrix-free.

        :func:`~pygeoinf2.numerics.preconditioners.sparse_approximation` of
        the covariance, with its options; see there. An operator and not a
        measure: thresholding does not keep a covariance positive definite,
        and a sparse matrix cannot be sampled from without a sparse Cholesky,
        which SciPy does not have.

        Args:
            **options: ``threshold``, ``max_per_column``, ``criterion`` and
                ``diagonal``, as for the function.

        Returns:
            The sparse operator on the space.

        Raises:
            ValueError: for a measure with no covariance, or a bad option.
        """
        from ..numerics.preconditioners import sparse_approximation

        return sparse_approximation(
            self._require_covariance("A sparse approximation"), **options
        )

    def credible_set(
        self, /, *, level: float = 0.95, solver: Any = None, unsafe: bool = False
    ) -> "Ellipsoid":
        """The region carrying a given share of the probability, as a set.

        One **hardening** of the measure: a set chosen to carry a given
        probability, here the Mahalanobis ellipsoid at the chi-squared
        quantile of *level*. A hardening is not canonical and it is not
        reversible -- the ellipsoid carries no memory of the distribution it
        came from -- which is why it is a named step rather than something a
        constructor does quietly. :meth:`ambient_ball`, in the space's norm
        or in the norm an operator defines, is the other.

        An ellipsoid is defined by a precision. A measure that has one uses
        it. A measure with only a covariance gets one as the covariance's
        inverse *through a solver*, conjugate gradients by default: nothing is
        assembled, and each membership test or support value then costs one
        solve, which is what the question costs on a space too large to
        invert. Pass a direct solver, ``CholeskySolver()``, to factorize the
        covariance once on a space small enough to hold it; that is the dense
        route, and it is taken only by name.

        Args:
            level: the probability the region carries, in ``(0, 1)``.
            solver: how to invert the covariance when the measure has no
                precision -- a linear solver, or a callable taking the
                covariance and returning one. Ignored when a precision exists.
            unsafe: proceed even when the domain discretises a function
                space (``domain.is_function_space``), where the set is an
                artefact of the truncation.

        Returns:
            The credible ellipsoid.

        Raises:
            ValueError: for a level outside ``(0, 1)``, a measure with
                neither a precision nor a covariance, or a domain that
                discretises a function space unless ``unsafe=True``: the
                chi-squared threshold grows with the dimension, so the set
                is an artefact of the truncation.
        """
        _require_density(self._domain, "A credible set", unsafe=unsafe)
        from scipy.stats import chi2

        from ..geometry.convex import Ellipsoid

        if not 0.0 < level < 1.0:
            raise ValueError(f"A credible level lies in (0, 1), got {level}.")
        threshold = float(chi2.ppf(level, self.domain.dim))
        # The ellipsoid is defined by its precision; the covariance only rides
        # along for whoever wants it. So a measure described by a precision
        # alone has a perfectly good credible set.
        covariance = None if self._covariance is None else self._covariance * threshold
        factor = (
            None
            if self._covariance_factor is None
            else self._covariance_factor * float(np.sqrt(threshold))
        )
        if self._precision is not None:
            precision = self._precision * (1.0 / threshold)
        else:
            # A measure built from a covariance alone has no precision, and an
            # ellipsoid is defined by one: the covariance's inverse, through
            # a solver, is an operator in the space's own metric by
            # construction. A dense inverse of the Galerkin matrix would not
            # be without two corrections by G.
            from ..numerics.solvers import resolve_solver

            definite = self._require_covariance("A credible set").with_traits(
                Traits.POSITIVE_DEFINITE
            )
            precision = resolve_solver(solver, definite)(definite) * (1.0 / threshold)
        return Ellipsoid(
            self.domain,
            precision,
            center=self.expectation,
            covariance=covariance,
            factor=factor,
        )

    def sample_pointwise_variance(
        self,
        samples: int,
        /,
        *,
        rng: Generator | None = None,
        n_jobs: int | None = None,
    ) -> X:
        """The pointwise variance field, estimated from draws.

        The mean over draws of ``(x - m) * (x - m)`` in the domain's pointwise
        product, so it needs a domain that is a module. The general answer
        on any space; on a symmetric space with an invariant measure,
        :meth:`~pygeoinf2.symmetric_space.SymmetricSpace.pointwise_variance`
        is the exact one and costs no draws.

        Args:
            samples: how many draws.
            rng: the generator.
            n_jobs: workers for the draws.

        Returns:
            A field of the domain.

        Raises:
            TypeError: if the domain has no pointwise product.
            ValueError: for fewer than one draw.
        """
        from ..algebra.spaces import HilbertModule

        space = self._domain
        if not isinstance(space, HilbertModule):
            raise TypeError(
                "A pointwise variance needs a domain with a pointwise product, "
                f"a HilbertModule; {type(space).__name__} is not one."
            )
        if samples < 1:
            raise ValueError("At least one draw is needed.")
        center = self.expectation
        total = space.zero()
        for draw in self.samples(samples, rng=rng, n_jobs=n_jobs):
            deviation = space.subtract(draw, center)
            total = space.axpy(1.0, space.multiply(deviation, deviation), total)
        return space.scale(1.0 / samples, total)

    def sample_pointwise_std(
        self,
        samples: int,
        /,
        *,
        rng: Generator | None = None,
        n_jobs: int | None = None,
    ) -> X:
        """The pointwise standard deviation field, the square root of the variance.

        Args:
            samples: how many draws.
            rng: the generator.
            n_jobs: workers for the draws.

        Returns:
            A field of the domain.
        """
        variance = self.sample_pointwise_variance(samples, rng=rng, n_jobs=n_jobs)
        return self._domain.sqrt(variance)

    def ambient_ball(
        self,
        /,
        *,
        level: float = 0.95,
        operator: LinearOperator | None = None,
        inverse: LinearOperator | None = None,
        method: str = "auto",
        quantile_method: str = "auto",
        rank: int | None = None,
        samples: int = 10_000,
        dense_limit: int = 1024,
        rng: Generator | None = None,
        n_jobs: int | None = None,
    ) -> Any:
        """The smallest ball about the mean carrying a given probability.

        A different hardening from :meth:`credible_set`, and a cruder one: that
        gives the ellipsoid the distribution's own shape, this gives a ball in
        the *space's* norm, or, given an *operator*, in the norm that operator
        defines. The credible set and the ball coincide only for an isotropic
        measure, and the ball is always the larger.

        It is worth having because a norm bound is what a set-theoretic prior
        is, so this is the bridge from a Gaussian belief to one, done in the
        geometry the constraint will be used in. It is also what
        :func:`~pygeoinf2.inference.backus.harden_error` calls on every Backus
        route whose error is a Gaussian, which is why the route taken here
        matters. On a space discretizing a function space the ball converges
        under refinement, the squared norm of a draw having a finite
        expectation whenever the covariance is trace class, which is what the
        credible set lacks there.

        With an *operator* ``B``, bounded, self-adjoint and positive definite
        on the space, the set is ``{ x : (B (x - m), x - m) <= r^2 }``, the
        ball of the inner product ``B`` defines: a constraint set of another
        shape than the space's own, such as the ``L^2`` ball of a Sobolev
        space, or a bound that weights one region or one component of a
        direct sum more than another. Because ``B`` is bounded and the
        covariance trace class, ``(B (x - m), x - m)`` has a finite
        expectation and the radius converges as the ball's does. The set is
        then an :class:`~pygeoinf2.geometry.convex.Ellipsoid`, which has a
        support function, and so serves a set estimator, only when its
        covariance is known: pass ``B``'s *inverse* for that.

        The radius is a quantile of ``sum_i lambda_i Z_i^2`` with the
        covariance's eigenvalues as weights, or those of ``B C`` with an
        operator, which is not a chi-square unless the measure is isotropic.
        Four ways to reach it:

        ``"diagonal"``
            The spectrum is already the covariance's eigenvalues, and the
            operator's if there is one. Exact, ``O(dim)``, no applications at
            all. This is the common case -- an isotropic error, a measure
            from :meth:`from_standard_deviations`, any invariant measure on a
            symmetric space, with an operator that is a spectral function
            there.
        ``"dense"``
            The generalized symmetric eigenproblem ``C_gal v == lambda G v``,
            whose eigenvalues are the operator's; with an operator, the
            eigenvalues of ``C^(1/2) B C^(1/2)`` from the matrices. Exact,
            ``O(dim^3)``, and it holds dense matrices.
        ``"spectral"``
            A randomized eigendecomposition truncated to *rank*, of the
            covariance, or of ``L* B L`` through the covariance factor. The
            dropped tail is positive, so the radius comes out **too small**:
            this is a lower bound on the true one, and a rank has to be large
            enough that what it leaves out is negligible.
        ``"sampling"``
            The empirical quantile of ``||x - m||^2``, or of ``(B (x - m),
            x - m)``, over draws. No spectrum at all, and the accuracy is that
            of an order statistic on *samples* draws.

        Args:
            level: the probability the ball carries, in ``(0, 1)``.
            operator: a bounded self-adjoint positive definite operator on
                the space whose inner product the ball is measured in; the
                space's own when omitted.
            inverse: the operator's inverse, which gives the ellipsoid its
                covariance and so its support function. Ignored without an
                operator.
            method: ``"auto"``, ``"diagonal"``, ``"dense"``, ``"spectral"`` or
                ``"sampling"``. ``"auto"`` reads a diagonal spectrum if there
                is one, then goes dense if the space has coordinates and is no
                larger than *dense_limit*, then samples if the measure can be
                sampled, then takes the randomized spectrum if a *rank* was
                given, and otherwise says which of those to supply.
            quantile_method: how to invert the weighted chi-squared -- see
                :func:`~pygeoinf2.numerics.quadratic_forms.weighted_chi2_quantile`.
            rank: eigenpairs to keep on the randomized route.
            samples: draws for the sampling route.
            dense_limit: the dimension above which ``"auto"`` stops forming
                matrices.
            rng: the generator for the probes or the draws.
            n_jobs: workers for them.

        Returns:
            The ball about the expectation carrying probability *level*; with
            an operator, the ellipsoid.

        Raises:
            ValueError: for an unknown method, a method whose ingredients this
                measure does not have, a measure with no covariance, or an
                operator that is not a self-adjoint positive definite
                operator on the space.
        """
        from ..geometry.convex import Ball, Ellipsoid

        if operator is not None:
            required = Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
            if operator.domain != self._domain or operator.codomain != self._domain:
                raise ValueError(
                    f"The operator must act on {self._domain!r}; it maps "
                    f"{operator.domain!r} to {operator.codomain!r}."
                )
            if required & operator.traits != required:
                raise ValueError(
                    f"The operator must claim {required!s}; it claims "
                    f"{operator.traits!s}."
                )
            if inverse is not None and (
                inverse.domain != self._domain or inverse.codomain != self._domain
            ):
                raise ValueError(f"The inverse must act on {self._domain!r}.")

        def finish(squared_radius: float) -> Any:
            radius = float(np.sqrt(max(squared_radius, 0.0)))
            if operator is None:
                return Ball(self._domain, radius=radius, center=self.expectation)
            if radius <= 0.0:
                raise ValueError("The ellipsoid's radius came out zero.")
            precision = (operator / radius**2).with_traits(required)
            covariance = (
                None if inverse is None else (inverse * radius**2).with_traits(required)
            )
            return Ellipsoid(
                self._domain,
                precision,
                center=self.expectation,
                covariance=covariance,
            )

        def from_spectrum(eigenvalues: np.ndarray) -> Any:
            return finish(
                self._squared_radius_from_spectrum(eigenvalues, level, quantile_method)
            )

        eigenvalues = self._diagonal_eigenvalues()
        if eigenvalues is not None and operator is not None:
            scaling = operator._known_scaling()
            eigenvalues = None if scaling is None else eigenvalues * np.asarray(scaling)
        if eigenvalues is not None and method in ("auto", "diagonal"):
            return from_spectrum(eigenvalues)
        if method == "diagonal":
            raise ValueError(
                "The diagonal route needs a covariance diagonal in the space's "
                "own basis, and an operator diagonal there too, and this one "
                "is not."
            )

        if method == "auto":
            affordable = (
                isinstance(self._domain, CoordinateSpace)
                and self._domain.dim <= dense_limit
            )
            if affordable:
                method = "dense"
            elif self.can_sample:
                method = "sampling"
            elif rank is not None:
                method = "spectral"
            else:
                raise ValueError(
                    f"No affordable route to an ambient ball on a space of "
                    f"dimension {self._domain.dim}: forming the covariance "
                    f"and taking its eigenvalues is cubic, and the two "
                    f"matrix-free routes need something this measure does not "
                    f"have. Give it a covariance factor so it can be sampled, "
                    f"pass rank= for a randomized spectrum, or raise "
                    f"dense_limit if the space can afford two dense matrices."
                )

        if method == "sampling":
            if not self.can_sample:
                raise ValueError(
                    "The sampling route needs a measure that can be drawn "
                    "from; supply a covariance factor or a sample callable."
                )
            center = self.expectation
            draws = self.samples(samples, rng=rng, n_jobs=n_jobs)
            if operator is None:
                squared = [
                    self._domain.squared_norm(self._domain.subtract(x, center))
                    for x in draws
                ]
            else:
                squared = []
                for x in draws:
                    offset = self._domain.subtract(x, center)
                    squared.append(self._domain.inner_product(operator(offset), offset))
            return finish(float(np.quantile(np.asarray(squared), level)))

        covariance = self._require_covariance("An ambient ball")
        if method == "spectral":
            from ..numerics.randomized import random_eig

            if operator is None:
                target = covariance
            else:
                factor = self.covariance_factor
                if factor is None:
                    raise ValueError(
                        "The spectral route with an operator needs a covariance "
                        "factor, to form L* B L."
                    )
                target = (factor.adjoint @ operator @ factor).with_traits(
                    Traits.SELF_ADJOINT | Traits.POSITIVE_SEMIDEFINITE
                )
            decomposition = random_eig(target, rank=rank, rng=rng, n_jobs=n_jobs)
            return from_spectrum(np.asarray(decomposition.eigenvalues, dtype=float))
        if method != "dense":
            raise ValueError(f"Unknown method {method!r}.")

        from scipy.linalg import eigh

        require_coordinates(self._domain)
        # The eigenvalues of the *operator* are those of C_c == G^-1 C_gal,
        # which is not symmetric on a space whose basis is not orthonormal.
        # They are the generalized eigenvalues of the symmetric pair
        # (C_gal, G), which is both cheaper and better conditioned than a
        # non-symmetric decomposition of C_c.
        galerkin = covariance.matrix(form="galerkin")
        galerkin = 0.5 * (galerkin + galerkin.T)
        gram = self._domain.gram_matrix()
        if operator is None:
            spectrum = eigh(galerkin, gram, eigvals_only=True)
            return from_spectrum(np.asarray(spectrum, dtype=float))
        # With an operator the weights are the eigenvalues of B C, which are
        # those of the symmetric C^(1/2) B C^(1/2): in Galerkin form, with
        # S the symmetric root of C_gal, the matrix S G^-1 B_gal G^-1 S.
        values, vectors = eigh(galerkin)
        root = (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.T
        from scipy.linalg import solve

        inner = solve(gram, operator.matrix(form="galerkin"), assume_a="pos")
        inner = solve(gram, inner.T, assume_a="pos").T
        symmetric = root @ inner @ root
        spectrum = eigh(0.5 * (symmetric + symmetric.T), eigvals_only=True)
        return from_spectrum(np.asarray(spectrum, dtype=float))

    @staticmethod
    def _squared_radius_from_spectrum(
        eigenvalues: np.ndarray, level: float, quantile_method: str, /
    ) -> float:
        """The *level* quantile of ``sum lambda_i Z_i^2``.

        Args:
            eigenvalues: the weights, or as many of them as are known.
            level: the probability the set carries.
            quantile_method: how to invert the weighted chi-squared.

        Returns:
            The squared radius.
        """
        from ..numerics.quadratic_forms import weighted_chi2_quantile

        weights = np.clip(np.asarray(eigenvalues, dtype=float), 0.0, None)
        if quantile_method == "auto":
            quantile_method = GaussianMeasure._quantile_method_for(weights)
        return float(weighted_chi2_quantile(weights, level, method=quantile_method))

    @staticmethod
    def _quantile_method_for(weights: np.ndarray, /) -> str:
        """Imhof or the moment-matched chi-square, decided by the spectrum.

        Imhof's integrand sums over every weight at every quadrature point, so
        inverting it costs time linear in the length of the spectrum *and* in
        the root find, which is unaffordable for the spectra of size ``10^5``
        a large inverse problem produces.

        A sum of many comparable weighted chi-squares is close to its
        moment-matched chi-square, by the same argument that makes it close to
        a normal, and the agreement improves as the sum lengthens. A spectrum
        dominated by a few modes is a different object -- one chi-square with
        one degree of freedom is not well matched by its own moments -- so the
        effective rank decides, not the length alone: ``"matched"`` above a
        thousand positive weights with effective rank
        ``(sum w)^2 / sum w^2`` above a hundred, ``"imhof"`` otherwise.

        Args:
            weights: the non-negative spectrum.

        Returns:
            ``"matched"`` or ``"imhof"``.
        """
        live = weights[weights > 0.0]
        if live.size <= 1000:
            return "imhof"
        effective = float(np.sum(live) ** 2 / np.sum(live**2))
        return "matched" if effective > 100.0 else "imhof"

    def as_multivariate_normal(self, /, *, n_jobs: int | None = None) -> Any:
        """The measure as a ``scipy.stats`` object, in components.

        For anything scipy already does — a density, a rank correlation, a
        statistical test. It is the *component* representation, so it is a
        statement about coefficients rather than about fields, and a metric
        that is not the identity does not travel with it.

        Args:
            n_jobs: workers for assembling the covariance matrix, one
                application per column. Serial by default.
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
        # right on the first and wrong on the second.
        def divide(matrix: np.ndarray) -> np.ndarray:
            return np.column_stack(
                [self._domain.solve_gram(column) for column in matrix.T]
            )

        galerkin = self._require_covariance("A multivariate normal").matrix(
            form="galerkin", n_jobs=n_jobs
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
        draw and one solve, with no factorization of the posterior covariance.
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
                is also accepted.

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
        # (I - C A* N^-1 A) C, not C - C A* N^-1 A C: the same operator, with
        # C applied twice per action instead of three times. Invisible for a
        # diagonal covariance and a third of the cost for any other.
        identity = LinearOperator.identity(self._domain)
        updated = (identity - cross @ inverse @ operator) @ covariance

        sample = None
        if self.can_sample and (noise is None or noise.can_sample):

            def sample(rng: Generator | None, _noise=noise) -> Any:
                """A *centered* posterior draw: ``(I - K A) dx - K de``.

                Centered because :meth:`sample` adds the expectation itself, and
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
        # eigenvalues -- so the marginal can still be sampled. Two conditions,
        # not one.
        #
        # The general route cannot supply a factor at all: if C == L L*, the
        # (i, i) block of C is a sum over the whole i-th row of L, not
        # L_ii L_ii*.
        #
        # And a diagonal square root is a square root only when the summand's
        # metric is diagonal. ``L`` with component matrix ``diag(r)`` has
        # adjoint ``G^-1 diag(r) G``, so ``L L*`` has component matrix
        # ``diag(r) G^-1 diag(r) G``, which is ``diag(r^2)`` exactly when the
        # two commute -- that is, when ``G`` is itself diagonal. Without this
        # test the marginal comes back with a factor that does not match its
        # covariance, and it is the *sampler* that is wrong, silently.
        factor = None
        if (
            isinstance(block, DiagonalLinearOperator)
            and block.domain.has_diagonal_metric
        ):
            factor = block.sqrt
        # And under the same condition the precision is the reciprocal
        # spectrum, when every variance is positive, so the marginal has a
        # density.
        precision = None
        if factor is not None and np.all(block.eigenvalues > 0.0):
            precision = DiagonalLinearOperator(block.domain, 1.0 / block.eigenvalues)
        return GaussianMeasure(
            block.domain,
            covariance=block,
            covariance_factor=factor,
            precision=precision,
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
        block = domain.projection(first) @ operator @ domain.projection(second).adjoint
        if left != right:
            return block
        # A diagonal block is a congruence of the operator, and inherits its
        # traits -- stated here as push_forward states them, because the
        # palindrome rule cannot see it: the composition flattens the
        # operator into its own factors, and a covariance whose traits were
        # claimed rather than derived (a posterior's) loses them on the way.
        # Without this the marginal of a posterior on a direct sum would claim
        # nothing, and the measure constructor would refuse it.
        return block.with_traits(
            congruence_traits(operator.traits, outer_invertible=False)
        )

    def mahalanobis_squared(self, x: X, *, unsafe: bool = False) -> float:
        """``(x - m, P (x - m))``, the squared Mahalanobis distance.

        Needs a precision, and says so rather than falling back to a dense
        solve. The fallback exists — :meth:`_weighted_squared` has it — but
        reaching it silently would mean a cubic cost incurred by a method that
        looks like a quadratic form. A measure whose covariance has no usable
        inverse has no Mahalanobis distance either; build one with a damped
        covariance, ``C + damping I``, and its inverse as the precision if an
        approximation is what you want.

        Args:
            x: the vector to measure.
            unsafe: proceed even when the domain discretises a function
                space (``domain.is_function_space``), where the value is an
                artefact of the truncation.

        Returns:
            The squared distance.

        Raises:
            NotImplementedError: if the measure has no precision.
            ValueError: on a domain that discretises a function space, unless
                ``unsafe=True``.
        """
        _require_density(self._domain, "A Mahalanobis distance", unsafe=unsafe)
        if self._precision is None:
            raise NotImplementedError(
                "This measure has no precision, so no Mahalanobis distance. "
                "Supply precision or precision_factor at construction."
            )
        deviation = self._deviation(x)
        return self._domain.inner_product(self._precision(deviation), deviation)

    def log_normalizing_constant(self, *, unsafe: bool = False) -> float:
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

        Args:
            unsafe: proceed even when the domain discretises a function
                space (``domain.is_function_space``), where the constant
                diverges with the dimension.

        Returns:
            The additive constant, in nats.

        Raises:
            NotImplementedError: if the measure has no covariance, so there is
                no determinant to take.
            ValueError: on a domain that discretises a function space, unless
                ``unsafe=True``.
        """
        _require_density(self._domain, "A normalizing constant", unsafe=unsafe)
        if self._log_normalization is None:
            self._log_normalization = self._compute_log_normalization()
        return self._log_normalization

    def _compute_log_normalization(self) -> float:
        """The constant, exactly for a diagonal covariance and otherwise not."""
        from ..numerics.functional_calculus import log_determinant

        gaussian = -0.5 * self._domain.dim * float(np.log(2.0 * np.pi))

        eigenvalues = self._diagonal_eigenvalues()
        if eigenvalues is not None:
            # Exact and cheap, and needs no positive-definiteness claim,
            # unlike the general route below.
            return gaussian - 0.5 * float(np.sum(np.log(eigenvalues)))

        if self._covariance is None:
            raise NotImplementedError(
                "This measure has no covariance, so no normalizing constant. "
                "Supply covariance or covariance_factor."
            )
        definite = self._covariance.with_traits(
            Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
        )
        return gaussian - 0.5 * log_determinant(definite).value

    def log_density(self, x: X, *, unsafe: bool = False) -> float:
        """The log density up to an additive constant.

        The constant is :meth:`log_normalizing_constant`, which depends on the
        covariance. Differences of this quantity are meaningful only *within*
        one measure; comparing two measures means adding each one's constant.

        Refused on a domain that discretises a function space unless
        ``unsafe=True``: the value depends on the truncation. The quantity
        that does not is a density relative to a reference measure,
        :class:`~pygeoinf2.probability.AbsolutelyContinuousMeasure`.

        Args:
            x: where to evaluate it.
            unsafe: proceed even when the domain discretises a function
                space (``domain.is_function_space``).

        Returns:
            ``-(x - m, P (x - m)) / 2``.

        Raises:
            NotImplementedError: if the measure has no precision.
            ValueError: on a domain that discretises a function space, unless
                ``unsafe=True``.
        """
        _require_density(self._domain, "A log density", unsafe=unsafe)
        return -0.5 * self.mahalanobis_squared(x, unsafe=True)

    def grad_log_density(self, x: X, *, unsafe: bool = False) -> X:
        """``-P (x - m)``, already a vector in the domain.

        No Riesz map is applied here because none is needed: the precision maps
        the space to itself, so its output is already the gradient, not a
        derivative.

        Args:
            x: where to evaluate it.
            unsafe: proceed even when the domain discretises a function
                space (``domain.is_function_space``).

        Returns:
            The gradient, a vector of the domain.

        Raises:
            NotImplementedError: if the measure has no precision, the
                gradient being ``-P (x - m)``.
            ValueError: on a domain that discretises a function space, unless
                ``unsafe=True``.
        """
        _require_density(self._domain, "A log-density gradient", unsafe=unsafe)
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

        Every algebraic operation goes through here, so the precision and its
        factor are passed through: an operation that preserves them (a
        translation, an invertible diagonal map, a scaling) keeps the density
        and everything built on the precision.
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
        diagonal. The functional calculus of a diagonalizable operator does not
        depend on the metric, so the eigenvalues are the honest thing to gate
        on.
        """
        from ..algebra.diagonal import DiagonalLinearOperator

        if not isinstance(covariance, DiagonalLinearOperator):
            return None, None
        values = np.asarray(covariance.eigenvalues, dtype=float)
        if np.any(values < 0.0):
            return None, None
        space = covariance.domain
        # A square root exists for any non-negative spectrum; a precision only
        # for a positive one. Gating both on positivity would lose the factor,
        # and with it sampling, for every sum involving a band-limited measure,
        # whose spectrum has zeros.
        factor = DiagonalLinearOperator(space, np.sqrt(values))
        precision = (
            DiagonalLinearOperator(space, 1.0 / values)
            if np.all(values > 0.0)
            else None
        )
        return factor, precision

    def _combine_affine(
        self, operator: LinearOperator, translation: X | None
    ) -> GaussianMeasure[X] | None:
        """A Gaussian stays Gaussian under an affine map.

        ``None`` for a precision-only measure under a map that does not carry
        the precision across, where the base class's push-forward takes over.
        """
        codomain = operator.codomain
        mean = operator(self.expectation)
        if translation is not None:
            mean = codomain.add(mean, translation)

        factor = (
            None
            if self._covariance_factor is None
            else operator @ self._covariance_factor
        )
        covariance = None
        if factor is None and self._covariance is not None:
            # The congruence rule, claimed here rather than left to the
            # composition: the palindrome is visible to the trait algebra only
            # when the covariance is a single factor, and a covariance written
            # as a product -- ``(I - K A) Q``, the posterior's -- is not.
            covariance = (operator @ self._covariance @ operator.adjoint).with_traits(
                congruence_traits(self._covariance.traits, outer_invertible=False)
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
          does not touch the covariance, and it is the commonest of these
          operations.
        * an invertible diagonal operator, which is what masking or rescaling a
          field by a spectral multiplier looks like.

        Anything else gives up, because inverting a general operator is a solve
        and doing one silently inside an algebraic operation is exactly the
        kind of hidden cost the library is trying not to have.
        """
        from ..algebra.diagonal import DiagonalLinearOperator

        if operator.is_identity:
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

        Works from whichever of the two the measure has, so a measure
        described by a covariance and a precision -- every conditioned or
        explicitly built one -- stays a Gaussian under a scalar multiple.
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
