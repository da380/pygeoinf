"""
The posterior of a linear problem with a Gaussian prior.

The Kalman update, assembled in the data space by default or in the model
space on request. With prior covariance ``Q`` and data covariance ``R``:

.. code-block:: text

    data space   N = A Q A* + R          K = Q A* N^-1
    model space  N = Q^-1 + A* R^-1 A    K = N^-1 A* R^-1

and the posterior is ``mu + K (d - A mu - mu_e)`` with covariance ``Q - K A Q``
— which in the model-space formalism is just ``N^-1``, so nothing extra is
formed there. That algebra lives on
:class:`~pygeoinf2.numerics.normal.NormalOperator`, which is also what the
class exposes so that preconditioners can be built against it.

The name says what the class does and what it needs. It is *linear* — the
forward operator is a :class:`LinearOperator`, not merely differentiable — and
it is *Gaussian* — the prior is a
:class:`~pygeoinf2.probability.gaussian.GaussianMeasure`, not any probability
measure, because the update below is the closed-form conjugate one and there is
no other case it covers. Bayesian inference in general is neither.

The estimator is a :class:`~pygeoinf2.inference.estimators.GaussianEstimator`,
a mapping from data to posterior measures: the covariance does not depend on
the data, so the mapping is a pair and only the mean moves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.random import Generator

from ..algebra.operators import AffineOperator, LinearOperator
from ..geometry.sets import Subset
from ..numerics.normal import Formalism, NormalOperator
from ..numerics.randomized import random_svd
from ..numerics.solvers import LinearSolver, resolve_solver
from ..probability.absolutely_continuous import (
    AbsolutelyContinuousMeasure,
    combine_log_normalizer,
)
from ..probability.base import ProbabilityMeasure
from ..probability.gaussian import GaussianMeasure
from ..traits import Traits
from .estimators import GaussianEstimator
from .problem import LinearForwardProblem

if TYPE_CHECKING:  # pragma: no cover
    from ..algebra.operators import Functional
    from ..algebra.spaces import HilbertSpace
    from ..numerics.randomized import Estimate

__all__ = ["GaussianPosterior", "LinearGaussianInversion"]


class GaussianPosterior(GaussianMeasure, AbsolutelyContinuousMeasure):
    """A Gaussian that also knows its density relative to a reference.

    What a Bayesian update returns: a ``GaussianMeasure`` first -- drawn from,
    pushed forward, conditioned exactly as any other -- and an
    :class:`~pygeoinf2.probability.absolutely_continuous.AbsolutelyContinuousMeasure`
    second, carrying the prior it came from, the log density ``-Phi`` against
    it (``Phi`` being the likelihood potential, half the chi-squared misfit),
    and the evidence as its normalizer. For a *linear* problem the two
    readings describe the same measure exactly. For a Laplace approximation
    (:meth:`~pygeoinf2.inference.laplace.LaplaceApproximation.at`) the Gaussian
    moments are the approximation at the mode while the relative density is
    the exact posterior's, which is the pair a sampler wants to start from.

    The algebra does not carry the density: a push-forward or a conditioning
    returns a plain ``GaussianMeasure``, because a density does not survive a
    general linear map.
    """

    def __init__(
        self,
        domain: "HilbertSpace",
        /,
        *,
        reference: ProbabilityMeasure,
        relative_log_density: "Functional",
        log_normalizer: "Estimate | Any | None" = None,
        **gaussian: Any,
    ) -> None:
        """
        Args:
            domain: the model space.
            reference: the prior.
            relative_log_density: ``-Phi``, the log density against it up to
                the normalizer.
            log_normalizer: the evidence in the density's convention, an
                ``Estimate`` or a zero-argument callable computing one on
                first use, or ``None`` when unknown.
            **gaussian: everything ``GaussianMeasure`` takes — expectation,
                covariance, precision, sample.
        """
        super().__init__(domain, **gaussian)
        self._init_density(
            reference, relative_log_density, log_normalizer=log_normalizer
        )


class LinearGaussianInversion(GaussianEstimator):
    """The posterior estimator for a linear problem with a Gaussian prior."""

    def __init__(
        self,
        problem: LinearForwardProblem,
        prior: GaussianMeasure,
        /,
        *,
        solver: LinearSolver | None = None,
        formalism: Formalism = "data_space",
    ) -> None:
        """
        Args:
            problem: the forward problem, whose forward operator must be
                linear and whose error measure, if any, must be Gaussian.
            prior: a Gaussian measure on the model space.
            solver: how to invert the normal operator. Conjugate gradients by
                default, since matrix-free is the right default and a direct
                solver forms the matrix. To precondition, pass an iterative
                solver carrying one: ``CGSolver().with_preconditioner(...)``.
                A preconditioner that is itself a ``LinearSolver`` is handed
                the normal operator at solve time, so most need nothing
                further. For one built from *other* factors — a surrogate on a
                coarser space — pass a **callable** here instead: it receives
                the assembled normal operator and returns the solver, so the
                preconditioner can be built from the very thing it will
                precondition. See
                :func:`~pygeoinf2.numerics.solvers.resolve_solver`.
            formalism: which space to assemble in: ``"data_space"`` (the
                default), ``"model_space"``, or ``"auto"`` to take the
                smaller, which for an underdetermined problem is the data
                space and for an overdetermined one the model space; see
                :func:`~pygeoinf2.numerics.normal.choose_formalism`.
        """
        if not isinstance(prior, GaussianMeasure):
            raise TypeError(
                f"The prior must be a GaussianMeasure -- this class computes "
                f"the closed-form conjugate update and covers no other case. "
                f"Got {type(prior).__name__}."
            )
        if prior.domain != problem.model_space:
            raise ValueError("The prior is not defined on the model space.")
        if problem.has_error and not isinstance(problem.error_measure, GaussianMeasure):
            raise TypeError(
                f"The data error must be a Gaussian measure, not "
                f"{type(problem.error).__name__}. For a set-valued error see "
                f"pygeoinf2.inference.backus."
            )

        # The normal operator needs no solver — a solver only inverts it — so
        # it is built first and the solver resolved against it. That is what
        # lets a caller hand in a factory and precondition with something built
        # from the very operator about to be inverted.
        normal = NormalOperator(
            problem.forward_operator,
            prior,
            error=problem.error_measure if problem.has_error else None,
            formalism=formalism,
        )
        solver = resolve_solver(solver, normal)
        inverse = solver(normal)
        gain = normal.kalman_operator(inverse)
        covariance = normal.posterior_covariance(inverse, gain)

        shift = self._data_shift(problem, prior)
        # The constant term of the posterior mean is one application of the
        # gain -- one solve -- so it is deferred until the estimator is used.
        # With a direct solver that solve is the factorization, and an
        # estimator built to be reduced, swept or made a surrogate of would
        # otherwise pay O(n^3) at construction for an inverse it never applies.
        model_space = problem.model_space

        def translation():
            return model_space.subtract(prior.expectation, gain(shift))

        # The sampler is passed *in* rather than attached on the way out, so
        # that push_forward carries it to the property posterior too.
        super().__init__(
            AffineOperator(gain, translation),
            covariance,
            centered_sample=(
                self._draw_fluctuation
                if self._prior_and_noise_can_be_drawn(problem, prior)
                else None
            ),
        )

        self._problem = problem
        self._prior = prior
        self._normal = normal
        self._inverse = inverse
        self._gain = gain
        self._formalism = normal.formalism
        # Log-determinants already computed, keyed on the settings that
        # produced them. See normal_log_determinant.
        self._log_determinants: dict[tuple, Any] = {}
        self._solver = solver
        # The one-entry memo of _residual_solve. Keyed on the *contents* of the
        # data vector: an id() key would return a stale answer once a freed
        # array's address is reused.
        self._solved_key: bytes | None = None
        self._solved: tuple[Any, Any, Any] | None = None

    @staticmethod
    def _data_shift(problem: LinearForwardProblem, prior: GaussianMeasure) -> Any:
        """``A mu + mu_e``: what the data would be with no signal and no noise."""
        shift = problem.forward_operator(prior.expectation)
        if problem.has_error:
            shift = problem.data_space.add(shift, problem.error_measure.expectation)
        return shift

    @property
    def kalman_operator(self) -> LinearOperator:
        """The Kalman operator: data residuals to model updates.

        A property rather than a method, because the solver it needs was given
        at construction. ``Q A* N^-1`` in the data-space formalism and
        ``N^-1 A* R^-1`` in the model-space one, which are the same operator
        written two ways.
        """
        return self._gain

    @property
    def data_prior(self) -> GaussianMeasure:
        """The prior predictive ``A Q A* + R``: the data before observing them.

        What the evidence is a density under, and what a synthetic dataset is
        drawn from.
        """
        return self._problem.data_measure_from_model_measure(self._prior)

    @property
    def joint_prior(self) -> Any:
        """The joint law of model and data before observing anything.

        Sampling it gives a consistent ``(model, data)`` pair, which is how a synthetic test is set up
        without smuggling in knowledge of the answer.
        """
        return self._problem.joint_measure(self._prior)

    @property
    def normal_operator(self) -> NormalOperator:
        """The assembled normal operator, with its factors still attached.

        What a preconditioner is built against. Generic preconditioners take it
        as a plain operator; structure-aware ones read ``forward``,
        ``prior_covariance`` and ``error_covariance`` off it. Also the thing to
        look at when a solve is behaving badly — its condition number is the
        problem's, not the forward operator's.
        """
        return self._normal

    @property
    def inverse_normal_operator(self) -> LinearOperator:
        """The inverse the solver produced, which in the model-space formalism
        *is* the posterior covariance."""
        return self._inverse

    @property
    def formalism(self) -> str:
        """Which space the normal equations were assembled in."""
        return self._formalism

    @property
    def prior(self) -> GaussianMeasure:
        """The prior this posterior updates."""
        return self._prior

    @property
    def forward_problem(self) -> LinearForwardProblem:
        """The forward problem being inverted."""
        return self._problem

    @property
    def solver(self) -> LinearSolver:
        """How the normal operator is inverted."""
        return self._solver

    def with_solver(self, solver: Any, /) -> "LinearGaussianInversion":
        """The same inversion, solved a different way.

        The whole point of a preconditioner: build one against
        :attr:`normal_operator` or a :meth:`surrogate`, then come back here.
        """
        return LinearGaussianInversion(
            self._problem, self._prior, solver=solver, formalism=self._formalism
        )

    def with_formalism(self, formalism: Formalism, /) -> "LinearGaussianInversion":
        """The same inversion, assembled in the other space.

        The two give the same answer, so this is a computational choice.
        """
        return LinearGaussianInversion(
            self._problem, self._prior, solver=self._solver, formalism=formalism
        )

    # ----------------------------------------------------------------- #
    #                            Surrogates                             #
    # ----------------------------------------------------------------- #

    def surrogate(
        self,
        /,
        *,
        forward: LinearOperator | None = None,
        prior: GaussianMeasure | None = None,
        error: GaussianMeasure | None = None,
        formalism: Formalism | None = None,
    ) -> NormalOperator:
        """A cheap stand-in for this inversion's normal operator.

        Returns the surrogate *normal operator* rather than a whole inversion,
        because that is what a preconditioner is built from and it is the only
        part of a surrogate problem that is ever used. The surrogate may live
        on a different model space; see
        :meth:`~pygeoinf2.numerics.normal.NormalOperator.surrogate`.

        Args:
            forward: a cheaper ``A``, sharing the data space.
            prior: a cheaper ``Q``, on the surrogate's model space.
            error: a cheaper ``R``.
            formalism: which space to assemble in. Inherited if omitted.

        Returns:
            The surrogate normal operator.
        """
        return self._normal.surrogate(
            forward=forward, prior=prior, error=error, formalism=formalism
        )

    def low_rank_surrogate(
        self,
        /,
        *,
        forward_rank: int | None = None,
        prior_rank: int | None = None,
        error_rank: int | None = None,
        rng: Generator | None = None,
        **kwargs: Any,
    ) -> NormalOperator:
        """A surrogate built by truncating each factor to a low-rank version.

        The general-purpose way to get a cheap surrogate when no cheaper
        *physics* is available: randomized SVD for the forward operator and
        randomized eigendecomposition for the measures. A problem-specific
        surrogate — a coarser mesh, a smoother kernel — is usually better, and
        this is what to reach for when there isn't one.

        Args:
            forward_rank: the rank to truncate ``A`` to, by randomized SVD.
            prior_rank: the rank for ``Q``, by randomized eigendecomposition.
            error_rank: the rank for ``R``, likewise.
            rng: the generator for the probes.
            **kwargs: passed to the randomized routines -- oversampling and
                power iterations.

        Returns:
            The surrogate normal operator. Each rank left as None leaves that
            factor exact.

        Raises:
            ValueError: if a rank exceeds the dimension of what it truncates,
                or the problem has no error measure to truncate.
        """
        forward = None
        if forward_rank is not None:
            forward = random_svd(
                self._normal.forward, rank=forward_rank, rng=rng, **kwargs
            )
        prior = None
        if prior_rank is not None:
            prior = self._prior.low_rank_approximation(
                rank=prior_rank, rng=rng, **kwargs
            )
        error = None
        if error_rank is not None:
            if not self._problem.has_error:
                raise ValueError("There is no data error measure to approximate.")
            error = self._problem.error_measure.low_rank_approximation(
                rank=error_rank, rng=rng, **kwargs
            )
        return self.surrogate(forward=forward, prior=prior, error=error)

    def parameterized(
        self,
        parameterization: LinearOperator,
        /,
        *,
        prior: GaussianMeasure,
    ) -> "LinearGaussianInversion":
        """The same inversion restricted to a parameter space.

        ``A P`` in place of ``A``, with a prior on the parameters. Not a
        preconditioner but a smaller *problem*, whose answer is a different
        (and generally worse) estimate; the point is that it may be the only
        one that fits in memory.

        Args:
            parameterization: maps the parameter space into the model space.
            prior: the prior on the parameters. Required: pulling the model
                prior back through the parameterization is not a pull-back and
                would be a different measure.

        Returns:
            The inversion of the parameterized problem.
        """
        reduced = self._problem.parameterized(parameterization)
        return LinearGaussianInversion(
            reduced, prior, solver=self._solver, formalism=self._formalism
        )

    def data_reduced(
        self,
        reduction: LinearOperator,
        /,
        *,
        error: ProbabilityMeasure | Subset | None = None,
    ) -> "LinearGaussianInversion":
        """The same inversion with the data compressed by *reduction*.

        ``R A`` in place of ``A``, with the error measure pushed through.

        Args:
            reduction: maps the data space to the reduced one.
            error: the reduced error. Defaults to pushing the current one
                forward, as on the problem itself.

        Returns:
            The inversion of the reduced problem.
        """
        reduced = self._problem.data_reduced(reduction, error=error)
        return LinearGaussianInversion(
            reduced, self._prior, solver=self._solver, formalism=self._formalism
        )

    def _data_key(self, data: Any, /) -> bytes | None:
        """A hashable summary of the data vector's contents, or None.

        The memo below must be keyed on what the vector *is*, not on where it
        happens to live: keying on ``id()`` returns the previous caller's
        answer as soon as an array is freed and its address reused. The price
        is one ``to_components`` and one buffer copy per call, small beside a
        solve. A space with no coordinate map cannot be summarized, and there
        the memo is simply off.

        Args:
            data: the data vector.

        Returns:
            The key, or None when the space has no components.
        """
        from ..algebra.spaces import CoordinateSpace

        space = self.data_space
        if not isinstance(space, CoordinateSpace):
            return None
        return np.ascontiguousarray(space.to_components(data)).tobytes()

    def _residual_solve(self, data: Any, /) -> tuple[Any, Any, Any]:
        """``(v, b, w)`` with ``v == d - shift``, ``b`` its right-hand side and
        ``N w == b``.

        The posterior mean is ``mu + K v`` and the data misfit is a pairing
        against ``N^-1`` of the same ``v``, so the two differ only in what they
        do *after* the solve. Memoizing it lets ``est(data)`` followed by
        ``log_evidence(data)`` cost one solve rather than two, and a mixture
        inversion over ``K`` components ``K`` solves rather than ``2K``.

        One entry is enough because the two calls that matter are adjacent.
        Nothing here is mutated in place, but the tuple is shared, so callers
        must treat it as read-only.

        Args:
            data: the observations.

        Returns:
            The residual, its right-hand side, and the solution.
        """
        key = self._data_key(data)
        if key is not None and self._solved_key == key:
            return self._solved
        residual = self.data_space.subtract(
            data, self._data_shift(self._problem, self._prior)
        )
        right_hand_side = self._normal.right_hand_side(residual)
        result = (residual, right_hand_side, self._inverse(right_hand_side))
        if key is not None:
            self._solved_key, self._solved = key, result
        return result

    def __call__(self, data: Any) -> GaussianPosterior:
        """The posterior measure for this data.

        As :meth:`~pygeoinf2.inference.estimators.GaussianEstimator.__call__`,
        but reaching the mean through :meth:`_residual_solve` so that the solve
        is shared with :meth:`mahalanobis`. :attr:`expectation_operator`
        computes the same mean without the shared solve, for a caller who
        wants the affine operator itself.

        The result is a :class:`GaussianPosterior`: the posterior Gaussian,
        also carrying the prior as its reference, ``-Phi`` as its density
        against it, and, when the problem has an error measure, the evidence
        as its normalizer -- computed lazily, since it costs log-determinants.

        Args:
            data: the observations.

        Returns:
            The posterior, Gaussian and samplable whenever the prior and the
            noise are.
        """
        _, _, solved = self._residual_solve(data)
        mean = self._problem.model_space.add(
            self._prior.expectation, self._normal.model_update(solved)
        )
        return GaussianPosterior(
            self.property_space,
            expectation=mean,
            covariance=self.covariance,
            sample=self._centered_sample,
            reference=self._prior,
            relative_log_density=-self._problem.potential(data),
            log_normalizer=(
                self._log_normalizer_for(data) if self._problem.has_error else None
            ),
        )

    def _log_normalizer_for(self, data: Any, /) -> Any:
        """The evidence in the potential's convention, lazily.

        With ``f == -chi^2/2``, the normalizer is ``log int exp(f) d prior ==
        (log|R| - log|S| - <v, S^-1 v>) / 2`` with ``S == A Q A* + R`` the
        data-space normal and ``v`` the shifted residual — all data-space
        quantities, so it costs two log-determinants and shares
        :meth:`_residual_solve`'s solve. It relates to :meth:`log_evidence`
        by the likelihood's own constant, ``-(n log 2pi + log|R|) / 2``,
        which the potential omits.
        """

        def compute() -> "Estimate":
            from ..numerics.functional_calculus import log_determinant

            misfit = self.mahalanobis(data)
            volume = self.normal_log_determinant()
            definite = self._problem.error_measure.covariance.with_traits(
                Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
            )
            error_volume = log_determinant(definite)
            return combine_log_normalizer(
                [(0.5, error_volume), (-0.5, volume), (-0.5, misfit)]
            )

        return compute

    def mahalanobis(self, data: Any, /) -> float:
        """``<v, N_d^-1 v>``, the misfit half of the evidence.

        The optimal penalized misfit: the minimum over models of the data
        misfit plus the prior penalty, which is minus twice the unnormalized
        log posterior at the posterior mean. Computed **matrix-free**, through the
        solver and preconditioner this estimator was given, so it costs one
        solve of the normal equations and never assembles anything.

        In the model-space formalism the data-space inverse is avoided
        entirely, by Woodbury:

        .. code-block:: text

            <v, N_d^-1 v> == <v, R^-1 v> - <A* R^-1 v, N_m^-1 A* R^-1 v>

        which is the whole point of that formalism — the data space is the
        large one there, and it is never inverted.

        Args:
            data: the observations.

        Returns:
            The misfit.

        Raises:
            ValueError: if the problem has no data error measure, there being
                no metric to measure the misfit in; or, in the model-space
                formalism, if the error has no precision, which the Woodbury
                reduction above needs.
        """
        if not self._problem.has_error:
            raise ValueError(
                "The evidence needs a data error measure: without one the "
                "data-space covariance A Q A* is singular and p(d) is not a "
                "density."
            )
        space = self.data_space
        residual, right_hand_side, solved = self._residual_solve(data)
        if self._formalism == "data_space":
            return float(space.inner_product(residual, solved))
        precision = self._problem.error_measure.precision
        if precision is None:
            raise ValueError(
                "The model-space formalism needs the error precision R^-1 to "
                "reduce the misfit through Woodbury."
            )
        base = space.inner_product(residual, precision(residual))
        reduction = self._problem.model_space.inner_product(right_hand_side, solved)
        return float(base - reduction)

    def normal_log_determinant(
        self,
        /,
        *,
        method: str = "auto",
        samples: int = 100,
        rng: Generator | None = None,
        **kwargs: Any,
    ) -> "Estimate":
        """``log |A Q A* + R|``, the volume half of the evidence.

        Densely when the space is small enough to afford it, and by stochastic
        Lanczos quadrature otherwise -- see
        :func:`~pygeoinf2.numerics.functional_calculus.log_determinant`. The
        stochastic route needs only applications of the operator, so the
        evidence is available on problems too large to assemble.

        In the model-space formalism it is reached by Sylvester's identity,

        .. code-block:: text

            |A Q A* + R| == |Q| |R| |Q^-1 + A* R^-1 A|

        so the data-space operator is never formed even to take its
        determinant. That costs two further log-determinants, of ``Q`` and
        ``R``, which are usually the cheap ones: a prior with a known spectrum
        and a diagonal noise covariance.

        Args:
            method: as for
                :func:`~pygeoinf2.numerics.functional_calculus.log_determinant`
                -- ``"dense"``, ``"stochastic"`` or ``"auto"``.
            samples: Hutchinson probes for the stochastic route.
            rng: the generator for those probes.
            **kwargs: passed through, including ``sample_rtol`` to sample to a
                tolerance rather than a count.

        Returns:
            An estimate, memoized on these settings -- the volume term does
            not depend on the data, and a mixture asks for it once per
            component per datum.

        Raises:
            ValueError: if the problem has no data error measure.
        """
        from ..numerics.functional_calculus import log_determinant

        if not self._problem.has_error:
            raise ValueError("The evidence needs a data error measure.")

        # The volume term does not depend on the data, and a mixture asks for
        # it once per component per call to weights() -- so without a memo the
        # single most expensive part of an evidence calculation is repeated
        # for every datum compared against the same model.
        #
        # Stochastic estimates are cached too, deliberately. Repeating them
        # would make a mixture's weights jitter between two calls with the
        # same data, which reads as instability in the model comparison rather
        # than in the estimator. Pass an ``rng`` to control what is cached;
        # the settings are part of the key, so asking for a different method
        # or more samples recomputes.
        key = (method, samples, tuple(sorted(kwargs.items())))
        cached = self._log_determinants.get(key)
        if cached is not None:
            return cached

        settings = dict(method=method, samples=samples, rng=rng, **kwargs)
        if self._formalism == "data_space":
            result = log_determinant(self._normal, **settings)
            self._log_determinants[key] = result
            return result

        from ..numerics.randomized import Estimate

        # Positive definiteness is claimed here rather than deduced: in this
        # formalism Q and R are inverted anyway, so a caller who has got this
        # far has already asserted it.
        definite = Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
        parts = [
            log_determinant(self._normal, **settings),
            log_determinant(self._prior.covariance.with_traits(definite), **settings),
            log_determinant(
                self._problem.error_measure.covariance.with_traits(definite),
                **settings,
            ),
        ]
        total = sum(part.value for part in parts)
        # Independent estimates, so the errors add in quadrature.
        error = float(np.sqrt(sum(part.standard_error**2 for part in parts)))
        result = Estimate(float(total), error, min(part.samples for part in parts))
        self._log_determinants[key] = result
        return result

    def evidence_terms(self, data: Any, /, **kwargs: Any) -> tuple[float, float]:
        """The two halves of the log evidence: misfit and volume.

        ``log p(d) == -(mahalanobis + log det N + dim log 2pi) / 2`` with ``N``
        the *data* prior covariance ``A Q A* + R``. Returned separately because
        they answer different questions: the first says whether the data are
        surprising under this model, the second penalizes a model flexible
        enough that they would not have been.

        Keyword arguments go to :meth:`normal_log_determinant`; pass
        ``method="stochastic"`` to keep the calculation matrix-free.
        """
        return self.mahalanobis(data), self.normal_log_determinant(**kwargs).value

    def log_evidence(self, data: Any, /, **kwargs: Any) -> float:
        """``log p(d)``: how well this model explains the data at all.

        The quantity model comparison needs, and the one a posterior cannot
        supply — a posterior is conditional on the model being right.
        """
        mahalanobis, volume = self.evidence_terms(data, **kwargs)
        dimension = self.data_space.dim
        return -0.5 * (mahalanobis + volume + dimension * np.log(2.0 * np.pi))

    @staticmethod
    def _prior_and_noise_can_be_drawn(
        problem: LinearForwardProblem, prior: GaussianMeasure
    ) -> bool:
        """Whether randomize-then-optimize has anything to randomize.

        The draw needs a sample of the prior and a sample of the noise; the
        posterior covariance itself is never factorized, which is the point.
        A static check because it is asked during construction, before the
        attributes it would otherwise read have been set.
        """
        if not prior.can_sample:
            return False
        return not problem.has_error or problem.error_measure.can_sample

    def _draw_fluctuation(self, rng: Generator | None, /) -> Any:
        """One draw of the posterior *fluctuation*, by randomize-then-optimize.

        Draw a model and a noise vector, form the residual they would have
        produced, and correct with the gain. Written centered, which makes
        something worth seeing explicit: **the fluctuation does not depend on
        the data.** Only the mean does — the same statement as the covariance
        being data-independent, which is why this estimator is a pair.

        Centered also because a ``sample`` callable handed to
        :class:`~pygeoinf2.probability.gaussian.GaussianMeasure` supplies the
        draw *about* the mean; the measure adds the expectation itself.
        """
        problem, prior, gain = self._problem, self._prior, self._gain
        model_space, data_space = problem.model_space, problem.data_space

        drawn = model_space.subtract(prior.sample(rng=rng), prior.expectation)
        residual = data_space.negative(problem.forward_operator(drawn))
        if problem.has_error:
            error = problem.error_measure
            noise = data_space.subtract(error.sample(rng=rng), error.expectation)
            residual = data_space.subtract(residual, noise)
        return model_space.add(drawn, gain(residual))


def _information_eigenpairs(
    forward: LinearOperator,
    prior: GaussianMeasure,
    error_precision: LinearOperator,
    /,
    *,
    rank: int | None,
    rng: Generator | None,
    n_jobs: int | None,
) -> tuple[LinearOperator, np.ndarray]:
    """The leading eigenpairs of ``L* A* R^-1 A L``, the misfit Hessian seen
    through the prior's factor ``L``: an isometry from ``R^r`` into the
    latent space whose columns are the eigenvectors, and the eigenvalues."""
    from ..numerics.randomized import random_eig

    factor = prior.covariance_factor
    if factor is None:
        raise ValueError("A low-rank posterior needs a prior with a covariance factor.")
    through = forward @ factor
    hessian = (through.adjoint @ error_precision @ through).with_traits(
        Traits.SELF_ADJOINT | Traits.POSITIVE_SEMIDEFINITE
    )
    decomposition = random_eig(hessian, rank=rank, rng=rng, n_jobs=n_jobs)
    eigenvalues = np.clip(np.asarray(decomposition.eigenvalues, dtype=float), 0.0, None)
    return decomposition.factor, eigenvalues


class LowRankGaussianPosterior(GaussianPosterior):
    """The prior corrected in the ``r`` directions the data inform most.

    The posterior of a Gaussian prior under a linear, or linearized, forward
    operator differs from the prior only where the data carry information,
    and the data carry it in at most as many directions as there are data,
    usually far fewer that matter. Those directions are the eigenvectors of
    the prior-preconditioned misfit Hessian ``H == L* A* R^-1 A L``, with
    ``L`` the prior's covariance factor, and its eigenvalues ``lambda`` say
    how much the data narrow each one. Keeping the ``r`` leading pairs
    ``(lambda_k, v_k)`` gives

    .. code-block:: text

        C == L (I - V D V*) L*,          D == diag(lambda / (1 + lambda))
        C^-1 == Li* (I + V Lambda V*) Li,  with  C_prior^-1 == Li* Li

    exact when ``r`` is the number of nonzero eigenvalues and otherwise the
    prior in the directions dropped, so always between the prior and the
    full posterior. The covariance factor is ``L (I - V a V*)`` with ``a ==
    1 - (1 + lambda)^-1/2``, so the measure samples at the cost of a prior
    draw; the precision factor is ``(I + V b V*) Li`` with ``b == (1 +
    lambda)^1/2 - 1``. Nothing here is assembled: every operator is the
    prior's factors composed with rank-``r`` corrections.

    The density relative to the prior is ``r``-dimensional, and the evidence
    in the potential's convention is ``-J(m) - sum log(1 + lambda) / 2`` with
    ``J`` the penalized misfit at the mean: exact for a linear problem at
    full rank, the Laplace estimate otherwise.

    Where the model space has a pointwise product, the amount by which the
    prior's pointwise variance exceeds this measure's is the field ``sum_k
    d_k (L v_k)^2``, from :meth:`pointwise_variance_reduction`: the
    posterior's pointwise variance on a space with no closed form for it,
    without sampling.
    """

    def __init__(
        self,
        prior: GaussianMeasure,
        mean: Any,
        eigenvalues: np.ndarray,
        directions: LinearOperator,
        /,
        *,
        relative_log_density: "Functional",
        penalized_misfit: float | None = None,
    ) -> None:
        """
        Args:
            prior: the Gaussian prior, with a covariance factor.
            mean: the posterior mean, or the mode the Laplace Gaussian is
                centered on.
            eigenvalues: the ``r`` leading eigenvalues of the
                prior-preconditioned misfit Hessian, non-negative.
            directions: an isometry from ``R^r`` into the prior factor's
                domain whose columns are the eigenvectors.
            relative_log_density: ``-Phi``, the log density against the
                prior up to the normalizer.
            penalized_misfit: ``J(m)``, the misfit at the mean plus half its
                squared prior norm about the prior's mean, for the evidence;
                computed from the mean when omitted and the prior has a
                precision or a precision factor, otherwise the evidence is
                unknown.

        Raises:
            ValueError: if the prior has no covariance factor, the
                directions do not map into the factor's domain, or an
                eigenvalue is negative.
        """
        from ..algebra.diagonal import DiagonalLinearOperator
        from ..numerics.randomized import Estimate

        factor = prior.covariance_factor
        if factor is None:
            raise ValueError(
                "A low-rank posterior needs a prior with a covariance factor."
            )
        latent = factor.domain
        if directions.codomain != latent:
            raise ValueError(
                f"The directions must map into the prior factor's domain "
                f"{latent!r}; they map into {directions.codomain!r}."
            )
        eigenvalues = np.asarray(eigenvalues, dtype=float)
        if eigenvalues.ndim != 1 or eigenvalues.size != directions.domain.dim:
            raise ValueError(
                f"Expected {directions.domain.dim} eigenvalues, one per direction, "
                f"got shape {eigenvalues.shape}."
            )
        if np.any(eigenvalues < 0.0):
            raise ValueError("The eigenvalues of a misfit Hessian are non-negative.")
        self._prior = prior
        self._eigenvalues = eigenvalues
        self._directions = directions
        space = prior.domain
        definite = Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE

        def corrected(weights: np.ndarray, sign: float) -> LinearOperator:
            """``I + sign V diag(weights) V*`` on the latent space."""
            inner = DiagonalLinearOperator(directions.domain, weights)
            return (
                LinearOperator.identity(latent)
                + sign * (directions @ inner @ directions.adjoint)
            ).with_traits(definite)

        shrink = eigenvalues / (1.0 + eigenvalues)
        covariance = (factor @ corrected(shrink, -1.0) @ factor.adjoint).with_traits(
            definite
        )
        covariance_factor = factor @ corrected(
            1.0 - 1.0 / np.sqrt(1.0 + eigenvalues), -1.0
        )
        precision = precision_factor = None
        whitening = prior.precision_factor
        prior_precision = prior.precision
        if whitening is not None:
            precision = (
                whitening.adjoint @ corrected(eigenvalues, 1.0) @ whitening
            ).with_traits(definite)
            precision_factor = (
                corrected(np.sqrt(1.0 + eigenvalues) - 1.0, 1.0) @ whitening
            )
        elif prior_precision is not None:
            # Li* V == P L V, the prior precision applied to the modes as
            # models, so C^-1 == P + (P W) Lambda (P W)* with W == L V: the
            # precision without a precision factor.
            pushed = prior_precision @ factor @ directions
            precision = (
                prior_precision
                + pushed
                @ DiagonalLinearOperator(directions.domain, eigenvalues)
                @ pushed.adjoint
            ).with_traits(definite)

        log_normalizer = None
        if penalized_misfit is None:
            offset = space.subtract(mean, prior.expectation)
            if whitening is not None:
                quadratic = latent.squared_norm(whitening(offset))
            elif prior_precision is not None:
                quadratic = space.inner_product(prior_precision(offset), offset)
            else:
                quadratic = None
            if quadratic is not None:
                penalized_misfit = -float(relative_log_density(mean)) + 0.5 * float(
                    quadratic
                )
        if penalized_misfit is not None:
            log_normalizer = Estimate(
                -float(penalized_misfit) - 0.5 * float(np.sum(np.log1p(eigenvalues))),
                0.0,
                0,
            )
        super().__init__(
            space,
            reference=prior,
            relative_log_density=relative_log_density,
            log_normalizer=log_normalizer,
            expectation=mean,
            covariance=covariance,
            covariance_factor=covariance_factor,
            precision=precision,
            precision_factor=precision_factor,
        )

    @classmethod
    def from_linear_problem(
        cls,
        problem: LinearForwardProblem,
        prior: GaussianMeasure,
        data: Any,
        /,
        *,
        rank: int | None = None,
        rng: Generator | None = None,
        n_jobs: int | None = None,
    ) -> "LowRankGaussianPosterior":
        """The low-rank posterior of a linear problem, with no solve at all.

        The mean is ``m_0 + C A* R^-1 (d - A m_0)`` with ``C`` the low-rank
        covariance, so once the eigenpairs are in hand it is a few
        applications; at full rank it is the posterior mean exactly.

        Args:
            problem: the linear forward problem, with a Gaussian error.
            prior: the Gaussian prior, with a covariance factor.
            data: the observations.
            rank: how many directions to keep; adaptive if omitted, as for
                :func:`~pygeoinf2.numerics.randomized.random_eig`.
            rng: the generator for the randomized eigendecomposition.
            n_jobs: workers for it.

        Returns:
            The posterior.

        Raises:
            ValueError: if the error measure has no precision, or the prior
                no covariance factor.
        """
        forward = problem.forward_operator
        error_precision = problem.error_measure.precision
        if error_precision is None:
            raise ValueError("The error measure needs a precision.")
        directions, eigenvalues = _information_eigenpairs(
            forward, prior, error_precision, rank=rank, rng=rng, n_jobs=n_jobs
        )
        factor = prior.covariance_factor
        latent = factor.domain
        space = prior.domain
        residual = forward.codomain.subtract(data, forward(prior.expectation))
        pulled = factor.adjoint(forward.adjoint(error_precision(residual)))
        weights = eigenvalues / (1.0 + eigenvalues)
        coefficients = directions.adjoint(pulled)
        corrected = latent.subtract(pulled, directions(weights * coefficients))
        mean = space.add(prior.expectation, factor(corrected))
        return cls(
            prior,
            mean,
            eigenvalues,
            directions,
            relative_log_density=-problem.potential(data),
        )

    @property
    def prior(self) -> GaussianMeasure:
        """The prior this measure corrects."""
        return self._prior

    @property
    def rank(self) -> int:
        """How many directions are kept."""
        return int(self._eigenvalues.size)

    @property
    def eigenvalues(self) -> np.ndarray:
        """The eigenvalues of the prior-preconditioned misfit Hessian kept,
        largest first: how much the data narrow each direction."""
        return self._eigenvalues.copy()

    @property
    def directions(self) -> LinearOperator:
        """The isometry from ``R^r`` into the latent space whose columns are
        the directions kept."""
        return self._directions

    def modes(self) -> list[Any]:
        """The directions as models, ``L v_k``: the fields the data inform."""
        factor = self._prior.covariance_factor
        unit = np.eye(self.rank)
        return [factor(self._directions(unit[:, k])) for k in range(self.rank)]

    def pointwise_variance_reduction(self) -> Any:
        """``sum_k d_k (L v_k)^2``, the field by which the prior's pointwise
        variance exceeds this measure's.

        Raises:
            TypeError: if the model space has no pointwise product.
        """
        from ..algebra.spaces import HilbertModule

        space = self.domain
        if not isinstance(space, HilbertModule):
            raise TypeError(
                f"{space!r} has no pointwise product, so no pointwise variance."
            )
        shrink = self._eigenvalues / (1.0 + self._eigenvalues)
        total = space.zero()
        for weight, mode in zip(shrink, self.modes()):
            total = space.axpy(float(weight), space.multiply(mode, mode), total)
        return total
