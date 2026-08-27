"""
The posterior of a linear problem with a Gaussian prior.

The Kalman update, assembled in whichever space is smaller. With prior
covariance ``Q`` and data covariance ``R``:

.. code-block:: text

    data space   N = A Q A* + R          K = Q A* N^-1
    model space  N = Q^-1 + A* R^-1 A    K = N^-1 A* R^-1

and the posterior is ``mu + K (d - A mu - mu_e)`` with covariance ``Q - K A Q``
— which in the model-space formalism is just ``N^-1``, so nothing extra is
formed there. That algebra lives on
:class:`~pygeoinf2.inference.normal.NormalOperator`, which is also what the
class exposes so that preconditioners can be built against it.

The name says what the class does and what it needs. It is *linear* — the
forward operator is a :class:`LinearOperator`, not merely differentiable — and
it is *Gaussian* — the prior is a
:class:`~pygeoinf2.probability.gaussian.GaussianMeasure`, not any probability
measure, because the update below is the closed-form conjugate one and there is
no other case it covers. Bayesian inference in general is neither.

The result is a :class:`~pygeoinf2.inference.estimators.GaussianEstimator`
rather than a measure: the covariance does not depend on the data, so the
mapping is a pair and only the mean moves. See DESIGN.md sections 18.7 and 23.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from numpy.random import Generator

from ..algebra.operators import AffineOperator, LinearOperator
from ..numerics.randomised import random_svd
from ..numerics.solvers import CholeskySolver, LinearSolver
from ..probability.gaussian import GaussianMeasure
from ..traits import Traits
from .estimators import GaussianEstimator
from .normal import Formalism, NormalOperator
from .problem import LinearForwardProblem

if TYPE_CHECKING:  # pragma: no cover
    from ..numerics.randomised import Estimate

__all__ = ["LinearGaussianInversion"]


class LinearGaussianInversion(GaussianEstimator):
    """The posterior estimator for a linear problem with a Gaussian prior."""

    def __init__(
        self,
        problem: LinearForwardProblem,
        prior: GaussianMeasure,
        /,
        *,
        solver: LinearSolver | None = None,
        formalism: Formalism = "auto",
    ) -> None:
        """
        Args:
            problem: the forward problem, whose forward operator must be
                linear and whose error measure, if any, must be Gaussian.
            prior: a Gaussian measure on the model space.
            solver: how to invert the normal operator. Cholesky by default.
                To precondition, pass an iterative solver carrying one:
                ``CGSolver().with_preconditioner(...)``. The preconditioner is
                handed the normal operator, so a generic one needs nothing
                further; a structure-aware one is built from
                :attr:`normal_operator` or a :meth:`surrogate` of it.
            formalism: which space to assemble in; ``"auto"`` takes the
                smaller, which for an underdetermined problem is the data
                space and for an overdetermined one the model space.
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

        solver = CholeskySolver() if solver is None else solver
        normal = NormalOperator(
            problem.forward_operator,
            prior,
            error=problem.error_measure if problem.has_error else None,
            formalism=formalism,
        )
        inverse = solver(normal)
        gain = normal.gain(inverse)
        covariance = normal.posterior_covariance(inverse, gain)

        shift = self._data_shift(problem, prior)
        translation = problem.model_space.subtract(prior.expectation, gain(shift))
        super().__init__(AffineOperator(gain, translation), covariance)

        self._problem = problem
        self._prior = prior
        self._normal = normal
        self._inverse = inverse
        self._gain = gain
        self._formalism = normal.formalism
        self._solver = solver

    @staticmethod
    def _data_shift(problem: LinearForwardProblem, prior: GaussianMeasure) -> Any:
        """``A mu + mu_e``: what the data would be with no signal and no noise."""
        shift = problem.forward_operator(prior.expectation)
        if problem.has_error:
            shift = problem.data_space.add(shift, problem.error_measure.expectation)
        return shift

    @property
    def gain(self) -> LinearOperator:
        """The Kalman gain: data residuals to model updates."""
        return self._gain

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
    def problem(self) -> LinearForwardProblem:
        """The forward problem being inverted."""
        return self._problem

    @property
    def solver(self) -> LinearSolver:
        """How the normal operator is inverted."""
        return self._solver

    def with_solver(self, solver: LinearSolver, /) -> "LinearGaussianInversion":
        """The same inversion, solved a different way.

        The whole point of a preconditioner: build one against
        :attr:`normal_operator` or a :meth:`surrogate`, then come back here.
        """
        return LinearGaussianInversion(
            self._problem, self._prior, solver=solver, formalism=self._formalism
        )

    def with_formalism(self, formalism: Formalism, /) -> "LinearGaussianInversion":
        """The same inversion, assembled in the other space.

        The two give the same answer, so this is a computational choice; the
        test suite checks the agreement.
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
        :meth:`~pygeoinf2.inference.normal.NormalOperator.surrogate`.
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
        *physics* is available: randomised SVD for the forward operator and
        randomised eigendecomposition for the measures. A problem-specific
        surrogate — a coarser mesh, a smoother kernel — is usually better, and
        this is what to reach for when there isn't one.

        Each rank left as None leaves that factor exact.
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

    def parameterised(
        self,
        parameterisation: LinearOperator,
        /,
        *,
        prior: GaussianMeasure,
        **kwargs: Any,
    ) -> "LinearGaussianInversion":
        """The same inversion restricted to a parameter space.

        ``A P`` in place of ``A``, with a prior on the parameters. Not a
        preconditioner but a smaller *problem*, whose answer is a different
        (and generally worse) estimate; the point is that it may be the only
        one that fits in memory.
        """
        reduced = self._problem.parameterised(parameterisation, **kwargs)
        return LinearGaussianInversion(
            reduced, prior, solver=self._solver, formalism=self._formalism
        )

    def data_reduced(
        self, reduction: LinearOperator, /, **kwargs: Any
    ) -> "LinearGaussianInversion":
        """The same inversion with the data compressed by *reduction*.

        ``R A`` in place of ``A``, with the error measure pushed through.
        """
        reduced = self._problem.data_reduced(reduction, **kwargs)
        return LinearGaussianInversion(
            reduced, self._prior, solver=self._solver, formalism=self._formalism
        )

    def mahalanobis(self, data: Any, /) -> float:
        """``<v, N_d^-1 v>``, the misfit half of the evidence.

        The optimal, penalty-balanced data misfit: the unnormalised log
        posterior at the posterior mean. Computed **matrix-free**, through the
        solver and preconditioner this estimator was given, so it costs one
        solve of the normal equations and never assembles anything.

        In the model-space formalism the data-space inverse is avoided
        entirely, by Woodbury:

        .. code-block:: text

            <v, N_d^-1 v> == <v, R^-1 v> - <A* R^-1 v, N_m^-1 A* R^-1 v>

        which is the whole point of that formalism — the data space is the
        large one there, and it is never inverted.
        """
        if not self._problem.has_error:
            raise ValueError(
                "The evidence needs a data error measure: without one the "
                "data-space covariance A Q A* is singular and p(d) is not a "
                "density."
            )
        space = self.data_space
        residual = space.subtract(data, self._data_shift(self._problem, self._prior))
        right_hand_side = self._normal.right_hand_side(residual)
        solved = self._inverse(right_hand_side)
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
        Lanczos quadrature otherwise — see
        :func:`~pygeoinf2.numerics.functional_calculus.log_determinant`. A
        log-determinant is the one part of an evidence calculation that looks
        like it needs the matrix, and it does not; without this the whole
        calculation is confined to problems small enough to assemble, which is
        not where model comparison is interesting.

        In the model-space formalism it is reached by Sylvester's identity,

        .. code-block:: text

            |A Q A* + R| == |Q| |R| |Q^-1 + A* R^-1 A|

        so the data-space operator is never formed even to take its
        determinant. That costs two further log-determinants, of ``Q`` and
        ``R``, which are usually the cheap ones: a prior with a known spectrum
        and a diagonal noise covariance.
        """
        from ..numerics.functional_calculus import log_determinant

        if not self._problem.has_error:
            raise ValueError("The evidence needs a data error measure.")
        settings = dict(method=method, samples=samples, rng=rng, **kwargs)
        if self._formalism == "data_space":
            return log_determinant(self._normal, **settings)

        from ..numerics.randomised import Estimate

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
        return Estimate(float(total), error, min(part.samples for part in parts))

    def evidence_terms(self, data: Any, /, **kwargs: Any) -> tuple[float, float]:
        """The two halves of the log evidence: misfit and volume.

        ``log p(d) == -(mahalanobis + log det N + dim log 2pi) / 2`` with ``N``
        the *data* prior covariance ``A Q A* + R``. Returned separately because
        they answer different questions: the first says whether the data are
        surprising under this model, the second penalises a model flexible
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

    @property
    def can_sample(self) -> bool:
        """Whether the posterior can be drawn from."""
        if not self._prior.can_sample:
            return False
        return not self._problem.has_error or self._problem.error_measure.can_sample

    def _centred_sample(self, rng: Generator | None, /) -> Any:
        """One draw of the posterior *fluctuation*, by randomise-then-optimise.

        Draw a model and a noise vector, form the residual they would have
        produced, and correct with the gain. Written centred, which makes
        something worth seeing explicit: **the fluctuation does not depend on
        the data.** Only the mean does — the same statement as the covariance
        being data-independent, which is why this estimator is a pair.

        Centred also because a ``sample`` callable handed to
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

    def __call__(self, data: Any) -> Any:
        """The posterior measure, with a sampler when one is available."""
        posterior = super().__call__(data)
        if not self.can_sample:
            return posterior

        return GaussianMeasure(
            self.target_space,
            expectation=posterior.expectation,
            covariance=posterior.covariance,
            sample=self._centred_sample,
        )
