"""
The posterior: a measure prior in, a measure out.

The Kalman update, assembled in whichever space is smaller. With prior
covariance ``Q`` and data covariance ``R``:

.. code-block:: text

    data space   N = A Q A* + R          K = Q A* N^-1
    model space  N = Q^-1 + A* R^-1 A    K = N^-1 A* R^-1

and the posterior is ``mu + K (d - A mu - mu_e)`` with covariance ``Q - K A Q``
— which in the model-space formalism is just ``N^-1``, so nothing extra is
formed there.

The result is a :class:`~pygeoinf2.inference.estimators.GaussianEstimator`
rather than a measure: the covariance does not depend on the data, so the
mapping is a pair and only the mean moves. See DESIGN.md section 18.7.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from numpy.random import Generator

from ..algebra.operators import AffineOperator, LinearOperator
from ..numerics.solvers import CholeskySolver, LinearSolver
from ..probability.base import ProbabilityMeasure
from ..traits import Traits
from .estimators import GaussianEstimator
from .point import Formalism, choose_formalism
from .problem import LinearForwardProblem

__all__ = ["Bayesian"]


class Bayesian(GaussianEstimator):
    """The posterior estimator for a linear problem with a Gaussian prior."""

    def __init__(
        self,
        problem: LinearForwardProblem,
        prior: ProbabilityMeasure,
        /,
        *,
        solver: LinearSolver | None = None,
        formalism: Formalism = "auto",
    ) -> None:
        """
        Args:
            problem: the forward problem.
            prior: a Gaussian measure on the model space.
            solver: how to invert the normal operator. Cholesky by default.
            formalism: which space to assemble in; ``"auto"`` takes the
                smaller, which for an underdetermined problem is the data
                space and for an overdetermined one the model space.
        """
        if prior.domain != problem.model_space:
            raise ValueError("The prior is not defined on the model space.")
        forward = problem.forward_operator
        model_space = problem.model_space
        solver = CholeskySolver() if solver is None else solver
        chosen = choose_formalism(problem, formalism=formalism)

        prior_covariance = prior.covariance
        if chosen == "data_space":
            normal = forward @ prior_covariance @ forward.adjoint
            if problem.has_error:
                normal = normal + problem.error_measure.covariance
            inverse = solver(normal.with_traits(Traits.POSITIVE_DEFINITE))
            gain = prior_covariance @ forward.adjoint @ inverse
            # Q - K A Q is a Schur complement, so it is positive semidefinite --
            # a posterior covariance always is. The trait algebra cannot see
            # that through a difference, so the claim is made here, where the
            # reason is known, rather than deduced from the operands.
            covariance = (
                prior_covariance - gain @ forward @ prior_covariance
            ).with_traits(Traits.SELF_ADJOINT | Traits.POSITIVE_SEMIDEFINITE)
        else:
            precision = prior.precision
            if problem.has_error:
                data_precision = problem.error_measure.precision
                normal = precision + forward.adjoint @ data_precision @ forward
                weighted = forward.adjoint @ data_precision
            else:
                normal = precision + forward.adjoint @ forward
                weighted = forward.adjoint
            inverse = solver(normal.with_traits(Traits.POSITIVE_DEFINITE))
            gain = inverse @ weighted
            # In this formalism the inverted normal operator *is* the
            # posterior covariance, so there is nothing further to form.
            covariance = inverse

        shift = self._data_shift(problem, prior)
        translation = model_space.subtract(prior.expectation, gain(shift))
        super().__init__(AffineOperator(gain, translation), covariance)

        self._problem = problem
        self._prior = prior
        self._gain = gain
        self._formalism = chosen
        self._solver = solver

    @staticmethod
    def _data_shift(problem: LinearForwardProblem, prior: ProbabilityMeasure) -> Any:
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
    def formalism(self) -> str:
        """Which space the normal equations were assembled in."""
        return self._formalism

    @property
    def prior(self) -> ProbabilityMeasure:
        """The prior this posterior updates."""
        return self._prior

    def evidence_terms(self, data: Any, /) -> tuple[float, float]:
        """The two halves of the log evidence: misfit and volume.

        ``log p(d) == -(mahalanobis + log det N + dim log 2pi) / 2`` with ``N``
        the *data* prior covariance ``A Q A* + R``. Returned separately because
        they answer different questions: the first says whether the data are
        surprising under this model, the second penalises a model flexible
        enough that they would not have been.
        """
        prior_data = self._problem.data_measure_from_model_measure(self._prior)
        residual = self.data_space.subtract(data, prior_data.expectation)
        mahalanobis = prior_data._weighted_squared(residual)
        matrix = prior_data.covariance.matrix(form="galerkin")
        sign, logarithm = np.linalg.slogdet(0.5 * (matrix + matrix.T))
        if sign <= 0:
            raise ValueError(
                "The data prior covariance is singular, so the evidence is "
                "not defined. Regularise it first."
            )
        gram = self.data_space.gram_matrix()
        _, gram_logarithm = np.linalg.slogdet(gram)
        return float(mahalanobis), float(logarithm - gram_logarithm)

    def log_evidence(self, data: Any, /) -> float:
        """``log p(d)``: how well this model explains the data at all.

        The quantity model comparison needs, and the one a posterior cannot
        supply — a posterior is conditional on the model being right.
        """
        mahalanobis, volume = self.evidence_terms(data)
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

        from ..probability.gaussian import GaussianMeasure

        return GaussianMeasure(
            self.target_space,
            expectation=posterior.expectation,
            covariance=posterior.covariance,
            sample=self._centred_sample,
        )
