"""
The posterior of a Gaussian-mixture prior, under a linear Gaussian likelihood.

Exact, and the reason mixtures are worth having in an inversion at all. With
prior ``sum_k w_k N(m_k, C_k)`` and data ``d = A m + e``:

.. code-block:: text

    posterior  sum_k w'_k N(m_k^post, C_k^post)
    w'_k       proportional to  w_k p(d | k)

Each component is updated by the usual Kalman formulas -- one
:class:`~pygeoinf.inference.gaussian.LinearGaussianInversion` per component,
with its solver and preconditioning -- and the weights are reweighted
by each component's *evidence*, which is exactly what
:meth:`~pygeoinf.inference.gaussian.LinearGaussianInversion.log_evidence`
computes. Nothing new is needed for the hard part.

The result is a **multimodal** posterior in closed form. A single Gaussian
posterior can only say "here, roughly this wide"; a mixture posterior can say
"either here or there, and the data prefer here four to one" — and the weights
are the quantitative form of that preference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..algebra.operators import LinearOperator
from ..algebra.spaces import HilbertSpace
from ..probability.mixture import GaussianMixture
from .estimators import MeasureEstimator, _PushedMeasureEstimator
from .gaussian import LinearGaussianInversion
from .problem import LinearForwardProblem

if TYPE_CHECKING:  # pragma: no cover
    from ..numerics.randomized import Estimate

__all__ = ["LinearGaussianMixtureInversion"]


class LinearGaussianMixtureInversion(MeasureEstimator):
    """The posterior for a mixture prior: components updated, weights rescored.

    Unlike :class:`~pygeoinf.inference.gaussian.LinearGaussianInversion` this
    is **not** a pair of a fixed covariance and a moving mean: the weights
    depend on the data, so the shape of the posterior does too. That is the
    point — a mixture posterior can change which mode it prefers when the data
    change, and a data-independent covariance is precisely what cannot.
    """

    def __init__(
        self,
        problem: LinearForwardProblem,
        prior: GaussianMixture,
        /,
        *,
        solver: Any = None,
        evidence: Any = None,
        n_jobs: int | None = None,
    ) -> None:
        """
        Args:
            problem: the forward problem.
            prior: a Gaussian mixture on the model space.
            solver: how each component's normal operator is inverted. Passed
                through unchanged, so a factory taking the normal operator
                works here too and is applied per component, and so does
                ``ModelSpace(solver)``.
            evidence: keyword arguments for each component's
                :meth:`~pygeoinf.inference.gaussian.LinearGaussianInversion.log_evidence`,
                as a dict. The direct determinant, taken by default wherever
                the data space can be assembled, is exact and the weights are
                then exact; ``{"method": "stochastic", "samples": 200}`` keeps
                the weight update matrix-free on a data space too large for
                that, at the cost of weights that carry the estimate's error. A
                generator under ``"rng"`` is split into one child per
                component, so that the components draw independent probes
                and a parallel run gives the numbers a serial one does.
            n_jobs: workers for the components, each one solve and one
                evidence. Serial by default. A worker answers on its own copy
                of the component's inversion, and the volume terms it
                computes are kept on this one, so a later call does not
                compute them again.

        Raises:
            TypeError: if the prior is not a ``GaussianMixture``.
            ValueError: if the prior is not defined on the model space.
        """
        if not isinstance(prior, GaussianMixture):
            raise TypeError(
                f"The prior must be a GaussianMixture; got "
                f"{type(prior).__name__}. For a single Gaussian use "
                f"LinearGaussianInversion."
            )
        if prior.domain != problem.model_space:
            raise ValueError("The prior is not defined on the model space.")
        self._problem = problem
        self._prior = prior
        self._evidence = {} if evidence is None else dict(evidence)
        self._n_jobs = n_jobs
        self._inversions = [
            LinearGaussianInversion(problem, component, solver=solver)
            for component in prior.components
        ]

    # ----------------------------------------------------------------- #
    #                             The pieces                            #
    # ----------------------------------------------------------------- #

    @property
    def prior(self) -> GaussianMixture:
        """The mixture prior."""
        return self._prior

    @property
    def forward_problem(self) -> LinearForwardProblem:
        """The forward problem."""
        return self._problem

    @property
    def inversions(self) -> list[LinearGaussianInversion]:
        """One inversion per component, each a full estimator in its own right.

        Each component is an ordinary linear Gaussian inversion: its normal
        operator, its Kalman operator, its evidence and its preconditioning
        are all reachable through it.
        """
        return list(self._inversions)

    @property
    def data_space(self) -> HilbertSpace:
        """The space the data live on."""
        return self._problem.data_space

    @property
    def property_space(self) -> HilbertSpace:
        """The space the posterior lives on."""
        return self._problem.model_space

    # ----------------------------------------------------------------- #
    #                            The posterior                          #
    # ----------------------------------------------------------------- #

    def _answers(self, data: Any, /, *, posterior: bool) -> list[tuple[Any, Any]]:
        """Each component's posterior, when asked for, and log evidence."""
        from numpy.random import Generator

        from ..parallel import parallel_map

        settings = [dict(self._evidence) for _ in self._inversions]
        rng = self._evidence.get("rng")
        if isinstance(rng, Generator):
            for setting, child in zip(settings, rng.spawn(len(settings))):
                setting["rng"] = child
        tasks = [
            (inversion, data, setting, posterior)
            for inversion, setting in zip(self._inversions, settings)
        ]
        results = parallel_map(_component_answer, tasks, n_jobs=self._n_jobs)
        for inversion, (_, _, volumes) in zip(self._inversions, results):
            inversion._log_determinants.update(volumes)
        return [(measure, evidence) for measure, evidence, _ in results]

    def _component_log_evidences(self, data: Any, /) -> list[Any]:
        """Each component's log evidence, as an ``Estimate``."""
        return [evidence for _, evidence in self._answers(data, posterior=False)]

    def log_evidence_terms(self, data: Any, /) -> np.ndarray:
        """``log p(d | k)`` for each component.

        The quantity that reweights the mixture, and separately the quantity a
        model comparison between the components would report. A component with
        a wide prior is penalized for it here, which is what stops the widest
        component from always winning.

        Args:
            data: the observations.

        Returns:
            The values, one per component. Each component's standard error,
            nonzero on the stochastic route, is on its own inversion's
            :meth:`~pygeoinf.inference.gaussian.LinearGaussianInversion.log_evidence`,
            and :meth:`log_evidence` carries their combination.
        """
        return np.array(
            [estimate.value for estimate in self._component_log_evidences(data)]
        )

    def weights(self, data: Any, /) -> np.ndarray:
        """The posterior mixing weights.

        ``w_k p(d | k)``, normalized — computed through a softmax on the logs,
        because the evidences of competing components routinely differ by
        hundreds of nats and the ratio is the only part that matters.
        """
        return self._weights_from(self.log_evidence_terms(data))

    def _weights_from(self, terms: np.ndarray, /) -> np.ndarray:
        from scipy.special import softmax

        with np.errstate(divide="ignore"):
            prior_logs = np.log(self._prior.weights)
        return softmax(prior_logs + terms)

    def log_evidence(self, data: Any, /) -> "Estimate":
        """``log p(d)`` for the mixture as a whole.

        The mixture's own evidence, for comparing it against a different prior
        entirely. It is *not* the largest component's: a mixture that hedges
        pays for the components that turned out to be wrong, and this is where
        that shows up.

        Args:
            data: the observations.

        Returns:
            An :class:`~pygeoinf.numerics.randomized.Estimate`. Its standard
            error combines the components' in quadrature, each weighted by
            its posterior weight, the derivative of the log-sum-exp; zero on
            the dense route.
        """
        from scipy.special import logsumexp, softmax

        from ..numerics.randomized import Estimate

        estimates = self._component_log_evidences(data)
        values = np.array([estimate.value for estimate in estimates])
        errors = np.array([estimate.standard_error for estimate in estimates])
        with np.errstate(divide="ignore"):
            logs = np.log(self._prior.weights) + values
        weights = softmax(logs)
        return Estimate(
            float(logsumexp(logs)),
            float(np.sqrt(np.sum((weights * errors) ** 2))),
            max(estimate.samples for estimate in estimates),
        )

    def __call__(self, data: Any) -> GaussianMixture:
        """The posterior mixture for this data.

        Each component's posterior and evidence come from one pass over the
        components, sharing each component's solve between the two.
        """
        answers = self._answers(data, posterior=True)
        terms = np.array([evidence.value for _, evidence in answers])
        return GaussianMixture(
            [measure for measure, _ in answers], weights=self._weights_from(terms)
        )

    def push_forward(self, operator: LinearOperator, /) -> MeasureEstimator:
        """The posterior for a property, component by component.

        The weights do not change: they are decided by the data through the
        evidence, and a property map is applied afterwards. Each component's
        *measure* is pushed forward, not its estimator: the two give the
        identical measure, but a pushed estimator would solve the normal
        equations again for data each component has already inverted.

        Args:
            operator: the property map ``T``, from the model space.

        Returns:
            The measure estimator of the property, whose answers are
            Gaussian mixtures on the property space.

        Raises:
            ValueError: if ``T`` does not act on the model space.
        """
        return _PushedMeasureEstimator(self, operator)


def _component_answer(task: tuple[Any, Any, dict, bool]) -> tuple[Any, Any, dict]:
    """One component's work: its posterior if asked for, its log evidence,
    and the volume terms its inversion holds afterwards, which a worker's
    copy must hand back for them to be kept."""
    inversion, data, evidence, posterior = task
    measure = inversion(data) if posterior else None
    log_evidence = inversion.log_evidence(data, **evidence)
    return measure, log_evidence, dict(inversion._log_determinants)
