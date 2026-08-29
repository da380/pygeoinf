"""
The posterior of a Gaussian-mixture prior, under a linear Gaussian likelihood.

Exact, and the reason mixtures are worth having in an inversion at all. With
prior ``sum_k w_k N(m_k, C_k)`` and data ``d = A m + e``:

.. code-block:: text

    posterior  sum_k w'_k N(m_k^post, C_k^post)
    w'_k       proportional to  w_k p(d | k)

Each component is updated by the usual Kalman formulas — one
:class:`~pygeoinf2.inference.gaussian.LinearGaussianInversion` per component,
with everything §23 through §28 gives them — and the weights are reweighted by
each component's *evidence*, which is exactly what
:meth:`~pygeoinf2.inference.gaussian.LinearGaussianInversion.log_evidence`
computes. Nothing new is needed for the hard part.

The result is a **multimodal** posterior in closed form. A single Gaussian
posterior can only say "here, roughly this wide"; a mixture posterior can say
"either here or there, and the data prefer here four to one" — and the weights
are the quantitative form of that preference.

See DESIGN.md section 31.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..algebra.spaces import HilbertSpace
from ..probability.mixture import GaussianMixture
from .estimators import MeasureEstimator
from .gaussian import LinearGaussianInversion
from .normal import Formalism
from .problem import LinearForwardProblem

__all__ = ["LinearGaussianMixtureInversion"]


class LinearGaussianMixtureInversion(MeasureEstimator):
    """The posterior for a mixture prior: components updated, weights rescored.

    Unlike :class:`~pygeoinf2.inference.gaussian.LinearGaussianInversion` this
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
        formalism: Formalism = "data_space",
        evidence: Any = None,
    ) -> None:
        """
        Args:
            problem: the forward problem.
            prior: a Gaussian mixture on the model space.
            solver: how each component's normal operator is inverted. Passed
                through unchanged, so a factory taking the normal operator
                works here too and is applied per component.
            formalism: which space to assemble each component in.
            evidence: keyword arguments for the evidence calculation, as a
                dict — ``{"method": "stochastic", "samples": 200}`` to keep the
                weight update matrix-free on a large data space.
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
        self._inversions = [
            LinearGaussianInversion(
                problem, component, solver=solver, formalism=formalism
            )
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
    def problem(self) -> LinearForwardProblem:
        """The forward problem."""
        return self._problem

    @property
    def inversions(self) -> list[LinearGaussianInversion]:
        """One inversion per component, each a full estimator in its own right.

        Exposed because a component *is* an ordinary linear Gaussian inversion:
        its normal operator, its gain, its evidence and its preconditioning are
        all reachable, and there is no reason to hide them behind the mixture.
        """
        return list(self._inversions)

    @property
    def data_space(self) -> HilbertSpace:
        """The space the data live on."""
        return self._problem.data_space

    @property
    def target_space(self) -> HilbertSpace:
        """The space the posterior lives on."""
        return self._problem.model_space

    # ----------------------------------------------------------------- #
    #                            The posterior                          #
    # ----------------------------------------------------------------- #

    def log_evidence_terms(self, data: Any, /) -> np.ndarray:
        """``log p(d | k)`` for each component.

        The quantity that reweights the mixture, and separately the quantity a
        model comparison between the components would report. A component with
        a wide prior is penalised for it here, which is what stops the widest
        component from always winning.
        """
        return np.array(
            [
                inversion.log_evidence(data, **self._evidence)
                for inversion in self._inversions
            ]
        )

    def weights(self, data: Any, /) -> np.ndarray:
        """The posterior mixing weights.

        ``w_k p(d | k)``, normalised — computed through a softmax on the logs,
        because the evidences of competing components routinely differ by
        hundreds of nats and the ratio is the only part that matters.
        """
        from scipy.special import softmax

        with np.errstate(divide="ignore"):
            prior_logs = np.log(self._prior.weights)
        return softmax(prior_logs + self.log_evidence_terms(data))

    def log_evidence(self, data: Any, /) -> float:
        """``log p(d)`` for the mixture as a whole.

        The mixture's own evidence, for comparing it against a different prior
        entirely. It is *not* the largest component's: a mixture that hedges
        pays for the components that turned out to be wrong, and this is where
        that shows up.
        """
        from scipy.special import logsumexp

        with np.errstate(divide="ignore"):
            prior_logs = np.log(self._prior.weights)
        return float(logsumexp(prior_logs + self.log_evidence_terms(data)))

    def __call__(self, data: Any) -> GaussianMixture:
        """The posterior mixture for this data."""
        return GaussianMixture(
            [inversion(data) for inversion in self._inversions],
            weights=self.weights(data),
        )

    def push_forward(self, operator: Any, /) -> "_PushedMixture":
        """The posterior for a property, component by component.

        The weights do not change: they are decided by the data through the
        evidence, and a property map is applied afterwards. So this pushes each
        component's estimator forward and leaves the scoring alone.
        """
        return _PushedMixture(self, operator)


class _PushedMixture(MeasureEstimator):
    """A mixture inversion seen through a property operator."""

    def __init__(self, base: LinearGaussianMixtureInversion, operator: Any, /) -> None:
        self._base = base
        self._operator = operator

    @property
    def data_space(self) -> HilbertSpace:
        """The space the data live on."""
        return self._base.data_space

    @property
    def target_space(self) -> HilbertSpace:
        """The property space."""
        return self._operator.codomain

    def weights(self, data: Any, /) -> np.ndarray:
        """The same weights as the model-space mixture: the data decide them."""
        return self._base.weights(data)

    def __call__(self, data: Any) -> GaussianMixture:
        """The posterior mixture on the property space.

        Each component's *measure* is pushed forward, not its estimator.
        The two give the identical measure -- same mean, same covariance
        ``T C T*``, same sampler -- but a pushed estimator re-solves the normal
        equations, while the measure is already the answer. With the weights
        needing the same solve again, that was ``2K`` solves for ``K``
        components where ``K`` will do.
        """
        return GaussianMixture(
            [
                inversion(data).push_forward(self._operator)
                for inversion in self._base.inversions
            ],
            weights=self._base.weights(data),
        )

    def push_forward(self, operator: Any, /) -> "_PushedMixture":
        """A further property of the same posterior."""
        return _PushedMixture(self._base, operator @ self._operator)
