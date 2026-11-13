"""
The posterior known through draws: a sampler run on a forward problem.

:class:`MonteCarloEstimator` is the measure estimator of the exact posterior
of a forward problem with a Gaussian prior, linear or not. Beside
:class:`~pygeoinf.inference.gaussian.LinearGaussianInversion`, the posterior
in closed form for a linear problem, and
:class:`~pygeoinf.inference.laplace.LaplaceApproximation`, the Gaussian about
the mode, it is the posterior known through its draws. What is computed is
fixed; how is the sampler, a knob as a solver or an optimizer is elsewhere::

    sampler = Chains(HMC(step=0.2, length=5), steps=2000, burn_in=500, rng=rng)
    posterior = MonteCarloEstimator(problem, prior, sampler=sampler)(data)

The sampler (:mod:`pygeoinf.sampling`) knows nothing of the problem: the
estimator builds the target for the data, the likelihood potential against
a reference, and hands it over. The reference is the Gaussian the sampler's
moves solve exactly, the prior by default. A Gaussian fitted to the
posterior makes the chains move at its scale, and given as an estimator it
is fitted to the data at each call::

    laplace = LaplaceApproximation.from_problem(problem, prior, rank=50)
    estimator = MonteCarloEstimator(problem, prior, sampler=sampler, reference=laplace)

A move that follows the state, the local MALA, moves about the Laplace
Gaussian at every state, which the estimator builds from the same Laplace
approximation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from ..algebra.operators import LinearOperator, Operator
from ..algebra.spaces import HilbertSpace
from ..probability.gaussian import GaussianMeasure
from ..probability.reference import GaussianReference, MixtureReference
from ..sampling.population import Sampler
from ..sampling.target import Target
from .estimators import MeasureEstimator, _PushedMeasureEstimator
from .gaussian import LinearGaussianInversion
from .laplace import LaplaceApproximation
from .problem import ForwardProblem

if TYPE_CHECKING:  # pragma: no cover
    from ..geometry.sets import Subset
    from ..sampling.results import SampledPosterior, WeightedEnsemble

__all__ = ["MonteCarloEstimator"]


class MonteCarloEstimator(MeasureEstimator):
    """The posterior of a forward problem, sampled: data to a measure known
    through its draws.

    Called on data, it builds the target, the likelihood potential
    ``Phi == chi^2 / 2`` against the reference, and runs the sampler on it.
    It returns what the sampler returns: a
    :class:`~pygeoinf.sampling.results.SampledPosterior` from
    :class:`~pygeoinf.sampling.population.Chains`, or a
    :class:`~pygeoinf.sampling.results.WeightedEnsemble`, with the evidence,
    from :class:`~pygeoinf.sampling.population.SMC`. Both are absolutely
    continuous with respect to the prior, with the density ``-Phi`` against
    it.

    Args:
        problem: the forward problem, with a Gaussian error measure; with a
            differentiable forward operator for a sampler that follows the
            gradient.
        prior: the Gaussian prior on the model space.
        sampler: how the posterior is sampled: a
            :class:`~pygeoinf.sampling.population.Chains` or an
            :class:`~pygeoinf.sampling.population.SMC` run of a move.
        reference: the Gaussian the sampler's moves solve exactly. The prior
            if omitted; a
            :class:`~pygeoinf.probability.reference.GaussianReference` or
            :class:`~pygeoinf.probability.reference.MixtureReference` over
            the prior, used as it is; or a
            :class:`~pygeoinf.inference.gaussian.LinearGaussianInversion` or
            :class:`~pygeoinf.inference.laplace.LaplaceApproximation` on the
            prior, whose ``reference`` is built for the data at each call. A
            move that follows the state needs a Laplace approximation, whose
            reference at each state it moves about.
        constraint: a subset of the model space outside which the posterior
            has no mass; a proposal outside it is rejected.
        record: a small map from the model space to a space with
            coordinates, linear or not, whose components a run of chains
            records at every step.
        record_names: names for the recorded components.

    Raises:
        TypeError: if the sampler is not a ``Sampler``, or the reference is
            none of the kinds above.
        ValueError: if the prior is not on the model space, the reference is
            over another prior, the sampler follows the gradient and the
            forward operator has no derivative, or the sampler's move
            follows the state and the reference is not a Laplace
            approximation.
    """

    def __init__(
        self,
        problem: ForwardProblem,
        prior: GaussianMeasure,
        /,
        *,
        sampler: Sampler,
        reference: Any = None,
        constraint: "Subset | None" = None,
        record: Operator | None = None,
        record_names: Sequence[str] | None = None,
    ) -> None:
        if prior.domain != problem.model_space:
            raise ValueError("The prior is not defined on the model space.")
        if not isinstance(sampler, Sampler):
            raise TypeError(
                "The sampler runs a move, Chains(HMC(step=0.2, length=5), "
                f"steps=1000) say; got {type(sampler).__name__}."
            )
        if sampler.uses_gradient and not problem.forward_operator.has_derivative:
            raise ValueError(
                "The sampler follows the gradient, and the forward operator has "
                "no derivative; PCN needs none."
            )
        _check_reference(reference, prior)
        if sampler.is_local and not isinstance(reference, LaplaceApproximation):
            raise ValueError(
                "The sampler's move follows the state, about the Laplace Gaussian "
                "at each one: give a LaplaceApproximation as the reference."
            )
        self._problem = problem
        self._prior = prior
        self._sampler = sampler
        self._reference = reference
        self._constraint = constraint
        self._record = record
        self._record_names = None if record_names is None else list(record_names)

    @property
    def forward_problem(self) -> ForwardProblem:
        """The forward problem."""
        return self._problem

    @property
    def prior(self) -> GaussianMeasure:
        """The prior."""
        return self._prior

    @property
    def sampler(self) -> Sampler:
        """How the posterior is sampled."""
        return self._sampler

    @property
    def reference(self) -> Any:
        """The reference as given: ``None`` for the prior, a measure, or an
        estimator whose reference is built for the data."""
        return self._reference

    @property
    def data_space(self) -> HilbertSpace:
        """Where the data live."""
        return self._problem.data_space

    @property
    def property_space(self) -> HilbertSpace:
        """The model space: the sampler draws the model itself."""
        return self._problem.model_space

    def target(self, data: Any, /) -> Target:
        """The residual potential for the data, against the reference.

        What a call hands the sampler; built here for a run made by hand, or
        to inspect the kernel a move builds on it.

        Args:
            data: the observations.

        Returns:
            The target.
        """
        potential = self._problem.potential(data)
        reference = self._reference
        if self._sampler.is_local:
            return Target(
                potential,
                self._prior,
                constraint=self._constraint,
                local=_LocalReference(reference, data),
            )
        if reference is None:
            reference = self._prior
        elif isinstance(reference, (LinearGaussianInversion, LaplaceApproximation)):
            reference = reference.reference(data)
        return Target(potential, reference, constraint=self._constraint)

    def __call__(
        self, data: Any, /, *, start: Any = None
    ) -> "SampledPosterior | WeightedEnsemble":
        """Sample the posterior for the data.

        Args:
            data: the observations.
            start: where chains start: a model, or a measure each chain
                draws its start from; the reference if omitted. Tempering
                always starts from the prior.

        Returns:
            The sampled posterior, or the weighted ensemble.

        Raises:
            ValueError: as for the sampler.
        """
        return self._sampler(
            self.target(data),
            start=start,
            record=self._record,
            record_names=self._record_names,
        )

    def push_forward(self, operator: LinearOperator, /) -> MeasureEstimator:
        """The estimator of the law of ``T m`` rather than of ``m``.

        Each sampled posterior is pushed through ``T``. To follow a property
        along the chains, with its diagnostics, record it instead.

        Args:
            operator: the property map ``T``, from the model space.

        Returns:
            The measure estimator of the property.

        Raises:
            ValueError: if ``T`` does not act on the model space.
        """
        return _PushedMeasureEstimator(self, operator)

    def __repr__(self) -> str:
        return f"MonteCarloEstimator(sampler={self._sampler!r})"


class _LocalReference:
    """The Laplace reference at a model, for the data: a reference that
    follows the state, as a callable a target can hold and a worker
    process can be sent."""

    def __init__(self, laplace: LaplaceApproximation, data: Any, /) -> None:
        self._laplace = laplace
        self._data = data

    def __call__(self, model: Any, /) -> GaussianReference:
        return self._laplace.reference(self._data, at=model)


def _check_reference(reference: Any, prior: GaussianMeasure, /) -> None:
    """Refuse a reference of the wrong kind, or over another prior.

    Raises:
        TypeError: for a reference that is none of the kinds offered.
        ValueError: for one over another prior.
    """
    if reference is None:
        return
    if isinstance(reference, (GaussianReference, MixtureReference)):
        over = reference.prior
    elif isinstance(reference, LaplaceApproximation):
        over = reference.estimator.prior
    elif isinstance(reference, LinearGaussianInversion):
        over = reference.prior
    elif reference is prior:
        return
    else:
        raise TypeError(
            "A reference is a GaussianReference or a MixtureReference, or a "
            "LinearGaussianInversion or LaplaceApproximation whose reference is "
            f"built for the data; got {type(reference).__name__}."
        )
    if over is not prior:
        raise ValueError("The reference must be built over the estimator's prior.")
