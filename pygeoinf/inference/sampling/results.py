"""
What a population of samplers returns: measures known through their draws.

Both results are absolutely continuous with respect to the prior, with the
density ``-Phi / T`` against it, and both hold the draws that represent them.
They differ in how. A :class:`SampledPosterior` comes from chains: its states
are exact in the long-run limit and correlated with one another. A
:class:`WeightedEnsemble` comes from tempering: its particles are weighted,
and its normalizer estimates the evidence.

A sampled posterior's states and a measure's draws have different
properties, so they are reached by different methods.
:meth:`SampledPosterior.sample` is a fresh draw, independent across calls;
:meth:`SampledPosterior.sample_thinned` reads the stored states, which costs
nothing and is correlated.

Both results fit a reference to what they hold: a Gaussian, or a Gaussian
mixture with as many components as the caller asks for, over a chart of a
few functionals, the run's recorded scalars or a map the caller gives. A
fitted reference is equivalent to the prior by construction, so it serves a
further run as its reference, the geometric presets' say, and it is where
:meth:`SampledPosterior.sample` starts its short chains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from numpy.random import Generator

from ...algebra.operators import Functional, LinearOperator
from ...algebra.spaces import CoordinateSpace, resolve_rng
from ...numerics.chain_diagnostics import (
    effective_sample_size,
    integrated_autocorrelation_time,
    split_rhat,
)
from ...numerics.mixture_fit import fit_gaussian_mixture
from ...numerics.randomized import Estimate
from ...probability.absolutely_continuous import AbsolutelyContinuousMeasure
from ...probability.gaussian import GaussianMeasure
from ..gaussian import GaussianReference, MixtureReference

if TYPE_CHECKING:  # pragma: no cover
    from .corrections import Kernel

__all__ = ["Chain", "ChainDiagnostics", "SampledPosterior", "WeightedEnsemble"]


@dataclass(frozen=True, eq=False)
class Chain:
    """One chain's record, after its burn-in.

    Attributes:
        potential: the likelihood potential ``Phi`` at every step.
        accepted: whether each step's proposal was accepted.
        acceptance: each step's acceptance probability.
        records: the recorded scalars at every step, shape
            ``(steps, count)``, or ``None`` when nothing was recorded.
        positions: the stored states, one every ``thin`` steps.
        step_size: the time step the chain ran at after burn-in.
    """

    potential: np.ndarray
    accepted: np.ndarray
    acceptance: np.ndarray
    records: np.ndarray | None
    positions: tuple
    step_size: float

    @property
    def steps(self) -> int:
        """How many steps were kept."""
        return int(self.potential.size)


@dataclass(frozen=True)
class ChainDiagnostics:
    """The diagnostics of each recorded scalar across the chains.

    The scalars are the potential first, then the recorded ones in order.

    Attributes:
        names: what each scalar is.
        autocorrelation_time: the integrated autocorrelation time of each.
        effective_sample_size: the effective sample size of each.
        rhat: the split, rank-normalized ``R-hat`` of each.
    """

    names: tuple[str, ...]
    autocorrelation_time: np.ndarray
    effective_sample_size: np.ndarray
    rhat: np.ndarray

    @property
    def steps_per_independent_draw(self) -> int:
        """``ceil(2 max tau)``: a lower bound, a slow direction that no
        recorded scalar sees being invisible here."""
        return int(np.ceil(2.0 * float(np.max(self.autocorrelation_time))))


class SampledPosterior(AbsolutelyContinuousMeasure):
    """A posterior known through the chains that sampled it.

    The density against the prior is the target's, ``-Phi / T``, with an
    unknown normalizer. The chains' records give its diagnostics and the
    means of the recorded scalars with their Monte Carlo standard errors,
    and a reference fitted to the recorded scalars, from which
    :meth:`sample` draws.

    Args:
        prior: the prior the density is taken against.
        relative_log_density: ``-Phi / T``.
        chains: the chains' records.
        kernel: the kernel the chains ran, at its starting step.
        record: the map whose components were recorded, if any.
        record_names: names for the recorded scalars.
    """

    def __init__(
        self,
        prior: GaussianMeasure,
        relative_log_density: Functional,
        chains: Sequence[Chain],
        /,
        *,
        kernel: "Kernel",
        record: LinearOperator | None = None,
        record_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__(prior, relative_log_density)
        self._chains = tuple(chains)
        self._kernel = kernel
        self._record = record
        self._draw_reference: GaussianReference | MixtureReference | None = None
        self._draw_steps: int | None = None
        count = 0
        if self._chains and self._chains[0].records is not None:
            count = self._chains[0].records.shape[1]
        if record_names is None:
            record_names = [f"record {k}" for k in range(count)]
        self._record_names = ("potential", *record_names)

    @property
    def chains(self) -> tuple[Chain, ...]:
        """The chains' records."""
        return self._chains

    @property
    def kernel(self) -> "Kernel":
        """The kernel the chains ran."""
        return self._kernel

    @property
    def acceptance_rate(self) -> float:
        """The mean acceptance probability over every kept step."""
        return float(np.mean([chain.acceptance.mean() for chain in self._chains]))

    def scalars(self) -> np.ndarray:
        """The potential and the recorded scalars, shape ``(count, chains, steps)``.

        Returns:
            The array.
        """
        columns = [np.stack([chain.potential for chain in self._chains])]
        if self._chains[0].records is not None:
            records = np.stack([chain.records for chain in self._chains])
            columns.extend(records[:, :, k] for k in range(records.shape[2]))
        return np.stack(columns)

    def diagnostics(self) -> ChainDiagnostics:
        """The autocorrelation time, effective sample size and ``R-hat`` of
        each scalar.

        Returns:
            The diagnostics.
        """
        values = self.scalars()
        return ChainDiagnostics(
            self._record_names,
            np.array([integrated_autocorrelation_time(v) for v in values]),
            np.array([effective_sample_size(v) for v in values]),
            np.array([split_rhat(v) for v in values]),
        )

    def scalar_means(self) -> list[Estimate]:
        """The mean of each scalar, with its Monte Carlo standard error.

        The error is ``sigma sqrt(tau / n)`` over all the chains' kept steps;
        the run length needed for a target error follows from it.

        Returns:
            One ``Estimate`` per scalar, the potential first.
        """
        values = self.scalars()
        estimates = []
        for v in values:
            total = v.size
            tau = integrated_autocorrelation_time(v)
            error = float(np.std(v, ddof=1)) * np.sqrt(tau / total)
            estimates.append(Estimate(float(np.mean(v)), error, total))
        return estimates

    @property
    def stored_positions(self) -> tuple:
        """Every stored state, chain after chain."""
        return tuple(x for chain in self._chains for x in chain.positions)

    def sample_thinned(self, *, rng: Generator | None = None) -> Any:
        """One of the stored states, chosen uniformly.

        No forward evaluation is made. The stored states are exact in the
        long-run limit but correlated with one another, by the chains'
        autocorrelation over the thinning interval; for a draw independent
        of the others, use :meth:`sample`.

        Args:
            rng: the generator for the choice.

        Returns:
            A stored model.

        Raises:
            ValueError: if no states were stored.
        """
        stored = self.stored_positions
        if not stored:
            raise ValueError("No states were stored; run the chains with thin= set.")
        return stored[int(resolve_rng(rng).integers(len(stored)))]

    @property
    def record(self) -> LinearOperator | None:
        """The map whose components were recorded at every step."""
        return self._record

    def fitted_reference(
        self,
        *,
        components: int = 1,
        chart: LinearOperator | None = None,
        rng: Generator | None = None,
    ) -> GaussianReference | MixtureReference:
        """A reference fitted to the run: the prior with a chart's law fitted.

        The chart is the recorded map's components, ``l(u) == R (u - m_0)``,
        read off the records of every kept step; or *chart*, another small
        map, read off the stored states. Its law is a Gaussian, or a
        mixture of *components* Gaussians fitted by expectation-maximization,
        and in every direction outside the chart the reference is the prior,
        so it is equivalent to the prior whatever the fit. How good the fit
        is decides only how fast a sampler built on it mixes, never what it
        samples.

        Args:
            components: how many Gaussians in the chart's law; more than one
                gives a :class:`~pygeoinf.inference.gaussian.MixtureReference`,
                one per mode.
            chart: a linear map from the model space to a small space with
                coordinates; the run's recorded map if omitted.
            rng: the generator for the mixture fit's seeding.

        Returns:
            The reference.

        Raises:
            ValueError: if the run recorded nothing and no chart is given, a
                chart is given and no states were stored, or there are fewer
                kept steps than components.
            TypeError: if the chart's codomain has no coordinates.
        """
        prior = self.reference
        if chart is None:
            if self._record is None:
                raise ValueError(
                    "A fitted reference needs a chart: run the chains with "
                    "record=, or pass chart=."
                )
            chart = self._record
            values = np.concatenate([c.records for c in self._chains])
        else:
            stored = self.stored_positions
            if not stored:
                raise ValueError(
                    "A chart other than the record is read off the stored "
                    "states; run the chains with thin= set."
                )
            values = np.array([chart.codomain.to_components(chart(x)) for x in stored])
        return _fit_reference(prior, chart, values, None, components, rng)

    @property
    def draw_reference(self) -> GaussianReference | MixtureReference:
        """Where :meth:`sample` starts its short chains: a Gaussian fitted to
        the recorded scalars, fitted on first use, unless another was set by
        :meth:`with_draw_reference`."""
        if self._draw_reference is None:
            self._draw_reference = self.fitted_reference()
        return self._draw_reference

    def with_draw_reference(
        self, reference: GaussianReference | MixtureReference, /
    ) -> "SampledPosterior":
        """The same run, drawing from short chains started at another
        reference: a mixture fitted by :meth:`fitted_reference`, for a
        posterior with several modes.

        Args:
            reference: the reference, over this run's prior.

        Returns:
            A new sampled posterior; this one is unchanged.

        Raises:
            TypeError: if the reference is not a ``GaussianReference`` or a
                ``MixtureReference``.
            ValueError: if it is over another prior.
        """
        if not isinstance(reference, (GaussianReference, MixtureReference)):
            raise TypeError(
                "A draw starts from a GaussianReference or a MixtureReference, "
                f"not a {type(reference).__name__}."
            )
        if reference.prior is not self.reference:
            raise ValueError("The reference must be built over the run's prior.")
        result = SampledPosterior(
            self.reference,
            self.relative_log_density,
            self._chains,
            kernel=self._kernel,
            record=self._record,
            record_names=list(self._record_names[1:]),
        )
        result._draw_reference = reference
        result._draw_steps = self._draw_steps
        return result

    @property
    def steps_per_draw(self) -> int:
        """How long each of :meth:`sample`'s chains runs: twice the largest
        integrated autocorrelation time of the recorded scalars, at least
        one."""
        if self._draw_steps is None:
            self._draw_steps = max(1, self.diagnostics().steps_per_independent_draw)
        return self._draw_steps

    def sample(self, *, rng: Generator | None = None) -> Any:
        """A fresh draw, independent of every other.

        A short chain of the run's kernel, at the step its chains adapted
        to, started from a draw of :attr:`draw_reference` and run for
        :attr:`steps_per_draw` steps; its last state is the draw. Each call
        is a new chain, so draws are independent and
        :meth:`~pygeoinf.probability.base.ProbabilityMeasure.samples` runs
        them in parallel. A draw is exact as far as the chain has
        equilibrated in those steps, which the fitted start makes short
        where the reference is close to the posterior; it costs that many
        kernel steps. :meth:`sample_thinned` costs nothing and is
        correlated.

        Args:
            rng: the generator for the start and the chain.

        Returns:
            A model.

        Raises:
            ValueError: as for :meth:`fitted_reference`, when the draw
                reference is fitted on first use, or if the start drawn is
                one at which the potential cannot be evaluated.
        """
        rng = resolve_rng(rng)
        reference = self.draw_reference
        steps = self.steps_per_draw
        step = float(np.mean([chain.step_size for chain in self._chains]))
        kernel = self._kernel.with_step(step)
        state = kernel.initial_state(reference.sample(rng=rng), rng=rng)
        if kernel.target.value(state) == float("inf"):
            raise ValueError(
                "The draw started where the potential cannot be evaluated; the "
                "draw reference reaches outside the posterior's support."
            )
        for _ in range(steps):
            state = kernel.step(state, rng=rng).state
        return state.position

    def __repr__(self) -> str:
        steps = sum(chain.steps for chain in self._chains)
        return f"SampledPosterior({len(self._chains)} chains, {steps} steps)"


class WeightedEnsemble(AbsolutelyContinuousMeasure):
    """Weighted particles, with the log evidence they estimate.

    The density against the prior is the target's, ``-Phi / T``, and the
    normalizer is the tempering path's estimate of ``log Z``, with no
    standard error: one run of a sequential Monte Carlo sampler does not
    estimate its own.

    Args:
        prior: the prior the density is taken against.
        relative_log_density: ``-Phi / T``.
        particles: the particles.
        log_weights: their unnormalized log weights.
        log_normalizer: the estimate of ``log Z``.
        temperatures: the inverse temperatures ``beta`` the path stopped at.
        effective_sizes: the effective sample size before each resampling.
    """

    def __init__(
        self,
        prior: GaussianMeasure,
        relative_log_density: Functional,
        particles: Sequence[Any],
        log_weights: np.ndarray,
        /,
        *,
        log_normalizer: Estimate,
        temperatures: np.ndarray,
        effective_sizes: np.ndarray,
    ) -> None:
        super().__init__(prior, relative_log_density, log_normalizer=log_normalizer)
        self._particles = tuple(particles)
        self._log_weights = np.asarray(log_weights, dtype=float)
        self._temperatures = np.asarray(temperatures, dtype=float)
        self._effective_sizes = np.asarray(effective_sizes, dtype=float)

    @property
    def particles(self) -> tuple:
        """The particles."""
        return self._particles

    @property
    def weights(self) -> np.ndarray:
        """The normalized weights."""
        shifted = np.exp(self._log_weights - self._log_weights.max())
        return shifted / shifted.sum()

    @property
    def effective_sample_size(self) -> float:
        """``1 / sum w^2`` of the normalized weights."""
        return float(1.0 / np.sum(self.weights**2))

    @property
    def temperatures(self) -> np.ndarray:
        """The inverse temperatures of the path, from the first above zero to one."""
        return self._temperatures.copy()

    @property
    def effective_sizes(self) -> np.ndarray:
        """The effective sample size at each stage, before resampling."""
        return self._effective_sizes.copy()

    def fitted_reference(
        self,
        chart: LinearOperator,
        /,
        *,
        components: int = 1,
        rng: Generator | None = None,
    ) -> GaussianReference | MixtureReference:
        """A reference fitted to the weighted particles over a chart.

        As :meth:`SampledPosterior.fitted_reference`, from the particles'
        chart coordinates with their weights.

        Args:
            chart: a linear map from the model space to a small space with
                coordinates.
            components: how many Gaussians in the chart's law.
            rng: the generator for the mixture fit's seeding.

        Returns:
            The reference.

        Raises:
            ValueError: for fewer particles than components.
            TypeError: if the chart's codomain has no coordinates.
        """
        values = np.array(
            [chart.codomain.to_components(chart(x)) for x in self._particles]
        )
        return _fit_reference(
            self.reference, chart, values, self.weights, components, rng
        )

    def weighted_mean(self, function: Any, /) -> float:
        """The weighted mean of a scalar function of the particles.

        Args:
            function: a callable from a model to a number, a functional say.

        Returns:
            The mean.
        """
        values = np.array([float(function(x)) for x in self._particles])
        return float(self.weights @ values)

    def __repr__(self) -> str:
        return (
            f"WeightedEnsemble({len(self._particles)} particles, "
            f"{self._temperatures.size} stages)"
        )


def _fit_reference(
    prior: GaussianMeasure,
    chart: LinearOperator,
    values: np.ndarray,
    weights: np.ndarray | None,
    components: int,
    rng: Generator | None,
) -> GaussianReference | MixtureReference:
    """The prior with the chart's law fitted to weighted values of the chart.

    The chart vectors are the representers of the chart's components,
    ``R* E* e_k``, so that ``(w_k, u) == (R u)_k``; the chart's coordinates
    are taken about the prior's mean, as a reference's are.

    Raises:
        TypeError: if the chart's codomain has no coordinates.
        ValueError: if the chart does not act on the prior's space.
    """
    space = chart.codomain
    if not isinstance(space, CoordinateSpace):
        raise TypeError("A chart's codomain needs coordinates.")
    if chart.domain != prior.domain:
        raise ValueError("The chart must act on the model space.")
    unit = np.identity(space.dim)
    vectors = [chart.adjoint(space.from_components(space.solve_gram(e))) for e in unit]
    centre = space.to_components(chart(prior.expectation))
    points = np.asarray(values, dtype=float).reshape(-1, space.dim) - centre
    fit = fit_gaussian_mixture(points, components, weights=weights, rng=rng)
    if components == 1:
        return GaussianReference(prior, vectors, fit.means[0], fit.covariances[0])
    return MixtureReference(prior, vectors, fit.weights, fit.means, fit.covariances)
