"""
Gaussian mixtures: a Gaussian whose parameters are themselves random.

A mixture couples a *parameterised* Gaussian, ``theta -> N(m(theta),
C(theta))``, with a measure on the parameter. The parameter space is
low-dimensional in practice and often finite — a handful of candidate
correlation lengths, a discrete choice between two geological scenarios — and
that is what makes the whole thing tractable, because a finite parameter
measure gives a mixture with closed forms for everything.

What it buys is **multimodality**, which a single Gaussian cannot express at
all. A prior saying "either a smooth model or a rough one, and I do not know
which" is a two-component mixture, and the data then decide: under a linear
Gaussian likelihood the posterior is again a mixture, with the *same*
components updated in the usual way and the weights reweighted by each
component's evidence. That is exact, not approximate, and it is the reason this
is worth having rather than a sampling scheme.

.. code-block:: text

    prior      sum_k w_k N(m_k, C_k)
    posterior  sum_k w'_k N(m_k^post, C_k^post)
    w'_k       proportional to  w_k p(d | component k)

and ``p(d | component k)`` is the evidence of §26, which is already computed
matrix-free.

See DESIGN.md section 31.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from numpy.random import Generator

from ..algebra.operators import LinearOperator
from ..traits import Traits
from .base import ProbabilityMeasure
from .gaussian import GaussianMeasure

__all__ = ["GaussianMixture"]


def _resolve_rng(rng: Generator | None) -> Generator:
    return np.random.default_rng() if rng is None else rng


class GaussianMixture[X](ProbabilityMeasure[X]):
    """A finite mixture of Gaussian measures on one space.

    The components need not share anything but their domain: different means,
    different covariances, different ranks. What they must share is the space,
    since a mixture is a statement about one random vector.
    """

    def __init__(
        self,
        components: Sequence[GaussianMeasure[X]],
        weights: Any = None,
        /,
    ) -> None:
        """
        Args:
            components: the Gaussian components, all on the same space.
            weights: the mixing weights. Normalised here, so they may be given
                unnormalised — which is what a reweighting produces. Equal
                weights if omitted.
        """
        components = list(components)
        if not components:
            raise ValueError("A mixture needs at least one component.")
        domain = components[0].domain
        for index, component in enumerate(components):
            if component.domain != domain:
                raise ValueError(
                    f"Component {index} lives on {component.domain!r}, but the "
                    f"first lives on {domain!r}. A mixture is a statement about "
                    f"one random vector, so they must share a space."
                )
        super().__init__(domain)

        if weights is None:
            weights = np.full(len(components), 1.0 / len(components))
        else:
            weights = np.asarray(weights, dtype=float)
            if weights.shape != (len(components),):
                raise ValueError(
                    f"{weights.size} weights for {len(components)} components."
                )
            if np.any(weights < 0.0):
                raise ValueError("The weights must be non-negative.")
            total = weights.sum()
            if total <= 0.0:
                raise ValueError("The weights must not all be zero.")
            weights = weights / total
        self._components = components
        self._weights = weights

    # ----------------------------------------------------------------- #
    #                            Construction                           #
    # ----------------------------------------------------------------- #

    @classmethod
    def from_family(
        cls,
        build: Callable[[Any], GaussianMeasure[X]],
        parameters: Sequence[Any],
        /,
        *,
        weights: Any = None,
    ) -> "GaussianMixture[X]":
        """A mixture from a parameterised Gaussian and a finite parameter set.

        The literal reading of "a parameterised Gaussian measure coupled with a
        distribution on the parameter space", when that distribution is
        discrete: *build* is the family and *parameters* carries its support.

        Args:
            build: ``theta -> N(m(theta), C(theta))``.
            parameters: the parameter values in the support.
            weights: their probabilities. Uniform if omitted.
        """
        return cls([build(parameter) for parameter in parameters], weights)

    @classmethod
    def from_parameter_samples(
        cls,
        build: Callable[[Any], GaussianMeasure[X]],
        parameters: ProbabilityMeasure,
        /,
        *,
        count: int = 32,
        rng: Generator | None = None,
    ) -> "GaussianMixture[X]":
        """A mixture from a *continuous* parameter measure, by sampling it.

        The mixture over a continuous parameter is an integral, and this is the
        Monte Carlo estimate of it: draw parameter values, weight them equally.
        Honest about being an approximation to a different object, and
        affordable exactly because the parameter space is small — a handful of
        draws covers one or two dimensions in a way it never would cover the
        model space.

        Args:
            build: ``theta -> N(m(theta), C(theta))``.
            parameters: a measure on the parameter space, which must be
                samplable.
            count: how many parameter values to draw.
            rng: the generator for those draws.
        """
        if not parameters.can_sample:
            raise ValueError(
                "The parameter measure must be samplable to discretise the "
                "mixture over it. Supply the support explicitly with "
                "from_family if it is finite."
            )
        if count < 1:
            raise ValueError(f"At least one parameter draw is needed, got {count}.")
        generator = _resolve_rng(rng)
        return cls([build(parameters.sample(rng=generator)) for _ in range(count)])

    # ----------------------------------------------------------------- #
    #                             The pieces                            #
    # ----------------------------------------------------------------- #

    @property
    def components(self) -> list[GaussianMeasure[X]]:
        """The Gaussian components."""
        return list(self._components)

    @property
    def weights(self) -> np.ndarray:
        """The mixing weights, normalised."""
        return self._weights.copy()

    def __len__(self) -> int:
        return len(self._components)

    def __repr__(self) -> str:
        return (
            f"GaussianMixture({len(self._components)} components on {self._domain!r})"
        )

    def with_weights(self, weights: Any, /) -> "GaussianMixture[X]":
        """The same components, weighted differently.

        What a data update produces: the components move, but a reweighting
        alone is the whole of what a *model comparison* does to a mixture.
        """
        return GaussianMixture(self._components, weights)

    # ----------------------------------------------------------------- #
    #                             Sampling                              #
    # ----------------------------------------------------------------- #

    @property
    def can_sample(self) -> bool:
        """Only if every component with non-zero weight can be."""
        return all(
            component.can_sample
            for component, weight in zip(self._components, self._weights)
            if weight > 0.0
        )

    def sample(self, *, rng: Generator | None = None) -> X:
        """Choose a component by its weight, then draw from it.

        Exact, and the whole of what makes a mixture easy to sample even when
        it is hard to write down.
        """
        generator = _resolve_rng(rng)
        index = int(generator.choice(len(self._components), p=self._weights))
        return self._components[index].sample(rng=generator)

    # ----------------------------------------------------------------- #
    #                              Moments                              #
    # ----------------------------------------------------------------- #

    @property
    def expectation(self) -> X:
        """``sum_k w_k m_k``."""
        total = self._domain.zero()
        for component, weight in zip(self._components, self._weights):
            if weight > 0.0:
                total = self._domain.add(
                    total, self._domain.scale(weight, component.expectation)
                )
        return total

    @property
    def covariance(self) -> LinearOperator[X, X]:
        """The law of total covariance, ``E[C | k] + Cov(m | k)``.

        .. code-block:: text

            C = sum_k w_k C_k + sum_k w_k (m_k - mbar) (m_k - mbar)*

        The second term is what a single Gaussian cannot have: the spread
        *between* the components, which is where a mixture's multimodality
        lives. It has rank at most ``K - 1``, so it is built as a low-rank
        factor rather than assembled — the components are few, which is the
        whole premise.
        """
        mean = self.expectation
        pieces = [
            weight * component.covariance
            for component, weight in zip(self._components, self._weights)
            if weight > 0.0
        ]
        within = pieces[0]
        for piece in pieces[1:]:
            within = within + piece

        offsets = [
            self._domain.scale(
                float(np.sqrt(weight)),
                self._domain.subtract(component.expectation, mean),
            )
            for component, weight in zip(self._components, self._weights)
            if weight > 0.0
        ]
        factor = LinearOperator.from_vectors(self._domain, offsets)
        between = factor @ factor.adjoint
        return (within + between).with_traits(
            Traits.SELF_ADJOINT | Traits.POSITIVE_SEMIDEFINITE
        )

    # ----------------------------------------------------------------- #
    #                              Density                              #
    # ----------------------------------------------------------------- #

    @property
    def has_log_density(self) -> bool:
        """True when every weighted component has one."""
        return all(
            component.has_log_density
            for component, weight in zip(self._components, self._weights)
            if weight > 0.0
        )

    def log_density(self, x: X) -> float:
        """``log sum_k w_k p_k(x)``, by log-sum-exp.

        Each component contributes its *fully normalised* log density. The
        constant a component's own :meth:`~GaussianMeasure.log_density` omits
        depends on its covariance, so it differs between components and cannot
        be left out of a sum over them: dropping it makes a broad component
        look as tall at its centre as a narrow one.

        Summed in the exponent rather than by exponentiating and adding, since
        a mixture's whole point is that one component may be many orders of
        magnitude more likely than another at a given point.
        """
        from scipy.special import logsumexp

        live = self._weights > 0.0
        if not self.has_log_density:
            raise NotImplementedError(
                "Some component has no log density, so the mixture has none. "
                "Each needs a precision; with_regularized_inverse supplies one."
            )
        densities = np.array(
            [
                component.log_density(x) + component.log_normalising_constant()
                for component, weight in zip(self._components, self._weights)
                if weight > 0.0
            ]
        )
        return float(logsumexp(densities, b=self._weights[live]))

    # ----------------------------------------------------------------- #
    #                          Transformations                          #
    # ----------------------------------------------------------------- #

    def _combine_affine(
        self, operator: LinearOperator, translation: X | None
    ) -> "GaussianMixture":
        """A mixture stays a mixture under an affine map, component by component.

        The weights do not move: which component a draw came from is not
        changed by what is then done to the draw.
        """
        return GaussianMixture(
            [
                component.affine_map(operator, translation=translation)
                for component in self._components
            ],
            self._weights,
        )

    def _combine_scale(self, alpha: float) -> "GaussianMixture":
        return GaussianMixture(
            [component * alpha for component in self._components], self._weights
        )

    def marginal_probabilities(self, x: X, /) -> np.ndarray:
        """Which component ``x`` probably came from.

        The posterior over the component label given the vector, by Bayes.
        Useful in its own right — it is how a mixture answers "which scenario
        is this?" — and it is what a classification built on one would report.
        """
        from scipy.special import softmax

        if not self.has_log_density:
            raise NotImplementedError(
                "Component probabilities need each component's density."
            )
        logs = np.full(len(self._components), -np.inf)
        for index, (component, weight) in enumerate(
            zip(self._components, self._weights)
        ):
            if weight > 0.0:
                # The normalising constant is per-component, so it does not
                # cancel in the softmax the way a shared one would.
                logs[index] = (
                    np.log(weight)
                    + component.log_density(x)
                    + component.log_normalising_constant()
                )
        return softmax(logs)
