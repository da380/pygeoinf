"""The problem: a field on ``[0, 1]`` observed at a dozen points.

The model space is ``H^1`` on the interval, piecewise-linear finite elements
over C++ vectors, on which a value at a point is a bounded functional; the
prior is a Matérn-type Gaussian measure on that space, of order one relative
to its metric and so trace class there. A truth drawn from the prior, values
at points drawn uniformly over the interval under Gaussian noise, and two
local averages to ask about. Two scales behind one ``build(scale)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pygeoinf2 import GaussianMeasure, LinearForwardProblem, LinearOperator

from .space import (
    CppSpace,
    local_average_operator,
    matern_measure,
    point_evaluation_operator,
)

# The two properties asked about: averages over these subintervals.
INTERVALS = ((0.2, 0.35), (0.6, 0.85))
# Where the observation points may fall: away from the natural ends.
SPAN = (0.03, 0.97)


@dataclass(frozen=True)
class Scale:
    """One size of the problem, and its physics: every field but the name is
    a command-line option of every script.

    Attributes:
        name: the preset's name, for the printed headings.
        seed: the seed of the points, the truth and the noise.
        n_elements: the elements of the uniform mesh of the interval.
        n_points: the number of observation points.
        noise_std: the standard deviation of the noise on each datum.
        space_length: the length l of the space's inner product
            x^T (M + l^2 K) y, above zero for a value at a point to be
            bounded.
        prior_length: the correlation length of the prior.
        prior_std: the prior's pointwise standard deviation at the middle of
            the interval.
        level: the credibility of the discrepancy level and of the sets the
            prior and the noise are hardened to.
    """

    name: str
    seed: int
    n_elements: int
    n_points: int
    noise_std: float
    space_length: float
    prior_length: float
    prior_std: float
    level: float


QUICK = Scale(
    name="quick",
    seed=2026,
    n_elements=100,
    n_points=12,
    noise_std=0.1,
    space_length=0.08,
    prior_length=0.08,
    prior_std=1.0,
    level=0.95,
)
FULL = Scale(
    name="full",
    seed=2026,
    n_elements=1000,
    n_points=12,
    noise_std=0.1,
    space_length=0.08,
    prior_length=0.08,
    prior_std=1.0,
    level=0.95,
)


@dataclass
class Problem:
    """Everything the parts share."""

    scale: Scale
    space: CppSpace
    prior: GaussianMeasure
    points: list[float]
    evaluate: LinearOperator
    averages: LinearOperator
    noise: GaussianMeasure
    problem: LinearForwardProblem
    truth: Any
    data: np.ndarray
    true_averages: np.ndarray

    @property
    def names(self) -> list[str]:
        """The two properties, by their intervals."""
        return [f"average over [{a:g}, {b:g}]" for a, b in INTERVALS]

    def relative_error(self, model: Any, /) -> float:
        """The error of a model against the truth, relative to the truth, in
        the ``L^2`` norm ``sqrt(x^T M x)``.

        Taken by the C++ vectors directly rather than through the space, so
        that it is not in the tally of what the library asked for; and in
        ``L^2`` rather than the space's norm, which weighs the derivative,
        and a dozen values do not resolve the derivative between them.
        """
        difference = model.copy()
        difference.axpy(-1.0, self.truth)
        mass = self.space.mass
        return float(
            np.sqrt(
                difference.dot_weighted(difference, mass)
                / self.truth.dot_weighted(self.truth, mass)
            )
        )


def build(scale: Scale, /) -> Problem:
    """The space, the prior, the points, the truth, the data and the averages.

    The points, the truth and the noise are drawn from the scale's seed, so
    that a part run alone sees the problem it sees in the sequence.
    """
    rng = np.random.default_rng(scale.seed)
    space = CppSpace(scale.n_elements, scale.space_length)
    prior = matern_measure(
        space, length_scale=scale.prior_length, pointwise_std=scale.prior_std
    )
    points = sorted(rng.uniform(*SPAN, scale.n_points).tolist())
    evaluate = point_evaluation_operator(space, points)
    averages = local_average_operator(space, INTERVALS)
    noise = GaussianMeasure.from_standard_deviation(evaluate.codomain, scale.noise_std)
    problem = LinearForwardProblem(evaluate, error=noise)
    truth, data = problem.synthetic_model_and_data(prior, rng=rng)
    true_averages = np.asarray(averages.codomain.to_components(averages(truth)))
    space.calls.clear()
    return Problem(
        scale=scale,
        space=space,
        prior=prior,
        points=points,
        evaluate=evaluate,
        averages=averages,
        noise=noise,
        problem=problem,
        truth=truth,
        data=data,
        true_averages=true_averages,
    )
