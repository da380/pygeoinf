"""The problem: a field on the sphere observed at ocean points.

The model space is the Sobolev space ``H^2`` of the unit sphere, on which
a value at a point exists, and the prior is a Sobolev measure of order two
on it, a measure of extra regularity whose covariance is trace class there,
calibrated to a pointwise standard deviation of one. The point and set
answers measure size by the prior, so that the one normal operator of the
Bayesian answer is also the one they sweep, and one preconditioner serves
them all. A truth drawn from the prior, point observations wherever a uniform draw of
points did not land on land, Gaussian noise, and three regional averages to
ask about. Two scales behind one ``build(scale=)``: the quick one runs in
seconds, the full one is the point.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pygeoinf2 import GaussianMeasure, LinearForwardProblem, LinearOperator
from pygeoinf2.symmetric_space.sphere import Sobolev

from .common import Counted
from .ocean_mask import area_weights, land_mask, land_polygons, ocean_points

# The regional averages: caps of one radius, the scale's, about named
# centers, two over ocean basins and one over a continent.
REGIONS = (
    ("North Pacific", np.array([20.0, -170.0])),
    ("South Atlantic", np.array([-25.0, -15.0])),
    ("Africa", np.array([5.0, 20.0])),
)
# Where the pointwise standard deviation is read exactly: deep inside two
# continents and in the middle of two oceans.
PROBES = (
    ("central Africa", np.array([5.0, 20.0])),
    ("central Asia", np.array([45.0, 90.0])),
    ("North Pacific", np.array([20.0, -170.0])),
    ("South Atlantic", np.array([-25.0, -15.0])),
)


@dataclass(frozen=True)
class Scale:
    """One size of the problem, and its physics: every field but the name is
    a command-line option of every script.

    Attributes:
        name: the preset's name, for the printed headings.
        seed: the seed of the truth, the points and the noise.
        lmax: the truncation degree of the model space.
        order: the Sobolev order of the model space, above one for a value
            at a point to exist.
        length_scale: the length scale of the space and the prior, on the
            unit sphere (0.06 is about 380 km).
        prior_order: the Sobolev order of the prior on the space, above one,
            half the sphere's dimension, for its covariance to be trace
            class there and its draws functions of the space.
        prior_std: the prior's pointwise standard deviation.
        n_data: the number of ocean observation points.
        noise_std: the standard deviation of the noise on each datum.
        cap_radius: the radius of the caps averaged over, on the unit sphere
            (0.3 is about 1900 km).
        level: the credibility of the discrepancy level and of the sets the
            prior and the noise are hardened to.
        set_power: the power of the prior's covariance that shapes its
            ellipsoid in the set answer, below one half for the ellipsoid to
            survive refinement.
        surrogate_degree: the degree of the coarse sphere of the
            preconditioner's surrogate.
        cg_cap: the iterations the plain solve of part B is capped at.
        std_samples: the posterior samples behind a pointwise standard
            deviation map.
        bump_width: the width of the bump whose resolution part C shows.
        n_jobs: the workers for the posterior samples and the support
            directions; none for serial.
    """

    name: str
    seed: int
    lmax: int
    order: float
    length_scale: float
    prior_order: float
    prior_std: float
    n_data: int
    noise_std: float
    cap_radius: float
    level: float
    set_power: float
    surrogate_degree: int
    cg_cap: int
    std_samples: int
    bump_width: float
    n_jobs: int | None


QUICK = Scale(
    name="quick",
    seed=2026,
    lmax=32,
    order=2.0,
    length_scale=0.15,
    prior_order=2.0,
    prior_std=1.0,
    n_data=400,
    noise_std=0.1,
    cap_radius=0.3,
    level=0.95,
    set_power=0.25,
    surrogate_degree=16,
    cg_cap=400,
    std_samples=32,
    bump_width=0.25,
    n_jobs=None,
)
FULL = Scale(
    name="full",
    seed=2026,
    lmax=128,
    order=2.0,
    length_scale=0.06,
    prior_order=2.0,
    prior_std=1.0,
    n_data=20_000,
    noise_std=0.1,
    cap_radius=0.3,
    level=0.95,
    set_power=0.25,
    surrogate_degree=48,
    cg_cap=500,
    std_samples=64,
    bump_width=0.1,
    n_jobs=8,
)


@dataclass
class Problem:
    """Everything ``build`` makes and the parts use."""

    scale: Scale
    space: Sobolev
    prior: GaussianMeasure
    land: Any
    on_land: np.ndarray
    points: np.ndarray
    counted: Counted
    problem: LinearForwardProblem
    truth: Any
    data: Any
    averages: LinearOperator
    true_averages: np.ndarray
    _surrogate: tuple[Sobolev, LinearOperator] | None = field(default=None, repr=False)

    def surrogate(self) -> tuple[Sobolev, LinearOperator]:
        """The coarse sphere and its dense point-evaluation operator, built once."""
        if self._surrogate is None:
            coarse = self.space.with_degree(self.scale.surrogate_degree)
            self._surrogate = (
                coarse,
                coarse.point_evaluation_operator(self.points, dense=True),
            )
        return self._surrogate

    def relative_error(self, model: Any) -> float:
        """``|m - truth| / |truth|`` in the ``L2`` norm, area-weighted on the grid.

        Not the space's own ``H^2`` norm, which the unresolved high degrees
        weigh more: that would read worse for a field that is visibly right
        wherever there are data.
        """
        weights = area_weights(self.space)
        difference = self.space.grid_values(model) - self.space.grid_values(self.truth)
        return float(
            np.sqrt(
                np.sum(weights * difference**2)
                / np.sum(weights * self.space.grid_values(self.truth) ** 2)
            )
        )


def build(*, scale: Scale = FULL, quiet: bool = False) -> Problem:
    """The space, the prior, the truth, the mask, the observations and the problem."""
    started = time.perf_counter()
    rng = np.random.default_rng([scale.seed, 0])
    say = (lambda *args: None) if quiet else print
    say(f"\n=== The setup ({scale.name} scale) ===")
    space = Sobolev(scale.lmax, scale.order, scale.length_scale)
    prior = space.sobolev_measure(
        scale.prior_order, scale.length_scale, pointwise_std=scale.prior_std
    )
    # A draw has the prior's order less half the dimension in derivatives
    # beyond the space's own.
    extra = scale.prior_order - 1.0
    needed = Sobolev.truncation_degree_for(
        scale.order + scale.prior_order, scale.length_scale, rtol=1e-6
    )
    say(
        f"space: H^{scale.order:g} on the unit sphere, degree {space.lmax}, "
        f"{space.dim} coefficients, grid {space.grid_shape}"
    )
    say(
        f"prior: a Sobolev measure of order {scale.prior_order:g} on "
        f"H^{scale.order:g}, length scale {scale.length_scale} "
        f"({scale.length_scale * 6371:.0f} km), pointwise std {scale.prior_std}; "
        f"its draws have nearly {extra:g} more derivative{'' if extra == 1 else 's'} "
        f"than the space asks, and a degree of {needed} holds their spectrum to "
        f"one part in a million"
    )

    land = land_polygons()
    mask = land_mask(space, land)
    say(
        f"mask: Natural Earth 110m land, "
        f"{np.average(mask, weights=area_weights(space)):.1%} of the area on land"
    )
    points = ocean_points(scale.n_data, land, rng=rng)
    say(f"observations: {len(points)} ocean points, uniform over the sea")

    counted = Counted(space.point_evaluation_operator(points))
    noise = GaussianMeasure.from_standard_deviation(
        counted.operator.codomain, scale.noise_std
    )
    problem = LinearForwardProblem(counted.operator, error=noise)
    truth, data = problem.synthetic_model_and_data(prior, rng=rng)
    values = space.grid_values(truth)
    say(
        f"truth: a draw from the prior, norm {space.norm(truth):.2f}, values from "
        f"{values.min():+.2f} to {values.max():+.2f}; noise std {scale.noise_std}; "
        f"chi-squared of the truth {problem.chi_squared(truth, data):.0f} on "
        f"{len(points)} data"
    )

    averages = space.geodesic_ball_average_operator(
        [center for _, center in REGIONS], scale.cap_radius
    )
    true_averages = averages(truth)
    for (name, _), value in zip(REGIONS, true_averages):
        say(f"  true average over {name}: {value:+.3f}")
    say(f"setup took {time.perf_counter() - started:.1f} s")
    counted.reset()
    return Problem(
        scale,
        space,
        prior,
        land,
        mask,
        points,
        counted,
        problem,
        truth,
        data,
        averages,
        true_averages,
    )
