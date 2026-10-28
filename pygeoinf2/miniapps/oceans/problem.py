"""The problem: a field on the sphere observed at ocean points.

A Sobolev prior on ``H^2`` of the unit sphere, a truth drawn from it, point
observations wherever a uniform draw of points did not land on land, Gaussian
noise, and three regional averages to ask about. Two scales behind one
``build(scale=)``: the quick one runs in seconds, the full one is the point.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pygeoinf2 import GaussianMeasure, LinearForwardProblem, LinearOperator
from pygeoinf2.symmetric_space.sphere import Sobolev

from common import Counted
from ocean_mask import area_weights, land_mask, land_polygons, ocean_points

NOISE_STD = 0.1
PRIOR_STD = 1.0
ORDER = 2.0
LEVEL = 0.95

# The regional averages: caps of one radius about named centers, two over
# ocean basins and one over a continent. Radii are physical, on the unit
# sphere (0.3 is about 1900 km).
REGIONS = (
    ("North Pacific", np.array([20.0, -170.0])),
    ("South Atlantic", np.array([-25.0, -15.0])),
    ("Africa", np.array([5.0, 20.0])),
)
CAP_RADIUS = 0.3
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
    """One size of the problem."""

    name: str
    lmax: int
    n_data: int
    length_scale: float
    surrogate_degree: int
    spectral_rank: int
    cg_cap: int
    std_samples: int
    n_jobs: int | None
    bump_width: float


QUICK = Scale(
    name="quick",
    lmax=32,
    n_data=400,
    length_scale=0.15,
    surrogate_degree=16,
    spectral_rank=60,
    cg_cap=400,
    std_samples=32,
    n_jobs=None,
    bump_width=0.25,
)
FULL = Scale(
    name="full",
    lmax=128,
    n_data=20_000,
    length_scale=0.06,
    surrogate_degree=48,
    spectral_rank=200,
    cg_cap=500,
    std_samples=64,
    n_jobs=8,
    bump_width=0.1,
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
        """``|m - truth| / |truth|`` in the space's norm."""
        space = self.space
        return space.norm(space.subtract(model, self.truth)) / space.norm(self.truth)


def build(*, scale: Scale = FULL, seed: int = 2026, quiet: bool = False) -> Problem:
    """The space, the prior, the truth, the mask, the observations and the problem."""
    started = time.perf_counter()
    rng = np.random.default_rng([seed, 0])
    say = (lambda *args: None) if quiet else print
    say(f"\n=== The setup ({scale.name} scale) ===")
    space = Sobolev(scale.lmax, ORDER, scale.length_scale)
    prior = space.sobolev_measure(ORDER, scale.length_scale, pointwise_std=PRIOR_STD)
    needed = Sobolev.truncation_degree_for(2.0 * ORDER, scale.length_scale, rtol=1e-6)
    say(
        f"space: H^{ORDER:g} on the unit sphere, degree {space.lmax}, "
        f"{space.dim} coefficients, grid {space.grid_shape}"
    )
    say(
        f"prior: Sobolev measure of order {ORDER:g} on top of the space's, length "
        f"scale {scale.length_scale} ({scale.length_scale * 6371:.0f} km), pointwise "
        f"std {PRIOR_STD}; a degree of {needed} holds its spectrum to one part in a "
        f"million"
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
        counted.operator.codomain, NOISE_STD
    )
    problem = LinearForwardProblem(counted.operator, error=noise)
    truth, data = problem.synthetic_model_and_data(prior, rng=rng)
    values = space.grid_values(truth)
    say(
        f"truth: a draw from the prior, norm {space.norm(truth):.2f}, values from "
        f"{values.min():+.2f} to {values.max():+.2f}; noise std {NOISE_STD}; "
        f"chi-squared of the truth {problem.chi_squared(truth, data):.0f} on "
        f"{len(points)} data"
    )

    averages = space.geodesic_ball_average_operator(
        [center for _, center in REGIONS], CAP_RADIUS
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
