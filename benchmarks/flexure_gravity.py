"""Flexure and gravity: two fields through one chain, on a degenerate prior.

Mag et al. (submitted), section 3.2: a density anomaly and a basal traction
on a sphere, observed through the geoid and the topography they jointly
produce, as in example 22. The prior is an ``L²`` amplitude bound on each
field -- an ellipsoid whose covariance grows with the degree, so the set
is bounded at every truncation and converges as the truncation is refined
-- and the noise set is the chi-squared ellipsoid of the Gaussian error in
whitened coordinates. The properties are paired cap averages of the two
fields.

What is measured, in this order:

1. at each truncation, the support values of the primal with the spectral
   solver, with conjugate gradients and with the dual, their brackets,
   applications and wall time, and the S-procedure ellipsoid's weight;
2. the contact-point hull of the two caps of one field, an inner bound,
   against the polytope of the same directions;
3. the change of the bounds between consecutive truncations, which must
   fall to a few per cent between the finest two.

Run from the repository root::

    poetry run python benchmarks/flexure_gravity.py --truncations 24 32 48 64

The slow tests run :func:`build` and :func:`measure` at reduced truncations.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    Counter,
    axis_directions,
    counted,
    parser,
    planar_directions,
    table,
    timed,
)

from pygeoinf2.algebra.direct_sum import (  # noqa: E402
    BlockDiagonalLinearOperator,
    BlockLinearOperator,
    ColumnLinearOperator,
    DirectSum,
)
from pygeoinf2.algebra.operators import LinearOperator  # noqa: E402
from pygeoinf2.algebra.spaces import EuclideanSpace  # noqa: E402
from pygeoinf2.geometry import Ellipsoid, FeasiblePropertySet  # noqa: E402
from pygeoinf2.geometry.feasible import Dual  # noqa: E402
from pygeoinf2.numerics.solvers import SpectralFamilySolver  # noqa: E402
from pygeoinf2.probability.gaussian import GaussianMeasure  # noqa: E402
from pygeoinf2.symmetric_space.sphere import Sobolev  # noqa: E402
from pygeoinf2.traits import Traits  # noqa: E402

GRAVITY = 9.8
NEWTON = 6.6743e-11
LAYER = 2.0e4
RADIUS = 6.371e6
POISSON = 0.25
DENSITY_LOAD, DENSITY_MANTLE = 3000.0, 3400.0
RIGIDITY = 1.0e23
GEOID_DEGREE, TOPOGRAPHY_DEGREE = 12, 20
CAPS = [np.array([20.0, -60.0]), np.array([-35.0, 140.0])]
CAP_RADIUS = 0.15 * RADIUS
# The cap averages are reported in units of each field's expected size, so
# that a direction mixing the two fields means something and the property
# space is not a mix of kg/m³ and Pa.
UNITS = (10.0, 2.0e6)


def chain(lmax: int, /) -> tuple[Sobolev, DirectSum, LinearOperator]:
    """The field space, the pair space and the forward operator at a truncation."""
    X = Sobolev(lmax, 2.0, 0.1 * RADIUS, radius=RADIUS)
    model_space = DirectSum([X, X], labels=("density", "traction"))
    flexure = X.inverse_flexural_operator(
        RIGIDITY, POISSON, (DENSITY_MANTLE - DENSITY_LOAD) * GRAVITY
    )
    density_of, traction_of = model_space.projection(0), model_space.projection(1)
    step_one = ColumnLinearOperator([density_of, flexure @ traction_of])
    degrees = X.degrees.astype(float)
    density_to_potential = X.spectral_operator(
        -4.0 * np.pi * NEWTON * RADIUS * LAYER / (2.0 * degrees + 1.0)
    )
    flexure_to_potential = X.spectral_operator(
        -4.0
        * np.pi
        * NEWTON
        * RADIUS
        * DENSITY_LOAD
        * (
            1.0
            + (DENSITY_MANTLE / DENSITY_LOAD - 1.0)
            * (1.0 - LAYER / RADIUS) ** (degrees + 2.0)
        )
        / (2.0 * degrees + 1.0)
    )
    step_two = BlockLinearOperator(
        [
            [density_to_potential, flexure_to_potential],
            [LinearOperator.zero(X), LinearOperator.identity(X)],
        ]
    )
    geoid = (-1.0 / GRAVITY) * X.coefficient_operator(lmax=GEOID_DEGREE)
    topography = X.coefficient_operator(lmax=TOPOGRAPHY_DEGREE)
    observation = BlockDiagonalLinearOperator([geoid, topography])
    return X, model_space, observation @ step_two @ step_one


def amplitude_prior(
    model_space: DirectSum, amplitudes: tuple[float, float], /
) -> Ellipsoid:
    """``{ (ρ, τ) : |ρ|²_{L²} / a_ρ² + |τ|²_{L²} / a_τ² <= 1 }`` on the Sobolev pair space.

    The ``L²`` inner product on a Sobolev space is the spectral operator
    ``1 / metric``, so the precision is diagonal and so is its inverse.
    """
    blocks, inverses = [], []
    for X, amplitude in zip(model_space.subspaces, amplitudes):
        blocks.append(X.spectral_operator(1.0 / (amplitude**2 * X.metric_values)))
        inverses.append(X.spectral_operator(amplitude**2 * X.metric_values))
    traits = Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
    return Ellipsoid(
        model_space,
        BlockDiagonalLinearOperator(blocks).with_traits(traits),
        covariance=BlockDiagonalLinearOperator(inverses).with_traits(traits),
    )


def l2_norm(X: Sobolev, field: Any, /) -> float:
    return float(
        np.sqrt(
            X.inner_product(X.spectral_operator(1.0 / X.metric_values)(field), field)
        )
    )


@dataclass
class Problem:
    """The data, made once at the finest truncation, and the pieces at one truncation."""

    lmax: int
    field_space: Sobolev
    model_space: DirectSum
    forward: Any
    counter: Counter
    prior: Ellipsoid
    noise: Ellipsoid
    target: Any
    data: Any
    truth_values: np.ndarray


@dataclass
class Data:
    """What is shared across truncations: the data and the bounds that made them feasible."""

    data: Any
    error: GaussianMeasure
    amplitudes: tuple[float, float]
    truth: Any
    truth_lmax: int


def make_data(
    *,
    truth_lmax: int,
    signal_to_noise: float = 20.0,
    amplitude_factor: float = 1.8,
    level: float = 0.95,
    rng: np.random.Generator,
) -> Data:
    """A truth at the finest truncation and data it fits inside both sets.

    The noise is white within each data block, with a standard deviation
    of the block's root-mean-square signal over ``signal_to_noise``. The
    amplitude bound on each field is ``amplitude_factor`` times the
    truth's ``L²`` norm, with the pair's combined level then
    ``2 / amplitude_factor²``, inside the ellipsoid for any factor above
    ``sqrt(2)``.
    """
    if amplitude_factor <= np.sqrt(2.0):
        raise ValueError(
            "The amplitude factor must exceed sqrt(2) for the truth to be feasible."
        )
    X, model_space, forward = chain(truth_lmax)
    prior = GaussianMeasure.from_product(
        [
            X.sobolev_measure(2.0, 5.0e5, pointwise_std=10.0),
            X.sobolev_measure(1.25, 1.0e5, pointwise_std=2.0e6),
        ]
    )
    truth = prior.sample(rng=rng)
    clean = forward(truth)
    error = GaussianMeasure.from_product(
        [
            GaussianMeasure.from_standard_deviation(
                block_space,
                float(np.sqrt(np.mean(np.asarray(block) ** 2))) / signal_to_noise,
            )
            for block_space, block in zip(forward.codomain.subspaces, clean)
        ]
    )
    # An error sample scaled to lie inside the credible ellipsoid, so that the
    # truth is feasible by construction rather than with probability 0.95.
    data_space = forward.codomain
    noise_set = error.credible_set(level=level)
    raw = error.sample(rng=rng)
    scale = 0.7 / np.sqrt(noise_set.mahalanobis_squared(raw))
    data = data_space.add(clean, data_space.scale(scale, raw))
    amplitudes = tuple(
        amplitude_factor * l2_norm(X, field) for field in (truth[0], truth[1])
    )
    return Data(
        data=data,
        error=error,
        amplitudes=amplitudes,
        truth=truth,
        truth_lmax=truth_lmax,
    )


def build(lmax: int, shared: Data, /, *, level: float = 0.95) -> Problem:
    """The problem at one truncation, on the shared data."""
    X, model_space, forward = chain(lmax)
    forward, counter = counted(forward)
    cap = X.geodesic_ball_average_operator(CAPS, CAP_RADIUS, dense=True)
    target = BlockDiagonalLinearOperator([cap * (1.0 / unit) for unit in UNITS])
    # The truth's property, on the truth's own truncation.
    X_truth = Sobolev(shared.truth_lmax, 2.0, 0.1 * RADIUS, radius=RADIUS)
    cap_truth = X_truth.geodesic_ball_average_operator(CAPS, CAP_RADIUS, dense=True)
    truth_values = np.concatenate(
        [
            cap_truth.codomain.to_components(cap_truth(field)) / unit
            for field, unit in zip(shared.truth, UNITS)
        ]
    )
    return Problem(
        lmax=lmax,
        field_space=X,
        model_space=model_space,
        forward=forward,
        counter=counter,
        prior=amplitude_prior(model_space, shared.amplitudes),
        noise=shared.error.credible_set(level=level),
        target=target,
        data=shared.data,
        truth_values=truth_values,
    )


def directions_for(space: Any, /, *, planar: int = 12) -> list[Any]:
    """The axes, the plane of one field's two caps, and the plane of one cap's two fields."""
    return (
        axis_directions(space)
        + planar_directions(space, 0, 1, planar)
        + planar_directions(space, 0, 2, planar)
    )


def measure(
    problems: list[Problem],
    *,
    n_jobs: int = 4,
    planar: int = 12,
    dual_iterations: int = 1500,
    quiet: bool = False,
) -> dict[str, Any]:
    """Every table of the module docstring, returned by name and printed."""
    out: dict[str, Any] = {}

    def say(text: str = "") -> None:
        if not quiet:
            print(text)

    # 1. Per truncation.
    rows, values_by_truncation = [], {}
    space = problems[0].target.codomain
    directions = directions_for(space, planar=planar)
    for problem in problems:
        reference = None
        for label, solver, via in (
            ("primal, spectral", SpectralFamilySolver(), "primal"),
            ("primal, CG", None, "primal"),
            (
                f"dual ({dual_iterations} iterations), spectral",
                SpectralFamilySolver(),
                Dual(max_iterations=dual_iterations),
            ),
        ):
            feasible = FeasiblePropertySet(
                problem.forward,
                problem.target,
                problem.prior,
                problem.data,
                noise=problem.noise,
                solver=solver,
            )
            problem.counter.reset()
            sweep, seconds = timed(
                lambda: feasible.support_sweep(directions, via=via, n_jobs=n_jobs)
            )
            values = sweep.values
            if reference is None:
                reference = values
                values_by_truncation[problem.lmax] = values
                truth_inside = all(
                    float(
                        space.inner_product(
                            q, space.from_components(problem.truth_values)
                        )
                    )
                    <= v + 1e-9 * abs(v)
                    for q, v in zip(directions, values)
                )
                ellipsoid, ellipsoid_seconds = timed(
                    lambda: feasible.outer_ellipsoid_result(criterion="volume")
                )
            gaps = np.array([r.gap for r in sweep.results])
            gaps = gaps[np.isfinite(gaps)]
            rows.append(
                [
                    problem.lmax,
                    problem.model_space.dim,
                    label,
                    float(
                        np.max(
                            np.abs(values - reference)
                            / np.maximum(np.abs(reference), 1e-300)
                        )
                    ),
                    float(np.max(gaps)) if gaps.size else np.inf,
                    float(np.mean([r.evaluations for r in sweep.results])),
                    problem.counter.total,
                    seconds,
                    sweep.converged,
                    truth_inside if label.startswith("primal, spectral") else "",
                    f"{ellipsoid.weight:.3f} ({ellipsoid_seconds:.1f} s)"
                    if label.startswith("primal, spectral")
                    else "",
                ]
            )
    out["per_truncation"] = rows
    say(
        "## Each truncation: the algorithms, brackets, cost, and the S-procedure weight"
    )
    say(
        table(
            [
                "lmax",
                "dim M",
                "algorithm",
                "max rel. diff to primal/spectral",
                "max bracket",
                "mean evaluations",
                "applications",
                "s",
                "converged",
                "truth inside",
                "S-procedure θ",
            ],
            rows,
        )
    )
    say()

    # 2. The contact-point hull against the polytope, on the finest truncation.
    finest = problems[-1]
    feasible = FeasiblePropertySet(
        finest.forward,
        finest.target,
        finest.prior,
        finest.data,
        noise=finest.noise,
        solver=SpectralFamilySolver(),
    )
    pair = LinearOperator.from_matrix(
        space, EuclideanSpace(2), np.eye(2, space.dim), form="components"
    )
    projected = feasible.push_forward(pair)
    plane = planar_directions(projected.domain, 0, 1, max(planar, 8))
    hull, hull_seconds = timed(lambda: projected.inner_hull(plane))
    polytope = projected.polytope(plane)
    inside = all(
        polytope.contains(
            hull.project(projected.domain.random(rng=np.random.default_rng(k))),
            rtol=1e-6,
        )
        for k in range(20)
    )
    rows = [[len(plane), inside, hull_seconds]]
    out["hull"] = rows
    say(
        "## The contact-point hull of the density caps (inner) inside the polytope (outer)"
    )
    say(table(["directions", "hull inside polytope", "hull s"], rows))
    say()

    # 3. Convergence under refinement.
    rows = []
    truncations = sorted(values_by_truncation)
    for coarse, fine in zip(truncations, truncations[1:]):
        a, b = values_by_truncation[coarse], values_by_truncation[fine]
        rows.append(
            [
                f"{coarse} → {fine}",
                float(np.max(np.abs(b - a) / np.maximum(np.abs(b), 1e-300))),
                float(np.mean(np.abs(b - a) / np.maximum(np.abs(b), 1e-300))),
            ]
        )
    out["refinement"] = rows
    say("## The bounds between consecutive truncations")
    say(table(["truncations", "max rel. change", "mean rel. change"], rows))
    say()
    return out


def main() -> None:
    arguments = parser(__doc__.split("\n")[0])
    arguments.add_argument(
        "--truncations", type=int, nargs="+", default=[24, 32, 48, 64]
    )
    arguments.add_argument(
        "--planar", type=int, default=12, help="directions per plane"
    )
    arguments.add_argument(
        "--snr", type=float, default=20.0, help="signal to noise per datum"
    )
    arguments.add_argument(
        "--dual-iterations", type=int, default=1500, help="the dual's bundle budget"
    )
    settings = arguments.parse_args()
    rng = np.random.default_rng(settings.seed)
    shared = make_data(
        truth_lmax=max(settings.truncations), signal_to_noise=settings.snr, rng=rng
    )
    print(
        f"truth at degree {shared.truth_lmax}; L² amplitudes {shared.amplitudes[0]:.3g} "
        f"kg/m³·m and {shared.amplitudes[1]:.3g} Pa·m; data space of dimension "
        f"{shared.data[0].size + shared.data[1].size}"
    )
    print()
    problems = [build(lmax, shared) for lmax in settings.truncations]
    measure(
        problems,
        n_jobs=settings.n_jobs,
        planar=settings.planar,
        dual_iterations=settings.dual_iterations,
    )


if __name__ == "__main__":
    main()
