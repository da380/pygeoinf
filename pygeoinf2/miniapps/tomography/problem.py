"""The problem: travel times through a layered disk, about its reference rays.

A planet of radius one: a mantle on ``[0.55, 1]`` whose velocity rises
linearly with depth from 1 at the surface to 1.8 at its base, over a slow
core of uniform velocity 1.2, so that the rays that turn in the mantle never
enter the core and a shadow lies beyond the farthest of them. Sources and
receivers on the surface; every pair between 20 and 80 degrees apart is a
ray, traced by :mod:`rays`. The model is the slowness perturbation, in
percent of the surface slowness, on a layered disk of two layers, the core
and the mantle, each in ``H^2`` with the prior's covariance the identity on
it, so that the space's own norm is the prior's and the point and set
answers damp towards the norm the Bayesian one assumes. The layers are
independent under the prior, which is what a discontinuity at the core's
edge means. A datum is the travel-time perturbation ``int ds dl`` along a
reference ray, in hundredths of a time unit; the truth is a draw from the
prior and the noise is Gaussian.

Two scales behind one ``build(scale=)``: the quick one runs in seconds, the
full one is the point. A third puts the same planet in a ball, the rays in
its equatorial plane and the angle a longitude: the tracer is radial and
does not change, and every part runs on it unchanged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.sparse import csr_matrix

from pygeoinf2 import (
    DiagonalLinearOperator,
    EuclideanSpace,
    GaussianMeasure,
    LinearForwardProblem,
    LinearOperator,
)
from pygeoinf2 import ball, disk
from pygeoinf2.spectral_element import Layered

from rays import RadialProfile, Ray, linear_gradient

# The structure.
RADIUS = 1.0
CORE_RADIUS = 0.55
SURFACE_VELOCITY = 1.0
BASE_VELOCITY = 1.8
CORE_VELOCITY = 1.2
GRADIENT = (BASE_VELOCITY - SURFACE_VELOCITY) / (RADIUS - CORE_RADIUS)
MANTLE = RadialProfile(
    linear_gradient(SURFACE_VELOCITY, GRADIENT, radius=RADIUS),
    radius=RADIUS,
    bottom=CORE_RADIUS,
)
# The rays kept: far enough apart to sample the mantle below the crust, and
# short of the shadow, which begins where the ray grazing the core emerges.
MIN_DISTANCE = 20.0
MAX_DISTANCE = 80.0

# The model space and the prior: the identity covariance on ``H^2``, scaled
# to the pointwise standard deviation below, in percent of the surface
# slowness; the noise in hundredths of a time unit.
ORDER = 2.0
PRIOR_STD = 1.0
NOISE_STD = 0.05
LEVEL = 0.95

# The properties: the mean perturbation over two polar boxes of the mantle,
# ``(r1, r2, angle1, angle2)`` with the angles in degrees, and over the core.
BOXES = (
    ("well covered", (0.75, 0.9, 60.0, 90.0)),
    ("poorly covered", (0.75, 0.9, 255.0, 285.0)),
)
PROPERTIES = ("well covered", "poorly covered", "core")


@dataclass(frozen=True)
class Scale:
    """One size of the problem."""

    name: str
    mmax: int
    length_scale: float
    sources: tuple[float, ...]
    receivers: tuple[float, ...]
    nodes: int
    std_samples: int
    n_jobs: int | None
    bump_width: float
    geometry: str = "disk"

    @property
    def is_ball(self) -> bool:
        return self.geometry == "ball"


# A continent of stations over the upper half of the planet and three island
# stations below; the sources spread round it, one far out.
QUICK = Scale(
    name="quick",
    mmax=16,
    length_scale=0.2,
    sources=(-40.0, 15.0, 75.0, 135.0, 190.0, 300.0),
    receivers=(*np.linspace(0.0, 150.0, 11), 270.0),
    nodes=201,
    std_samples=128,
    n_jobs=None,
    bump_width=0.12,
)
FULL = Scale(
    name="full",
    mmax=96,
    length_scale=0.1,
    sources=(
        -60.0,
        -35.0,
        -10.0,
        15.0,
        45.0,
        75.0,
        105.0,
        135.0,
        165.0,
        190.0,
        215.0,
        300.0,
    ),
    receivers=(*np.linspace(0.0, 150.0, 61), 235.0, 270.0, 320.0),
    nodes=301,
    std_samples=256,
    n_jobs=None,
    bump_width=0.06,
)
# The ball: the quick network in the equatorial plane, a few more stations,
# and the angular truncation a degree.
BALL = Scale(
    name="ball",
    mmax=12,
    length_scale=0.25,
    sources=QUICK.sources,
    receivers=(*np.linspace(0.0, 150.0, 16), 270.0),
    nodes=201,
    std_samples=128,
    n_jobs=None,
    bump_width=0.15,
    geometry="ball",
)


def network(
    sources: Sequence[float],
    receivers: Sequence[float],
    /,
    *,
    max_distance: float = MAX_DISTANCE,
) -> list[tuple[float, float]]:
    """Every source-receiver pair between :data:`MIN_DISTANCE` and
    ``max_distance`` degrees apart, once: where a source sits at a
    receiver, the ray from it to another receiver at a source is the ray
    back, the same path, and is kept in one direction only."""
    pairs, seen = [], set()
    for source in sources:
        for receiver in receivers:
            apart = abs((receiver - source + 180.0) % 360.0 - 180.0)
            ends = frozenset((round(source % 360.0, 9), round(receiver % 360.0, 9)))
            if MIN_DISTANCE <= apart <= max_distance and ends not in seen:
                seen.add(ends)
                pairs.append((float(source), float(receiver)))
    return pairs


def path_weights(rays: Sequence[Ray], /) -> csr_matrix:
    """The sparse ``(rays, nodes)`` matrix of the path-integral weights, the
    nodes of all the rays stacked in order."""
    counts = [len(ray.weights) for ray in rays]
    columns = np.arange(sum(counts))
    rows = np.repeat(np.arange(len(rays)), counts)
    weights = np.concatenate([ray.weights for ray in rays])
    return csr_matrix((weights, (rows, columns)), shape=(len(rays), sum(counts)))


def lift(scale: Scale, nodes: np.ndarray, /) -> list[tuple[float, ...]]:
    """Points ``(radius, angle)`` of the plane of the rays as points of the
    model space: themselves in a disk, ``(radius, 0, angle)`` on the
    equatorial plane of a ball."""
    nodes = np.asarray(nodes, dtype=float).reshape(-1, 2)
    if scale.is_ball:
        return [(r, 0.0, a) for r, a in nodes]
    return [(r, a) for r, a in nodes]


def composed_operator(
    scale: Scale, space: Any, rays: Sequence[Ray], /
) -> LinearOperator:
    """The path integrals as the composition of point evaluation at every
    node with the sparse weights: what the forward operator is, written as
    it reads. Its point evaluation is a dense ``(nodes, dim)`` matrix, which
    at the full scale would be gigabytes; :func:`path_integral_operator`
    forms the same product ray by ray."""
    nodes = np.concatenate([ray.nodes for ray in rays])
    evaluation = space.point_evaluation_operator(lift(scale, nodes))
    weights = LinearOperator.from_matrix(
        evaluation.codomain,
        EuclideanSpace(len(rays)),
        path_weights(rays),
        form="components",
    )
    return weights @ evaluation


def path_integral_operator(
    scale: Scale, space: Any, rays: Sequence[Ray], /
) -> LinearOperator:
    """The travel-time perturbations ``int ds dl`` along the rays.

    Row ``i`` of its Galerkin matrix is the weighted sum of the basis at the
    nodes of ray ``i``: the derivative components of the path integral,
    which is what point evaluation at the nodes followed by the weights
    gives, formed one ray at a time so that the evaluation at every node is
    never held at once. The adjoint is the library's, from those rows.
    """
    rows = np.empty((len(rays), space.dim))
    for index, ray in enumerate(rays):
        rows[index] = ray.weights @ space.basis_matrix(lift(scale, ray.nodes))
    return LinearOperator.from_matrix(
        space, EuclideanSpace(len(rays)), rows, form="galerkin"
    )


def box_average_row(
    scale: Scale,
    space: Any,
    box: tuple[float, float, float, float],
    /,
    *,
    size: int = 24,
) -> np.ndarray:
    """The derivative components of the mean over a polar box
    ``(r1, r2, angle1, angle2)``, by Gauss-Legendre quadrature.

    In a disk the box is that annular sector, under ``r dr dangle``. In a
    ball it is the sector of a shell that straddles the plane of the rays,
    as far either side of it in latitude as the box is wide in longitude,
    under ``r^2 cos(latitude) dr dlatitude dlongitude``.
    """
    r1, r2, a1, a2 = box
    x, w = np.polynomial.legendre.leggauss(size)

    def nodes(low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
        return 0.5 * (high - low) * (x + 1.0) + low, 0.5 * (high - low) * w

    radii, radial = nodes(r1, r2)
    angles, angular = nodes(a1, a2)
    angular = np.radians(1.0) * angular
    if not scale.is_ball:
        points = [(r, a) for r in radii for a in angles]
        weights = np.einsum("i,j->ij", radial * radii, angular).ravel()
    else:
        half = 0.5 * (a2 - a1)
        latitudes, polar = nodes(-half, half)
        polar = np.radians(1.0) * polar * np.cos(np.radians(latitudes))
        points = [(r, b, a) for r in radii for b in latitudes for a in angles]
        weights = np.einsum("i,j,k->ijk", radial * radii**2, polar, angular).ravel()
    return weights @ space.basis_matrix(points) / weights.sum()


def core_average_row(space: Layered, /) -> np.ndarray:
    """The derivative components of the mean over the core: its layer's
    integral, over the area."""
    row = np.zeros(space.dim)
    core = space.layers[space.index("core")]
    row[space.columns("core")] = (
        core.integral_functional().derivative_components / core.domain_volume
    )
    return row


def property_operator(scale: Scale, space: Layered, /) -> LinearOperator:
    """The three means: over the well-covered box, the poorly covered one and
    the core."""
    rows = [box_average_row(scale, space, box) for _, box in BOXES]
    rows.append(core_average_row(space))
    return LinearOperator.from_matrix(
        space, EuclideanSpace(len(rows)), np.array(rows), form="galerkin"
    )


def model_space(scale: Scale, /) -> Layered:
    """The core and the mantle, each in ``H^2``; the core's angular truncation
    in proportion to its radius, so that both are truncated at one
    wavelength. Disks, or balls."""
    inner = max(2, int(round(CORE_RADIUS / RADIUS * scale.mmax)))
    kind = ball.Layered if scale.is_ball else disk.Layered
    return kind(
        [0.0, CORE_RADIUS, RADIUS],
        [inner, scale.mmax],
        order=ORDER,
        length_scale=scale.length_scale,
        labels=["core", "mantle"],
    )


def identity_prior(space: Layered, /, *, std: float = PRIOR_STD) -> GaussianMeasure:
    """The identity covariance on each layer, scaled so that its pointwise
    standard deviation, nearly uniform over each layer, averages ``std``.

    One diagonal operator over the components of both layers, so that the
    layers are independent and the covariance is a multiple of the space's
    own inner product on each, with its spectrum and its precision in hand,
    and so that the measure lives on the layered space itself, whose
    pointwise algebra its draws and its posteriors keep.
    """
    variances = []
    for layer in space.layers:
        raw = layer.interior_values(layer.pointwise_variance(np.ones(layer.dim)))
        variances.append(np.full(layer.dim, std**2 / float(raw.mean())))
    variances = np.concatenate(variances)
    return GaussianMeasure(
        space,
        covariance_factor=DiagonalLinearOperator(space, np.sqrt(variances)),
        precision=DiagonalLinearOperator(space, 1.0 / variances),
    )


def inner_truncation(layer: Any, /) -> int:
    """A layer's angular truncation: its largest order, or degree."""
    return int(getattr(layer, "lmax", getattr(layer, "mmax", 0)))


@dataclass
class Problem:
    """Everything ``build`` makes and the parts use."""

    scale: Scale
    space: Layered
    prior: GaussianMeasure
    pairs: list[tuple[float, float]]
    rays: list[Ray]
    forward: LinearOperator
    problem: LinearForwardProblem
    truth: Any
    data: np.ndarray
    properties: LinearOperator
    true_properties: np.ndarray

    @property
    def nodes(self) -> np.ndarray:
        """Every node of every ray, stacked, as ``(radius, angle)`` in the
        plane of the rays."""
        return np.concatenate([ray.nodes for ray in self.rays])

    @property
    def tag(self) -> str:
        """What the figures' names begin with: nothing for the disk."""
        return "ball_" if self.scale.is_ball else ""

    def lift(self, points: Any, /) -> list[tuple[float, ...]]:
        """Points ``(radius, angle)`` of the plane of the rays as points of
        the model space."""
        return lift(self.scale, points)

    def relative_error(self, model: Any) -> float:
        """``|m - truth| / |truth|`` over the mantle, in the ``L2`` norm by the
        quadrature of its own part of the grid: where the data are, and in a
        norm the eye reads, where the ``H^2`` norm is dominated by the
        unresolved short wavelengths."""
        index = self.space.index("mantle")
        mantle = self.space.layers[index]
        radial = mantle.basis[0].weights()
        if self.scale.is_ball:
            weights = radial[:, None, None] * mantle.grid.weights[None, :, None]
        else:
            weights = radial[:, None] * mantle.grid.weight
        truth = mantle.interior_values(self.truth[index])
        difference = mantle.interior_values(model[index]) - truth
        return float(
            np.sqrt(np.sum(weights * difference**2) / np.sum(weights * truth**2))
        )


def build(*, scale: Scale = FULL, seed: int = 2026, quiet: bool = False) -> Problem:
    """The structure, the rays, the space, the prior, the truth and the data."""
    started = time.perf_counter()
    rng = np.random.default_rng([seed, 0])
    say = (lambda *args: None) if quiet else print
    say(f"\n=== The setup ({scale.name} scale) ===")
    say(
        f"structure: a mantle on [{CORE_RADIUS}, {RADIUS}] with velocity "
        f"{SURFACE_VELOCITY} + {GRADIENT:.3f} (1 - r), {BASE_VELOCITY} at its base, "
        f"over a core of {CORE_VELOCITY}; the ray grazing the core emerges "
        f"{MANTLE.grazing_distance:.1f} degrees away, and beyond is the shadow"
    )

    t0 = time.perf_counter()
    pairs = network(scale.sources, scale.receivers)
    rays = [
        MANTLE.trace(source, receiver, nodes=scale.nodes) for source, receiver in pairs
    ]
    deepest = min(float(ray.nodes[:, 0].min()) for ray in rays)
    say(
        f"rays: {len(scale.sources)} sources, {len(scale.receivers)} receivers, "
        f"{len(rays)} pairs between {MIN_DISTANCE:g} and {MAX_DISTANCE:g} degrees "
        f"apart, {scale.nodes} nodes each, traced in {time.perf_counter() - t0:.1f} s; "
        f"the deepest turns at r = {deepest:.3f}"
    )

    space = model_space(scale)
    prior = identity_prior(space)
    core, mantle = space.layers
    truncation = "lmax" if scale.is_ball else "mmax"
    say(
        f"space: the layered {scale.geometry}, core {core.dim} and mantle "
        f"{mantle.dim} coefficients ({truncation} {inner_truncation(core)} and "
        f"{inner_truncation(mantle)}), H^{ORDER:g} with "
        f"length scale {scale.length_scale}; prior: the identity covariance on "
        f"it, pointwise std {PRIOR_STD:g}% of the surface slowness, the layers "
        f"independent"
    )

    t0 = time.perf_counter()
    forward = path_integral_operator(scale, space, rays)
    say(
        f"forward: the path integral along each ray, {len(rays)} rows assembled "
        f"from {len(rays) * scale.nodes} node evaluations in "
        f"{time.perf_counter() - t0:.1f} s"
    )
    noise = GaussianMeasure.from_standard_deviation(forward.codomain, NOISE_STD)
    problem = LinearForwardProblem(forward, error=noise)
    truth, data = problem.synthetic_model_and_data(prior, rng=rng)
    signal = forward(truth)
    say(
        f"truth: a draw from the prior; travel-time perturbations of std "
        f"{np.std(signal):.3f} against noise of {NOISE_STD}; chi-squared of the "
        f"truth {problem.chi_squared(truth, data):.1f} on {len(rays)} data"
    )

    properties = property_operator(scale, space)
    true_properties = properties(truth)
    for name, value in zip(PROPERTIES, true_properties):
        say(f"  true mean perturbation, {name}: {value:+.3f}")
    say(f"setup took {time.perf_counter() - started:.1f} s")
    return Problem(
        scale,
        space,
        prior,
        pairs,
        rays,
        forward,
        problem,
        truth,
        data,
        properties,
        true_properties,
    )
