"""The problem: travel times through a layered disk, about its reference rays.

A planet of radius one: a mantle on ``[0.55, 1]`` whose velocity rises
linearly with depth from 1 at the surface to 1.8 at its base, over a slow
core of uniform velocity 1.2, so that the rays that turn in the mantle never
enter the core and a shadow lies beyond the farthest of them. Sources and
receivers on the surface; every pair between 20 and 80 degrees apart is a
ray, traced by :mod:`.rays`. The model is the slowness perturbation, in
percent of the surface slowness, on a layered disk of two layers, the core
and the mantle, each in ``H^2``, on which a value at a ray's node exists.
The prior is a Sobolev measure of order two on each layer, a measure of
extra regularity whose covariance is trace class there, its amplitude set
so that its pointwise standard deviation, nearly uniform, averages the
scale's. The layers are independent under the prior, which is what a
discontinuity at the core's edge means. A datum is the travel-time
perturbation ``int ds dl`` along a
reference ray, in hundredths of a time unit; the truth is a draw from the
prior and the noise is Gaussian.

Two scales behind one ``build(scale=)``: the quick one runs in seconds, the
full one is the point. The same planet in a ball, with its stations spread
over the sphere, is built in :mod:`ball` from the pieces here: the layered
space, the prior, the path integrals and the means over boxes serve a disk
and a ball alike.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.sparse import csr_matrix

from pygeoinf2 import (
    EuclideanSpace,
    GaussianMeasure,
    LinearForwardProblem,
    LinearOperator,
)
from pygeoinf2 import ball, disk
from pygeoinf2.spectral_element import Layered

from .rays import RadialProfile, Ray, linear_gradient

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
# The properties: the mean perturbation over two polar boxes of the mantle,
# ``(r1, r2, angle1, angle2)`` with the angles in degrees, and over the core.
BOXES = (
    ("well covered", (0.75, 0.9, 60.0, 90.0)),
    ("poorly covered", (0.75, 0.9, 255.0, 285.0)),
)
PROPERTIES = ("well covered", "poorly covered", "core")


@dataclass(frozen=True)
class Scale:
    """One size of the problem, and its physics: every field but the name,
    the geometry and the candidates is a command-line option of every
    script.

    The model space is the slowness perturbation, in percent of the surface
    slowness, on the layered disk of the core and the mantle, each in
    ``H^order``; the prior is a Sobolev measure of order ``prior_order`` on
    each layer, scaled to a pointwise standard deviation; the noise is in
    hundredths of a time unit.

    Attributes:
        name: the preset's name, for the printed headings.
        seed: the seed of the truth and the noise.
        mmax: the mantle's angular truncation; the core's is in proportion to
            its radius.
        order: the Sobolev order of each layer of the model space, above one
            in the disk and three halves in the ball for a value at a point
            to exist.
        length_scale: the length scale of the space and the prior.
        prior_order: the Sobolev order of the prior on each layer, above
            one in the disk and three halves in the ball, half the
            dimension, for its covariance to be trace class there and its
            draws functions of the space.
        prior_std: the prior's pointwise standard deviation, in percent of
            the surface slowness, as its average over each layer.
        noise_std: the standard deviation of the noise on each travel time,
            in hundredths of a time unit.
        level: the credibility of the discrepancy level and of the sets the
            prior and the noise are hardened to.
        sources: the sources' angles on the surface, in degrees.
        receivers: the receivers' angles on the surface, in degrees.
        min_distance: the shortest ray kept, in degrees between its ends: far
            enough apart to sample the mantle below the crust.
        max_distance: the longest ray kept, in degrees between its ends:
            short of the shadow, which begins where the ray grazing the core
            emerges.
        nodes: the nodes along each ray.
        std_samples: the posterior samples behind a pointwise standard
            deviation map.
        bump_width: the width of the bumps whose resolution part B shows.
        candidates: part E's candidate angles for the plume, in degrees: the
            axes of the conduits it may rise through.
        plume_width: the width of part E's plume, across its conduit.
        plume_peak: the plume's peak, in multiples of the prior's pointwise
            standard deviation.
        background: the size of the draw from the shared covariance beneath
            part E's true plume.
        redraws: how many times part E redraws the data at each noise level
            of its sweep.
        n_jobs: the workers for the posterior samples, the support
            directions and the mixture's components; none for serial.
        geometry: ``"disk"``, or ``"ball"`` for part F.
    """

    name: str
    seed: int
    mmax: int
    order: float
    length_scale: float
    prior_order: float
    prior_std: float
    noise_std: float
    level: float
    sources: tuple[Any, ...]
    receivers: tuple[Any, ...]
    min_distance: float
    max_distance: float
    nodes: int
    std_samples: int
    bump_width: float
    candidates: tuple[float, ...]
    plume_width: float
    plume_peak: float
    background: float
    redraws: int
    n_jobs: int | None
    geometry: str = "disk"

    @property
    def is_ball(self) -> bool:
        return self.geometry == "ball"


# A continent of stations over the upper half of the planet and three island
# stations below; the sources spread round it, one far out. The candidate
# angles of part E's plume: two under the continent, where the rays cross in
# every direction; one at its edge, under moderate coverage; and two opposite
# it, where a handful of rays pass. The quick scale keeps three of them.
QUICK = Scale(
    name="quick",
    seed=2026,
    mmax=16,
    order=2.0,
    length_scale=0.2,
    prior_order=2.0,
    prior_std=1.0,
    noise_std=0.05,
    level=0.95,
    sources=(-40.0, 15.0, 75.0, 135.0, 190.0, 300.0),
    receivers=(*(float(a) for a in np.linspace(0.0, 150.0, 11)), 270.0),
    min_distance=20.0,
    max_distance=80.0,
    nodes=201,
    std_samples=128,
    bump_width=0.12,
    candidates=(75.0, 330.0, 270.0),
    plume_width=0.12,
    plume_peak=3.0,
    background=0.5,
    redraws=8,
    n_jobs=None,
)
FULL = dataclasses.replace(
    QUICK,
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
    receivers=(
        *(float(a) for a in np.linspace(0.0, 150.0, 61)),
        235.0,
        270.0,
        320.0,
    ),
    nodes=301,
    std_samples=256,
    bump_width=0.06,
    candidates=(75.0, 120.0, 330.0, 200.0, 270.0),
    plume_width=0.06,
    redraws=32,
)


def network(
    sources: Sequence[float],
    receivers: Sequence[float],
    /,
    *,
    min_distance: float,
    max_distance: float,
) -> list[tuple[float, float]]:
    """Every source-receiver pair between ``min_distance`` and
    ``max_distance`` degrees apart, once: where a source sits at a
    receiver, the ray from it to another receiver at a source is the ray
    back, the same path, and is kept in one direction only."""
    pairs, seen = [], set()
    for source in sources:
        for receiver in receivers:
            apart = abs((receiver - source + 180.0) % 360.0 - 180.0)
            ends = frozenset((round(source % 360.0, 9), round(receiver % 360.0, 9)))
            if min_distance <= apart <= max_distance and ends not in seen:
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
    """Ray nodes as points of the model space: ``(radius, angle)`` in a disk,
    ``(radius, latitude, longitude)`` in a ball."""
    width = 3 if scale.is_ball else 2
    nodes = np.asarray(nodes, dtype=float).reshape(-1, width)
    return [tuple(float(value) for value in node) for node in nodes]


def composed_operator(
    scale: Scale, space: Any, rays: Sequence[Ray], /
) -> LinearOperator:
    """The path integrals as the composition of point evaluation at every
    node with the sparse weights: what the forward operator is, written as
    it reads. Its point evaluation as a dense ``(nodes, dim)`` matrix would
    be gigabytes at the full scale, where the library evaluates locally
    instead; :func:`path_integral_operator` forms the same product ray by
    ray as rows."""
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
    box: tuple[float, ...],
    /,
    *,
    size: int = 24,
) -> np.ndarray:
    """The derivative components of the mean over a box, by Gauss-Legendre
    quadrature.

    In a disk the box is an annular sector ``(r1, r2, angle1, angle2)``,
    under ``r dr dangle``. In a ball it is the sector of a shell ``(r1, r2,
    latitude1, latitude2, longitude1, longitude2)``, under ``r^2
    cos(latitude) dr dlatitude dlongitude``. Angles in degrees.
    """
    x, w = np.polynomial.legendre.leggauss(size)

    def nodes(low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
        return 0.5 * (high - low) * (x + 1.0) + low, 0.5 * (high - low) * w

    degree = np.radians(1.0)
    if not scale.is_ball:
        r1, r2, a1, a2 = box
        radii, radial = nodes(r1, r2)
        angles, angular = nodes(a1, a2)
        points = [(r, a) for r in radii for a in angles]
        weights = np.einsum("i,j->ij", radial * radii, degree * angular).ravel()
    else:
        r1, r2, b1, b2, a1, a2 = box
        radii, radial = nodes(r1, r2)
        latitudes, polar = nodes(b1, b2)
        longitudes, angular = nodes(a1, a2)
        polar = degree * polar * np.cos(np.radians(latitudes))
        points = [(r, b, a) for r in radii for b in latitudes for a in longitudes]
        weights = np.einsum(
            "i,j,k->ijk", radial * radii**2, polar, degree * angular
        ).ravel()
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


def property_operator(
    scale: Scale, space: Layered, /, *, boxes: Sequence[tuple[str, tuple]] = BOXES
) -> LinearOperator:
    """The three means: over the well-covered box, the poorly covered one and
    the core."""
    rows = [box_average_row(scale, space, box) for _, box in boxes]
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
        order=scale.order,
        length_scale=scale.length_scale,
        labels=["core", "mantle"],
    )


def layer_amplitudes(space: Layered, /, *, order: float, std: float) -> list[float]:
    """Each layer's amplitude for a Sobolev measure of the given order whose
    pointwise standard deviation, nearly uniform over the layer, averages
    ``std``: the raw measure's mean over the layer's interior, scaled."""
    amplitudes = []
    for layer in space.layers:
        raw = layer.interior_values(layer.pointwise_variance(layer.eigenvalues**-order))
        amplitudes.append(std / float(np.sqrt(raw.mean())))
    return amplitudes


def sobolev_prior(space: Layered, /, *, order: float, std: float) -> GaussianMeasure:
    """A Sobolev measure of the given order on each layer, its pointwise
    standard deviation averaging ``std`` over each.

    The layered space's own Sobolev measure, which refuses an order at or
    below half the dimension: the layers independent, each with covariance
    ``a^2 A^-order`` of its own ``A``, diagonal in each layer's basis, so
    that its factor, its precision and its spectrum are in hand without a
    solve. It is calibrated by an amplitude per layer rather than pointwise,
    which would rescale it from place to place and leave it neither diagonal
    nor with a precision; its pointwise standard deviation stays within two
    per cent of uniform. The measure is on the layered space itself, so its
    draws and its posteriors keep the space's pointwise algebra, which a
    pointwise standard deviation needs.
    """
    return space.sobolev_measure(
        order, amplitude=layer_amplitudes(space, order=order, std=std)
    )


def prior_pointwise_std(space: Layered, /, *, order: float, std: float) -> tuple:
    """The pointwise standard deviation of :func:`sobolev_prior`, exactly,
    as a field: its covariance is diagonal in each layer's basis."""
    return tuple(
        amplitude * np.sqrt(layer.pointwise_variance(layer.eigenvalues**-order))
        for layer, amplitude in zip(
            space.layers, layer_amplitudes(space, order=order, std=std)
        )
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
        """Every node of every ray, stacked, as points of the model space."""
        return np.concatenate([ray.nodes for ray in self.rays])

    def lift(self, points: Any, /) -> list[tuple[float, ...]]:
        """Points as tuples of the model space's coordinates."""
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


def build(*, scale: Scale = FULL, quiet: bool = False) -> Problem:
    """The structure, the rays, the space, the prior, the truth and the data."""
    started = time.perf_counter()
    rng = np.random.default_rng([scale.seed, 0])
    say = (lambda *args: None) if quiet else print
    say(f"\n=== The setup ({scale.name} scale) ===")
    say(
        f"structure: a mantle on [{CORE_RADIUS}, {RADIUS}] with velocity "
        f"{SURFACE_VELOCITY} + {GRADIENT:.3f} (1 - r), {BASE_VELOCITY} at its base, "
        f"over a core of {CORE_VELOCITY}; the ray grazing the core emerges "
        f"{MANTLE.grazing_distance:.1f} degrees away, and beyond is the shadow"
    )

    t0 = time.perf_counter()
    pairs = network(
        scale.sources,
        scale.receivers,
        min_distance=scale.min_distance,
        max_distance=scale.max_distance,
    )
    rays = [
        MANTLE.trace(source, receiver, nodes=scale.nodes) for source, receiver in pairs
    ]
    deepest = min(float(ray.nodes[:, 0].min()) for ray in rays)
    say(
        f"rays: {len(scale.sources)} sources, {len(scale.receivers)} receivers, "
        f"{len(rays)} pairs between {scale.min_distance:g} and "
        f"{scale.max_distance:g} degrees "
        f"apart, {scale.nodes} nodes each, traced in {time.perf_counter() - t0:.1f} s; "
        f"the deepest turns at r = {deepest:.3f}"
    )

    space = model_space(scale)
    prior = sobolev_prior(space, order=scale.prior_order, std=scale.prior_std)
    core, mantle = space.layers
    truncation = "lmax" if scale.is_ball else "mmax"
    say(
        f"space: the layered {scale.geometry}, core {core.dim} and mantle "
        f"{mantle.dim} coefficients ({truncation} {inner_truncation(core)} and "
        f"{inner_truncation(mantle)}), H^{scale.order:g} with "
        f"length scale {scale.length_scale}; prior: a Sobolev measure of order "
        f"{scale.prior_order:g} on each layer, pointwise std "
        f"{scale.prior_std:g}% of the surface slowness, the layers independent"
    )

    t0 = time.perf_counter()
    forward = path_integral_operator(scale, space, rays)
    say(
        f"forward: the path integral along each ray, {len(rays)} rows assembled "
        f"from {len(rays) * scale.nodes} node evaluations in "
        f"{time.perf_counter() - t0:.1f} s"
    )
    noise = GaussianMeasure.from_standard_deviation(forward.codomain, scale.noise_std)
    problem = LinearForwardProblem(forward, error=noise)
    truth, data = problem.synthetic_model_and_data(prior, rng=rng)
    signal = forward(truth)
    say(
        f"truth: a draw from the prior; travel-time perturbations of std "
        f"{np.std(signal):.3f} against noise of {scale.noise_std}; chi-squared of the "
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
