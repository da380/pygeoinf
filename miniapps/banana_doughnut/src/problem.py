"""The problem: each traveltime as two functionals on one model space.

The planet is :mod:`.planet`'s: a mantle on ``[0.55, 1]`` whose velocity
rises linearly with depth, the source at the surface on the equator and the
receivers on the surface east of it. A measurement is a receiver's
distance and a band, :mod:`.pulse`'s spectrum about a dominant period; each
has ray theory's functional, from the ray, and finite-frequency theory's,
from the quadrature of the paraxial kernel about it (:mod:`.functionals`).

The model space is ``m = dln c`` on the mantle annulus in ``H^2``, where a
value at a point exists and both functionals are bounded: a
:class:`pygeoinf.ball.Ball` with an inner radius, its inner product ``(A^2
m, m)`` with ``A = 1 - L^2 Laplacian`` and ``L`` the scale's one length
scale. It is the norm the Sobolev gradients and the dual norms are measured
in, and it is the problem's: no prior changes it.

A prior is a Sobolev measure of order two on that one space with
correlation lengths of its own, ``L_r`` along a radius and ``L_h`` across
one: covariance ``(1 - div(L_r^2 grad) - L_h^2 Laplacian_h)^-2``, block by
degree in the space's basis, its pointwise standard deviation set to the
scale's at every point. The family is the scale's isotropic lengths, ``L_r
= L_h``, and its anisotropic pairs, a radial length shorter than the
lateral one as in the mantle.

Two scales behind one ``build(scale=)``: the quick one is a single ray in
the middle band on a ball of degree 16, the full one four rays in three
bands on a ball of degree 96.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from pygeoinf import GaussianMeasure, LinearOperator, ball

from . import functionals
from .functionals import Quadrature, Rule, thread_map
from .planet import (
    BASE_VELOCITY,
    CORE_RADIUS,
    GRADIENT,
    MANTLE,
    RADIUS,
    SURFACE_VELOCITY,
    Path,
    trace,
)
from .pulse import Band


@dataclass(frozen=True)
class Scale:
    """One size of the problem, and its physics: every field but the name
    is a command-line option of every script.

    Attributes:
        name: the preset's name, for the printed headings.
        seed: the seed of the draws from the priors.
        distances: the receivers' distances from the source along the
            equator, in degrees, between about 20 and 75: every ray turns in
            the mantle, which none reaches beyond 82.
        source_depth: the source's depth below the surface, in units of the
            planet's radius; the receivers are at the surface.
        periods: the bands' dominant periods, in units of time.
        lmax: the model space's largest spherical-harmonic degree.
        order: the Sobolev order of the model space, above three halves for
            a value at a point to exist.
        length_scale: the length scale of the model space's norm, the one
            the Sobolev gradients and the dual norms are measured in; no
            prior changes it.
        prior_lengths: the correlation lengths of the isotropic priors, the
            same along a radius and across one.
        prior_pairs: the anisotropic priors, each a correlation length along
            a radius and one across it, written ``radial,lateral``.
        prior_order: the Sobolev order of the priors, above three halves,
            half the dimension, for the covariance to be trace class.
        prior_std: the priors' pointwise standard deviation of ``dln c``, the
            same at every point of the annulus.
        nodes: the nodes along each ray, the points of both functionals'
            quadrature in arc length.
        detour_nodes: Gauss-Legendre nodes in the detour time about each
            node of a ray.
        angle_nodes: the trapezoid rule's nodes in the angle about a ray.
        frequency_nodes: Gauss-Legendre nodes in frequency for the band's
            average over its spectrum.
        cutoff: the largest detour time kept, as the exponent of the
            spectrum's Gaussian envelope there.
        n_jobs: threads for the rays, the quadratures and the rows of the
            functionals; none for serial.
    """

    name: str
    seed: int
    distances: tuple[float, ...]
    source_depth: float
    periods: tuple[float, ...]
    lmax: int
    order: float
    length_scale: float
    prior_lengths: tuple[float, ...]
    prior_pairs: tuple[tuple[float, float], ...]
    prior_order: float
    prior_std: float
    nodes: int
    detour_nodes: int
    angle_nodes: int
    frequency_nodes: int
    cutoff: float
    n_jobs: int | None

    @property
    def rule(self) -> Rule:
        """The quadrature's counts about each node of a ray."""
        return Rule(
            self.detour_nodes, self.angle_nodes, self.frequency_nodes, self.cutoff
        )

    @property
    def prior_scales(self) -> list[tuple[float, float]]:
        """Every prior's correlation lengths along a radius and across one:
        the isotropic ones, then the pairs."""
        isotropic = [(float(length), float(length)) for length in self.prior_lengths]
        pairs = [
            (float(radial), float(lateral)) for radial, lateral in self.prior_pairs
        ]
        return isotropic + pairs


# The bands: the dominant periods whose first Fresnel zone at mid-path is
# about 0.04, 0.08 and 0.16 of the radius across at 45 degrees, where the
# travel time is 0.669; the quick scale keeps the middle one. Degree 96
# resolves about 0.033, below the narrowest band and the shortest prior.
QUICK = Scale(
    name="quick",
    seed=2026,
    distances=(45.0,),
    source_depth=0.0,
    periods=(0.024,),
    lmax=16,
    order=2.0,
    length_scale=0.1,
    prior_lengths=(0.08, 0.32),
    prior_pairs=((0.05, 0.2),),
    prior_order=2.0,
    prior_std=0.02,
    nodes=101,
    detour_nodes=32,
    angle_nodes=16,
    frequency_nodes=128,
    cutoff=30.0,
    n_jobs=None,
)
FULL = dataclasses.replace(
    QUICK,
    name="full",
    distances=(30.0, 45.0, 60.0, 75.0),
    periods=(0.006, 0.024, 0.096),
    lmax=96,
    prior_lengths=(0.04, 0.08, 0.16, 0.32),
    prior_pairs=((0.05, 0.2), (0.05, 0.5)),
    nodes=201,
    angle_nodes=32,
    n_jobs=4,
)


def model_space(scale: Scale, /) -> Any:
    """The mantle annulus in ``H^order`` of the scale's length scale."""
    return ball.Sobolev(
        scale.lmax,
        scale.order,
        scale.length_scale,
        radius=RADIUS,
        inner_radius=CORE_RADIUS,
    )


def sobolev_prior(
    space: Any, radial: float, lateral: float, /, *, order: float, std: float
) -> GaussianMeasure:
    """The Sobolev measure of an order on the space with correlation lengths
    of its own along a radius and across one, its pointwise standard
    deviation ``std`` everywhere.

    The covariance before the calibration is ``(1 - div(L_r^2 grad) - L_h^2
    Laplacian_h)^-order``; its pointwise variance is a function of radius
    alone, and the calibration multiplies the draws by the radial profile
    that brings it to ``std^2``, which leaves the correlations as they were.
    """
    return space.sobolev_measure(
        float(order),
        length_scale=float(radial),
        lateral_length_scale=float(lateral),
        pointwise_std=float(std),
    )


def raw_pointwise_std(
    space: Any, radial: float, lateral: float, /, *, order: float
) -> np.ndarray:
    """The pointwise standard deviation over the annulus's own grid of the
    Sobolev measure of an order with those correlation lengths, before its
    calibration: how far from uniform the calibration had to bring it."""
    variance = space.sobolev_pointwise_variance(
        float(order), length_scale=float(radial), lateral_length_scale=float(lateral)
    )
    return np.sqrt(space.interior_values(variance))


@dataclass
class Problem:
    """Everything ``build`` makes and the parts use.

    Attributes:
        scale: the scale it was built at.
        paths: each receiver's ray and dynamic ray tracing along it, by
            distance.
        bands: the bands, by period.
        quadratures: ``quadratures[band][path]``, the finite-frequency
            functional's quadrature of each ray in each band.
        space: the model space.
        priors: the priors on it, by their correlation lengths along a
            radius and across one, every one of the scale's.
    """

    scale: Scale
    paths: list[Path]
    bands: list[Band]
    quadratures: list[list[Quadrature]]
    space: Any
    priors: dict[tuple[float, float], GaussianMeasure]
    _assembled: dict = field(default_factory=dict, repr=False)

    def prior(self, length: float, lateral: float | None = None, /) -> GaussianMeasure:
        """The prior of a correlation length along a radius and one across
        it, the same if not given: one of the scale's, or made now and kept.
        """
        key = (float(length), float(length if lateral is None else lateral))
        if key not in self.priors:
            self.priors[key] = sobolev_prior(
                self.space,
                *key,
                order=self.scale.prior_order,
                std=self.scale.prior_std,
            )
        return self.priors[key]

    def ray_pieces(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """The points and weights of ray theory's functional of each ray."""
        return [
            (functionals.ray_points(path), functionals.ray_weights(path))
            for path in self.paths
        ]

    def finite_frequency_pieces(
        self, band: int, /
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """The points kept and their weights of finite-frequency theory's
        functional of each ray in one band, by its index."""
        return [(q.lifted, q.kept_weights) for q in self.quadratures[band]]

    def _operator(
        self,
        key: Any,
        pieces: Callable[[], list[tuple[np.ndarray, np.ndarray]]],
        assembled: bool,
    ) -> LinearOperator:
        if not assembled:
            return functionals.composed_operator(self.space, pieces())
        if key not in self._assembled:
            self._assembled[key] = functionals.assembled_operator(
                self.space, pieces(), n_jobs=self.scale.n_jobs
            )
        return self._assembled[key]

    def ray_operator(self, *, assembled: bool = False) -> LinearOperator:
        """Ray theory's functional of every ray, one row a ray, by distance.

        Args:
            assembled: from the rows, formed once and kept; otherwise point
                evaluation at the nodes composed with the weights.
        """
        return self._operator("ray", self.ray_pieces, assembled)

    def finite_frequency_operator(
        self, band: int, /, *, assembled: bool = False
    ) -> LinearOperator:
        """Finite-frequency theory's functional of every ray in one band, by
        its index, one row a ray.

        Args:
            band: the band's index.
            assembled: from the rows, formed once and kept; otherwise point
                evaluation at the quadrature's points composed with the
                weights.
        """
        return self._operator(
            ("finite frequency", band),
            lambda: self.finite_frequency_pieces(band),
            assembled,
        )


def build(*, scale: Scale = FULL, quiet: bool = False) -> Problem:
    """The rays, the bands, the quadratures, the space and the prior."""
    started = time.perf_counter()
    say: Callable[..., None] = (lambda *args: None) if quiet else print
    say(f"\n=== The setup ({scale.name} scale) ===")
    say(
        f"planet: a mantle on [{CORE_RADIUS}, {RADIUS}] with velocity "
        f"{SURFACE_VELOCITY} + {GRADIENT:.3f} (1 - r), {BASE_VELOCITY} at its base; "
        f"the rays that turn in it reach {MANTLE.grazing_distance:.1f} degrees"
    )

    t0 = time.perf_counter()
    paths = thread_map(
        lambda d: trace(d, nodes=scale.nodes, source_depth=scale.source_depth),
        scale.distances,
        n_jobs=scale.n_jobs,
    )
    say(
        f"rays: {len(paths)} from the source at depth {scale.source_depth:g}, "
        f"{scale.nodes} nodes each, traced with their Hessians in "
        f"{time.perf_counter() - t0:.2f} s"
    )
    for path in paths:
        sums = path.dynamic.sums[1:-1]
        say(
            f"  {path.distance:g} degrees: length {path.length:.4f}, travel time "
            f"{path.travel_time:.4f}, turning at r = {path.turning_radius:.4f}; "
            f"A1 from {sums[:, 0].min():.3f}, A2 from {sums[:, 1].min():.3f}, "
            "both positive"
        )

    bands = [Band(period) for period in scale.periods]
    rule = scale.rule
    for band in bands:
        widths = ", ".join(f"{p.fresnel_half_width(band.period):.3f}" for p in paths)
        fractions = ", ".join(f"{band.period / p.travel_time:.3f}" for p in paths)
        say(
            f"band T0 = {band.period:g}: Fresnel half-width at mid-path {widths}; "
            f"T0 / T = {fractions}"
        )
    defect = bands[0].gaussian_defect(
        rule.detours, cutoff=rule.cutoff, frequencies=rule.frequencies
    )
    say(
        f"quadrature: {rule.detours} detour times to sigma^2 u^2 / 2 = "
        f"{rule.cutoff:g}, {rule.angles} angles, {rule.frequencies} frequencies; "
        f"sum G(u) w - 1 = {defect:.2e}, the same in every band"
    )

    t0 = time.perf_counter()
    pairs = [(b, p) for b in range(len(bands)) for p in range(len(paths))]
    made = thread_map(
        lambda pair: functionals.quadrature(paths[pair[1]], bands[pair[0]], rule),
        pairs,
        n_jobs=scale.n_jobs,
    )
    quadratures = [
        made[b * len(paths) : (b + 1) * len(paths)] for b in range(len(bands))
    ]
    say(f"  {len(made)} quadratures made in {time.perf_counter() - t0:.2f} s")
    for band, row in zip(bands, quadratures):
        for path, q in zip(paths, row):
            say(
                f"  {path.distance:g} degrees, T0 = {band.period:g}: "
                f"{q.kept.sum()} of {q.kept.size} points in the annulus; weight "
                f"dropped above the surface {100 * q.surface_clipped:+.3f}% "
                f"(absolute {100 * q.surface_clipped_absolute:.2f}%), below the "
                f"core-mantle boundary {100 * q.core_clipped:+.3f}% (absolute "
                f"{100 * q.core_clipped_absolute:.2f}%)"
            )

    t0 = time.perf_counter()
    space = model_space(scale)
    say(
        f"space: the annulus in H^{scale.order:g}, lmax {scale.lmax}, length scale "
        f"{scale.length_scale:g}, {space.dim} coefficients"
    )
    priors = {
        key: sobolev_prior(space, *key, order=scale.prior_order, std=scale.prior_std)
        for key in scale.prior_scales
    }
    say(
        f"priors: Sobolev measures of order {scale.prior_order:g} on it, pointwise "
        f"std {scale.prior_std:g} everywhere; space and {len(priors)} priors built "
        f"in {time.perf_counter() - t0:.2f} s"
    )
    for radial, lateral in priors:
        raw = raw_pointwise_std(space, radial, lateral, order=scale.prior_order)
        lengths = (
            f"correlation length {radial:g}"
            if radial == lateral
            else f"correlation lengths {radial:g} radial, {lateral:g} lateral"
        )
        say(
            f"  {lengths}: pointwise std before calibration from "
            f"{raw.min() / raw.mean():.3f} to {raw.max() / raw.mean():.3f} of its "
            "average"
        )
    say(f"setup took {time.perf_counter() - started:.2f} s")
    return Problem(scale, paths, bands, quadratures, space, priors)
