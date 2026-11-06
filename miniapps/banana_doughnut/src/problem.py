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

A prior is a Sobolev measure of order two on that one space with length
scales of its own, ``L_r`` along a radius and ``L_h`` across one: its
covariance is the operator ``(1 - div(L_r^2 grad) - L_h^2
Laplacian_h)^-2`` in the space's metric, block by degree in the space's
basis, its pointwise standard deviation set to the scale's at every point.
Being an operator in that metric, it carries the space's own symbol into
the draws, so the space's length scale is the shortest over which a draw
can decorrelate, and a prior's nominal lengths are not its draws'
correlation lengths: those are measured, from the covariance itself
(:mod:`.correlations`). The family is the scale's isotropic lengths, ``L_r
= L_h``, and its anisotropic pairs, a radial length shorter than the
lateral one as in the mantle.

Two scales behind one ``build(scale=)``: the quick one is a single ray in
the middle band on a ball of degree 16, the full one four rays in three
bands on a ball of degree 160.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from pygeoinf import EuclideanSpace, GaussianMeasure, LinearOperator, ball

from . import correlations, functionals
from .functionals import Quadrature, Rule, Surface, thread_map
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
        reference_distance: the distance, one of ``distances``, of the one
            geometry the comparison draws.
        source_depth: the source's depth below the surface, in units of the
            planet's radius; the receivers are at the surface.
        periods: the bands' dominant periods, in units of time.
        lmax: the model space's largest spherical-harmonic degree.
        order: the Sobolev order of the model space, above three halves for
            a value at a point to exist.
        length_scale: the length scale of the model space's norm, the one
            the Sobolev gradients and the dual norms are measured in, and
            the shortest over which a prior's draws can decorrelate; no
            prior changes it.
        padding: how far the space's radial mesh reaches past each end of
            the annulus, out of the domain; none for the space's own
            default, two of its length scales. A prior much longer than that
            would feel the ends of the mesh.
        prior_lengths: the nominal length scales of the isotropic priors'
            covariances, the same along a radius and across one; their
            draws' correlation lengths are measured.
        prior_pairs: the anisotropic priors, each a nominal length scale
            along a radius and one across it, written ``radial,lateral``.
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
        surface: what becomes of the quadrature's points above the free
            surface: ``fold``, moved to the mirror radius ``2 - r`` along
            their own direction, the free surface by images; or ``clip``,
            dropped.
        draws: the draws from each prior whose delays part D scatters, each
            through both theories' functionals of every ray in every band.
        n_jobs: threads for the rays, the quadratures, the rows of the
            functionals, the push-forwards and the draws; none for serial.
    """

    name: str
    seed: int
    distances: tuple[float, ...]
    reference_distance: float
    source_depth: float
    periods: tuple[float, ...]
    lmax: int
    order: float
    length_scale: float
    padding: float | None
    prior_lengths: tuple[float, ...]
    prior_pairs: tuple[tuple[float, float], ...]
    prior_order: float
    prior_std: float
    nodes: int
    detour_nodes: int
    angle_nodes: int
    frequency_nodes: int
    cutoff: float
    surface: Surface
    draws: int
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
# travel time is 0.669; the quick scale keeps the middle one. The space's
# length scale, 0.02, is below every Fresnel half-width and every prior of
# interest: a prior's covariance is an operator in the space's metric, and
# the space's scale is the shortest over which its draws can decorrelate.
# Degree 160 resolves about pi / 160 = 0.02 at the surface. The quick
# scale's priors are smoke-test values, long enough for degree 16 to
# resolve. A draw from a prior costs about a second at degree 160 and a
# hundredth of one at degree 16, so the full scale draws fewer.
QUICK = Scale(
    name="quick",
    seed=2026,
    distances=(45.0,),
    reference_distance=45.0,
    source_depth=0.0,
    periods=(0.024,),
    lmax=16,
    order=2.0,
    length_scale=0.02,
    padding=0.32,
    prior_lengths=(0.2, 0.4),
    prior_pairs=((0.1, 0.4),),
    prior_order=2.0,
    prior_std=0.02,
    nodes=101,
    detour_nodes=32,
    angle_nodes=16,
    frequency_nodes=128,
    cutoff=30.0,
    surface="fold",
    draws=300,
    n_jobs=None,
)
FULL = dataclasses.replace(
    QUICK,
    name="full",
    distances=(30.0, 45.0, 60.0, 75.0),
    periods=(0.006, 0.024, 0.096),
    lmax=160,
    prior_lengths=(0.02, 0.04, 0.08, 0.16, 0.32),
    prior_pairs=((0.03, 0.2), (0.03, 0.5)),
    nodes=201,
    angle_nodes=32,
    draws=100,
    n_jobs=4,
)


def model_space(scale: Scale, /) -> Any:
    """The mantle annulus in ``H^order`` of the scale's length scale, its
    radial mesh padded as the scale says."""
    return ball.Sobolev(
        scale.lmax,
        scale.order,
        scale.length_scale,
        radius=RADIUS,
        inner_radius=CORE_RADIUS,
        padding=scale.padding,
    )


def resolution(scale: Scale, /) -> float:
    """The shortest length the space resolves at the surface, about ``pi a /
    lmax``, half the shortest wavelength of its harmonics there."""
    return float(np.pi * RADIUS / scale.lmax)


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
    _rows: dict = field(default_factory=dict, repr=False)
    _correlations: dict = field(default_factory=dict, repr=False)

    @property
    def reference(self) -> int:
        """The index of the ray at the scale's reference distance.

        Raises:
            ValueError: if that distance is not one of the scale's.
        """
        for index, path in enumerate(self.paths):
            if np.isclose(path.distance, self.scale.reference_distance):
                return index
        raise ValueError(
            f"The reference distance {self.scale.reference_distance:g} is not one "
            f"of the distances {self.scale.distances}."
        )

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

    def correlation(
        self, key: tuple[float, float], index: int | None = None, /
    ) -> correlations.Correlation:
        """The correlation of the prior of those nominal lengths, along a
        radius and across one, from the turning point of a ray, by its index
        (the reference ray's by default): computed once and kept."""
        index = self.reference if index is None else int(index)
        key = (float(key[0]), float(key[1]))
        if (key, index) not in self._correlations:
            self._correlations[(key, index)] = correlations.correlation(
                self.space,
                self.prior(*key),
                correlations.turning_point(self.paths[index]),
            )
        return self._correlations[(key, index)]

    def ray_pieces(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """The points and weights of ray theory's functional of each ray."""
        return [self.piece(index) for index in range(len(self.paths))]

    def finite_frequency_pieces(
        self, band: int, /
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """The points kept and their weights of finite-frequency theory's
        functional of each ray in one band, by its index."""
        return [self.piece(index, band) for index in range(len(self.paths))]

    def piece(
        self, index: int, band: int | None = None, /
    ) -> tuple[np.ndarray, np.ndarray]:
        """The points ``(radius, latitude, longitude)`` and weights of one
        functional: ray theory's of a ray by its index, or with a band's
        index finite-frequency theory's in that band."""
        if band is None:
            path = self.paths[index]
            return functionals.ray_points(path), functionals.ray_weights(path)
        quadrature = self.quadratures[band][index]
        return quadrature.lifted, quadrature.kept_weights

    def rows(
        self, keys: list[tuple[int, int | None]], /, *, n_jobs: int | None = None
    ) -> np.ndarray:
        """The derivative components of functionals, one row each, so that
        a functional is ``row @ components(m)``: each read off its own
        composition by one application of the adjoint, formed in ``n_jobs``
        threads (the scale's by default) and kept.

        Args:
            keys: ``(index, band)`` of each functional, as :meth:`piece`
                takes them, ``band`` ``None`` for ray theory's.
            n_jobs: threads for the rows not yet formed.

        Returns:
            ``(len(keys), dim)``.
        """
        keys = [
            (int(index), None if band is None else int(band)) for index, band in keys
        ]
        missing = list(dict.fromkeys(key for key in keys if key not in self._rows))

        def row(key: tuple[int, int | None]) -> np.ndarray:
            operator = functionals.composed_operator(self.space, [self.piece(*key)])
            return operator.matrix(form="galerkin")[0]

        jobs = self.scale.n_jobs if n_jobs is None else n_jobs
        for key, made in zip(missing, thread_map(row, missing, n_jobs=jobs)):
            self._rows[key] = made
        if not keys:
            return np.zeros((0, self.space.dim))
        return np.stack([self._rows[key] for key in keys])

    def representer(self, index: int, band: int | None = None, /) -> np.ndarray:
        """The Sobolev gradient of one functional, as :meth:`piece` names it:
        its representer in the space's own inner product, the vector ``g``
        with ``(g, m) = l(m)`` for every model."""
        return self.space.representer(self.rows([(index, band)])[0])

    def _from_rows(self, keys: list[tuple[int, int | None]]) -> LinearOperator:
        return LinearOperator.from_matrix(
            self.space, EuclideanSpace(len(keys)), self.rows(keys), form="galerkin"
        )

    def _composed(self, keys: list[tuple[int, int | None]]) -> LinearOperator:
        return functionals.composed_operator(
            self.space, [self.piece(*key) for key in keys]
        )

    def ray_operator(self, *, assembled: bool = False) -> LinearOperator:
        """Ray theory's functional of every ray, one row a ray, by distance.

        Args:
            assembled: from the rows, formed once and kept; otherwise point
                evaluation at the nodes composed with the weights.
        """
        keys = [(index, None) for index in range(len(self.paths))]
        return self._from_rows(keys) if assembled else self._composed(keys)

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
        keys = [(index, band) for index in range(len(self.paths))]
        return self._from_rows(keys) if assembled else self._composed(keys)

    @property
    def stacked_keys(self) -> list[tuple[int, int | None]]:
        """The functionals of :meth:`stacked_operator`, in its order, as
        :meth:`piece` names them: ray theory's of each ray, then
        finite-frequency theory's of each ray in each band in turn."""
        rays = range(len(self.paths))
        return [(index, None) for index in rays] + [
            (index, band) for band in range(len(self.bands)) for index in rays
        ]

    def stacked_operator(self, *, assembled: bool = True) -> LinearOperator:
        """Every functional of the scale stacked, onto ``R^(N (1 + B))`` for
        ``N`` rays and ``B`` bands: ray theory's of every ray, by distance,
        then finite-frequency theory's of every ray in the first band, and so
        on through the bands, so that the ``N`` rows from ``N k`` on are
        :meth:`ray_operator`'s for ``k`` zero and
        :meth:`finite_frequency_operator`'s of band ``k - 1`` after.

        Args:
            assembled: from the rows, formed once and kept; otherwise point
                evaluation at every functional's points composed with the
                rows of weights.
        """
        keys = self.stacked_keys
        return self._from_rows(keys) if assembled else self._composed(keys)

    def pair_operator(
        self, index: int, band: int, /, *, assembled: bool = True
    ) -> LinearOperator:
        """The two functionals of one ray in one band stacked, ``m -> (dT_ray,
        dT_FF)``, an operator onto ``R^2``.

        Args:
            index: the ray's index.
            band: the band's index.
            assembled: from the two rows, formed once and kept, so that an
                application is a product with them; otherwise point
                evaluation at both functionals' points at once composed
                with the two rows of weights.
        """
        keys = [(index, None), (index, band)]
        return self._from_rows(keys) if assembled else self._composed(keys)


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
        widths = ", ".join(
            f"{p.fresnel_half_width(band.period):.3f} "
            f"({p.fresnel_half_width(band.period) / p.wavelength(band.period):.2f})"
            for p in paths
        )
        waves = ", ".join(f"{p.wavelength(band.period):.4f}" for p in paths)
        fractions = ", ".join(f"{band.period / p.travel_time:.3f}" for p in paths)
        say(
            f"band T0 = {band.period:g}: wavelength {SURFACE_VELOCITY * band.period:.4f} "
            f"at the surface, {waves} at mid-path; Fresnel half-width at mid-path "
            f"{widths} (in wavelengths); T0 / T = {fractions}"
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
        lambda pair: functionals.quadrature(
            paths[pair[1]], bands[pair[0]], rule, surface=scale.surface
        ),
        pairs,
        n_jobs=scale.n_jobs,
    )
    quadratures = [
        made[b * len(paths) : (b + 1) * len(paths)] for b in range(len(bands))
    ]
    above = "folded back" if scale.surface == "fold" else "dropped"
    fate = (
        "folded back to their mirror images below it, the free surface by images"
        if scale.surface == "fold"
        else "dropped"
    )
    say(
        f"  {len(made)} quadratures made in {time.perf_counter() - t0:.2f} s; the "
        f"points above the surface {fate}"
    )
    for band, row in zip(bands, quadratures):
        for path, q in zip(paths, row):
            folded = (
                f"; {q.folded_below_core} points folded below the core-mantle "
                "boundary and dropped there"
                if q.folded_below_core
                else ""
            )
            say(
                f"  {path.distance:g} degrees, T0 = {band.period:g}: "
                f"{q.kept.sum()} of {q.kept.size} points kept; weight {above} "
                f"above the surface {100 * q.surface_fraction:+.3f}% "
                f"(absolute {100 * q.surface_fraction_absolute:.2f}%), dropped "
                f"below the core-mantle boundary {100 * q.core_fraction:+.3f}% "
                f"(absolute {100 * q.core_fraction_absolute:.2f}%){folded}"
            )

    t0 = time.perf_counter()
    space = model_space(scale)
    say(
        f"space: the annulus in H^{scale.order:g}, lmax {scale.lmax} (resolving about "
        f"{resolution(scale):.3f} at the surface), length scale "
        f"{scale.length_scale:g}, padding {space.padding[0]:g} below and "
        f"{space.padding[1]:g} above, {space.dim} coefficients"
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
            f"nominal length scale {radial:g}"
            if radial == lateral
            else f"nominal length scales {radial:g} radial, {lateral:g} lateral"
        )
        say(
            f"  {lengths}: pointwise std before calibration from "
            f"{raw.min() / raw.mean():.3f} to {raw.max() / raw.mean():.3f} of its "
            "average"
        )
    say(f"setup took {time.perf_counter() - started:.2f} s")
    return Problem(scale, paths, bands, quadratures, space, priors)
