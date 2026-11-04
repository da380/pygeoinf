"""The experiment: a thin shell, a synthetic network and the mode kernels.

The D″ layer, the lowermost 300 km of the mantle, is treated as one sphere.
Every field lives on it, and a datum is an average through the layer, so that
the layer's radial structure is folded into the data's weights.

The network is synthetic on a realistic geometry. Events lie along the plate
boundaries, stations on the continents. A diffracted wave, Sdiff or Pdiff,
sees the arc of the great circle from its event to its station that remains
when the descent through the mantle is taken off each end; a core reflection,
ScS or PcP, sees a small cap about the midpoint of a shorter path. So the
coverage of the base of the mantle is what the map of the world's earthquakes
and seismometers leaves it, without any data from either.

The modes are illustrative: each is three weights, on ``dlnVs``, ``dlnVp`` and
``dlnrho`` through the layer, chosen to span what real structure coefficients
see, from mostly shear to mostly density. They are not the kernels of any
named mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import shapely
from cartopy.io import shapereader

# The three thermal fields, in the order of every block and column below.
FIELDS = ("vs", "vp", "rho")
# Plate boundaries, as polylines of (latitude, longitude) in degrees, coarse
# and hand-placed: subduction zones for the most part, as the catalogue is,
# then the Alpine-Himalayan belt and the longest ridges.
BOUNDARIES = (
    ((-15, -173), (-22, -175), (-30, -178), (-38, 178)),
    ((-20, 169), (-12, 166), (-7, 155), (-5, 150)),
    ((-4, 140), (-6, 130), (-9, 122)),
    ((-10, 115), (-8, 105), (-2, 98), (5, 94)),
    ((6, 127), (15, 121), (24, 123), (31, 141), (36, 142)),
    ((43, 148), (50, 158), (55, 162)),
    ((52, 178), (52, -170), (55, -158), (60, -147)),
    ((45, -128), (20, -105), (15, -95), (12, -88), (8, -83)),
    ((5, -78), (-5, -81), (-15, -75), (-25, -70), (-35, -73), (-45, -75)),
    ((18, -65), (12, -60)),
    ((28, 88), (34, 72), (37, 55), (38, 40), (38, 25), (38, 15)),
    ((60, -30), (35, -37), (10, -40), (-20, -13), (-50, -8)),
    ((-10, 67), (-30, 75), (-45, 100), (-55, 140)),
    ((10, -104), (-20, -112), (-40, -105)),
)
# The modes' weights on (dlnVs, dlnVp, dlnrho), the first few mostly shear,
# then compressional, then the density-sensitive ones the question rests on.
KERNELS = (
    (1.0, 0.1, 0.1),
    (0.2, 0.2, 1.0),
    (0.1, 1.0, 0.1),
    (0.9, 0.2, 0.4),
    (0.4, 0.8, 0.3),
    (0.3, 0.1, 0.8),
    (0.8, 0.3, 0.2),
    (0.2, 0.9, 0.4),
    (0.6, 0.1, 0.6),
    (0.1, 0.4, 0.9),
)
# Where the provinces lie in the truth, (latitude, longitude) in degrees, and
# the caps asked about: over each, and over a place between them.
PROVINCES = ((-5.0, 20.0), (-10.0, -160.0))
REGIONS = (
    ("Africa", (-5.0, 20.0)),
    ("Pacific", (-10.0, -160.0)),
    ("Asia", (35.0, 100.0)),
)


@dataclass(frozen=True)
class Scale:
    """One size of the problem, its physics, and how long its chains run:
    every field but the name is a command-line option of every script.

    Attributes:
        name: the preset's name, for the printed headings.
        lmax: the truncation degree of the fields.
        arcs: the diffracted paths, Sdiff and Pdiff together.
        caps: the core reflections, ScS and PcP together.
        modes: the modes of the table of kernels, the first so many.
        arc_nodes: the quadrature nodes along each arc.
        cap_nodes: the quadrature nodes of the indicator over each cap.
        indicator_degree: the degree of the fine grid the modes' coefficients
            of the indicator are integrated on, the same at every truncation.
        seed: the seed of the network, the truth, the noise and every
            part's draws.
        events: the events along the plate boundaries.
        stations: the stations on land.
        p_share: the share of the arcs and of the caps that are P waves.
        descent: the arc taken off each end of a diffracted path for its
            descent through the mantle, degrees.
        arc_range: the distances of the diffracted paths, degrees.
        cap_range: the distances of the core reflections, degrees.
        cap_radius: the radius of a core reflection's cap, on the unit
            sphere (0.09 is about 330 km at the layer).
        degrees: the degrees of the modes' structure coefficients.
        order: the Sobolev order of the fields' space, above one for a value
            at a point to exist.
        length_scale: the length scale of the space and the thermal prior,
            on the unit sphere (0.3 is about 1100 km at the layer).
        prior_order: the Sobolev order of the priors on the space, above
            one for their covariances to be trace class there.
        thermal_std: the pointwise standard deviations of the thermal
            fields, in percent, in the order vs, vp, rho.
        thermal_correlation: the thermal fields' correlations, in the order
            vs-vp, vs-rho, vp-rho.
        level_set_length_scale: the length scale of the level set's prior.
        chemistry_std: the prior's standard deviations of the contrasts
            inside the provinces, in percent, in the order vs, vp, rho.
        contrasts: the true contrasts inside the provinces, in percent.
        province_width: the width of the true level set's bumps, on the
            unit sphere.
        province_level: what is taken off the true level set's bumps, of
            unit peak: the higher it is, the smaller the provinces.
        noise_s: the noise on the S data, as a fraction of their rms.
        noise_p: the noise on the P data, as a fraction of their rms.
        noise_modes: the noise on the modes' coefficients, as a fraction of
            their rms.
        region_radius: the radius of the caps asked about, on the unit
            sphere.
        weak_noise: what part C's weak problem multiplies the noise by.
        chains: the chains of each sampler of part C.
        budget: the forward evaluations of each chain of a sampler whose
            reference is the prior.
        burn_share: the share of those a chain takes first and discards.
        geometric_steps: the steps each chain of GeometricHMC keeps, which
            are counted apart since its reference is what it spends time on.
        geometric_burn_in: the steps it takes first and discards.
        reference_rank: the rank of the low-rank reference on the data as
            given.
        weak_rank: the rank of the low-rank reference on the weak problem.
        pcn_step: the time step pCN's burn-in starts from.
        mala_step: the time step MALA's burn-in starts from.
        hmc_step: the time step HMC's burn-in starts from.
        hmc_length: the leapfrog steps of an HMC trajectory.
        ghmc_step: the time step GeometricHMC's burn-in starts from.
        ghmc_length: the leapfrog steps of a GeometricHMC trajectory.
        ladder: the degrees of part C's ladder.
        ladder_steps: the steps each chain keeps on the ladder.
        start_share: the share of the sphere inside the provinces of part D's
            start, the slowest of the thermal-only posterior mean.
        start_spread: the fraction of a prior draw added to that start, for
            each chain's own.
        gibbs_chains: the chains of part D's main run.
        gibbs_steps: the sweeps each of them keeps.
        gibbs_burn_in: the sweeps each of them takes first and discards.
        gibbs_thin: the sweeps between the states they store.
        linear_step: the time step the linear block's burn-in starts from.
        linear_length: the leapfrog steps of the linear block's trajectory.
        level_step: the time step the level set's burn-in starts from.
        level_steps: the pCN steps at each visit to the level set.
        thermal_step: the time step of the thermal fields' own block, in part
            D's comparison of three blocks, where its burn-in starts.
        thermal_length: the leapfrog steps of that block's trajectory.
        contrast_step: the time step of the contrasts' own block, likewise.
        contrast_steps: the MALA steps at each visit to it.
        level_ladder: the pCN steps per visit part D compares.
        compare_steps: the sweeps of each of part D's comparisons.
        compare_burn_in: the sweeps they take first and discard.
        n_jobs: the workers the chains run on; none for one after another.
    """

    name: str
    lmax: int
    arcs: int
    caps: int
    modes: int
    arc_nodes: int
    cap_nodes: int
    indicator_degree: int = 128
    seed: int = 2026
    events: int = 400
    stations: int = 300
    p_share: float = 0.4
    descent: float = 40.0
    arc_range: tuple[float, ...] = (100.0, 160.0)
    cap_range: tuple[float, ...] = (30.0, 80.0)
    cap_radius: float = 0.09
    degrees: tuple[int, ...] = (2, 4, 6)
    order: float = 2.0
    length_scale: float = 0.3
    prior_order: float = 2.0
    thermal_std: tuple[float, ...] = (1.0, 0.5, 0.3)
    thermal_correlation: tuple[float, ...] = (0.95, 0.8, 0.8)
    level_set_length_scale: float = 0.4
    chemistry_std: tuple[float, ...] = (2.0, 1.0, 1.0)
    contrasts: tuple[float, ...] = (-2.0, -0.5, 1.0)
    province_width: float = 0.55
    province_level: float = 0.4
    noise_s: float = 0.2
    noise_p: float = 0.3
    noise_modes: float = 0.2
    region_radius: float = 0.3
    weak_noise: float = 15.0
    chains: int = 2
    budget: int = 2000
    burn_share: float = 0.2
    geometric_steps: int = 80
    geometric_burn_in: int = 20
    reference_rank: int = 50
    weak_rank: int = 10
    pcn_step: float = 0.01
    mala_step: float = 0.01
    hmc_step: float = 0.1
    hmc_length: int = 8
    ghmc_step: float = 0.3
    ghmc_length: int = 4
    ladder: tuple[int, ...] = (12, 24)
    ladder_steps: int = 100
    start_share: float = 0.25
    start_spread: float = 0.1
    gibbs_chains: int = 4
    gibbs_steps: int = 800
    gibbs_burn_in: int = 200
    gibbs_thin: int = 4
    linear_step: float = 0.02
    linear_length: int = 16
    level_step: float = 0.002
    level_steps: int = 5
    thermal_step: float = 0.03
    thermal_length: int = 8
    contrast_step: float = 0.001
    contrast_steps: int = 5
    level_ladder: tuple[int, ...] = (1, 5, 20)
    compare_steps: int = 100
    compare_burn_in: int = 50
    n_jobs: int | None = None


PROOF = Scale(
    name="proof",
    lmax=12,
    arcs=40,
    caps=20,
    modes=2,
    arc_nodes=32,
    cap_nodes=16,
    events=40,
    stations=40,
)
QUICK = Scale(
    name="quick",
    lmax=24,
    arcs=600,
    caps=300,
    modes=6,
    arc_nodes=64,
    cap_nodes=32,
    n_jobs=4,
)
FULL = Scale(
    name="full",
    lmax=48,
    arcs=3000,
    caps=1500,
    modes=10,
    arc_nodes=96,
    cap_nodes=48,
    events=1000,
    stations=800,
    chains=4,
    budget=1000,
    geometric_steps=40,
    geometric_burn_in=10,
    reference_rank=80,
    weak_rank=20,
    ladder=(12, 24, 48),
    n_jobs=4,
)


def to_vector(points: Any, /) -> np.ndarray:
    """``(latitude, longitude)`` pairs in degrees as unit vectors, row by row."""
    latitude, longitude = np.radians(np.asarray(points, dtype=float)).T
    return np.column_stack(
        [
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ]
    )


def to_points(vectors: np.ndarray, /) -> np.ndarray:
    """Unit vectors as ``(latitude, longitude)`` pairs in degrees, row by row."""
    x, y, z = np.asarray(vectors).T
    return np.column_stack(
        [np.degrees(np.arcsin(np.clip(z, -1.0, 1.0))), np.degrees(np.arctan2(y, x))]
    )


def separation(first: np.ndarray, second: np.ndarray, /) -> np.ndarray:
    """The angle between unit vectors, row by row, in radians."""
    cross = np.linalg.norm(np.cross(first, second), axis=-1)
    return np.arctan2(cross, np.sum(first * second, axis=-1))


def along(first: np.ndarray, second: np.ndarray, fraction: Any, /) -> np.ndarray:
    """The point a fraction of the way along each great circle, row by row."""
    angle = separation(first, second)[:, None]
    fraction = np.broadcast_to(np.asarray(fraction, dtype=float), angle.shape)
    return (
        np.sin((1.0 - fraction) * angle) * first + np.sin(fraction * angle) * second
    ) / np.sin(angle)


def events(count: int, /, *, rng: np.random.Generator) -> np.ndarray:
    """Events along the plate boundaries, uniform by length with a little
    scatter.

    Returns:
        A ``(count, 2)`` array of ``(latitude, longitude)`` in degrees.
    """
    starts = np.vstack(
        [to_vector(line[:-1]) for line in BOUNDARIES if len(line) > 1],
    )
    ends = np.vstack([to_vector(line[1:]) for line in BOUNDARIES if len(line) > 1])
    lengths = separation(starts, ends)
    chosen = rng.choice(lengths.size, size=count, p=lengths / lengths.sum())
    vectors = along(starts[chosen], ends[chosen], rng.uniform(size=(count, 1)))
    vectors = vectors + rng.normal(scale=np.radians(1.5), size=vectors.shape)
    return to_points(vectors / np.linalg.norm(vectors, axis=1, keepdims=True))


def land_polygons() -> Any:
    """The Natural Earth land, as one shapely geometry."""
    path = shapereader.natural_earth(
        resolution="110m", category="physical", name="land"
    )
    return shapely.union_all(list(shapereader.Reader(path).geometries()))


def stations(count: int, /, *, rng: np.random.Generator) -> np.ndarray:
    """Stations uniform over the land north of Antarctica.

    Returns:
        A ``(count, 2)`` array of ``(latitude, longitude)`` in degrees.
    """
    land = land_polygons()
    kept: list[np.ndarray] = []
    total = 0
    while total < count:
        draw = max(200, 6 * (count - total))
        latitudes = np.degrees(np.arcsin(rng.uniform(-0.85, 1.0, draw)))
        longitudes = rng.uniform(-180.0, 180.0, draw)
        inside = shapely.contains_xy(land, longitudes, latitudes)
        kept.append(np.column_stack([latitudes, longitudes])[inside])
        total += int(inside.sum())
    return np.concatenate(kept)[:count]


def _pairs(
    sources: np.ndarray,
    receivers: np.ndarray,
    count: int,
    limits: tuple[float, ...],
    /,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Event-station pairs at a range of distances, as two arrays of vectors."""
    first, second = to_vector(sources), to_vector(receivers)
    low, high = np.radians(limits)
    found_first: list[np.ndarray] = []
    found_second: list[np.ndarray] = []
    total = 0
    while total < count:
        draw = 4 * count
        a = first[rng.integers(len(first), size=draw)]
        b = second[rng.integers(len(second), size=draw)]
        angle = separation(a, b)
        inside = (angle >= low) & (angle <= high)
        found_first.append(a[inside])
        found_second.append(b[inside])
        total += int(inside.sum())
    return np.concatenate(found_first)[:count], np.concatenate(found_second)[:count]


@dataclass(frozen=True)
class Network:
    """Where the data are: what each kind of wave sees on the shell.

    Attributes:
        events: the events, ``(latitude, longitude)`` in degrees.
        stations: the stations, likewise.
        s_arcs: the Sdiff arcs on the shell, ``(start, end)`` pairs.
        p_arcs: the Pdiff arcs.
        s_caps: the centres of the ScS caps, an array of points.
        p_caps: the centres of the PcP caps.
        kernels: the modes' weights, one row per mode, on (vs, vp, rho).
    """

    events: np.ndarray
    stations: np.ndarray
    s_arcs: tuple
    p_arcs: tuple
    s_caps: np.ndarray
    p_caps: np.ndarray
    kernels: np.ndarray


def _diffracted(
    sources: np.ndarray,
    receivers: np.ndarray,
    count: int,
    scale: Scale,
    /,
    *,
    rng: np.random.Generator,
) -> tuple:
    """The arcs of diffracted paths on the shell, as ``(start, end)`` pairs."""
    if count == 0:
        return ()
    a, b = _pairs(sources, receivers, count, scale.arc_range, rng=rng)
    fraction = np.radians(scale.descent) / separation(a, b)[:, None]
    starts = to_points(along(a, b, fraction))
    ends = to_points(along(a, b, 1.0 - fraction))
    return tuple((start, end) for start, end in zip(starts, ends))


def _reflected(
    sources: np.ndarray,
    receivers: np.ndarray,
    count: int,
    scale: Scale,
    /,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """The bounce points of core reflections on the shell."""
    if count == 0:
        return np.empty((0, 2))
    a, b = _pairs(sources, receivers, count, scale.cap_range, rng=rng)
    return to_points(along(a, b, 0.5))


def network(scale: Scale, /) -> Network:
    """The events, the stations and what their waves see, from the seed alone.

    Args:
        scale: the counts and the seed.

    Returns:
        The network, the same at every truncation degree.
    """
    rng = np.random.default_rng([scale.seed, 0])
    sources = events(scale.events, rng=rng)
    receivers = stations(scale.stations, rng=rng)
    p_arcs = int(round(scale.p_share * scale.arcs))
    p_caps = int(round(scale.p_share * scale.caps))
    return Network(
        sources,
        receivers,
        _diffracted(sources, receivers, scale.arcs - p_arcs, scale, rng=rng),
        _diffracted(sources, receivers, p_arcs, scale, rng=rng),
        _reflected(sources, receivers, scale.caps - p_caps, scale, rng=rng),
        _reflected(sources, receivers, p_caps, scale, rng=rng),
        np.array(KERNELS[: scale.modes], dtype=float),
    )
