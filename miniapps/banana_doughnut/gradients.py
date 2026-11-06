"""Part B: the two functionals' kernels in L2 and their Sobolev equivalents.

A traveltime is a linear functional on the model space, ``dln c`` on the
mantle annulus in ``H^2``. Its ``L2`` kernel is what seismology draws: for
ray theory a line mass on the ray, nothing to draw but the ray, and for
finite-frequency theory the hollow banana of part A, a doughnut across the
ray. In the space the functional has a representer: in the space's own
metric its Sobolev gradient ``g``, with ``(g, m) = dT(m)`` for every model,
which the library gives as the functional's adjoint applied to one; under a
prior with covariance ``C``, its representer in the prior's Cameron-Martin
space, ``C g``, which is the covariance between the field at each point and
the delay, what a datum depends on.

One figure per band and prior, and section: a two-by-two grid, the ``L2``
kernels in the top row, ray theory's on the left and finite-frequency
theory's on the right, and their Sobolev equivalents below, ``C g`` of each
under the prior on one color scale, so that the two compare; and the same
for the gradients in the space's own metric, the case of no prior. The
space's length scale, 0.02, is below every Fresnel half-width, so the
gradient keeps the doughnut's hole in the wider bands, and what fills it is
the prior. The two sections are part A's: across the ray at its turning
point, which is mid-path for a source at the surface, the figure that
carries the point; and the ray's own plane, with the ray as a line. Both
reach three Fresnel half-widths of the band from the ray, as part A's do.
The middle band under the middle isotropic prior is drawn first, at the
reference distance, then every other band and prior of the scale.

Last, one picture of the whole mantle for that first band and prior and
the scale's deepest ray, the one to the largest distance: the two ``C g``
on the faces of the mantle with a wedge cut from it north of the equator,
a quarter of the way round from mid-path back past the source, so that the
wedge's second face, seen nearly square-on, is the section across the ray
at mid-path and its floor the ray's own plane from the source to mid-path,
with the ray drawn on it and the core a plain sphere standing in the cut,
as :func:`pygeoinf.plotting.plot_cutaway` draws a field of the ball
through ``planetmodel``; ray theory's beside finite-frequency theory's, on
one color scale.
"""

from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from setup import DRAWN_NODES
from setup import REACH as KERNEL_REACH
from setup import across as kernel_across
from setup import ray_plane as kernel_plane
from src import sections
from src.common import parse, save
from src.correlations import turning_point
from src.functionals import thread_map
from src.planet import CORE_RADIUS, trace
from src.problem import FULL, QUICK, Problem, Scale, build

#: How a field is named in the files: none for the space's own gradient.
Key = tuple[float, float] | None
#: The fill on the ray above which a field counts as filled in, and below
#: which its hole persists, in the captions.
FILLED, HOLLOW = 0.95, 0.5
#: The contours drawn on each Sobolev field, as fractions of its own extreme.
SHAPE = (0.25, 0.5, 0.75)
#: The cut-away: the wedge's opening in degrees of longitude, west of the
#: ray's turning point, where its second face is; the pixels across the
#: picture; and the colors of the surface and of the core, plain, the
#: field being on the faces of the cut alone. The view is planetmodel's
#: default, which shows the second face nearly square-on.
WEDGE_OPENING = 90.0
CUTAWAY_PIXELS = 400
SURFACE_COLOR = "0.85"
CORE_COLOR = "0.6"


def first(problem: Problem, /) -> tuple[int, tuple[float, float]]:
    """The band and prior drawn first: the middle band, by its index, and
    the middle isotropic prior."""
    lengths = sorted(problem.scale.prior_lengths)
    middle = float(lengths[(len(lengths) - 1) // 2])
    return (len(problem.bands) - 1) // 2, (middle, middle)


def prior_tag(key: Key, /) -> str:
    """A prior in a file's name."""
    if key is None:
        return "noprior"
    radial, lateral = key
    return f"prior{radial:g}" if radial == lateral else f"prior{radial:g}x{lateral:g}"


def short_name(key: Key, /) -> str:
    """A prior in a figure's title: its nominal lengths, or no prior."""
    if key is None:
        return "no prior (H^2 gradient)"
    radial, lateral = key
    return (
        f"prior {radial:g}" if radial == lateral else f"prior {radial:g} x {lateral:g}"
    )


def prior_name(key: Key, /) -> str:
    """A prior by its nominal lengths, as the titles name it."""
    if key is None:
        return "no prior"
    radial, lateral = key
    if radial == lateral:
        return f"the prior of nominal length {radial:g}"
    return f"the prior of nominal lengths {radial:g} radial, {lateral:g} lateral"


@dataclass
class Fields:
    """The fields part B draws, for one ray.

    Attributes:
        index: the ray's index.
        gradients: by band (``None`` for ray theory), the Sobolev gradient.
        images: by prior and band, ``C g``, the covariance of the field with
            the delay.
        stds: by prior and band, the delay's standard deviation.
    """

    index: int
    gradients: dict[int | None, np.ndarray]
    images: dict[tuple[tuple[float, float], int | None], np.ndarray]
    stds: dict[tuple[tuple[float, float], int | None], float]
    wholes: dict = field(default_factory=dict, repr=False)

    def of(self, key: Key, band: int | None, /) -> np.ndarray:
        """The field drawn for a prior, or none, and a band."""
        return self.gradients[band] if key is None else self.images[(key, band)]


def fields(
    problem: Problem,
    index: int,
    /,
    *,
    priors: Sequence[tuple[float, float]],
    bands: Sequence[int] | None = None,
    n_jobs: int | None = None,
) -> Fields:
    """The Sobolev gradients of ray theory's functional and of
    finite-frequency theory's in each band (or the bands given, by index),
    for one ray, and each prior's covariance applied to them; the rows and
    the covariances in ``n_jobs`` threads."""
    space = problem.space
    chosen = range(len(problem.bands)) if bands is None else bands
    bands = [None, *chosen]
    problem.rows([(index, band) for band in bands], n_jobs=n_jobs)
    gradients = {band: problem.representer(index, band) for band in bands}
    for key in priors:
        problem.prior(*key)
    cases = [(key, band) for key in priors for band in bands]
    made = thread_map(
        lambda case: problem.prior(*case[0]).covariance(gradients[case[1]]),
        cases,
        n_jobs=n_jobs,
    )
    images = dict(zip(cases, made))
    stds = {
        case: float(np.sqrt(space.inner_product(images[case], gradients[case[1]])))
        for case in cases
    }
    return Fields(index, gradients, images, stds)


def whole_sections(
    problem: Problem,
    found: Fields,
    keys: Sequence[Key],
    /,
    *,
    n_jobs: int | None = None,
) -> None:
    """Every field's two sections, whole, kept on ``found``: computed in
    ``n_jobs`` threads, each field once whatever the band of its
    figure."""
    path = problem.paths[found.index]
    bands: list[int | None] = [None, *range(len(problem.bands))]
    wanted = [
        (key, band, which)
        for key in keys
        for band in bands
        for which in sections.SECTIONS
        if (key, band, which) not in found.wholes
    ]
    made = thread_map(
        lambda case: sections.section(
            problem.space, found.of(case[0], case[1]), path, case[2]
        ),
        wanted,
        n_jobs=n_jobs,
    )
    found.wholes.update(zip(wanted, made))


def fill_state(fill: float, /) -> str:
    """What a field does on the ray, in words: filled in, partly filled in,
    or the hole persisting."""
    if fill >= FILLED:
        return "filled in"
    if fill < HOLLOW:
        return "the hole persists"
    return "partly filled in"


def figure(
    problem: Problem,
    found: Fields,
    band: int,
    key: Key,
    which: str,
    /,
    *,
    drawn: Any,
    outdir: pathlib.Path,
) -> tuple[pathlib.Path, float, float]:
    """One band and prior on one section: the ``L2`` kernels above, their
    Sobolev equivalents below.

    Args:
        problem: the problem.
        found: the fields, their sections made.
        band: the band's index.
        key: the prior's nominal lengths, or ``None`` for the gradients in
            the space's own metric.
        which: ``"across"`` or ``"plane"``.
        drawn: the ray traced finely, for the ``L2`` kernel.
        outdir: where the figure goes.

    Returns:
        The file, and the fill on the ray of ray theory's field and of
        finite-frequency theory's.
    """
    space, scale = problem.space, problem.scale
    path = problem.paths[found.index]
    period = problem.bands[band].period
    reach = KERNEL_REACH * path.fresnel_half_width(period)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.6, 10.6) if which == "across" else (14.0, 8.4),
        layout="constrained",
    )
    (ray_kernel, ff_kernel), (ray_field, ff_field) = axes
    sections.frame(ray_kernel, path, which, reach=reach)
    ray_kernel.set_title("ray kernel", fontsize=9)
    if which == "across":
        kernel_across(ff_kernel, drawn, problem.bands[band], scale=scale)
    else:
        kernel_plane(ff_kernel, drawn, problem.bands[band], scale=scale)
    ff_kernel.set_title("finite-frequency kernel", fontsize=9)
    ff_kernel.tick_params(labelsize=7)
    wholes = [found.wholes[(key, None, which)], found.wholes[(key, band, which)]]
    limit = max(sections.largest(whole, path, which, reach=reach) for whole in wholes)
    label = "g" if key is None else "C ℓ"
    for ax, whole, name in zip(
        (ray_field, ff_field), wholes, ("ray", "finite frequency")
    ):
        sections.draw(ax, whole, path, which, reach=reach, limit=limit, label=label)
        sections.shape(ax, whole, path, which, reach=reach, levels=SHAPE)
        ax.set_title(f"{name}: {label}", fontsize=9)
    across = (
        wholes
        if which == "across"
        else [found.wholes[(key, None, "across")], found.wholes[(key, band, "across")]]
    )
    fills = [
        sections.filled(space, found.of(key, b), whole, path, reach=reach)
        for b, whole in zip((None, band), across)
    ]
    place = "across the ray" if which == "across" else "ray plane"
    name = f"gradients_{which}_{path.distance:g}deg_T{period:g}_{prior_tag(key)}.png"
    save(
        fig,
        outdir / name,
        title=f"Kernels and Sobolev equivalents, {place}: T0 = {period:g}, "
        f"{short_name(key)}",
    )
    return outdir / name, fills[0], fills[1]


def deepest(problem: Problem, /) -> int:
    """The index of the scale's deepest ray, the one to the largest
    distance."""
    return int(np.argmax([path.distance for path in problem.paths]))


def cutaway_geometry(
    space: Any,
    field: np.ndarray,
    /,
    *,
    longitudes: tuple[float, float],
    latitudes: tuple[float, float],
    inner: float,
    view: tuple[float, float] | None = None,
) -> Any:
    """The geometry of the cut-away :func:`pygeoinf.plotting.plot_cutaway`
    draws for the same wedge, view and inner radius, in degrees and
    latitudes, the view ``planetmodel``'s default if none is given:
    ``planetmodel``'s ``CutawaySlice`` of the field, a picture of two
    pixels, which projects points as the picture does and says which of
    them it shows."""
    from planetmodel import slices
    from pygeoinf import plotting

    options = {}
    if view is not None:
        options["view"] = (np.radians(90.0 - view[0]), np.radians(view[1]))
    return slices.cutaway(
        plotting.planet_model(space, field),
        "field",
        longitudes=tuple(np.radians(value) for value in longitudes),
        colatitudes=tuple(sorted(np.radians(90.0 - value) for value in latitudes)),
        inner=inner,
        n=2,
        **options,
    )


def cutaway_figure(
    problem: Problem,
    found: Fields,
    band: int,
    key: tuple[float, float],
    /,
    *,
    outdir: pathlib.Path,
) -> tuple[pathlib.Path, int]:
    """The covariance of the field with each delay of one ray, ``C g``, ray
    theory's beside finite-frequency theory's, on the faces of the mantle
    with a wedge cut from it north of the equator between the ray's turning
    point and a quarter of the way round west of it: the second face the
    section across the ray there, the floor the ray's own plane with the
    ray drawn on it where the picture shows it; on one color scale, the
    largest size either reaches in that plane.

    Returns:
        The file, and how many of the ray's nodes the picture shows.
    """
    from pygeoinf import plotting

    space = problem.space
    path = problem.paths[found.index]
    period = problem.bands[band].period
    turn = turning_point(path)[2]
    longitudes = (turn - WEDGE_OPENING, turn)
    latitudes = (0.0, 90.0)
    view = None
    pictured = [found.of(key, None), found.of(key, band)]
    limit = max(
        float(np.abs(sections.section(space, field, path, "plane")[2]).max())
        for field in pictured
    )
    geometry = cutaway_geometry(
        space,
        pictured[0],
        longitudes=longitudes,
        latitudes=latitudes,
        view=view,
        inner=CORE_RADIUS,
    )
    nodes = path.dynamic.frame.point
    seen = geometry.visible(nodes)
    x, y = geometry.project(nodes)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.6), layout="constrained")
    for ax, image, name in zip(axes, pictured, ("ray", "finite frequency")):
        plotting.plot_cutaway(
            space,
            image,
            ax=ax,
            longitudes=longitudes,
            latitudes=latitudes,
            view=view,
            inner=CORE_RADIUS,
            n=CUTAWAY_PIXELS,
            inner_color=CORE_COLOR,
            surface=SURFACE_COLOR,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        ax.plot(np.where(seen, x, np.nan), np.where(seen, y, np.nan), "k-", lw=0.9)
        ax.set_title(f"{name}: C ℓ", fontsize=9)
    fig.colorbar(
        ScalarMappable(Normalize(-limit, limit), "RdBu_r"),
        ax=axes,
        shrink=0.7,
        label="C ℓ",
    )
    name = f"gradients_cutaway_{path.distance:g}deg_T{period:g}_{prior_tag(key)}.png"
    save(
        fig,
        outdir / name,
        title=f"Sobolev equivalents in a cut-away: T0 = {period:g}, {short_name(key)}",
    )
    return outdir / name, int(np.count_nonzero(seen))


def run(
    scale: Scale,
    /,
    *,
    outdir: pathlib.Path,
    problem: Problem | None = None,
    bands: Sequence[int] | None = None,
    priors: Sequence[Key] | None = None,
) -> dict:
    """The figures of the reference ray: the middle band under the middle
    isotropic prior first, then every other band and prior of the scale, and
    the gradients in the space's own metric for each band.

    Args:
        scale: the scale.
        outdir: where the figures go.
        problem: the problem, built if not given.
        bands: the bands' indices to draw, every one by default.
        priors: the priors to draw, by their nominal lengths, ``None`` among
            them for the gradients; every one of the scale's and the
            gradients by default.
    """
    if problem is None:
        problem = build(scale=scale)
    print(
        f"\n=== Part B: the kernels and their Sobolev equivalents ({scale.name} "
        "scale) ==="
    )
    index = problem.reference
    path = problem.paths[index]
    band0, key0 = first(problem)
    bands = list(range(len(problem.bands))) if bands is None else list(bands)
    keys: list[Key] = [*scale.prior_scales, None] if priors is None else list(priors)
    order = [(band0, key0)] if band0 in bands and key0 in keys else []
    order += [(b, k) for k in keys for b in bands if (b, k) != (band0, key0)]
    t0 = time.perf_counter()
    found = fields(
        problem,
        index,
        priors=[k for k in keys if k is not None],
        n_jobs=scale.n_jobs,
    )
    print(
        f"{len(found.gradients)} gradients and {len(found.images)} images under "
        f"{len(keys) - (None in keys)} priors in {time.perf_counter() - t0:.1f} s"
    )
    t0 = time.perf_counter()
    whole_sections(problem, found, keys, n_jobs=scale.n_jobs)
    for key in keys:
        if key is not None:
            problem.correlation(key, index)
    print(f"their sections in {time.perf_counter() - t0:.1f} s")
    for b, band in enumerate(problem.bands):
        wave = path.wavelength(band.period)
        width = path.fresnel_half_width(band.period)
        print(
            f"  T0 = {band.period:g}: wavelength {wave:.4f} at mid-path, Fresnel "
            f"half-width {width:.3f} ({width / wave:.2f} wavelengths); drawn to "
            f"{KERNEL_REACH:g} half-widths from the ray"
        )
    for key in keys:
        if key is not None:
            c = problem.correlation(key, index)
            print(
                f"  {prior_name(key)}: 1/e at {c.shell_distance:.3f} along the "
                f"shell, {c.radial_distance:.3f} radially"
            )
    drawn = trace(path.distance, nodes=DRAWN_NODES, source_depth=scale.source_depth)
    t0 = time.perf_counter()
    numbers: dict[str, Any] = {"figures": [], "fills": {}}
    print(
        f"  at {path.distance:g} degrees, on the ray over the extreme across it "
        "(1 filled in, 0 hollow), ray theory's field and finite frequency's:"
    )
    for band, key in order:
        fills = []
        for which in sections.SECTIONS:
            file, ray_fill, ff_fill = figure(
                problem, found, band, key, which, drawn=drawn, outdir=outdir
            )
            numbers["figures"].append(file)
            fills = [ray_fill, ff_fill]
        numbers["fills"][(band, key)] = fills
        std = (
            ""
            if key is None
            else (
                f", delay std ray {found.stds[(key, None)]:.3e}, FF "
                f"{found.stds[(key, band)]:.3e}"
            )
        )
        print(
            f"    T0 = {problem.bands[band].period:<6g} {prior_name(key):<58s} "
            f"{fills[0]:+.2f} {fills[1]:+.2f} ({fill_state(fills[1])}){std}"
        )
    print(f"{len(numbers['figures'])} figures in {time.perf_counter() - t0:.1f} s")
    if (band0, key0) in order and key0 is not None:
        t0 = time.perf_counter()
        far = deepest(problem)
        pictured = (
            found
            if far == index
            else fields(problem, far, priors=[key0], bands=[band0], n_jobs=scale.n_jobs)
        )
        file, seen = cutaway_figure(problem, pictured, band0, key0, outdir=outdir)
        numbers["cutaway"] = file
        print(
            f"the cut-away of the ray at {problem.paths[far].distance:g} degrees, "
            f"T0 = {problem.bands[band0].period:g}, under {prior_name(key0)}, the "
            f"ray seen at {seen} of its "
            f"{len(problem.paths[far].dynamic.arc_length)} nodes, in "
            f"{time.perf_counter() - t0:.1f} s"
        )
    return numbers


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
