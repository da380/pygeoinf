"""Part C: how far ray theory predicts the finite-frequency delay, under a
prior.

For one ray and one band the two functionals stacked are one operator, ``m
-> (dT_ray, dT_FF)``, from the model space onto ``R^2``, and a prior pushed
forward through it is a Gaussian on ``R^2``: the joint law of the two delays
before any datum. From its covariance come the two standard deviations,
their correlation ``rho``, the regression slope of the finite-frequency
delay on the ray delay, and the standard deviation of the finite-frequency
delay given the ray delay over its own, ``sqrt(1 - rho^2)``: the share of
the finite-frequency delay that ray theory cannot predict. The relative
standard deviation of the difference ``dT_FF - dT_ray`` is a reading of the
same covariance; its deterministic counterpart, which no prior enters, is
the relative dual norm of the difference of the two functionals in the
space's own norm.

A prior is placed by what its draws do, not by its nominal lengths: the
distance at which its correlation from the ray's turning point falls to
``1/e`` (:mod:`src.correlations`), along the shell for an isotropic prior
and along the radius for an anisotropic one, whose radial length is the
shorter; and that distance is measured in the band's wavelength at
mid-path, ``c T0``, the one length the waves set. Each band's Fresnel
half-width sits at its own place on that axis, so the same prior sits at
three places, one a band, and where the physics is the same the bands'
curves line up. A second figure takes the Fresnel half-width as the unit
instead.

The figures, at each distance: the Gaussian's density in every band under
the shortest isotropic prior, and under every prior in the widest band,
with the diagonal ``dT_FF = dT_ray``; and the numbers against the prior's
correlation distance in wavelengths, then in Fresnel half-widths. The
numbers for every distance, band and prior go to a table beside the
figures, every length in units of the radius and in wavelengths.
"""

from __future__ import annotations

import csv
import pathlib
import time
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from src.common import parse, save
from src.functionals import thread_map
from src.problem import FULL, QUICK, Problem, Scale, build, resolution

#: The probability inside each contour drawn, by its Mahalanobis radius.
SIGMAS = (1.0, 2.0, 3.0)
#: How far the density panels reach, in the ray delay's standard deviations.
WIDTH = 3.6
#: The delays are drawn in units of this.
UNIT = 1e-2
#: The table's name in the output directory.
TABLE = "agreement.csv"


@dataclass(frozen=True)
class Agreement:
    """The two delays of one ray in one band under one prior.

    Attributes:
        distance: the ray's distance, in degrees.
        period: the band's dominant period.
        wavelength: the band's wavelength at mid-path, ``c T0``.
        fresnel_half_width: the band's Fresnel half-width at mid-path.
        radial_length: the prior's nominal length scale along a radius.
        lateral_length: its nominal length scale across one.
        shell_distance: where the prior's correlation from the ray's
            turning point falls to ``1/e`` along the shell.
        radial_distance: where it does along the radius, NaN if not inside
            the mantle.
        std_ray: the ray delay's standard deviation.
        std_ff: the finite-frequency delay's standard deviation.
        correlation: their correlation.
        slope: the regression slope of the finite-frequency delay on the ray
            delay.
        conditional_std_ratio: the finite-frequency delay's standard
            deviation given the ray delay over its own,
            ``sqrt(1 - correlation^2)``.
        difference_std_ratio: the standard deviation of ``dT_FF - dT_ray``
            over the ray delay's.
        dual_norm_ratio: the norm of ``l_FF - l_ray`` in the dual of the
            model space over ``l_ray``'s, which no prior enters.
    """

    distance: float
    period: float
    wavelength: float
    fresnel_half_width: float
    radial_length: float
    lateral_length: float
    shell_distance: float
    radial_distance: float
    std_ray: float
    std_ff: float
    correlation: float
    slope: float
    conditional_std_ratio: float
    difference_std_ratio: float
    dual_norm_ratio: float

    @property
    def isotropic(self) -> bool:
        """Whether the prior has one nominal length."""
        return self.radial_length == self.lateral_length

    @property
    def reach(self) -> float:
        """The prior's correlation distance on the curves: along the shell
        for an isotropic prior, along the radius for an anisotropic one."""
        return self.shell_distance if self.isotropic else self.radial_distance

    @property
    def in_wavelengths(self) -> float:
        """That distance in the band's wavelengths."""
        return self.reach / self.wavelength

    @property
    def in_fresnel_widths(self) -> float:
        """That distance in the band's Fresnel half-widths."""
        return self.reach / self.fresnel_half_width


def statistics(covariance: np.ndarray, /) -> dict[str, float]:
    """The standard deviations, the correlation, the slope and the two
    ratios of a covariance of ``(dT_ray, dT_FF)``."""
    a, b, c = (
        covariance[0, 0],
        covariance[1, 1],
        0.5 * (covariance[0, 1] + covariance[1, 0]),
    )
    rho = c / np.sqrt(a * b)
    return {
        "std_ray": float(np.sqrt(a)),
        "std_ff": float(np.sqrt(b)),
        "correlation": float(rho),
        "slope": float(c / a),
        "conditional_std_ratio": float(np.sqrt(max(0.0, (a * b - c * c) / (a * b)))),
        "difference_std_ratio": float(np.sqrt(max(0.0, a + b - 2.0 * c) / a)),
    }


def pushed_covariance(
    problem: Problem, index: int, band: int, key: tuple[float, float], /
) -> np.ndarray:
    """The covariance of the prior of those nominal lengths pushed forward
    through the two functionals of one ray in one band, a Gaussian on
    ``R^2``."""
    pushed = problem.prior(*key).push_forward(problem.pair_operator(index, band))
    return np.asarray(pushed.covariance.matrix(form="components"))


def dual_norm_ratio(problem: Problem, index: int, band: int, /) -> float:
    """``||l_FF - l_ray|| / ||l_ray||`` in the dual of the model space: the
    norm of the difference of the two Sobolev gradients over the ray's."""
    space = problem.space
    ray = problem.representer(index)
    difference = space.subtract(problem.representer(index, band), ray)
    return float(space.norm(difference) / space.norm(ray))


def sweep(
    problem: Problem,
    indices: Sequence[int],
    /,
    *,
    priors: Sequence[tuple[float, float]] | None = None,
    n_jobs: int | None = None,
) -> list[Agreement]:
    """The agreement of the two delays of each ray chosen, in each band,
    under each prior (the scale's by default).

    The functionals' rows are formed first, in ``n_jobs`` threads, which is
    most of the cost; then each prior's correlation from each ray's turning
    point, and every push-forward, likewise, each the prior's covariance
    applied to two rows.
    """
    priors = problem.scale.prior_scales if priors is None else list(priors)
    pairs = [(i, b) for i in indices for b in range(len(problem.bands))]
    problem.rows(
        [(i, None) for i in indices] + [(i, b) for i, b in pairs], n_jobs=n_jobs
    )
    duals = {pair: dual_norm_ratio(problem, *pair) for pair in pairs}
    # Every prior made here, before the threads share the problem's store.
    for key in priors:
        problem.prior(*key)
    places = [(key, i) for i in indices for key in priors]
    reached = thread_map(
        lambda place: problem.correlation(*place), places, n_jobs=n_jobs
    )
    found = dict(zip(places, reached))
    cases = [(i, b, key) for i, b in pairs for key in priors]
    covariances = thread_map(
        lambda case: pushed_covariance(problem, *case), cases, n_jobs=n_jobs
    )
    rows = []
    for (i, b, key), covariance in zip(cases, covariances):
        path, band = problem.paths[i], problem.bands[b]
        rows.append(
            Agreement(
                distance=path.distance,
                period=band.period,
                wavelength=path.wavelength(band.period),
                fresnel_half_width=path.fresnel_half_width(band.period),
                radial_length=key[0],
                lateral_length=key[1],
                shell_distance=found[(key, i)].shell_distance,
                radial_distance=found[(key, i)].radial_distance,
                dual_norm_ratio=duals[(i, b)],
                **statistics(covariance),
            )
        )
    return rows


def covariance_of(row: Agreement, /) -> np.ndarray:
    """The covariance of ``(dT_ray, dT_FF)`` back from its numbers."""
    c = row.correlation * row.std_ray * row.std_ff
    return np.array([[row.std_ray**2, c], [c, row.std_ff**2]])


# --------------------------------------------------------------------------- #
#                                  Drawing                                    #
# --------------------------------------------------------------------------- #


def prior_label(row: Agreement, /) -> str:
    """A prior by its nominal lengths, short."""
    if row.isotropic:
        return f"prior {row.radial_length:g}"
    return f"prior {row.radial_length:g} x {row.lateral_length:g}"


def density_panel(ax: Any, row: Agreement, /, *, title: str) -> None:
    """The density of ``(dT_ray, dT_FF)`` as its contours at one, two and
    three standard deviations, filled, with the diagonal and the line of
    regression."""
    covariance = covariance_of(row)
    values, vectors = np.linalg.eigh(covariance)
    root = vectors * np.sqrt(np.clip(values, 0.0, None))
    t = np.linspace(0.0, 2.0 * np.pi, 721)
    circle = np.stack([np.cos(t), np.sin(t)])
    for sigma, shade in zip(SIGMAS[::-1], ("#c6dbef", "#6baed6", "#2171b5")):
        x, y = sigma * (root @ circle)
        ax.fill(x, y, color=shade, lw=0.6, ec="#08306b")
    reach = WIDTH * max(row.std_ray, row.std_ff)
    line = np.array([-reach, reach])
    ax.plot(line, line, "k--", lw=0.8, label="δT_FF = δT_ray")
    ax.plot(line, row.slope * line, color="C3", lw=0.8, label="regression")
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal")
    ax.set_xlabel("δT_ray", fontsize=8)
    ax.set_ylabel("δT_FF", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.ticklabel_format(style="sci", scilimits=(-2, 2))
    ax.set_title(title, fontsize=9)


def densities_figure(
    rows: Sequence[Agreement],
    distance: float,
    /,
    *,
    prior: tuple[float, float],
    period: float,
    outdir: pathlib.Path,
) -> pathlib.Path:
    """At one distance: the density in every band under one prior, and under
    every prior in one band."""
    here = [row for row in rows if row.distance == distance]
    by_band = [row for row in here if (row.radial_length, row.lateral_length) == prior]
    by_prior = [row for row in here if row.period == period]
    columns = max(3, len(by_band))
    lower = -(-len(by_prior) // columns)
    fig = plt.figure(figsize=(3.4 * columns, 3.6 + 3.8 * lower), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[3.6, 3.8 * lower])
    for part, chosen, heading, name in (
        (top, by_band, prior_label(by_band[0]), lambda r: f"T0 = {r.period:g}"),
        (bottom, by_prior, f"T0 = {period:g}", prior_label),
    ):
        axes = np.atleast_1d(part.subplots(-(-len(chosen) // columns), columns)).ravel()
        for ax in axes[len(chosen) :]:
            ax.set_axis_off()
        for ax, row in zip(axes, chosen):
            density_panel(ax, row, title=name(row))
        part.supylabel(heading, fontsize=10, fontweight="bold")
        if part is top:
            axes[0].legend(fontsize=7, loc="lower right")
    file = f"agreement_{distance:g}deg.png"
    save(fig, outdir / file, title=f"Joint law of the two delays, {distance:g} degrees")
    return outdir / file


#: The numbers the curves show: the field, the title and the vertical scale.
PANELS = (
    ("std_ff", "std of δT", "linear"),
    ("correlation", "1 − ρ", "log"),
    ("slope", "slope of δT_FF on δT_ray", "linear"),
    ("conditional_std_ratio", "conditional std ratio", "log"),
    ("difference_std_ratio", "difference std ratio", "log"),
)


def curves_figure(
    rows: Sequence[Agreement],
    distance: float,
    /,
    *,
    scale: Scale,
    outdir: pathlib.Path,
    unit: str = "wavelength",
) -> pathlib.Path:
    """At one distance: the numbers against the prior's measured correlation
    distance, a curve a band through the isotropic priors and a marker a
    pair.

    Args:
        rows: the numbers.
        distance: the distance drawn.
        scale: the scale, for the title.
        outdir: where the figure goes.
        unit: ``"wavelength"``, the distance in the band's wavelengths at
            mid-path, each band's Fresnel half-width marked in its color; or
            ``"fresnel"``, in the band's Fresnel half-widths, one marked.
    """
    here = [row for row in rows if row.distance == distance]
    periods = sorted({row.period for row in here})
    pairs = sorted(
        {(r.radial_length, r.lateral_length) for r in here if not r.isotropic}
    )
    markers = ("s", "D", "^", "v", "P", "X")
    waves = unit == "wavelength"

    def where(r: Agreement) -> float:
        return r.in_wavelengths if waves else r.in_fresnel_widths

    def value(r: Agreement, field: str) -> float:
        return 1.0 - r.correlation if field == "correlation" else getattr(r, field)

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.6), layout="constrained")
    axes = axes.ravel()
    for ax, (field, label, axis_scale) in zip(axes, PANELS):
        for k, period in enumerate(periods):
            color = f"C{k}"
            band = [r for r in here if r.period == period]
            line = sorted(
                (r for r in band if r.isotropic and np.isfinite(where(r))), key=where
            )
            ax.plot(
                [where(r) for r in line],
                [value(r, field) for r in line],
                "o-",
                color=color,
                ms=4,
                label=f"T0 = {period:g}",
            )
            if field == "std_ff":
                ax.plot(
                    [where(r) for r in line],
                    [r.std_ray for r in line],
                    "--",
                    color=color,
                    lw=0.9,
                )
            for pair, marker in zip(pairs, markers):
                for r in band:
                    if (r.radial_length, r.lateral_length) == pair and np.isfinite(
                        where(r)
                    ):
                        ax.plot(
                            where(r),
                            value(r, field),
                            marker,
                            color=color,
                            ms=6,
                            mfc="none",
                        )
            if waves:
                ax.axvline(
                    band[0].fresnel_half_width / band[0].wavelength,
                    color=color,
                    ls=":",
                    lw=1.0,
                )
            if field == "difference_std_ratio":
                ax.axhline(band[0].dual_norm_ratio, color=color, ls="--", lw=0.9)
        if not waves:
            ax.axvline(1.0, color="black", ls=":", lw=1.0)
        ax.set_xscale("log")
        low, high = ax.get_xlim()
        ticks = [t for t in 2.0 ** np.arange(-8, 9) if low <= t <= high]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(
            "ratio to wavelength" if waves else "ratio to Fresnel half-width",
            fontsize=8,
        )
        ax.set_yscale(axis_scale)
        ax.set_title(label, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, which="both", lw=0.3, alpha=0.5)
    key = axes[-1]
    key.set_axis_off()
    for k, period in enumerate(periods):
        key.plot([], [], "o-", color=f"C{k}", label=f"T0 = {period:g}")
    if waves:
        key.plot([], [], ":", color="0.3", label="Fresnel half-width")
    else:
        key.plot([], [], ":", color="black", label="Fresnel half-width")
    key.plot([], [], "--", color="0.3", lw=0.9, label="ray std")
    for pair, marker in zip(pairs, markers):
        key.plot(
            [],
            [],
            marker,
            color="0.3",
            mfc="none",
            label=f"prior {pair[0]:g} x {pair[1]:g}",
        )
    key.plot([], [], "k--", lw=0.9, label="dual norm ratio")
    key.legend(loc="center", fontsize=9, frameon=False)
    name = f"agreement_{'curves' if waves else 'fresnel'}_{distance:g}deg.png"
    save(fig, outdir / name, title=f"Agreement of the two delays, {distance:g} degrees")
    return outdir / name


def write_table(
    rows: Sequence[Agreement], path: pathlib.Path, /, *, scale: Scale
) -> None:
    """The numbers as a table with a title, one row a distance, band and
    prior, every length in units of the radius and in the band's
    wavelengths."""
    extra = [
        "fresnel_half_width_wavelengths",
        "shell_distance_wavelengths",
        "radial_distance_wavelengths",
        "length_scale_wavelengths",
        "resolution_wavelengths",
    ]
    fields = list(asdict(rows[0])) + extra
    finest = resolution(scale)
    with open(path, "w", newline="") as stream:
        stream.write(
            "# Banana-doughnut, part C: the ray and finite-frequency delays under "
            f"Sobolev priors of order {scale.prior_order:g} and pointwise std "
            f"{scale.prior_std:g}, on H^{scale.order:g} of length scale "
            f"{scale.length_scale:g} at lmax {scale.lmax} (resolving about "
            f"{finest:.4f} at the surface), the surface by "
            f"{'images' if scale.surface == 'fold' else 'clipping'}\n"
        )
        stream.write(
            "# distances in degrees; lengths in units of the planet's radius, and "
            "in the band's wavelength at mid-path where the name says so; standard "
            "deviations in units of time; shell_distance and radial_distance are "
            "where each prior's correlation from the ray's turning point falls to "
            "1/e along the shell and along the radius (nan: not inside the "
            "mantle)\n"
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            values = asdict(row)
            values["fresnel_half_width_wavelengths"] = (
                row.fresnel_half_width / row.wavelength
            )
            values["shell_distance_wavelengths"] = row.shell_distance / row.wavelength
            values["radial_distance_wavelengths"] = row.radial_distance / row.wavelength
            values["length_scale_wavelengths"] = scale.length_scale / row.wavelength
            values["resolution_wavelengths"] = finest / row.wavelength
            writer.writerow({k: f"{v:.10g}" for k, v in values.items()})
    print(f"table written to {path}")


def print_table(rows: Sequence[Agreement], distance: float, /) -> None:
    """The numbers at one distance."""
    print(
        f"  at {distance:g} degrees: {'T0':>6s} {'prior':>9s} {'1/e at':>7s} "
        f"{'in wl':>6s} {'in Fr':>6s} {'std ray':>9s} {'std FF':>9s} "
        f"{'rho':>10s} {'slope':>7s} {'cond':>8s} {'diff':>8s} {'dual':>8s}"
    )
    for row in rows:
        if row.distance != distance:
            continue
        prior = (
            f"{row.radial_length:g}"
            if row.isotropic
            else f"{row.radial_length:g},{row.lateral_length:g}"
        )
        print(
            f"  {'':>15s} {row.period:6g} {prior:>9s} {row.reach:7.3f} "
            f"{row.in_wavelengths:6.2f} {row.in_fresnel_widths:6.2f} "
            f"{row.std_ray:9.3e} {row.std_ff:9.3e} {row.correlation:10.7f} "
            f"{row.slope:7.4f} {row.conditional_std_ratio:8.2e} "
            f"{row.difference_std_ratio:8.2e} {row.dual_norm_ratio:8.2e}"
        )


def run(
    scale: Scale,
    /,
    *,
    outdir: pathlib.Path,
    problem: Problem | None = None,
    distances: Sequence[float] | None = None,
) -> list[Agreement]:
    """The agreement at every distance of the scale (or those given), the
    table, and the figures of each distance."""
    if problem is None:
        problem = build(scale=scale)
    print(f"\n=== Part C: the agreement of the two delays ({scale.name} scale) ===")
    wanted = scale.distances if distances is None else distances
    indices = [
        i
        for i, path in enumerate(problem.paths)
        if any(np.isclose(path.distance, d) for d in wanted)
    ]
    t0 = time.perf_counter()
    rows = sweep(problem, indices, n_jobs=scale.n_jobs)
    print(
        f"{len(indices)} rays, {len(problem.bands)} bands and "
        f"{len(scale.prior_scales)} priors: {len(rows)} push-forwards in "
        f"{time.perf_counter() - t0:.1f} s; a prior placed by its 1/e distance "
        "along the shell, a pair by the radial one, in units of the radius, in "
        "wavelengths (wl) and in Fresnel half-widths (Fr)"
    )
    finest = resolution(scale)
    for band in problem.bands:
        path = problem.paths[problem.reference]
        wave = path.wavelength(band.period)
        print(
            f"  T0 = {band.period:g}: wavelength {band.period:g} at the surface, "
            f"{wave:.4f} at mid-path at {path.distance:g} degrees; there the Fresnel "
            f"half-width {path.fresnel_half_width(band.period) / wave:.2f}, the "
            f"space's length scale {scale.length_scale / wave:.3f} and its "
            f"resolution {finest / wave:.3f} wavelengths"
        )
    reference = problem.paths[problem.reference].distance
    order = [reference] + [
        problem.paths[i].distance
        for i in indices
        if problem.paths[i].distance != reference
    ]
    for distance in order:
        if any(row.distance == distance for row in rows):
            print_table(rows, distance)
    write_table(rows, outdir / TABLE, scale=scale)
    shortest = (float(min(scale.prior_lengths)),) * 2
    widest = float(max(scale.periods))
    for distance in order:
        if not any(row.distance == distance for row in rows):
            continue
        densities_figure(rows, distance, prior=shortest, period=widest, outdir=outdir)
        for unit in ("wavelength", "fresnel"):
            curves_figure(rows, distance, scale=scale, outdir=outdir, unit=unit)
    return rows


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
