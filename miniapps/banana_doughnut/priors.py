"""The priors: what their draws do.

A prior here is a Sobolev measure of order two on the model space with
length scales of its own, its covariance an operator in the space's metric,
so its nominal lengths are not the lengths over which its draws decorrelate:
the space's own scale enters too. Each prior is therefore characterized by
its draws, at the reference ray's turning point, the middle of the path:
its correlation from there along the shell through the point, in the ray's
direction, and along the radius both ways, computed exactly from the
covariance (:mod:`src.correlations`), with the distance at which it falls to
``1/e`` in each direction; and one draw, on the two sections the other parts
use, across the ray at its turning point and in the ray's plane. Those
measured distances, not the nominal lengths, are what the comparison of
part C is laid out against. They are printed, with each band's wavelength
and Fresnel half-width at mid-path, in the radius's units and in
wavelengths, and written to a table beside the figure.
"""

from __future__ import annotations

import csv
import pathlib
import time
from typing import Any

import numpy as np

import matplotlib.pyplot as plt

from src import sections
from src.common import parse, save
from src.correlations import LEVEL, Correlation, describe
from src.functionals import thread_map
from src.problem import FULL, QUICK, Problem, Scale, build

#: How far from the ray the draws are drawn, in units of the radius.
REACH = 0.3
#: The draws' color scale, in the priors' pointwise standard deviations.
SPREAD = 3.0
#: The table's name in the output directory.
TABLE = "priors.csv"


def short(key: tuple[float, float], /) -> str:
    """A prior in a panel's title, by its nominal lengths."""
    radial, lateral = key
    return (
        f"prior {radial:g}" if radial == lateral else f"prior {radial:g} x {lateral:g}"
    )


def nominal(key: tuple[float, float], /) -> str:
    """A prior by its nominal lengths."""
    radial, lateral = key
    if radial == lateral:
        return f"nominal length {radial:g}"
    return f"nominal lengths {radial:g} radial, {lateral:g} lateral"


def correlation_panel(
    ax: Any, problem: Problem, key: tuple[float, float], found: Correlation, /
) -> None:
    """The correlation along the shell and the radius, the level ``1/e``,
    the nominal lengths and each band's Fresnel half-width at mid-path."""
    path = problem.paths[problem.reference]
    ax.plot(found.shell_offsets, found.shell, "k-", lw=1.2, label="shell")
    ax.plot(found.down_offsets, found.down, "k--", lw=1.0, label="down")
    ax.plot(found.up_offsets, found.up, "k:", lw=1.0, label="up")
    ax.axhline(LEVEL, color="0.5", lw=0.7)
    for distance in (found.shell_distance, found.radial_distance):
        if np.isfinite(distance):
            ax.plot(distance, LEVEL, "o", color="C3", ms=4)
    for length in sorted(set(key)):
        ax.axvline(length, color="0.6", lw=0.8, ls="-.")
    for k, band in enumerate(problem.bands):
        ax.axvline(
            path.fresnel_half_width(band.period),
            color=f"C{k}",
            lw=0.9,
            ls=":",
            label=f"Fresnel, T0 = {band.period:g}",
        )
    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(-0.2, 1.02)
    ax.set_xlabel("distance", fontsize=8)
    ax.set_ylabel("correlation", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(short(key), fontsize=9)


def figure(
    problem: Problem,
    keys: list[tuple[float, float]],
    found: dict,
    wholes: dict,
    /,
    *,
    outdir: pathlib.Path,
) -> pathlib.Path:
    """A row a prior: its correlation, and a draw across the ray and in the
    ray's plane."""
    scale = problem.scale
    path = problem.paths[problem.reference]
    limit = SPREAD * scale.prior_std
    fig, axes = plt.subplots(
        len(keys),
        3,
        figsize=(15.0, 2.9 * len(keys)),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.2, 1.0, 1.6]},
        layout="constrained",
    )
    for row, key in zip(axes, keys):
        correlation_panel(row[0], problem, key, found[key])
        for ax, which in zip(row[1:], ("across", "plane")):
            sections.draw(
                ax,
                wholes[(key, which)],
                path,
                which,
                reach=REACH,
                limit=limit,
                label="δln c",
            )
            place = "across the ray" if which == "across" else "ray plane"
            ax.set_title(f"{short(key)}, draw, {place}", fontsize=9)
    axes[0, 0].legend(fontsize=6.5, loc="upper right")
    name = f"priors_{path.distance:g}deg.png"
    save(fig, outdir / name, title="Priors: correlation and a draw")
    return outdir / name


def write_table(
    problem: Problem,
    keys: list[tuple[float, float]],
    found: dict,
    path: pathlib.Path,
    /,
) -> None:
    """The measured distances of every prior, in the radius's units and in
    each band's wavelengths at mid-path, with a title."""
    scale = problem.scale
    ray = problem.paths[problem.reference]
    waves = {band.period: ray.wavelength(band.period) for band in problem.bands}
    fields = [
        "radial_length",
        "lateral_length",
        "shell_distance",
        "down_distance",
        "up_distance",
        "radial_distance",
    ] + [f"shell_distance_wavelengths_T{period:g}" for period in waves]
    with open(path, "w", newline="") as stream:
        stream.write(
            "# Banana-doughnut, the priors: where the correlation of each prior's "
            f"draws from the turning point of the ray at {ray.distance:g} degrees "
            "falls to 1/e, along the shell and down and up the radius, in units of "
            "the radius (nan: not inside the mantle) and in each band's "
            "wavelength at mid-path ("
            + ", ".join(f"T0 {p:g}: {w:.4f}" for p, w in waves.items())
            + f"); space H^{scale.order:g}, length scale {scale.length_scale:g}, "
            f"lmax {scale.lmax}\n"
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key in keys:
            c = found[key]
            row = {
                "radial_length": key[0],
                "lateral_length": key[1],
                "shell_distance": c.shell_distance,
                "down_distance": c.down_distance,
                "up_distance": c.up_distance,
                "radial_distance": c.radial_distance,
            }
            for period, wave in waves.items():
                row[f"shell_distance_wavelengths_T{period:g}"] = c.shell_distance / wave
            writer.writerow({k: f"{v:.6g}" for k, v in row.items()})
    print(f"table written to {path}")


def run(
    scale: Scale, /, *, outdir: pathlib.Path, problem: Problem | None = None
) -> dict:
    """Every prior of the scale: its correlation, its distances and a draw,
    in a figure and a table."""
    if problem is None:
        problem = build(scale=scale)
    print(f"\n=== The priors ({scale.name} scale) ===")
    keys = scale.prior_scales
    index = problem.reference
    path = problem.paths[index]
    t0 = time.perf_counter()
    for key in keys:
        problem.prior(*key)
    thread_map(lambda key: problem.correlation(key, index), keys, n_jobs=scale.n_jobs)
    found = {key: problem.correlation(key, index) for key in keys}
    draws = {
        key: problem.prior(*key).sample(rng=np.random.default_rng([scale.seed, k]))
        for k, key in enumerate(keys)
    }
    cases = [(key, which) for key in keys for which in ("across", "plane")]
    made = thread_map(
        lambda case: sections.section(problem.space, draws[case[0]], path, case[1]),
        cases,
        n_jobs=scale.n_jobs,
    )
    wholes = dict(zip(cases, made))
    print(
        f"{len(keys)} priors' correlations and draws in "
        f"{time.perf_counter() - t0:.1f} s"
    )
    for band in problem.bands:
        wave = path.wavelength(band.period)
        width = path.fresnel_half_width(band.period)
        print(
            f"  T0 = {band.period:g}: wavelength {wave:.4f} at mid-path "
            f"({band.period:g} at the surface), Fresnel half-width {width:.3f} "
            f"({width / wave:.2f} wavelengths)"
        )
    print(
        f"  the space's length scale {scale.length_scale:g}, its resolution about "
        f"{np.pi / scale.lmax:.3f} at the surface"
    )
    for key in keys:
        c = found[key]
        waves = ", ".join(
            f"{c.shell_distance / path.wavelength(band.period):.2f}"
            for band in problem.bands
        )
        print(
            f"  {nominal(key):<36s} {describe(c)}; along the shell in each band's "
            f"wavelengths {waves}"
        )
    write_table(problem, keys, found, outdir / TABLE)
    file = figure(problem, keys, found, wholes, outdir=outdir)
    return {"correlations": found, "figure": file, "table": outdir / TABLE}


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
