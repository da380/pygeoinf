"""What the part scripts share: the command line, the generators, the figures."""

from __future__ import annotations

import argparse
import os
import pathlib
import platform
from typing import Any, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pygeoinf2 import plotting  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
SEED = 2026


def machine_name() -> str:
    """The processor's name, from ``/proc/cpuinfo`` where there is one."""
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def parse(description: str) -> argparse.Namespace:
    """The command line every part script takes: ``--quick``, ``--outdir``, ``--seed``."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--quick", action="store_true", help="the small scale")
    parser.add_argument(
        "--outdir", type=pathlib.Path, default=None, help="where the figures go"
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def output_directory(quick: bool, outdir: pathlib.Path | None) -> pathlib.Path:
    """Next to the sources at the full scale, in ``quick/`` at the quick one."""
    if outdir is None:
        outdir = HERE / "quick" if quick else HERE
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def part_rng(seed: int, part: str) -> np.random.Generator:
    """One generator per part, so a part gives the same answer run alone or in order."""
    return np.random.default_rng([seed, ord(part)])


def save(fig: Any, path: pathlib.Path) -> None:
    """Write the figure and close it; nothing is shown on screen."""
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"figure written to {path}")


def is_ball(space: Any, /) -> bool:
    """Whether a layered space is of balls, which are drawn by a section."""
    return space.spatial_dimension == 3


#: The section of a ball that holds the rays: its equatorial plane, seen from
#: above the north pole, longitude zero on the right and ninety at the top,
#: so that it is drawn as the disk is.
EQUATOR = (90.0, 0.0)


def panels(space: Any, rows: int, columns: int, /) -> tuple[Any, np.ndarray]:
    """A grid of panels for a disk, or for the section of a ball, as a
    two-dimensional array of axes."""
    if is_ball(space):
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(4.4 * columns, 4.0 * rows),
            layout="constrained",
            subplot_kw={"aspect": "equal"},
        )
    else:
        fig, axes = plotting.subplots(
            space, rows=rows, columns=columns, figsize=(4.4 * columns, 4.0 * rows)
        )
    return fig, np.atleast_1d(axes).reshape(rows, columns)


def draw(
    space: Any,
    field: Any,
    ax: Any,
    /,
    *,
    title: str,
    limit: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | None = None,
    label: str | None = None,
) -> None:
    """One disk, or the section of a ball through the plane of the rays, with
    a colorbar: signed about zero, or not."""
    signed = limit is not None
    options = dict(
        ax=ax,
        symmetric=signed,
        vmin=-limit if signed else vmin,
        vmax=limit if signed else vmax,
        cmap=cmap or ("RdBu_r" if signed else "viridis"),
        colorbar_label=label,
    )
    if is_ball(space):
        plotting.plot_section(space, field, pole=EQUATOR, **options)
    else:
        plotting.plot(space, field, **options)
    ax.set_title(title, fontsize=10)


def draw_rays(ax: Any, rays: Sequence[Any], /, **style: Any) -> None:
    """Rays as polylines through their nodes ``(radius, angle)``, in the plane
    of the drawing."""
    from matplotlib.collections import LineCollection

    segments = []
    for ray in rays:
        t = np.radians(ray.nodes[:, 1])
        segments.append(
            np.column_stack([ray.nodes[:, 0] * np.cos(t), ray.nodes[:, 0] * np.sin(t)])
        )
    ax.add_collection(LineCollection(segments, **style))


def outline_boxes(
    ax: Any, boxes: Sequence[tuple[str, tuple[float, float, float, float]]], /, **style
) -> None:
    """The edges of polar boxes ``(r1, r2, angle1, angle2)``, angles in degrees."""
    style = {"color": "black", "linewidth": 1.0, **style}
    for _, (r1, r2, a1, a2) in boxes:
        t = np.radians(np.linspace(a1, a2, 60))
        edge_r = np.concatenate([np.full(60, r1), np.full(60, r2)[::-1], [r1]])
        edge_t = np.concatenate([t, t[::-1], t[:1]])
        ax.plot(edge_r * np.cos(edge_t), edge_r * np.sin(edge_t), **style)


def stations(
    ax: Any, sources: Sequence[float], receivers: Sequence[float], /, *, radius: float
) -> None:
    """Sources as stars and receivers as triangles, just outside the surface."""
    for angles, marker, color, size in (
        (receivers, "v", "C0", 18),
        (sources, "*", "C3", 70),
    ):
        t = np.radians(np.asarray(angles, dtype=float))
        ax.scatter(
            1.04 * radius * np.cos(t),
            1.04 * radius * np.sin(t),
            marker=marker,
            s=size,
            color=color,
            zorder=5,
            clip_on=False,
        )


def variances(covariance: Any, /) -> np.ndarray:
    """The diagonal of a covariance on a Euclidean space of a few dimensions,
    read off its matrix: one application per dimension."""
    return np.clip(np.diag(covariance.matrix(form="components")), 0.0, None)
