"""What the part scripts share: the command line, the generators, the figures.

Each part writes its own figures, one method to a figure: the point
estimate with its error budget, the measure as prior against posterior, the
set as the bound the prior's ball admits alone against the bracket the data
leave.
"""

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


def save(fig: Any, path: pathlib.Path, /, *, title: str) -> None:
    """Title the figure, write it and close it; nothing is shown on screen."""
    fig.suptitle(title)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"figure written to {path}")


def panels(space: Any, rows: int, columns: int, /) -> tuple[Any, np.ndarray]:
    """A grid of panels for a disk, as a two-dimensional array of axes."""
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
    """One disk with a colorbar: signed about zero, or not."""
    signed = limit is not None
    options = dict(
        ax=ax,
        symmetric=signed,
        vmin=-limit if signed else vmin,
        vmax=limit if signed else vmax,
        cmap=cmap or ("RdBu_r" if signed else "viridis"),
        colorbar_label=label,
    )
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


def error_budget_panels(
    axes: Any,
    names: Sequence[str],
    estimate: Any,
    propagated: Any,
    bias: Any,
    truth: Any,
    /,
    *,
    ylabel: str,
) -> None:
    """A point estimate of each property with its error budget, a panel each.

    The estimate with twice the standard deviation of the noise it
    propagates, and what it would have been on exact data, the truth plus
    the bias, which only the truth reveals.
    """
    for ax, name, e, s, b, t in zip(axes, names, estimate, propagated, bias, truth):
        ax.errorbar(
            0.0,
            e,
            yerr=2.0 * s,
            fmt="o",
            color="C1",
            capsize=4,
            label="estimate, 2 propagated sd",
        )
        ax.plot(
            0.0,
            t + b,
            "D",
            mfc="none",
            color="C1",
            ms=8,
            label="on exact data: truth plus bias",
        )
        ax.axhline(t, color="black", ls="--", lw=1.0, label="truth")
        ax.set_xticks([])
        ax.set_xlim(-1.0, 1.0)
        reach = max(abs(e - t) + 2.0 * s, abs(b), 2.0 * s, 1e-3)
        ax.set_ylim(t - 1.4 * reach, t + 1.4 * reach)
        ax.set_title(name, fontsize=10)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(fontsize=7, loc="lower left", frameon=False)


def bound_panels(
    axes: Any,
    names: Sequence[str],
    alone: Any,
    brackets: Any,
    truth: Any,
    /,
    *,
    ylabel: str,
) -> None:
    """For each property, a panel: the bound the prior's ball admits alone
    against the bracket the data leave, and the truth.

    Both are drawn as bars on an axis linear within one and logarithmic
    beyond, since a bracket the data narrow can be a small part of the bound
    it came from.
    """
    for ax, name, (a0, a1), (b0, b1), t in zip(axes, names, alone, brackets, truth):
        ax.plot([0, 0], [a0, a1], color="0.6", lw=10, solid_capstyle="butt")
        ax.plot([1, 1], [b0, b1], color="C0", lw=10, solid_capstyle="butt")
        ax.axhline(t, color="black", ls="--", lw=1.0, label="truth")
        ax.set_xticks([0, 1], ["the prior's\nball alone", "with the\ndata"])
        ax.set_xlim(-0.7, 1.7)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_title(name, fontsize=10)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(fontsize=7, loc="lower left", frameon=False)


def marginal_panels(
    axes: Any,
    names: Sequence[str],
    posterior: Any,
    prior: Any,
    truth: Any,
    /,
    *,
    xlabel: str,
) -> None:
    """Each property's marginal before and after, the prior's on its own
    axis to the right, with the truth."""
    for index, (ax, name) in enumerate(zip(axes, names)):
        plotting.plot_densities(
            posterior,
            prior=prior,
            truth=float(truth[index]),
            index=index,
            ax=ax,
            labels=["posterior"],
            prior_labels=["prior"],
            xlabel=xlabel,
            title=name,
        )


def mixture_marginal(mixture: Any, index: int, /) -> tuple[np.ndarray, ...]:
    """One coordinate's marginal of a Gaussian mixture on a Euclidean space:
    the weights, and each component's mean and standard deviation there.
    The marginal is the mixture of those Gaussians, exactly."""
    space = mixture.domain
    direction = space.basis_vector(index)
    means = np.array(
        [space.to_components(c.expectation)[index] for c in mixture.components]
    )
    deviations = np.array(
        [
            np.sqrt(max(space.to_components(c.covariance(direction))[index], 0.0))
            for c in mixture.components
        ]
    )
    return mixture.weights, means, deviations


def mixture_density(
    weights: np.ndarray, means: np.ndarray, deviations: np.ndarray, x: np.ndarray, /
) -> np.ndarray:
    """The density of a mixture of Gaussians on the line at ``x``."""
    z = (x[:, None] - means[None, :]) / deviations[None, :]
    return np.exp(-0.5 * z**2) / (np.sqrt(2.0 * np.pi) * deviations) @ weights


def grouped_components(
    weights: np.ndarray, means: np.ndarray, deviations: np.ndarray, /
) -> list[tuple[float, float, float]]:
    """A mixture's components on the line, those that coincide there merged:
    ``(weight, mean, deviation)``, components with the same deviation and
    means within a hundredth of it counted as one."""
    groups: list[list[float]] = []
    for w, m, s in sorted(zip(weights, means, deviations), key=lambda c: c[1]):
        last = groups[-1] if groups else None
        if (
            last is not None
            and abs(m - last[1]) < 1e-2 * s
            and abs(s - last[2]) < 1e-6 * s
        ):
            last[0] += w
        else:
            groups.append([w, m, s])
    return [tuple(group) for group in groups]


def mixture_marginal_panels(
    axes: Any,
    names: Sequence[str],
    posterior: Any,
    prior: Any,
    truth: Any,
    /,
    *,
    xlabel: str,
) -> None:
    """Each property's marginal before and after for Gaussian mixtures,
    drawn as the mixture's density with its weighted components beneath it,
    the prior's on its own axis to the right, and the truth.

    A mixture's marginal is the weighted sum of its components' Gaussians,
    evaluated here exactly; drawn as the Gaussian of the same mean and
    variance it would lose the modes that are the point of it. Components
    that coincide in the coordinate are drawn as one.
    """
    for index, (ax, name) in enumerate(zip(axes, names)):
        after = mixture_marginal(posterior, index)
        before = mixture_marginal(prior, index)
        spans = []
        for weights, means, deviations in (after, before):
            live = weights > 1e-4
            spans += [
                (means - 4.0 * deviations)[live].min(),
                (means + 4.0 * deviations)[live].max(),
            ]
        low = min(min(spans), float(truth[index]))
        high = max(max(spans), float(truth[index]))
        x = np.linspace(low, high, 2001)
        twin = ax.twinx()
        for target, marginal, color, style, label in (
            (twin, before, "green", ":", "prior"),
            (ax, after, "tab:blue", "-", "posterior"),
        ):
            target.plot(
                x, mixture_density(*marginal, x), style, color=color, label=label
            )
            groups = grouped_components(*marginal)
            if len(groups) > 1:
                for position, (w, m, s) in enumerate(groups):
                    if w < 1e-3:
                        continue
                    target.plot(
                        x,
                        w
                        * np.exp(-0.5 * ((x - m) / s) ** 2)
                        / (np.sqrt(2 * np.pi) * s),
                        style,
                        color=color,
                        lw=0.7,
                        alpha=0.6,
                        label=f"{label}'s components" if position == 0 else None,
                    )
            target.set_ylim(bottom=0.0)
        twin.set_ylabel("prior density", color="green")
        twin.tick_params(axis="y", labelcolor="green")
        ax.axvline(float(truth[index]), color="black", lw=1.5, ls="--", label="truth")
        ax.set_xlim(low, high)
        ax.set_ylabel("posterior density")
        ax.set_xlabel(xlabel)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_title(name, fontsize=10)
        handles, labels = ax.get_legend_handles_labels()
        extra, extra_labels = twin.get_legend_handles_labels()
        ax.legend(handles + extra, labels + extra_labels, loc="upper right", fontsize=7)


def variances(covariance: Any, /) -> np.ndarray:
    """The diagonal of a covariance on a Euclidean space of a few dimensions,
    read off its matrix: one application per dimension."""
    return np.clip(np.diag(covariance.matrix(form="components")), 0.0, None)
