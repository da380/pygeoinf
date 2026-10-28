"""What the part scripts share: the command line, profiles and their bands,
and the property panels."""

from __future__ import annotations

import argparse
import os
import pathlib
from typing import Any, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pygeoinf2 import plotting  # noqa: E402

from reference import BOUNDARIES, R_CMB, R_ICB, prem  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
SEED = 2026

# How many radii per layer a profile and its band are drawn at.
_DRAWN = 120


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


def layer_radii(space: Any, /) -> list[np.ndarray]:
    """Radii inside each layer, never on an interface, for drawing."""
    radii = []
    for lower, upper in space.layer_bounds:
        margin = 1e-6 * (upper - lower)
        radii.append(np.linspace(lower + margin, upper - margin, _DRAWN))
    return radii


def pointwise_std(space: Any, measure: Any, /) -> list[np.ndarray]:
    """A measure's exact pointwise standard deviation at :func:`layer_radii`,
    layer by layer: the covariance pushed through point evaluation."""
    radii = layer_radii(space)
    evaluation = space.point_evaluation_operator(list(np.concatenate(radii)))
    variance = measure.push_forward(evaluation).covariance.matrix(form="components")
    std = np.sqrt(np.clip(np.diag(variance), 0.0, None))
    return np.split(std, np.cumsum([r.size for r in radii])[:-1])


def draw_profile(space: Any, model: Any, ax: Any, /, **kwargs: Any) -> None:
    """A layered profile as a line per layer, broken at each interface."""
    plotting.plot(space, model, ax=ax, **kwargs)


def draw_band(
    space: Any, model: Any, std: Sequence[np.ndarray], ax: Any, /, **kwargs: Any
) -> None:
    """Two standard deviations either side of a profile, layer by layer."""
    label = kwargs.pop("label", None)
    for index, (radii, deviation) in enumerate(zip(layer_radii(space), std)):
        values = space.layers[index].evaluate(model[index], radii)
        ax.fill_between(
            radii,
            values - 2.0 * deviation,
            values + 2.0 * deviation,
            linewidth=0.0,
            label=label if index == 0 else None,
            **kwargs,
        )


def draw_prem(ax: Any, /, **kwargs: Any) -> None:
    """PREM's own density, with all its discontinuities."""
    for layer in prem().layers:
        lower, upper = layer.interval
        r = np.linspace(lower, upper, 60)
        ax.plot(r / 1e3, layer.fields["rho"](r) / 1e3, **kwargs)
        kwargs = {**kwargs, "label": None}
    ax.set_xlim(BOUNDARIES[0], BOUNDARIES[-1])


def finish_profile_axes(ax: Any, /, *, title: str) -> None:
    """Labels, the two boundaries marked, a legend."""
    for radius in (R_ICB, R_CMB):
        ax.axvline(radius, color="0.6", lw=0.6, ls=":")
    ax.set_xlabel("radius (km)")
    ax.set_ylabel("density (g/cm³)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, frameon=False)


def interval_bar(
    ax: Any,
    position: float,
    center: float,
    half_width: float,
    /,
    *,
    marker: str,
    color: str,
    label: str,
) -> None:
    """A point with a symmetric bar."""
    ax.errorbar(
        position,
        center,
        yerr=half_width,
        fmt=marker,
        color=color,
        capsize=4,
        label=label,
    )


def bracket_bar(
    ax: Any, position: float, bracket: Sequence[float], /, *, color: str, label: str
) -> None:
    """A certified bracket, as a thick bar."""
    ax.plot(
        [position, position],
        bracket,
        color=color,
        lw=6,
        alpha=0.6,
        solid_capstyle="butt",
        label=label,
    )
