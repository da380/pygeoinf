"""What the part scripts share: the command line, the timer, the figures."""

from __future__ import annotations

import argparse
import os
import pathlib
import platform
import time
from collections import Counter
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.layout_engine import ConstrainedLayoutEngine  # noqa: E402
import numpy as np  # noqa: E402

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


class Stopwatch:
    """Wall time and kernel calls since the last ``lap``."""

    def __init__(self, calls: Counter) -> None:
        self._calls = calls
        self._start = time.perf_counter()
        self._seen = Counter(calls)

    def lap(self) -> tuple[float, Counter]:
        now = time.perf_counter()
        spent = now - self._start
        used = Counter(self._calls)
        used.subtract(self._seen)
        self._start, self._seen = now, Counter(self._calls)
        return spent, +used


def describe(calls: Counter) -> str:
    """Kernel runs as ``forward 4, born 12, ...``, each one shot."""
    order = ("forward", "born", "adjoint", "second_born", "curvature")
    return ", ".join(f"{name} {calls[name]}" for name in order if calls[name])


def section(
    ax: Any, survey: Any, values: np.ndarray, /, *, whole: bool = False, **options: Any
) -> Any:
    """A field on the section, depth down, x and z in km; on the whole grid,
    sponge and all, with ``whole``."""
    values = np.asarray(values)
    x, z = survey.x, survey.z
    if not whole:
        values = values[survey.window]
        x, z = x[survey.window[0]], z[survey.window[1]]
    h = survey.scale.h
    image = ax.imshow(
        values.T,
        extent=(x[0] - h / 2, x[-1] + h / 2, z[-1] + h / 2, z[0] - h / 2),
        aspect="equal",
        interpolation="nearest",
        **options,
    )
    ax.set_xlabel("x (km)")
    ax.set_ylabel("depth (km)")
    return image


def sponge_outline(ax: Any, survey: Any, /, *, color: str = "0.45") -> None:
    """On the whole grid: the section outlined, and the sponge beyond it hatched."""
    from matplotlib.patches import Rectangle

    h = survey.scale.h
    width, depth = survey.section_lengths
    left, right = survey.x[0] - h / 2, survey.x[-1] + h / 2
    bottom = survey.z[-1] + h / 2
    for x0, x1, z0 in ((left, -h / 2, -h / 2), (width - h / 2, right, -h / 2)):
        ax.add_patch(
            Rectangle(
                (x0, z0),
                x1 - x0,
                bottom - z0,
                fill=False,
                hatch="//",
                lw=0,
                color=color,
            )
        )
    ax.add_patch(
        Rectangle(
            (-h / 2, depth - h / 2),
            width,
            bottom - depth + h / 2,
            fill=False,
            hatch="//",
            lw=0,
            color=color,
        )
    )
    ax.plot(
        [-h / 2, -h / 2, width - h / 2, width - h / 2],
        [-h / 2, depth - h / 2, depth - h / 2, -h / 2],
        color=color,
        lw=1.0,
    )
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, -h / 2)


def save(fig: Any, path: pathlib.Path, /, *, title: str) -> None:
    """Title the figure, lay it out, write it and close it; nothing is shown
    on screen."""
    fig.suptitle(title)
    if not isinstance(fig.get_layout_engine(), ConstrainedLayoutEngine):
        fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"figure written to {path}")
