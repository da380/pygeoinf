"""What the part scripts share: the command line, the counter, the figures."""

from __future__ import annotations

import argparse
import os
import pathlib
import platform
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from cartopy import crs  # noqa: E402

from pygeoinf2 import LinearOperator, plotting  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
SEED = 2026


class Counted:
    """A forward operator that counts its applications and its adjoint's."""

    def __init__(self, operator: LinearOperator) -> None:
        self.forward = 0
        self.adjoint = 0
        self._inner = operator
        self.operator = LinearOperator.from_callables(
            operator.domain, operator.codomain, self._value, adjoint=self._adjoint
        )

    def _value(self, x: Any) -> Any:
        self.forward += 1
        return self._inner(x)

    def _adjoint(self, y: Any) -> Any:
        self.adjoint += 1
        return self._inner.adjoint(y)

    def reset(self) -> None:
        self.forward = self.adjoint = 0


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


def map_panels(space: Any, rows: int, columns: int, /) -> tuple[Any, np.ndarray]:
    """A grid of Robinson maps."""
    fig, axes = plotting.subplots(
        space,
        rows=rows,
        columns=columns,
        projection=crs.Robinson(),
        figsize=(6.0 * columns, 3.1 * rows),
    )
    return fig, np.atleast_1d(axes).reshape(rows, columns)


def draw_map(
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
) -> None:
    """One map with coasts and a colorbar: signed about zero, or positive."""
    signed = limit is not None
    plotting.plot(
        space,
        field,
        ax=ax,
        symmetric=signed,
        vmin=-limit if signed else vmin,
        vmax=limit if signed else vmax,
        cmap=cmap or ("RdBu" if signed else "viridis"),
        coasts=True,
        gridlines=False,
        colorbar=True,
    )
    ax.set_title(title, fontsize=10)


def save(fig: Any, path: pathlib.Path) -> None:
    """Write the figure and close it; nothing is shown on screen."""
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"figure written to {path}")
