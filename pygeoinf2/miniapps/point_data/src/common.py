"""What the part scripts share: the command line, the counter, the figures.

Each part writes its own figures, one method to a figure: the point
estimate with its error budget, the measure as prior against posterior, the
set as the bound the prior's ellipsoid admits alone against the bracket the
data leave.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import os
import pathlib
import platform
import re
import typing
from typing import Any, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from cartopy import crs  # noqa: E402

from pygeoinf2 import LinearOperator, plotting  # noqa: E402

# The app's directory, above this package, and where a run writes.
APP = pathlib.Path(__file__).resolve().parents[1]
RESULTS = APP / "results"
# The scale's fields that are not command-line options.
FIXED = frozenset({"name"})


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


def _field_help(kind: type, /) -> dict[str, str]:
    """Each field's line under ``Attributes:`` in a dataclass's docstring."""
    found: dict[str, str] = {}
    name, inside = None, False
    for line in (inspect.getdoc(kind) or "").splitlines():
        if line.strip() == "Attributes:":
            inside = True
        elif inside and (match := re.match(r" {4}(\w+): (.*)", line)):
            name = match[1]
            found[name] = match[2].strip()
        elif inside and name and line.startswith(" " * 8):
            found[name] += " " + line.strip()
        elif inside and line.strip():
            break
    return found


# The types a tuple's items are read as, a bool before the int it is.
_ITEMS = (bool, int, float, str)


def _reader(hint: Any, value: Any, /) -> dict[str, Any]:
    """How a field of a given annotation and preset value is read from the
    command line: a flag pair for a bool, one or more values for a tuple,
    ``none`` for an optional field's ``None``, else the type itself. A
    tuple's items are of its annotation's type, or of its preset's first
    item's where the annotation does not say."""
    if hint is bool:
        return {"action": argparse.BooleanOptionalAction}
    arguments = typing.get_args(hint)
    if typing.get_origin(hint) is tuple:
        item = arguments[0] if arguments[0] in _ITEMS else type(value[0])
        item = next((kind for kind in _ITEMS if issubclass(item, kind)), str)
        return {"type": item, "nargs": "+", "metavar": item.__name__}
    if type(None) in arguments:
        (inner,) = (a for a in arguments if a is not type(None))

        def optional(text: str) -> Any:
            return None if text.lower() == "none" else inner(text)

        optional.__name__ = inner.__name__
        return {"type": optional, "metavar": f"{inner.__name__}|none"}
    return {"type": hint, "metavar": hint.__name__}


def _shown(value: Any, /) -> str:
    """A preset's value as ``--help`` shows it: a tuple as its items, a long
    one cut short."""
    if not isinstance(value, tuple):
        return str(value)
    items = [str(item) for item in value]
    if len(items) > 12:
        items = items[:4] + ["...", items[-1], f"({len(items)} values)"]
    return " ".join(items)


def parse(
    description: str,
    /,
    *,
    quick: Any,
    full: Any,
    fixed: frozenset[str] = FIXED,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """The command line every script takes, built from the scale's dataclass.

    ``--quick`` or ``--full`` (the default) picks a preset; then every field
    of the scale but those in *fixed* is an option named after it
    (``noise_std`` is ``--noise-std``), typed by its annotation, its default
    the chosen preset's value, and its help the field's line in the
    dataclass's docstring. The options given are applied to the preset by
    :func:`dataclasses.replace`. ``--outdir`` says where the figures go.

    Args:
        description: what ``--help`` prints first, a script's docstring.
        quick: the quick preset.
        full: the full preset, of the same dataclass.
        fixed: the fields that are not options.
        argv: the arguments, or ``None`` for the command line's own.

    Returns:
        The parsed arguments, with ``scale`` the preset with the options
        applied and ``outdir`` the directory, made.
    """
    choice = argparse.ArgumentParser(add_help=False)
    presets = choice.add_mutually_exclusive_group()
    presets.add_argument("--quick", action="store_true", help="the quick scale")
    presets.add_argument(
        "--full", action="store_true", help="the full scale (the default)"
    )
    preset = quick if choice.parse_known_args(argv)[0].quick else full
    parser = argparse.ArgumentParser(
        description=description,
        parents=[choice],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--outdir",
        type=pathlib.Path,
        default=None,
        help="where the figures go (default: results/, results/quick/ at the "
        "quick scale)",
    )
    group = parser.add_argument_group(
        f"the scale, each default the {preset.name} preset's"
    )
    hints = typing.get_type_hints(type(preset))
    helps = _field_help(type(preset))
    for field in dataclasses.fields(preset):
        if field.name in fixed:
            continue
        value = getattr(preset, field.name)
        action = group.add_argument(
            "--" + field.name.replace("_", "-"),
            dest=field.name,
            default=value,
            help=helps.get(field.name, field.name).rstrip(".").replace("%", "%%"),
            **_reader(hints[field.name], value),
        )
        if "%(default)" not in action.help:
            action.help += f" (default: {_shown(value)})".replace("%", "%%")
    arguments = parser.parse_args(argv)
    changes = {
        field.name: getattr(arguments, field.name)
        for field in dataclasses.fields(preset)
        if field.name not in fixed
    }
    for name, value in changes.items():
        if isinstance(getattr(preset, name), tuple):
            changes[name] = tuple(value)
    arguments.scale = dataclasses.replace(preset, **changes)
    arguments.outdir = output_directory(arguments.quick, arguments.outdir)
    return arguments


def output_directory(quick: bool, outdir: pathlib.Path | None) -> pathlib.Path:
    """``results/`` beside the scripts at the full scale, ``results/quick/`` at
    the quick one, unless a directory is given; made if it is not there."""
    if outdir is None:
        outdir = RESULTS / "quick" if quick else RESULTS
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def part_rng(seed: int, part: str) -> np.random.Generator:
    """One generator per part, so a part gives the same answer run alone or in order."""
    return np.random.default_rng([seed, ord(part)])


def map_axes(fig: Any, spec: Any, /) -> Any:
    """One Robinson map in a cell of a figure's grid."""
    return fig.add_subplot(spec, projection=crs.Robinson())


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


def error_budget_panels(
    axes: Any,
    names: Any,
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
    names: Any,
    alone: Any,
    brackets: Any,
    truth: Any,
    /,
    *,
    ylabel: str,
) -> None:
    """For each property, a panel: the bound the prior's ellipsoid admits
    alone against the bracket the data leave, and the truth.

    Both are drawn as bars on an axis linear about zero and logarithmic
    beyond one, since a bracket the data narrow can be a hundredth of the
    bound it came from.
    """
    for ax, name, (a0, a1), (b0, b1), t in zip(axes, names, alone, brackets, truth):
        ax.plot([0, 0], [a0, a1], color="0.6", lw=10, solid_capstyle="butt")
        ax.plot([1, 1], [b0, b1], color="C0", lw=10, solid_capstyle="butt")
        ax.axhline(t, color="black", ls="--", lw=1.0, label="truth")
        ax.set_xticks([0, 1], ["the prior's\nellipsoid alone", "with the\ndata"])
        ax.set_xlim(-0.7, 1.7)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_title(name, fontsize=10)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(fontsize=7, loc="lower left", frameon=False)


def save(fig: Any, path: pathlib.Path, /, *, title: str) -> None:
    """Title the figure, write it and close it; nothing is shown on screen."""
    fig.suptitle(title)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"figure written to {path}")
