"""What the part scripts share: the command line, the timer, the figures."""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import os
import pathlib
import platform
import re
import time
import typing
from collections import Counter
from typing import Any, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.layout_engine import ConstrainedLayoutEngine  # noqa: E402
import numpy as np  # noqa: E402

# Where a run writes: ``results/`` in the app's directory, above this package.
RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
# The scale's fields that are not command-line options.
FIXED = frozenset({"name"})


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
