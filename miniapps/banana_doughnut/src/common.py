"""What the part scripts share: the command line, where a run writes, and
the figures' helpers."""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import os
import pathlib
import re
import typing
from typing import Any, Callable, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Where a run writes: ``results/`` in the app's directory, above this package.
RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
# The scale's fields that are not command-line options: the preset's name.
FIXED = frozenset({"name"})


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


def _group(kinds: tuple[type, ...], /) -> Callable[[str], tuple[Any, ...]]:
    """A reader of one group of values of these types written with commas
    between them, ``0.05,0.2`` for a pair of floats; named for the help."""

    def read(text: str) -> tuple[Any, ...]:
        parts = text.split(",")
        if len(parts) != len(kinds):
            raise argparse.ArgumentTypeError(
                f"{text!r} is not {len(kinds)} values separated by commas"
            )
        return tuple(kind(part) for kind, part in zip(kinds, parts))

    read.__name__ = ",".join(kind.__name__ for kind in kinds)
    return read


def _reader(hint: Any, value: Any, /) -> dict[str, Any]:
    """How a field of a given annotation and preset value is read from the
    command line: a flag pair for a bool, one or more values for a tuple,
    ``none`` for an optional field's ``None``, else the type itself. A
    tuple's items are of its annotation's type, or of its preset's first
    item's where the annotation does not say; a ``Literal`` is one of its
    values; items that are tuples
    themselves, of an annotation such as ``tuple[float, float]``, are each
    one word with commas between its values."""
    if hint is bool:
        return {"action": argparse.BooleanOptionalAction}
    arguments = typing.get_args(hint)
    if typing.get_origin(hint) is typing.Literal:
        return {"choices": arguments, "metavar": "|".join(map(str, arguments))}
    if typing.get_origin(hint) is tuple and typing.get_origin(arguments[0]) is tuple:
        read = _group(typing.get_args(arguments[0]))
        return {"type": read, "nargs": "+", "metavar": read.__name__}
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
    """A preset's value as ``--help`` shows it: a tuple as its items, an
    item that is a tuple as its values with commas between them, as it is
    written, and a long one cut short."""
    if not isinstance(value, tuple):
        return str(value)
    items = [
        ",".join(map(str, item)) if isinstance(item, tuple) else str(item)
        for item in value
    ]
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
    (``prior_std`` is ``--prior-std``), typed by its annotation, its default
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


def save(fig: Any, path: pathlib.Path, /, *, title: str) -> None:
    """Title the figure, write it and close it; nothing is shown on screen."""
    fig.suptitle(title)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"figure written to {path}")


def radius_contours(
    ax: Any, x: np.ndarray, y: np.ndarray, radii: np.ndarray, /, **style: Any
) -> None:
    """The surface and the core-mantle boundary on a section: the contours
    of the radius at those two values, on the section's own grid."""
    from .planet import CORE_RADIUS, RADIUS

    style = {"colors": "0.3", "linewidths": 0.8, **style}
    ax.contour(x, y, radii, levels=[CORE_RADIUS, RADIUS], **style)


def symmetric_mesh(
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    /,
    *,
    limit: float,
    label: str,
    shading: str = "auto",
) -> Any:
    """Values on a grid, curvilinear or not, signed about zero under a
    diverging map with a colorbar of the drawing's own height, at equal
    scales on the two axes; NaN where nothing is drawn, or ``shading=
    "gouraud"`` to interpolate between the nodes of a grid with no gaps."""
    mesh = ax.pcolormesh(
        x,
        y,
        np.ma.masked_invalid(values),
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        shading=shading,
        rasterized=True,
    )
    ax.set_aspect("equal")
    # In the drawing's own coordinates, so that it follows the drawing's box
    # once the equal scales have fixed it.
    bar = ax.inset_axes((1.03, 0.0, 0.035, 1.0))
    ax.figure.colorbar(mesh, cax=bar, label=label)
    return mesh
