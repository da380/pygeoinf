"""What the part scripts share: the command line, the generators, profiles
and their bands, and the figure titles.

The space's radius is in Earth radii; every figure is drawn against the
radius in km, converted here at the plot.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import os
import pathlib
import re
import typing
from typing import Any, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .reference import R_CMB, R_ICB, R_SURFACE, prem  # noqa: E402

# Where a run writes: ``results/`` in the app's directory, above this package.
RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
# The scale's fields that are not command-line options.
FIXED = frozenset({"name"})
APP = "Planet density"

# How many radii per layer a profile's band is drawn at.
_DRAWN = 120


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


def title(part: str, what: str, /) -> str:
    """A figure's title: the app, the part and what the figure shows."""
    return f"{APP}, {part}: {what}"


def save(fig: Any, path: pathlib.Path) -> None:
    """Write the figure and close it; nothing is shown on screen."""
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"figure written to {path}")


def layer_radii(space: Any, /) -> list[np.ndarray]:
    """Radii inside each layer, in the space's Earth radii, never on an
    interface, for drawing."""
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
    """A layered profile as a line per layer in one color, broken at each
    interface, against the radius in km."""
    label = kwargs.pop("label", None)
    for index, radii in enumerate(layer_radii(space)):
        values = space.layers[index].evaluate(model[index], radii)
        (line,) = ax.plot(
            R_SURFACE * radii, values, label=label if index == 0 else None, **kwargs
        )
        kwargs["color"] = line.get_color()
    ax.set_xlim(0.0, R_SURFACE)


def draw_band(
    space: Any, model: Any, std: Sequence[np.ndarray], ax: Any, /, **kwargs: Any
) -> None:
    """Two standard deviations either side of a profile, layer by layer,
    against the radius in km."""
    label = kwargs.pop("label", None)
    for index, (radii, deviation) in enumerate(zip(layer_radii(space), std)):
        values = space.layers[index].evaluate(model[index], radii)
        ax.fill_between(
            R_SURFACE * radii,
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
    ax.set_xlim(0.0, R_SURFACE)


def finish_profile_axes(ax: Any, /, *, subtitle: str = "", legend: bool = True) -> None:
    """Labels, the two boundaries marked, a legend."""
    for radius in (R_ICB, R_CMB):
        ax.axvline(radius, color="0.6", lw=0.6, ls=":")
    ax.set_xlabel("radius (km)")
    ax.set_ylabel("density (g/cm³)")
    if subtitle:
        ax.set_title(subtitle, fontsize=10)
    if legend:
        ax.legend(fontsize=7, frameon=False)
