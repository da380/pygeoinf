"""What the part scripts share: the command line, the scales, the figures, and
the sensors of the unit-square problem that parts A and C invert.

Each part writes its own figures, one method to a figure: the point
estimate with its error budget, the measure as prior against posterior, the
set as the bound the prior's ball admits alone against the bracket the
data leave.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import os
import pathlib
import re
import typing
from dataclasses import dataclass
from typing import Any, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as tri  # noqa: E402
import mfem.ser as mfem  # noqa: E402
import numpy as np  # noqa: E402

# Where a run writes: ``results/`` in the app's directory, above this package.
RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
# The scale's fields that are not command-line options.
FIXED = frozenset({"name"})


@dataclass(frozen=True)
class Scale:
    """One size of every part, and its physics: every field but the name is
    a command-line option of every script.

    No mesh of parts D and E lines up with the box's grid; see part D.

    Attributes:
        name: the preset's name, for the printed headings.
        seed: the seed of every part's draws.
        divisions: parts A to C's quadrilaterals along each side of the unit
            square.
        line_elements: part B's elements of the interval.
        conductivity: the conductivity of part C's diffusion equation.
        orders: the polynomial orders of part C's study of the same data.
        source_strength: the amplitude of parts A and C's Matern prior on
            the source.
        smoothness: the smoothness of that Matern prior.
        correlation_length: the correlation length of that Matern prior.
        sensor_width: the width of each of parts A and C's sixteen sensors'
            Gaussian windows.
        noise_std: the standard deviation of parts A and C's noise.
        box_grid: parts D and E's grid points along each side of the
            periodic box.
        box_elements: MFEM's quadrilaterals along each side of the box's
            mesh.
        refinements: the meshes of part D's refinement study.
        box_order: the Sobolev order of the box's space, above one for a
            value at a point to exist.
        length_scale: the length scale of the box's space and prior.
        box_prior_order: the Sobolev order of part E's prior on the box's
            space, above one, half the box's dimension, for its covariance
            to be trace class there and its draws functions of the space.
        box_prior_std: the pointwise standard deviation of part E's prior.
        screening_length: the screening length of the box's solve.
        sensors: part E's sensors, scattered outside the gap.
        box_noise_std: the standard deviation of part E's noise.
        gap_radius: the radius of the disc part E's sensors avoid.
        region_radius: the radius of the discs part E averages over.
        level: the credibility of part E's discrepancy level and of the sets
            its prior and noise are hardened to.
        std_samples: the posterior samples behind part E's standard
            deviation map.
    """

    name: str
    seed: int
    divisions: int
    line_elements: int
    conductivity: float
    orders: tuple[int, ...]
    source_strength: float
    smoothness: float
    correlation_length: float
    sensor_width: float
    noise_std: float
    box_grid: int
    box_elements: int
    refinements: tuple[int, ...]
    box_order: float
    length_scale: float
    box_prior_order: float
    box_prior_std: float
    screening_length: float
    sensors: int
    box_noise_std: float
    gap_radius: float
    region_radius: float
    level: float
    std_samples: int


QUICK = Scale(
    name="quick",
    seed=2026,
    divisions=6,
    line_elements=16,
    conductivity=1.0,
    orders=(1, 2, 3),
    source_strength=4.0,
    smoothness=1.0,
    correlation_length=0.12,
    sensor_width=0.06,
    noise_std=2.0e-4,
    box_grid=32,
    box_elements=24,
    refinements=(12, 24, 48),
    box_order=2.0,
    length_scale=0.1,
    box_prior_order=2.0,
    box_prior_std=1.0,
    screening_length=0.05,
    sensors=150,
    box_noise_std=0.05,
    gap_radius=0.18,
    region_radius=0.08,
    level=0.95,
    std_samples=16,
)
FULL = dataclasses.replace(
    QUICK,
    name="full",
    divisions=10,
    line_elements=32,
    box_grid=64,
    box_elements=48,
    refinements=(24, 48, 96),
    length_scale=0.05,
    screening_length=0.025,
    sensors=600,
    std_samples=48,
)


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


def save(fig: Any, path: pathlib.Path, /, *, title: str) -> None:
    """Title the figure, write it and close it; nothing is shown on screen."""
    fig.suptitle(title)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"figure written to {path}")


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
    """For each property, a panel: the bound the prior's ball admits
    alone against the bracket the data leave, and the truth, on an axis
    linear within one and logarithmic beyond."""
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
    from pygeoinf import plotting

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


def prior_alone(prior_set: Any, properties: Any, /) -> np.ndarray:
    """The bound on each property that the prior's set admits with no data:
    its support function either way along the property's representer."""
    support = prior_set.support_function()
    P = properties.codomain
    bounds = np.zeros((P.dim, 2))
    for index in range(P.dim):
        q = P.basis_vector(index)
        bounds[index] = (
            -support(properties.adjoint(P.scale(-1.0, q))),
            support(properties.adjoint(q)),
        )
    return bounds


def values(vector: Any) -> np.ndarray:
    """An ``mfem.Vector``'s entries, copied out of MFEM's memory."""
    return np.array(vector.GetDataArray(), copy=True)


# --------------------------------------------------------------------- #
#                          Finite element figures                        #
# --------------------------------------------------------------------- #


class _Coordinate(mfem.PyCoefficient):
    """One coordinate of the point, as an MFEM coefficient."""

    def __init__(self, axis: int) -> None:
        super().__init__()
        self._axis = axis

    def EvalValue(self, x: Any) -> float:
        return float(x[self._axis])


def dof_coordinates(elements: Any, /) -> np.ndarray:
    """Where each degree of freedom of a nodal ``H1`` space sits, one row each.

    The coordinate functions projected onto the space: for a Lagrange basis
    the projection interpolates, so each value is its node's coordinate.
    """
    dimension = elements.GetMesh().Dimension()
    columns = []
    for axis in range(dimension):
        function = mfem.GridFunction(elements)
        coefficient = _Coordinate(axis)
        function.ProjectCoefficient(coefficient)
        columns.append(values(function))
    return np.stack(columns, axis=1)


def draw_field(
    ax: Any,
    coordinates: np.ndarray,
    field: np.ndarray,
    /,
    *,
    title: str,
    limit: float | None = None,
    cmap: str | None = None,
    label: str | None = None,
) -> None:
    """A field on the unit square from its nodal values, with a colorbar."""
    mesh = tri.Triangulation(coordinates[:, 0], coordinates[:, 1])
    signed = limit is not None
    shading = ax.tripcolor(
        mesh,
        field,
        shading="gouraud",
        cmap=cmap or ("RdBu_r" if signed else "viridis"),
        vmin=-limit if signed else None,
        vmax=limit if signed else None,
    )
    bar = ax.figure.colorbar(shading, ax=ax, shrink=0.85)
    if label:
        bar.set_label(label)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)


# --------------------------------------------------------------------- #
#                       The periodic box of D and E                      #
# --------------------------------------------------------------------- #

# The box is the unit square with periodic sides; its space is the Sobolev
# space of the scale's order, ``H^2``, whose metric is far from the
# identity, and the prior of part E is a Sobolev measure of order two on it.


def box_space(scale: Scale, /) -> Any:
    """The Sobolev space ``H^2`` on the unit periodic box, at the scale's grid."""
    from pygeoinf.symmetric_space.torus import Sobolev

    return Sobolev(
        (scale.box_grid, scale.box_grid),
        scale.box_order,
        scale.length_scale,
        lengths=(1.0, 1.0),
    )


def box_prior(space: Any, scale: Scale, /) -> Any:
    """A Sobolev measure of order two on the box's ``H^2``, at the scale's
    pointwise standard deviation, one.

    A measure of extra regularity on the space: its covariance is trace
    class there, the order exceeding half the box's dimension, and its
    draws have nearly one derivative more than the space asks.
    """
    return space.sobolev_measure(
        scale.box_prior_order,
        space.length_scale,
        pointwise_std=scale.box_prior_std,
    )


# --------------------------------------------------------------------- #
#                      The unit-square problem of A and C                #
# --------------------------------------------------------------------- #

# Sixteen sensors on a grid, each reporting a Gaussian-weighted average of
# the temperature over a window of the scale's width; the property asked
# about is the total source over the central square.
SENSOR_POSITIONS = tuple(
    (float(x), float(y))
    for x in np.linspace(0.2, 0.8, 4)
    for y in np.linspace(0.2, 0.8, 4)
)
REGION = (0.3, 0.7)


class Bump(mfem.PyCoefficient):
    """A normalized Gaussian window, standing for a sensor's footprint."""

    def __init__(self, center: Sequence[float], width: float) -> None:
        super().__init__()
        self._center = np.asarray(center, dtype=float)
        self._width = float(width)

    def EvalValue(self, x: Any) -> float:
        offset = np.asarray([x[0], x[1]]) - self._center
        return float(
            np.exp(-0.5 * offset @ offset / self._width**2)
            / (2.0 * np.pi * self._width**2)
        )


class Region(mfem.PyCoefficient):
    """One inside the central square, zero outside."""

    def EvalValue(self, x: Any) -> float:
        low, high = REGION
        return 1.0 if low <= x[0] <= high and low <= x[1] <= high else 0.0


def sensor_forms(
    elements: Any,
    /,
    *,
    width: float,
    centers: Sequence[Sequence[float]] = SENSOR_POSITIONS,
) -> list[Any]:
    """One assembled ``mfem.LinearForm`` per sensor: its footprint integrated
    against each basis function, which is what the sensor reports."""
    forms = []
    for center in centers:
        form = mfem.LinearForm(elements)
        form.AddDomainIntegrator(mfem.DomainLFIntegrator(Bump(center, width)))
        form.Assemble()
        forms.append(form)
    return forms


def region_form(elements: Any, /) -> Any:
    """The assembled ``mfem.LinearForm`` of the total source over the central square."""
    form = mfem.LinearForm(elements)
    form.AddDomainIntegrator(mfem.DomainLFIntegrator(Region()))
    form.Assemble()
    return form


def unit_square(order: int, divisions: int, /) -> Any:
    """An ``H1`` space of the given order on a square mesh of the unit square."""
    # PyMFEM's element space keeps references to the mesh and the collection
    # it was built on, so neither is collected while the space is in use.
    mesh = mfem.Mesh.MakeCartesian2D(divisions, divisions, mfem.Element.QUADRILATERAL)
    return mfem.FiniteElementSpace(mesh, mfem.H1_FECollection(order, mesh.Dimension()))
