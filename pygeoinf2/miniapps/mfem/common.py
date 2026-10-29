"""What the part scripts share: the command line, the scales, the figures, and
the sensors of the unit-square problem that parts A and C invert.

Each part writes its own figures, one method to a figure: the point
estimate with its error budget, the measure as prior against posterior, the
set as the bound the prior's ball admits alone against the bracket the data
leave.
"""

from __future__ import annotations

import argparse
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as tri  # noqa: E402
import mfem.ser as mfem  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
SEED = 2026


@dataclass(frozen=True)
class Scale:
    """One size of every part."""

    name: str
    # Parts A to C: quadrilaterals along each side of the unit square, and
    # elements of the interval in part B.
    divisions: int
    line_elements: int
    # Parts D and E: grid points along each side of the periodic box, MFEM's
    # quadrilaterals along each side of its mesh, and the meshes of part D's
    # refinement study. No mesh here lines up with the grid; see part D.
    box_grid: int
    box_elements: int
    refinements: tuple[int, ...]
    # Part E: the length scale of the box's Sobolev space and prior, the
    # screening length of the solve, the number of sensors and of the
    # posterior samples behind the standard deviation map.
    length_scale: float
    screening_length: float
    sensors: int
    std_samples: int


QUICK = Scale(
    name="quick",
    divisions=6,
    line_elements=16,
    box_grid=32,
    box_elements=24,
    refinements=(12, 24, 48),
    length_scale=0.1,
    screening_length=0.05,
    sensors=150,
    std_samples=16,
)
FULL = Scale(
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
    against the bracket the data leave, and the truth, on an axis linear
    within one and logarithmic beyond."""
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
    from pygeoinf2 import plotting

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


def ball_alone(ball: Any, properties: Any, /) -> np.ndarray:
    """The bound on each property that a ball admits with no data: its
    support function either way along the property's representer."""
    support = ball.support_function()
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
# space of this order, whose metric is far from the identity, and the prior
# of part E is the identity covariance on it.
BOX_ORDER = 2.0


def box_space(scale: Scale, /) -> Any:
    """The Sobolev space ``H^2`` on the unit periodic box, at the scale's grid."""
    from pygeoinf2.symmetric_space.torus import Sobolev

    return Sobolev(
        (scale.box_grid, scale.box_grid),
        BOX_ORDER,
        scale.length_scale,
        lengths=(1.0, 1.0),
    )


def box_prior(space: Any, /) -> Any:
    """The identity covariance on the box's space, at a pointwise standard deviation of one.

    The space's own norm is then the prior's, so the minimum-norm estimate
    and the prior's ball measure size as the Bayesian answer does.
    """
    return space.sobolev_measure(0.0, space.length_scale, pointwise_std=1.0)


# --------------------------------------------------------------------- #
#                      The unit-square problem of A and C                #
# --------------------------------------------------------------------- #

# Sixteen sensors on a grid, each reporting a Gaussian-weighted average of
# the temperature over a window of this width; the property asked about is
# the total source over the central square.
SENSOR_POSITIONS = tuple(
    (float(x), float(y))
    for x in np.linspace(0.2, 0.8, 4)
    for y in np.linspace(0.2, 0.8, 4)
)
SENSOR_WIDTH = 0.06
REGION = (0.3, 0.7)
CORRELATION = 0.12
SOURCE_STRENGTH = 4.0
NOISE = 2.0e-4


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
    centers: Sequence[Sequence[float]] = SENSOR_POSITIONS,
    width: float = SENSOR_WIDTH,
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
