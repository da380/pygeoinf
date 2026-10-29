"""Part D: an operator through MFEM on the library's own space, and its check ladder.

The second way of wrapping MFEM. The space is the library's periodic box, a
Sobolev space whose metric is far from the identity, and MFEM appears only
inside an operator on it: the screened Poisson solve ``(1 - s^2 Laplacian)
u = f``, whose action takes the field to MFEM's quadrature points, solves
there, and reads the finite element solution back at the box's grid. The
adjoint is written from the pieces: the transpose of each transfer with the
box's Gram factors where they belong, and MFEM's own solve again for the
symmetric system (see ``box_operator``).

The ladder runs the library's operator check on each piece and on the
composition, with the prior of part E as the source of probe fields. Then
three things it can tell apart: the adjoint with the Gram factors left out,
which it refuses; the adjoint the continuum promises, the solve being
self-adjoint before discretization, which holds only to the discretization
error and only falls under mesh refinement; and the true adjoint at a range
of MFEM tolerances, whose defect follows the tolerance down.
"""

from __future__ import annotations

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf2 import LinearOperator, plotting
from pygeoinf2.testing import check_operator, check_traits

from src.box_operator import ScreenedPoisson
from src.common import (
    FULL,
    QUICK,
    Scale,
    box_prior,
    box_space,
    parse,
    part_rng,
    save,
)

ADJOINT = "the adjoint satisfies (A x, y) == (x, A* y)"
TOLERANCES = (1e-6, 1e-9, 1e-12)


def plain_transpose(box: ScreenedPoisson, /) -> LinearOperator:
    """The operator with the transpose of its component matrices as its adjoint.

    What the matrices give when the metric is forgotten: ``B^T Q^T K^-1
    P^T T^T`` applied to the codomain's components, with neither the Gram
    factor on the right nor its inverse on the left.
    """
    space = box.space

    def adjoint(y: np.ndarray) -> np.ndarray:
        load = box.evaluation_matrix.T @ (box.cell_volume * np.asarray(y).ravel())
        solution = box.solve(load)
        return space.from_components(
            space.accumulate(box.load_matrix.T @ solution, box.quadrature_points)
        )

    return LinearOperator.from_callables(space, space, box.operator, adjoint=adjoint)


def as_its_own_adjoint(operator: LinearOperator, /) -> LinearOperator:
    """The operator paired with itself as its adjoint, as the continuum would have it."""
    return LinearOperator.from_callables(
        operator.domain, operator.codomain, operator, adjoint=operator
    )


def measured(operator: LinearOperator, /, **options: object) -> float:
    """The adjoint defect the check measures, without a tolerance to fail."""
    return check_operator(operator, rtol=np.inf, **options).defects[ADJOINT]


def box_ladder(
    scale: Scale, /, *, rng: np.random.Generator, outdir: pathlib.Path
) -> dict:
    """The check ladder on the pieces and the whole, and what it tells apart."""
    started = time.perf_counter()
    print(
        f"\n=== D. An operator through MFEM, and its check ladder ({scale.name} scale) ==="
    )
    X = box_space(scale)
    prior = box_prior(X, scale)
    t0 = time.perf_counter()
    box = ScreenedPoisson(
        X, elements=scale.box_elements, screening_length=scale.screening_length
    )
    print(
        f"the box: H^{X.order:g} on the unit periodic square, a {X.shape[0]} x "
        f"{X.shape[1]} grid, length scale {X.length_scale}, Gram weights from 1 to "
        f"{X.apply_gram(np.ones(X.dim)).max():.1e}"
    )
    print(
        f"MFEM: a periodic mesh of {scale.box_elements} x {scale.box_elements} "
        f"quadratic elements, {box.size} degrees of freedom, "
        f"{len(box.quadrature_points)} quadrature points; screening length "
        f"{box.screening_length}; built in {time.perf_counter() - t0:.2f} s"
    )

    box.reset_counts()
    # The solve's traits are checked as a claim of symmetry alone. It is
    # positive definite too, but that claim implies invertibility, which
    # check_traits verifies on a coordinate space by forming the whole
    # matrix: one MFEM solve per degree of freedom.
    symmetric = LinearOperator.self_adjoint(box.coefficients, box.solve)
    ladder = {
        "sampling": check_operator(box.sampling, rng=rng, measure=prior),
        "solve": check_operator(box.solve, rng=rng),
        "solve traits": check_traits(symmetric, rng=rng),
        "evaluation": check_operator(box.evaluation, rng=rng),
        "operator": check_operator(box.operator, rng=rng, measure=prior),
    }
    print(
        f"the ladder, at MFEM's tolerance of {box.rtol:g} ({box.solves} solves, "
        f"{box.iterations / max(box.solves, 1):.0f} iterations each):"
    )
    for name, report in ladder.items():
        detail = ", ".join(
            f"{key} {value:.1e}" for key, value in report.defects.items()
        )
        print(f"  {name:<13s} passes: {detail}")

    try:
        check_operator(plain_transpose(box), rng=rng, measure=prior)
        refusal = ""
    except AssertionError as failure:
        refusal = str(failure)
    plain_defect = measured(plain_transpose(box), rng=rng, measure=prior)
    print(
        f"the plain transpose, Gram factors left out: refused, defect "
        f"{plain_defect:.1e} ({refusal[:60]}...)"
    )

    print(
        "the operator against the continuum it discretizes, the Fourier multiplier "
        "1 / (1 + s^2 |k|^2), as the mesh is refined:"
    )
    f = prior.sample(rng=rng)
    exact = box.continuum(f)
    refinement = []
    meshes = [(n, False) for n in scale.refinements] + [(scale.box_grid, True)]
    for elements, aligned in meshes:
        refined = ScreenedPoisson(
            X, elements=elements, screening_length=scale.screening_length
        )
        operator = refined.operator
        image = operator(f)
        difference = X.subtract(image, exact)
        row = {
            "elements": elements,
            "aligned": aligned,
            "dofs": refined.size,
            "l2": float(np.linalg.norm(difference) / np.linalg.norm(exact)),
            "norm": X.norm(difference) / X.norm(exact),
            "adjoint": measured(operator, rng=rng, measure=prior),
            "continuum adjoint": measured(
                as_its_own_adjoint(operator), rng=rng, measure=prior
            ),
        }
        refinement.append(row)
        label = " (aligned with the grid)" if aligned else ""
        print(
            f"  {elements:3d} elements a side, {refined.size:6d} dofs{label}: "
            f"relative difference {row['l2']:.1e} on the grid, {row['norm']:.1e} in "
            f"H^{X.order:g}; adjoint defect {row['adjoint']:.1e}, and {row['continuum adjoint']:.1e} "
            f"with the operator as its own adjoint"
        )

    print("the true adjoint at MFEM tolerances from loose to tight:")
    tolerances = []
    for tolerance in (*TOLERANCES, box.rtol):
        box.reset_counts()
        defect = measured(box.operator_at(tolerance), rng=rng, measure=prior)
        tolerances.append((tolerance, defect, box.iterations / max(box.solves, 1)))
        print(
            f"  rtol {tolerance:7.0e}: adjoint defect {defect:.1e}, "
            f"{tolerances[-1][2]:.0f} iterations a solve"
        )

    fig = plt.figure(figsize=(12.0, 7.2), layout="constrained")
    grid = fig.add_gridspec(2, 6)
    maps = [fig.add_subplot(grid[0, 2 * i : 2 * i + 2]) for i in range(3)]
    image = box.operator(f)
    limit = float(np.abs(np.asarray(image)).max())
    for ax, field, title, bound in (
        (maps[0], f, "a source f drawn from the prior", float(np.abs(f).max())),
        (maps[1], image, "u = A f, solved by MFEM", limit),
        (maps[2], X.subtract(image, exact), "A f minus the exact multiplier", None),
    ):
        values = np.asarray(field)
        plotting.plot(
            X,
            values,
            ax=ax,
            cmap="RdBu_r",
            symmetric=True,
            vmin=-(bound or float(np.abs(values).max())),
            vmax=bound or float(np.abs(values).max()),
        )
        ax.set_title(title, fontsize=10)
    left = fig.add_subplot(grid[1, :3])
    right = fig.add_subplot(grid[1, 3:])
    tolerance_values = np.array([t for t, _, _ in tolerances])
    left.loglog(
        tolerance_values, [d for _, d, _ in tolerances], "o-", label="adjoint defect"
    )
    left.loglog(tolerance_values, tolerance_values, "k:", lw=0.8, label="the tolerance")
    left.set_xlabel("MFEM's relative tolerance")
    left.set_title("The true adjoint follows the solve's tolerance", fontsize=10)
    left.legend(fontsize=8, frameon=False)
    misaligned = [row for row in refinement if not row["aligned"]]
    elements = np.array([row["elements"] for row in misaligned])
    right.loglog(
        elements,
        [row["l2"] for row in misaligned],
        "o-",
        label="A f against the multiplier",
    )
    right.loglog(
        elements,
        [row["continuum adjoint"] for row in misaligned],
        "s-",
        label="A as its own adjoint: defect",
    )
    right.loglog(
        elements,
        [row["adjoint"] for row in misaligned],
        "^-",
        label="the true adjoint: defect",
    )
    right.set_xticks(elements, [str(n) for n in elements])
    right.minorticks_off()
    right.set_xlabel("elements along a side")
    right.set_title(
        "Refined, the discretization error falls; the adjoint always holds", fontsize=10
    )
    right.legend(fontsize=8, frameon=False)
    save(
        fig,
        outdir / "box_ladder.png",
        title="MFEM, part D: the box operator and its check ladder",
    )
    print(f"D took {time.perf_counter() - started:.1f} s")
    return {
        "dofs": box.size,
        "ladder": ladder,
        "plain_defect": plain_defect,
        "refusal": refusal,
        "refinement": refinement,
        "tolerances": tolerances,
    }


def run(scale: Scale, /, *, outdir: pathlib.Path) -> dict:
    return box_ladder(scale, rng=part_rng(scale.seed, "d"), outdir=outdir)


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
