"""Part B: the matrix form, a finite element space as a coordinate space.

In a finite element space the inner product is not the dot product of the
degree-of-freedom values: it is ``(u, v) == u^T M v`` with ``M`` the mass
matrix, so the mass matrix is the space's Gram matrix. Three things a finite
element code writes out by hand then come out of the library's general
machinery: a linear form is a derivative, whose gradient is a mass solve; a
bilinear form is the Galerkin matrix of its operator, whose adjoint carries
the metric; and a Krylov solve on that operator works in the right inner
product without being told. On quadratic elements over an interval, with
``mfem.Vector`` objects as the vectors.
"""

from __future__ import annotations

import pathlib
import time

import matplotlib.pyplot as plt
import mfem.ser as mfem
import numpy as np

from pygeoinf2 import CGSolver, Traits
from pygeoinf2.testing import check_coordinates, check_operator, check_space

from common import (
    FULL,
    QUICK,
    Scale,
    dof_coordinates,
    output_directory,
    parse,
    part_rng,
    save,
)
from matrix_space import (
    MfemSpace,
    functional_from_linear_form,
    operator_from_bilinear_form,
)


class Window(mfem.PyCoefficient):
    """A Gaussian window on the interval, centered at 0.3."""

    def EvalValue(self, x: object) -> float:
        return float(np.exp(-0.5 * ((x[0] - 0.3) / 0.08) ** 2))


def matrix_form(
    scale: Scale, /, *, rng: np.random.Generator, outdir: pathlib.Path
) -> dict:
    """The space, its checks, a linear and a bilinear form, and a solve."""
    started = time.perf_counter()
    print(f"\n=== B. The matrix form ({scale.name} scale) ===")
    mesh = mfem.Mesh.MakeCartesian1D(scale.line_elements, 1.0)
    elements = mfem.FiniteElementSpace(mesh, mfem.H1_FECollection(2, mesh.Dimension()))
    V = MfemSpace(elements)
    print(
        f"{V.dim} degrees of freedom, quadratic elements on [0, 1]; vectors are "
        f"{type(V.zero()).__name__} objects, not arrays; an orthonormal basis: "
        f"{V.is_orthonormal}, because the mass matrix is the metric"
    )
    space_report = check_space(V, rng=rng)
    coordinate_report = check_coordinates(V, rng=rng)
    print(f"  {space_report}\n  {coordinate_report}")

    # A linear form is a derivative: its load vector holds l(phi_i), and its
    # representer, the gradient, is the mass solve of that vector.
    window = Window()
    load = mfem.LinearForm(elements)
    load.AddDomainIntegrator(mfem.DomainLFIntegrator(window))
    load.Assemble()
    entries = np.array(load.GetDataArray(), copy=True)
    functional = functional_from_linear_form(V, load)
    u = V.random(rng=rng)
    action_defect = abs(functional(u) - float(entries @ V.to_components(u)))
    representer = V.to_components(functional.representer)
    mass_solve_defect = float(np.max(np.abs(representer - V.solve_gram(entries))))
    print(
        "the load of l(u) = integral of g u, g a window: l(u) == b . u to "
        f"{action_defect:.1e}; its representer is M^-1 b to {mass_solve_defect:.1e}, "
        f"and peaks at {representer.max():.2f} where b peaks at {entries.max():.3f}"
    )

    # A bilinear form is the Galerkin matrix of its operator: K == M A_c.
    stiffness = mfem.BilinearForm(elements)
    stiffness.AddDomainIntegrator(mfem.DiffusionIntegrator())
    stiffness.AddDomainIntegrator(mfem.MassIntegrator())
    stiffness.Assemble()
    stiffness.Finalize()
    A = operator_from_bilinear_form(V, stiffness, traits=Traits.POSITIVE_DEFINITE)
    operator_report = check_operator(A, rng=rng)
    print(
        "the operator of the form -u'' + u, from its Galerkin matrix: "
        f"{operator_report}; the adjoint identity holds in the mass metric"
    )

    # Conjugate gradients on that operator, against a direct solve of the
    # Galerkin system: A x == M^-1 K x, so x == K^-1 M b.
    right_hand_side = functional.representer
    result = CGSolver(rtol=1e-12)(A).solve(right_hand_side)
    direct = np.linalg.solve(
        A.matrix(form="galerkin"), V.gram_matrix() @ V.to_components(right_hand_side)
    )
    agreement = float(np.max(np.abs(V.to_components(result.solution) - direct)))
    print(
        f"conjugate gradients on A x = representer: {result.iterations} iterations, "
        f"agreeing with a direct solve to {agreement:.1e}; every inner product in "
        f"it was u^T M v"
    )

    x = dof_coordinates(elements)[:, 0]
    order = np.argsort(x)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4), layout="constrained")
    axes[0].plot(x[order], entries[order], ".-", label="load vector b: the derivative")
    axes[0].plot(
        x[order], representer[order], ".-", label="representer M^-1 b: the gradient"
    )
    axes[0].set_xlabel("x")
    axes[0].set_title("The functional u -> integral of g u", fontsize=10)
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].plot(
        x[order],
        V.to_components(result.solution)[order],
        "-",
        lw=3,
        alpha=0.5,
        label=f"conjugate gradients, {result.iterations} iterations",
    )
    axes[1].plot(x[order], direct[order], "k--", lw=1, label="direct solve")
    axes[1].set_xlabel("x")
    axes[1].set_title("-u'' + u = g with natural ends, solved two ways", fontsize=10)
    axes[1].legend(fontsize=8, frameon=False)
    save(fig, outdir / "matrix_form.png")
    print(f"B took {time.perf_counter() - started:.1f} s")
    return {
        "dim": V.dim,
        "orthonormal": V.is_orthonormal,
        "space": space_report,
        "coordinates": coordinate_report,
        "action_defect": action_defect,
        "mass_solve_defect": mass_solve_defect,
        "representer_differs": float(np.max(np.abs(representer - entries))),
        "operator": operator_report,
        "iterations": result.iterations,
        "agreement": agreement,
    }


def run(scale: Scale, /, *, seed: int, outdir: pathlib.Path) -> dict:
    return matrix_form(scale, rng=part_rng(seed, "b"), outdir=outdir)


def main() -> None:
    args = parse(__doc__)
    run(
        QUICK if args.quick else FULL,
        seed=args.seed,
        outdir=output_directory(args.quick, args.outdir),
    )


if __name__ == "__main__":
    main()
