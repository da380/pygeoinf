"""Part A: the Hilbert-space form, where the library conducts and MFEM computes.

The finite element space is a ``HilbertSpace`` and nothing more: no
component map, no Gram matrix, no CSR array ever read. The inner product is
MFEM's mass operator applied by MFEM, the mass solve is MFEM's conjugate
gradients, forms become operators through ``FormSystemMatrix`` (MFEM's own
way of imposing boundary conditions, as happy with a matrix-free partially
assembled operator as with a sparse matrix), and white noise is MFEM's own
integrator. The library supplies the composition, the adjoints, the prior
and the inversion, and never sees a matrix.

The problem is the one part C solves through the matrix form: the heat
source on the unit square behind sixteen local temperature readings, with
the source zero on the boundary. Here it is solved on partially assembled
operators, and the answer is checked against the matrix form's on the same
data. Nothing in this arrangement asks whether a vector is local or
distributed, which is what would let the same code run over MFEM's parallel
spaces; that path is not run here.
"""

from __future__ import annotations

import pathlib
import time

import matplotlib.pyplot as plt
import mfem.ser as mfem
import numpy as np

from pygeoinf2 import (
    CoordinateSpace,
    EuclideanSpace,
    GaussianMeasure,
    LinearForwardProblem,
    LinearGaussianInversion,
    LinearOperator,
    Traits,
)
from pygeoinf2.testing import check_operator, check_space, check_traits

import matrix_space as coordinate
from common import (
    CORRELATION,
    FULL,
    NOISE,
    QUICK,
    SENSOR_POSITIONS,
    SOURCE_STRENGTH,
    Scale,
    dof_coordinates,
    draw_field,
    output_directory,
    parse,
    part_rng,
    region_form,
    save,
    sensor_forms,
    unit_square,
    values,
)
from hilbert_space import (
    MfemHilbertSpace,
    matern_measure,
    operator_from_bilinear_form,
    operator_from_linear_forms,
    solver_from_bilinear_form,
)

DEFINITE = Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE


def partially_assembled(elements: object, *integrators: object) -> object:
    """A partially assembled form: MFEM keeps its action, never its matrix."""
    form = mfem.BilinearForm(elements)
    form.SetAssemblyLevel(mfem.AssemblyLevel_PARTIAL)
    for integrator in integrators:
        form.AddDomainIntegrator(integrator)
    form.Assemble()
    return form


def hilbert_form(
    scale: Scale, /, *, rng: np.random.Generator, outdir: pathlib.Path
) -> dict:
    """The space, its operators and their checks, and the inversion on it."""
    started = time.perf_counter()
    print(f"\n=== A. The Hilbert-space form ({scale.name} scale) ===")
    elements = unit_square(2, scale.divisions)
    essential = coordinate.essential_dofs_of(elements)
    V = MfemHilbertSpace(elements, essential_dofs=essential, assembly="partial")
    print(
        f"{V}: quadratic elements on a {scale.divisions} x {scale.divisions} mesh of "
        f"the unit square, zero on the boundary; a CoordinateSpace: "
        f"{isinstance(V, CoordinateSpace)}"
    )
    space_report = check_space(V, rng=rng)
    print(f"  {space_report}: every inner product was an MFEM Mult")
    try:
        LinearOperator.identity(V).matrix()
        refused = ""
    except TypeError as refusal:
        refused = str(refusal).split(".")[0]
    print(f"  asked for a matrix: {refused}")

    # The matrix form on the same elements and boundary, for comparison.
    reference = coordinate.MfemSpace(elements, essential_dofs=essential)
    u, w = V.random(rng=rng), V.random(rng=rng)
    inner_difference = abs(V.inner_product(u, w) - reference.inner_product(u, w))
    print(
        f"  (u, w) here {V.inner_product(u, w):+.10f}, through the matrix form's "
        f"Gram matrix {reference.inner_product(u, w):+.10f}"
    )

    # A convection term makes the form non-symmetric: the adjoint of its
    # operator in this inner product is M^-1 K^T, neither K nor K^T, and it
    # comes from MFEM's MultTranspose and one mass solve.
    wind = mfem.VectorConstantCoefficient(mfem.Vector([1.0, 0.5]))
    transport = operator_from_bilinear_form(
        V,
        partially_assembled(
            elements, mfem.ConvectionIntegrator(wind), mfem.MassIntegrator()
        ),
    )
    transport_report = check_operator(transport, rng=rng)
    print(f"a convection-mass form, not symmetric: {transport_report}")

    # The Laplacian, positive definite because of the boundary condition, and
    # its inverse: MFEM's conjugate gradients on MFEM's constrained operator,
    # with the Jacobi smoother MFEM builds without a matrix. One system serves
    # both.
    system = V.new_system(partially_assembled(elements, mfem.DiffusionIntegrator()))
    stiffness = operator_from_bilinear_form(V, system, traits=DEFINITE)
    solve_pde = solver_from_bilinear_form(V, system, rtol=1e-10)
    traits_report = check_traits(stiffness, rng=rng)
    inverse_residual = V.norm(V.subtract(stiffness(solve_pde(u)), u)) / V.norm(u)
    print(
        f"the Laplacian, positive definite on the constrained space: {traits_report}; "
        f"MFEM inverts it, |A S u - u| / |u| = {inverse_residual:.1e}"
    )

    # The inverse problem: sensors are linear forms, so the observation
    # operator's action is a list of dot products and its adjoint one mass
    # solve however many sensors there are.
    forms = sensor_forms(elements)
    sensors = operator_from_linear_forms(V, forms)
    forward = sensors @ solve_pde
    prior = SOURCE_STRENGTH * matern_measure(
        V, smoothness=1.0, correlation_length=CORRELATION
    )
    noise = GaussianMeasure.from_standard_deviation(forward.codomain, NOISE)
    problem = LinearForwardProblem(forward, error=noise)
    truth, data = problem.synthetic_model_and_data(prior, rng=rng)
    chi_squared = problem.chi_squared(truth, data)
    t0 = time.perf_counter()
    inversion = LinearGaussianInversion(problem, prior)
    recovered = inversion(data).expectation
    error = V.norm(V.subtract(recovered, truth)) / V.norm(truth)
    boundary = float(np.abs(values(recovered)[essential]).max())
    print(
        f"{len(SENSOR_POSITIONS)} sensors, a Matern prior sampled by MFEM's white "
        f"noise and elliptic solves; chi-squared of the truth {chi_squared:.1f} on "
        f"{len(data)} data; the posterior mean in {time.perf_counter() - t0:.1f} s, "
        f"relative error {error:.3f}, zero on the boundary to {boundary:.1e}"
    )

    total = region_form(elements)
    property_operator = operator_from_linear_forms(V, [total])
    answer = inversion.push_forward(property_operator)(data)
    one = EuclideanSpace(1)
    estimate = float(one.to_components(answer.expectation)[0])
    spread = float(np.sqrt(answer.covariance.matrix(form="components")[0, 0]))
    actual = float(one.to_components(property_operator(truth))[0])
    print(
        f"the total source over the central square: posterior {estimate:+.4f} +/- "
        f"{spread:.4f}, truth {actual:+.4f}"
    )

    # The same data through the matrix form: matrices read out of MFEM and
    # inverted by the library.
    stiffness_form = mfem.BilinearForm(elements)
    stiffness_form.AddDomainIntegrator(mfem.DiffusionIntegrator())
    stiffness_form.Assemble()
    stiffness_form.Finalize()
    their_forward = coordinate.operator_from_linear_forms(
        reference, forms
    ) @ coordinate.solver_from_bilinear_form(reference, stiffness_form, rtol=1e-10)
    their_prior = SOURCE_STRENGTH * coordinate.matern_measure(
        reference, smoothness=1.0, correlation_length=CORRELATION
    )
    their_inversion = LinearGaussianInversion(
        LinearForwardProblem(their_forward, error=noise), their_prior
    )
    their_mean = their_inversion(data).expectation
    their_answer = their_inversion.push_forward(
        coordinate.operator_from_linear_forms(reference, [total])
    )(data)
    their_estimate = float(one.to_components(their_answer.expectation)[0])
    their_spread = float(
        np.sqrt(their_answer.covariance.matrix(form="components")[0, 0])
    )
    mean_difference = V.norm(V.subtract(recovered, their_mean)) / V.norm(their_mean)
    print(
        f"the matrix form on the same data: {their_estimate:+.4f} +/- "
        f"{their_spread:.4f}; the two posterior means differ by {mean_difference:.1e} "
        f"relative, the two estimates by {abs(estimate - their_estimate):.1e}"
    )

    coordinates = dof_coordinates(elements)
    limit = float(np.abs(values(truth)).max())
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), layout="constrained")
    draw_field(
        axes[0], coordinates, values(truth), title="the true source", limit=limit
    )
    draw_field(
        axes[1],
        coordinates,
        values(recovered),
        title="posterior mean, Hilbert-space form",
        limit=limit,
    )
    draw_field(
        axes[2],
        coordinates,
        values(recovered) - values(their_mean),
        title="minus the matrix form's posterior mean",
        limit=max(float(np.abs(values(recovered) - values(their_mean)).max()), 1e-300),
    )
    sensor_x, sensor_y = np.array(SENSOR_POSITIONS).T
    for ax in axes[:2]:
        ax.plot(sensor_x, sensor_y, "k+", ms=6)
    save(fig, outdir / "hilbert_form.png")
    print(f"A took {time.perf_counter() - started:.1f} s")
    return {
        "dim": V.dim,
        "space": space_report,
        "refused": refused,
        "inner_difference": inner_difference,
        "transport": transport_report,
        "traits": traits_report,
        "inverse_residual": inverse_residual,
        "chi_squared": chi_squared,
        "error": error,
        "boundary": boundary,
        "estimate": estimate,
        "spread": spread,
        "actual": actual,
        "their_estimate": their_estimate,
        "their_spread": their_spread,
        "mean_difference": mean_difference,
    }


def run(scale: Scale, /, *, seed: int, outdir: pathlib.Path) -> dict:
    return hilbert_form(scale, rng=part_rng(seed, "a"), outdir=outdir)


def main() -> None:
    args = parse(__doc__)
    run(
        QUICK if args.quick else FULL,
        seed=args.seed,
        outdir=output_directory(args.quick, args.outdir),
    )


if __name__ == "__main__":
    main()
