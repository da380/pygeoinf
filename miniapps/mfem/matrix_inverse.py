"""Part C: an inverse problem on the matrix form.

Steady diffusion on the unit square,

    -div(kappa grad u) = f  in the square,   u = 0 on its boundary,

and the inverse question: given the temperature reported by sixteen sensors,
each a local average, where was the heat put in? The unknown is the source
``f``, a function on the same finite element space as ``u``.

Three things make it a test of the design rather than a demonstration. The
boundary condition is a subspace, not a special case: the functions that
vanish on the boundary form a Hilbert space of their own, on which the
Laplacian is positive definite, which the trait check verifies rather than
assumes. The PDE is solved by MFEM: the forward operator is the sensors
composed with MFEM's own conjugate gradients on MFEM's assembled system,
wrapped as an operator with an adjoint. And nothing converts between a load
vector and a function by hand: each sensor is a linear form, whose mass
solve happens inside the operator. The same question is then asked at three
polynomial orders, on the same data.
"""

from __future__ import annotations

import pathlib
import time

import matplotlib.pyplot as plt
import mfem.ser as mfem
import numpy as np

from pygeoinf import (
    EuclideanSpace,
    GaussianMeasure,
    LinearForwardProblem,
    LinearGaussianInversion,
    Traits,
)
from pygeoinf.testing import check_traits

from src.common import (
    FULL,
    QUICK,
    SENSOR_POSITIONS,
    Scale,
    dof_coordinates,
    draw_field,
    parse,
    part_rng,
    region_form,
    save,
    sensor_forms,
    unit_square,
    values,
)
from src.matrix_space import (
    MfemSpace,
    essential_dofs_of,
    matern_measure,
    operator_from_bilinear_form,
    operator_from_linear_forms,
    solver_from_bilinear_form,
)

DEFINITE = Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE


def constrained_space(order: int, divisions: int, /) -> tuple[object, MfemSpace]:
    """An ``H1`` space of the given order, held at zero on the whole boundary."""
    elements = unit_square(order, divisions)
    return elements, MfemSpace(elements, essential_dofs=essential_dofs_of(elements))


def stiffness_form(elements: object, /, *, conductivity: float) -> object:
    """The diffusion form, assembled and finalized by MFEM."""
    form = mfem.BilinearForm(elements)
    form.AddDomainIntegrator(
        mfem.DiffusionIntegrator(mfem.ConstantCoefficient(conductivity))
    )
    form.Assemble()
    form.Finalize()
    return form


def the_problem(elements: object, space: MfemSpace, scale: Scale, /) -> tuple:
    """The forward problem, the prior and the property operator on one space."""
    solve = solver_from_bilinear_form(
        space,
        stiffness_form(elements, conductivity=scale.conductivity),
        rtol=1e-10,
    )
    forms = sensor_forms(elements, width=scale.sensor_width)
    forward = operator_from_linear_forms(space, forms) @ solve
    prior = scale.source_strength * matern_measure(
        space, smoothness=scale.smoothness, correlation_length=scale.correlation_length
    )
    noise = GaussianMeasure.from_standard_deviation(forward.codomain, scale.noise_std)
    problem = LinearForwardProblem(forward, error=noise)
    property_operator = operator_from_linear_forms(space, [region_form(elements)])
    return problem, prior, property_operator


def on_property(
    inversion: LinearGaussianInversion, operator: object, data: object
) -> tuple:
    """The posterior on the one-number property: its mean and standard deviation."""
    answer = inversion.push_forward(operator)(data)
    one = EuclideanSpace(1)
    return (
        float(one.to_components(answer.expectation)[0]),
        float(np.sqrt(answer.covariance.matrix(form="components")[0, 0])),
    )


def prior_on_property(prior: object, operator: object, /) -> tuple:
    """The prior pushed to the one-number property: its mean and standard
    deviation."""
    pushed = prior.affine_map(operator)
    one = EuclideanSpace(1)
    return (
        float(one.to_components(pushed.expectation)[0]),
        float(np.sqrt(pushed.covariance.matrix(form="components")[0, 0])),
    )


def matrix_inverse(
    scale: Scale, /, *, rng: np.random.Generator, outdir: pathlib.Path
) -> dict:
    """The inversion at order two, then the same data at three orders."""
    started = time.perf_counter()
    print(f"\n=== C. An inverse problem on the matrix form ({scale.name} scale) ===")
    elements, V = constrained_space(2, scale.divisions)
    print(
        f"H1 order 2 on a {scale.divisions} x {scale.divisions} mesh of the unit square: "
        f"{elements.GetTrueVSize()} degrees of freedom, {len(V.essential_dofs)} on "
        f"the boundary, so the constrained space has dimension {V.dim}"
    )
    stiffness = operator_from_bilinear_form(
        V, stiffness_form(elements, conductivity=scale.conductivity), traits=DEFINITE
    )
    traits_report = check_traits(stiffness, rng=rng)
    print(
        f"the Laplacian is positive definite on it: {traits_report}; on the "
        f"unconstrained space the constants would be in its kernel"
    )

    problem, prior, property_operator = the_problem(elements, V, scale)
    truth, data = problem.synthetic_model_and_data(prior, rng=rng)
    chi_squared = problem.chi_squared(truth, data)
    print(
        f"{len(SENSOR_POSITIONS)} sensors, forward = sensors @ MFEM's solve; a "
        f"Matern prior, correlation length {scale.correlation_length}, amplitude "
        f"{scale.source_strength}, its covariance, factor and precision all elliptic "
        f"solves; chi-squared of the truth {chi_squared:.1f} on {len(data)} data"
    )
    t0 = time.perf_counter()
    inversion = LinearGaussianInversion(problem, prior)
    recovered = inversion(data).expectation
    error = V.norm(V.subtract(recovered, truth)) / V.norm(truth)
    boundary = float(np.abs(values(recovered)[V.essential_dofs]).max())
    print(
        f"the posterior mean in "
        f"{time.perf_counter() - t0:.1f} s: relative error {error:.3f} (sixteen "
        f"numbers cannot fix a field of {V.dim} degrees of freedom), zero on the "
        f"boundary to {boundary:.1e} because it never left the space"
    )
    estimate, spread = on_property(inversion, property_operator, data)
    actual = float(EuclideanSpace(1).to_components(property_operator(truth))[0])
    print(
        f"what the data do fix, the total source over the central square: "
        f"posterior {estimate:+.4f} +/- {spread:.4f}, truth {actual:+.4f}, "
        f"{abs(estimate - actual) / spread:.1f} standard deviations out"
    )

    print("the same property on the same data, at three polynomial orders:")
    orders = []
    priors = []
    for order in scale.orders:
        order_elements, space = constrained_space(order, scale.divisions)
        order_problem, order_prior, order_property = the_problem(
            order_elements, space, scale
        )
        value, deviation = on_property(
            LinearGaussianInversion(order_problem, order_prior), order_property, data
        )
        orders.append((order, space.dim, value, deviation))
        priors.append(prior_on_property(order_prior, order_property))
        print(
            f"  order {order}: {space.dim:5d} free dofs   {value:+.4f} +/- {deviation:.4f}"
        )
    narrowest = min(orders, key=lambda row: row[3])[0]
    if narrowest == scale.orders[0]:
        print(
            "  the lowest order's bar is the narrowest, which is not better "
            "information but overconfidence: a coarse space cannot represent the "
            "sources it is therefore certain do not exist, and the discretization "
            "acts as a prior nobody wrote down"
        )
    else:
        print(f"  order {narrowest} has the narrowest bar")

    coordinates = dof_coordinates(elements)
    limit = float(np.abs(values(truth)).max())
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), layout="constrained")
    draw_field(
        axes[0], coordinates, values(truth), title="the true source", limit=limit
    )
    draw_field(
        axes[1], coordinates, values(recovered), title="posterior mean", limit=limit
    )
    sensor_x, sensor_y = np.array(SENSOR_POSITIONS).T
    for ax in axes[:2]:
        ax.plot(sensor_x, sensor_y, "k+", ms=6)
    for (order, dim, value, deviation), (center, spread) in zip(orders, priors):
        axes[2].errorbar(
            order - 0.1,
            center,
            yerr=1.96 * spread,
            fmt="s",
            capsize=4,
            color="0.55",
            label="prior" if order == scale.orders[0] else None,
        )
        axes[2].errorbar(
            order + 0.1,
            value,
            yerr=1.96 * deviation,
            fmt="o",
            capsize=4,
            color="C0",
            label="posterior" if order == scale.orders[0] else None,
        )
    axes[2].axhline(actual, color="black", ls="--", lw=1.0, label="truth")
    axes[2].set_xticks(list(scale.orders))
    axes[2].set_xlabel("polynomial order")
    axes[2].set_ylabel("total source over the central square")
    axes[2].set_title(
        "The property at three orders, 95% intervals before and after", fontsize=10
    )
    axes[2].legend(fontsize=8, frameon=False)
    save(
        fig,
        outdir / "matrix_inverse.png",
        title="MFEM, part C: an inverse problem on the matrix form, prior against "
        "posterior. The posterior mean, and the property at three orders",
    )
    print(f"C took {time.perf_counter() - started:.1f} s")
    return {
        "dim": V.dim,
        "traits": traits_report,
        "chi_squared": chi_squared,
        "error": error,
        "boundary": boundary,
        "estimate": estimate,
        "spread": spread,
        "actual": actual,
        "orders": orders,
    }


def run(scale: Scale, /, *, outdir: pathlib.Path) -> dict:
    return matrix_inverse(scale, rng=part_rng(scale.seed, "c"), outdir=outdir)


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
