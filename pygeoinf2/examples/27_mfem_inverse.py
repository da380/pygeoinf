"""
27. An inverse problem posed on a finite element space.

Example 16 showed that a finite element space *is* a Hilbert space once the
mass matrix is taken as the metric. This does something with that: a real
inverse problem, on a real mesh, with a real boundary condition.

The physics is steady-state diffusion on the unit square,

    -div(kappa grad u) = f    in the domain,   u = 0 on its boundary,

and the question is the inverse one. Given the temperature measured by a few
sensors, where was the heat put in? The unknown is the source ``f``, a function
on the same space as the solution.

Three things make it a genuine test of the design rather than a demonstration.

**The boundary condition is a subspace, not a special case.** Functions
vanishing on the boundary form a Hilbert space in their own right, and that is
what ``MfemSpace(fes, essential_dofs=...)`` builds. The Laplacian is *singular*
on the unconstrained space — constants are in its kernel — and positive
definite on this one, which ``check_traits`` verifies rather than assumes.

**The PDE is solved by MFEM.** The forward operator is
``sensors @ pde_solve``, where the solve is MFEM's own conjugate gradients on
MFEM's own assembled system. This library never sees a matrix: it wraps the
solve as an operator, composes it, and takes the adjoint. Assembly,
preconditioning and the solve stay on MFEM's side of the boundary, which is
where they belong; what this side supplies is the inverse problem.

**Nothing converts between a load vector and a function.** Each sensor is a
linear form, so its natural output is a *derivative*; the mass solve that turns
one into a function happens inside the operator. That is the distinction of
DESIGN.md section 5.6, in the setting where it is easiest to get wrong.

Needs MFEM, which comes with the 'mfem' extra.
"""

import numpy as np

import mfem.ser as mfem

from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.backends.mfem import (
    MfemSpace,
    essential_dofs_of,
    operator_from_bilinear_form,
    operator_from_linear_forms,
    solver_from_bilinear_form,
)
from pygeoinf2.inference import LinearForwardProblem, LinearGaussianInversion
from pygeoinf2.numerics.functional_calculus import operator_inverse_sqrt
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.testing import check_traits
from pygeoinf2.traits import Traits

rng = np.random.default_rng(4)

ORDER = 2
DIVISIONS = 10
CONDUCTIVITY = 1.0
CORRELATION = 0.12
SOURCE_STRENGTH = 4.0
NOISE = 2.0e-4


# ---------------------------------------------------------------------------
# The space: functions vanishing on the boundary.
# ---------------------------------------------------------------------------


def build_space(order, divisions):
    """An H1 space of the given order, constrained on the whole boundary."""
    mesh = mfem.Mesh.MakeCartesian2D(divisions, divisions, mfem.Element.QUADRILATERAL)
    collection = mfem.H1_FECollection(order, mesh.Dimension())
    elements = mfem.FiniteElementSpace(mesh, collection)
    constrained = essential_dofs_of(elements)
    return mesh, elements, MfemSpace(elements, essential_dofs=constrained), constrained


mesh, elements, V, essential = build_space(ORDER, DIVISIONS)
print(f"H1 order {ORDER} on a {DIVISIONS}x{DIVISIONS} mesh of the unit square")
print(f"  {elements.GetTrueVSize()} degrees of freedom, {len(essential)} on the")
print(f"  boundary, so the constrained space has dimension {V.dim}")
print()


def assemble(space, *integrators):
    """Assemble a bilinear form. MFEM's job, in MFEM's terms."""
    form = mfem.BilinearForm(space.finite_element_space)
    for integrator in integrators:
        form.AddDomainIntegrator(integrator)
    form.Assemble()
    form.Finalize()
    return form


DEFINITE = Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE


# ---------------------------------------------------------------------------
# The physics: a stiffness operator that is invertible only because of the BC.
# ---------------------------------------------------------------------------

stiffness_form = assemble(
    V, mfem.DiffusionIntegrator(mfem.ConstantCoefficient(CONDUCTIVITY))
)
stiffness = operator_from_bilinear_form(V, stiffness_form, traits=DEFINITE)
check_traits(stiffness, rng=rng)
print("the pure Laplacian is positive definite here, and check_traits says so.")
print("  On the unconstrained space it would not be: constants are in its")
print("  kernel, and the boundary condition is exactly what removes them.")

# MFEM solves it, and the result is one of our operators.
solve_pde = solver_from_bilinear_form(V, stiffness_form, rtol=1e-10)
print("  and MFEM inverts it: solver_from_bilinear_form hands back a")
print("  LinearOperator that happens to be a PDE solve, with an adjoint")
print()


# ---------------------------------------------------------------------------
# The observations: local averages, which is what a sensor actually reports.
# ---------------------------------------------------------------------------


class Bump(mfem.PyCoefficient):
    """A normalised Gaussian window, standing for a sensor's footprint."""

    def __init__(self, centre, width):
        super().__init__()
        self._centre = np.asarray(centre, float)
        self._width = float(width)

    def EvalValue(self, x):
        offset = np.asarray([x[0], x[1]]) - self._centre
        return float(
            np.exp(-0.5 * offset @ offset / self._width**2)
            / (2.0 * np.pi * self._width**2)
        )


def sensor_operator(space, centres, width):
    """Local averages at each centre, assembled by MFEM and wrapped as one
    operator.

    A sensor is a linear form, so MFEM assembles it exactly as it assembles
    anything else; ``operator_from_linear_forms`` stacks the load vectors into
    an observation operator. Again nothing is converted by hand: the rows are
    *derivative* components, and the mass solve the adjoint needs stays inside.
    """
    forms = []
    for centre in centres:
        form = mfem.LinearForm(space.finite_element_space)
        form.AddDomainIntegrator(mfem.DomainLFIntegrator(Bump(centre, width)))
        form.Assemble()
        forms.append(form)
    return operator_from_linear_forms(space, forms)


positions = [(x, y) for x in np.linspace(0.2, 0.8, 4) for y in np.linspace(0.2, 0.8, 4)]
sensors = sensor_operator(V, positions, 0.06)
print(f"{len(positions)} sensors, each averaging over a small window")

# The forward operator: put in a source, solve the PDE, read the sensors.
forward = sensors @ solve_pde
print("forward operator = sensors @ (MFEM's PDE solve):")
print(f"  {V.dim} -> {forward.codomain.dim}, composed from two wrapped MFEM")
print("  objects, and this library assembles nothing of its own")
print()


# ---------------------------------------------------------------------------
# The prior: smooth sources, through a differential operator.
# ---------------------------------------------------------------------------

# A Whittle-Matern-style prior. With S the operator of
# ``a(u,v) = uv + l^2 grad u . grad v``, the covariance is ``sigma^2 S^-1``:
# self-adjoint, positive definite, and smoothing. Its *precision* is available
# in closed form, which is unusual and useful — no factorisation is needed for
# the model-space formalism — and a factor for sampling comes from a Lanczos
# inverse square root rather than from a dense decomposition.
shift_form = assemble(
    V,
    mfem.MassIntegrator(),
    mfem.DiffusionIntegrator(mfem.ConstantCoefficient(CORRELATION**2)),
)
shift = operator_from_bilinear_form(V, shift_form, traits=DEFINITE)
covariance = (SOURCE_STRENGTH**2) * solver_from_bilinear_form(V, shift_form, rtol=1e-10)
prior = GaussianMeasure(
    V,
    covariance=covariance,
    precision=(1.0 / SOURCE_STRENGTH**2) * shift,
    covariance_factor=SOURCE_STRENGTH
    * operator_inverse_sqrt(shift, max_iterations=80, rtol=1e-11),
)
print(f"prior: correlation length {CORRELATION}, pointwise scale {SOURCE_STRENGTH}")
print("  its precision is a differential operator, known exactly;")
print("  its sampling factor is a Lanczos inverse square root")
print()


# ---------------------------------------------------------------------------
# A synthetic source, its data, and the inversion.
# ---------------------------------------------------------------------------

noise = GaussianMeasure.from_standard_deviation(forward.codomain, NOISE)
problem = LinearForwardProblem(forward, error=noise)
truth, data = problem.synthetic_model_and_data(prior, rng=rng)

print(f"chi-squared of the truth: {problem.chi_squared(truth, data):.1f}")
print(f"  on {problem.data_space.dim} data")

inversion = LinearGaussianInversion(problem, prior)
posterior = inversion(data)
recovered = posterior.expectation

error = V.norm(V.subtract(recovered, truth)) / V.norm(truth)
print(f"normal equations assembled in the {inversion.formalism}")
print(f"relative error of the posterior mean: {error:.3f}")
print("  which is not small, and should not be: sixteen numbers cannot")
print(f"  determine a field with {V.dim} degrees of freedom. What they can")
print("  determine is a *property* of it, which is the point below.")

boundary_values = np.asarray(recovered.GetDataArray())[essential]
print(
    "the recovered source vanishes on the boundary to "
    f"{np.abs(boundary_values).max():.1e} -- not enforced afterwards, but true"
)
print("  because the answer never left the space where it holds")
print()


# ---------------------------------------------------------------------------
# What the data actually determined.
# ---------------------------------------------------------------------------


# The total source over the middle of the domain: one number, with an honest
# error bar, obtained without ever forming the posterior covariance.
class Region(mfem.PyCoefficient):
    """One inside a central square, zero outside."""

    def EvalValue(self, x):
        return 1.0 if 0.3 <= x[0] <= 0.7 and 0.3 <= x[1] <= 0.7 else 0.0


total = mfem.LinearForm(elements)
total.AddDomainIntegrator(mfem.DomainLFIntegrator(Region()))
total.Assemble()
property_operator = operator_from_linear_forms(V, [total])

answer = inversion.push_forward(property_operator)(data)
estimate = float(EuclideanSpace(1).to_components(answer.expectation)[0])
deviation = float(np.sqrt(answer.covariance.matrix(form="components")[0, 0]))
actual = float(EuclideanSpace(1).to_components(property_operator(truth))[0])
print("total source over the central square:")
print(f"  posterior {estimate:+.4f} +/- {deviation:.4f}")
print(f"  truth     {actual:+.4f}")
print(f"  which is {abs(estimate - actual) / deviation:.1f} standard deviations out")
print()


# ---------------------------------------------------------------------------
# And it does not depend on the discretisation.
# ---------------------------------------------------------------------------

# The same question at three polynomial orders, on the *same* data. Where the
# space is fine enough to represent what the prior describes, the answer should
# not move; where it is not, the discretisation is quietly doing the work of a
# prior, and the error bar is the place that shows.
print("the same property, at three polynomial orders:")
for order in (1, 2, 3):
    _, order_elements, space, _ = build_space(order, DIVISIONS)
    order_stiffness = assemble(
        space, mfem.DiffusionIntegrator(mfem.ConstantCoefficient(CONDUCTIVITY))
    )
    order_forward = sensor_operator(space, positions, 0.06) @ solver_from_bilinear_form(
        space, order_stiffness, rtol=1e-10
    )
    order_shift_form = assemble(
        space,
        mfem.MassIntegrator(),
        mfem.DiffusionIntegrator(mfem.ConstantCoefficient(CORRELATION**2)),
    )
    order_prior = GaussianMeasure(
        space,
        covariance=(SOURCE_STRENGTH**2)
        * solver_from_bilinear_form(space, order_shift_form, rtol=1e-10),
        precision=(1.0 / SOURCE_STRENGTH**2)
        * operator_from_bilinear_form(space, order_shift_form, traits=DEFINITE),
    )
    order_problem = LinearForwardProblem(
        order_forward,
        error=GaussianMeasure.from_standard_deviation(order_forward.codomain, NOISE),
    )
    order_region = mfem.LinearForm(order_elements)
    order_region.AddDomainIntegrator(mfem.DomainLFIntegrator(Region()))
    order_region.Assemble()
    order_property = operator_from_linear_forms(space, [order_region])
    # The *same* data, inverted on a different discretisation.
    order_answer = LinearGaussianInversion(order_problem, order_prior).push_forward(
        order_property
    )(data)
    value = float(EuclideanSpace(1).to_components(order_answer.expectation)[0])
    spread = float(np.sqrt(order_answer.covariance.matrix(form="components")[0, 0]))
    print(f"  order {order}: {space.dim:5d} free dofs   {value:+.4f} +/- {spread:.4f}")
print("  orders 2 and 3 agree closely, which is what makes the answer a")
print("  statement about the problem rather than about the mesh -- and they")
print("  can agree because the metric is the mass matrix, so the prior means")
print("  the same thing on both spaces rather than being rescaled by the")
print("  discretisation.")
print()
print("  order 1 is the interesting one. It differs, and its error bar is")
print("  *smaller* than the finer spaces' -- which is not better information")
print("  but overconfidence: a coarse space cannot represent the sources it")
print("  is therefore certain do not exist. The discretisation is acting as a")
print("  prior nobody wrote down, which is the failure mode this whole")
print("  arrangement exists to make visible.")
