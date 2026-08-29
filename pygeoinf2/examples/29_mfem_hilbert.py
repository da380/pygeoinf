"""
29. MFEM as a plain Hilbert space: this library conducts, MFEM computes.

Examples 16 and 27 present a finite element space through its coordinates: the
mass matrix is read out of MFEM, factorised here, and every assembled form
becomes a Galerkin matrix this library multiplies and inverts. That is a fine
arrangement for a serial problem of modest size.

This example does the same inverse problem the other way. The space is a
HilbertSpace and nothing more -- no component map, no Gram matrix, no CSR
arrays are ever read. The inner product is MFEM's mass operator applied by
MFEM; the mass solve is MFEM's conjugate gradients; forms become operators
through FormSystemMatrix, which is how MFEM itself imposes boundary
conditions and which is just as happy to hand back a matrix-free operator as
a sparse matrix; white noise is MFEM's own integrator. This library supplies
the composition, the adjoints, the prior and the inversion, and never sees a
matrix.

Why bother, when the coordinate version works? Because nothing in this
arrangement asks whether a vector is local or distributed. Over a
ParFiniteElementSpace with HypreParMatrix operators, the same code is the MPI
version, with this library orchestrating solves across ranks it knows nothing
about. That path is not run here -- it needs mfem.par -- but the serial code
below is written so that nothing would have to change.

Needs MFEM, which comes with the 'mfem' extra.
"""

import numpy as np

import mfem.ser as mfem

from pygeoinf2.algebra.spaces import CoordinateSpace, EuclideanSpace
from pygeoinf2.backends import mfem as coordinate
from pygeoinf2.backends.mfem import essential_dofs_of
from pygeoinf2.backends.mfem_hilbert import (
    MfemHilbertSpace,
    matern_measure,
    operator_from_bilinear_form,
    operator_from_linear_forms,
    solver_from_bilinear_form,
)
from pygeoinf2.inference import LinearForwardProblem, LinearGaussianInversion
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.testing import check_operator, check_space, check_traits
from pygeoinf2.traits import Traits

rng = np.random.default_rng(4)
DEFINITE = Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE

# ---------------------------------------------------------------------------
# The space: the same mesh and boundary condition as example 27.
# ---------------------------------------------------------------------------

mesh = mfem.Mesh.MakeCartesian2D(10, 10, mfem.Element.QUADRILATERAL)
elements = mfem.FiniteElementSpace(mesh, mfem.H1_FECollection(2, mesh.Dimension()))
essential = essential_dofs_of(elements)

# "partial" assembly: MFEM keeps no matrix at all, only the action. On a
# tensor-product mesh this is the arrangement that scales, and the space does
# not care which it was given.
V = MfemHilbertSpace(elements, essential_dofs=essential, assembly="partial")
print(V)
print("  a CoordinateSpace?", isinstance(V, CoordinateSpace))
print(
    "  vectors are", type(V.zero()).__name__, "-- MFEM's, and MFEM applies the metric"
)
check_space(V, rng=rng)
print("  check_space passes; every inner product was an MFEM Mult")
try:
    from pygeoinf2.algebra.operators import LinearOperator

    LinearOperator.identity(V).matrix()
except TypeError as refusal:
    print("  matrix():", str(refusal).split(".")[0])
print()

# The coordinate backend on the same elements, to show the two agree.
reference = coordinate.MfemSpace(elements, essential_dofs=essential)
u, w = V.random(rng=rng), V.random(rng=rng)
print("(u, w) here:               ", f"{V.inner_product(u, w):+.10f}")
print("(u, w) via the Gram matrix:", f"{reference.inner_product(u, w):+.10f}")
print()


# ---------------------------------------------------------------------------
# Forms become operators through FormSystemMatrix, and the adjoint is right.
# ---------------------------------------------------------------------------


def assemble(*integrators):
    """A partially assembled form: MFEM never builds its matrix."""
    form = mfem.BilinearForm(elements)
    form.SetAssemblyLevel(mfem.AssemblyLevel_PARTIAL)
    for integrator in integrators:
        form.AddDomainIntegrator(integrator)
    form.Assemble()
    return form


# A convection term makes the form non-symmetric. The adjoint of its operator
# in this inner product is M^-1 K^T -- neither K nor K^T -- and it comes out
# of MFEM's MultTranspose and one mass solve. check_operator verifies the
# adjoint identity in the finite element inner product.
wind = mfem.VectorConstantCoefficient(mfem.Vector([1.0, 0.5]))
transport = operator_from_bilinear_form(
    V, assemble(mfem.ConvectionIntegrator(wind), mfem.MassIntegrator())
)
check_operator(transport, rng=rng)
print("a convection-mass form: not symmetric, adjoint verified by check_operator")
print("  (A u, w) =", f"{V.inner_product(transport(u), w):+.6f}")
print("  (u, A*w) =", f"{V.inner_product(u, transport.adjoint(w)):+.6f}")
print()

# The Laplacian, positive definite because of the boundary condition, and
# its inverse: MFEM's CG on MFEM's constrained operator, with the Jacobi
# smoother MFEM builds without a matrix. One system serves both.
system = V.new_system(assemble(mfem.DiffusionIntegrator()))
stiffness = operator_from_bilinear_form(V, system, traits=DEFINITE)
solve_pde = solver_from_bilinear_form(V, system, rtol=1e-10)
check_traits(stiffness, rng=rng)
residual = V.norm(V.subtract(stiffness(solve_pde(u)), u)) / V.norm(u)
print("the Laplacian is positive definite here (check_traits), and MFEM inverts it:")
print(f"  |A S u - u| / |u| = {residual:.1e}, no matrix on either side")
print()


# ---------------------------------------------------------------------------
# The inverse problem of example 27, conducted rather than computed.
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


def sensor_forms(centres, width):
    forms = []
    for centre in centres:
        form = mfem.LinearForm(elements)
        form.AddDomainIntegrator(mfem.DomainLFIntegrator(Bump(centre, width)))
        form.Assemble()
        forms.append(form)
    return forms


positions = [(x, y) for x in np.linspace(0.2, 0.8, 4) for y in np.linspace(0.2, 0.8, 4)]
forms = sensor_forms(positions, 0.06)

# Each sensor is a load vector; the observation operator's action is a list
# of dot products and its adjoint is one mass solve, however many sensors.
sensors = operator_from_linear_forms(V, forms)
forward = sensors @ solve_pde
prior = 4.0 * matern_measure(V, smoothness=1.0, correlation_length=0.12)
noise = GaussianMeasure.from_standard_deviation(forward.codomain, 2.0e-4)
problem = LinearForwardProblem(forward, error=noise)
truth, data = problem.synthetic_model_and_data(prior, rng=rng)
print(f"{len(positions)} sensors, a Matern prior sampled by MFEM's white noise and")
print("MFEM's elliptic solves, synthetic data, and an inversion:")

inversion = LinearGaussianInversion(problem, prior)
posterior = inversion(data)
recovered = posterior.expectation
print(
    f"  chi-squared of the truth {problem.chi_squared(truth, data):.1f} on {len(data)} data"
)
print(
    "  relative error of the posterior mean",
    f"{V.norm(V.subtract(recovered, truth)) / V.norm(truth):.3f}",
)
print("  the recovered source vanishes on the boundary to", end=" ")
print(f"{np.abs(np.asarray(recovered.GetDataArray())[essential]).max():.1e}")


class Region(mfem.PyCoefficient):
    def EvalValue(self, x):
        return 1.0 if 0.3 <= x[0] <= 0.7 and 0.3 <= x[1] <= 0.7 else 0.0


total = mfem.LinearForm(elements)
total.AddDomainIntegrator(mfem.DomainLFIntegrator(Region()))
total.Assemble()
property_operator = operator_from_linear_forms(V, [total])
answer = inversion.push_forward(property_operator)(data)
one = EuclideanSpace(1)
estimate = float(one.to_components(answer.expectation)[0])
spread = float(np.sqrt(answer.covariance.matrix(form="components")[0, 0]))
actual = float(one.to_components(property_operator(truth))[0])
print("  total source over the central square:")
print(f"    posterior {estimate:+.4f} +/- {spread:.4f}, truth {actual:+.4f}")
print()

# ---------------------------------------------------------------------------
# The same data through the coordinate backend, for the record.
# ---------------------------------------------------------------------------


def coordinate_answer():
    """Example 27's route: matrices read out of MFEM and inverted here."""
    form = mfem.BilinearForm(elements)
    form.AddDomainIntegrator(mfem.DiffusionIntegrator())
    form.Assemble()
    form.Finalize()
    their_forward = coordinate.operator_from_linear_forms(
        reference, forms
    ) @ coordinate.solver_from_bilinear_form(reference, form, rtol=1e-10)
    their_prior = 4.0 * coordinate.matern_measure(
        reference, smoothness=1.0, correlation_length=0.12
    )
    their_problem = LinearForwardProblem(their_forward, error=noise)
    their_property = coordinate.operator_from_linear_forms(reference, [total])
    their_answer = LinearGaussianInversion(their_problem, their_prior).push_forward(
        their_property
    )(data)
    return float(one.to_components(their_answer.expectation)[0]), float(
        np.sqrt(their_answer.covariance.matrix(form="components")[0, 0])
    )


their_estimate, their_spread = coordinate_answer()
print("the coordinate backend, on the same data:")
print(f"    posterior {their_estimate:+.4f} +/- {their_spread:.4f}")
print(f"  which agrees to {abs(estimate - their_estimate):.1e}: two arrangements,")
print("  one answer. The difference is who did the arithmetic -- and that")
print("  here nobody could have formed a matrix, because there was none.")
