"""
MFEM as a plain Hilbert space: the conductor arrangement.

``backends.mfem`` reads MFEM's matrices and does the linear algebra here.
``backends.mfem_hilbert`` reads nothing: the space has no coordinates, and
every operation is an MFEM ``Mult`` or an MFEM solve. These tests establish
that the second arrangement gives the same answers as the first on the same
element space — inner products, operators, adjoints of non-symmetric forms,
functionals, observation operators, white noise, Matern fields and a whole
inversion — and that it does so without ever forming a matrix.

The comparison against the coordinate backend is the metric rule in its
strongest available form: that backend's Gram matrix is the assembled mass
matrix, which is as far from diagonal as a Gram matrix gets.

MFEM is an optional dependency, so these skip when it is absent.
"""

import gc

import numpy as np
import pytest

from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.algebra.spaces import CoordinateSpace, MassWeightedSpace
from pygeoinf2.inference import LinearForwardProblem, LinearGaussianInversion
from pygeoinf2.numerics import CGSolver
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.testing import (
    check_operator,
    check_space,
    check_traits,
    check_white_noise,
)
from pygeoinf2.traits import Traits

mfem = pytest.importorskip("mfem.ser")

from pygeoinf2.backends import mfem as coordinate  # noqa: E402
from pygeoinf2.backends import mfem_hilbert as opaque  # noqa: E402
from pygeoinf2.backends.mfem import essential_dofs_of  # noqa: E402
from pygeoinf2.backends.mfem_hilbert import (  # noqa: E402
    MfemDofSpace,
    MfemHilbertSpace,
    functional_from_linear_form,
    matern_measure,
    operator_from_bilinear_form,
    operator_from_linear_forms,
    solver_from_bilinear_form,
    white_noise_load,
)

DEFINITE = Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE


def values(x):
    """A vector's degree-of-freedom values, copied out of MFEM's memory."""
    return np.array(x.GetDataArray(), copy=True)


@pytest.fixture(scope="module")
def elements():
    mesh = mfem.Mesh.MakeCartesian2D(5, 5, mfem.Element.QUADRILATERAL)
    collection = mfem.H1_FECollection(2, mesh.Dimension())
    return mfem.FiniteElementSpace(mesh, collection)


@pytest.fixture(scope="module")
def essential(elements):
    return essential_dofs_of(elements)


@pytest.fixture(params=["full", "partial"])
def assembly(request):
    return request.param


@pytest.fixture
def V(elements, essential, assembly):
    """The constrained space, in both assembly modes."""
    return MfemHilbertSpace(elements, essential_dofs=essential, assembly=assembly)


@pytest.fixture
def reference(elements, essential):
    """The coordinate backend on the same element space and constraint."""
    return coordinate.MfemSpace(elements, essential_dofs=essential)


def bilinear(elements, assembly, *integrators):
    """An assembled form at the given level. Not finalised: FormSystemMatrix
    does that itself, and the opaque backend never reads ``SpMat``."""
    form = mfem.BilinearForm(elements)
    if assembly == "partial":
        form.SetAssemblyLevel(mfem.AssemblyLevel_PARTIAL)
    for integrator in integrators:
        form.AddDomainIntegrator(integrator)
    form.Assemble()
    return form


def finalised(elements, *integrators):
    """The same form for the coordinate backend, which reads the matrix."""
    form = bilinear(elements, "full", *integrators)
    form.Finalize()
    return form


def linear(elements, coefficient):
    form = mfem.LinearForm(elements)
    form.AddDomainIntegrator(mfem.DomainLFIntegrator(coefficient))
    form.Assemble()
    return form


def convection():
    """A non-symmetric integrator: its operator's adjoint is neither K nor K^T."""
    return mfem.ConvectionIntegrator(
        mfem.VectorConstantCoefficient(mfem.Vector([1.0, 0.5]))
    )


class TestTheSpace:
    def test_it_is_a_hilbert_space_and_not_a_coordinate_space(self, V):
        assert isinstance(V, MassWeightedSpace)
        assert not isinstance(V, CoordinateSpace)
        assert isinstance(V.dof_space, MfemDofSpace)
        assert type(V.zero()).__name__ == "Vector"

    def test_the_axioms_hold(self, V, rng):
        check_space(V, rng=rng)

    def test_the_dof_space_axioms_hold_too(self, V, rng):
        check_space(V.dof_space, rng=rng)

    def test_the_inner_product_is_the_coordinate_backends(self, V, reference, rng):
        """``(u, v) == u^T M v`` without anybody forming ``M``."""
        u, w = V.random(rng=rng), V.random(rng=rng)
        assert V.inner_product(u, w) == pytest.approx(reference.inner_product(u, w))
        assert not np.isclose(V.inner_product(u, w), V.dof_space.inner_product(u, w))

    def test_the_mass_solve_inverts_the_mass_operator(self, V, rng):
        x = V.random(rng=rng)
        back = V.mass_inverse(V.mass(x))
        assert V.norm(V.subtract(back, x)) < 1e-12 * V.norm(x)

    def test_the_dimension_is_the_free_count(self, V, elements, essential):
        assert V.dim == elements.GetTrueVSize() - len(essential)
        assert V.dof_space.size == elements.GetTrueVSize()

    def test_every_vector_vanishes_on_the_boundary(self, V, essential, rng):
        for draw in (V.random(rng=rng), V.white_noise(rng=rng), V.zero()):
            assert np.all(values(draw)[essential] == 0.0)

    def test_two_constructions_are_the_same_space(self, elements, essential, assembly):
        a = MfemHilbertSpace(elements, essential_dofs=essential, assembly=assembly)
        b = MfemHilbertSpace(elements, essential_dofs=essential, assembly=assembly)
        assert a == b and hash(a) == hash(b)
        assert a.shares_vectors_with(b.dof_space)
        assert a != MfemHilbertSpace(elements, assembly=assembly)

    def test_no_matrix_can_be_formed(self, V):
        A = LinearOperator.identity(V)
        with pytest.raises(TypeError, match="no coordinate map"):
            A.matrix()

    def test_nonsense_is_refused(self, elements):
        with pytest.raises(ValueError, match="assembly"):
            MfemHilbertSpace(elements, assembly="sparse")
        with pytest.raises(ValueError, match="out of range|must lie"):
            MfemHilbertSpace(elements, essential_dofs=[elements.GetTrueVSize()])
        with pytest.raises(ValueError, match="dimension zero"):
            MfemHilbertSpace(elements, essential_dofs=range(elements.GetTrueVSize()))


class TestWhiteNoise:
    def test_it_is_white_on_the_space(self, V, rng):
        check_white_noise(V, rng=rng, samples=3000, rtol=0.2)

    def test_the_load_has_the_mass_operator_as_covariance(self, V, reference):
        """Against the coordinate backend's Gram matrix, which is ``M``'s
        free block: the one place a matrix appears in this file, and it is
        the other backend's."""
        rng = np.random.default_rng(11)
        free = reference.free_dofs
        loads = np.array(
            [values(white_noise_load(V, rng=rng))[free] for _ in range(4000)]
        )
        covariance = loads.T @ loads / len(loads)
        gram = reference.gram_matrix()
        scale = np.abs(np.diag(gram)).max()
        assert np.abs(covariance - gram).max() < 0.1 * scale


class TestOperators:
    def test_a_symmetric_form_gives_the_coordinate_backends_operator(
        self, V, reference, elements, assembly, rng
    ):
        A = operator_from_bilinear_form(
            V,
            bilinear(
                elements, assembly, mfem.DiffusionIntegrator(), mfem.MassIntegrator()
            ),
            traits=DEFINITE,
        )
        B = coordinate.operator_from_bilinear_form(
            reference,
            finalised(elements, mfem.DiffusionIntegrator(), mfem.MassIntegrator()),
            traits=DEFINITE,
        )
        x = V.random(rng=rng)
        assert np.allclose(values(A(x)), values(B(x)))
        check_traits(A, rng=rng)

    def test_a_non_symmetric_form_gets_its_true_adjoint(
        self, V, reference, elements, assembly, rng
    ):
        """``M^-1 K^T``, through MFEM's ``MultTranspose`` and a mass solve.
        Neither ``K`` nor ``K^T`` alone satisfies the adjoint identity here."""
        A = operator_from_bilinear_form(
            V, bilinear(elements, assembly, convection(), mfem.MassIntegrator())
        )
        B = coordinate.operator_from_bilinear_form(
            reference, finalised(elements, convection(), mfem.MassIntegrator())
        )
        check_operator(A, rng=rng)
        x = V.random(rng=rng)
        assert np.allclose(values(A(x)), values(B(x)))
        assert np.allclose(values(A.adjoint(x)), values(B.adjoint(x)))
        assert not np.allclose(values(A.adjoint(x)), values(A(x)))

    def test_the_operator_outlives_the_form_it_came_from(
        self, V, elements, assembly, rng
    ):
        """The eliminated matrix belongs to the form and the constrained
        operator refers back to it; without the retained references the
        first application after collection reads freed memory."""
        form = bilinear(elements, assembly, mfem.DiffusionIntegrator())
        A = operator_from_bilinear_form(V, form, traits=DEFINITE)
        x = V.random(rng=rng)
        before = values(A(x))
        del form
        gc.collect()
        assert np.allclose(values(A(x)), before)

    def test_partial_assembly_is_full_assembly_without_the_matrix(
        self, elements, essential, rng
    ):
        spaces = {
            level: MfemHilbertSpace(elements, essential_dofs=essential, assembly=level)
            for level in ("full", "partial")
        }
        results = {}
        for level, space in spaces.items():
            A = operator_from_bilinear_form(
                space,
                bilinear(elements, level, mfem.DiffusionIntegrator()),
                traits=DEFINITE,
            )
            results[level] = values(A(space.random(rng=np.random.default_rng(3))))
        assert np.allclose(results["full"], results["partial"])


class TestSolvingWithMfem:
    def test_it_inverts_the_operator_the_form_defines(self, V, elements, assembly, rng):
        system = V.new_system(bilinear(elements, assembly, mfem.DiffusionIntegrator()))
        A = operator_from_bilinear_form(V, system, traits=DEFINITE)
        S = solver_from_bilinear_form(V, system, rtol=1e-13)
        w = V.random(rng=rng)
        assert V.norm(V.subtract(A(S(w)), w)) < 1e-9 * V.norm(w)
        assert V.norm(V.subtract(S(A(w)), w)) < 1e-9 * V.norm(w)
        check_operator(S, rng=rng)

    def test_it_agrees_with_solving_here_by_cg(self, V, elements, assembly, rng):
        """The core's coordinate-free CG on the wrapped operator against
        MFEM's on its own system: same operator, two Krylov loops."""
        system = V.new_system(bilinear(elements, assembly, mfem.DiffusionIntegrator()))
        A = operator_from_bilinear_form(V, system, traits=DEFINITE)
        S = solver_from_bilinear_form(V, system, rtol=1e-13)
        w = V.random(rng=rng)
        ours = CGSolver(rtol=1e-12)(A).solve(w).solution
        assert V.norm(V.subtract(ours, S(w))) < 1e-8 * V.norm(w)

    def test_a_system_from_another_space_is_refused(
        self, elements, essential, assembly
    ):
        a = MfemHilbertSpace(elements, essential_dofs=essential, assembly=assembly)
        b = MfemHilbertSpace(elements, assembly=assembly)
        system = a.new_system(bilinear(elements, assembly, mfem.DiffusionIntegrator()))
        with pytest.raises(ValueError, match="different space"):
            operator_from_bilinear_form(b, system)

    def test_a_solve_that_does_not_converge_is_reported(
        self, V, elements, assembly, rng
    ):
        S = solver_from_bilinear_form(
            V,
            bilinear(elements, assembly, mfem.DiffusionIntegrator()),
            max_iterations=1,
        )
        with pytest.raises(RuntimeError, match="did not converge"):
            S(V.random(rng=rng))

    def test_a_caller_may_supply_the_solver(self, V, elements, assembly, rng):
        seen = []

        def make(operator):
            solver = mfem.CGSolver()
            solver.SetOperator(operator)
            solver.SetRelTol(1e-13)
            solver.SetMaxIter(2000)
            solver.SetPrintLevel(-1)
            seen.append(solver)
            return solver

        system = V.new_system(bilinear(elements, assembly, mfem.DiffusionIntegrator()))
        A = operator_from_bilinear_form(V, system, traits=DEFINITE)
        S = solver_from_bilinear_form(V, system, make_solver=make)
        w = V.random(rng=rng)
        assert len(seen) == 1
        assert V.norm(V.subtract(A(S(w)), w)) < 1e-8 * V.norm(w)


class TestFunctionalsAndObservations:
    def test_a_linear_form_is_a_derivative(self, V, reference, elements, rng):
        form = linear(elements, mfem.ConstantCoefficient(1.0))
        f = functional_from_linear_form(V, form)
        g = coordinate.functional_from_linear_form(reference, form)
        u = V.random(rng=rng)
        assert f(u) == pytest.approx(g(u))
        assert np.allclose(values(f.representer), values(g.representer), atol=1e-10)
        # the load vector itself is not the representer: the mass solve matters
        load = values(mfem.Vector(form))
        assert not np.allclose(values(f.representer), load)
        assert f(u) == pytest.approx(float(load @ values(u)))

    def test_observation_rows_are_the_functionals(self, V, reference, elements, rng):
        forms = [linear(elements, mfem.ConstantCoefficient(c)) for c in (0.5, 1.0, 2.0)]
        A = operator_from_linear_forms(V, forms)
        B = coordinate.operator_from_linear_forms(reference, forms)
        x = V.random(rng=rng)
        assert np.allclose(A(x), B(x))
        for row, form in zip(A(x), forms):
            assert row == pytest.approx(functional_from_linear_form(V, form)(x))

    def test_the_adjoint_carries_the_metric(self, V, reference, elements, rng):
        forms = [linear(elements, mfem.ConstantCoefficient(c)) for c in (0.5, 2.0)]
        A = operator_from_linear_forms(V, forms)
        B = coordinate.operator_from_linear_forms(reference, forms)
        check_operator(A, rng=rng)
        d = rng.normal(size=2)
        assert np.allclose(values(A.adjoint(d)), values(B.adjoint(d)), atol=1e-10)

    def test_nonsense_is_refused(self, V, elements):
        from pygeoinf2.algebra.spaces import EuclideanSpace

        with pytest.raises(ValueError, match="At least one"):
            operator_from_linear_forms(V, [])
        with pytest.raises(ValueError, match="data space of dimension"):
            operator_from_linear_forms(
                V,
                [linear(elements, mfem.ConstantCoefficient(1.0))],
                codomain=EuclideanSpace(3),
            )


class TestMaternMeasure:
    def test_it_matches_the_coordinate_backends_field(self, V, reference, rng):
        ours = matern_measure(V, smoothness=1.0, correlation_length=0.2)
        theirs = coordinate.matern_measure(
            reference, smoothness=1.0, correlation_length=0.2
        )
        x = V.random(rng=rng)
        assert np.allclose(
            values(ours.covariance(x)),
            values(theirs.covariance(x)),
            rtol=1e-6,
            atol=1e-9,
        )
        assert np.allclose(
            values(ours.precision(x)), values(theirs.precision(x)), rtol=1e-6, atol=1e-9
        )

    def test_the_precision_inverts_the_covariance(self, V, rng):
        field = matern_measure(V, smoothness=1.0, correlation_length=0.2)
        x = V.random(rng=rng)
        back = field.precision(field.covariance(x))
        assert V.norm(V.subtract(back, x)) < 1e-7 * V.norm(x)

    def test_it_samples_on_the_space(self, V, essential, rng):
        field = matern_measure(V, smoothness=1.0, correlation_length=0.2)
        draw = field.sample(rng=rng)
        assert type(draw).__name__ == "Vector"
        assert np.all(values(draw)[essential] == 0.0)
        assert V.norm(draw) > 0.0

    def test_anisotropy_needs_a_matrix_coefficient_and_full_assembly(
        self, elements, essential, reference, rng
    ):
        full = MfemHilbertSpace(elements, essential_dofs=essential, assembly="full")
        ours = matern_measure(full, smoothness=1.0, correlation_length=(0.3, 0.1))
        theirs = coordinate.matern_measure(
            reference, smoothness=1.0, correlation_length=(0.3, 0.1)
        )
        x = full.random(rng=rng)
        assert np.allclose(
            values(ours.covariance(x)),
            values(theirs.covariance(x)),
            rtol=1e-6,
            atol=1e-9,
        )
        partial = MfemHilbertSpace(
            elements, essential_dofs=essential, assembly="partial"
        )
        with pytest.raises(ValueError, match="partial"):
            matern_measure(partial, smoothness=1.0, correlation_length=(0.3, 0.1))

    def test_bad_parameters_are_refused(self, V):
        with pytest.raises(ValueError, match="positive integer"):
            matern_measure(V, smoothness=0.5)
        with pytest.raises(ValueError, match="positive"):
            matern_measure(V, correlation_length=-1.0)


class TestAWholeInversion:
    """The conductor arrangement end to end, against the coordinate one.

    Sensors, a PDE solved by MFEM, a Matern prior and a Bayesian inversion
    — the shape of example 27 — on the space with no coordinates. The
    posterior mean must agree with the one the coordinate backend produces
    from the same data, and the inversion must never have asked for a
    matrix, which it cannot have because there is none to give.
    """

    @pytest.fixture
    def positions(self):
        return [(x, y) for x in (0.3, 0.7) for y in (0.3, 0.7)]

    @staticmethod
    def sensors(elements, positions):
        class Bump(mfem.PyCoefficient):
            def __init__(self, centre):
                super().__init__()
                self._centre = np.asarray(centre, float)

            def EvalValue(self, x):
                offset = np.asarray([x[0], x[1]]) - self._centre
                return float(np.exp(-0.5 * offset @ offset / 0.1**2))

        return [linear(elements, Bump(centre)) for centre in positions]

    def test_the_posterior_mean_agrees_with_the_coordinate_backend(
        self, V, reference, elements, assembly, positions
    ):
        forms = self.sensors(elements, positions)
        ours = self.inversion(
            V, opaque, forms, bilinear(elements, assembly, mfem.DiffusionIntegrator())
        )
        theirs = self.inversion(
            reference,
            coordinate,
            forms,
            finalised(elements, mfem.DiffusionIntegrator()),
        )
        rng = np.random.default_rng(7)
        truth = theirs.prior.sample(rng=rng)
        data = theirs.problem.forward_operator(truth) + 1e-3 * rng.normal(
            size=len(forms)
        )
        mine = ours.inversion(data).expectation
        other = theirs.inversion(data).expectation
        assert V.norm(V.subtract(mine, other)) < 1e-6 * V.norm(other)

    @staticmethod
    def inversion(space, backend, forms, stiffness):
        class Setup:
            pass

        setup = Setup()
        solve = backend.solver_from_bilinear_form(space, stiffness, rtol=1e-12)
        forward = backend.operator_from_linear_forms(space, forms) @ solve
        setup.prior = backend.matern_measure(
            space, smoothness=1.0, correlation_length=0.2
        )
        setup.problem = LinearForwardProblem(
            forward,
            error=GaussianMeasure.from_standard_deviation(forward.codomain, 1e-3),
        )
        setup.inversion = LinearGaussianInversion(setup.problem, setup.prior)
        return setup
