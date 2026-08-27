"""
Foreign backends: the claim the test doubles cannot settle.

``OpaqueSpace`` shows that the core does not *need* components. These tests
show that it works when the vectors genuinely belong to someone else — an
``mfem.Vector`` with its own memory and its own idea of what a vector is.

MFEM is an optional dependency, so these skip when it is absent.
"""

import numpy as np
import pytest

from pygeoinf2.numerics import CGSolver
from pygeoinf2.testing import (
    check_coordinates,
    check_operator,
    check_space,
    check_traits,
    check_white_noise,
)
from pygeoinf2.traits import Traits

mfem = pytest.importorskip("mfem.ser")

from pygeoinf2.backends.mfem import (  # noqa: E402
    essential_dofs_of,
    operator_from_linear_forms,
    solver_from_bilinear_form,
    MfemSpace,
    _to_scipy,
    functional_from_linear_form,
    operator_from_bilinear_form,
)


@pytest.fixture(scope="module")
def elements():
    mesh = mfem.Mesh.MakeCartesian1D(16, 1.0)
    collection = mfem.H1_FECollection(2, mesh.Dimension())
    return mfem.FiniteElementSpace(mesh, collection)


@pytest.fixture
def V(elements):
    return MfemSpace(elements)


class TestMfemSpace:
    def test_vectors_are_not_arrays(self, V):
        """Which is the point: the core never touches them directly."""
        assert type(V.zero()).__name__ == "Vector"
        assert not isinstance(V.zero(), np.ndarray)

    def test_the_axioms_hold(self, V, rng):
        check_space(V, rng=rng)

    def test_the_coordinate_axioms_hold(self, V, rng):
        check_coordinates(V, rng=rng)

    def test_the_mass_matrix_is_the_metric(self, V, rng):
        """``(u, v) == u^T M v``, not a dot product of coefficients."""
        assert not V.is_orthonormal
        u, w = V.random(rng=rng), V.random(rng=rng)
        components = (V.to_components(u), V.to_components(w))
        assert V.inner_product(u, w) == pytest.approx(
            float(components[0] @ V.gram_matrix() @ components[1])
        )
        assert not np.isclose(
            V.inner_product(u, w), float(components[0] @ components[1])
        )

    def test_to_components_returns_a_copy(self, V, rng):
        """A view into MFEM-owned memory can outlive its owner.

        ``to_components(from_components(c))`` collects the temporary vector
        before the view is read, and the result is plausible wrong numbers
        rather than an error. Pinned here because the failure is silent.
        """
        c = rng.normal(size=V.dim)
        assert np.allclose(V.to_components(V.from_components(c)), c)

    def test_white_noise_is_white(self, V, rng):
        check_white_noise(V, rng=rng, samples=4000, rtol=0.18)


class TestMfemForms:
    def test_a_linear_form_is_a_derivative(self, V, elements, rng):
        load = mfem.LinearForm(elements)
        load.AddDomainIntegrator(mfem.DomainLFIntegrator(mfem.ConstantCoefficient(1.0)))
        load.Assemble()
        functional = functional_from_linear_form(V, load)

        entries = np.asarray(load.GetDataArray())
        u = V.random(rng=rng)
        assert functional(u) == pytest.approx(float(entries @ V.to_components(u)))

        # And its representer is the mass solve, not the load vector.
        assert np.allclose(
            V.to_components(functional.representer), V.solve_gram(entries)
        )
        assert not np.allclose(V.to_components(functional.representer), entries)

    def test_a_bilinear_form_is_a_galerkin_matrix(self, V, elements, rng):
        """``a(u, v) == u^T K v`` means ``K == M A_c``."""
        form = mfem.BilinearForm(elements)
        form.AddDomainIntegrator(mfem.DiffusionIntegrator())
        form.AddDomainIntegrator(mfem.MassIntegrator())
        form.Assemble()
        form.Finalize()

        A = operator_from_bilinear_form(V, form, traits=Traits.POSITIVE_DEFINITE)
        stiffness = _to_scipy(form.SpMat()).toarray()

        x = V.random(rng=rng)
        assert np.allclose(
            V.to_components(A(x)),
            np.linalg.solve(V.gram_matrix(), stiffness @ V.to_components(x)),
        )
        check_operator(A, rng=rng)
        check_traits(A, rng=rng)

    def test_cg_solves_on_the_finite_element_space(self, V, elements, rng):
        form = mfem.BilinearForm(elements)
        form.AddDomainIntegrator(mfem.DiffusionIntegrator())
        form.AddDomainIntegrator(mfem.MassIntegrator())
        form.Assemble()
        form.Finalize()
        A = operator_from_bilinear_form(V, form, traits=Traits.POSITIVE_DEFINITE)

        b = V.random(rng=rng)
        result = CGSolver(rtol=1e-12)(A).solve(b)
        assert result.converged
        assert V.norm(V.subtract(A(result.solution), b)) < 1e-8 * V.norm(b)

        stiffness = _to_scipy(form.SpMat()).toarray()
        direct = np.linalg.solve(stiffness, V.gram_matrix() @ V.to_components(b))
        assert np.allclose(V.to_components(result.solution), direct, atol=1e-8)


def square(order=2, divisions=6):
    """A 2-D H1 space, which is where boundary conditions become interesting."""
    mesh = mfem.Mesh.MakeCartesian2D(divisions, divisions, mfem.Element.QUADRILATERAL)
    collection = mfem.H1_FECollection(order, mesh.Dimension())
    return mfem.FiniteElementSpace(mesh, collection)


class TestEssentialBoundaryConditions:
    """A homogeneous Dirichlet condition as a *subspace*, not a special case.

    The functions vanishing on the boundary form a Hilbert space in their own
    right, and building it is the whole of what the condition does: the mass
    matrix restricted to the free degrees of freedom is its Gram matrix, and a
    bilinear form restricted to the same block is the Galerkin matrix of the
    operator on it. So everything else in the library applies unchanged, which
    is what these tests check.
    """

    @pytest.fixture
    def constrained(self):
        elements = square()
        dofs = essential_dofs_of(elements)
        return elements, dofs, MfemSpace(elements, essential_dofs=dofs)

    def test_the_dimension_drops_by_the_constrained_dofs(self, constrained):
        elements, dofs, space = constrained
        assert dofs.size > 0
        assert space.dim == elements.GetTrueVSize() - dofs.size
        assert space.is_constrained

    def test_it_is_a_hilbert_space_in_its_own_right(self, constrained, rng):
        _, _, space = constrained
        check_space(space, rng=rng)
        check_coordinates(space, rng=rng)

    def test_every_vector_vanishes_on_the_boundary(self, constrained, rng):
        """Not enforced afterwards — there is nowhere for a non-zero boundary
        value to be stored, because it is not a coordinate of this space."""
        elements, dofs, space = constrained
        for vector in (space.zero(), space.random(rng=rng)):
            values = np.asarray(vector.GetDataArray())
            assert values.size == elements.GetTrueVSize()
            assert np.abs(values[dofs]).max() == 0.0

    def test_vectors_stay_the_length_mfem_expects(self, constrained, rng):
        """So that a vector of this space can go straight into a GridFunction,
        which a vector of free values alone could not."""
        elements, _, space = constrained
        vector = space.random(rng=rng)
        function = mfem.GridFunction(elements)
        function.Assign(vector)
        assert function.Size() == elements.GetTrueVSize()

    def test_the_laplacian_is_definite_only_once_constrained(self, constrained, rng):
        """The point of the boundary condition, stated as an operator property:
        constants are in the Laplacian's kernel, and removing them is what the
        condition does. ``check_traits`` verifies the claim rather than taking
        it."""
        elements, _, space = constrained
        form = mfem.BilinearForm(elements)
        form.AddDomainIntegrator(mfem.DiffusionIntegrator())
        form.Assemble()
        form.Finalize()

        operator = operator_from_bilinear_form(
            space, form, traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
        )
        check_operator(operator, rng=rng)
        check_traits(operator, rng=rng)

        # Unconstrained, the same form is singular: a constant function is in
        # its kernel, so the smallest eigenvalue of the Galerkin matrix is zero.
        whole = MfemSpace(elements)
        matrix = whole.restrict(form.SpMat())
        assert np.linalg.eigvalsh(0.5 * (matrix + matrix.T)).min() < 1e-10
        restricted = space.restrict(form.SpMat())
        assert np.linalg.eigvalsh(0.5 * (restricted + restricted.T)).min() > 1e-6

    def test_a_load_vector_is_restricted_too(self, constrained):
        elements, dofs, space = constrained
        form = mfem.LinearForm(elements)
        form.AddDomainIntegrator(mfem.DomainLFIntegrator(mfem.ConstantCoefficient(1.0)))
        form.Assemble()
        functional = functional_from_linear_form(space, form)
        assert functional.derivative_components.size == space.dim
        # And it still evaluates correctly: the constrained entries multiply
        # coefficients that are zero anyway.
        vector = space.random(rng=np.random.default_rng(1))
        assert functional(vector) == pytest.approx(
            float(np.asarray(form.GetDataArray()) @ np.asarray(vector.GetDataArray()))
        )

    def test_only_some_of_the_boundary_can_be_constrained(self):
        """A mixed problem: essential on part of the boundary, natural on the
        rest, which is the usual case and not an exotic one."""
        elements = square()
        every = essential_dofs_of(elements)
        one_side = essential_dofs_of(elements, attributes=[1])
        assert 0 < one_side.size < every.size
        assert set(one_side.tolist()) <= set(every.tolist())
        assert (
            MfemSpace(elements, essential_dofs=one_side).dim
            > MfemSpace(elements, essential_dofs=every).dim
        )

    def test_the_unconstrained_space_is_unchanged(self, rng):
        """The default is exactly what it was before boundary conditions
        existed, so nothing that did not ask for one is affected."""
        elements = square()
        space = MfemSpace(elements)
        assert not space.is_constrained
        assert space.dim == elements.GetTrueVSize()
        assert space.essential_dofs.size == 0
        check_space(space, rng=rng)

    def test_two_boundary_conditions_are_two_spaces(self):
        """They must not compare equal, or an operator built for one would be
        accepted by the other."""
        elements = square()
        whole = MfemSpace(elements)
        part = MfemSpace(elements, essential_dofs=essential_dofs_of(elements))
        assert whole != part
        assert MfemSpace(elements) == whole

    def test_nonsense_is_refused(self):
        elements = square()
        with pytest.raises(ValueError, match="must lie in"):
            MfemSpace(elements, essential_dofs=[elements.GetTrueVSize()])
        with pytest.raises(ValueError, match="dimension zero"):
            MfemSpace(elements, essential_dofs=range(elements.GetTrueVSize()))
        with pytest.raises(ValueError, match="out of range"):
            essential_dofs_of(elements, attributes=[99])


class TestSolvingWithMfem:
    """The PDE solve left to MFEM and wrapped as one of our operators.

    The division of labour: MFEM assembles, preconditions and solves; this
    library says what the result *is* — an operator in the right metric, with
    an adjoint, that composes. So the tests are that it really is the inverse,
    that it really is in the metric, and that MFEM's objects come back
    unharmed.
    """

    @pytest.fixture
    def constrained(self):
        elements = square()
        space = MfemSpace(elements, essential_dofs=essential_dofs_of(elements))
        form = mfem.BilinearForm(elements)
        form.AddDomainIntegrator(mfem.DiffusionIntegrator())
        form.Assemble()
        form.Finalize()
        return elements, space, form

    def test_it_inverts_the_operator_the_form_defines(self, constrained, rng):
        _, space, form = constrained
        operator = operator_from_bilinear_form(
            space, form, traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
        )
        inverse = solver_from_bilinear_form(space, form, rtol=1e-12)
        for _ in range(4):
            vector = space.random(rng=rng)
            recovered = inverse(operator(vector))
            assert space.norm(space.subtract(recovered, vector)) == pytest.approx(
                0.0, abs=1e-8 * space.norm(vector)
            )

    def test_it_agrees_with_solving_the_same_operator_here(self, constrained, rng):
        """Two solvers, two libraries, one operator. Neither can be adjusted to
        match the other, so agreement is evidence that the metric bookkeeping —
        the mass multiply turning a function into a load vector — is right."""
        _, space, form = constrained
        operator = operator_from_bilinear_form(
            space, form, traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
        )
        theirs = solver_from_bilinear_form(space, form, rtol=1e-12)
        ours = CGSolver(rtol=1e-12)(operator)
        for _ in range(4):
            vector = space.random(rng=rng)
            expected = ours(vector)
            assert space.norm(
                space.subtract(theirs(vector), expected)
            ) == pytest.approx(0.0, abs=1e-8 * space.norm(expected))

    def test_it_is_an_operator_like_any_other(self, constrained, rng):
        _, space, form = constrained
        inverse = solver_from_bilinear_form(space, form, rtol=1e-12)
        check_operator(inverse, rng=rng)
        check_traits(inverse, rng=rng)

    def test_the_form_it_was_given_is_left_alone(self, constrained):
        """MFEM's own ``FormSystemMatrix`` takes ownership of the form's
        matrix, so reading it afterwards is a use-after-free — a segfault, not
        an exception. This route does not use it, and the test says so."""
        _, space, form = constrained
        before = _to_scipy(form.SpMat()).toarray().copy()
        solver_from_bilinear_form(space, form, rtol=1e-12)
        assert np.allclose(_to_scipy(form.SpMat()).toarray(), before)

    def test_an_unfinalised_form_is_refused_rather_than_crashing(self):
        """Before ``Finalize`` there are no CSR arrays to read, and reading
        them anyway segfaults with no traceback."""
        elements = square()
        space = MfemSpace(elements)
        form = mfem.BilinearForm(elements)
        form.AddDomainIntegrator(mfem.MassIntegrator())
        form.Assemble()
        assert not form.SpMat().Finalized()
        with pytest.raises(ValueError, match="not been finalised"):
            operator_from_bilinear_form(space, form)

    def test_a_solve_that_does_not_converge_is_reported(self, constrained, rng):
        """An unconverged solve inside an operator is an operator that is
        quietly not the one it claims to be."""
        _, space, form = constrained
        inverse = solver_from_bilinear_form(space, form, rtol=1e-16, max_iterations=1)
        with pytest.raises(RuntimeError, match="did not converge"):
            inverse(space.random(rng=rng))


class TestObservationOperators:
    """Linear forms stacked into an observation operator."""

    def test_the_rows_are_the_functionals_they_came_from(self, rng):
        elements = square()
        space = MfemSpace(elements, essential_dofs=essential_dofs_of(elements))
        forms = []
        for value in (1.0, 2.5):
            form = mfem.LinearForm(elements)
            form.AddDomainIntegrator(
                mfem.DomainLFIntegrator(mfem.ConstantCoefficient(value))
            )
            form.Assemble()
            forms.append(form)

        operator = operator_from_linear_forms(space, forms)
        assert operator.codomain.dim == 2
        vector = space.random(rng=rng)
        observed = operator.codomain.to_components(operator(vector))
        for index, form in enumerate(forms):
            functional = functional_from_linear_form(space, form)
            assert observed[index] == pytest.approx(functional(vector))

    def test_its_adjoint_carries_the_metric(self, rng):
        """The whole reason the rows are derivative components. An adjoint that
        forgot the mass solve would still be a linear map, still have the right
        shape, and be wrong."""
        elements = square()
        space = MfemSpace(elements, essential_dofs=essential_dofs_of(elements))
        form = mfem.LinearForm(elements)
        form.AddDomainIntegrator(mfem.DomainLFIntegrator(mfem.ConstantCoefficient(1.0)))
        form.Assemble()
        operator = operator_from_linear_forms(space, [form])
        check_operator(operator, rng=rng)

    def test_nonsense_is_refused(self):
        elements = square()
        space = MfemSpace(elements)
        with pytest.raises(ValueError, match="At least one"):
            operator_from_linear_forms(space, [])
