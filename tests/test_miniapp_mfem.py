"""
The MFEM mini-app keeps working.

``miniapps/mfem`` wraps MFEM two ways. A finite element space
becomes a Hilbert space, either through its matrices (``src/matrix_space``:
the mass matrix as the Gram matrix of a coordinate space, assembled forms as
Galerkin matrices) or with no coordinates at all (``src/hilbert_space``:
every inner product, mass solve and form application is MFEM's own). And an
operator on the library's own periodic box has an MFEM solve as its action
(``src/box_operator``). The modules are tested class by class: the space axioms,
a linear form as a derivative and a bilinear form as a Galerkin matrix,
boundary conditions as subspaces, MFEM's solves wrapped as operators, white
noise and Matern fields from MFEM's own pieces, the two forms agreeing on
everything they share, and the box operator's steps and their adjoints.
Then every part of the app runs once at the quick scale, built through the
scripts' own command line, each through its own script's entry point, and
what it shows is asserted.

Needs PyMFEM, whose wheels are built for Linux only; without it the tests
skip and say so.
"""

from __future__ import annotations

import dataclasses
import gc
import importlib
import pathlib
import sys

import numpy as np
import pytest

from pygeoinf.algebra.operators import LinearOperator
from pygeoinf.algebra.spaces import CoordinateSpace, EuclideanSpace, MassWeightedSpace
from pygeoinf.inference import LinearForwardProblem, LinearGaussianInversion
from pygeoinf.numerics import CGSolver
from pygeoinf.numerics.solvers import ConvergenceError
from pygeoinf.probability.gaussian import GaussianMeasure
from pygeoinf.testing import (
    check_coordinates,
    check_operator,
    check_space,
    check_traits,
    check_white_noise,
)
from pygeoinf.traits import Traits

mfem = pytest.importorskip(
    "mfem.ser", reason="The MFEM mini-app needs PyMFEM, the mfem extra (Linux only)."
)

MINIAPP = pathlib.Path(__file__).resolve().parents[1] / "miniapps" / "mfem"
if not (MINIAPP / "src" / "matrix_space.py").exists():
    pytest.skip(f"The mini-app is not at {MINIAPP}.", allow_module_level=True)

pytestmark = pytest.mark.slow

PARTS = {
    "a": "hilbert_form",
    "b": "matrix_form",
    "c": "matrix_inverse",
    "d": "box_ladder",
    "e": "box_inverse",
}
# The app's modules are the package ``src``, which its part scripts import by
# that plain name; other mini-apps' packages have the same name. What this
# module has loaded, kept here so that every test sees one copy of each.
_LOADED: dict = {}


def _ours(name: str) -> bool:
    """Whether a module name is the app's package, one of its modules, or one
    of its part scripts."""
    return name == "src" or name.startswith("src.") or name in PARTS.values()


def _module(name: str):
    """One of the mini-app's modules or scripts, ``src.common`` or
    ``box_ladder``, imported by name with the app's directory on the path for
    the duration, as a script run there would find it.

    Another app's package of the same name is dropped first, and what this
    module loaded before is put back, should another app's test have dropped
    it in between.
    """
    if sys.modules.get("src") is not _LOADED.get("src"):
        for loaded in [loaded for loaded in sys.modules if _ours(loaded)]:
            del sys.modules[loaded]
        sys.modules.update(_LOADED)
    sys.path.insert(0, str(MINIAPP))
    try:
        module = importlib.import_module(name)
    finally:
        sys.path.remove(str(MINIAPP))
    _LOADED.update({loaded: m for loaded, m in sys.modules.items() if _ours(loaded)})
    return module


# The two space modules are needed here, for the classes below. The rest are
# imported by the fixtures that need them.
coordinate = _module("src.matrix_space")
opaque = _module("src.hilbert_space")


@pytest.fixture(scope="module", autouse=True)
def _forget_the_app():
    """Nothing after this module may see the app's modules by their plain names."""
    yield
    for name in [name for name in sys.modules if _ours(name)]:
        if sys.modules[name] is _LOADED.get(name):
            del sys.modules[name]


# --------------------------------------------------------------------- #
#                 The matrix form: a finite element space                #
#                    as a coordinate space (matrix_space)                #
# --------------------------------------------------------------------- #


class _OnAnInterval:
    """Quadratic elements on [0, 1], the matrix form's simplest space."""

    @pytest.fixture(scope="class")
    @classmethod
    def elements(cls):
        mesh = mfem.Mesh.MakeCartesian1D(16, 1.0)
        collection = mfem.H1_FECollection(2, mesh.Dimension())
        return mfem.FiniteElementSpace(mesh, collection)

    @pytest.fixture
    def V(self, elements):
        return coordinate.MfemSpace(elements)


class TestMatrixSpace(_OnAnInterval):
    def test_vectors_are_not_arrays(self, V):
        """Which is the point: the library never touches them directly."""
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


class TestMatrixForms(_OnAnInterval):
    def test_a_linear_form_is_a_derivative(self, V, elements, rng):
        load = mfem.LinearForm(elements)
        load.AddDomainIntegrator(mfem.DomainLFIntegrator(mfem.ConstantCoefficient(1.0)))
        load.Assemble()
        functional = coordinate.functional_from_linear_form(V, load)

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

        A = coordinate.operator_from_bilinear_form(
            V, form, traits=Traits.POSITIVE_DEFINITE
        )
        stiffness = coordinate._to_scipy(form.SpMat()).toarray()

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
        A = coordinate.operator_from_bilinear_form(
            V, form, traits=Traits.POSITIVE_DEFINITE
        )

        b = V.random(rng=rng)
        result = CGSolver(rtol=1e-12)(A).solve(b)
        assert result.converged
        assert V.norm(V.subtract(A(result.solution), b)) < 1e-8 * V.norm(b)

        stiffness = coordinate._to_scipy(form.SpMat()).toarray()
        direct = np.linalg.solve(stiffness, V.gram_matrix() @ V.to_components(b))
        assert np.allclose(V.to_components(result.solution), direct, atol=1e-8)


def square(order=2, divisions=6):
    """A 2-D H1 space, which is where boundary conditions become interesting."""
    mesh = mfem.Mesh.MakeCartesian2D(divisions, divisions, mfem.Element.QUADRILATERAL)
    collection = mfem.H1_FECollection(order, mesh.Dimension())
    return mfem.FiniteElementSpace(mesh, collection)


class TestMatrixEssentialBoundaryConditions:
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
        dofs = coordinate.essential_dofs_of(elements)
        return elements, dofs, coordinate.MfemSpace(elements, essential_dofs=dofs)

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

        operator = coordinate.operator_from_bilinear_form(
            space, form, traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
        )
        check_operator(operator, rng=rng)
        check_traits(operator, rng=rng)

        # Unconstrained, the same form is singular: a constant function is in
        # its kernel, so the smallest eigenvalue of the Galerkin matrix is zero.
        whole = coordinate.MfemSpace(elements)
        # restrict keeps the matrix sparse, so densify here rather than there.
        matrix = np.asarray(whole.restrict(form.SpMat()).todense())
        assert np.linalg.eigvalsh(0.5 * (matrix + matrix.T)).min() < 1e-10
        restricted = np.asarray(space.restrict(form.SpMat()).todense())
        assert np.linalg.eigvalsh(0.5 * (restricted + restricted.T)).min() > 1e-6

    def test_a_load_vector_is_restricted_too(self, constrained):
        elements, dofs, space = constrained
        form = mfem.LinearForm(elements)
        form.AddDomainIntegrator(mfem.DomainLFIntegrator(mfem.ConstantCoefficient(1.0)))
        form.Assemble()
        functional = coordinate.functional_from_linear_form(space, form)
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
        every = coordinate.essential_dofs_of(elements)
        one_side = coordinate.essential_dofs_of(elements, attributes=[1])
        assert 0 < one_side.size < every.size
        assert set(one_side.tolist()) <= set(every.tolist())
        assert (
            coordinate.MfemSpace(elements, essential_dofs=one_side).dim
            > coordinate.MfemSpace(elements, essential_dofs=every).dim
        )

    def test_the_unconstrained_space_is_unchanged(self, rng):
        """The default is exactly what it was before boundary conditions
        existed, so nothing that did not ask for one is affected."""
        elements = square()
        space = coordinate.MfemSpace(elements)
        assert not space.is_constrained
        assert space.dim == elements.GetTrueVSize()
        assert space.essential_dofs.size == 0
        check_space(space, rng=rng)

    def test_two_boundary_conditions_are_two_spaces(self):
        """They must not compare equal, or an operator built for one would be
        accepted by the other."""
        elements = square()
        whole = coordinate.MfemSpace(elements)
        part = coordinate.MfemSpace(
            elements, essential_dofs=coordinate.essential_dofs_of(elements)
        )
        assert whole != part
        assert coordinate.MfemSpace(elements) == whole

    def test_nonsense_is_refused(self):
        elements = square()
        with pytest.raises(ValueError, match="must lie in"):
            coordinate.MfemSpace(elements, essential_dofs=[elements.GetTrueVSize()])
        with pytest.raises(ValueError, match="dimension zero"):
            coordinate.MfemSpace(
                elements, essential_dofs=range(elements.GetTrueVSize())
            )
        with pytest.raises(ValueError, match="out of range"):
            coordinate.essential_dofs_of(elements, attributes=[99])


class TestMatrixSolvingWithMfem:
    """The PDE solve left to MFEM and wrapped as one of our operators.

    The division of labor: MFEM assembles, preconditions and solves; this
    library says what the result *is* — an operator in the right metric, with
    an adjoint, that composes. So the tests are that it really is the inverse,
    that it really is in the metric, and that MFEM's objects come back
    unharmed.
    """

    @pytest.fixture
    def constrained(self):
        elements = square()
        space = coordinate.MfemSpace(
            elements, essential_dofs=coordinate.essential_dofs_of(elements)
        )
        form = mfem.BilinearForm(elements)
        form.AddDomainIntegrator(mfem.DiffusionIntegrator())
        form.Assemble()
        form.Finalize()
        return elements, space, form

    def test_it_inverts_the_operator_the_form_defines(self, constrained, rng):
        _, space, form = constrained
        operator = coordinate.operator_from_bilinear_form(
            space, form, traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
        )
        inverse = coordinate.solver_from_bilinear_form(space, form, rtol=1e-12)
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
        operator = coordinate.operator_from_bilinear_form(
            space, form, traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
        )
        theirs = coordinate.solver_from_bilinear_form(space, form, rtol=1e-12)
        ours = CGSolver(rtol=1e-12)(operator)
        for _ in range(4):
            vector = space.random(rng=rng)
            expected = ours(vector)
            assert space.norm(
                space.subtract(theirs(vector), expected)
            ) == pytest.approx(0.0, abs=1e-8 * space.norm(expected))

    def test_it_is_an_operator_like_any_other(self, constrained, rng):
        _, space, form = constrained
        inverse = coordinate.solver_from_bilinear_form(space, form, rtol=1e-12)
        check_operator(inverse, rng=rng)
        check_traits(inverse, rng=rng)

    def test_the_form_it_was_given_is_left_alone(self, constrained):
        """MFEM's own ``FormSystemMatrix`` takes ownership of the form's
        matrix, so reading it afterwards is a use-after-free — a segfault, not
        an exception. This route does not use it, and the test says so."""
        _, space, form = constrained
        before = coordinate._to_scipy(form.SpMat()).toarray().copy()
        coordinate.solver_from_bilinear_form(space, form, rtol=1e-12)
        assert np.allclose(coordinate._to_scipy(form.SpMat()).toarray(), before)

    def test_an_unfinalized_form_is_refused_rather_than_crashing(self):
        """Before ``Finalize`` there are no CSR arrays to read, and reading
        them anyway segfaults with no traceback."""
        elements = square()
        space = coordinate.MfemSpace(elements)
        form = mfem.BilinearForm(elements)
        form.AddDomainIntegrator(mfem.MassIntegrator())
        form.Assemble()
        assert not form.SpMat().Finalized()
        with pytest.raises(ValueError, match="not been finalized"):
            coordinate.operator_from_bilinear_form(space, form)

    def test_a_solve_that_does_not_converge_is_reported(self, constrained, rng):
        """An unconverged solve inside an operator is an operator that is
        quietly not the one it claims to be."""
        _, space, form = constrained
        inverse = coordinate.solver_from_bilinear_form(
            space, form, rtol=1e-16, max_iterations=1
        )
        with pytest.raises(ConvergenceError, match="did not converge"):
            inverse(space.random(rng=rng))


class TestMatrixObservationOperators:
    """Linear forms stacked into an observation operator."""

    def test_the_rows_are_the_functionals_they_came_from(self, rng):
        elements = square()
        space = coordinate.MfemSpace(
            elements, essential_dofs=coordinate.essential_dofs_of(elements)
        )
        forms = []
        for value in (1.0, 2.5):
            form = mfem.LinearForm(elements)
            form.AddDomainIntegrator(
                mfem.DomainLFIntegrator(mfem.ConstantCoefficient(value))
            )
            form.Assemble()
            forms.append(form)

        operator = coordinate.operator_from_linear_forms(space, forms)
        assert operator.codomain.dim == 2
        vector = space.random(rng=rng)
        observed = operator.codomain.to_components(operator(vector))
        for index, form in enumerate(forms):
            functional = coordinate.functional_from_linear_form(space, form)
            assert observed[index] == pytest.approx(functional(vector))

    def test_its_adjoint_carries_the_metric(self, rng):
        """The whole reason the rows are derivative components. An adjoint that
        forgot the mass solve would still be a linear map, still have the right
        shape, and be wrong."""
        elements = square()
        space = coordinate.MfemSpace(
            elements, essential_dofs=coordinate.essential_dofs_of(elements)
        )
        form = mfem.LinearForm(elements)
        form.AddDomainIntegrator(mfem.DomainLFIntegrator(mfem.ConstantCoefficient(1.0)))
        form.Assemble()
        operator = coordinate.operator_from_linear_forms(space, [form])
        check_operator(operator, rng=rng)

    def test_nonsense_is_refused(self):
        elements = square()
        space = coordinate.MfemSpace(elements)
        with pytest.raises(ValueError, match="At least one"):
            coordinate.operator_from_linear_forms(space, [])


class TestMatrixWhiteNoise:
    """White noise through MFEM rather than through a factorization."""

    def test_the_load_has_the_mass_matrix_as_its_covariance(self):
        """Which is the whole content of a finite element white noise: the
        right-hand side of ``(W, phi_i)`` has covariance ``(phi_i, phi_j)``,
        not the identity. Slow because it is a Monte Carlo statement."""
        elements = square(order=1, divisions=4)
        space = coordinate.MfemSpace(elements)
        rng = np.random.default_rng(0)
        draws = np.stack(
            [coordinate.white_noise_load(space, rng=rng) for _ in range(20000)]
        )
        mass = coordinate._to_scipy(space.mass_matrix).toarray()
        scale = np.abs(mass).max()
        assert np.abs(np.cov(draws.T) - mass).max() < 0.08 * scale
        # And emphatically not the identity, which is the mistake it prevents.
        assert np.abs(np.cov(draws.T) - np.identity(space.dim)).max() > scale

    def test_the_load_is_a_copy_and_survives_its_form(self, rng):
        """The form that assembled it is long gone by the time the caller sees
        the array, and MFEM does not keep it alive."""
        space = coordinate.MfemSpace(square(order=1, divisions=4))
        first = coordinate.white_noise_load(space, rng=rng)
        for _ in range(50):
            coordinate.white_noise_load(space, rng=rng)
        assert np.isfinite(first).all()
        assert np.abs(first).max() < 10.0

    @pytest.mark.parametrize("constrained", [False, True])
    def test_components_are_white_noise_on_the_space(self, constrained):
        elements = square(order=1, divisions=4)
        space = (
            coordinate.MfemSpace(
                elements, essential_dofs=coordinate.essential_dofs_of(elements)
            )
            if constrained
            else coordinate.MfemSpace(elements)
        )
        check_white_noise(space, rng=np.random.default_rng(1), samples=20000, rtol=0.2)


class TestMatrixMaternMeasure:
    """The Lindgren SPDE method, wrapped from MFEM's own pieces."""

    @pytest.fixture(scope="class")
    @classmethod
    def field(cls):
        mesh = mfem.Mesh.MakeCartesian2D(32, 32, mfem.Element.QUADRILATERAL)
        elements = mfem.FiniteElementSpace(mesh, mfem.H1_FECollection(1, 2))
        space = coordinate.MfemSpace(elements)
        coordinates = np.array(mesh.GetVertexArray())
        interior = np.all((coordinates > 0.3) & (coordinates < 0.7), axis=1)
        return space, coordinates, interior

    def test_it_is_a_gaussian_measure_with_all_three_pieces(self, field):
        """Covariance, factor *and* precision — so it can be sampled, and used
        in a model-space formalism, without anything being formed."""
        space, _, _ = field
        measure = coordinate.matern_measure(
            space, smoothness=1.0, correlation_length=0.2
        )
        assert measure.domain == space
        assert measure.can_sample
        assert measure.covariance is not None
        assert measure.precision is not None

    def test_the_precision_inverts_the_covariance(self, field, rng):
        """They are built from opposite ends — the precision from the assembled
        operator, the covariance from MFEM's solve — so agreement is a
        statement about the wrapper rather than an identity by construction."""
        space, _, _ = field
        measure = coordinate.matern_measure(
            space, smoothness=1.0, correlation_length=0.2
        )
        for _ in range(3):
            vector = space.random(rng=rng)
            recovered = measure.precision(measure.covariance(vector))
            assert space.norm(space.subtract(recovered, vector)) == pytest.approx(
                0.0, abs=1e-6 * space.norm(vector)
            )

    @pytest.mark.parametrize("smoothness", [1.0, 3.0])
    def test_the_pointwise_variance_is_one_away_from_the_boundary(
        self, field, smoothness
    ):
        """MFEM's normalization coefficient, checked rather than trusted. Away
        from the boundary, because the SPDE on a bounded domain is not
        stationary near it — which the docstring says and this respects."""
        space, _, interior = field
        measure = coordinate.matern_measure(
            space, smoothness=smoothness, correlation_length=0.15
        )
        rng = np.random.default_rng(0)
        draws = np.stack(
            [space.to_components(measure.sample(rng=rng)) for _ in range(2000)]
        )
        assert draws[:, interior].var(axis=0).mean() == pytest.approx(1.0, abs=0.1)

    def test_the_correlation_is_the_matern_one(self, field):
        """Against the analytic formula, Bessel function and all. Nothing in
        the implementation knows that formula, so this is independent."""
        from scipy.special import gamma, kv

        space, coordinates, interior = field
        smoothness, length = 1.0, 0.15
        measure = coordinate.matern_measure(
            space, smoothness=smoothness, correlation_length=length
        )
        rng = np.random.default_rng(0)
        draws = np.stack(
            [space.to_components(measure.sample(rng=rng)) for _ in range(3000)]
        )

        chosen = np.flatnonzero(interior)[:150]
        correlation = np.corrcoef(draws[:, chosen].T)
        separation = np.linalg.norm(
            coordinates[chosen][:, None, :] - coordinates[chosen][None, :, :],
            axis=-1,
        )
        upper = np.triu_indices(len(chosen), 1)
        distances, values = separation[upper], correlation[upper]

        def matern(radius):
            scaled = np.sqrt(2.0 * smoothness) / length * radius
            return (
                2.0 ** (1 - smoothness)
                / gamma(smoothness)
                * scaled**smoothness
                * kv(smoothness, scaled)
            )

        for low, high in [(0.05, 0.09), (0.09, 0.13), (0.15, 0.21)]:
            band = (distances >= low) & (distances < high)
            assert band.sum() > 30
            middle = 0.5 * (low + high)
            assert values[band].mean() == pytest.approx(matern(middle), abs=0.06)

    def test_anisotropy_stretches_the_field(self, field):
        space, coordinates, interior = field
        measure = coordinate.matern_measure(
            space, smoothness=1.0, correlation_length=[0.30, 0.05]
        )
        rng = np.random.default_rng(1)
        draws = np.stack(
            [space.to_components(measure.sample(rng=rng)) for _ in range(2500)]
        )
        chosen = np.flatnonzero(interior)
        correlation = np.corrcoef(draws[:, chosen].T)
        offsets = coordinates[chosen][:, None, :] - coordinates[chosen][None, :, :]
        upper = np.triu_indices(len(chosen), 1)
        across = np.abs(offsets[..., 0][upper])
        along = np.abs(offsets[..., 1][upper])
        values = correlation[upper]
        horizontal = (along < 0.02) & (across > 0.08) & (across < 0.13)
        vertical = (across < 0.02) & (along > 0.08) & (along < 0.13)
        assert values[horizontal].mean() > 0.6
        assert values[vertical].mean() < 0.3

    def test_a_fractional_exponent_is_refused_with_a_reason(self, field):
        """MFEM reaches it with a rational approximation its Python bindings do
        not expose. Refusing beats reimplementing it badly."""
        space, _, _ = field
        with pytest.raises(ValueError, match="not a positive integer"):
            coordinate.matern_measure(space, smoothness=2.0)

    def test_bad_parameters_are_refused(self, field):
        space, _, _ = field
        with pytest.raises(ValueError, match="must be positive"):
            coordinate.matern_measure(space, smoothness=-1.0)
        with pytest.raises(ValueError, match="must be positive"):
            coordinate.matern_measure(space, correlation_length=0.0)
        with pytest.raises(ValueError, match="correlation lengths for"):
            coordinate.matern_measure(space, correlation_length=[0.1, 0.2, 0.3])


class TestMatrixSparsitySurvives:
    """A finite element matrix is sparse, and must stay sparse.

    The one property nothing else in the suite pins, which is how ``restrict``
    came to call ``.toarray()`` unconditionally and go unnoticed: every test
    ran on a mesh small enough that a dense block was merely wasteful. At 1e5
    degrees of freedom it is 80 GB.
    """

    @pytest.fixture
    def poisson(self):
        mesh = mfem.Mesh.MakeCartesian2D(30, 30, mfem.Element.QUADRILATERAL)
        collection = mfem.H1_FECollection(1, mesh.Dimension())
        elements = mfem.FiniteElementSpace(mesh, collection)
        form = mfem.BilinearForm(elements)
        form.AddDomainIntegrator(mfem.DiffusionIntegrator())
        form.Assemble()
        form.Finalize()
        return elements, form

    def test_restrict_keeps_the_matrix_sparse(self, poisson):
        import scipy.sparse as sp

        elements, form = poisson
        space = coordinate.MfemSpace(elements)
        restricted = space.restrict(form.SpMat())

        assert sp.issparse(restricted)
        size = restricted.shape[0]
        # A P1 Laplacian on a quadrilateral mesh has at most nine entries a row.
        assert restricted.nnz < 10 * size
        assert restricted.nnz < 0.05 * size * size

    def test_the_assembled_operator_holds_the_sparse_matrix(self, poisson):
        import scipy.sparse as sp

        from pygeoinf.algebra.operators import MatrixLinearOperator

        elements, form = poisson
        space = coordinate.MfemSpace(elements)
        operator = coordinate.operator_from_bilinear_form(
            space, form, traits=Traits.SELF_ADJOINT | Traits.POSITIVE_SEMIDEFINITE
        )

        assert isinstance(operator, MatrixLinearOperator)
        assert sp.issparse(operator.stored_matrix)
        assert operator.stored_matrix.nnz < 10 * space.dim


# --------------------------------------------------------------------- #
#               The Hilbert-space form: no coordinates at all            #
#                              (hilbert_space)                           #
# --------------------------------------------------------------------- #


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
    return coordinate.essential_dofs_of(elements)


@pytest.fixture(params=["full", "partial"])
def assembly(request):
    return request.param


@pytest.fixture
def V(elements, essential, assembly):
    """The constrained space, in both assembly modes."""
    return opaque.MfemHilbertSpace(
        elements, essential_dofs=essential, assembly=assembly
    )


@pytest.fixture
def reference(elements, essential):
    """The matrix form on the same element space and constraint."""
    return coordinate.MfemSpace(elements, essential_dofs=essential)


def bilinear(elements, assembly, *integrators):
    """An assembled form at the given level. Not finalized: FormSystemMatrix
    does that itself, and the Hilbert-space form never reads ``SpMat``."""
    form = mfem.BilinearForm(elements)
    if assembly == "partial":
        form.SetAssemblyLevel(mfem.AssemblyLevel_PARTIAL)
    for integrator in integrators:
        form.AddDomainIntegrator(integrator)
    form.Assemble()
    return form


def finalized(elements, *integrators):
    """The same form for the matrix form, which reads the matrix."""
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


class TestHilbertSpace:
    def test_it_is_a_hilbert_space_and_not_a_coordinate_space(self, V):
        assert isinstance(V, MassWeightedSpace)
        assert not isinstance(V, CoordinateSpace)
        assert isinstance(V.dof_space, opaque.MfemDofSpace)
        assert type(V.zero()).__name__ == "Vector"

    def test_the_axioms_hold(self, V, rng):
        check_space(V, rng=rng)

    def test_the_dof_space_axioms_hold_too(self, V, rng):
        check_space(V.dof_space, rng=rng)

    def test_the_inner_product_is_the_matrix_forms(self, V, reference, rng):
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
        a = opaque.MfemHilbertSpace(
            elements, essential_dofs=essential, assembly=assembly
        )
        b = opaque.MfemHilbertSpace(
            elements, essential_dofs=essential, assembly=assembly
        )
        assert a == b and hash(a) == hash(b)
        assert a.shares_vectors_with(b.dof_space)
        assert a != opaque.MfemHilbertSpace(elements, assembly=assembly)

    def test_no_matrix_can_be_formed(self, V):
        A = LinearOperator.identity(V)
        with pytest.raises(TypeError, match="no coordinate map"):
            A.matrix()

    def test_nonsense_is_refused(self, elements):
        with pytest.raises(ValueError, match="assembly"):
            opaque.MfemHilbertSpace(elements, assembly="sparse")
        with pytest.raises(ValueError, match="out of range|must lie"):
            opaque.MfemHilbertSpace(elements, essential_dofs=[elements.GetTrueVSize()])
        with pytest.raises(ValueError, match="dimension zero"):
            opaque.MfemHilbertSpace(
                elements, essential_dofs=range(elements.GetTrueVSize())
            )


class TestHilbertWhiteNoise:
    def test_it_is_white_on_the_space(self, V, rng):
        check_white_noise(V, rng=rng, samples=3000, rtol=0.2)

    def test_the_load_has_the_mass_operator_as_covariance(self, V, reference):
        """Against the matrix form's Gram matrix, which is ``M``'s free
        block: the one place a matrix appears in these tests of the
        Hilbert-space form, and it is the other form's."""
        rng = np.random.default_rng(11)
        free = reference.free_dofs
        loads = np.array(
            [values(opaque.white_noise_load(V, rng=rng))[free] for _ in range(4000)]
        )
        covariance = loads.T @ loads / len(loads)
        gram = reference.gram_matrix()
        scale = np.abs(np.diag(gram)).max()
        assert np.abs(covariance - gram).max() < 0.1 * scale


class TestHilbertOperators:
    def test_a_symmetric_form_gives_the_matrix_forms_operator(
        self, V, reference, elements, assembly, rng
    ):
        A = opaque.operator_from_bilinear_form(
            V,
            bilinear(
                elements, assembly, mfem.DiffusionIntegrator(), mfem.MassIntegrator()
            ),
            traits=DEFINITE,
        )
        B = coordinate.operator_from_bilinear_form(
            reference,
            finalized(elements, mfem.DiffusionIntegrator(), mfem.MassIntegrator()),
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
        A = opaque.operator_from_bilinear_form(
            V, bilinear(elements, assembly, convection(), mfem.MassIntegrator())
        )
        B = coordinate.operator_from_bilinear_form(
            reference, finalized(elements, convection(), mfem.MassIntegrator())
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
        A = opaque.operator_from_bilinear_form(V, form, traits=DEFINITE)
        x = V.random(rng=rng)
        before = values(A(x))
        del form
        gc.collect()
        assert np.allclose(values(A(x)), before)

    def test_partial_assembly_is_full_assembly_without_the_matrix(
        self, elements, essential, rng
    ):
        spaces = {
            level: opaque.MfemHilbertSpace(
                elements, essential_dofs=essential, assembly=level
            )
            for level in ("full", "partial")
        }
        results = {}
        for level, space in spaces.items():
            A = opaque.operator_from_bilinear_form(
                space,
                bilinear(elements, level, mfem.DiffusionIntegrator()),
                traits=DEFINITE,
            )
            results[level] = values(A(space.random(rng=np.random.default_rng(3))))
        assert np.allclose(results["full"], results["partial"])


class TestHilbertSolvingWithMfem:
    def test_it_inverts_the_operator_the_form_defines(self, V, elements, assembly, rng):
        system = V.new_system(bilinear(elements, assembly, mfem.DiffusionIntegrator()))
        A = opaque.operator_from_bilinear_form(V, system, traits=DEFINITE)
        S = opaque.solver_from_bilinear_form(V, system, rtol=1e-13)
        w = V.random(rng=rng)
        assert V.norm(V.subtract(A(S(w)), w)) < 1e-9 * V.norm(w)
        assert V.norm(V.subtract(S(A(w)), w)) < 1e-9 * V.norm(w)
        check_operator(S, rng=rng)

    def test_it_agrees_with_solving_here_by_cg(self, V, elements, assembly, rng):
        """The core's coordinate-free CG on the wrapped operator against
        MFEM's on its own system: same operator, two Krylov loops."""
        system = V.new_system(bilinear(elements, assembly, mfem.DiffusionIntegrator()))
        A = opaque.operator_from_bilinear_form(V, system, traits=DEFINITE)
        S = opaque.solver_from_bilinear_form(V, system, rtol=1e-13)
        w = V.random(rng=rng)
        ours = CGSolver(rtol=1e-12)(A).solve(w).solution
        assert V.norm(V.subtract(ours, S(w))) < 1e-8 * V.norm(w)

    def test_a_system_from_another_space_is_refused(
        self, elements, essential, assembly
    ):
        a = opaque.MfemHilbertSpace(
            elements, essential_dofs=essential, assembly=assembly
        )
        b = opaque.MfemHilbertSpace(elements, assembly=assembly)
        system = a.new_system(bilinear(elements, assembly, mfem.DiffusionIntegrator()))
        with pytest.raises(ValueError, match="different space"):
            opaque.operator_from_bilinear_form(b, system)

    def test_a_solve_that_does_not_converge_is_reported(
        self, V, elements, assembly, rng
    ):
        S = opaque.solver_from_bilinear_form(
            V,
            bilinear(elements, assembly, mfem.DiffusionIntegrator()),
            max_iterations=1,
        )
        with pytest.raises(ConvergenceError, match="did not converge"):
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
        A = opaque.operator_from_bilinear_form(V, system, traits=DEFINITE)
        S = opaque.solver_from_bilinear_form(V, system, make_solver=make)
        w = V.random(rng=rng)
        assert len(seen) == 1
        assert V.norm(V.subtract(A(S(w)), w)) < 1e-8 * V.norm(w)


class TestHilbertFunctionalsAndObservations:
    def test_a_linear_form_is_a_derivative(self, V, reference, elements, rng):
        form = linear(elements, mfem.ConstantCoefficient(1.0))
        f = opaque.functional_from_linear_form(V, form)
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
        A = opaque.operator_from_linear_forms(V, forms)
        B = coordinate.operator_from_linear_forms(reference, forms)
        x = V.random(rng=rng)
        assert np.allclose(A(x), B(x))
        for row, form in zip(A(x), forms):
            assert row == pytest.approx(opaque.functional_from_linear_form(V, form)(x))

    def test_the_adjoint_carries_the_metric(self, V, reference, elements, rng):
        forms = [linear(elements, mfem.ConstantCoefficient(c)) for c in (0.5, 2.0)]
        A = opaque.operator_from_linear_forms(V, forms)
        B = coordinate.operator_from_linear_forms(reference, forms)
        check_operator(A, rng=rng)
        d = rng.normal(size=2)
        assert np.allclose(values(A.adjoint(d)), values(B.adjoint(d)), atol=1e-10)

    def test_nonsense_is_refused(self, V, elements):
        from pygeoinf.algebra.spaces import EuclideanSpace

        with pytest.raises(ValueError, match="At least one"):
            opaque.operator_from_linear_forms(V, [])
        with pytest.raises(ValueError, match="data space of dimension"):
            opaque.operator_from_linear_forms(
                V,
                [linear(elements, mfem.ConstantCoefficient(1.0))],
                codomain=EuclideanSpace(3),
            )


class TestHilbertMaternMeasure:
    def test_it_matches_the_matrix_forms_field(self, V, reference, rng):
        ours = opaque.matern_measure(V, smoothness=1.0, correlation_length=0.2)
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
        field = opaque.matern_measure(V, smoothness=1.0, correlation_length=0.2)
        x = V.random(rng=rng)
        back = field.precision(field.covariance(x))
        assert V.norm(V.subtract(back, x)) < 1e-7 * V.norm(x)

    def test_it_samples_on_the_space(self, V, essential, rng):
        field = opaque.matern_measure(V, smoothness=1.0, correlation_length=0.2)
        draw = field.sample(rng=rng)
        assert type(draw).__name__ == "Vector"
        assert np.all(values(draw)[essential] == 0.0)
        assert V.norm(draw) > 0.0

    def test_anisotropy_needs_a_matrix_coefficient_and_full_assembly(
        self, elements, essential, reference, rng
    ):
        full = opaque.MfemHilbertSpace(
            elements, essential_dofs=essential, assembly="full"
        )
        ours = opaque.matern_measure(
            full, smoothness=1.0, correlation_length=(0.3, 0.1)
        )
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
        partial = opaque.MfemHilbertSpace(
            elements, essential_dofs=essential, assembly="partial"
        )
        with pytest.raises(ValueError, match="partial"):
            opaque.matern_measure(
                partial, smoothness=1.0, correlation_length=(0.3, 0.1)
            )

    def test_bad_parameters_are_refused(self, V):
        with pytest.raises(ValueError, match="positive integer"):
            opaque.matern_measure(V, smoothness=0.5)
        with pytest.raises(ValueError, match="positive"):
            opaque.matern_measure(V, correlation_length=-1.0)


class TestHilbertWholeInversion:
    """The Hilbert-space form end to end, against the matrix form.

    Sensors, a PDE solved by MFEM, a Matern prior and a Bayesian inversion
    — the shape of part C — on the space with no coordinates. The
    posterior mean must agree with the one the matrix form produces
    from the same data, and the inversion must never have asked for a
    matrix, which it cannot have because there is none to give.
    """

    @pytest.fixture
    def positions(self):
        return [(x, y) for x in (0.3, 0.7) for y in (0.3, 0.7)]

    @staticmethod
    def sensors(elements, positions):
        class Bump(mfem.PyCoefficient):
            def __init__(self, center):
                super().__init__()
                self._center = np.asarray(center, float)

            def EvalValue(self, x):
                offset = np.asarray([x[0], x[1]]) - self._center
                return float(np.exp(-0.5 * offset @ offset / 0.1**2))

        return [linear(elements, Bump(center)) for center in positions]

    def test_the_posterior_mean_agrees_with_the_matrix_form(
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
            finalized(elements, mfem.DiffusionIntegrator()),
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
    def inversion(space, module, forms, stiffness):
        class Setup:
            pass

        setup = Setup()
        solve = module.solver_from_bilinear_form(space, stiffness, rtol=1e-12)
        forward = module.operator_from_linear_forms(space, forms) @ solve
        setup.prior = module.matern_measure(
            space, smoothness=1.0, correlation_length=0.2
        )
        setup.problem = LinearForwardProblem(
            forward,
            error=GaussianMeasure.from_standard_deviation(forward.codomain, 1e-3),
        )
        setup.inversion = LinearGaussianInversion(setup.problem, setup.prior)
        return setup


# --------------------------------------------------------------------- #
#           An operator on the library's own box, through MFEM          #
#                              (box_operator)                            #
# --------------------------------------------------------------------- #


def a_box(grid=16, order=2.0, length_scale=0.1):
    from pygeoinf.symmetric_space.torus import Sobolev

    return Sobolev((grid, grid), order, length_scale, lengths=(1.0, 1.0))


class TestBoxOperator:
    """The screened Poisson solve on a periodic box, with MFEM inside."""

    @pytest.fixture(scope="class")
    @classmethod
    def box(cls):
        return _module("src.box_operator").ScreenedPoisson(
            a_box(), elements=12, screening_length=0.05
        )

    @pytest.fixture(scope="class")
    @classmethod
    def prior(cls, box):
        """The probe fields' measure, part E's prior: a Sobolev measure of
        order two on the box's H^2."""
        return box.space.sobolev_measure(2.0, 0.1, pointwise_std=1.0)

    def test_the_library_sees_no_mfem_space(self, box):
        """The operator is on the box; the steps between pass plain arrays."""
        operator = box.operator
        assert operator.domain == box.space and operator.codomain == box.space
        assert box.sampling.codomain == EuclideanSpace(box.size)
        assert box.evaluation.domain == EuclideanSpace(box.size)
        assert isinstance(operator(box.space.zero()), np.ndarray)

    def test_each_step_has_its_adjoint(self, box, prior, rng):
        for step, measure in (
            (box.sampling, prior),
            (box.solve, None),
            (box.evaluation, None),
            (box.operator, prior),
        ):
            report = check_operator(step, rng=rng, measure=measure)
            assert report.worst < 1e-9

    def test_the_solve_is_symmetric_and_positive_definite(self, box, rng):
        required = Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
        assert box.solve.traits & required == required
        check_traits(box.solve, rng=rng)

    def test_it_approximates_the_multiplier(self, box, prior, rng):
        f = prior.sample(rng=rng)
        exact = box.continuum(f)
        difference = np.linalg.norm(box.operator(f) - exact) / np.linalg.norm(exact)
        assert difference < 0.05

    def test_forgetting_the_metric_is_caught(self, box, prior, rng):
        """The component matrices' transpose, without the box's Gram factors,
        is not the adjoint on a Sobolev box."""
        space = box.space

        def transpose(y):
            load = box.evaluation_matrix.T @ (box.cell_volume * np.asarray(y).ravel())
            return space.from_components(
                space.accumulate(
                    box.load_matrix.T @ box.solve(load), box.quadrature_points
                )
            )

        wrong = LinearOperator.from_callables(
            space, space, box.operator, adjoint=transpose
        )
        with pytest.raises(AssertionError, match="adjoint"):
            check_operator(wrong, rng=rng, measure=prior)

    def test_an_unconverged_solve_is_reported(self):
        box = _module("src.box_operator").ScreenedPoisson(
            a_box(), elements=8, screening_length=0.05, max_iterations=1
        )
        with pytest.raises(ConvergenceError, match="did not reach"):
            box.solve(np.ones(box.size))

    def test_nonsense_is_refused(self):
        from pygeoinf.symmetric_space.circle import Sobolev as OnACircle

        ScreenedPoisson = _module("src.box_operator").ScreenedPoisson
        with pytest.raises(ValueError, match="two-dimensional"):
            ScreenedPoisson(OnACircle(16, 2.0, 0.1), elements=8, screening_length=0.05)
        with pytest.raises(ValueError, match="two elements"):
            ScreenedPoisson(a_box(), elements=1, screening_length=0.05)
        with pytest.raises(ValueError, match="screening length"):
            ScreenedPoisson(a_box(), elements=8, screening_length=0.0)
        with pytest.raises(ValueError, match="order"):
            ScreenedPoisson(a_box(), elements=8, screening_length=0.05, order=0)


# --------------------------------------------------------------------- #
#                       The parts, at the quick scale                    #
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def scale(tmp_path_factory):
    """The quick preset, built through the scripts' own command line with
    options given at their preset values."""
    common = _module("src.common")
    quick = common.QUICK
    arguments = common.parse(
        "the quick scale",
        quick=quick,
        full=common.FULL,
        argv=[
            "--quick",
            "--noise-std",
            str(quick.noise_std),
            "--orders",
            *map(str, quick.orders),
            "--outdir",
            str(tmp_path_factory.mktemp("mfem_options")),
        ],
    )
    assert arguments.scale == quick
    return arguments.scale


def test_every_field_of_the_scale_is_an_option(tmp_path):
    common = _module("src.common")
    arguments = common.parse(
        "",
        quick=common.QUICK,
        full=common.FULL,
        argv=[
            "--box-noise-std",
            "0.1",
            "--orders",
            "2",
            "3",
            "--outdir",
            str(tmp_path),
        ],
    )
    assert arguments.scale == dataclasses.replace(
        common.FULL, box_noise_std=0.1, orders=(2, 3)
    )


@pytest.fixture(scope="module")
def results(scale, tmp_path_factory):
    """Every part at the quick scale, once, each through its own entry point."""
    outdir = tmp_path_factory.mktemp("mfem")
    return {
        part: _module(name).run(scale, outdir=outdir) for part, name in PARTS.items()
    }, outdir


def test_a_the_hilbert_space_form_agrees_with_the_matrix_form(results):
    part = results[0]["a"]
    assert part["space"].worst < 1e-12
    assert "no coordinate map" in part["refused"]
    assert part["inner_difference"] < 1e-12
    assert part["transport"].worst < 1e-9
    assert part["traits"].worst < 1e-9
    assert part["inverse_residual"] < 1e-8
    assert part["boundary"] == 0.0
    assert part["mean_difference"] < 1e-6
    assert part["estimate"] == pytest.approx(part["their_estimate"], abs=1e-8)
    assert part["spread"] == pytest.approx(part["their_spread"], rel=1e-6)
    assert abs(part["estimate"] - part["actual"]) < 4.0 * part["spread"]


def test_b_the_matrix_form_keeps_derivative_and_gradient_apart(results):
    part = results[0]["b"]
    assert not part["orthonormal"]
    assert part["space"].worst < 1e-12 and part["coordinates"].worst < 1e-12
    assert part["action_defect"] < 1e-12 and part["mass_solve_defect"] < 1e-12
    assert part["representer_differs"] > 0.1
    assert part["operator"].worst < 1e-12
    assert part["agreement"] < 1e-10


def test_c_the_inverse_problem_on_the_matrix_form(results):
    part = results[0]["c"]
    assert part["traits"].worst < 1e-9
    assert part["boundary"] == 0.0
    assert abs(part["estimate"] - part["actual"]) < 4.0 * part["spread"]
    # The lowest order is the most confident, which is not the same as right.
    deviations = [deviation for *_, deviation in part["orders"]]
    assert deviations[0] < min(deviations[1:])


def test_d_the_ladder_proves_the_box_operators_adjoint(results):
    part = results[0]["d"]
    adjoint = "the adjoint satisfies (A x, y) == (x, A* y)"
    for name in ("sampling", "evaluation"):
        assert part["ladder"][name].defects[adjoint] < 1e-13, name
    assert part["ladder"]["solve"].defects[adjoint] < 1e-12
    assert part["ladder"]["operator"].defects[adjoint] < 1e-9
    # Without the Gram factors, refused; with them, at every mesh, held.
    assert "adjoint" in part["refusal"] and part["plain_defect"] > 1e-2
    misaligned = [row for row in part["refinement"] if not row["aligned"]]
    assert all(row["adjoint"] < 1e-9 for row in part["refinement"])
    # The discretization error, and the continuum's adjoint with it, fall
    # under refinement; on a mesh aligned with the grid the operator is a
    # Fourier multiplier and its own adjoint to rounding.
    differences = [row["l2"] for row in misaligned]
    assert differences == sorted(differences, reverse=True)
    assert (
        misaligned[-1]["continuum adjoint"] < 0.1 * misaligned[0]["continuum adjoint"]
    )
    (aligned,) = [row for row in part["refinement"] if row["aligned"]]
    assert aligned["continuum adjoint"] < 1e-9
    # The true adjoint's defect follows MFEM's tolerance down.
    defects = [defect for _, defect, _ in part["tolerances"]]
    assert defects == sorted(defects, reverse=True)
    assert defects[0] > 1e3 * defects[-1]


def test_e_three_answers_on_the_box(results):
    part = results[0]["e"]
    truth = part["true_averages"]
    # The minimum-norm model meets the discrepancy principle.
    assert part["smallest_chi_squared"] == pytest.approx(part["critical"], rel=1e-2)
    # The data fix the source among the sensors better than in the gap.
    observed, gap = part["exact_std"]
    assert observed < gap < 1.0
    assert part["deviation"][0] < part["deviation"][1]
    inside = np.abs(part["center"] - truth) <= 1.96 * part["deviation"]
    assert inside.sum() >= 1
    # The certified brackets hold the truth, and a set bound admits more
    # than the Gaussian posterior does.
    low, high = part["brackets"][:, 0], part["brackets"][:, 1]
    assert np.all(low <= truth) and np.all(truth <= high)
    assert part["inside"]
    assert np.all(high - low > 2 * 1.96 * part["deviation"])
    assert high[1] - low[1] > high[0] - low[0]
    # The truth lies in the prior's ball, whose bound alone is the radius
    # times the norm of each average's representer, the same either way;
    # the data narrow it, most among the sensors.
    assert 0.0 < part["radius"] and part["gauge"] <= 1.0
    alone = part["alone"]
    assert np.allclose(alone[:, 0], -alone[:, 1])
    narrowed = (high - low) / (alone[:, 1] - alone[:, 0])
    assert np.all(narrowed < 1.0) and narrowed[0] < narrowed[1]


def test_the_figures_were_written(results):
    outdir = results[1]
    names = [name for name in PARTS.values() if name != "box_inverse"]
    names += [f"box_inverse_{kind}" for kind in ("point", "posterior", "brackets")]
    for name in names:
        assert (outdir / f"{name}.png").exists(), name
