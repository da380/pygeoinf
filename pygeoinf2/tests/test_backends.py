"""
Foreign backends: the claim the test doubles cannot settle.

``OpaqueSpace`` shows that the core does not *need* components. These tests
show that it works when the vectors genuinely belong to someone else — an
``mfem.Vector`` with its own memory, a ``PETSc.Vec`` that may be distributed.

Both are optional dependencies and skip when absent.
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


class TestPetsc:
    """Written against petsc4py; skipped where PETSc is not built."""

    @pytest.fixture
    def petsc_spaces(self, rng):
        pytest.importorskip("petsc4py")
        from petsc4py import PETSc

        from pygeoinf2.backends.petsc import PetscSpace, PetscWeightedSpace

        n = 8

        def matrix(array):
            result = PETSc.Mat().createDense([n, n], array=np.ascontiguousarray(array))
            result.assemble()
            return result

        root = rng.normal(size=(n, n))
        mass = root @ root.T + n * np.identity(n)
        return PetscSpace(n), PetscWeightedSpace(matrix(mass)), mass, matrix

    def test_the_axioms_hold(self, petsc_spaces, rng):
        plain, weighted, _, _ = petsc_spaces
        check_space(plain, rng=rng)
        check_space(weighted, rng=rng)
        check_coordinates(weighted, rng=rng)

    def test_the_adjoint_is_not_the_transpose(self, petsc_spaces, rng):
        """The point of the weighted space, and of DESIGN.md 5.6."""
        from pygeoinf2.backends.petsc import operator_from_matrix

        _, weighted, mass, matrix = petsc_spaces
        array = rng.normal(size=mass.shape)
        A = operator_from_matrix(weighted, matrix(array))
        check_operator(A, rng=rng)

        y = weighted.random(rng=rng)
        adjoint = weighted.to_components(A.adjoint(y))
        components = weighted.to_components(y)
        assert np.allclose(adjoint, np.linalg.solve(mass, array.T @ mass @ components))
        assert not np.allclose(adjoint, array.T @ components)
