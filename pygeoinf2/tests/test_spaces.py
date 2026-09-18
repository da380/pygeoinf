"""Spaces: the axioms, the coordinate layer, and the metric."""

from typing import Hashable

import numpy as np
import pytest

from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.algebra.spaces import EuclideanSpace, HilbertSpace, Reals
from pygeoinf2.traits import Traits
from pygeoinf2.testing import (
    check_coordinates,
    check_representer,
    check_space,
    check_white_noise,
)

from .conftest import (
    WeightedSpace,
    make_dense_metric_space,
    make_weighted_space,
)


SPACES = {
    "euclidean": (lambda: EuclideanSpace(4)),
    "reals": (lambda: Reals()),
    "weighted": make_weighted_space,
    "dense_metric": make_dense_metric_space,
}


@pytest.mark.parametrize("name", list(SPACES))
class TestAxioms:
    def test_space_axioms(self, name, rng):
        build = SPACES[name]
        check_space(build(), rng=rng, rebuild=build)

    def test_coordinate_axioms(self, name, rng):
        check_coordinates(SPACES[name](), rng=rng)


class TestIdentity:
    def test_spaces_are_hashable(self):
        """v1 declares __eq__ without __hash__, so every space is unhashable."""
        assert {EuclideanSpace(3): "value"}[EuclideanSpace(3)] == "value"
        assert len({EuclideanSpace(3), EuclideanSpace(3), EuclideanSpace(4)}) == 2

    def test_structural_equality(self):
        assert EuclideanSpace(3) == EuclideanSpace(3)
        assert EuclideanSpace(3) != EuclideanSpace(4)
        assert Reals() == Reals()

    def test_different_types_are_unequal(self):
        assert EuclideanSpace(1) != Reals()
        assert EuclideanSpace(3) != "not a space"

    def test_equality_is_not_identity_based(self):
        a, b = make_weighted_space(), make_weighted_space()
        assert a is not b
        assert a == b and hash(a) == hash(b)


class TestIdentityFailureIsCaught:
    """The check must catch the v1 mass-weighted-space equality defect."""

    def test_identity_keyed_space_is_rejected(self, rng):
        class IdentityKeyedSpace(WeightedSpace):
            def _key(self) -> Hashable:
                return id(self)  # what comparing by operator identity amounts to

        def build():
            return IdentityKeyedSpace(np.array([1.0, 4.0, 9.0]))

        # It passes on its own...
        check_space(build(), rng=rng)
        # ...but not against an independently constructed copy.
        with pytest.raises(
            AssertionError, match="structurally identical spaces are equal"
        ):
            check_space(build(), rng=rng, rebuild=build)


class TestMetric:
    def test_orthonormal_space_has_identity_gram(self):
        assert np.allclose(EuclideanSpace(4).gram_matrix(), np.identity(4))
        assert EuclideanSpace(4).is_orthonormal

    def test_weighted_space_gram_is_the_metric(self):
        space = make_weighted_space()
        assert not space.is_orthonormal
        assert np.allclose(space.gram_matrix(), np.diag(space.metric_values))

    def test_inner_product_carries_the_metric(self):
        space = make_weighted_space()
        x = space.from_components(np.array([1.0, 1.0, 1.0, 1.0]))
        assert space.inner_product(x, x) == pytest.approx(space.metric_values.sum())

    def test_dense_gram_round_trips(self):
        space = make_dense_metric_space()
        c = np.array([1.0, -2.0, 0.5])
        assert np.allclose(space.solve_gram(space.apply_gram(c)), c)

    def test_metric_values_must_be_positive(self):
        with pytest.raises(ValueError, match="strictly positive"):
            WeightedSpace(np.array([1.0, -1.0]))


class TestRepresenter:
    """The distinction between a derivative and a gradient. See DECISIONS.md D-26."""

    def test_representer_pairs_as_the_derivative_does(self, rng):
        for build in (
            make_weighted_space,
            make_dense_metric_space,
            lambda: EuclideanSpace(4),
        ):
            space = build()
            g = rng.normal(size=space.dim)
            check_representer(space, g, rng=rng)

    def test_representer_differs_from_the_raw_components(self):
        """On a non-orthonormal basis, using g as a gradient is wrong by G."""
        space = make_weighted_space()
        g = np.ones(space.dim)
        representer = space.representer(g)
        naive = space.from_components(g)  # the classic adjoint-method error
        assert not np.allclose(space.to_components(representer), g)
        assert np.allclose(space.to_components(representer), g / space.metric_values)

        x = space.from_components(np.array([1.0, 1.0, 1.0, 1.0]))
        exact = float(g @ space.to_components(x))
        assert space.inner_product(representer, x) == pytest.approx(exact)
        assert space.inner_product(naive, x) != pytest.approx(exact)

    def test_the_two_coincide_on_an_orthonormal_basis(self):
        """Which is why the error survives: it is invisible in the toy case."""
        space = EuclideanSpace(4)
        g = np.array([1.0, -2.0, 3.0, 0.5])
        assert np.allclose(space.to_components(space.representer(g)), g)


class TestWhiteNoise:
    """v1 gets this wrong on every mass-weighted space. See DECISIONS.md D-19."""

    @pytest.mark.parametrize("name", ["euclidean", "weighted", "dense_metric"])
    def test_white_noise_has_identity_covariance(self, name, rng):
        check_white_noise(SPACES[name](), rng=rng, samples=40000, rtol=0.05)

    def test_the_check_catches_the_v1_construction(self, rng):
        """Drawing standard normal components gives covariance G, not I."""

        class V1StyleSpace(WeightedSpace):
            def white_noise(self, *, rng=None):
                rng = np.random.default_rng() if rng is None else rng
                return self.from_components(rng.standard_normal(self.dim))

        with pytest.raises(AssertionError, match="white noise has identity covariance"):
            check_white_noise(
                V1StyleSpace(np.array([1.0, 4.0, 9.0])),
                rng=rng,
                samples=20000,
                rtol=0.05,
            )

    @pytest.mark.parametrize("size", [1.0e-4, 1.0, 1.0e4])
    def test_the_tolerance_scales_with_the_metric(self, size, rng):
        """The sampling error of an entry is ``sqrt(2 C_ii C_jj / n)`` at
        most, so the agreement asked for is a fraction of ``sqrt(C_ii C_jj)``
        and one ``rtol`` serves a metric of any size. Against a fixed
        tolerance a large metric failed by chance -- it did, on a Sobolev
        space whose eigenvalues a Robin condition had lifted -- and a small
        one could not fail at all."""
        space = WeightedSpace(size * np.array([1.0, 4.0, 9.0]))
        check_white_noise(space, rng=rng, samples=20000, rtol=0.06)

    @pytest.mark.parametrize("size", [1.0e-4, 1.0e4])
    def test_the_v1_construction_is_caught_at_any_size_of_metric(self, size, rng):
        class V1StyleSpace(WeightedSpace):
            def white_noise(self, *, rng=None):
                rng = np.random.default_rng() if rng is None else rng
                return self.from_components(rng.standard_normal(self.dim))

        with pytest.raises(AssertionError, match="white noise has identity covariance"):
            check_white_noise(
                V1StyleSpace(size * np.array([1.0, 4.0, 9.0])),
                rng=rng,
                samples=20000,
                rtol=0.06,
            )

    def test_random_is_not_advertised_as_white_noise(self, rng):
        """random() draws standard normal components and makes no claim."""
        space = make_weighted_space()
        c = space.to_components(space.random(rng=rng))
        assert c.shape == (space.dim,)


class TestReals:
    def test_vectors_are_plain_floats(self):
        space = Reals()
        assert isinstance(space.zero(), float)
        assert isinstance(space.add(1.5, 2.0), float)
        assert space.inner_product(3.0, 4.0) == pytest.approx(12.0)

    def test_immutable_backend_returns_rather_than_mutates(self):
        """The in-place contract: always use the return value."""
        space = Reals()
        y = 1.0
        result = space.axpy(2.0, 3.0, y)
        assert result == pytest.approx(7.0)
        assert y == 1.0  # unchanged, because floats cannot be mutated

    def test_round_trips_through_components(self):
        space = Reals()
        assert space.from_components(space.to_components(2.5)) == pytest.approx(2.5)


class TestDerivedOperations:
    def test_gram_schmidt_orthonormalizes(self, rng):
        space = make_weighted_space()
        vectors = [space.random(rng=rng) for _ in range(3)]
        basis = space.gram_schmidt(vectors)
        for i, u in enumerate(basis):
            for j, v in enumerate(basis):
                assert space.inner_product(u, v) == pytest.approx(
                    1.0 if i == j else 0.0, abs=1e-10
                )

    def test_gram_schmidt_rejects_dependent_vectors(self):
        space = make_weighted_space()
        x = space.from_components(np.array([1.0, 2.0, 3.0, 4.0]))
        with pytest.raises(ValueError, match="linearly dependent"):
            space.gram_schmidt([x, space.scale(2.0, x)])

    def test_mean(self, rng):
        space = make_weighted_space()
        vectors = [space.random(rng=rng) for _ in range(5)]
        expected = np.mean([space.to_components(v) for v in vectors], axis=0)
        assert np.allclose(space.to_components(space.mean(vectors)), expected)

    def test_mean_of_nothing_is_an_error(self):
        with pytest.raises(ValueError, match="empty"):
            EuclideanSpace(3).mean([])

    def test_basis_vector_bounds(self):
        with pytest.raises(IndexError):
            EuclideanSpace(3).basis_vector(3)


class TestCoordinateFreeSpace:
    """A space with no coordinates still satisfies the core axioms."""

    def test_random_is_optional(self):
        from pygeoinf2.algebra.spaces import HilbertSpace

        class Opaque(HilbertSpace[np.ndarray]):
            @property
            def dim(self):
                return 2

            def _key(self):
                return ()

            def zero(self):
                return np.zeros(2)

            def copy(self, x):
                return x.copy()

            def inner_product(self, x, y):
                return float(np.dot(x, y))

            def axpy(self, a, x, y):
                y += a * x
                return y

            def scale_inplace(self, a, x):
                x *= a
                return x

        space = Opaque()
        assert space.norm(np.array([3.0, 4.0])) == pytest.approx(5.0)
        with pytest.raises(NotImplementedError, match="random"):
            space.random()


class TestADiagonalMetricKnowsItsGramMatrix:
    """Written down from the diagonal, not probed: the base class applied the
    metric to every basis vector, dim^2 multiplications to learn the dim
    numbers a diagonal-metric space already holds, and the dense
    log-determinant then took an O(dim^3) slogdet of the result."""

    def test_the_gram_matrix_is_the_diagonal_without_a_probe(self, monkeypatch):
        space = make_weighted_space()
        monkeypatch.setattr(
            type(space),
            "apply_gram",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("probed")),
        )
        assert np.allclose(space.gram_matrix(), np.diag(space.metric_values))

    def test_the_dense_log_determinant_never_builds_the_gram_matrix(
        self, rng, monkeypatch
    ):
        from pygeoinf2.algebra.diagonal import DiagonalLinearOperator
        from pygeoinf2.numerics.functional_calculus import log_determinant

        space = make_weighted_space()
        values = rng.uniform(1.0, 3.0, space.dim)
        # A non-diagonal operator on a diagonal-metric space, so the dense
        # route is the one taken and the metric correction is needed.
        root = rng.normal(size=(space.dim, space.dim))
        galerkin = root @ root.T + space.dim * np.identity(space.dim)
        operator = LinearOperator.from_matrix(
            space, space, galerkin, traits=Traits.POSITIVE_DEFINITE, form="galerkin"
        )
        components = np.linalg.solve(np.diag(space.metric_values), galerkin)
        expected = float(np.linalg.slogdet(components)[1])
        monkeypatch.setattr(
            type(space),
            "gram_matrix",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("dense gram")),
        )
        assert log_determinant(operator, method="dense").value == pytest.approx(
            expected
        )
        del values, DiagonalLinearOperator


class TestMassWeightedSpace:
    """One inner product against another, on the same vectors."""

    @pytest.fixture
    def weighted(self):
        from pygeoinf2.algebra.diagonal import DiagonalLinearOperator
        from pygeoinf2.algebra.spaces import MassWeightedSpace

        base = EuclideanSpace(4)
        mass = DiagonalLinearOperator(base, np.array([1.0, 4.0, 9.0, 0.25]))
        return base, mass, MassWeightedSpace(base, mass)

    def test_a_diagonal_mass_is_inverted_exactly_and_for_free(
        self, weighted, monkeypatch
    ):
        """As the docstring always promised, and as v1 had by taking the
        inverse from the caller; the default used to be a conjugate-gradient
        solve on every application."""
        from pygeoinf2.algebra.diagonal import DiagonalLinearOperator
        from pygeoinf2.numerics import solvers

        base, mass, space = weighted
        monkeypatch.setattr(
            solvers.CGSolver,
            "__init__",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("CG built")),
        )
        inverse = space.mass_inverse
        assert isinstance(inverse, DiagonalLinearOperator)
        assert np.allclose(inverse.eigenvalues, 1.0 / mass.eigenvalues)

    def test_a_general_mass_still_falls_back_to_a_solve(self, rng):
        from pygeoinf2.algebra.spaces import MassWeightedSpace
        from pygeoinf2.numerics.solvers import InverseOperator

        base = EuclideanSpace(4)
        root = rng.normal(size=(4, 4))
        mass = LinearOperator.from_matrix(
            base,
            base,
            root @ root.T + 4.0 * np.identity(4),
            traits=Traits.POSITIVE_DEFINITE,
            form="galerkin",
        )
        space = MassWeightedSpace(base, mass)
        assert isinstance(space.mass_inverse, InverseOperator)
        x = base.random(rng=rng)
        assert base.norm(
            base.subtract(mass(space.mass_inverse(x)), x)
        ) < 1e-8 * base.norm(x)

    def test_it_is_a_hilbert_space(self, weighted, rng):
        _, _, space = weighted
        check_space(space, rng=rng)

    def test_the_inner_product_is_the_weighted_one(self, weighted, rng):
        base, mass, space = weighted
        x, y = base.random(rng=rng), base.random(rng=rng)
        assert space.inner_product(x, y) == pytest.approx(
            base.inner_product(mass(x), y)
        )

    def test_the_inverse_mass_is_derived(self, weighted, rng):
        """v1 makes the caller supply it. Deriving it is one fewer thing to get
        wrong, and it makes the construction usable when the inverse has no
        closed form."""
        base, mass, space = weighted
        probe = base.random(rng=rng)
        recovered = space.mass_inverse(mass(probe))
        assert base.norm(base.subtract(recovered, probe)) < 1e-10 * base.norm(probe)

    def test_it_shares_its_vectors_with_its_base(self, weighted):
        base, _, space = weighted
        assert space.shares_vectors_with(base)

    def test_it_shares_its_vectors_with_an_equal_base(self, weighted):
        """An equal-but-distinct base is the same space by every property
        that matters, and ``EuclideanSpace(n)`` is minted freely inside the
        library, so identity is the wrong test."""
        _, _, space = weighted
        assert space.shares_vectors_with(EuclideanSpace(4))

    def test_equal_spaces_share_vectors_by_default(self):
        """The default on :class:`HilbertSpace`, which the mass-weighted
        space delegates to for its base."""
        assert EuclideanSpace(4).shares_vectors_with(EuclideanSpace(4))
        assert not EuclideanSpace(4).shares_vectors_with(EuclideanSpace(5))

    def test_equal_mass_operators_give_equal_spaces(self, weighted):
        """The key holds the mass operator, not ``id`` of it: identical
        equality, but no stale address to alias a different operator."""
        from pygeoinf2.algebra.spaces import MassWeightedSpace

        base, mass, space = weighted
        assert space == MassWeightedSpace(EuclideanSpace(4), mass)
        assert hash(space) == hash(MassWeightedSpace(EuclideanSpace(4), mass))

    def test_a_mass_operator_must_claim_what_it_needs(self, weighted):
        from pygeoinf2.algebra.spaces import MassWeightedSpace

        base, _, _ = weighted
        unclaimed = LinearOperator.from_matrix(
            base, base, np.identity(4), form="components"
        )
        with pytest.raises(ValueError, match="must claim"):
            MassWeightedSpace(base, unclaimed)

    def test_the_mass_must_act_on_the_base(self, weighted):
        from pygeoinf2.algebra.diagonal import DiagonalLinearOperator
        from pygeoinf2.algebra.spaces import MassWeightedSpace

        base, _, _ = weighted
        elsewhere = DiagonalLinearOperator(EuclideanSpace(3), np.ones(3))
        with pytest.raises(ValueError, match="map .* to itself"):
            MassWeightedSpace(base, elsewhere)


class TestOrthonormalisationOnComponents:
    """``gram_schmidt`` and ``orthonormal_basis`` on a coordinate space work
    on component arrays, converting each vector once. They must agree with
    the coordinate-free versions on a Gram matrix that is not diagonal."""

    @pytest.fixture
    def dense(self):
        return make_dense_metric_space(40)

    def test_gram_schmidt_matches_the_generic_route(self, dense, rng):
        vectors = [dense.random(rng=rng) for _ in range(12)]
        fast = dense.gram_schmidt(vectors)
        slow = HilbertSpace.gram_schmidt(dense, vectors)
        for a, b in zip(fast, slow):
            assert dense.norm(dense.subtract(a, b)) < 1e-10
        gram = np.array([[dense.inner_product(a, b) for b in fast] for a in fast])
        assert gram == pytest.approx(np.eye(12), abs=1e-10)

    def test_orthonormal_basis_drops_dependent_vectors(self, dense, rng):
        x, y = dense.random(rng=rng), dense.random(rng=rng)
        basis = dense.orthonormal_basis([x, dense.scale(2.0, x), y, dense.add(x, y)])
        assert len(basis) == 2
        slow = HilbertSpace.orthonormal_basis(dense, [x, dense.scale(2.0, x), y])
        for a, b in zip(basis, slow):
            assert dense.norm(dense.subtract(a, b)) < 1e-10

    def test_gram_schmidt_rejects_dependence_by_index(self, dense, rng):
        x = dense.random(rng=rng)
        with pytest.raises(ValueError, match="Vector 1"):
            dense.gram_schmidt([x, dense.scale(-3.0, x)])

    def test_the_vectors_do_not_alias_the_array(self, rng):
        """EuclideanSpace's coordinate map does not copy, so the fast path
        must, or the returned vectors would be views of one array."""
        space = EuclideanSpace(5)
        basis = space.orthonormal_basis([space.random(rng=rng) for _ in range(3)])
        before = basis[1].copy()
        space.scale_inplace(10.0, basis[0])
        assert np.array_equal(basis[1], before)


class TestMassWeightedCoordinates:
    """Over a coordinate base the weighted space is a coordinate space, with
    the Gram map the base's composed with the mass operator; over a module it
    keeps the pointwise operations. And the lift from the base stays exact
    and stays on the component route."""

    @pytest.fixture
    def dense(self, rng):
        from pygeoinf2.algebra.spaces import CoordinateSpace, MassWeightedSpace

        base = make_dense_metric_space(4)
        root = rng.normal(size=(4, 4)) + 3.0 * np.identity(4)
        mass = LinearOperator.from_matrix(
            base,
            base,
            root @ root.T,
            form="galerkin",
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
        )
        space = MassWeightedSpace(base, mass)
        assert isinstance(space, CoordinateSpace)
        return base, mass, space

    def test_it_passes_the_coordinate_checks(self, dense, rng):
        base, mass, space = dense
        check_space(space, rng=rng)
        check_coordinates(space, rng=rng)

    def test_the_gram_map_is_the_base_gram_after_the_mass(self, dense):
        base, mass, space = dense
        expected = base.gram_matrix() @ mass.matrix(form="components")
        assert np.allclose(space.gram_matrix(), expected)
        assert np.allclose(expected, expected.T)

    def test_a_direct_solver_and_a_matrix_work_over_it(self, dense, rng):
        from pygeoinf2.numerics.solvers import CholeskySolver

        base, mass, space = dense
        root = rng.normal(size=(4, 4)) + 4.0 * np.identity(4)
        operator = LinearOperator.from_matrix(
            space,
            space,
            root @ root.T,
            form="galerkin",
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
        )
        y = space.random(rng=rng)
        x = CholeskySolver()(operator)(y)
        assert space.norm(space.subtract(operator(x), y)) < 1e-9 * space.norm(y)
        assert operator.matrix(form="galerkin").shape == (4, 4)

    def test_the_lift_from_the_base_is_the_mass_formula(self, dense, rng):
        """``A^{*V} == M^-1 A^{*U} M``, and the lifted operator acts on
        components without leaving them, which is what keeps the lift cheap
        (David: the automated lift is key)."""
        from pygeoinf2.testing import check_operator

        base, mass, space = dense
        on_base = LinearOperator.from_matrix(
            base, base, rng.normal(size=(4, 4)), form="components"
        )
        lifted = LinearOperator.from_formal_adjoint(space, space, on_base)
        check_operator(lifted, rng=rng)
        y = space.random(rng=rng)
        expected = space.mass_inverse(on_base.adjoint(mass(y)))
        assert space.norm(space.subtract(lifted.adjoint(y), expected)) < 1e-9 * (
            space.norm(expected) + 1.0
        )
        assert lifted._components_action() is not None
        assert lifted._components_adjoint_action() is not None

    def test_over_a_module_the_pointwise_operations_survive(self):
        from pygeoinf2.algebra.diagonal import DiagonalLinearOperator
        from pygeoinf2.algebra.spaces import HilbertModule, MassWeightedSpace
        from pygeoinf2.symmetric_space.fourier import Lebesgue

        base = Lebesgue((8,), lengths=(1.0,))
        mass = DiagonalLinearOperator(
            base,
            1.0 + base.laplacian_eigenvalues,
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
        )
        space = MassWeightedSpace(base, mass)
        assert isinstance(space, HilbertModule)
        f = base.project_function(lambda t: 1.0 + 0.5 * np.cos(2.0 * np.pi * t))
        assert np.allclose(space.multiply(f, f), base.multiply(f, f))
        assert np.allclose(space.sqrt(f), base.sqrt(f))
        assert space.has_diagonal_metric
        assert np.allclose(space.gram_diagonal(), 1.0 + base.laplacian_eigenvalues)
        assert np.allclose(np.diag(space.gram_matrix()), space.gram_diagonal())

    def test_a_subclass_is_left_as_written(self):
        from pygeoinf2.algebra.diagonal import DiagonalLinearOperator
        from pygeoinf2.algebra.spaces import CoordinateSpace, MassWeightedSpace

        class Plain(MassWeightedSpace):
            pass

        base = EuclideanSpace(3)
        mass = DiagonalLinearOperator(base, np.array([1.0, 2.0, 3.0]))
        assert type(Plain(base, mass)) is Plain
        assert not isinstance(Plain(base, mass), CoordinateSpace)


class TestCoordinateSelection:
    """v1's ``subspace_projection``, on any coordinate space."""

    def test_it_selects_and_its_adjoint_carries_the_metric(self, rng):
        from pygeoinf2.testing import check_operator

        space = make_weighted_space()
        selection = space.coordinate_selection([3, 1])
        check_operator(selection, rng=rng)
        x = space.random(rng=rng)
        assert np.allclose(selection(x), space.to_components(x)[[3, 1]])
        pulled = space.to_components(selection.adjoint(np.array([1.0, 0.0])))
        expected = np.zeros(4)
        expected[3] = 1.0 / space.metric_values[3]
        assert np.allclose(pulled, expected)
        assert selection._components_action() is not None

    def test_bad_positions_are_refused(self):
        space = make_weighted_space()
        with pytest.raises(ValueError, match="repeat"):
            space.coordinate_selection([1, 1])
        with pytest.raises(ValueError, match="lie in"):
            space.coordinate_selection([4])
