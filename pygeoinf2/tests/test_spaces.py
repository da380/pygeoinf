"""Spaces: the axioms, the coordinate layer, and the metric."""

from typing import Hashable

import numpy as np
import pytest

from pygeoinf2.algebra.spaces import EuclideanSpace, Reals
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
    """The distinction between a derivative and a gradient. See DESIGN.md 5.6."""

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
    """v1 gets this wrong on every mass-weighted space. See DESIGN.md 9."""

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
    def test_gram_schmidt_orthonormalises(self, rng):
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
