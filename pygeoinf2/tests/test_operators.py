"""Operators: adjoints, traits, the algebra, and the evaluation paths."""

import numpy as np
import pytest

from pygeoinf2.algebra.nodes import _Composition, _Sum
from pygeoinf2.algebra.operators import (
    LinearOperator,
    require_coordinates,
)
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.testing import (
    check_operator,
    check_traits,
)
from pygeoinf2.traits import Traits

from .conftest import make_dense_metric_space, make_weighted_space
from .doubles import OpaqueSpace


def spd_matrix(rng, n):
    root = rng.normal(size=(n, n))
    return root @ root.T + n * np.identity(n)


@pytest.fixture
def spaces():
    return EuclideanSpace(4), EuclideanSpace(3)


class TestAdjoint:
    def test_adjoint_identity_on_orthonormal_spaces(self, spaces, rng):
        X, Y = spaces
        A = LinearOperator.from_component_matrix(X, Y, rng.normal(size=(3, 4)))
        check_operator(A, rng=rng)

    def test_adjoint_identity_on_a_weighted_space(self, rng):
        """Where a hand-written adjoint most often goes wrong."""
        X = make_weighted_space()
        Y = make_dense_metric_space()
        A = LinearOperator.from_component_matrix(X, Y, rng.normal(size=(Y.dim, X.dim)))
        check_operator(A, rng=rng)

    def test_adjoint_is_memoised(self, spaces, rng):
        X, Y = spaces
        A = LinearOperator.from_component_matrix(X, Y, rng.normal(size=(3, 4)))
        assert A.adjoint is A.adjoint
        assert A.adjoint.adjoint is A

    def test_self_adjoint_operator_is_its_own_adjoint(self, rng):
        X = make_weighted_space()
        C = LinearOperator.from_component_matrix(
            X, X, spd_matrix(rng, X.dim), traits=Traits.SELF_ADJOINT
        )
        # The claim is not verified at construction, so check it separately.
        A = LinearOperator.self_adjoint(X, lambda x: X.scale(2.0, x))
        assert A.adjoint is A
        assert C.adjoint is C

    def test_adjoint_action_must_be_supplied(self, spaces):
        X, Y = spaces
        A = LinearOperator.from_callables(X, Y, lambda x: np.zeros(3))
        with pytest.raises(NotImplementedError, match="adjoint action"):
            A.adjoint(np.zeros(3))


class TestTraitPropagation:
    def test_gramian_is_semidefinite(self, spaces, rng):
        X, Y = spaces
        A = LinearOperator.from_component_matrix(X, Y, rng.normal(size=(3, 4)))
        G = A @ A.adjoint
        assert Traits.SELF_ADJOINT & G.traits
        assert Traits.POSITIVE_SEMIDEFINITE & G.traits
        check_traits(G, rng=rng)

    def test_congruence_is_recognised_after_the_fact(self, spaces, rng):
        """A @ C @ A.adjoint, built in two steps. The M1 acceptance case.

        This is what closures cannot do: the congruence is assembled as
        (A @ C) @ A.adjoint, so the pattern only exists once the composition is
        complete.
        """
        X, Y = spaces
        A = LinearOperator.from_component_matrix(X, Y, rng.normal(size=(3, 4)))
        C = LinearOperator.from_component_matrix(
            X, X, spd_matrix(rng, 4), traits=Traits.POSITIVE_SEMIDEFINITE
        )
        pushforward = A @ C @ A.adjoint
        assert Traits.SELF_ADJOINT & pushforward.traits
        assert Traits.POSITIVE_SEMIDEFINITE & pushforward.traits
        check_traits(pushforward, rng=rng)
        check_operator(pushforward, rng=rng)

    def test_bayesian_normal_operator(self, spaces, rng):
        """A Q A* + R, the operator the whole inversion layer inverts."""
        X, Y = spaces
        A = LinearOperator.from_component_matrix(X, Y, rng.normal(size=(3, 4)))
        Q = LinearOperator.from_component_matrix(
            X, X, spd_matrix(rng, 4), traits=Traits.POSITIVE_SEMIDEFINITE
        )
        R = LinearOperator.from_component_matrix(
            Y, Y, spd_matrix(rng, 3), traits=Traits.POSITIVE_DEFINITE
        )
        normal = A @ Q @ A.adjoint + R
        assert Traits.SELF_ADJOINT & normal.traits
        assert Traits.POSITIVE_DEFINITE & normal.traits
        check_traits(normal, rng=rng)

    def test_a_plain_product_claims_nothing(self, rng):
        X = EuclideanSpace(4)
        A = LinearOperator.from_component_matrix(
            X, X, spd_matrix(rng, 4), traits=Traits.SELF_ADJOINT
        )
        B = LinearOperator.from_component_matrix(
            X, X, spd_matrix(rng, 4), traits=Traits.SELF_ADJOINT
        )
        assert not (Traits.SELF_ADJOINT & (A @ B).traits)

    def test_negative_scaling_drops_definiteness(self, rng):
        X = EuclideanSpace(4)
        C = LinearOperator.from_component_matrix(
            X, X, spd_matrix(rng, 4), traits=Traits.POSITIVE_DEFINITE
        )
        assert Traits.SELF_ADJOINT & (-C).traits
        assert not (Traits.POSITIVE_SEMIDEFINITE & (-C).traits)

    def test_self_adjoint_claim_requires_an_endomorphism(self, spaces):
        X, Y = spaces
        with pytest.raises(ValueError, match="SELF_ADJOINT"):
            LinearOperator.from_callables(
                X, Y, lambda x: np.zeros(3), traits=Traits.SELF_ADJOINT
            )


class TestTraitsAreClaimsNotProofs:
    def test_a_false_claim_is_caught(self, rng):
        """Nothing verifies traits at construction. check_traits is the net."""
        X = EuclideanSpace(4)
        asymmetric = rng.normal(size=(4, 4))
        liar = LinearOperator.from_component_matrix(
            X, X, asymmetric, traits=Traits.SELF_ADJOINT
        )
        with pytest.raises(AssertionError, match="SELF_ADJOINT"):
            check_traits(liar, rng=rng)

    def test_a_false_definiteness_claim_is_caught(self, rng):
        X = EuclideanSpace(4)
        negative = -spd_matrix(rng, 4)
        liar = LinearOperator.from_component_matrix(
            X, X, negative, traits=Traits.POSITIVE_DEFINITE
        )
        with pytest.raises(AssertionError, match="POSITIVE"):
            check_traits(liar, rng=rng)


class TestNodes:
    def test_identity(self, rng):
        X = make_weighted_space()
        identity = LinearOperator.identity(X)
        check_operator(identity, rng=rng)
        check_traits(identity, rng=rng)
        x = X.random(rng)
        assert identity(x) is x

    def test_identity_disappears_from_compositions(self, spaces, rng):
        X, Y = spaces
        A = LinearOperator.from_component_matrix(X, Y, rng.normal(size=(3, 4)))
        assert A @ LinearOperator.identity(X) is A
        assert LinearOperator.identity(Y) @ A is A

    def test_zero(self, spaces, rng):
        X, Y = spaces
        zero = LinearOperator.zero(X, Y)
        check_operator(zero, rng=rng)
        assert np.allclose(zero(X.random(rng)), np.zeros(3))

    def test_zero_disappears_from_sums(self, spaces, rng):
        X, Y = spaces
        A = LinearOperator.from_component_matrix(X, Y, rng.normal(size=(3, 4)))
        assert A + LinearOperator.zero(X, Y) is A
        assert LinearOperator.zero(X, Y) + A is A

    def test_sums_flatten(self, spaces, rng):
        X, Y = spaces
        ops = [
            LinearOperator.from_component_matrix(X, Y, rng.normal(size=(3, 4)))
            for _ in range(3)
        ]
        total = ops[0] + ops[1] + ops[2]
        assert isinstance(total, _Sum)
        assert len(total.terms) == 3

    def test_compositions_flatten(self, rng):
        X = EuclideanSpace(4)
        ops = [
            LinearOperator.from_component_matrix(X, X, rng.normal(size=(4, 4)))
            for _ in range(3)
        ]
        product = ops[0] @ ops[1] @ ops[2]
        assert isinstance(product, _Composition)
        assert len(product.factors) == 3

    def test_nested_scalings_fold(self, spaces, rng):
        X, Y = spaces
        A = LinearOperator.from_component_matrix(X, Y, rng.normal(size=(3, 4)))
        assert (2.0 * (3.0 * A)).alpha == pytest.approx(6.0)
        assert 0.5 * (2.0 * A) is A
        assert 1.0 * A is A

    def test_composition_adjoint_reverses(self, rng):
        X = EuclideanSpace(4)
        A = LinearOperator.from_component_matrix(X, X, rng.normal(size=(4, 4)))
        B = LinearOperator.from_component_matrix(X, X, rng.normal(size=(4, 4)))
        product = A @ B
        check_operator(product, rng=rng)
        assert product.adjoint.factors == (B.adjoint, A.adjoint)

    def test_repr_names_real_objects(self, spaces, rng):
        X, Y = spaces
        A = LinearOperator.from_component_matrix(X, Y, rng.normal(size=(3, 4)))
        assert "Composition" in repr(A @ A.adjoint)
        assert "Adjoint" in repr(A.adjoint)


class TestSpecialisationProtocol:
    """Structure must survive the algebra regardless of operand order."""

    def test_order_independence(self, rng):
        """v1 loses structure for `generic + special` but not the reverse."""
        X = EuclideanSpace(4)

        class Diagonal(LinearOperator):
            """A family closed under addition, like InvariantLinearAutomorphism."""

            def __init__(self, domain, values, **kwargs):
                super().__init__(domain, domain, **kwargs)
                self.values = np.asarray(values, dtype=float)

            def _value(self, x):
                return self.values * x

            def _adjoint_value(self, y):
                return self.values * y

            def _combine_add(self, other):
                if isinstance(other, Diagonal) and other.domain == self.domain:
                    return Diagonal(
                        self.domain,
                        self.values + other.values,
                        traits=Traits.SELF_ADJOINT,
                    )
                return None

            def _combine_radd(self, other):
                return self._combine_add(other)

        a = Diagonal(X, [1.0, 2, 3, 4], traits=Traits.SELF_ADJOINT)
        b = Diagonal(X, [5.0, 6, 7, 8], traits=Traits.SELF_ADJOINT)
        generic = LinearOperator.from_component_matrix(X, X, rng.normal(size=(4, 4)))

        assert isinstance(a + b, Diagonal)
        # And the structure is not lost just because the special operand is on
        # the right, which is the v1 failure mode.
        assert isinstance(generic + a, _Sum)
        assert isinstance(a + generic, _Sum)
        assert type(generic + a) is type(a + generic)


class TestMatrixRepresentations:
    def test_component_matrix_round_trip(self, rng):
        X, Y = make_weighted_space(), make_dense_metric_space()
        M = rng.normal(size=(Y.dim, X.dim))
        A = LinearOperator.from_component_matrix(X, Y, M)
        assert np.allclose(A.matrix(form="components"), M)

    def test_galerkin_form_is_the_gram_weighted_matrix(self, rng):
        X, Y = make_weighted_space(), make_dense_metric_space()
        M = rng.normal(size=(Y.dim, X.dim))
        A = LinearOperator.from_component_matrix(X, Y, M)
        assert np.allclose(A.matrix(form="galerkin"), Y.gram_matrix() @ M)

    def test_auto_picks_galerkin_for_self_adjoint_operators(self, rng):
        """Self-adjointness shows as matrix symmetry only in Galerkin form."""
        X = make_weighted_space()
        S = spd_matrix(rng, X.dim)
        A = LinearOperator.self_adjoint(
            X, lambda x: X.solve_gram(S @ X.to_components(x))
        )
        auto = A.matrix()
        assert np.allclose(auto, auto.T)
        assert not np.allclose(
            A.matrix(form="components"), A.matrix(form="components").T
        )

    def test_derivative_matrix_adjoint_returns_representers(self, rng):
        """The operator-level form of the derivative/gradient distinction."""
        X, Y = make_weighted_space(), EuclideanSpace(2)
        M = rng.normal(size=(2, X.dim))
        A = LinearOperator.from_derivative_matrix(X, Y, M)
        check_operator(A, rng=rng)
        # Row i acts as the i-th derivative functional...
        x = X.random(rng)
        assert np.allclose(A(x), M @ X.to_components(x))
        # ...and the adjoint of a basis covector is that row's representer.
        e0 = np.array([1.0, 0.0])
        assert np.allclose(X.to_components(A.adjoint(e0)), X.solve_gram(M[0]))

    def test_matrix_requires_coordinates(self, rng):
        X = OpaqueSpace(np.array([1.0, 2.0, 3.0]))
        A = LinearOperator.self_adjoint(X, lambda x: x)
        with pytest.raises(TypeError, match="no coordinate map"):
            A.matrix()

    def test_require_coordinates_names_the_capability(self):
        with pytest.raises(TypeError, match="requires one"):
            require_coordinates(OpaqueSpace(np.array([1.0])))
