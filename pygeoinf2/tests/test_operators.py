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
from .doubles import Opaque, OpaqueSpace


def spd_matrix(rng, n):
    root = rng.normal(size=(n, n))
    return root @ root.T + n * np.identity(n)


@pytest.fixture
def spaces():
    return EuclideanSpace(4), EuclideanSpace(3)


class TestAdjoint:
    def test_adjoint_identity_on_orthonormal_spaces(self, spaces, rng):
        X, Y = spaces
        A = LinearOperator.from_matrix(X, Y, rng.normal(size=(3, 4)), form="components")
        check_operator(A, rng=rng)

    def test_adjoint_identity_on_a_weighted_space(self, rng):
        """Where a hand-written adjoint most often goes wrong."""
        X = make_weighted_space()
        Y = make_dense_metric_space()
        A = LinearOperator.from_matrix(
            X, Y, rng.normal(size=(Y.dim, X.dim)), form="components"
        )
        check_operator(A, rng=rng)

    def test_adjoint_is_memoised(self, spaces, rng):
        X, Y = spaces
        A = LinearOperator.from_matrix(X, Y, rng.normal(size=(3, 4)), form="components")
        assert A.adjoint is A.adjoint
        assert A.adjoint.adjoint is A

    def test_self_adjoint_operator_is_its_own_adjoint(self, rng):
        X = make_weighted_space()
        C = LinearOperator.from_matrix(
            X, X, spd_matrix(rng, X.dim), traits=Traits.SELF_ADJOINT, form="components"
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
        A = LinearOperator.from_matrix(X, Y, rng.normal(size=(3, 4)), form="components")
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
        A = LinearOperator.from_matrix(X, Y, rng.normal(size=(3, 4)), form="components")
        C = LinearOperator.from_matrix(
            X,
            X,
            spd_matrix(rng, 4),
            traits=Traits.POSITIVE_SEMIDEFINITE,
            form="components",
        )
        pushforward = A @ C @ A.adjoint
        assert Traits.SELF_ADJOINT & pushforward.traits
        assert Traits.POSITIVE_SEMIDEFINITE & pushforward.traits
        check_traits(pushforward, rng=rng)
        check_operator(pushforward, rng=rng)

    def test_bayesian_normal_operator(self, spaces, rng):
        """A Q A* + R, the operator the whole inversion layer inverts."""
        X, Y = spaces
        A = LinearOperator.from_matrix(X, Y, rng.normal(size=(3, 4)), form="components")
        Q = LinearOperator.from_matrix(
            X,
            X,
            spd_matrix(rng, 4),
            traits=Traits.POSITIVE_SEMIDEFINITE,
            form="components",
        )
        R = LinearOperator.from_matrix(
            Y, Y, spd_matrix(rng, 3), traits=Traits.POSITIVE_DEFINITE, form="components"
        )
        normal = A @ Q @ A.adjoint + R
        assert Traits.SELF_ADJOINT & normal.traits
        assert Traits.POSITIVE_DEFINITE & normal.traits
        check_traits(normal, rng=rng)

    def test_a_plain_product_claims_nothing(self, rng):
        X = EuclideanSpace(4)
        A = LinearOperator.from_matrix(
            X, X, spd_matrix(rng, 4), traits=Traits.SELF_ADJOINT, form="components"
        )
        B = LinearOperator.from_matrix(
            X, X, spd_matrix(rng, 4), traits=Traits.SELF_ADJOINT, form="components"
        )
        assert not (Traits.SELF_ADJOINT & (A @ B).traits)

    def test_negative_scaling_drops_definiteness(self, rng):
        X = EuclideanSpace(4)
        C = LinearOperator.from_matrix(
            X, X, spd_matrix(rng, 4), traits=Traits.POSITIVE_DEFINITE, form="components"
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
        liar = LinearOperator.from_matrix(
            X, X, asymmetric, traits=Traits.SELF_ADJOINT, form="components"
        )
        with pytest.raises(AssertionError, match="SELF_ADJOINT"):
            check_traits(liar, rng=rng)

    def test_a_false_definiteness_claim_is_caught(self, rng):
        X = EuclideanSpace(4)
        negative = -spd_matrix(rng, 4)
        liar = LinearOperator.from_matrix(
            X, X, negative, traits=Traits.POSITIVE_DEFINITE, form="components"
        )
        with pytest.raises(AssertionError, match="POSITIVE"):
            check_traits(liar, rng=rng)


class TestNodes:
    def test_identity(self, rng):
        X = make_weighted_space()
        identity = LinearOperator.identity(X)
        check_operator(identity, rng=rng)
        check_traits(identity, rng=rng)
        x = X.random(rng=rng)
        assert identity(x) is x

    def test_identity_disappears_from_compositions(self, spaces, rng):
        X, Y = spaces
        A = LinearOperator.from_matrix(X, Y, rng.normal(size=(3, 4)), form="components")
        assert A @ LinearOperator.identity(X) is A
        assert LinearOperator.identity(Y) @ A is A

    def test_zero(self, spaces, rng):
        X, Y = spaces
        zero = LinearOperator.zero(X, codomain=Y)
        check_operator(zero, rng=rng)
        assert np.allclose(zero(X.random(rng=rng)), np.zeros(3))

    def test_zero_disappears_from_sums(self, spaces, rng):
        X, Y = spaces
        A = LinearOperator.from_matrix(X, Y, rng.normal(size=(3, 4)), form="components")
        assert A + LinearOperator.zero(X, codomain=Y) is A
        assert LinearOperator.zero(X, codomain=Y) + A is A

    def test_sums_flatten(self, spaces, rng):
        X, Y = spaces
        ops = [
            LinearOperator.from_matrix(X, Y, rng.normal(size=(3, 4)), form="components")
            for _ in range(3)
        ]
        total = ops[0] + ops[1] + ops[2]
        assert isinstance(total, _Sum)
        assert len(total.terms) == 3

    def test_compositions_flatten(self, rng):
        X = EuclideanSpace(4)
        ops = [
            LinearOperator.from_matrix(X, X, rng.normal(size=(4, 4)), form="components")
            for _ in range(3)
        ]
        product = ops[0] @ ops[1] @ ops[2]
        assert isinstance(product, _Composition)
        assert len(product.factors) == 3

    def test_nested_scalings_fold(self, spaces, rng):
        X, Y = spaces
        A = LinearOperator.from_matrix(X, Y, rng.normal(size=(3, 4)), form="components")
        assert (2.0 * (3.0 * A)).alpha == pytest.approx(6.0)
        assert 0.5 * (2.0 * A) is A
        assert 1.0 * A is A

    def test_composition_adjoint_reverses(self, rng):
        X = EuclideanSpace(4)
        A = LinearOperator.from_matrix(X, X, rng.normal(size=(4, 4)), form="components")
        B = LinearOperator.from_matrix(X, X, rng.normal(size=(4, 4)), form="components")
        product = A @ B
        check_operator(product, rng=rng)
        assert product.adjoint.factors == (B.adjoint, A.adjoint)

    def test_repr_names_real_objects(self, spaces, rng):
        X, Y = spaces
        A = LinearOperator.from_matrix(X, Y, rng.normal(size=(3, 4)), form="components")
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
        generic = LinearOperator.from_matrix(
            X, X, rng.normal(size=(4, 4)), form="components"
        )

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
        A = LinearOperator.from_matrix(X, Y, M, form="components")
        assert np.allclose(A.matrix(form="components"), M)

    def test_galerkin_form_is_the_gram_weighted_matrix(self, rng):
        X, Y = make_weighted_space(), make_dense_metric_space()
        M = rng.normal(size=(Y.dim, X.dim))
        A = LinearOperator.from_matrix(X, Y, M, form="components")
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
        A = LinearOperator.from_matrix(X, Y, M, form="galerkin")
        check_operator(A, rng=rng)
        # Row i acts as the i-th derivative functional...
        x = X.random(rng=rng)
        assert np.allclose(A(x), M @ X.to_components(x))
        # ...and the adjoint of a basis covector is that row's representer.
        e0 = np.array([1.0, 0.0])
        assert np.allclose(X.to_components(A.adjoint(e0)), X.solve_gram(M[0]))

    def test_derivative_callables_match_the_assembled_operator(self, rng):
        """The matrix-free path must agree with the one it stands in for."""
        X, Y = make_weighted_space(), EuclideanSpace(2)
        M = rng.normal(size=(2, X.dim))
        assembled = LinearOperator.from_matrix(X, Y, M, form="galerkin")
        matrix_free = LinearOperator.from_derivative_callables(
            X,
            Y,
            lambda x: M @ X.to_components(x),
            lambda y: M.T @ y,
        )
        check_operator(matrix_free, rng=rng)

        x, y = X.random(rng=rng), Y.random(rng=rng)
        assert np.allclose(matrix_free(x), assembled(x))
        assert np.allclose(
            X.to_components(matrix_free.adjoint(y)),
            X.to_components(assembled.adjoint(y)),
        )

    def test_a_metric_free_adjoint_is_caught(self, rng):
        """The negative control: the guard must be watched to fail.

        Building the same operator through ``from_callables`` and handing it
        the derivative components *as if* they were a gradient is the mistake
        ``from_derivative_callables`` exists to prevent. If ``check_operator``
        passed here too, the guard would be proving nothing.
        """
        X, Y = make_weighted_space(), EuclideanSpace(2)
        M = rng.normal(size=(2, X.dim))
        wrong = LinearOperator.from_callables(
            X,
            Y,
            lambda x: M @ X.to_components(x),
            adjoint=lambda y: X.from_components(M.T @ y),  # no inverse metric
        )
        with pytest.raises(AssertionError):
            check_operator(wrong, rng=rng)

    def test_derivative_callables_needs_domain_coordinates_only(self, rng):
        """The codomain may be opaque; the domain is where the metric lives.

        The codomain weight appears in ``derivative_components`` because that
        callable is the derivative of ``(A x, y)_Y``, so the caller owns the
        codomain side of the inner product. Only the domain metric is applied
        for them.
        """
        X = make_weighted_space()
        weights = np.array([2.0, 3.0])
        Y = OpaqueSpace(weights)
        M = rng.normal(size=(2, X.dim))
        A = LinearOperator.from_derivative_callables(
            X,
            Y,
            lambda x: Opaque(M @ X.to_components(x)),
            lambda y: M.T @ (weights * y.data),
        )
        check_operator(A, rng=rng)

        with pytest.raises(TypeError, match="no coordinate map"):
            LinearOperator.from_derivative_callables(
                Y, X, lambda y: X.zero, lambda x: np.zeros(2)
            )

    def test_derivative_callables_checks_the_component_shape(self, rng):
        X, Y = make_weighted_space(), EuclideanSpace(2)
        A = LinearOperator.from_derivative_callables(
            X, Y, lambda x: np.zeros(2), lambda y: np.zeros(X.dim + 1)
        )
        with pytest.raises(ValueError, match="expected"):
            A.adjoint(np.ones(2))

    def test_the_two_fill_directions_agree(self, rng):
        """Rows cost dim(Y) adjoint applications, columns cost dim(X)."""
        X, Y = make_weighted_space(), make_dense_metric_space()
        M = rng.normal(size=(Y.dim, X.dim))
        A = LinearOperator.from_matrix(X, Y, M, form="components")
        for form in ("components", "galerkin"):
            by_columns = A.matrix(form=form, by="columns")
            by_rows = A.matrix(form=form, by="rows")
            assert np.allclose(by_columns, by_rows)
        assert np.allclose(A.matrix(form="components", by="rows"), M)

    def test_auto_fills_a_tall_operator_by_rows(self, rng):
        """The cheap direction is taken without being asked for.

        Counted rather than asserted: an observation operator has far fewer
        data than model components, and filling it by columns would apply the
        forward map once per component.
        """
        X, Y = make_weighted_space(), EuclideanSpace(2)
        assert Y.dim < X.dim
        M = rng.normal(size=(Y.dim, X.dim))
        forward, backward = 0, 0

        def value(x):
            nonlocal forward
            forward += 1
            return M @ X.to_components(x)

        def derivative(y):
            nonlocal backward
            backward += 1
            return M.T @ y

        A = LinearOperator.from_derivative_callables(X, Y, value, derivative)
        assert np.allclose(A.matrix(form="galerkin"), M)
        assert (forward, backward) == (0, Y.dim)

    def test_assembled_reproduces_the_operator(self, rng):
        X, Y = make_weighted_space(), EuclideanSpace(3)
        M = rng.normal(size=(3, X.dim))
        A = LinearOperator.from_derivative_callables(
            X, Y, lambda x: M @ X.to_components(x), lambda y: M.T @ y
        )
        B = A.assembled()
        check_operator(B, rng=rng)
        x, y = X.random(rng=rng), Y.random(rng=rng)
        assert np.allclose(A(x), B(x))
        assert np.allclose(X.to_components(A.adjoint(y)), X.to_components(B.adjoint(y)))

    def test_assembled_carries_the_traits(self, rng):
        X = make_weighted_space()
        S = spd_matrix(rng, X.dim)
        A = LinearOperator.self_adjoint(
            X, lambda x: X.solve_gram(S @ X.to_components(x))
        )
        assert Traits.SELF_ADJOINT & A.assembled().traits

    def test_matrix_requires_coordinates(self, rng):
        X = OpaqueSpace(np.array([1.0, 2.0, 3.0]))
        A = LinearOperator.self_adjoint(X, lambda x: x)
        with pytest.raises(TypeError, match="no coordinate map"):
            A.matrix()

    def test_require_coordinates_names_the_capability(self):
        with pytest.raises(TypeError, match="requires one"):
            require_coordinates(OpaqueSpace(np.array([1.0])))


class TestWithTraits:
    """Adding a claim must not cost the operator its class.

    The specialisation protocol dispatches on type, so an operator that arrives
    at a fast path as a wrapper does not take it. ``with_traits`` returned a
    wrapper, and that one fact was responsible for a diagonal covariance losing
    its exact log-determinant and a normal operator losing its factors.
    """

    def test_a_diagonal_operator_stays_diagonal(self):
        from pygeoinf2.algebra.diagonal import DiagonalLinearOperator

        space = EuclideanSpace(6)
        values = np.linspace(1.0, 2.0, 6)
        claimed = DiagonalLinearOperator(space, values).with_traits(
            Traits.POSITIVE_DEFINITE
        )

        assert isinstance(claimed, DiagonalLinearOperator)
        assert claimed.eigenvalues == pytest.approx(values)
        assert Traits.POSITIVE_DEFINITE & claimed.traits
        assert claimed.log_determinant == pytest.approx(float(np.sum(np.log(values))))

    def test_the_original_is_left_alone(self):
        from pygeoinf2.algebra.diagonal import DiagonalLinearOperator

        space = EuclideanSpace(4)
        original = DiagonalLinearOperator(space, np.ones(4))
        before = original.traits
        original.with_traits(Traits.POSITIVE_DEFINITE)
        assert original.traits == before

    def test_it_still_acts_the_same_way(self, rng):
        space = EuclideanSpace(5)
        matrix = rng.normal(size=(5, 5))
        matrix = matrix @ matrix.T + 5.0 * np.identity(5)
        operator = LinearOperator.from_matrix(space, space, matrix, form="components")
        claimed = operator.with_traits(Traits.SELF_ADJOINT)

        vector = rng.normal(size=5)
        assert claimed(vector) == pytest.approx(operator(vector))
        # SELF_ADJOINT means the adjoint is the operator itself, so the stale
        # adjoint the original had memoised must not have been carried over.
        assert claimed.adjoint is claimed

    def test_claiming_self_adjointness_off_the_diagonal_is_refused(self):
        operator = LinearOperator.zero(EuclideanSpace(3), codomain=EuclideanSpace(4))
        with pytest.raises(ValueError, match="SELF_ADJOINT"):
            operator.with_traits(Traits.SELF_ADJOINT)


class TestCompositionCosts:
    """Building an expression must not do the work of applying it."""

    def test_composing_with_an_inverse_costs_no_applications(self, rng):
        """The palindrome rule compares factors by identity, and asking an
        operator for its adjoint *builds* one. For the inverse of a direct
        solver that means extracting a second matrix and factorising it, so
        testing whether a composition happened to be a palindrome cost an
        O(n^3) detour at composition time -- on an expression that might never
        be applied. Measured at dimension 60: 60 applications for
        ``inverse @ B`` and none for ``B @ inverse``.
        """
        from pygeoinf2.numerics.solvers import LUSolver

        space = EuclideanSpace(40)
        matrix = rng.normal(size=(40, 40)) + 40.0 * np.identity(40)
        applications = 0

        def value(x):
            nonlocal applications
            applications += 1
            return matrix @ x

        def adjoint(y):
            nonlocal applications
            applications += 1
            return matrix.T @ y

        operator = LinearOperator.from_callables(space, space, value, adjoint=adjoint)
        other = LinearOperator.from_matrix(
            space, space, rng.normal(size=(40, 40)), form="components"
        )
        inverse = LUSolver()(operator)

        applications = 0
        _ = inverse @ other
        assert applications == 0
        _ = other @ inverse
        assert applications == 0

    def test_the_palindrome_traits_still_fire(self, rng):
        """Reading the link rather than building it must not cost deduction.
        The links that matter are made by writing ``A.adjoint`` in the
        expression, which happens before the composition is built."""
        space = EuclideanSpace(5)
        operator = LinearOperator.from_matrix(
            space, space, rng.normal(size=(5, 5)), form="components"
        )
        middle = LinearOperator.from_matrix(
            space,
            space,
            np.identity(5) * 2.0,
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_SEMIDEFINITE,
            form="components",
        )

        assert Traits.POSITIVE_SEMIDEFINITE & (operator @ operator.adjoint).traits
        assert (
            Traits.POSITIVE_SEMIDEFINITE & (operator @ middle @ operator.adjoint).traits
        )
        assert Traits.SELF_ADJOINT & (operator.adjoint @ operator).traits


class TestMatrixLinearOperator:
    """An operator built from a matrix must be able to produce one."""

    @pytest.fixture(params=["euclidean", "weighted", "dense-metric"])
    def domain(self, request):
        from .conftest import make_dense_metric_space, make_weighted_space

        return {
            "euclidean": lambda: EuclideanSpace(4),
            "weighted": make_weighted_space,
            "dense-metric": make_dense_metric_space,
        }[request.param]()

    @pytest.mark.parametrize("form", ["components", "galerkin"])
    def test_the_matrix_round_trips_in_its_own_form(self, domain, form, rng):
        codomain = EuclideanSpace(3)
        matrix = rng.normal(size=(3, domain.dim))
        operator = LinearOperator.from_matrix(domain, codomain, matrix, form=form)
        assert operator.matrix(form=form) == pytest.approx(matrix)

    @pytest.mark.parametrize("form", ["components", "galerkin"])
    def test_it_agrees_with_the_probed_route(self, domain, form, rng):
        """Both forms and both diagonals, against the generic implementation
        that fills the matrix in by applying the operator. A dense Gram is what
        separates the two representations, so it has to be one of the cases."""
        codomain = EuclideanSpace(3)
        matrix = rng.normal(size=(3, domain.dim))
        stored = LinearOperator.from_matrix(domain, codomain, matrix, form=form)
        probed = LinearOperator.from_callables(
            domain,
            codomain,
            stored,
            adjoint=lambda y, s=stored: s.adjoint(y),
        )

        for wanted in ("components", "galerkin"):
            assert stored.matrix(form=wanted) == pytest.approx(
                probed.matrix(form=wanted)
            )
            assert stored.diagonals(offsets=(0,), form=wanted) == pytest.approx(
                probed.diagonals(offsets=(0,), form=wanted)
            )

    def test_the_matrix_is_read_rather_than_re_derived(self, rng):
        """It was captured in a closure, so an operator built from a matrix
        could not produce one: ``matrix()`` re-derived it by ``dim``
        applications, and so did every direct solver before factorising."""
        space = EuclideanSpace(40)
        matrix = rng.normal(size=(40, 40))
        applications = 0

        class Counting(EuclideanSpace):
            pass

        operator = LinearOperator.from_matrix(space, space, matrix, form="components")
        original = type(operator)._value

        def counting(self, x):
            nonlocal applications
            applications += 1
            return original(self, x)

        type(operator)._value = counting
        try:
            operator.matrix(form="components")
            operator.diagonals(offsets=(0,), form="components")
            assembled = operator.assembled()
        finally:
            type(operator)._value = original

        assert applications == 0
        assert assembled is operator

    def test_a_sparse_matrix_stays_sparse(self, rng):
        import scipy.sparse as sp

        space = EuclideanSpace(50)
        sparse = sp.diags([np.ones(49), np.full(50, 2.0), np.ones(49)], [-1, 0, 1])
        operator = LinearOperator.from_matrix(space, space, sparse, form="components")
        assert sp.issparse(operator.stored_matrix)
        assert operator.stored_matrix.nnz == 148
        # And it still applies correctly.
        probe = rng.normal(size=50)
        assert operator(probe) == pytest.approx(sparse @ probe)

    def test_the_form_must_be_given(self, rng):
        space = EuclideanSpace(3)
        with pytest.raises(TypeError):
            LinearOperator.from_matrix(space, space, np.identity(3))
        with pytest.raises(ValueError, match="components' or 'galerkin"):
            LinearOperator.from_matrix(space, space, np.identity(3), form="derivative")


class TestFormalAdjointLift:
    """Reusing an operator, and its adjoint, under a different inner product."""

    def test_it_lifts_between_sobolev_orders(self, rng):
        from pygeoinf2.symmetric_space.sphere import Lebesgue, Sobolev
        from pygeoinf2.testing import check_operator

        pytest.importorskip("pyshtools")
        base = Lebesgue(12)
        target = Sobolev(12, 2.0, 0.2)
        operator = LinearOperator.from_matrix(
            base, base, rng.normal(size=(base.dim, base.dim)), form="components"
        )
        lifted = LinearOperator.from_formal_adjoint(target, target, operator)

        # Same action, adjoint taken in the new inner product.
        probe = target.random(rng=rng)
        assert target.grid_values(lifted(probe)) == pytest.approx(
            target.grid_values(operator(probe))
        )
        check_operator(lifted, rng=rng)

    def test_it_lifts_onto_a_direct_sum_with_a_euclidean_summand(self, rng):
        """pyslfp's central idiom, which had no equivalent at all.

        Its fingerprint operator maps a ``EuclideanSpace(2)`` of parameters into
        a direct sum of three Sobolev fields and a pair of scalars, and it is
        built on the L2 spaces and lifted. ``lift_formal_adjoint`` accepts only
        a single symmetric space on each side, so neither shape was reachable.
        """
        from pygeoinf2.algebra.direct_sum import DirectSum
        from pygeoinf2.symmetric_space.sphere import Lebesgue, Sobolev
        from pygeoinf2.testing import check_operator

        pytest.importorskip("pyshtools")
        base_field, field = Lebesgue(10), Sobolev(10, 2.0, 0.2)
        scalars = EuclideanSpace(2)
        base = DirectSum([base_field, base_field, base_field, scalars])
        target = DirectSum([field, field, field, scalars])

        columns = [base.random(rng=rng) for _ in range(2)]

        def value(c):
            out = base.zero()
            for weight, column in zip(c, columns):
                out = base.axpy(float(weight), column, out)
            return out

        def adjoint(y):
            return np.array([base.inner_product(column, y) for column in columns])

        operator = LinearOperator.from_callables(scalars, base, value, adjoint=adjoint)
        lifted = LinearOperator.from_formal_adjoint(scalars, target, operator)

        assert lifted.domain is scalars
        check_operator(lifted, rng=rng)

    def test_the_forward_action_costs_nothing_extra(self, rng):
        """Only the adjoint is reweighted, so on a shared grid the forward
        action is the operator's own and nothing else. It used to round-trip
        through components on both sides, four transforms per application doing
        no work."""
        pyshtools = pytest.importorskip("pyshtools")
        from pygeoinf2.symmetric_space.sphere import Lebesgue, Sobolev

        counts = {"n": 0}
        originals = {}
        for name in ("SHExpandDH", "MakeGridDH"):
            originals[name] = getattr(pyshtools.expand, name)

            def wrap(inner):
                def counted(*args, **kwargs):
                    counts["n"] += 1
                    return inner(*args, **kwargs)

                return counted

            setattr(pyshtools.expand, name, wrap(originals[name]))
        try:
            base, target = Lebesgue(12), Sobolev(12, 2.0, 0.2)
            lifted = LinearOperator.from_formal_adjoint(
                target, target, LinearOperator.identity(base)
            )
            probe = target.random(rng=rng)
            counts["n"] = 0
            lifted(probe)
            forward = counts["n"]
        finally:
            for name, inner in originals.items():
                setattr(pyshtools.expand, name, inner)
        assert forward == 0

    def test_a_mass_weighted_space_lifts_without_coordinates(self, rng):
        """The coordinate-free route, which is what the construction is for:
        a mass-weighted space over a backend with no component map."""
        from pygeoinf2.algebra.diagonal import DiagonalLinearOperator
        from pygeoinf2.algebra.spaces import MassWeightedSpace
        from pygeoinf2.testing import check_operator

        base = EuclideanSpace(4)
        mass = DiagonalLinearOperator(base, np.array([1.0, 4.0, 9.0, 0.25]))
        weighted = MassWeightedSpace(base, mass)
        operator = LinearOperator.from_matrix(
            base, base, rng.normal(size=(4, 4)), form="components"
        )
        lifted = LinearOperator.from_formal_adjoint(weighted, weighted, operator)

        probe = weighted.random(rng=rng)
        assert lifted(probe) == pytest.approx(operator(probe))
        check_operator(lifted, rng=rng)

    def test_mismatched_dimensions_are_refused(self, rng):
        base = EuclideanSpace(4)
        operator = LinearOperator.from_matrix(
            base, base, np.identity(4), form="components"
        )
        with pytest.raises(ValueError, match="same vectors"):
            LinearOperator.from_formal_adjoint(EuclideanSpace(5), base, operator)


class TestColumnOperator:
    """``from_vectors`` on a coordinate space stores the components and
    takes its adjoint through one analysis and one metric application."""

    def test_it_agrees_with_the_coordinate_free_construction(self, rng):
        from pygeoinf2.algebra.spaces import CoordinateSpace

        space = make_dense_metric_space(20)
        vectors = [space.random(rng=rng) for _ in range(5)]
        fast = LinearOperator.from_vectors(space, vectors)
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            CoordinateSpace, "uses_component_fast_paths", property(lambda self: False)
        )
        try:
            slow = LinearOperator.from_vectors(space, vectors)
        finally:
            monkeypatch.undo()
        c = rng.standard_normal(5)
        assert space.norm(space.subtract(fast(c), slow(c))) < 1e-12
        y = space.random(rng=rng)
        assert fast.adjoint(y) == pytest.approx(slow.adjoint(y), rel=1e-12, abs=1e-12)
        check_operator(fast, rng=rng)

    def test_it_can_be_built_from_columns(self, rng):
        space = make_dense_metric_space(20)
        columns = rng.standard_normal((20, 4))
        operator = LinearOperator.from_component_columns(space, columns)
        assert np.array_equal(operator.columns, columns)
        assert len(operator.vectors) == 4
        check_operator(operator, rng=rng)
        with pytest.raises(ValueError):
            LinearOperator.from_component_columns(space, np.zeros((20, 0)))
        with pytest.raises(ValueError):
            LinearOperator.from_component_columns(space, np.zeros((7, 2)))

    def test_the_strict_space_falls_back(self, rng):
        from .doubles import StrictSpace

        strict = StrictSpace(make_weighted_space())
        operator = LinearOperator.from_vectors(strict, [strict.random(rng=rng)])
        assert not hasattr(operator, "columns")
        operator(np.ones(1))
