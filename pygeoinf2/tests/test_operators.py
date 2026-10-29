"""Operators: adjoints, traits, the algebra, and the evaluation paths."""

import numpy as np
import pytest

from pygeoinf2.algebra.nodes import _Composition, _Sum
from pygeoinf2.algebra.operators import (
    AffineOperator,
    LinearOperator,
    require_coordinates,
)
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.testing import (
    check_operator,
    check_traits,
)
from pygeoinf2.traits import Traits

from .conftest import make_dense_metric_space, make_weighted_space, spd_matrix
from .doubles import Opaque, OpaqueSpace


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

    def test_adjoint_is_memoized(self, spaces, rng):
        X, Y = spaces
        A = LinearOperator.from_matrix(X, Y, rng.normal(size=(3, 4)), form="components")
        assert A.adjoint is A.adjoint
        assert A.adjoint.adjoint is A

    def test_self_adjoint_operator_is_its_own_adjoint(self, rng):
        X = make_weighted_space()
        # Symmetric in the Galerkin form is what self-adjoint means on a
        # weighted space; the same matrix as a components form would not be.
        C = LinearOperator.from_matrix(
            X, X, spd_matrix(rng, X.dim), traits=Traits.SELF_ADJOINT, form="galerkin"
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


class TestProbedAdjoint:
    """A missing adjoint is refused rather than derived silently by probing
    every basis vector on every application. The opt-in assembles the matrix once
    and derives the adjoint from it, and its cost is in its name."""

    @staticmethod
    def action_only(X, Y, matrix):
        return LinearOperator.from_callables(
            X, Y, lambda x: Y.from_components(matrix @ X.to_components(x))
        )

    def test_it_agrees_with_the_matrix_built_adjoint_on_a_dense_metric(self, rng):
        X, Y = make_dense_metric_space(5), make_weighted_space()
        matrix = rng.normal(size=(Y.dim, X.dim))
        probed = self.action_only(X, Y, matrix).with_probed_adjoint()
        reference = LinearOperator.from_matrix(X, Y, matrix, form="components")
        check_operator(probed, rng=rng)
        for _ in range(3):
            y = Y.random(rng=rng)
            assert X.norm(
                X.subtract(probed.adjoint(y), reference.adjoint(y))
            ) < 1e-10 * X.norm(reference.adjoint(y))

    def test_the_matrix_is_assembled_once(self, rng):
        X, Y = make_dense_metric_space(6), make_weighted_space()
        matrix = rng.normal(size=(Y.dim, X.dim))
        applications = []
        operator = LinearOperator.from_callables(
            X,
            Y,
            lambda x: (
                applications.append(1),
                Y.from_components(matrix @ X.to_components(x)),
            )[1],
        ).with_probed_adjoint()
        y = Y.random(rng=rng)
        operator.adjoint(y)
        assert len(applications) == X.dim
        operator.adjoint(y)
        operator.matrix(form="galerkin")
        assert len(applications) == X.dim

    def test_the_refusal_names_the_opt_in(self, spaces):
        X, Y = spaces
        A = LinearOperator.from_callables(X, Y, lambda x: np.zeros(3))
        with pytest.raises(NotImplementedError, match="with_probed_adjoint"):
            A.adjoint(np.zeros(3))

    def test_a_wide_operator_gets_its_matrix_by_rows_after_opting_in(self, rng):
        """Filling by rows goes through the adjoint, which an operator without
        one cannot do; with the opt-in it can, and gets the assembled matrix.
        The default fill falls back to the columns instead."""
        X, Y = EuclideanSpace(8), EuclideanSpace(3)
        matrix = rng.normal(size=(3, 8))
        bare = self.action_only(X, Y, matrix)
        with pytest.raises(NotImplementedError):
            bare.matrix(by="rows")
        assert np.allclose(bare.matrix(), matrix)
        assert np.allclose(bare.with_probed_adjoint().matrix(by="rows"), matrix)

    def test_a_space_without_coordinates_is_refused(self):
        X = OpaqueSpace(np.ones(4))
        A = LinearOperator.from_callables(X, X, lambda x: x)
        with pytest.raises(TypeError, match="coordinate"):
            A.with_probed_adjoint()


class TestTraitPropagation:
    def test_gramian_is_semidefinite(self, spaces, rng):
        X, Y = spaces
        A = LinearOperator.from_matrix(X, Y, rng.normal(size=(3, 4)), form="components")
        G = A @ A.adjoint
        assert Traits.SELF_ADJOINT & G.traits
        assert Traits.POSITIVE_SEMIDEFINITE & G.traits
        check_traits(G, rng=rng)

    def test_congruence_is_recognized_after_the_fact(self, spaces, rng):
        """A @ C @ A.adjoint, built in two steps.

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
        """Structure survives `generic + special` as it does the reverse."""
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
        # the right.
        assert isinstance(generic + a, _Sum)
        assert isinstance(a + generic, _Sum)
        assert type(generic + a) is type(a + generic)


class TestScipyBridge:
    """The matrix-free scipy bridge, ``as_scipy`` and its converse:
    the metric factors of each form are put in and taken out correctly on a
    space whose Gram matrix is not diagonal."""

    @pytest.fixture
    def pieces(self, rng):
        X = make_dense_metric_space(4)
        Y = make_weighted_space()
        A = LinearOperator.from_matrix(
            X, Y, rng.normal(size=(Y.dim, X.dim)), form="components"
        )
        return X, Y, A

    @pytest.mark.parametrize("form", ["components", "galerkin"])
    def test_it_agrees_with_the_dense_matrix_and_its_transpose(self, pieces, form, rng):
        X, Y, A = pieces
        bridge = A.as_scipy(form=form)
        dense = A.matrix(form=form)
        assert bridge.shape == dense.shape
        c = rng.normal(size=X.dim)
        y = rng.normal(size=Y.dim)
        assert np.allclose(bridge.matvec(c), dense @ c)
        assert np.allclose(bridge.rmatvec(y), dense.T @ y)
        block = rng.normal(size=(X.dim, 3))
        assert np.allclose(bridge.matmat(block), dense @ block)
        assert np.allclose(bridge.rmatmat(np.eye(Y.dim)), dense.T)

    def test_it_feeds_scipy_eigsh_on_a_dense_metric(self, rng):
        """The Galerkin form of a self-adjoint operator is symmetric, so
        scipy's Lanczos with the Gram matrix as the mass recovers the
        operator's own eigenvalues -- the generalized ones of (S, G), which
        the component matrix, unsymmetric on this metric, would not give."""
        from scipy.linalg import eigh
        from scipy.sparse.linalg import eigsh

        X = make_dense_metric_space(6)
        raw = rng.normal(size=(6, 6))
        symmetric = raw @ raw.T + 6.0 * np.eye(6)
        A = LinearOperator.from_matrix(
            X,
            X,
            symmetric,
            form="galerkin",
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
        )
        bridge = A.as_scipy()  # auto: Galerkin for a self-adjoint operator
        largest = eigsh(
            bridge, k=1, M=X.gram_matrix(), which="LA", return_eigenvectors=False
        )
        expected = eigh(symmetric, X.gram_matrix(), eigvals_only=True).max()
        assert largest[0] == pytest.approx(expected, rel=1e-8)

    @pytest.mark.parametrize("form", ["components", "galerkin"])
    def test_the_converse_round_trips(self, pieces, form, rng):
        X, Y, A = pieces
        back = LinearOperator.from_scipy(X, Y, A.as_scipy(form=form), form=form)
        x = X.random(rng=rng)
        assert Y.norm(Y.subtract(back(x), A(x))) < 1e-10 * max(Y.norm(A(x)), 1.0)
        y = Y.random(rng=rng)
        assert X.norm(X.subtract(back.adjoint(y), A.adjoint(y))) < 1e-10 * max(
            X.norm(A.adjoint(y)), 1.0
        )
        check_operator(back, rng=rng)

    def test_a_scipy_operator_from_outside(self, rng):
        """An array wrapped by scipy, entering as the components matrix."""
        from scipy.sparse.linalg import aslinearoperator

        X, Y = make_dense_metric_space(4), EuclideanSpace(3)
        matrix = rng.normal(size=(3, 4))
        A = LinearOperator.from_scipy(X, Y, aslinearoperator(matrix), form="components")
        assert np.allclose(A.matrix(form="components"), matrix)
        check_operator(A, rng=rng)
        with pytest.raises(ValueError, match="shape"):
            LinearOperator.from_scipy(
                X, Y, aslinearoperator(matrix.T), form="components"
            )


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

    The specialization protocol dispatches on type, so an operator that arrives
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
        # adjoint the original had memoized must not have been carried over.
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
        solver that means extracting a second matrix and factorizing it, so
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
        applications, and so did every direct solver before factorizing."""
        space = EuclideanSpace(40)
        matrix = rng.normal(size=(40, 40))
        applications = 0

        operator = LinearOperator.from_matrix(space, space, matrix, form="components")
        original = type(operator)._value

        def counting(self, x):
            nonlocal applications
            applications += 1
            return original(self, x)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(type(operator), "_value", counting)
            operator.matrix(form="components")
            operator.diagonals(offsets=(0,), form="components")
            assembled = operator.assembled()

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
        action is the operator's own and nothing else: no round trip through
        components on both sides, which would be four transforms per
        application doing no work."""
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

    def test_the_base_may_be_an_equal_but_distinct_space(self, rng):
        """``EuclideanSpace(n)`` is minted freely -- ``from_vectors`` and
        ``coordinate_projection`` both do it -- so a lift over a mass-weighted
        space must not require the *same object* as the operator's base."""
        from pygeoinf2.algebra.spaces import MassWeightedSpace
        from pygeoinf2.testing import check_operator

        base = make_dense_metric_space(4)
        symmetric = rng.normal(size=(4, 4))
        symmetric = symmetric @ symmetric.T + 4.0 * np.identity(4)
        mass = LinearOperator.from_matrix(
            base, base, symmetric, form="galerkin"
        ).with_traits(Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE)
        weighted = MassWeightedSpace(base, mass)

        # A second, equal base: what the caller normally has to hand.
        other = make_dense_metric_space(4)
        assert other == base and other is not base
        operator = LinearOperator.from_matrix(
            other, other, rng.normal(size=(4, 4)), form="components"
        )
        lifted = LinearOperator.from_formal_adjoint(weighted, other, operator)
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


class TestLinearisationIdentity:
    """``Linearization`` and ``QuadraticModel`` are frozen dataclasses whose
    fields hold *vectors*. With the default ``eq=True`` the generated ``==``
    compared them field by field, which on an array-backed space returns an
    array whose truth value is an error, and the ``__hash__`` that
    ``frozen=True`` generates alongside it raised ``TypeError: unhashable
    type: 'numpy.ndarray'``. ``eq=False`` gives identity for both, which is
    the only equality a record of one evaluation can honestly offer."""

    @staticmethod
    def model(rng):
        from pygeoinf2.algebra.linearization import Linearization, QuadraticModel

        space = make_dense_metric_space(4)
        derivative = LinearOperator.from_matrix(
            space, space, rng.standard_normal((4, 4)), form="components"
        )
        point = space.random(rng=rng)
        from pygeoinf2.algebra.operators import LinearFunctional

        functional = LinearFunctional.from_representer(space, space.random(rng=rng))
        return (
            Linearization(point, space.random(rng=rng), derivative),
            QuadraticModel(point, 1.0, functional),
        )

    @pytest.mark.parametrize("index", [0, 1])
    def test_it_hashes(self, index, rng):
        record = self.model(rng)[index]
        assert hash(record) == hash(record)
        assert {record: 1}[record] == 1

    @pytest.mark.parametrize("index", [0, 1])
    def test_equality_is_identity_and_does_not_raise(self, index, rng):
        record = self.model(rng)[index]
        other = self.model(rng)[index]
        assert record == record
        assert record != other
        assert bool(record != other) is True


class TestFunctionalConstructorSignature:
    """The codomain of a functional is always ``Reals``. It stays positional
    -- ``LinearOperator.__init__`` calls the next ``__init__`` in the MRO as
    ``__init__(domain, codomain)``, and twenty-odd subclasses call
    ``super().__init__(domain)`` -- but it is now positional-*only*, so the
    parameter's name is not part of the API."""

    def test_the_codomain_is_positional_only(self):
        import inspect

        from pygeoinf2.algebra.operators import Functional, LinearFunctional

        for cls in (Functional, LinearFunctional):
            parameter = inspect.signature(cls.__init__).parameters["codomain"]
            assert parameter.kind is inspect.Parameter.POSITIONAL_ONLY, cls.__name__

    def test_both_call_shapes_still_work(self):
        from pygeoinf2.algebra.operators import Functional
        from pygeoinf2.algebra.spaces import Reals

        space = EuclideanSpace(3)
        assert Functional(space).codomain == Reals()
        assert Functional(space, Reals()).codomain == Reals()
        with pytest.raises(ValueError, match="maps into Reals"):
            Functional(space, space)


class TestTheChecksReportWhatTheyMeasured:
    """``check_*`` take ``rtol=`` and return a ``CheckReport``. The use is a backend
    whose adjoint is consistent only to discretisation level: pass the
    ``rtol`` it can meet, read the defect back, record it falling under
    refinement."""

    @staticmethod
    def imperfect(space, defect):
        """A self-map whose declared adjoint is off by ``defect``, relatively,
        on a non-diagonal metric -- the shape of a continuous adjoint."""
        values = np.linspace(1.0, 2.0, space.dim)

        def act(x):
            return space.from_components(values * space.to_components(x))

        def adjoint(y):
            g = space.gram_matrix()
            c = space.to_components(y)
            exact = np.linalg.solve(g, values * (g @ c))
            return space.from_components(exact * (1.0 + defect))

        return LinearOperator.from_callables(space, space, act, adjoint=adjoint)

    def test_an_exact_operator_reports_a_roundoff_defect(self, rng):
        space = make_dense_metric_space(5)
        A = LinearOperator.from_matrix(
            space, space, rng.normal(size=(5, 5)), form="components"
        )
        report = check_operator(A, rng=rng)
        assert report.check == "check_operator"
        assert "the adjoint satisfies (A x, y) == (x, A* y)" in report.defects
        assert report.worst < 1e-9

    def test_a_loose_rtol_passes_and_reports_the_defect(self, rng):
        space = make_dense_metric_space(5)
        A = self.imperfect(space, 1e-4)
        report = check_operator(A, rng=rng, rtol=1e-2)
        defect = report.defects["the adjoint satisfies (A x, y) == (x, A* y)"]
        assert 1e-6 < defect < 1e-2

    def test_the_default_rtol_still_fails_it(self, rng):
        space = make_dense_metric_space(5)
        A = self.imperfect(space, 1e-4)
        with pytest.raises(AssertionError, match="adjoint"):
            check_operator(A, rng=rng)

    def test_every_check_returns_a_report(self, rng):
        from pygeoinf2.testing import CheckReport, check_coordinates, check_space

        space = make_dense_metric_space(4)
        for report in (
            check_space(space, rng=rng),
            check_coordinates(space, rng=rng),
            check_traits(
                LinearOperator.from_matrix(
                    space, space, np.identity(4), form="components"
                ).with_traits(Traits.SELF_ADJOINT),
                rng=rng,
            ),
        ):
            assert isinstance(report, CheckReport)
            assert report.worst < 1e-8

    def test_an_empty_report_has_a_zero_worst(self):
        from pygeoinf2.testing import CheckReport

        assert CheckReport("check_space").worst == 0.0


class TestAFunctionalBuiltFromOneLinearization:
    """``from_callables(linearize=)``: one call yields the value and the
    derivative together, so ``at()`` costs one evaluation -- the arrangement
    an objective over a PDE solve needs -- while ``value`` stays the cheap
    route a backtracking search pays for."""

    @staticmethod
    def build(space, counts):
        from pygeoinf2.algebra.linearization import QuadraticModel
        from pygeoinf2.algebra.operators import Functional, LinearFunctional

        def value(x):
            counts["value"] += 1
            return space.squared_norm(x)

        def linearize(x):
            counts["linearize"] += 1
            return QuadraticModel(
                x,
                space.squared_norm(x),
                LinearFunctional.from_representer(space, space.scale(2.0, x)),
            )

        return Functional.from_callables(space, value, linearize=linearize)

    def test_at_is_the_one_call(self, rng):
        space = EuclideanSpace(4)
        counts = {"value": 0, "linearize": 0}
        f = self.build(space, counts)
        assert f.has_derivative
        x = space.random(rng=rng)
        model = f.at(x)
        assert counts == {"value": 0, "linearize": 1}
        assert model.value == pytest.approx(space.squared_norm(x))
        assert space.to_components(model.gradient) == pytest.approx(
            2.0 * space.to_components(x)
        )

    def test_the_value_stays_the_cheap_route(self, rng):
        space = EuclideanSpace(4)
        counts = {"value": 0, "linearize": 0}
        f = self.build(space, counts)
        f(space.random(rng=rng))
        assert counts == {"value": 1, "linearize": 0}

    def test_the_derivative_falls_back_to_it(self, rng):
        space = EuclideanSpace(4)
        counts = {"value": 0, "linearize": 0}
        f = self.build(space, counts)
        x = space.random(rng=rng)
        df = f.derivative(x)
        assert counts == {"value": 0, "linearize": 1}
        probe = space.random(rng=rng)
        assert df(probe) == pytest.approx(2.0 * space.inner_product(x, probe))


class TestAssembledStoresTheComponentsForm:
    """``assembled()`` exists to make applications cheap, and the components
    form is the one in which an application is a bare matrix product. Storing
    the Galerkin form ``G A_c`` meant undoing the metric again on every
    forward application: measured at dimension 2000 over a dense Gram, 3.80 ms
    per application against 0.77 ms, and CG 624 ms against 511 ms.

    The Galerkin form is still what a symmetric factorization wants, and it
    still gets it — converted once, at build time, through ``_in_form``.
    """

    @staticmethod
    def problem(rng, dim=12):
        space = make_dense_metric_space(dim)
        symmetric = rng.standard_normal((dim, dim))
        symmetric = symmetric @ symmetric.T + dim * np.identity(dim)
        stored = LinearOperator.from_matrix(
            space,
            space,
            symmetric,
            form="galerkin",
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
        )
        # Matrix-free, so that `assembled` has something to assemble.
        free = LinearOperator.self_adjoint(
            space,
            lambda x, s=stored: s(x),
            traits=Traits.POSITIVE_DEFINITE,
        )
        return space, symmetric, free

    def test_it_stores_components_and_reproduces_the_operator(self, rng):
        space, _, free = self.problem(rng)
        assembled = free.assembled()
        assert assembled.stored_form == "components"
        probe = space.random(rng=rng)
        assert space.to_components(assembled(probe)) == pytest.approx(
            space.to_components(free(probe))
        )
        check_operator(assembled, rng=rng)
        check_traits(assembled, rng=rng)

    def test_a_forward_application_no_longer_undoes_the_metric(self, rng):
        """The whole point: one ``solve_gram`` per application, gone."""
        space, _, free = self.problem(rng)
        assembled = free.assembled()
        solves = 0
        original = type(space).solve_gram

        def counting(self, c):
            nonlocal solves
            solves += 1
            return original(self, c)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(type(space), "solve_gram", counting)
            assembled(space.random(rng=rng))
        assert solves == 0

    def test_a_direct_solver_still_gets_the_symmetric_matrix(self, rng):
        """Converted once through ``_in_form``, against the factorization's
        own cost."""
        from pygeoinf2.numerics import CholeskySolver

        space, symmetric, free = self.problem(rng)
        assembled = free.assembled()
        assert assembled.matrix(form="galerkin") == pytest.approx(symmetric)

        right = space.random(rng=rng)
        solution = CholeskySolver()(assembled)(right)
        residual = space.subtract(assembled(solution), right)
        assert space.norm(residual) < 1e-9 * space.norm(right)


class TestChecksTakeAMeasure:
    """``measure=`` on every check: probes drawn from a measure on the
    space rather than as white noise, which on a function space is rough in
    every mode and not what an operator will meet."""

    @pytest.fixture
    def field(self, rng):
        from pygeoinf2.symmetric_space import Lebesgue as BoxLebesgue

        X = BoxLebesgue((16,), lengths=(1.0,))
        measure = X.invariant_measure(lambda k: 1.0 / (1.0 + k) ** 2)
        return X, measure

    def test_the_probes_come_from_the_measure(self, field, rng, monkeypatch):
        from pygeoinf2.testing import check_operator, check_space

        X, measure = field
        drawn = {"n": 0}
        real = measure.sample

        def counting(*, rng=None):
            drawn["n"] += 1
            return real(rng=rng)

        monkeypatch.setattr(measure, "sample", counting)
        A = X.spectral_operator(1.0 / (1.0 + X.degrees.astype(float)))
        check_operator(A, rng=rng, measure=measure)
        assert drawn["n"] > 0
        before = drawn["n"]
        check_space(X, rng=rng, measure=measure)
        assert drawn["n"] > before

    def test_a_measure_on_another_space_is_refused(self, field, rng):
        from pygeoinf2.testing import check_operator

        X, measure = field
        Y = EuclideanSpace(3)
        B = LinearOperator.from_matrix(Y, Y, np.eye(3), form="components")
        with pytest.raises(ValueError, match="lives on"):
            check_operator(B, rng=rng, measure=measure)

    def test_a_wrong_adjoint_still_fails_under_smooth_probes(self, field, rng):
        from pygeoinf2.testing import check_operator

        X, measure = field
        values = 1.0 / (1.0 + X.degrees.astype(float))
        wrong = LinearOperator.from_callables(
            X,
            X,
            lambda x: X.from_components(values * X.to_components(x)),
            adjoint=lambda y: X.from_components(2.0 * values * X.to_components(y)),
        )
        with pytest.raises(AssertionError, match="adjoint"):
            check_operator(wrong, rng=rng, measure=measure)

    def test_the_derivative_checks_take_it_too(self, field, rng):
        from pygeoinf2.algebra.operators import Functional
        from pygeoinf2.testing import check_affine, check_derivative, check_gradient

        X, measure = field
        f = Functional.from_callables(
            X, lambda x: X.squared_norm(x), gradient=lambda x: X.scale(2.0, x)
        )
        point = measure.sample(rng=rng)
        check_gradient(f, point, rng=rng, measure=measure)
        A = X.spectral_operator(1.0 / (1.0 + X.degrees.astype(float)))
        check_derivative(A, point, rng=rng, measure=measure)
        check_affine(
            AffineOperator(A, measure.sample(rng=rng)), rng=rng, measure=measure
        )


class TestPublicStructureAccessors:
    """What an operator will say about itself without being applied: the
    matrix it already holds, its action on component arrays, and whether it
    is the identity. Read-only views on the protected extension hooks."""

    def test_known_matrix_is_the_stored_matrix_or_none(self, rng):
        X = make_dense_metric_space(4)
        matrix = rng.normal(size=(4, 4))
        stored = LinearOperator.from_matrix(X, X, matrix, form="components")
        assert np.allclose(stored.known_matrix(form="components"), matrix)
        assert np.allclose(
            stored.known_matrix(form="galerkin"), X.gram_matrix() @ matrix
        )
        callable_only = LinearOperator.from_callables(
            X, X, lambda x: X.from_components(matrix @ X.to_components(x))
        )
        assert callable_only.known_matrix(form="components") is None
        with pytest.raises(ValueError, match="form"):
            stored.known_matrix(form="dense")

    def test_from_callables_takes_the_component_actions(self, rng):
        """Supplied through the constructor, not written onto the result."""
        X = make_dense_metric_space(4)
        matrix = rng.normal(size=(4, 4))
        plain = LinearOperator.from_callables(
            X, X, lambda x: X.from_components(matrix @ X.to_components(x))
        )
        assert plain.components_action() is None
        assert plain.components_adjoint_action() is None
        adjoint_matrix = X.solve_gram_to_columns(matrix.T @ X.gram_matrix())
        fused = LinearOperator.from_callables(
            X,
            X,
            lambda x: X.from_components(matrix @ X.to_components(x)),
            adjoint=lambda y: X.from_components(adjoint_matrix @ X.to_components(y)),
            components_action=lambda c: matrix @ c,
            components_adjoint_action=lambda c: adjoint_matrix @ c,
        )
        check_operator(fused, rng=rng)
        x = X.random(rng=rng)
        assert np.allclose(
            fused.components_action()(X.to_components(x)), X.to_components(fused(x))
        )
        assert np.allclose(
            fused.components_adjoint_action()(X.to_components(x)),
            X.to_components(fused.adjoint(x)),
        )

    def test_is_identity_is_true_only_for_the_identity(self, rng):
        X = make_dense_metric_space(3)
        assert LinearOperator.identity(X).is_identity
        assert not LinearOperator.from_matrix(
            X, X, np.identity(3), form="components"
        ).is_identity
        assert not LinearOperator.zero(X).is_identity
        assert not (2.0 * LinearOperator.identity(X)).is_identity
        assert not LinearOperator.from_callables(X, X, lambda x: x).is_identity


class TestCheckAffineForwardsItsKnobs:
    """The nested ``check_operator`` on the linear part is held to the caller's
    tolerance and probed from the caller's measure, not to the defaults."""

    def test_the_tolerance_reaches_the_linear_part(self, rng):
        X = make_dense_metric_space(4)
        matrix = rng.normal(size=(4, 4))
        exact_adjoint = X.solve_gram_to_columns(matrix.T @ X.gram_matrix())
        slightly_off = exact_adjoint * (1.0 + 1e-6)
        inexact = LinearOperator.from_callables(
            X,
            X,
            lambda x: X.from_components(matrix @ X.to_components(x)),
            adjoint=lambda y: X.from_components(slightly_off @ X.to_components(y)),
        )
        from pygeoinf2.testing import check_affine

        affine = AffineOperator(inexact, X.random(rng=rng))
        with pytest.raises(AssertionError, match="adjoint"):
            check_affine(affine, rng=rng)
        report = check_affine(affine, rng=rng, rtol=1e-4)
        assert report.defects["the adjoint satisfies (A x, y) == (x, A* y)"] > 1e-8

    def test_the_measure_reaches_the_linear_part(self, rng, monkeypatch):
        from pygeoinf2.probability.gaussian import GaussianMeasure
        from pygeoinf2.testing import check_affine

        X = make_dense_metric_space(4)
        identity = LinearOperator.identity(X)
        measure = GaussianMeasure(X, covariance=identity, covariance_factor=identity)
        drawn = {"n": 0}
        real = measure.sample

        def counting(*, rng=None):
            drawn["n"] += 1
            return real(rng=rng)

        monkeypatch.setattr(measure, "sample", counting)
        matrix = rng.normal(size=(4, 4))
        linear = LinearOperator.from_matrix(X, X, matrix, form="components")
        check_affine(
            AffineOperator(linear, X.random(rng=rng)),
            rng=rng,
            measure=measure,
            trials=2,
        )
        # Three draws per affine trial, and check_operator's own on top.
        assert drawn["n"] > 6


class TestRankOneAndTheAdjointMemo:
    def test_the_tensor_product_is_the_rank_one_operator(self, rng):
        """``x -> (v, x) u`` in the spaces' own inner products, with the
        adjoint ``y -> (u, y) v``, and positive semidefinite when ``u`` is
        ``v`` on one space."""
        X, Y = make_weighted_space(), make_dense_metric_space(3)
        u, v = Y.random(rng=rng), X.random(rng=rng)
        operator = LinearOperator.from_tensor_product(u, v, domain=X, codomain=Y)
        x, y = X.random(rng=rng), Y.random(rng=rng)
        assert Y.norm(
            Y.subtract(operator(x), Y.scale(X.inner_product(v, x), u))
        ) < 1e-12 * Y.norm(u)
        assert X.norm(
            X.subtract(operator.adjoint(y), X.scale(Y.inner_product(u, y), v))
        ) < 1e-12 * X.norm(v)
        same = LinearOperator.from_tensor_product(v, v, domain=X, codomain=X)
        assert Traits.POSITIVE_SEMIDEFINITE & same.traits
        assert not (Traits.POSITIVE_SEMIDEFINITE & operator.traits)

    def test_adjoints_are_linked_reads_the_memo_and_never_fills_it(self, rng):
        X = make_weighted_space()
        A = LinearOperator.from_matrix(
            X, X, rng.normal(size=(X.dim, X.dim)), form="components"
        )
        B = LinearOperator.from_matrix(
            X, X, rng.normal(size=(X.dim, X.dim)), form="components"
        )
        assert not LinearOperator.adjoints_are_linked(A, B)
        assert not LinearOperator.adjoints_are_linked(A, A)
        adjoint = A.adjoint
        assert LinearOperator.adjoints_are_linked(A, adjoint)
        assert LinearOperator.adjoints_are_linked(adjoint, A)
        assert not LinearOperator.adjoints_are_linked(A, B)


class TestCoordinateOperators:
    def test_they_invert_one_another(self, rng):
        X = make_weighted_space()
        projection, inclusion = X.coordinate_projection(), X.coordinate_inclusion()
        check_operator(projection, rng=rng)
        check_operator(inclusion, rng=rng)
        x = X.random(rng=rng)
        assert np.allclose(inclusion(projection(x)), x)

    def test_the_adjoint_of_inclusion_is_not_the_projection(self, rng):
        """The adjoint of an inclusion, in its smallest possible setting.

        The adjoint gives *derivative* components, ``G c_x``; the projection
        gives the components themselves. They coincide only when the metric is
        the identity.
        """
        X = make_weighted_space()
        x = X.random(rng=rng)
        assert not np.allclose(
            X.coordinate_inclusion().adjoint(x), X.coordinate_projection()(x)
        )
        assert np.allclose(
            X.coordinate_inclusion().adjoint(x),
            X.apply_gram(X.to_components(x)),
        )


class TestDiagonals:
    @pytest.fixture
    def banded(self, rng):
        X = make_weighted_space()
        full = rng.normal(size=(X.dim, X.dim))
        matrix = np.triu(np.tril(full, 1), -1)
        return X, matrix, LinearOperator.from_matrix(X, X, matrix, form="components")

    def test_the_exact_route_reads_the_matrix(self, banded):
        X, matrix, operator = banded
        got = operator.diagonals(offsets=[-1, 0, 1], form="components")
        for index, offset in enumerate([-1, 0, 1]):
            diagonal = np.diag(matrix, offset)
            if offset >= 0:
                assert np.allclose(
                    got[index, offset : offset + diagonal.size], diagonal
                )
            else:
                assert np.allclose(got[index, : diagonal.size], diagonal)

    def test_the_banded_probe_agrees_on_a_banded_operator(self, banded):
        X, matrix, operator = banded
        # An operator holding its matrix reads the band and ignores the
        # probe; the probe is exercised on one that must be applied.
        probed = LinearOperator.from_callables(X, X, operator, adjoint=operator.adjoint)
        assert np.allclose(
            operator.diagonals(offsets=[-1, 0, 1], form="components"),
            probed.diagonals(offsets=[-1, 0, 1], form="components", probe="banded"),
        )

    def test_the_banded_probe_is_wrong_on_a_full_one(self, rng):
        """The negative control. It sums in the out-of-band entries, and says so.

        Built from callables rather than from a matrix: an operator that holds
        its matrix reads the diagonals off it and is exact whatever ``probe``
        asks for, so there is no approximation left to catch.
        """
        X = make_weighted_space()
        matrix = rng.normal(size=(X.dim, X.dim))
        operator = LinearOperator.from_callables(
            X,
            X,
            lambda x: X.from_components(matrix @ X.to_components(x)),
            adjoint=lambda y: X.from_components(
                X.solve_gram(matrix.T @ X.apply_gram(X.to_components(y)))
            ),
        )
        assert not np.allclose(
            operator.diagonals(offsets=[-1, 0, 1], form="components"),
            operator.diagonals(offsets=[-1, 0, 1], form="components", probe="banded"),
        )

    def test_no_offsets_is_refused(self, banded):
        _, _, operator = banded
        with pytest.raises(ValueError, match="At least one offset"):
            operator.diagonals(offsets=[])


class TestDerivativeOperator:
    def test_it_differentiates(self):
        from pygeoinf2.symmetric_space import Lebesgue as BoxLebesgue

        X = BoxLebesgue((256,), lengths=(2.0 * np.pi,))
        axis = X.grid_axes[0]
        assert np.allclose(
            X.derivative_operator()(X.project_function(np.sin)),
            np.cos(axis),
            atol=1e-9,
        )

    def test_it_is_anti_self_adjoint_in_l2(self, rng):
        """``int f' g == -int f g'``, which is what its adjoint must reflect."""
        from pygeoinf2.symmetric_space import Lebesgue as BoxLebesgue

        X = BoxLebesgue((64,), lengths=(1.0,))
        derivative = X.derivative_operator()
        check_operator(derivative, rng=rng)
        u, v = X.random(rng=rng), X.random(rng=rng)
        assert X.inner_product(derivative(u), v) == pytest.approx(
            -X.inner_product(u, derivative(v))
        )

    def test_it_picks_the_right_axis(self):
        from pygeoinf2.symmetric_space import Lebesgue as BoxLebesgue

        X = BoxLebesgue((32, 32), lengths=(2.0 * np.pi, 2.0 * np.pi))
        first, second = np.meshgrid(*X.grid_axes, indexing="ij")
        field = X.project_function(lambda p: np.sin(p[0]) * np.cos(3.0 * p[1]))
        assert np.allclose(
            X.derivative_operator(axis=1)(field),
            -3.0 * np.sin(first) * np.sin(3.0 * second),
            atol=1e-9,
        )

    def test_the_sobolev_version_has_a_working_adjoint(self, rng):
        from pygeoinf2.symmetric_space import Sobolev as BoxSobolev

        X = BoxSobolev((64,), 2.0, 0.1, lengths=(1.0,))
        check_operator(X.derivative_operator(), rng=rng)

    def test_a_missing_axis_is_refused(self):
        from pygeoinf2.symmetric_space import Lebesgue as BoxLebesgue

        with pytest.raises(ValueError, match="axes"):
            BoxLebesgue((16,), lengths=(1.0,)).derivative_operator(axis=2)


class TestSpectralBlockOperator:
    """a correlated measure's operators act on all their
    fields at once -- one analysis and one synthesis per field -- and its
    draws are synthesized straight from white-noise components."""

    @staticmethod
    def space():
        pytest.importorskip("pyshtools")
        from pygeoinf2.symmetric_space.sphere import Sobolev

        return Sobolev(10, 2.0, 0.2)

    @staticmethod
    def counting(X):
        counts = {"analyses": 0, "syntheses": 0}
        to_components, from_components = X.to_components, X.from_components

        def counting_to(x):
            counts["analyses"] += 1
            return to_components(x)

        def counting_from(c):
            counts["syntheses"] += 1
            return from_components(c)

        X.to_components, X.from_components = counting_to, counting_from
        return counts

    def test_it_agrees_with_the_grid_of_diagonal_blocks(self, rng):
        from pygeoinf2.algebra.diagonal import DiagonalLinearOperator
        from pygeoinf2.algebra.direct_sum import BlockLinearOperator, DirectSum
        from pygeoinf2.symmetric_space.base import SpectralBlockLinearOperator
        from pygeoinf2.testing import check_operator

        X = self.space()
        slices = rng.normal(size=(X.dim, 3, 3))
        space = DirectSum([X, X, X])
        fast = SpectralBlockLinearOperator(space, slices)
        grid = BlockLinearOperator(
            [
                [DiagonalLinearOperator(X, slices[:, i, j].copy()) for j in range(3)]
                for i in range(3)
            ]
        )
        x = space.random(rng=rng)
        for a, b in zip(fast(x), grid(x)):
            assert np.allclose(X.to_components(a), X.to_components(b))
        for a, b in zip(fast.adjoint(x), grid.adjoint(x)):
            assert np.allclose(X.to_components(a), X.to_components(b))
        assert fast.matrix(form="galerkin") == pytest.approx(
            grid.matrix(form="galerkin")
        )
        assert fast.diagonals(offsets=(0,), form="galerkin") == pytest.approx(
            grid.diagonals(offsets=(0,), form="galerkin")
        )
        action = fast._components_action()
        assert np.allclose(action(space.to_components(x)), space.to_components(fast(x)))
        check_operator(fast, rng=rng)

    def test_an_application_analyzes_each_field_once(self, rng):
        X = self.space()
        measure = X.correlated_measure_from_correlations(
            [X.heat_symbol(0.14), X.sobolev_symbol(-2.0, 0.2)],
            np.array([[1.0, 0.6], [0.6, 1.0]]),
        )
        x = measure.domain.random(rng=rng)
        counts = self.counting(X)
        try:
            measure.covariance(x)
            assert counts == {"analyses": 2, "syntheses": 2}
            counts["analyses"] = counts["syntheses"] = 0
            measure.sample(rng=rng)
            assert counts == {"analyses": 0, "syntheses": 2}
        finally:
            del X.to_components, X.from_components

    def test_the_precision_is_carried_and_inverts_the_covariance(self, rng):
        from pygeoinf2.traits import Traits

        X = self.space()
        measure = X.correlated_measure_from_correlations(
            [X.heat_symbol(0.14), X.sobolev_symbol(-2.0, 0.2)],
            np.array([[1.0, 0.6], [0.6, 1.0]]),
        )
        assert measure.precision is not None
        assert Traits.POSITIVE_DEFINITE & measure.precision.traits
        x = measure.domain.random(rng=rng)
        back = measure.precision(measure.covariance(x))
        for a, b in zip(back, x):
            assert np.allclose(X.to_components(a), X.to_components(b))

    def test_marginals_and_cross_covariances_are_still_read_off_blocks(self, rng):
        X = self.space()
        first = X.heat_symbol(0.14)
        measure = X.correlated_measure_from_correlations(
            [first, X.sobolev_symbol(-2.0, 0.2)], np.array([[1.0, 0.3], [0.3, 1.0]])
        )
        assert np.allclose(measure.marginal(0).covariance.eigenvalues, first)
        assert np.allclose(X.spectral_correlations(measure)[X.degrees > 0], 0.3)

    def test_the_draws_have_the_requested_correlation(self, rng):
        X = self.space()
        measure = X.correlated_measure_from_correlations(
            [X.heat_symbol(0.14), X.heat_symbol(0.14)],
            np.array([[1.0, -0.8], [-0.8, 1.0]]),
        )
        draws = measure.samples(3000, rng=rng)
        u = np.array([X.to_components(draw[0]) for draw in draws])
        v = np.array([X.to_components(draw[1]) for draw in draws])
        assert np.corrcoef(u[:, 5], v[:, 5])[0, 1] == pytest.approx(-0.8, abs=0.06)


class TestNaNIsNotADefectOfZero:
    def test_a_nan_valued_operator_is_refused(self, rng):
        X = EuclideanSpace(3)
        broken = LinearOperator.from_callables(
            X,
            X,
            lambda x: np.full(3, np.nan),
            adjoint=lambda y: np.full(3, np.nan),
        )
        with pytest.raises(AssertionError, match="not a finite number"):
            check_operator(broken, rng=rng)


class TestATraitCheckDoesNotExtractTheMatrix:
    def test_a_definite_claim_costs_a_few_applications(self, rng):
        """Positive definite implies invertible, and the invertibility check
        used to extract the matrix to compute a rank: as many applications as
        the dimension, paid by every covariance that claimed its traits."""
        X = EuclideanSpace(300)
        count = [0]

        def apply(x):
            count[0] += 1
            return 2.0 * x

        operator = LinearOperator.self_adjoint(
            X, apply, traits=Traits.POSITIVE_DEFINITE
        )
        check_traits(operator, rng=rng)
        assert count[0] < 50


class TestTheInvertibilityClaimIsCheckedOnlyWhereCheap:
    """Every covariance claims invertibility through positive definiteness,
    so ``check_traits`` must not form and decompose a dense matrix for it."""

    @pytest.fixture
    def no_dense_rank(self, monkeypatch):
        def refuse(*args, **kwargs):
            raise AssertionError("a dense rank decomposition was computed")

        monkeypatch.setattr(np.linalg, "matrix_rank", refuse)

    def test_a_diagonal_claim_is_read_off_its_scalings(self, rng, no_dense_rank):
        from pygeoinf2.algebra.diagonal import DiagonalLinearOperator

        X = make_weighted_space()
        D = DiagonalLinearOperator(X, rng.uniform(1.0, 2.0, X.dim))
        assert Traits.INVERTIBLE & D.traits
        check_traits(D, rng=rng)
        check_traits(D @ (2.0 * D), rng=rng)

    def test_a_zero_scaling_is_caught(self, rng, no_dense_rank):
        from pygeoinf2.algebra.diagonal import DiagonalLinearOperator

        X = EuclideanSpace(3)
        singular = DiagonalLinearOperator(X, np.array([0.0, 1.0, 2.0]))
        with pytest.raises(AssertionError, match="scalings are zero"):
            check_traits(singular.with_traits(Traits.INVERTIBLE), rng=rng)

    def test_a_sparse_array_is_not_densified(self, rng, no_dense_rank):
        import scipy.sparse

        n = 50
        X = EuclideanSpace(n)
        matrix = scipy.sparse.diags(
            [np.full(n - 1, -1.0), np.full(n, 4.0), np.full(n - 1, -1.0)],
            [-1, 0, 1],
            format="csr",
        )
        A = LinearOperator.from_matrix(
            X, X, matrix, form="components", traits=Traits.INVERTIBLE
        )
        check_traits(A, rng=rng)

    def test_a_dense_stored_array_is_still_ranked(self, rng):
        X = EuclideanSpace(3)
        singular = LinearOperator.from_matrix(
            X, X, np.diag([1.0, 1.0, 0.0]), form="components", traits=Traits.INVERTIBLE
        )
        with pytest.raises(AssertionError, match="full rank"):
            check_traits(singular, rng=rng)


class TestDiagonalsThroughAScaledComposition:
    def test_a_diagonal_factor_scales_rows_or_columns(self, rng):
        from pygeoinf2.algebra.diagonal import DiagonalLinearOperator

        X, Y = EuclideanSpace(5), EuclideanSpace(3)
        A = LinearOperator.from_matrix(X, Y, rng.normal(size=(3, 5)), form="components")
        D = DiagonalLinearOperator(X, rng.uniform(1.0, 2.0, 5))
        for operator in (A @ D @ A.adjoint, A.adjoint @ A @ D, D @ A.adjoint @ A):
            expected = np.diag(operator.matrix(form="components"))
            assert np.allclose(np.ravel(operator.diagonals()), expected)


class TestTheComparisonScaleIsRelative:
    def test_a_wrong_adjoint_on_a_tiny_operator_is_caught(self, rng):
        """The pairings are of order 1e-10, so an absolute floor at one would
        read their disagreement as a defect of 1e-10 and pass it."""
        X = EuclideanSpace(4)
        wrong = LinearOperator.from_callables(
            X, X, lambda x: 1e-10 * x, adjoint=lambda y: -1e-10 * y
        )
        with pytest.raises(AssertionError, match="adjoint"):
            check_operator(wrong, rng=rng)

    def test_a_tiny_operator_with_the_right_adjoint_passes(self, rng):
        X = EuclideanSpace(4)
        tiny = LinearOperator.from_callables(
            X, X, lambda x: 1e-10 * x, adjoint=lambda y: 1e-10 * y
        )
        report = check_operator(tiny, rng=rng)
        assert report.worst < 1e-12

    def test_the_absolute_floor_is_atol(self, rng):
        X = EuclideanSpace(4)
        wrong = LinearOperator.from_callables(
            X, X, lambda x: 1e-10 * x, adjoint=lambda y: -1e-10 * y
        )
        check_operator(wrong, rng=rng, atol=1e-8)


class TestTheShortcutsAreChecked:
    """A product or a sum applies its factors through their components
    actions and block applications whenever it can, so ``check_operator``
    holds each to the operator's own action."""

    def test_a_components_action_that_disagrees_is_caught(self, rng):
        X = make_dense_metric_space(4)
        G = X.gram_matrix()
        wrong = LinearOperator.from_callables(
            X,
            X,
            lambda x: 2.0 * x,
            adjoint=lambda y: 2.0 * y,
            components_action=lambda c: 2.0 * G @ c,
            components_adjoint_action=lambda c: 2.0 * c,
        )
        with pytest.raises(AssertionError, match="components action"):
            check_operator(wrong, rng=rng)

    def test_an_adjoint_components_action_that_disagrees_is_caught(self, rng):
        X = make_dense_metric_space(4)
        G = X.gram_matrix()
        wrong = LinearOperator.from_callables(
            X,
            X,
            lambda x: 2.0 * x,
            adjoint=lambda y: 2.0 * y,
            components_action=lambda c: 2.0 * c,
            components_adjoint_action=lambda c: 2.0 * G @ c,
        )
        with pytest.raises(AssertionError, match="adjoint's components action"):
            check_operator(wrong, rng=rng)

    def test_agreeing_shortcuts_pass_and_are_reported(self, rng):
        from pygeoinf2.algebra.diagonal import DiagonalLinearOperator

        X = make_dense_metric_space(4)
        right = LinearOperator.from_callables(
            X,
            X,
            lambda x: 2.0 * x,
            adjoint=lambda y: 2.0 * y,
            components_action=lambda c: 2.0 * c,
            components_adjoint_action=lambda c: 2.0 * c,
        )
        report = check_operator(right, rng=rng)
        assert "the components action agrees with the action" in report.defects
        W = make_weighted_space()
        D = DiagonalLinearOperator(W, rng.uniform(1.0, 2.0, W.dim))
        report = check_operator(D, rng=rng)
        assert "the block application agrees with single applications" in (
            report.defects
        )

    def test_a_block_application_that_disagrees_is_caught(self, rng):
        X = make_weighted_space()

        class Wrong(LinearOperator):
            def _value(self, x):
                return X.copy(x)

            def _adjoint_value(self, y):
                return X.copy(y)

            def apply_block(self, vectors, /, *, n_jobs=None):
                return [X.scale(2.0, x) for x in vectors]

        with pytest.raises(AssertionError, match="block application"):
            check_operator(Wrong(X, X), rng=rng)


class TestAMatrixWithoutAnAdjoint:
    def test_the_default_fill_falls_back_to_columns(self):
        """A forward operator into a smaller data space, built without an
        adjoint: the rows cannot be filled, the columns can, and the
        defaults should not ask for an adjoint the matrix does not need."""
        X, Y = EuclideanSpace(5), EuclideanSpace(3)
        A = LinearOperator.from_callables(X, Y, lambda x: x[:3].copy())
        expected = np.eye(3, 5)
        assert np.array_equal(A.matrix(), expected)
        assert np.array_equal(A.matrix(by="columns"), expected)
        with pytest.raises(NotImplementedError):
            A.matrix(by="rows")


class TestADiagonalOnADenseMetric:
    """Invertibility and idempotence are properties of the map, and a
    constant spectrum is a multiple of the identity, on any metric; a
    diagonal on a finite element space keeps them, and a congruence of a
    definite operator by it is definite."""

    def test_metric_independent_traits_are_deduced(self, rng):
        from pygeoinf2.algebra.diagonal import DiagonalLinearOperator

        X = make_dense_metric_space(5)
        D = DiagonalLinearOperator(X, rng.uniform(1.0, 2.0, 5))
        assert D.traits & Traits.INVERTIBLE
        assert not D.traits & Traits.SELF_ADJOINT
        mask = DiagonalLinearOperator(X, np.array([0.0, 1.0, 1.0, 0.0, 1.0]))
        assert mask.traits & Traits.IDEMPOTENT
        constant = DiagonalLinearOperator(X, np.full(5, 2.0))
        assert constant.traits & Traits.SELF_ADJOINT
        assert constant.traits & Traits.POSITIVE_DEFINITE
        for operator in (D, mask, constant):
            check_traits(operator, rng=rng)

    def test_a_congruence_is_definite_and_solvable(self, rng):
        from pygeoinf2.algebra.diagonal import DiagonalLinearOperator
        from pygeoinf2.numerics.solvers import CholeskySolver

        from .conftest import spd_operator

        X = make_dense_metric_space(5)
        D = DiagonalLinearOperator(X, rng.uniform(1.0, 2.0, 5))
        congruence = D.adjoint @ spd_operator(X, rng) @ D
        assert congruence.traits & Traits.POSITIVE_DEFINITE
        check_traits(congruence, rng=rng)
        inverse = CholeskySolver()(congruence)
        x = X.random(rng=rng)
        assert np.allclose(inverse(congruence(x)), x)


class TestDiagonalsRefuseAnUnknownForm:
    def test_an_unknown_form_is_refused(self, rng):
        X = EuclideanSpace(4)
        with pytest.raises(ValueError, match="form"):
            (2.0 * LinearOperator.identity(X)).diagonals(form="nonsense")


class TestEveryCheckTakesBothTolerances:
    def test_the_vector_comparing_checks_take_atol(self):
        import inspect

        from pygeoinf2 import testing

        for name in (
            "check_space",
            "check_coordinates",
            "check_operator",
            "check_traits",
            "check_projection",
            "check_affine",
            "check_hessian",
        ):
            parameters = inspect.signature(getattr(testing, name)).parameters
            assert "rtol" in parameters and "atol" in parameters, name

    def test_check_traits_honors_atol(self, rng):
        """A projector off by a part in 1e8 of its output: refused at the
        default tolerances, accepted with an absolute allowance that covers
        it."""
        X = EuclideanSpace(4)
        nearly = LinearOperator.from_matrix(
            X,
            X,
            np.diag([1.0, 1.0 + 1e-6, 0.0, 0.0]),
            form="components",
            traits=Traits.IDEMPOTENT,
        )
        with pytest.raises(AssertionError, match="IDEMPOTENT"):
            check_traits(nearly, rng=rng)
        check_traits(nearly, rng=rng, atol=1e-4)


class TestANegativePowerOfASingularDiagonal:
    def test_it_is_refused(self):
        from pygeoinf2.algebra.diagonal import DiagonalLinearOperator

        D = DiagonalLinearOperator(EuclideanSpace(3), np.array([0.0, 1.0, 2.0]))
        with pytest.raises(ValueError, match="non-zero"):
            D**-1
        assert np.allclose((D**2).eigenvalues, [0.0, 1.0, 4.0])
        assert np.allclose(
            (
                DiagonalLinearOperator(EuclideanSpace(2), np.array([2.0, 4.0])) ** -1
            ).eigenvalues,
            [0.5, 0.25],
        )
