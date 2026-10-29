"""Direct sums, block operators, and the joint model they exist for."""

import numpy as np
import pytest

from pygeoinf2.algebra.direct_sum import (
    BlockDiagonalLinearOperator,
    BlockDiagonalOperator,
    BlockLinearOperator,
    BlockOperator,
    ColumnLinearOperator,
    ColumnOperator,
    DirectSum,
    RowLinearOperator,
    RowOperator,
)
from pygeoinf2.algebra.operators import LinearOperator, Operator
from pygeoinf2.algebra.spaces import CoordinateSpace, EuclideanSpace
from pygeoinf2.probability import (
    GaussianMeasure,
    ProductMeasure,
    PushForwardMeasure,
    product,
)
from pygeoinf2.testing import (
    check_coordinates,
    check_derivative,
    check_measure,
    check_operator,
    check_space,
    check_traits,
)
from pygeoinf2.traits import Traits

from .conftest import make_dense_metric_space, make_weighted_space
from .doubles import OpaqueSpace


def build_sum(labels=None, first=None):
    return DirectSum(
        [make_weighted_space() if first is None else first, EuclideanSpace(3)],
        labels=labels,
    )


@pytest.fixture(
    params=[lambda dim: make_weighted_space(), make_dense_metric_space],
    ids=["weighted", "dense-metric"],
)
def summand(request):
    """The first summand's metric.

    Over both: a direct sum's Gram is block diagonal over the summands, and a
    summand whose own Gram is diagonal cannot tell whether the block structure
    is being handled or merely the diagonal. Four dimensions either way, so the
    tests that count them stay honest.
    """
    return lambda: request.param(4)


@pytest.fixture
def S(summand):
    return build_sum(("model", "data"), first=summand())


@pytest.fixture(params=["weighted", "dense"])
def model_space(request):
    """A diagonal metric and a dense one on the first summand: a block
    operator's adjoint carries each summand's metric, and only a dense one
    tells that from the raw components."""
    return (
        make_weighted_space()
        if request.param == "weighted"
        else make_dense_metric_space(4)
    )


class TestDirectSumSpace:
    def test_axioms(self, S, summand, rng):
        # The rebuild has to use the same summand, or the check is comparing
        # two different spaces and reporting the difference as a key failure.
        check_space(
            S, rng=rng, rebuild=lambda: build_sum(("model", "data"), first=summand())
        )

    def test_coordinates(self, S, rng):
        check_coordinates(S, rng=rng)

    def test_dimension_and_summands(self, S):
        assert S.dim == 4 + 3
        assert len(S) == 2

    def test_vectors_are_tuples(self, S, rng):
        x = S.random(rng=rng)
        assert isinstance(x, tuple)
        assert len(x) == 2

    def test_inner_product_is_the_sum_of_the_summands(self, S, rng):
        x, y = S.random(rng=rng), S.random(rng=rng)
        expected = sum(
            space.inner_product(xi, yi) for space, xi, yi in zip(S.subspaces, x, y)
        )
        assert S.inner_product(x, y) == pytest.approx(expected)

    def test_named_access(self, S, rng):
        x = S.random(rng=rng)
        assert S.component(x, "model") is x[0]
        assert S.component(x, "data") is x[1]
        assert S.subspace("data") == EuclideanSpace(3)

    def test_unknown_label(self, S, rng):
        with pytest.raises(KeyError, match="junk"):
            S.component(S.random(rng=rng), "junk")

    def test_labels_are_not_part_of_identity(self):
        """A block operator cannot know what its user called things.

        Two sums over the same summands are the same space, whatever their
        components are named; making labels structural would put a block
        operator on a space unequal to the one its vectors came from.
        """
        assert build_sum(("model", "data")) == build_sum()
        assert hash(build_sum(("a", "b"))) == hash(build_sum())

    def test_duplicate_labels_are_refused(self):
        with pytest.raises(ValueError, match="distinct"):
            DirectSum([EuclideanSpace(2), EuclideanSpace(3)], labels=("x", "x"))

    def test_wrong_number_of_labels(self):
        with pytest.raises(ValueError, match="labels for"):
            DirectSum([EuclideanSpace(2), EuclideanSpace(3)], labels=("only_one",))

    def test_empty_sums_are_refused(self):
        with pytest.raises(ValueError, match="at least one"):
            DirectSum([])


class TestASumPickles:
    def test_a_round_trip_keeps_the_summands_and_the_dispatch(self):
        import pickle

        space = DirectSum([EuclideanSpace(2), make_weighted_space()], labels=("a", "b"))
        copy = pickle.loads(pickle.dumps(space))
        assert isinstance(copy, CoordinateSpace)
        assert copy.dim == space.dim and copy.labels == ("a", "b")
        assert [s.dim for s in copy.subspaces] == [s.dim for s in space.subspaces]


class TestCoordinateDispatch:
    def test_all_coordinate_summands_give_a_coordinate_sum(self):
        assert isinstance(build_sum(), CoordinateSpace)

    def test_one_coordinate_free_summand_makes_the_sum_coordinate_free(self):
        """The honest answer, and the one require_coordinates depends on."""
        mixed = DirectSum([EuclideanSpace(2), OpaqueSpace(np.array([1.0, 2.0]))])
        assert not isinstance(mixed, CoordinateSpace)

    def test_a_coordinate_free_sum_still_satisfies_the_axioms(self, rng):
        mixed = DirectSum([EuclideanSpace(2), OpaqueSpace(np.array([1.0, 2.0]))])
        check_space(mixed, rng=rng)

    def test_gram_is_block_diagonal(self, S):
        gram = S.gram_matrix()
        assert np.allclose(gram[:4, 4:], 0.0)
        assert np.allclose(gram[4:, :4], 0.0)

    def test_orthonormality_needs_every_summand(self):
        assert DirectSum([EuclideanSpace(2), EuclideanSpace(3)]).is_orthonormal
        assert not build_sum().is_orthonormal


class TestProjections:
    def test_projection_and_inclusion_are_adjoint(self, S, rng):
        for key in ("model", "data"):
            check_operator(S.projection(key), rng=rng)
            assert S.projection(key).adjoint is S.inclusion(key)

    def test_projections_are_memoized(self, S):
        """So that P @ C @ P.adjoint is recognizable as a congruence."""
        assert S.projection("model") is S.projection("model")
        assert S.projection(0) is S.projection("model")

    def test_a_congruence_through_a_projection_is_recognized(self, S, rng):
        C = LinearOperator.self_adjoint(
            S, lambda x: S.scale(2.0, x), traits=Traits.POSITIVE_DEFINITE
        )
        P = S.projection("data")
        assert Traits.POSITIVE_SEMIDEFINITE & (P @ C @ P.adjoint).traits

    def test_inclusion_then_projection_is_the_identity(self, S, rng):
        x = S.subspace("data").random(rng=rng)
        assert np.allclose(S.projection("data")(S.inclusion("data")(x)), x)


class TestBlockOperators:
    @pytest.fixture
    def pieces(self, rng, model_space):
        X, Y = model_space, EuclideanSpace(3)
        A = LinearOperator.from_matrix(
            X, Y, rng.normal(size=(3, X.dim)), form="components"
        )
        return X, Y, A

    def test_linear_blocks_dispatch_to_the_linear_class(self, pieces):
        X, Y, A = pieces
        op = BlockOperator(
            [
                [LinearOperator.identity(X), LinearOperator.zero(Y, codomain=X)],
                [A, LinearOperator.identity(Y)],
            ]
        )
        assert isinstance(op, BlockLinearOperator)

    def test_adjoint_is_the_transposed_grid(self, pieces, rng):
        X, Y, A = pieces
        op = BlockOperator(
            [
                [LinearOperator.identity(X), LinearOperator.zero(Y, codomain=X)],
                [A, LinearOperator.identity(Y)],
            ]
        )
        check_operator(op, rng=rng)
        assert op.adjoint.block(0, 1) is A.adjoint
        assert op.adjoint.adjoint is op

    def test_a_symmetric_grid_is_recognized_as_self_adjoint(self, rng):
        X, Y = EuclideanSpace(3), EuclideanSpace(2)
        A = LinearOperator.from_matrix(X, Y, rng.normal(size=(2, 3)), form="components")
        C = LinearOperator.self_adjoint(X, lambda x: 2.0 * x)
        D = LinearOperator.self_adjoint(Y, lambda y: 3.0 * y)
        op = BlockOperator([[C, A.adjoint], [A, D]])
        assert Traits.SELF_ADJOINT & op.traits
        check_traits(op, rng=rng)

    def test_an_asymmetric_grid_claims_nothing(self, pieces):
        X, Y, A = pieces
        op = BlockOperator(
            [
                [LinearOperator.identity(X), LinearOperator.zero(Y, codomain=X)],
                [A, LinearOperator.identity(Y)],
            ]
        )
        assert not (Traits.SELF_ADJOINT & op.traits)

    def test_mismatched_blocks_are_refused(self, pieces):
        X, Y, A = pieces
        with pytest.raises(ValueError, match="domain"):
            BlockOperator([[LinearOperator.identity(X), A]])

    def test_ragged_grids_are_refused(self, pieces):
        X, Y, A = pieces
        with pytest.raises(ValueError, match="blocks, but row 0"):
            BlockOperator(
                [[LinearOperator.identity(X), LinearOperator.zero(Y, codomain=X)], [A]]
            )


class TestColumnRowAndDiagonal:
    @pytest.fixture
    def pieces(self, rng, model_space):
        X, Y, Z = model_space, EuclideanSpace(3), EuclideanSpace(2)
        A = LinearOperator.from_matrix(
            X, Y, rng.normal(size=(3, X.dim)), form="components"
        )
        B = LinearOperator.from_matrix(
            X, Z, rng.normal(size=(2, X.dim)), form="components"
        )
        return X, Y, Z, A, B

    def test_column_maps_into_a_sum(self, pieces, rng):
        X, Y, Z, A, B = pieces
        op = ColumnOperator([A, B])
        assert isinstance(op, ColumnLinearOperator)
        assert op.domain == X
        check_operator(op, rng=rng)
        x = X.random(rng=rng)
        assert np.allclose(op(x)[0], A(x)) and np.allclose(op(x)[1], B(x))

    def test_column_adjoint_is_a_row(self, pieces, rng):
        _, _, _, A, B = pieces
        op = ColumnOperator([A, B])
        assert isinstance(op.adjoint, RowLinearOperator)
        assert op.adjoint.adjoint is op

    def test_row_maps_out_of_a_sum(self, pieces, rng):
        X, Y, Z, A, B = pieces
        op = RowOperator([A.adjoint, B.adjoint])
        assert isinstance(op, RowLinearOperator)
        assert op.codomain == X
        check_operator(op, rng=rng)

    def test_block_diagonal_traits_intersect(self, rng):
        X, Y = EuclideanSpace(3), EuclideanSpace(2)
        C = LinearOperator.self_adjoint(
            X, lambda x: 2.0 * x, traits=Traits.POSITIVE_DEFINITE
        )
        D = LinearOperator.self_adjoint(
            Y, lambda y: 3.0 * y, traits=Traits.POSITIVE_DEFINITE
        )
        op = BlockDiagonalOperator([C, D])
        assert isinstance(op, BlockDiagonalLinearOperator)
        assert Traits.POSITIVE_DEFINITE & op.traits
        check_traits(op, rng=rng)
        check_operator(op, rng=rng)

    def test_one_unstructured_block_loses_the_claim(self, rng):
        X, Y = EuclideanSpace(3), EuclideanSpace(2)
        C = LinearOperator.self_adjoint(
            X, lambda x: 2.0 * x, traits=Traits.POSITIVE_DEFINITE
        )
        D = LinearOperator.from_matrix(Y, Y, rng.normal(size=(2, 2)), form="components")
        assert not (Traits.SELF_ADJOINT & BlockDiagonalOperator([C, D]).traits)


class TestNonlinearBlocks:
    """The reason block operators are nonlinear by default."""

    @pytest.fixture
    def pieces(self, rng, model_space):
        X, Y = model_space, EuclideanSpace(3)

        def value(m):
            c = X.to_components(m)
            return np.array([float(c @ c), c[0], c[1]])

        def derivative(m):
            c = X.to_components(m)
            rows = np.vstack([2.0 * c, np.eye(X.dim)[0], np.eye(X.dim)[1]])
            return LinearOperator.from_matrix(X, Y, rows, form="components")

        F = Operator.from_callables(X, Y, value, derivative=derivative)
        return X, Y, F

    def test_a_nonlinear_block_stays_nonlinear(self, pieces):
        X, Y, F = pieces
        op = BlockOperator(
            [
                [LinearOperator.identity(X), LinearOperator.zero(Y, codomain=X)],
                [F, LinearOperator.identity(Y)],
            ]
        )
        assert type(op) is BlockOperator
        assert not isinstance(op, LinearOperator)

    def test_the_joint_model_evaluates(self, pieces, rng):
        X, Y, F = pieces
        op = BlockOperator(
            [
                [LinearOperator.identity(X), LinearOperator.zero(Y, codomain=X)],
                [F, LinearOperator.identity(Y)],
            ]
        )
        m, e = X.random(rng=rng), Y.random(rng=rng)
        model, data = op((m, e))
        assert np.allclose(model, m)
        assert np.allclose(data, F(m) + e)

    def test_the_derivative_is_the_block_of_derivatives(self, pieces, rng):
        """[[I, 0], [F'(m), I]] -- the linearized joint model, from the same object."""
        X, Y, F = pieces
        op = BlockOperator(
            [
                [LinearOperator.identity(X), LinearOperator.zero(Y, codomain=X)],
                [F, LinearOperator.identity(Y)],
            ]
        )
        point = (X.random(rng=rng), Y.random(rng=rng))
        jacobian = op.derivative(point)
        assert isinstance(jacobian, BlockLinearOperator)
        check_operator(jacobian, rng=rng)
        check_derivative(op, point, rng=rng)

    def test_a_nonlinear_column(self, pieces, rng):
        X, Y, F = pieces
        A = LinearOperator.from_matrix(
            X, Y, rng.normal(size=(3, X.dim)), form="components"
        )
        op = ColumnOperator([F, A])
        assert type(op) is ColumnOperator
        check_derivative(op, X.random(rng=rng), rng=rng)


class TestProductMeasures:
    def test_a_product_of_gaussians_is_gaussian(self, rng):
        X, Y = make_weighted_space(), EuclideanSpace(3)
        joint = product(
            [
                GaussianMeasure.from_standard_deviation(X, 1.5),
                GaussianMeasure.from_standard_deviation(Y, 0.4),
            ],
            labels=("model", "noise"),
        )
        assert isinstance(joint, GaussianMeasure)
        assert Traits.POSITIVE_DEFINITE & joint.covariance.traits
        check_measure(joint, rng=rng, samples=20000, rtol=0.1)

    def test_a_product_of_anything_else_is_still_samplable(self, rng):
        """A product of non-Gaussian factors is sampled factor by factor."""
        X, Y = EuclideanSpace(2), EuclideanSpace(3)
        base = GaussianMeasure.from_standard_deviation(X, 1.0)
        nonlinear = Operator.from_callables(X, X, lambda x: np.abs(x))
        skewed = nonlinear @ base

        joint = product([skewed, GaussianMeasure.from_standard_deviation(Y, 1.0)])
        assert isinstance(joint, ProductMeasure)
        assert not joint.has_covariance  # honestly unavailable
        draws = joint.samples(200, rng=rng)
        assert all(np.all(d[0] >= 0.0) for d in draws)

    def test_factors_are_reachable_by_name(self, rng):
        X, Y = EuclideanSpace(2), EuclideanSpace(3)
        joint = ProductMeasure(
            [
                GaussianMeasure.from_standard_deviation(X, 1.0),
                GaussianMeasure.from_standard_deviation(Y, 1.0),
            ],
            labels=("model", "noise"),
        )
        assert joint.factor("noise").domain == Y


class TestAProductOnAGivenDomain:
    def test_the_measure_lives_on_the_domain_given(self, rng):
        spaces = [EuclideanSpace(2), EuclideanSpace(3)]
        measures = [GaussianMeasure.from_standard_deviation(s, 1.0) for s in spaces]
        domain = DirectSum(spaces, labels=("a", "b"))
        measure = GaussianMeasure.from_product(measures, domain=domain)
        assert measure.domain is domain
        check_measure(measure, rng=rng, samples=2000, rtol=0.3)

    def test_a_domain_with_other_summands_is_refused(self):
        spaces = [EuclideanSpace(2), EuclideanSpace(3)]
        measures = [GaussianMeasure.from_standard_deviation(s, 1.0) for s in spaces]
        with pytest.raises(ValueError, match="Summand 1"):
            GaussianMeasure.from_product(
                measures, domain=DirectSum([EuclideanSpace(2), EuclideanSpace(4)])
            )
        with pytest.raises(ValueError, match="summands"):
            GaussianMeasure.from_product(measures, domain=DirectSum(spaces[:1]))
        with pytest.raises(ValueError, match="labels or a domain"):
            GaussianMeasure.from_product(
                measures, domain=DirectSum(spaces), labels=("a", "b")
            )


class TestJointModel:
    """The construction direct sums exist for: the joint law of model and data."""

    def test_the_linear_joint_law_has_the_block_shape(self, rng):
        X, Y = make_weighted_space(), EuclideanSpace(3)
        A = LinearOperator.from_matrix(
            X, Y, rng.normal(size=(3, X.dim)), form="components"
        )

        prior = GaussianMeasure.from_standard_deviation(X, 1.5)
        noise = GaussianMeasure.from_standard_deviation(Y, 0.3)

        op = BlockOperator(
            [
                [LinearOperator.identity(X), LinearOperator.zero(Y, codomain=X)],
                [A, LinearOperator.identity(Y)],
            ]
        )
        joint = op @ product([prior, noise], labels=("model", "data"))

        assert isinstance(joint, GaussianMeasure)
        check_measure(joint, rng=rng, samples=20000, rtol=0.12)

        # The data block of the joint covariance is A C A* + R.
        data = joint.domain.inclusion(1)
        expected = A @ prior.covariance @ A.adjoint + noise.covariance
        y = Y.random(rng=rng)
        assert np.allclose(
            joint.domain.component(joint.covariance(data(y)), 1), expected(y)
        )

    def test_the_nonlinear_joint_law_is_samplable(self, rng):
        """The same shape, with F in place of A. No closed density, but samples."""
        X, Y = EuclideanSpace(2), EuclideanSpace(2)

        def value(m):
            return np.array([float(m @ m), float(m[0])])

        F = Operator.from_callables(X, Y, value)
        op = BlockOperator(
            [
                [LinearOperator.identity(X), LinearOperator.zero(Y, codomain=X)],
                [F, LinearOperator.identity(Y)],
            ]
        )
        joint = op @ product(
            [
                GaussianMeasure.from_standard_deviation(X, 1.0),
                GaussianMeasure.from_standard_deviation(Y, 0.1),
            ]
        )

        assert isinstance(joint, PushForwardMeasure)
        draws = joint.samples(4000, rng=rng)
        # E[d_0] == E[|m|^2] + 0 == 2 for a standard normal on R^2.
        assert np.mean([d[1][0] for d in draws]) == pytest.approx(2.0, rel=0.1)
        # And the model half is untouched by the map.
        assert np.mean([d[0][0] for d in draws]) == pytest.approx(0.0, abs=0.1)


class TestBlocksDoNotAllocateForNothing:
    """A block row must not start from ``codomain.zero()`` — a synthesis on a
    spectral space — and then ``axpy`` every block including the structurally
    zero ones, whose value is another fresh zero vector.
    ``[[I, 0], [I, I]]`` on two spheres cost three syntheses per application
    where none are needed: 12.2 ms against 0.13 ms at lmax 128.
    """

    @staticmethod
    def pieces(dim=6):
        from pygeoinf2.algebra.nodes import _Identity, _Zero

        space = make_dense_metric_space(dim)
        return space, _Identity(space), _Zero(space, space)

    @staticmethod
    def naive(codomain, images):
        """The plain route, kept as the reference."""
        total = codomain.zero()
        for image in images:
            total = codomain.axpy(1.0, image, total)
        return total

    def test_a_block_operator_agrees_with_the_naive_sum(self, rng):
        space, identity, zero = self.pieces()
        matrix = LinearOperator.from_matrix(
            space, space, rng.standard_normal((6, 6)), form="components"
        )
        operator = BlockLinearOperator([[identity, zero], [matrix, identity]])
        x = (space.random(rng=rng), space.random(rng=rng))

        got = operator(x)
        expected = (
            self.naive(space, [identity(x[0]), zero(x[1])]),
            self.naive(space, [matrix(x[0]), identity(x[1])]),
        )
        for a, b in zip(got, expected):
            assert space.to_components(a) == pytest.approx(space.to_components(b))

    def test_the_adjoint_agrees_and_is_still_the_adjoint(self, rng):
        """On a Gram matrix that is not diagonal, which is where an adjoint
        that skipped the wrong term would show."""
        space, identity, zero = self.pieces()
        matrix = LinearOperator.from_matrix(
            space, space, rng.standard_normal((6, 6)), form="components"
        )
        operator = BlockLinearOperator([[identity, zero], [matrix, identity]])
        check_operator(operator, rng=rng)
        check_operator(ColumnLinearOperator([identity, zero]), rng=rng)
        check_operator(RowLinearOperator([zero, matrix]), rng=rng)

    def test_a_zero_block_is_never_applied(self, rng):
        """Its value is a fresh zero vector, which is the allocation being
        avoided; adding it changes nothing."""
        from pygeoinf2.algebra.nodes import _Zero

        space, identity, zero = self.pieces()
        applications = []
        original = _Zero._value

        def counting(self, x):
            applications.append(1)
            return original(self, x)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(_Zero, "_value", counting)
            x = (space.random(rng=rng), space.random(rng=rng))
            BlockLinearOperator([[identity, zero], [zero, identity]])(x)
            RowLinearOperator([identity, zero])(x)
            ColumnLinearOperator([identity, zero]).adjoint(x)
        assert applications == []

    def test_the_result_does_not_alias_the_argument(self, rng):
        """``_Identity`` returns its argument unchanged, so a row that starts
        from the first image rather than from a zero has to copy it."""
        space, identity, zero = self.pieces()
        x = (space.random(rng=rng), space.random(rng=rng))
        before = space.to_components(x[0]).copy()

        y = BlockLinearOperator([[identity, zero], [zero, identity]])(x)
        assert y[0] is not x[0]
        space.scale_inplace(2.0, y[0])
        assert space.to_components(x[0]) == pytest.approx(before)

    def test_a_row_of_nothing_but_zeros_is_still_zero(self, rng):
        space, _, zero = self.pieces()
        x = (space.random(rng=rng), space.random(rng=rng))
        y = BlockLinearOperator([[zero, zero], [zero, zero]])(x)
        for part in y:
            assert space.norm(part) == pytest.approx(0.0)


class TestCoordinateDirectSumIsPublic:
    """A direct sum of coordinate spaces is a named, subclassable class, and
    a subclass that compares equal to the plain sum hashes equal to it
    without restating the formula."""

    def test_the_sum_of_coordinate_spaces_is_one(self):
        from pygeoinf2.algebra import CoordinateDirectSum

        plain = DirectSum([EuclideanSpace(2), EuclideanSpace(3)])
        assert type(plain) is CoordinateDirectSum
        assert isinstance(plain, CoordinateSpace)

    def test_a_subclass_hashes_as_the_plain_sum(self):
        from pygeoinf2.algebra import CoordinateDirectSum

        class Tagged(CoordinateDirectSum):
            def __eq__(self, other):
                if not isinstance(other, CoordinateDirectSum):
                    return NotImplemented
                return self._key() == other._key()

            __hash__ = CoordinateDirectSum.__hash__

        summands = [EuclideanSpace(2), EuclideanSpace(3)]
        plain = DirectSum(summands)
        tagged = Tagged(summands)
        assert tagged == plain and plain == tagged
        assert hash(tagged) == hash(plain)
        assert len({plain, tagged}) == 1


class TestTheBlockGrid:
    def test_blocks_is_the_row_major_grid(self, rng):
        X, Y = make_weighted_space(), EuclideanSpace(3)
        A = LinearOperator.from_matrix(
            X, Y, rng.normal(size=(3, X.dim)), form="components"
        )
        operator = BlockOperator(
            [[LinearOperator.identity(X), A.adjoint], [A, LinearOperator.identity(Y)]]
        )
        grid = operator.blocks
        assert len(grid) == 2 and all(len(row) == 2 for row in grid)
        for i in range(2):
            for j in range(2):
                assert grid[i][j] is operator.block(i, j)


class TestABlockDiagonalOnACallersSum:
    def test_it_acts_on_the_sum_given(self):
        X, Y = EuclideanSpace(2), EuclideanSpace(3)
        plain = DirectSum([X, Y])
        labeled = DirectSum([X, Y], labels=("a", "b"))
        blocks = [LinearOperator.identity(X), LinearOperator.identity(Y)]
        seated = BlockDiagonalLinearOperator(blocks, domain=labeled, codomain=labeled)
        assert seated.domain is labeled
        assert seated.adjoint.domain is labeled
        assert BlockDiagonalLinearOperator(blocks).domain == plain

    def test_a_sum_of_other_summands_is_refused(self):
        X, Y = EuclideanSpace(2), EuclideanSpace(3)
        blocks = [LinearOperator.identity(X), LinearOperator.identity(Y)]
        with pytest.raises(ValueError, match="direct sum of the blocks"):
            BlockDiagonalLinearOperator(blocks, domain=DirectSum([Y, X]))


class TestBlockOperatorsPickle:
    """Process-parallel loops send operators to workers by pickling them, so
    every block class must survive a round trip, linear or not."""

    @staticmethod
    def _round_trip(operator):
        import pickle

        return pickle.loads(pickle.dumps(operator))

    def _blocks(self, rng):
        X = make_dense_metric_space(3)
        Y = make_weighted_space()
        A = LinearOperator.from_matrix(
            X, Y, rng.normal(size=(Y.dim, X.dim)), form="components"
        )
        return X, Y, A

    def test_every_linear_block_class_round_trips(self, rng):
        X, Y, A = self._blocks(rng)
        IX, IY = LinearOperator.identity(X), LinearOperator.identity(Y)
        operators = [
            BlockOperator([[IX, A.adjoint], [A, IY]]),
            ColumnOperator([IX, A]),
            RowOperator([A, IY]),
            BlockDiagonalOperator([IX, IY]),
        ]
        expected = [
            BlockLinearOperator,
            ColumnLinearOperator,
            RowLinearOperator,
            BlockDiagonalLinearOperator,
        ]
        for operator, cls in zip(operators, expected):
            copy = self._round_trip(operator)
            assert type(copy) is cls
            x = operator.domain.random(rng=rng)
            y = operator.codomain.random(rng=rng)
            assert np.allclose(
                operator.codomain.to_components(copy(x)),
                operator.codomain.to_components(operator(x)),
            )
            assert np.allclose(
                operator.domain.to_components(copy.adjoint(y)),
                operator.domain.to_components(operator.adjoint(y)),
            )

    def test_every_nonlinear_block_class_round_trips(self, rng):
        X, Y, A = self._blocks(rng)
        F = Operator.from_callables(X, X, np.abs)
        G = Operator.from_callables(Y, Y, np.abs)
        operators = [
            BlockOperator([[F, A.adjoint], [A, G]]),
            ColumnOperator([F, A]),
            RowOperator([A.adjoint, F]),
            BlockDiagonalOperator([F, G]),
        ]
        expected = [BlockOperator, ColumnOperator, RowOperator, BlockDiagonalOperator]
        for operator, cls in zip(operators, expected):
            copy = self._round_trip(operator)
            assert type(copy) is cls
            x = operator.domain.random(rng=rng)
            assert np.allclose(
                operator.codomain.to_components(copy(x)),
                operator.codomain.to_components(operator(x)),
            )

    def test_a_product_measure_samples_in_worker_processes(self):
        E = EuclideanSpace(2)
        factor = GaussianMeasure.from_standard_deviation(E, 1.0)
        measure = GaussianMeasure.from_product([factor, factor])
        draws = measure.samples(4, rng=np.random.default_rng(0), n_jobs=2)
        assert len(draws) == 4
        assert all(len(draw) == 2 for draw in draws)


class TestANonlinearBlockIsLinearizedOnce:
    """``at()`` on a block operator asks each block for its ``at()`` once,
    so a block that is one solve yielding value and derivative is solved
    once, as it is on its own."""

    @staticmethod
    def counted(space, counts):
        def value(x):
            counts["value"] += 1
            return space.from_components(space.to_components(x) ** 2)

        def linearize(x):
            from pygeoinf2.algebra.linearization import Linearization

            counts["linearize"] += 1
            c = space.to_components(x)
            derivative = LinearOperator.from_matrix(
                space, space, np.diag(2.0 * c), form="components"
            )
            return Linearization(x, space.from_components(c**2), derivative)

        return Operator.from_callables(space, space, value, linearize=linearize)

    def test_every_block_arrangement_solves_once(self, rng):
        X = make_weighted_space()
        identity = LinearOperator.identity(X)
        counts = {"value": 0, "linearize": 0}
        F = self.counted(X, counts)
        arrangements = [
            BlockOperator([[identity, LinearOperator.zero(X)], [F, identity]]),
            ColumnOperator([identity, F]),
            RowOperator([F, identity]),
            BlockDiagonalOperator([F, identity]),
        ]
        for operator in arrangements:
            counts.update(value=0, linearize=0)
            x = operator.domain.random(rng=rng)
            model = operator.at(x)
            assert counts == {"value": 0, "linearize": 1}, operator
            expected = operator(x)
            for got, want in zip(
                model.value if isinstance(model.value, tuple) else (model.value,),
                expected if isinstance(expected, tuple) else (expected,),
            ):
                assert np.allclose(got, want)
            check_derivative(operator, x, rng=rng)

    def test_a_linear_block_operator_is_its_own_derivative(self, rng):
        X = make_weighted_space()
        identity = LinearOperator.identity(X)
        operator = BlockDiagonalOperator([identity, 2.0 * identity])
        x = operator.domain.random(rng=rng)
        assert operator.at(x).derivative is operator
        assert operator.derivative(x) is operator


class TestAnEmptyBlockOperatorIsRefusedByName:
    @pytest.mark.parametrize(
        "build", [ColumnOperator, RowOperator, BlockDiagonalOperator, BlockOperator]
    )
    def test_no_blocks(self, build):
        with pytest.raises(ValueError, match="at least one"):
            build([])
