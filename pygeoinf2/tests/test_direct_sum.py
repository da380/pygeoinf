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

from .conftest import make_weighted_space
from .doubles import OpaqueSpace


def build_sum(labels=None):
    return DirectSum([make_weighted_space(), EuclideanSpace(3)], labels=labels)


@pytest.fixture
def S():
    return build_sum(("model", "data"))


class TestDirectSumSpace:
    def test_axioms(self, S, rng):
        check_space(S, rng=rng, rebuild=lambda: build_sum(("model", "data")))

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

    def test_projections_are_memoised(self, S):
        """So that P @ C @ P.adjoint is recognisable as a congruence."""
        assert S.projection("model") is S.projection("model")
        assert S.projection(0) is S.projection("model")

    def test_a_congruence_through_a_projection_is_recognised(self, S, rng):
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
    def pieces(self, rng):
        X, Y = make_weighted_space(), EuclideanSpace(3)
        A = LinearOperator.from_component_matrix(X, Y, rng.normal(size=(3, X.dim)))
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

    def test_a_symmetric_grid_is_recognised_as_self_adjoint(self, rng):
        X, Y = EuclideanSpace(3), EuclideanSpace(2)
        A = LinearOperator.from_component_matrix(X, Y, rng.normal(size=(2, 3)))
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
    def pieces(self, rng):
        X, Y, Z = make_weighted_space(), EuclideanSpace(3), EuclideanSpace(2)
        A = LinearOperator.from_component_matrix(X, Y, rng.normal(size=(3, X.dim)))
        B = LinearOperator.from_component_matrix(X, Z, rng.normal(size=(2, X.dim)))
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
        D = LinearOperator.from_component_matrix(Y, Y, rng.normal(size=(2, 2)))
        assert not (Traits.SELF_ADJOINT & BlockDiagonalOperator([C, D]).traits)


class TestNonlinearBlocks:
    """The reason block operators are nonlinear by default."""

    @pytest.fixture
    def pieces(self, rng):
        X, Y = make_weighted_space(), EuclideanSpace(3)

        def value(m):
            c = X.to_components(m)
            return np.array([float(c @ c), c[0], c[1]])

        def derivative(m):
            c = X.to_components(m)
            rows = np.vstack([2.0 * c, np.eye(X.dim)[0], np.eye(X.dim)[1]])
            return LinearOperator.from_component_matrix(X, Y, rows)

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
        """[[I, 0], [F'(m), I]] -- the linearised joint model, from the same object."""
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
        A = LinearOperator.from_component_matrix(X, Y, rng.normal(size=(3, X.dim)))
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
        """v1 has this only for Gaussians; a non-Gaussian prior needs it."""
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


class TestJointModel:
    """The construction the whole thing is for. See DESIGN.md 3.3."""

    def test_the_linear_joint_law_matches_v1s_shape(self, rng):
        X, Y = make_weighted_space(), EuclideanSpace(3)
        A = LinearOperator.from_component_matrix(X, Y, rng.normal(size=(3, X.dim)))

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
