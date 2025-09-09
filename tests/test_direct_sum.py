"""
Tests for the direct_sum module.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from pygeoinf.hilbert_space import EuclideanSpace, HilbertSpace
from pygeoinf.symmetric_space.circle import Sobolev
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.direct_sum import (
    HilbertSpaceDirectSum,
    BlockLinearOperator,
    BlockDiagonalLinearOperator,
    RowLinearOperator,
    ColumnLinearOperator,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def euclidean_subspaces() -> list[EuclideanSpace]:
    """Provides a list of simple Euclidean spaces to form a direct sum."""
    return [EuclideanSpace(dim=2), EuclideanSpace(dim=3)]


@pytest.fixture(scope="module")
def mixed_subspaces() -> list[HilbertSpace]:
    """Provides a mixed list of Euclidean and Sobolev spaces."""
    return [EuclideanSpace(dim=3), Sobolev(8, 1.0, 0.1)]


@pytest.fixture(scope="module")
def block_operators(euclidean_subspaces) -> list[list[LinearOperator]]:
    """Provides a 2x2 block matrix of linear operators."""
    s1, s2 = euclidean_subspaces
    op11 = LinearOperator.from_matrix(s1, s1, np.random.randn(2, 2))
    op12 = LinearOperator.from_matrix(s2, s1, np.random.randn(2, 3))
    op21 = LinearOperator.from_matrix(s1, s2, np.random.randn(3, 2))
    op22 = LinearOperator.from_matrix(s2, s2, np.random.randn(3, 3))
    return [[op11, op12], [op21, op22]]


# =============================================================================
# Test Suite 1: Hilbert Space Axiom Checks
# =============================================================================


def test_direct_sum_euclidean_axioms(euclidean_subspaces: list[EuclideanSpace]):
    """
    Verifies axioms for a direct sum of Euclidean spaces.
    """
    space = HilbertSpaceDirectSum(euclidean_subspaces)
    space.check(n_checks=5)


def test_direct_sum_mixed_axioms(mixed_subspaces: list[HilbertSpace]):
    """
    Verifies axioms for a direct sum of mixed space types.
    """
    space = HilbertSpaceDirectSum(mixed_subspaces)
    space.check(n_checks=5)


# =============================================================================
# Test Suite 2: Linear Operator Axiom Checks
# =============================================================================


def test_block_linear_operator_axioms(block_operators: list[list[LinearOperator]]):
    """Verifies that the BlockLinearOperator satisfies all axioms."""
    operator = BlockLinearOperator(block_operators)
    # Create a second, compatible operator for algebraic checks
    s1 = operator.domain.subspace(0)
    s2 = operator.domain.subspace(1)
    op11 = LinearOperator.from_matrix(s1, s1, np.random.randn(2, 2))
    op12 = LinearOperator.from_matrix(s2, s1, np.random.randn(2, 3))
    op21 = LinearOperator.from_matrix(s1, s2, np.random.randn(3, 2))
    op22 = LinearOperator.from_matrix(s2, s2, np.random.randn(3, 3))
    operator2 = BlockLinearOperator([[op11, op12], [op21, op22]])
    operator.check(n_checks=3, op2=operator2)


def test_block_diagonal_operator_axioms(euclidean_subspaces: list[EuclideanSpace]):
    """Verifies that the BlockDiagonalLinearOperator satisfies all axioms."""
    s1, s2 = euclidean_subspaces
    op1 = LinearOperator.from_matrix(s1, s1, np.random.randn(2, 2))
    op2 = LinearOperator.from_matrix(s2, s2, np.random.randn(3, 3))
    operator = BlockDiagonalLinearOperator([op1, op2])
    # Create a second, compatible operator for algebraic checks
    op3 = LinearOperator.from_matrix(s1, s1, np.random.randn(2, 2))
    op4 = LinearOperator.from_matrix(s2, s2, np.random.randn(3, 3))
    operator2 = BlockDiagonalLinearOperator([op3, op4])
    operator.check(n_checks=3, op2=operator2)


def test_row_linear_operator_axioms(euclidean_subspaces: list[EuclideanSpace]):
    """Verifies that the RowLinearOperator satisfies all axioms."""
    s1, s2 = euclidean_subspaces
    op1 = LinearOperator.from_matrix(s1, s1, np.random.randn(2, 2))
    op2 = LinearOperator.from_matrix(s2, s1, np.random.randn(2, 3))
    operator = RowLinearOperator([op1, op2])
    # Create a second, compatible operator for algebraic checks
    op3 = LinearOperator.from_matrix(s1, s1, np.random.randn(2, 2))
    op4 = LinearOperator.from_matrix(s2, s1, np.random.randn(2, 3))
    operator2 = RowLinearOperator([op3, op4])
    operator.check(n_checks=3, op2=operator2)


def test_column_linear_operator_axioms(euclidean_subspaces: list[EuclideanSpace]):
    """Verifies that the ColumnLinearOperator satisfies all axioms."""
    s1, s2 = euclidean_subspaces
    op1 = LinearOperator.from_matrix(s1, s2, np.random.randn(3, 2))
    op2 = LinearOperator.from_matrix(s1, s1, np.random.randn(2, 2))
    operator = ColumnLinearOperator([op1, op2])
    # Create a second, compatible operator for algebraic checks
    op3 = LinearOperator.from_matrix(s1, s2, np.random.randn(3, 2))
    op4 = LinearOperator.from_matrix(s1, s1, np.random.randn(2, 2))
    operator2 = ColumnLinearOperator([op3, op4])
    operator.check(n_checks=3, op2=operator2)


# =============================================================================
# Test Suite 3: Specific Properties and Composition Rules
# =============================================================================


class TestDirectSumProperties:
    """Tests specific mathematical properties of direct sums and block operators."""

    def test_projection_inclusion_identity(self, euclidean_subspaces):
        """Tests that P_i @ J_i = Id_i for projection and inclusion."""
        direct_sum = HilbertSpaceDirectSum(euclidean_subspaces)
        for i, subspace in enumerate(euclidean_subspaces):
            proj = direct_sum.subspace_projection(i)
            incl = direct_sum.subspace_inclusion(i)
            composed_op = proj @ incl

            identity_matrix = np.eye(subspace.dim)
            assert_allclose(composed_op.matrix(dense=True), identity_matrix)

    def test_partition_of_unity(self, euclidean_subspaces):
        """Tests that sum(J_i @ P_i) = Id on the direct sum space."""
        direct_sum = HilbertSpaceDirectSum(euclidean_subspaces)
        identity_sum = direct_sum.zero_operator(direct_sum)

        for i in range(len(euclidean_subspaces)):
            proj = direct_sum.subspace_projection(i)
            incl = direct_sum.subspace_inclusion(i)
            identity_sum += incl @ proj

        identity_matrix = np.eye(direct_sum.dim)
        assert_allclose(identity_sum.matrix(dense=True), identity_matrix)

    def test_block_operator_composition(self, block_operators):
        """Tests that (P_i @ A @ J_j) is equivalent to A_ij."""
        block_op = BlockLinearOperator(block_operators)
        direct_sum_domain = block_op.domain
        direct_sum_codomain = block_op.codomain

        for i in range(block_op.row_dim):
            for j in range(block_op.col_dim):
                proj_i = direct_sum_codomain.subspace_projection(i)
                incl_j = direct_sum_domain.subspace_inclusion(j)

                extracted_block = proj_i @ block_op @ incl_j
                original_block = block_op.block(i, j)

                assert_allclose(
                    extracted_block.matrix(dense=True),
                    original_block.matrix(dense=True),
                )

    def test_block_diagonal_off_diagonal_is_zero(self, euclidean_subspaces):
        """Tests that off-diagonal blocks of a BlockDiagonalLinearOperator are zero."""
        s1, s2 = euclidean_subspaces
        op1 = LinearOperator.from_matrix(s1, s1, np.random.randn(2, 2))
        op2 = LinearOperator.from_matrix(s2, s2, np.random.randn(3, 3))
        block_diag_op = BlockDiagonalLinearOperator([op1, op2])

        for i in range(block_diag_op.row_dim):
            for j in range(block_diag_op.col_dim):
                if i != j:
                    block = block_diag_op.block(i, j)
                    zero_matrix = np.zeros((block.codomain.dim, block.domain.dim))
                    assert_allclose(block.matrix(dense=True), zero_matrix)

    def test_adjoint_of_column_is_row(self, euclidean_subspaces):
        """Tests that the adjoint of a ColumnLinearOperator is a RowLinearOperator."""
        s1, s2 = euclidean_subspaces
        op1 = LinearOperator.from_matrix(s1, s2, np.random.randn(3, 2))
        op2 = LinearOperator.from_matrix(s1, s1, np.random.randn(2, 2))

        column_op = ColumnLinearOperator([op1, op2])
        adjoint_op = column_op.adjoint

        assert isinstance(adjoint_op.domain, HilbertSpaceDirectSum)
        assert not isinstance(adjoint_op.codomain, HilbertSpaceDirectSum)

        manual_row_op = RowLinearOperator([op1.adjoint, op2.adjoint])

        assert_allclose(adjoint_op.matrix(dense=True), manual_row_op.matrix(dense=True))


# =============================================================================
# Test Suite 4: Error Handling and Edge Cases
# =============================================================================


class TestDirectSumErrorHandling:
    """Tests for exceptions and invalid inputs."""

    def test_block_operator_mismatched_domains(self, euclidean_subspaces):
        """Tests ValueError for inconsistent domains in a column."""
        s1, s2 = euclidean_subspaces
        s3 = EuclideanSpace(4)
        op11 = LinearOperator.from_matrix(s1, s1, np.random.randn(2, 2))
        op21_bad = LinearOperator.from_matrix(s3, s2, np.random.randn(3, 4))
        with pytest.raises(ValueError):
            BlockLinearOperator([[op11], [op21_bad]])

    def test_block_operator_mismatched_codomains(self, euclidean_subspaces):
        """Tests ValueError for inconsistent codomains in a row."""
        s1, s2 = euclidean_subspaces
        s3 = EuclideanSpace(4)
        op11 = LinearOperator.from_matrix(s1, s1, np.random.randn(2, 2))
        op12_bad = LinearOperator.from_matrix(s2, s3, np.random.randn(4, 3))
        with pytest.raises(ValueError):
            BlockLinearOperator([[op11, op12_bad]])

    def test_row_operator_mismatched_codomains(self, euclidean_subspaces):
        """Tests ValueError for inconsistent codomains in a RowLinearOperator."""
        s1, s2 = euclidean_subspaces
        op1 = LinearOperator.from_matrix(s1, s1, np.random.randn(2, 2))
        op2 = LinearOperator.from_matrix(s2, s2, np.random.randn(3, 3))
        with pytest.raises(ValueError):
            RowLinearOperator([op1, op2])

    def test_column_operator_mismatched_domains(self, euclidean_subspaces):
        """Tests ValueError for inconsistent domains in a ColumnLinearOperator."""
        s1, s2 = euclidean_subspaces
        op1 = LinearOperator.from_matrix(s1, s1, np.random.randn(2, 2))
        op2 = LinearOperator.from_matrix(s2, s2, np.random.randn(3, 3))
        with pytest.raises(ValueError):
            ColumnLinearOperator([op1, op2])

    @pytest.mark.parametrize(
        "op_class", [BlockLinearOperator, RowLinearOperator, ColumnLinearOperator]
    )
    def test_empty_operator_list_raises_error(self, op_class):
        """Tests ValueError when initializing block operators with empty lists."""
        with pytest.raises(ValueError):
            op_class([])
        if op_class == BlockLinearOperator:
            with pytest.raises(ValueError):
                op_class([[]])

    def test_block_indexing_errors(self, block_operators):
        """Tests that out-of-bounds block indexing raises an error."""
        block_op = BlockLinearOperator(block_operators)
        with pytest.raises(ValueError):
            block_op.block(2, 0)
        with pytest.raises(ValueError):
            block_op.block(0, 2)

    def test_dual_isomorphism_wrong_number_of_forms(self, euclidean_subspaces):
        """Tests ValueError for canonical_dual_isomorphism with wrong input size."""
        direct_sum = HilbertSpaceDirectSum(euclidean_subspaces)
        s1, s2 = euclidean_subspaces
        form1 = s1.to_dual(s1.random())
        # Provide only one form when two are expected
        with pytest.raises(ValueError):
            direct_sum.canonical_dual_isomorphism([form1])
