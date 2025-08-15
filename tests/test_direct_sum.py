"""
Tests for the direct_sum module.
"""
import pytest
import numpy as np
from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.symmetric_space.circle import Sobolev
from pygeoinf.operators import LinearOperator
from pygeoinf.direct_sum import (
    HilbertSpaceDirectSum,
    BlockLinearOperator,
    BlockDiagonalLinearOperator,
    RowLinearOperator,
    ColumnLinearOperator,
)
from .checks.hilbert_space import HilbertSpaceChecks
from .checks.linear_operator import LinearOperatorChecks


# =============================================================================
# Fixtures for building direct sum spaces and block operators
# =============================================================================

@pytest.fixture
def euclidean_subspaces() -> list[EuclideanSpace]:
    """Provides a list of simple Euclidean spaces to form a direct sum."""
    return [EuclideanSpace(dim=2), EuclideanSpace(dim=3)]


@pytest.fixture
def mixed_subspaces() -> list[EuclideanSpace | Sobolev]:
    """Provides a mixed list of Euclidean and Sobolev spaces."""
    return [EuclideanSpace(dim=3), Sobolev(8, 1.0, 0.1)]


@pytest.fixture
def block_operators(euclidean_subspaces) -> list[list[LinearOperator]]:
    """Provides a 2x2 block matrix of linear operators between Euclidean spaces."""
    s1, s2 = euclidean_subspaces
    op11 = LinearOperator.from_matrix(s1, s1, np.random.randn(2, 2))
    op12 = LinearOperator.from_matrix(s2, s1, np.random.randn(2, 3))
    op21 = LinearOperator.from_matrix(s1, s2, np.random.randn(3, 2))
    op22 = LinearOperator.from_matrix(s2, s2, np.random.randn(3, 3))
    return [[op11, op12], [op21, op22]]


# =============================================================================
# Test Suite 1: Euclidean Spaces Only
# =============================================================================

class TestHilbertSpaceDirectSumEuclidean(HilbertSpaceChecks):
    """
    Runs Hilbert space checks on a direct sum of EuclideanSpaces.
    """
    @pytest.fixture
    def space(self, euclidean_subspaces: list[EuclideanSpace]) -> HilbertSpaceDirectSum:
        return HilbertSpaceDirectSum(euclidean_subspaces)


class TestBlockLinearOperatorEuclidean(LinearOperatorChecks):
    """
    Runs linear operator checks on a BlockLinearOperator between Euclidean spaces.
    """
    @pytest.fixture
    def operator(
        self, block_operators: list[list[LinearOperator]]
    ) -> BlockLinearOperator:
        return BlockLinearOperator(block_operators)


class TestBlockDiagonalLinearOperatorEuclidean(LinearOperatorChecks):
    """
    Runs linear operator checks on a BlockDiagonalLinearOperator.
    """
    @pytest.fixture
    def operator(self, euclidean_subspaces: list[EuclideanSpace]) -> BlockDiagonalLinearOperator:
        s1, s2 = euclidean_subspaces
        op1 = LinearOperator.from_matrix(s1, s1, np.random.randn(2, 2))
        op2 = LinearOperator.from_matrix(s2, s2, np.random.randn(3, 3))
        return BlockDiagonalLinearOperator([op1, op2])


# =============================================================================
# Test Suite 2: Mixed Euclidean and Sobolev Spaces
# =============================================================================

class TestHilbertSpaceDirectSumMixed(HilbertSpaceChecks):
    """
    Runs Hilbert space checks on a direct sum of a EuclideanSpace and a Sobolev space.
    """
    @pytest.fixture
    def space(self, mixed_subspaces: list[EuclideanSpace | Sobolev]) -> HilbertSpaceDirectSum:
        return HilbertSpaceDirectSum(mixed_subspaces)


# =============================================================================
# Simple instantiation tests for Row and Column operators
# =============================================================================

def test_row_linear_operator_instantiation(euclidean_subspaces):
    """Tests that a RowLinearOperator can be created successfully."""
    s1, s2 = euclidean_subspaces
    op1 = LinearOperator.from_matrix(s1, s1, np.random.randn(2, 2))
    op2 = LinearOperator.from_matrix(s2, s1, np.random.randn(2, 3))
    row_op = RowLinearOperator([op1, op2])
    assert row_op is not None
    assert row_op.row_dim == 1
    assert row_op.col_dim == 2


def test_column_linear_operator_instantiation(euclidean_subspaces):
    """Tests that a ColumnLinearOperator can be created successfully."""
    s1, s2 = euclidean_subspaces
    op1 = LinearOperator.from_matrix(s1, s2, np.random.randn(3, 2))
    op2 = LinearOperator.from_matrix(s1, s1, np.random.randn(2, 2))
    col_op = ColumnLinearOperator([op1, op2])
    assert col_op is not None
    assert col_op.row_dim == 2
    assert col_op.col_dim == 1
