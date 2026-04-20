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

import pickle

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


class _ZeroDimScalarSpace(HilbertSpace):
    """Test-only zero-dimensional space that forbids component access."""

    @property
    def dim(self) -> int:
        return 0

    @property
    def zero(self) -> float:
        return 0.0

    def to_dual(self, x: float):
        return lambda y: x * y

    def from_dual(self, xp):
        raise RuntimeError("from_dual should not be called in this test")

    def to_components(self, x: float) -> np.ndarray:
        raise RuntimeError("to_components should not be called in this test")

    def from_components(self, c: np.ndarray) -> float:
        raise RuntimeError("from_components should not be called in this test")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ZeroDimScalarSpace)

    def is_element(self, x) -> bool:
        return isinstance(x, float)

    def add(self, x: float, y: float) -> float:
        return x + y

    def subtract(self, x: float, y: float) -> float:
        return x - y

    def multiply(self, a: float, x: float) -> float:
        return a * x

    def negative(self, x: float) -> float:
        return -x

    def copy(self, x: float) -> float:
        return float(x)

    def inner_product(self, x1: float, x2: float) -> float:
        return float(x1 * x2)


class TestDirectSumProperties:
    """Tests specific mathematical properties of direct sums and block operators."""

    def test_zero_componentwise_for_zero_dim_subspaces(self):
        """zero should not route through from_components for basis-free blocks."""
        direct_sum = HilbertSpaceDirectSum([_ZeroDimScalarSpace(), _ZeroDimScalarSpace()])

        assert direct_sum.zero == [0.0, 0.0]

    def test_inner_product_componentwise_for_zero_dim_subspaces(self):
        """inner_product should not route through to_components for basis-free blocks."""
        direct_sum = HilbertSpaceDirectSum([_ZeroDimScalarSpace(), _ZeroDimScalarSpace()])

        value = direct_sum.inner_product([1.5, -2.0], [3.0, 4.0])
        assert_allclose(value, 1.5 * 3.0 + (-2.0) * 4.0)

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
# Additional Regression Coverage: Basis-Free Direct Sums
# =============================================================================


class _BasisFreeSpace(HilbertSpace):
    """
    Minimal HilbertSpace with dim=0 and no basis.

    to_components / from_components raise AssertionError to detect any
    path that inadvertently calls them. zero, random, and inner_product
    are provided directly so that HilbertSpaceDirectSum can be tested
    without touching the component representations.
    """

    def __init__(self, scale: float = 1.0):
        self._scale = scale

    @property
    def dim(self) -> int:
        return 0

    def to_components(self, x):
        raise AssertionError(
            "_BasisFreeSpace.to_components must not be called in basis-free mode"
        )

    def from_components(self, c):
        raise AssertionError(
            "_BasisFreeSpace.from_components must not be called in basis-free mode"
        )

    @property
    def zero(self):
        return np.array(0.0)

    def random(self):
        return np.array(np.random.randn())

    def inner_product(self, x, y):
        return float(x) * float(y) * self._scale

    def to_dual(self, x):
        from pygeoinf.linear_forms import LinearForm as LF

        lf = LF.__new__(LF)
        lf.__dict__["_domain"] = self
        lf.__dict__["_components"] = np.array([])
        lf.__dict__["_compute"] = (
            lambda v: float(v) * self._scale * float(x) / self._scale
        )
        return lf

    def from_dual(self, xp):
        return np.array(xp(np.array(1.0)) / self._scale)

    def add(self, x, y):
        return np.array(float(x) + float(y))

    def subtract(self, x, y):
        return np.array(float(x) - float(y))

    def multiply(self, a, x):
        return np.array(a * float(x))

    def ax(self, a, x):
        x *= a

    def axpy(self, a, x, y):
        result = np.array(float(y) + a * float(x))
        y[()] = result
        return y

    def copy(self, x):
        return np.array(float(x))

    def is_element(self, x):
        return isinstance(x, (float, np.floating, np.ndarray))

    def __eq__(self, other):
        if not isinstance(other, _BasisFreeSpace):
            return NotImplemented
        return self._scale == other._scale


class TestBasisFreeDirectSum:
    """
    Regression tests: HilbertSpaceDirectSum must not route zero / random /
    inner_product through to_components / from_components when the subspace
    is basis-free (dim=0).
    """

    def test_zero_does_not_call_from_components(self):
        """zero must be computed from subspace zeros, not from_components."""
        space = _BasisFreeSpace()
        ds = HilbertSpaceDirectSum([space, space])
        z = ds.zero
        assert isinstance(z, list)
        assert len(z) == 2
        assert_allclose(float(z[0]), 0.0)
        assert_allclose(float(z[1]), 0.0)

    def test_random_does_not_call_from_components(self):
        """random must be computed from subspace randoms, not from_components."""
        space = _BasisFreeSpace()
        ds = HilbertSpaceDirectSum([space, space])
        np.random.seed(42)
        v = ds.random()
        assert isinstance(v, list)
        assert len(v) == 2
        assert np.isfinite(float(v[0]))
        assert np.isfinite(float(v[1]))

    def test_inner_product_does_not_call_to_components(self):
        """inner_product must use subspace inner products, not to_components."""
        space_a = _BasisFreeSpace(scale=2.0)
        space_b = _BasisFreeSpace(scale=3.0)
        ds = HilbertSpaceDirectSum([space_a, space_b])

        xs = [np.array(2.0), np.array(4.0)]
        ys = [np.array(3.0), np.array(5.0)]

        ip = ds.inner_product(xs, ys)
        assert_allclose(ip, 72.0)

    def test_nested_basis_free_direct_sum_zero(self):
        """Nested direct sums of basis-free spaces return nested zero lists."""
        space = _BasisFreeSpace()
        inner_ds = HilbertSpaceDirectSum([space, space])
        outer_ds = HilbertSpaceDirectSum([inner_ds, space])

        z = outer_ds.zero
        assert isinstance(z, list) and len(z) == 2
        inner_z = z[0]
        assert isinstance(inner_z, list) and len(inner_z) == 2
        assert_allclose(float(inner_z[0]), 0.0)
        assert_allclose(float(inner_z[1]), 0.0)
        assert_allclose(float(z[1]), 0.0)

    def test_mixed_basis_free_and_euclidean_inner_product(self):
        """Direct sum of a basis-free space + EuclideanSpace must work."""
        bf = _BasisFreeSpace(scale=1.0)
        eu = EuclideanSpace(2)
        ds = HilbertSpaceDirectSum([bf, eu])

        z = ds.zero
        assert isinstance(z, list) and len(z) == 2
        assert_allclose(float(z[0]), 0.0)
        assert_allclose(z[1], np.zeros(2))

        xs = [np.array(3.0), np.array([1.0, 2.0])]
        ys = [np.array(4.0), np.array([2.0, 3.0])]
        assert_allclose(ds.inner_product(xs, ys), 20.0)


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


# =============================================================================
# Test Suite 5: Serialization and Pickling Robustness
# =============================================================================


class TestSerialization:
    """
    Tests to ensure that direct sum spaces and block operators can be safely
    pickled and unpickled. This guarantees compatibility with multiprocessing
    libraries like joblib.
    """

    def test_pickle_direct_sum_space(self, euclidean_subspaces):
        """Tests pickling of HilbertSpaceDirectSum."""
        space = HilbertSpaceDirectSum(euclidean_subspaces)

        # Serialize and deserialize
        pickled_space = pickle.dumps(space)
        unpickled_space = pickle.loads(pickled_space)

        # Verify properties
        assert unpickled_space.dim == space.dim
        assert unpickled_space == space

        # Verify functional mapping
        x = space.random()
        assert np.allclose(space.to_components(x), unpickled_space.to_components(x))

    def test_pickle_block_linear_operator(self, block_operators):
        """Tests pickling of BlockLinearOperator."""
        operator = BlockLinearOperator(block_operators)

        unpickled_op = pickle.loads(pickle.dumps(operator))

        # Verify matrix assembly survived
        assert_allclose(operator.matrix(dense=True), unpickled_op.matrix(dense=True))

        # Verify forward mapping execution
        x = operator.domain.random()
        assert_allclose(
            operator.codomain.to_components(operator(x)),
            unpickled_op.codomain.to_components(unpickled_op(x)),
        )
        # Verify adjoint mapping execution
        y = operator.codomain.random()
        assert_allclose(
            operator.domain.to_components(operator.adjoint(y)),
            unpickled_op.domain.to_components(unpickled_op.adjoint(y)),
        )

    def test_pickle_block_diagonal_operator(self, euclidean_subspaces):
        """Tests pickling of BlockDiagonalLinearOperator."""
        s1, s2 = euclidean_subspaces
        op1 = LinearOperator.from_matrix(s1, s1, np.random.randn(2, 2))
        op2 = LinearOperator.from_matrix(s2, s2, np.random.randn(3, 3))
        operator = BlockDiagonalLinearOperator([op1, op2])

        unpickled_op = pickle.loads(pickle.dumps(operator))

        assert_allclose(operator.matrix(dense=True), unpickled_op.matrix(dense=True))

        x = operator.domain.random()
        assert_allclose(
            operator.codomain.to_components(operator(x)),
            unpickled_op.codomain.to_components(unpickled_op(x)),
        )

    def test_pickle_row_linear_operator(self, euclidean_subspaces):
        """Tests pickling of RowLinearOperator."""
        s1, s2 = euclidean_subspaces
        op1 = LinearOperator.from_matrix(s1, s1, np.random.randn(2, 2))
        op2 = LinearOperator.from_matrix(s2, s1, np.random.randn(2, 3))
        operator = RowLinearOperator([op1, op2])

        unpickled_op = pickle.loads(pickle.dumps(operator))

        assert_allclose(operator.matrix(dense=True), unpickled_op.matrix(dense=True))

        x = operator.domain.random()
        assert_allclose(
            operator.codomain.to_components(operator(x)),
            unpickled_op.codomain.to_components(unpickled_op(x)),
        )

    def test_pickle_column_linear_operator(self, euclidean_subspaces):
        """Tests pickling of ColumnLinearOperator."""
        s1, s2 = euclidean_subspaces
        op1 = LinearOperator.from_matrix(s1, s2, np.random.randn(3, 2))
        op2 = LinearOperator.from_matrix(s1, s1, np.random.randn(2, 2))
        operator = ColumnLinearOperator([op1, op2])

        unpickled_op = pickle.loads(pickle.dumps(operator))

        assert_allclose(operator.matrix(dense=True), unpickled_op.matrix(dense=True))

        x = operator.domain.random()
        assert_allclose(
            operator.codomain.to_components(operator(x)),
            unpickled_op.codomain.to_components(unpickled_op(x)),
        )

