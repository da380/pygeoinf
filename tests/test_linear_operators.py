"""
A comprehensive test suite for the LinearOperator class, its factories,
and its specialized subclasses, consistent with the HilbertSpace API.
"""

import pytest
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator as ScipyLinOp


import pygeoinf as inf
from pygeoinf.symmetric_space.circle import Sobolev as CircleSobolev
from pygeoinf.symmetric_space.sphere import Sobolev as SphereSobolev


# For reproducibility in all tests
@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(42)


class TestOperatorFactories:
    """
    Tests for the various static factory and construction methods.
    """

    def test_from_matrix_standard(self):
        domain, codomain = inf.EuclideanSpace(3), inf.EuclideanSpace(2)
        M = np.random.randn(2, 3)
        op = inf.LinearOperator.from_matrix(domain, codomain, M, galerkin=False)
        op2 = inf.LinearOperator.from_matrix(
            domain, codomain, np.random.randn(2, 3), galerkin=False
        )

        assert np.allclose(op.matrix(dense=True), M)
        op.check(n_checks=3, op2=op2)

    def test_from_matrix_galerkin(self):
        domain, codomain = inf.EuclideanSpace(2), inf.EuclideanSpace(2)
        M = np.random.randn(2, 2)
        op = inf.LinearOperator.from_matrix(domain, codomain, M, galerkin=True)
        op2 = inf.LinearOperator.from_matrix(
            domain, codomain, np.random.randn(2, 2), galerkin=True
        )

        assert np.allclose(op.matrix(dense=True, galerkin=True), M)
        op.check(n_checks=3, op2=op2)

    def test_self_adjoint_from_matrix(self):
        """Tests creating a self-adjoint operator from a symmetric matrix."""
        space = inf.EuclideanSpace(3)
        M = np.random.randn(3, 3)
        M_symm = M + M.T
        op = inf.LinearOperator.self_adjoint_from_matrix(space, M_symm)
        M2 = np.random.randn(3, 3)
        M2_symm = M2 + M2.T
        op2 = inf.LinearOperator.self_adjoint_from_matrix(space, M2_symm)

        x, y = space.random(), space.random()
        assert np.isclose(space.inner_product(op(x), y), space.inner_product(x, op(y)))

        op.check(n_checks=3, op2=op2)

    def test_from_linear_forms(self):
        """Tests creating an operator from a list of LinearForms."""
        domain = inf.EuclideanSpace(3)
        form1 = inf.LinearForm(domain, components=np.array([1, 2, 3]))
        form2 = inf.LinearForm(domain, components=np.array([4, 5, 6]))
        op = inf.LinearOperator.from_linear_forms([form1, form2])
        form3 = inf.LinearForm(domain, components=np.random.randn(3))
        form4 = inf.LinearForm(domain, components=np.random.randn(3))
        op2 = inf.LinearOperator.from_linear_forms([form3, form4])

        expected_matrix = np.array([[1, 2, 3], [4, 5, 6]])
        assert np.allclose(op.matrix(dense=True), expected_matrix)

        op.check(n_checks=3, op2=op2)

    def test_from_tensor_product(self):
        """Tests creating an operator from a sum of tensor products."""
        domain, codomain = inf.EuclideanSpace(2), inf.EuclideanSpace(3)
        u1, v1 = codomain.random(), domain.random()
        u2, v2 = codomain.random(), domain.random()
        op = inf.LinearOperator.from_tensor_product(
            domain, codomain, [(u1, v1), (u2, v2)]
        )
        u3, v3 = codomain.random(), domain.random()
        u4, v4 = codomain.random(), domain.random()
        op2 = inf.LinearOperator.from_tensor_product(
            domain, codomain, [(u3, v3), (u4, v4)]
        )

        x = domain.random()
        y_op = op(x)
        y_manual = domain.inner_product(x, v1) * u1 + domain.inner_product(x, v2) * u2
        assert np.allclose(y_op, y_manual)

        op.check(n_checks=3, op2=op2)

    def test_from_formal_adjoint_simple(self):
        """Tests from_formal_adjoint on a simple MassWeightedHilbertSpace."""
        base_space = inf.EuclideanSpace(3)
        mass_mat = np.array([[3, 0.1, 0.2], [0.1, 2, 0], [0.2, 0, 1]])
        mass_op = inf.LinearOperator.from_matrix(
            base_space, base_space, mass_mat, galerkin=True
        )
        inv_mass_op = inf.LinearOperator.from_matrix(
            base_space, base_space, np.linalg.inv(mass_mat), galerkin=True
        )
        weighted_space = inf.MassWeightedHilbertSpace(base_space, mass_op, inv_mass_op)
        A_base = inf.LinearOperator.from_matrix(
            base_space, base_space, np.random.randn(3, 3)
        )
        A_weighted = inf.LinearOperator.from_formal_adjoint(
            weighted_space, weighted_space, A_base
        )
        A_base2 = inf.LinearOperator.from_matrix(
            base_space, base_space, np.random.randn(3, 3)
        )
        A_weighted2 = inf.LinearOperator.from_formal_adjoint(
            weighted_space, weighted_space, A_base2
        )

        A_weighted.check(n_checks=3, op2=A_weighted2)

    def test_from_formal_adjoint_symmetric_direct_sum(self):
        """Tests from_formal_adjoint on a direct sum of a circle and a sphere
        Sobolev space, representing a complex, mixed-geometry model space.
        """
        circle_sob = CircleSobolev(16, 2.0, 0.1)
        sphere_sob = SphereSobolev(16, 2.0, 0.2)
        domain_full = inf.HilbertSpaceDirectSum([circle_sob, sphere_sob])

        circle_leb = circle_sob.underlying_space
        sphere_leb = sphere_sob.underlying_space
        domain_base = inf.HilbertSpaceDirectSum([circle_leb, sphere_leb])

        op1_base = circle_leb.invariant_automorphism(lambda eig: 1.0 / (1.0 + eig))
        op2_base = sphere_leb.invariant_automorphism(lambda eig: 1.0 / (1.0 + eig))
        A_base = inf.BlockDiagonalLinearOperator([op1_base, op2_base])

        A_full = inf.LinearOperator.from_formal_adjoint(
            domain_full, domain_full, A_base
        )

        op3_base = circle_leb.invariant_automorphism(
            lambda eig: 1.0 / (1.0 + 0.5 * eig)
        )
        op4_base = sphere_leb.invariant_automorphism(
            lambda eig: 1.0 / (1.0 + 0.2 * eig)
        )
        A_base2 = inf.BlockDiagonalLinearOperator([op3_base, op4_base])
        A_full2 = inf.LinearOperator.from_formal_adjoint(
            domain_full, domain_full, A_base2
        )

        A_full.check(n_checks=3, op2=A_full2)

    def test_from_tensor_product(self):
        domain, codomain = inf.EuclideanSpace(2), inf.EuclideanSpace(3)
        u1, v1 = codomain.random(), domain.random()
        u2, v2 = codomain.random(), domain.random()
        op = inf.LinearOperator.from_tensor_product(
            domain, codomain, [(u1, v1), (u2, v2)]
        )
        u3, v3 = codomain.random(), domain.random()
        op2 = inf.LinearOperator.from_tensor_product(domain, codomain, [(u3, v3)])

        x = domain.random()
        y_op = op(x)
        y_manual = domain.inner_product(x, v1) * u1 + domain.inner_product(x, v2) * u2
        assert np.allclose(y_op, y_manual)
        op.check(n_checks=3, op2=op2)

    def test_from_matrix_dispatcher(self):
        """
        Tests that the from_matrix factory dispatches to the correct subclass
        based on the input matrix type.
        """
        domain = inf.EuclideanSpace(3)
        codomain = inf.EuclideanSpace(3)

        # 1. Test Dense Case -> DenseMatrixLinearOperator
        dense_mat = np.eye(3)
        dense_op = inf.LinearOperator.from_matrix(domain, codomain, dense_mat)
        assert isinstance(dense_op, inf.DenseMatrixLinearOperator)
        assert not dense_op.is_galerkin

        # 2. Test Sparse Case -> SparseMatrixLinearOperator
        sparse_mat = sp.csr_array(dense_mat)
        sparse_op = inf.LinearOperator.from_matrix(
            domain, codomain, sparse_mat, galerkin=True
        )
        assert isinstance(sparse_op, inf.SparseMatrixLinearOperator)
        assert sparse_op.is_galerkin

        # 3. Test Diagonal Sparse Case -> DiagonalSparseMatrixLinearOperator
        diag_mat = sp.dia_array(dense_mat)
        diag_op = inf.LinearOperator.from_matrix(domain, codomain, diag_mat)
        assert isinstance(diag_op, inf.DiagonalSparseMatrixLinearOperator)

        # 4. Test Matrix-Free Case -> MatrixLinearOperator
        # This requires the alias from your file:
        # from scipy.sparse.linalg import LinearOperator as ScipyLinOp
        scipy_op = ScipyLinOp((3, 3), matvec=lambda x: x)
        free_op = inf.LinearOperator.from_matrix(domain, codomain, scipy_op)
        # It should be the base MatrixLinearOperator, not a more specific one
        assert type(free_op) is inf.MatrixLinearOperator

    def test_self_adjoint_from_matrix_dispatcher(self):
        """Tests that the self_adjoint factory dispatches to the correct class."""
        space = inf.EuclideanSpace(3)

        # Test dense case
        dense_mat = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
        dense_op = inf.LinearOperator.self_adjoint_from_matrix(space, dense_mat)
        assert isinstance(dense_op, inf.DenseMatrixLinearOperator)
        assert dense_op.is_galerkin

        # Test sparse case
        sparse_mat = sp.csr_array(dense_mat)
        sparse_op = inf.LinearOperator.self_adjoint_from_matrix(space, sparse_mat)
        assert isinstance(sparse_op, inf.SparseMatrixLinearOperator)
        assert sparse_op.is_galerkin


class TestMatrixRepresentationLogic:
    """
    Tests to explicitly verify the correctness of matrix representations,
    especially in non-Euclidean spaces where it matters most.
    """

    def test_galerkin_vs_standard_with_mass_matrix(self):
        """
        CRITICAL TEST: Verifies that the Galerkin matrix (G) is correctly
        computed as M @ A, where M is the mass matrix and A is the standard matrix.

        This test is now corrected to use the proper MassWeightedHilbertSpace API.
        """

        underlying_space = inf.EuclideanSpace(2)
        mass_matrix = np.array([[2.0, 0.5], [0.5, 3.0]])
        inv_mass_matrix = np.linalg.inv(mass_matrix)

        mass_op = inf.LinearOperator.from_matrix(
            underlying_space, underlying_space, mass_matrix, galerkin=True
        )
        inv_mass_op = inf.LinearOperator.from_matrix(
            underlying_space, underlying_space, inv_mass_matrix, galerkin=True
        )

        space = inf.MassWeightedHilbertSpace(underlying_space, mass_op, inv_mass_op)

        std_matrix = np.array([[1, 2], [3, 4]])
        op = inf.LinearOperator.from_matrix(space, space, std_matrix, galerkin=False)

        galerkin_matrix_from_op = op.matrix(dense=True, galerkin=True)

        expected_galerkin_matrix = mass_matrix @ std_matrix

        assert np.allclose(galerkin_matrix_from_op, expected_galerkin_matrix)
        assert not np.allclose(galerkin_matrix_from_op, std_matrix)


class TestMatrixOperatorSpecializations:
    """
    Tests for the specialized MatrixLinearOperator and its subclasses.
    """

    def test_dense_matrix_op_getitem(self):
        """Tests component-wise access for DenseMatrixLinearOperator."""
        domain = inf.EuclideanSpace(2)
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        op = inf.DenseMatrixLinearOperator(domain, domain, matrix)

        assert op[0, 1] == 2.0
        assert np.allclose(op[1, :], np.array([3.0, 4.0]))

    def test_matrix_op_optimized_diagonals(self):
        """Confirms the optimized diagonal extraction override works correctly."""
        domain = inf.EuclideanSpace(3)
        matrix = np.array([[1, 2, 3], [2, 5, 6], [3, 6, 9]])
        op = inf.LinearOperator.from_matrix(domain, domain, matrix, galerkin=True)

        diagonals, offsets = op.extract_diagonals([-1, 0, 1], galerkin=True)
        expected_diagonals = np.array(
            [[2.0, 6.0, 0.0], [1.0, 5.0, 9.0], [0.0, 2.0, 6.0]]
        )
        assert np.allclose(diagonals, expected_diagonals)


@pytest.fixture
def sparse_op_fixture():
    """Provides a standard SparseMatrixLinearOperator for testing."""
    domain = inf.EuclideanSpace(3)
    codomain = inf.EuclideanSpace(3)
    # A simple sparse array in COO format to initialize
    row = np.array([0, 0, 1, 2, 2])
    col = np.array([0, 2, 1, 0, 2])
    data = np.array([1, 2, 3, 4, 5])
    sparse_array = sp.coo_array((data, (row, col)), shape=(3, 3))

    # The operator will internally convert this to csr_array
    op = inf.SparseMatrixLinearOperator(domain, codomain, sparse_array)
    return op, sparse_array.toarray()


class TestSparseMatrixOperator:
    """
    Tests the specialized SparseMatrixLinearOperator subclass.
    """

    def test_initialization_with_sparray(self, sparse_op_fixture):
        op, _ = sparse_op_fixture
        assert isinstance(op, inf.SparseMatrixLinearOperator)
        # Check that the internal format is csr_array as designed
        assert isinstance(op._matrix, sp.csr_array)

    def test_initialization_fails_with_dense(self):
        """Verifies the strict type check for modern sparray objects."""
        domain = inf.EuclideanSpace(2)
        dense_matrix = np.array([[1, 2], [3, 4]])
        with pytest.raises(TypeError):
            inf.SparseMatrixLinearOperator(domain, domain, dense_matrix)

    def test_compute_dense_matrix(self, sparse_op_fixture):
        """Tests the optimized .toarray() path."""
        op, expected_dense = sparse_op_fixture
        dense_from_op = op.matrix(dense=True)
        assert np.allclose(dense_from_op, expected_dense)

    def test_extract_diagonal(self, sparse_op_fixture):
        """Tests the optimized main diagonal extraction."""
        op, expected_dense = sparse_op_fixture
        diagonal_from_op = op.extract_diagonal()
        expected_diagonal = np.diag(expected_dense)
        assert np.allclose(diagonal_from_op, expected_diagonal)

    def test_extract_diagonals(self, sparse_op_fixture):
        """Tests the optimized extraction of multiple diagonals."""
        op, _ = sparse_op_fixture
        diagonals, offsets = op.extract_diagonals([-1, 0, 1])

        expected_diagonals = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 3.0, 5.0],
                [0.0, 0.0, 0.0],
            ]
        )
        assert np.allclose(diagonals, expected_diagonals)
        assert offsets == [-1, 0, 1]

    def test_getitem_access(self, sparse_op_fixture):
        """Tests component and slice access."""
        op, _ = sparse_op_fixture

        # Single element access
        assert op[0, 2] == 2.0
        assert op[1, 0] == 0.0  # Test a zero element

        # Row slice access
        row_slice = op[1, :]
        assert isinstance(row_slice, sp.sparray)
        assert np.allclose(row_slice.toarray().flatten(), np.array([0, 3, 0]))


@pytest.fixture
def diag_sparse_op_fixture():
    """Provides a DiagonalSparseMatrixLinearOperator with non-contiguous diagonals."""
    domain = inf.EuclideanSpace(4)
    # Diagonals are k=-1, k=0, and k=2
    offsets = [-1, 0, 2]
    # Note the required padding for the data array
    data = np.array(
        [
            [1.0, 2.0, 3.0, 0.0],  # k = -1
            [4.0, 5.0, 6.0, 7.0],  # k = 0
            [0.0, 0.0, 8.0, 9.0],  # k = 2
        ]
    )
    diagonals_tuple = (data, offsets)
    op = inf.DiagonalSparseMatrixLinearOperator(domain, domain, diagonals_tuple)
    return op, diagonals_tuple


class TestDiagonalSparseOperator:
    """
    Tests the specialized DiagonalSparseMatrixLinearOperator subclass.
    """

    def test_initialization(self, diag_sparse_op_fixture):
        """Tests that the operator is initialized correctly."""
        op, (data, offsets) = diag_sparse_op_fixture
        assert isinstance(op, inf.DiagonalSparseMatrixLinearOperator)
        assert isinstance(op._matrix, sp.dia_array)
        assert np.array_equal(op.offsets, offsets)

    def test_convenience_properties(self):
        """Tests the .is_strictly_diagonal property."""
        domain = inf.EuclideanSpace(3)
        # Strictly diagonal case
        op1 = inf.DiagonalSparseMatrixLinearOperator(
            domain, domain, (np.array([[1, 2, 3]]), [0])
        )
        # Multi-diagonal case
        op2 = inf.DiagonalSparseMatrixLinearOperator(
            domain, domain, (np.array([[1, 2, 0], [0, 3, 4]]), [-1, 1])
        )
        assert op1.is_strictly_diagonal
        assert not op2.is_strictly_diagonal

    def test_optimized_extract_diagonals_full(self, diag_sparse_op_fixture):
        """
        Tests extracting the same diagonals that are natively stored,
        which should be a very fast operation.
        """
        op, (expected_data, expected_offsets) = diag_sparse_op_fixture
        data, offsets = op.extract_diagonals(expected_offsets)
        assert np.allclose(data, expected_data)
        assert offsets == expected_offsets

    def test_optimized_extract_diagonals_subset_and_missing(
        self, diag_sparse_op_fixture
    ):
        """
        Tests extracting a subset of stored diagonals and asking for one
        that does not exist (which should be all zeros).
        """
        op, _ = diag_sparse_op_fixture

        # Ask for k=0 (stored), k=1 (not stored), and k=2 (stored)
        requested_offsets = [0, 1, 2]
        data, offsets = op.extract_diagonals(requested_offsets)

        expected_data = np.array(
            [
                [4.0, 5.0, 6.0, 7.0],  # k = 0 (from fixture)
                [0.0, 0.0, 0.0, 0.0],  # k = 1 (should be zero)
                [0.0, 0.0, 8.0, 9.0],  # k = 2 (from fixture)
            ]
        )

        assert np.allclose(data, expected_data)
        assert offsets == requested_offsets

    def test_from_operator_factory(self, sparse_op_fixture):
        """
        Tests the factory method that creates a diagonal approximation
        of another operator.
        """
        source_op, _ = sparse_op_fixture
        offsets_to_keep = [-1, 0, 1]

        # Create the diagonal approximation
        diag_op = inf.DiagonalSparseMatrixLinearOperator.from_operator(
            source_op, offsets_to_keep
        )

        assert isinstance(diag_op, inf.DiagonalSparseMatrixLinearOperator)

        # Get the expected diagonals from the source
        expected_diagonals, _ = source_op.extract_diagonals(offsets_to_keep)

        # Get the actual diagonals from the new operator
        actual_diagonals, _ = diag_op.extract_diagonals(offsets_to_keep)

        assert np.allclose(actual_diagonals, expected_diagonals)

    def test_from_diagonal_values_factory(self):
        """
        Tests the factory method for creating purely diagonal operators.
        """
        domain = inf.EuclideanSpace(3)
        diagonal_values = np.array([1.0, 5.0, 9.0])

        # Test creation with default (galerkin=False)
        op = inf.DiagonalSparseMatrixLinearOperator.from_diagonal_values(
            domain, domain, diagonal_values
        )

        assert isinstance(op, inf.DiagonalSparseMatrixLinearOperator)
        assert op.is_strictly_diagonal
        assert not op.is_galerkin
        assert np.allclose(op.extract_diagonal(), diagonal_values)

        # Test creation with galerkin=True
        op_galerkin = inf.DiagonalSparseMatrixLinearOperator.from_diagonal_values(
            domain, domain, diagonal_values, galerkin=True
        )
        assert op_galerkin.is_galerkin
        assert np.allclose(op_galerkin.extract_diagonal(galerkin=True), diagonal_values)

        # Test for dimension mismatch error
        short_values = np.array([1.0, 2.0])
        with pytest.raises(ValueError):
            inf.DiagonalSparseMatrixLinearOperator.from_diagonal_values(
                domain, domain, short_values
            )

    def test_getattr_proxy_methods(self):
        """
        Tests the __getattr__ proxy for element-wise functions and properties.
        """
        domain = inf.EuclideanSpace(3)
        # Use values suitable for both abs() and sqrt() tests
        op_pos = inf.DiagonalSparseMatrixLinearOperator.from_diagonal_values(
            domain, domain, np.array([4.0, 9.0, 16.0])
        )
        op_mix = inf.DiagonalSparseMatrixLinearOperator.from_diagonal_values(
            domain, domain, np.array([-4.0, 9.0, -16.0])
        )

        # Test the __abs__ dunder method
        abs_op = abs(op_mix)
        assert isinstance(abs_op, inf.DiagonalSparseMatrixLinearOperator)
        assert np.allclose(abs_op.extract_diagonal(), [4.0, 9.0, 16.0])

        # Test a method that returns a raw value
        total = op_pos.sum()
        assert isinstance(total, (float, np.number))
        assert np.isclose(total, 29.0)

        # Test proxying a non-callable attribute
        assert op_pos.shape == (3, 3)

    def test_inverse_property(self):
        """
        Tests the .inverse property for strictly diagonal operators.
        """
        domain = inf.EuclideanSpace(3)
        diagonal_values = np.array([2.0, 4.0, 5.0])
        op = inf.DiagonalSparseMatrixLinearOperator.from_diagonal_values(
            domain, domain, diagonal_values
        )

        # Compute the inverse
        inv_op = op.inverse

        # Check that the product is the identity
        identity_op = op @ inv_op
        identity_matrix = identity_op.matrix(dense=True)

        assert isinstance(inv_op, inf.DiagonalSparseMatrixLinearOperator)
        assert np.allclose(identity_matrix, np.eye(3))
        assert np.allclose(inv_op.extract_diagonal(), [0.5, 0.25, 0.2])

        # Test that it fails for a non-diagonal operator
        multi_diag_op = inf.DiagonalSparseMatrixLinearOperator(
            domain, domain, (np.array([[1, 2, 0], [0, 3, 4]]), [-1, 1])
        )
        with pytest.raises(NotImplementedError):
            _ = multi_diag_op.inverse

        # Test that it fails for a diagonal with zeros
        zero_diag_op = inf.DiagonalSparseMatrixLinearOperator.from_diagonal_values(
            domain, domain, np.array([1.0, 0.0, 3.0])
        )
        with pytest.raises(ValueError):
            _ = zero_diag_op.inverse

    def test_sqrt_property(self):
        """
        Tests the .sqrt property for strictly diagonal operators.
        """
        domain = inf.EuclideanSpace(3)
        diagonal_values = np.array([4.0, 9.0, 16.0])
        op = inf.DiagonalSparseMatrixLinearOperator.from_diagonal_values(
            domain, domain, diagonal_values
        )

        # Compute the square root
        sqrt_op = op.sqrt

        # Check that squaring the result gives back the original
        reconstructed_op = sqrt_op**2

        assert isinstance(sqrt_op, inf.DiagonalSparseMatrixLinearOperator)
        assert np.allclose(reconstructed_op.matrix(dense=True), op.matrix(dense=True))
        assert np.allclose(sqrt_op.extract_diagonal(), [2.0, 3.0, 4.0])

        # NEW: Test that it fails for a non-diagonal operator
        multi_diag_op = inf.DiagonalSparseMatrixLinearOperator(
            domain, domain, (np.array([[1, 2, 0], [0, 3, 4]]), [-1, 1])
        )
        with pytest.raises(NotImplementedError):
            _ = multi_diag_op.sqrt

        # Test that it fails for a diagonal with negative entries
        neg_val_op = inf.DiagonalSparseMatrixLinearOperator.from_diagonal_values(
            domain, domain, np.array([1.0, -4.0, 9.0])
        )
        with pytest.raises(ValueError):
            _ = neg_val_op.sqrt
