import pytest
import numpy as np
from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.linear_operators import (
    LinearOperator,
    DiagonalLinearOperator,
    NormalSumOperator,
)


@pytest.fixture
def setup_spaces():
    domain = EuclideanSpace(3)
    codomain = EuclideanSpace(2)
    return domain, codomain


@pytest.fixture
def setup_operator(setup_spaces):
    domain, codomain = setup_spaces
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    return LinearOperator.from_matrix(domain, codomain, matrix)


class TestLinearOperatorMatrix:
    def test_matrix_dense(self, setup_operator):
        op = setup_operator
        matrix = op.matrix(dense=True)
        expected_matrix = np.array([[1, 2, 3], [4, 5, 6]])
        assert np.allclose(matrix, expected_matrix)

    def test_matrix_dense_galerkin(self, setup_operator):
        op = setup_operator
        # The Galerkin matrix should be different from the standard matrix
        # because the spaces are Euclidean, so the dual is the same as the space
        # and the Galerkin matrix is just the standard matrix.
        matrix = op.matrix(dense=True, galerkin=True)
        expected_matrix = np.array([[1, 2, 3], [4, 5, 6]])
        assert np.allclose(matrix, expected_matrix)

    def test_extract_diagonal(self, setup_operator):
        op = setup_operator
        diagonal = op.extract_diagonal(galerkin=False)
        expected_diagonal = np.array([1, 5])
        assert np.allclose(diagonal, expected_diagonal)

    def test_extract_diagonal_galerkin(self):
        domain = EuclideanSpace(3)
        matrix = np.array([[1, 2, 3], [2, 5, 6], [3, 6, 9]])
        op = LinearOperator.from_matrix(domain, domain, matrix, galerkin=True)
        diagonal = op.extract_diagonal(galerkin=True)
        expected_diagonal = np.array([1, 5, 9])
        assert np.allclose(diagonal, expected_diagonal)

    def test_extract_diagonals(self, setup_operator):
        op = setup_operator
        diagonals, offsets = op.extract_diagonals([0, 1], galerkin=False)
        assert np.allclose(diagonals, np.array([[1.0, 5.0], [0.0, 2.0]]))
        assert offsets == [0, 1]

    def test_extract_diagonals_galerkin(self):
        domain = EuclideanSpace(3)
        matrix = np.array([[1, 2, 3], [2, 5, 6], [3, 6, 9]])
        op = LinearOperator.from_matrix(domain, domain, matrix, galerkin=True)
        diagonals, offsets = op.extract_diagonals([-1, 0, 1], galerkin=True)
        assert np.allclose(
            diagonals, np.array([[2.0, 6.0, 0.0], [1.0, 5.0, 9.0], [0.0, 2.0, 6.0]])
        )
        assert offsets == [-1, 0, 1]

    def test_extract_diagonal_parallel(self, setup_operator):
        op = setup_operator
        diagonal = op.extract_diagonal(galerkin=False, parallel=True, n_jobs=-1)
        expected_diagonal = np.array([1, 5])
        assert np.allclose(diagonal, expected_diagonal)

    def test_extract_diagonals_parallel(self, setup_operator):
        op = setup_operator
        diagonals, offsets = op.extract_diagonals(
            [0, 1], galerkin=False, parallel=True, n_jobs=-1
        )
        assert np.allclose(diagonals, np.array([[1.0, 5.0], [0.0, 2.0]]))
        assert offsets == [0, 1]

    def test_extract_diagonals_single_negative_offset(self):
        domain = EuclideanSpace(3)
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        op = LinearOperator.from_matrix(domain, domain, matrix, galerkin=True)
        diagonals, offsets = op.extract_diagonals([-1], galerkin=True)
        assert np.allclose(diagonals, np.array([[4.0, 8.0, 0.0]]))
        assert offsets == [-1]


class TestDiagonalLinearOperator:
    @pytest.fixture
    def setup_diag_operator(self):
        domain = EuclideanSpace(3)
        diagonal_values = np.array([1.0, 2.0, 3.0])
        return DiagonalLinearOperator(domain, domain, diagonal_values)

    def test_extract_diagonal_galerkin(self, setup_diag_operator):
        op = setup_diag_operator
        diagonal = op.extract_diagonal(galerkin=True)
        expected_diagonal = np.array([1.0, 2.0, 3.0])
        assert np.allclose(diagonal, expected_diagonal)

    def test_extract_diagonal_no_galerkin(self, setup_diag_operator):
        op = setup_diag_operator
        diagonal = op.extract_diagonal(galerkin=False)
        expected_diagonal = np.array([1.0, 2.0, 3.0])
        assert np.allclose(diagonal, expected_diagonal)

    def test_extract_diagonals_galerkin(self, setup_diag_operator):
        op = setup_diag_operator
        diagonals, offsets = op.extract_diagonals([-1, 0, 1], galerkin=True)
        expected_diagonals = np.array(
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]
        )
        assert np.allclose(diagonals, expected_diagonals)
        assert offsets == [-1, 0, 1]

    def test_extract_diagonals_no_galerkin(self, setup_diag_operator):
        op = setup_diag_operator
        diagonals, offsets = op.extract_diagonals([-1, 0, 1], galerkin=False)
        expected_diagonals = np.array(
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]
        )
        assert np.allclose(diagonals, expected_diagonals)


class TestNormalSumOperator:
    @pytest.fixture
    def setup_normal_sum_operator(self):
        domain_A = EuclideanSpace(2)
        codomain_A = EuclideanSpace(3)
        A_matrix = np.array([[1, 2], [3, 4], [5, 6]])
        A = LinearOperator.from_matrix(domain_A, codomain_A, A_matrix)

        Q_matrix = np.array([[2, 0], [0, 3]])
        Q = LinearOperator.from_matrix(domain_A, domain_A, Q_matrix, galerkin=True)

        B_matrix = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 3]])
        B = LinearOperator.from_matrix(codomain_A, codomain_A, B_matrix, galerkin=True)

        return NormalSumOperator(A, Q, B)

    def test_extract_diagonal(self, setup_normal_sum_operator):
        op = setup_normal_sum_operator
        # A @ Q @ A.T + B
        # A = [[1, 2], [3, 4], [5, 6]]
        # Q = [[2, 0], [0, 3]]
        # B = [[1, 1, 1], [1, 2, 1], [1, 1, 3]]
        # A @ Q = [[2, 6], [6, 12], [10, 18]]
        # A.T = [[1, 3, 5], [2, 4, 6]]
        # A @ Q @ A.T = [[14, 30, 46], [30, 66, 102], [46, 102, 158]]
        # diag(A @ Q @ A.T) = [14, 66, 158]
        # diag(B) = [1, 2, 3]
        # expected_diagonal = [15, 68, 161]
        diagonal = op.extract_diagonal(galerkin=True)
        expected_diagonal = np.array([15.0, 68.0, 161.0])
        assert np.allclose(diagonal, expected_diagonal)

    def test_extract_diagonals(self, setup_normal_sum_operator):
        op = setup_normal_sum_operator
        diagonals, offsets = op.extract_diagonals([-1, 0, 1], galerkin=True)
        # diag(B) = [[1, 1, 0], [1, 2, 3], [0, 1, 1]]
        # diag(AQA.T) = [[30, 102, 0], [14, 66, 158], [0, 30, 102]]
        expected_diagonals = np.array(
            [[31.0, 103.0, 0.0], [15.0, 68.0, 161.0], [0.0, 31.0, 103.0]]
        )
        assert np.allclose(diagonals, expected_diagonals)
