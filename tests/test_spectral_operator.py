"""Tests for SpectralFractionalOperator."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.linear_operators import (
    DiagonalSparseMatrixLinearOperator,
    LinearOperator,
)
from pygeoinf.low_rank import LowRankEig
from pygeoinf.spectral_operator import (
    SpectralFractionalOperator,
    fractional_operators_from_eig,
)


def _make_low_rank_eig_from_dense(matrix: np.ndarray) -> LowRankEig:
    """Build a LowRankEig from a dense symmetric matrix (full rank)."""
    n = matrix.shape[0]
    space = EuclideanSpace(n)
    eigvals, U = np.linalg.eigh(matrix)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    U = U[:, idx]
    u_op = LinearOperator.from_matrix(space, space, U)
    d_op = DiagonalSparseMatrixLinearOperator.from_diagonal_values(
        space, space, eigvals
    )
    return LowRankEig(u_op, d_op)


def _spd_matrix(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return A.T @ A + 0.1 * np.eye(n)


@pytest.mark.parametrize("theta", [-1.0, -0.5, 0.5, 1.0, 2.0])
def test_fractional_power_matches_dense_eigh(theta):
    """A^theta via SpectralFractionalOperator matches np.linalg.eigh fallback."""
    n = 6
    C = _spd_matrix(n, seed=7)
    eigvals, U = np.linalg.eigh(C)
    expected = U @ np.diag(eigvals ** theta) @ U.T

    eig = _make_low_rank_eig_from_dense(C)
    op = SpectralFractionalOperator.from_low_rank_eig(eig, theta)

    rng = np.random.default_rng(0)
    for _ in range(5):
        v = rng.standard_normal(n)
        assert_allclose(op(v), expected @ v, rtol=1e-10, atol=1e-10)


def test_inverse_round_trip_on_eigenspace():
    """C^theta @ C^{-theta} is the identity on the eigenvector range."""
    n = 5
    C = _spd_matrix(n, seed=1)
    eig = _make_low_rank_eig_from_dense(C)

    A = SpectralFractionalOperator.from_low_rank_eig(eig, -0.4)
    A_inv = SpectralFractionalOperator.from_low_rank_eig(eig, 0.4)

    rng = np.random.default_rng(2)
    for _ in range(5):
        v = rng.standard_normal(n)
        round_trip = A_inv(A(v))
        assert_allclose(round_trip, v, rtol=1e-9, atol=1e-9)


def test_adjointness_symmetric_func():
    """For symmetric f, <A x, y> == <x, A y>."""
    n = 7
    C = _spd_matrix(n, seed=3)
    eig = _make_low_rank_eig_from_dense(C)
    op = SpectralFractionalOperator.from_low_rank_eig(eig, 0.3)

    rng = np.random.default_rng(4)
    for _ in range(5):
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        assert_allclose(
            np.dot(op(x), y), np.dot(x, op(y)), rtol=1e-10, atol=1e-10
        )


def test_match_diagonal_operator_power():
    """For diagonal A, our operator matches DiagonalSparseMatrixLinearOperator.

    DiagonalSparseMatrixLinearOperator supports __pow__ for fractional powers
    via functional calculus; we should agree on diagonal inputs.
    """
    n = 4
    diag = np.array([4.0, 1.0, 0.5, 0.25])
    C = np.diag(diag)
    eig = _make_low_rank_eig_from_dense(C)
    op = SpectralFractionalOperator.from_low_rank_eig(eig, -0.5)

    space = EuclideanSpace(n)
    direct = DiagonalSparseMatrixLinearOperator.from_diagonal_values(
        space, space, diag
    )
    direct_power = direct**-0.5

    rng = np.random.default_rng(5)
    for _ in range(3):
        v = rng.standard_normal(n)
        assert_allclose(op(v), direct_power(v), rtol=1e-10, atol=1e-10)


def test_from_callable():
    """Custom callable f, e.g. exp(-theta * lambda), is handled correctly."""
    n = 5
    diag = np.linspace(0.5, 5.0, n)
    C = np.diag(diag)
    eig = _make_low_rank_eig_from_dense(C)

    op = SpectralFractionalOperator.from_callable(
        eig.u_factor, eig.eigenvalues, lambda x: np.exp(-0.3 * x)
    )
    expected_diag = np.exp(-0.3 * eig.eigenvalues)

    rng = np.random.default_rng(6)
    for _ in range(3):
        v = rng.standard_normal(n)
        coords = eig.u_factor.adjoint(v)
        expected = eig.u_factor(expected_diag * coords)
        assert_allclose(op(v), expected, rtol=1e-10, atol=1e-10)


def test_quadratic_form_squared():
    """quadratic_form_squared matches the direct inner product evaluation."""
    n = 6
    C = _spd_matrix(n, seed=9)
    eig = _make_low_rank_eig_from_dense(C)
    op = SpectralFractionalOperator.from_low_rank_eig(eig, -0.5)

    rng = np.random.default_rng(10)
    for _ in range(5):
        v = rng.standard_normal(n)
        direct = float(np.dot(v, op(v)))
        via_helper = op.quadratic_form_squared(v)
        assert_allclose(via_helper, direct, rtol=1e-10, atol=1e-10)


def test_fractional_operators_from_eig_triple():
    """fractional_operators_from_eig returns the consistent A, A^{-1}, A^{-1/2}."""
    n = 5
    C = _spd_matrix(n, seed=11)
    eig = _make_low_rank_eig_from_dense(C)
    theta = 0.4

    A, A_inv, A_inv_sqrt = fractional_operators_from_eig(eig, theta)

    rng = np.random.default_rng(12)
    v = rng.standard_normal(n)
    # A is C^{-theta}; A_inv is C^{theta}; A_inv_sqrt is C^{theta/2}.
    assert_allclose(A(A_inv(v)), v, rtol=1e-9, atol=1e-9)
    assert_allclose(
        A_inv_sqrt(A_inv_sqrt(v)), A_inv(v), rtol=1e-9, atol=1e-9
    )


def test_eigenvalue_length_mismatch_raises():
    space = EuclideanSpace(3)
    identity_op = space.identity_operator()
    with pytest.raises(ValueError, match="length"):
        SpectralFractionalOperator(
            identity_op, np.array([1.0, 2.0]), lambda x: x
        )


def test_negative_power_requires_positive_eigenvalues():
    """A negative power on zero eigenvalues raises unless regularized."""
    space = EuclideanSpace(3)
    eigvals = np.array([1.0, 0.0, 0.0])
    U = np.eye(3)
    u_op = LinearOperator.from_matrix(space, space, U)
    d_op = DiagonalSparseMatrixLinearOperator.from_diagonal_values(
        space, space, eigvals
    )
    eig = LowRankEig(u_op, d_op)

    with pytest.raises(ValueError, match="positive eigenvalues"):
        SpectralFractionalOperator.from_low_rank_eig(eig, -0.5)
    op = SpectralFractionalOperator.from_low_rank_eig(
        eig, -0.5, regularization=1e-3
    )
    assert isinstance(op, SpectralFractionalOperator)


def test_func_must_be_vectorised():
    """A scalar-only callable raises a helpful error."""
    diag = np.array([1.0, 2.0, 3.0])
    eig = _make_low_rank_eig_from_dense(np.diag(diag))
    with pytest.raises(ValueError, match="vectorised"):
        SpectralFractionalOperator(
            eig.u_factor, eig.eigenvalues, lambda x: 1.0  # scalar return
        )
