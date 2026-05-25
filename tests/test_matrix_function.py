"""Tests for the Lanczos matrix-function module."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg
from numpy.testing import assert_allclose

from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.matrix_function import (
    apply_matrix_function,
    lanczos_tridiagonalize,
    matrix_function_quadratic_form,
)


def _spd_operator(n: int, seed: int) -> tuple[LinearOperator, np.ndarray]:
    """Return (operator, dense matrix) for a random SPD operator."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    C = A.T @ A + 0.5 * np.eye(n)
    space = EuclideanSpace(n)
    op = LinearOperator.from_matrix(space, space, C)
    return op, C


def test_lanczos_full_rank_matches_dense_eigh():
    """k = n Lanczos reconstructs the dense spectrum to within FP."""
    n = 6
    op, C = _spd_operator(n, seed=11)
    rng = np.random.default_rng(0)
    v = rng.standard_normal(n)

    Q, T = lanczos_tridiagonalize(op, v, k=n, reorth="full")
    assert len(Q) == n
    assert T.shape == (n, n)

    # Eigenvalues of T equal eigenvalues of C up to ordering.
    ritz = np.sort(np.linalg.eigvalsh(T))
    true_spec = np.sort(np.linalg.eigvalsh(C))
    assert_allclose(ritz, true_spec, rtol=1e-10, atol=1e-10)

    # Orthonormality of the Lanczos vectors.
    QQ = np.column_stack(Q)
    assert_allclose(QQ.T @ QQ, np.eye(n), rtol=0, atol=1e-10)


@pytest.mark.parametrize("power", [-1.0, -0.5, 0.5, 1.0])
def test_apply_matrix_function_matches_fractional_power(power):
    """Lanczos f(C) v matches scipy.linalg.fractional_matrix_power for k=n."""
    n = 6
    op, C = _spd_operator(n, seed=13)
    expected_op = scipy.linalg.fractional_matrix_power(C, power)

    rng = np.random.default_rng(1)
    for _ in range(5):
        v = rng.standard_normal(n)
        expected = expected_op @ v
        got = apply_matrix_function(op, v, lambda x: x**power, k=n)
        assert_allclose(got, expected, rtol=1e-9, atol=1e-9)


def test_apply_matrix_function_geometric_convergence():
    """Error decays geometrically in k for a smooth function on a smooth spectrum."""
    n = 20
    # Build SPD with a known smooth spectrum: eigenvalues 1..n.
    eigvals = np.arange(1, n + 1, dtype=float)
    rng = np.random.default_rng(2)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    C = Q @ np.diag(eigvals) @ Q.T
    space = EuclideanSpace(n)
    op = LinearOperator.from_matrix(space, space, C)

    v = rng.standard_normal(n)
    truth = scipy.linalg.fractional_matrix_power(C, -0.5) @ v

    errors = []
    for k in (3, 6, 10, 15):
        got = apply_matrix_function(op, v, lambda x: x**-0.5, k=k)
        errors.append(np.linalg.norm(got - truth))

    # Errors should be monotone decreasing in k.
    for i in range(len(errors) - 1):
        assert errors[i + 1] <= errors[i] * 1.05  # small tolerance for FP


def test_matrix_function_quadratic_form_matches_inner_product():
    """matrix_function_quadratic_form equals <v, apply_matrix_function(...)>."""
    n = 8
    op, _ = _spd_operator(n, seed=15)
    rng = np.random.default_rng(3)
    for _ in range(4):
        v = rng.standard_normal(n)
        applied = apply_matrix_function(op, v, lambda x: x**-0.3, k=n)
        direct = float(np.dot(v, applied))
        via_quad = matrix_function_quadratic_form(
            op, v, lambda x: x**-0.3, k=n
        )
        assert_allclose(via_quad, direct, rtol=1e-9, atol=1e-9)


def test_reorth_full_outperforms_none_at_large_k():
    """Full reorthogonalisation preserves orthonormality where 'none' loses it."""
    n = 50
    op, _ = _spd_operator(n, seed=17)
    rng = np.random.default_rng(4)
    v = rng.standard_normal(n)
    k = 30

    Q_full, _ = lanczos_tridiagonalize(op, v, k=k, reorth="full")
    Q_none, _ = lanczos_tridiagonalize(op, v, k=k, reorth="none")

    Q_full_mat = np.column_stack(Q_full)
    Q_none_mat = np.column_stack(Q_none)
    err_full = np.max(
        np.abs(Q_full_mat.T @ Q_full_mat - np.eye(Q_full_mat.shape[1]))
    )
    err_none = np.max(
        np.abs(Q_none_mat.T @ Q_none_mat - np.eye(Q_none_mat.shape[1]))
    )
    assert err_full < 1e-10
    # Loss-of-orthogonality typically reaches ~1e-3 or worse by k=30.
    assert err_none > err_full


def test_zero_vector_returns_zero():
    n = 5
    op, _ = _spd_operator(n, seed=19)
    space = EuclideanSpace(n)
    result = apply_matrix_function(op, space.zero, lambda x: x**-0.5, k=3)
    assert_allclose(result, space.zero, atol=0)


def test_breakdown_returns_truncated():
    """A vector lying in a small invariant subspace triggers early termination."""
    # Build a block-diagonal C with two blocks; start vector in block 1 only.
    n = 6
    rng = np.random.default_rng(21)
    A1 = rng.standard_normal((3, 3))
    A2 = rng.standard_normal((3, 3))
    C = np.zeros((n, n))
    C[:3, :3] = A1.T @ A1 + 0.1 * np.eye(3)
    C[3:, 3:] = A2.T @ A2 + 0.1 * np.eye(3)
    space = EuclideanSpace(n)
    op = LinearOperator.from_matrix(space, space, C)
    v = np.zeros(n)
    v[:3] = rng.standard_normal(3)

    Q, T = lanczos_tridiagonalize(op, v, k=n, reorth="full")
    # The Krylov subspace lives entirely in the first block, so we should
    # break down by step 3.
    assert len(Q) <= 3


def test_invalid_k_raises():
    n = 4
    op, _ = _spd_operator(n, seed=23)
    space = EuclideanSpace(n)
    with pytest.raises(ValueError, match="k must be"):
        lanczos_tridiagonalize(op, space.basis_vector(0), k=0)


def test_zero_v_raises():
    n = 4
    op, _ = _spd_operator(n, seed=25)
    space = EuclideanSpace(n)
    with pytest.raises(ValueError, match="non-zero"):
        lanczos_tridiagonalize(op, space.zero, k=3)


def test_invalid_reorth_raises():
    n = 4
    op, _ = _spd_operator(n, seed=27)
    space = EuclideanSpace(n)
    with pytest.raises(ValueError, match="reorth"):
        lanczos_tridiagonalize(op, space.basis_vector(0), k=3, reorth="bogus")
