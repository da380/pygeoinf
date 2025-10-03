"""
Tests for the NormalSumOperator class, refactored with pytest fixtures
and parameterized dimensions.
"""

import pytest
import numpy as np
import pygeoinf as inf


# For reproducibility in all tests
@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(42)


@pytest.fixture(
    params=[
        (3, 2),  # Original small case
        (5, 5),  # Square case
        (50, 20),  # A larger, non-square case
    ],
    ids=["dims(3,2)", "dims(5,5)", "dims(50,20)"],
)
def normal_sum_components(request):
    """
    Provides a standard set of correctly-dimensioned operators for tests.
    This fixture is parameterized to run with multiple dimension sets.
    """
    domain_dim, codomain_dim = request.param
    domain_A, codomain_A = inf.EuclideanSpace(domain_dim), inf.EuclideanSpace(
        codomain_dim
    )

    A = inf.LinearOperator.from_matrix(
        domain_A, codomain_A, np.random.randn(codomain_dim, domain_dim)
    )

    # Q must be an operator on the DOMAIN of A.
    Q_mat = np.random.randn(domain_dim, domain_dim)
    Q = inf.LinearOperator.self_adjoint_from_matrix(domain_A, Q_mat + Q_mat.T)

    # B must be an operator on the CODOMAIN of A.
    B_mat = np.random.randn(codomain_dim, codomain_dim)
    B = inf.LinearOperator.self_adjoint_from_matrix(codomain_A, B_mat + B_mat.T)

    # The domain of the final NormalSumOperator is the codomain of A.
    return {"A": A, "Q": Q, "B": B, "op_domain": codomain_A}


# --- Parameterized Test for Standard Euclidean Cases ---
@pytest.mark.parametrize(
    "op_args_keys, manual_check_func",
    [
        (
            ("A", "Q", "B"),  # Full form: N = AQA* + B
            lambda A, Q, B, x: A(Q(A.adjoint(x))) + B(x),
        ),
        (
            ("A", "B"),  # Default Q: N = AA* + B
            lambda A, Q, B, x: A(A.adjoint(x)) + B(x),
        ),
        (("A", "Q"), lambda A, Q, B, x: A(Q(A.adjoint(x)))),  # Default B: N = AQA*
        (("A",), lambda A, Q, B, x: A(A.adjoint(x))),  # Both defaults: N = AA*
    ],
    ids=["full_form", "default_q", "default_b", "both_defaults"],
)
def test_normalsum_euclidean_cases(
    normal_sum_components, op_args_keys, manual_check_func
):
    """
    Tests all standard variations of the NormalSumOperator in Euclidean space,
    including a direct check of the computed dense matrix.
    """
    op_args = {key: normal_sum_components[key] for key in op_args_keys}
    op = inf.NormalSumOperator(**op_args)
    op_domain = normal_sum_components["op_domain"]

    # --- Specific Action Check ---
    x = op_domain.random()
    A, Q, B = (normal_sum_components[k] for k in ("A", "Q", "B"))

    y_op = op(x)
    y_manual = manual_check_func(A, Q, B, x)
    assert np.allclose(y_op, y_manual)

    # --- Dense Matrix Check ---
    # Manually build the expected Galerkin matrix
    dim = op_domain.dim
    basis = [op_domain.basis_vector(i) for i in range(dim)]
    expected_matrix = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            # M_ij = <e_i, op(e_j)>
            op_e_j = op(basis[j])
            expected_matrix[i, j] = op_domain.inner_product(basis[i], op_e_j)

    # Get the matrix from our optimized implementation
    computed_matrix = op.matrix(dense=True, galerkin=True, parallel=False)
    assert np.allclose(computed_matrix, expected_matrix)

    # --- Axiom Check ---
    op.check(n_checks=3)


def test_normalsum_fails_check_with_nonsymmetric_q(normal_sum_components):
    """
    Tests that .check() fails if a non-self-adjoint Q is provided.
    """
    A = normal_sum_components["A"]
    domain_A = A.domain
    dim = domain_A.dim

    # Create a Q on the domain of A that is explicitly NOT self-adjoint
    Q_mat = np.random.randn(dim, dim)
    Q = inf.LinearOperator.from_matrix(domain_A, domain_A, Q_mat)

    op = inf.NormalSumOperator(A, Q=Q)

    # The check should fail because the resulting operator is not self-adjoint.
    with pytest.raises(AssertionError, match="Adjoint definition failed"):
        op.check(n_checks=1)
