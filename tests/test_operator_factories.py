"""
Tests for the various static factory and construction methods on the LinearOperator class.
"""

import pytest

import numpy as np
import pygeoinf as inf


def test_from_matrix_standard():
    """Tests creating an operator from a matrix with standard mapping."""
    domain, codomain = inf.EuclideanSpace(3), inf.EuclideanSpace(2)
    M = np.random.randn(2, 3)

    op = inf.LinearOperator.from_matrix(domain, codomain, M, galerkin=False)

    x = domain.random()
    y_op = op(x)
    y_np = M @ x

    assert np.allclose(y_op, y_np)
    assert np.allclose(op.matrix(dense=True), M)


def test_from_matrix_galerkin():
    """Tests creating an operator from a matrix with Galerkin mapping."""
    domain, codomain = inf.EuclideanSpace(2), inf.EuclideanSpace(2)
    M = np.random.randn(2, 2)

    op = inf.LinearOperator.from_matrix(domain, codomain, M, galerkin=True)

    x = domain.random()
    y_op = op(x)
    y_expected = M @ domain.to_components(x)

    assert np.allclose(domain.to_components(y_op), y_expected)
    assert np.allclose(op.matrix(dense=True, galerkin=True), M)


def test_self_adjoint_from_matrix():
    """Tests creating a self-adjoint operator from a symmetric matrix."""
    space = inf.EuclideanSpace(3)
    M = np.random.randn(3, 3)
    M_symm = M + M.T

    op = inf.LinearOperator.self_adjoint_from_matrix(space, M_symm)

    x, y = space.random(), space.random()
    lhs = space.inner_product(op(x), y)
    rhs = space.inner_product(x, op(y))

    assert np.isclose(lhs, rhs)
    assert np.allclose(op.matrix(dense=True, galerkin=True), M_symm)


def test_from_linear_forms():
    """Tests creating an operator from a list of LinearForms."""
    domain = inf.EuclideanSpace(3)

    form1 = inf.LinearForm(domain, components=np.array([1, 2, 3]))
    form2 = inf.LinearForm(domain, components=np.array([4, 5, 6]))

    op = inf.LinearOperator.from_linear_forms([form1, form2])

    assert isinstance(op.codomain, inf.EuclideanSpace) and op.codomain.dim == 2

    expected_matrix = np.array([[1, 2, 3], [4, 5, 6]])
    assert np.allclose(op.matrix(dense=True), expected_matrix)


def test_from_tensor_product():
    """Tests creating an operator from a sum of tensor products."""
    domain, codomain = inf.EuclideanSpace(2), inf.EuclideanSpace(3)

    u1, v1 = codomain.random(), domain.random()
    u2, v2 = codomain.random(), domain.random()

    op = inf.LinearOperator.from_tensor_product(domain, codomain, [(u1, v1), (u2, v2)])

    x = domain.random()
    y_op = op(x)
    y_manual = domain.inner_product(x, v1) * u1 + domain.inner_product(x, v2) * u2

    assert np.allclose(y_op, y_manual)


def test_self_adjoint_from_tensor_product():
    """Tests creating a self-adjoint operator from a tensor product sum."""
    space = inf.EuclideanSpace(3)

    v1, v2 = space.random(), space.random()

    op = inf.LinearOperator.self_adjoint_from_tensor_product(space, [v1, v2])

    x = space.random()
    y_op = op(x)
    y_manual = space.inner_product(x, v1) * v1 + space.inner_product(x, v2) * v2

    assert np.allclose(y_op, y_manual)

    y = space.random()
    lhs = space.inner_product(op(x), y)
    rhs = space.inner_product(x, op(y))
    assert np.isclose(lhs, rhs)


def test_from_formal_adjoint_simple():
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

    M = np.random.randn(3, 3)
    A_base = inf.LinearOperator.from_matrix(base_space, base_space, M)

    A_weighted = inf.LinearOperator.from_formal_adjoint(
        weighted_space, weighted_space, A_base
    )

    x, y = weighted_space.random(), weighted_space.random()
    lhs = weighted_space.inner_product(A_weighted(x), y)
    rhs = weighted_space.inner_product(x, A_weighted.adjoint(y))

    assert np.isclose(lhs, rhs)


def test_from_formal_adjoint_with_direct_sum():
    """Tests from_formal_adjoint with a direct sum of MassWeightedHilbertSpaces."""
    domain_base = inf.EuclideanSpace(2)
    codomain_base1 = inf.EuclideanSpace(3)
    codomain_base2 = inf.EuclideanSpace(2)
    codomain_base = inf.HilbertSpaceDirectSum([codomain_base1, codomain_base2])

    M1, M2 = np.random.randn(3, 2), np.random.randn(2, 2)
    op1_base = inf.LinearOperator.from_matrix(domain_base, codomain_base1, M1)
    op2_base = inf.LinearOperator.from_matrix(domain_base, codomain_base2, M2)
    A_base = inf.ColumnLinearOperator([op1_base, op2_base])

    domain_full = domain_base
    mass_mat1 = np.array([[2, 0.1, 0], [0.1, 1, 0.2], [0, 0.2, 2]])
    mass_op1 = inf.LinearOperator.from_matrix(
        codomain_base1, codomain_base1, mass_mat1, galerkin=True
    )
    inv_mass_op1 = inf.LinearOperator.from_matrix(
        codomain_base1, codomain_base1, np.linalg.inv(mass_mat1), galerkin=True
    )
    codomain_full1 = inf.MassWeightedHilbertSpace(
        codomain_base1, mass_op1, inv_mass_op1
    )
    codomain_full2 = codomain_base2
    codomain_full = inf.HilbertSpaceDirectSum([codomain_full1, codomain_full2])

    A_full = inf.LinearOperator.from_formal_adjoint(domain_full, codomain_full, A_base)

    x, y = domain_full.random(), codomain_full.random()
    lhs = codomain_full.inner_product(A_full(x), y)
    rhs = domain_full.inner_product(x, A_full.adjoint(y))

    assert np.isclose(lhs, rhs)
