"""
Tests for the various static factory and construction methods on the LinearOperator class.
"""

import pytest
import numpy as np
import pygeoinf as inf
from pygeoinf.symmetric_space.circle import Sobolev as CircleSobolev
from pygeoinf.symmetric_space.sphere import Sobolev as SphereSobolev


# For reproducibility in all tests
@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(42)


def test_from_matrix_standard():
    """Tests creating an operator from a matrix with standard mapping."""
    domain, codomain = inf.EuclideanSpace(3), inf.EuclideanSpace(2)
    M = np.random.randn(2, 3)
    op = inf.LinearOperator.from_matrix(domain, codomain, M, galerkin=False)
    op2 = inf.LinearOperator.from_matrix(domain, codomain, np.random.randn(2, 3), galerkin=False)

    # Specific check: Ensure the matrix representation is correct
    assert np.allclose(op.matrix(dense=True), M)
    # Axiom check: Ensure the resulting operator is mathematically valid
    op.check(n_checks=3, op2=op2)


def test_from_matrix_galerkin():
    """Tests creating an operator from a matrix with Galerkin mapping."""
    domain, codomain = inf.EuclideanSpace(2), inf.EuclideanSpace(2)
    M = np.random.randn(2, 2)
    op = inf.LinearOperator.from_matrix(domain, codomain, M, galerkin=True)
    op2 = inf.LinearOperator.from_matrix(domain, codomain, np.random.randn(2, 2), galerkin=True)

    # Specific check: Ensure the Galerkin matrix representation is correct
    assert np.allclose(op.matrix(dense=True, galerkin=True), M)
    # Axiom check: Ensure the resulting operator is mathematically valid
    op.check(n_checks=3, op2=op2)


def test_self_adjoint_from_matrix():
    """Tests creating a self-adjoint operator from a symmetric matrix."""
    space = inf.EuclideanSpace(3)
    M = np.random.randn(3, 3)
    M_symm = M + M.T
    op = inf.LinearOperator.self_adjoint_from_matrix(space, M_symm)
    M2 = np.random.randn(3, 3)
    M2_symm = M2 + M2.T
    op2 = inf.LinearOperator.self_adjoint_from_matrix(space, M2_symm)

    # Specific check: Ensure the operator is self-adjoint
    x, y = space.random(), space.random()
    assert np.isclose(space.inner_product(op(x), y), space.inner_product(x, op(y)))
    # Axiom check: Ensure the resulting operator is mathematically valid
    op.check(n_checks=3, op2=op2)


def test_from_linear_forms():
    """Tests creating an operator from a list of LinearForms."""
    domain = inf.EuclideanSpace(3)
    form1 = inf.LinearForm(domain, components=np.array([1, 2, 3]))
    form2 = inf.LinearForm(domain, components=np.array([4, 5, 6]))
    op = inf.LinearOperator.from_linear_forms([form1, form2])
    form3 = inf.LinearForm(domain, components=np.random.randn(3))
    form4 = inf.LinearForm(domain, components=np.random.randn(3))
    op2 = inf.LinearOperator.from_linear_forms([form3, form4])

    # Specific check: Ensure the matrix is constructed correctly
    expected_matrix = np.array([[1, 2, 3], [4, 5, 6]])
    assert np.allclose(op.matrix(dense=True), expected_matrix)
    # Axiom check: Ensure the resulting operator is mathematically valid
    op.check(n_checks=3, op2=op2)


def test_from_tensor_product():
    """Tests creating an operator from a sum of tensor products."""
    domain, codomain = inf.EuclideanSpace(2), inf.EuclideanSpace(3)
    u1, v1 = codomain.random(), domain.random()
    u2, v2 = codomain.random(), domain.random()
    op = inf.LinearOperator.from_tensor_product(domain, codomain, [(u1, v1), (u2, v2)])
    u3, v3 = codomain.random(), domain.random()
    u4, v4 = codomain.random(), domain.random()
    op2 = inf.LinearOperator.from_tensor_product(domain, codomain, [(u3, v3), (u4, v4)])

    # Specific check: Ensure the operator action is correct
    x = domain.random()
    y_op = op(x)
    y_manual = domain.inner_product(x, v1) * u1 + domain.inner_product(x, v2) * u2
    assert np.allclose(y_op, y_manual)
    # Axiom check: Ensure the resulting operator is mathematically valid
    op.check(n_checks=3, op2=op2)


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

    # Axiom check: The most important check is that the adjoint is now correct
    A_weighted.check(n_checks=3, op2=A_weighted2)


def test_from_formal_adjoint_symmetric_direct_sum():
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

    A_full = inf.LinearOperator.from_formal_adjoint(domain_full, domain_full, A_base)

    op3_base = circle_leb.invariant_automorphism(lambda eig: 1.0 / (1.0 + 0.5 * eig))
    op4_base = sphere_leb.invariant_automorphism(lambda eig: 1.0 / (1.0 + 0.2 * eig))
    A_base2 = inf.BlockDiagonalLinearOperator([op3_base, op4_base])
    A_full2 = inf.LinearOperator.from_formal_adjoint(domain_full, domain_full, A_base2)

    # Axiom check: Verify the resulting operator is mathematically sound
    A_full.check(n_checks=3, op2=A_full2)
