"""
Concrete tests for the AffineOperator implementation.
"""

import pytest
import numpy as np

from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.linear_operators import LinearOperator

# Assuming you placed AffineOperator in affine_operators.py
from pygeoinf.affine_operators import AffineOperator


@pytest.fixture
def affine_operator() -> AffineOperator:
    """
    Provides a simple AffineOperator mapping from R^3 to R^2.
    F(x) = A(x) + b
    """
    domain = EuclideanSpace(3)
    codomain = EuclideanSpace(2)

    # Linear part A: simple 2x3 matrix
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    linear_part = LinearOperator.from_matrix(domain, codomain, matrix)

    # Translation part b: vector in R^2
    translation = np.array([10.0, 20.0])

    return AffineOperator(linear_part, translation)


def test_affine_operator_axioms(affine_operator: AffineOperator):
    """
    Verifies that the operator satisfies all non-linear and affine-specific
    axioms by calling its internal self-check method.
    """
    # This will run the finite difference checks AND the new affine checks
    affine_operator.check(n_checks=5)


def test_affine_evaluation(affine_operator: AffineOperator):
    """
    Explicitly tests that F(x) = A(x) + b computes the expected result.
    """
    # x = [1, 0, -1]
    x = np.array([1.0, 0.0, -1.0])

    # A(x) = [1(1)+3(-1), 4(1)+6(-1)] = [-2, -2]
    # F(x) = [-2, -2] + [10, 20] = [8, 18]
    expected_result = np.array([8.0, 18.0])

    actual_result = affine_operator(x)

    assert np.allclose(actual_result, expected_result)


def test_affine_properties_access(affine_operator: AffineOperator):
    """
    Verifies that the linear part and translation part are correctly accessible.
    """
    # Extract translation
    b = affine_operator.translation_part
    assert np.allclose(b, np.array([10.0, 20.0]))

    # Extract linear part and test its evaluation
    A = affine_operator.linear_part
    x = np.array([1.0, 0.0, -1.0])
    assert np.allclose(A(x), np.array([-2.0, -2.0]))


def test_translation_domain_validation():
    """
    Ensures that an AffineOperator cannot be constructed if the translation
    vector doesn't belong to the codomain.
    """
    domain = EuclideanSpace(3)
    codomain = EuclideanSpace(2)

    linear_part = LinearOperator.from_matrix(domain, codomain, np.ones((2, 3)))

    # Invalid translation: a vector in R^3 instead of R^2
    invalid_translation = np.array([1.0, 2.0, 3.0])

    with pytest.raises(TypeError, match="element of the linear operator's codomain"):
        AffineOperator(linear_part, invalid_translation)


def test_affine_algebraic_operations():
    """
    Verifies that algebraic operations (+, -, *, @) between AffineOperators
    and LinearOperators correctly return new AffineOperators with the proper
    internal structure.
    """
    domain = EuclideanSpace(3)
    codomain = EuclideanSpace(3)

    # Base operators for testing
    L1 = LinearOperator.from_matrix(domain, codomain, np.eye(3) * 2.0)
    L2 = LinearOperator.from_matrix(domain, codomain, np.eye(3) * 3.0)

    b1 = np.array([1.0, 2.0, 3.0])
    b2 = np.array([-1.0, 0.0, 1.0])

    F1 = AffineOperator(L1, b1)
    F2 = AffineOperator(L2, b2)

    # 1. Addition (Affine + Affine)
    # (2I x + b1) + (3I x + b2) = 5I x + (b1 + b2)
    F_add = F1 + F2
    assert isinstance(F_add, AffineOperator)
    assert np.allclose(F_add.linear_part.matrix(dense=True), np.eye(3) * 5.0)
    assert np.allclose(F_add.translation_part, np.array([0.0, 2.0, 4.0]))

    # 2. Addition (Affine + Linear / Linear + Affine)
    # (2I x + b1) + 3I x = 5I x + b1
    F_add_lin1 = F1 + L2
    F_add_lin2 = L2 + F1
    assert isinstance(F_add_lin1, AffineOperator)
    assert isinstance(F_add_lin2, AffineOperator)
    assert np.allclose(F_add_lin1.translation_part, b1)
    assert np.allclose(F_add_lin2.translation_part, b1)

    # 3. Scalar Multiplication
    # 3 * (2I x + b1) = 6I x + 3*b1
    F_mul = 3.0 * F1
    assert isinstance(F_mul, AffineOperator)
    assert np.allclose(F_mul.linear_part.matrix(dense=True), np.eye(3) * 6.0)
    assert np.allclose(F_mul.translation_part, np.array([3.0, 6.0, 9.0]))

    # 4. Composition (Affine @ Affine)
    # F1(F2(x)) = L1(L2 x + b2) + b1 = (L1@L2)x + (L1(b2) + b1)
    F_comp = F1 @ F2
    assert isinstance(F_comp, AffineOperator)
    assert np.allclose(F_comp.linear_part.matrix(dense=True), np.eye(3) * 6.0)
    expected_b_comp = L1(b2) + b1
    assert np.allclose(F_comp.translation_part, expected_b_comp)

    # 5. Composition (Linear @ Affine)
    # L2(F1(x)) = L2(L1 x + b1) = (L2@L1)x + L2(b1)
    F_comp_lin = L2 @ F1
    assert isinstance(F_comp_lin, AffineOperator)
    assert np.allclose(F_comp_lin.linear_part.matrix(dense=True), np.eye(3) * 6.0)
    expected_b_comp_lin = L2(b1)
    assert np.allclose(F_comp_lin.translation_part, expected_b_comp_lin)

    # Run the rigorous internal checks on one of the composites to be safe
    F_comp.check(n_checks=2)
