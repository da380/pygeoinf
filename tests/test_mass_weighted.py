"""
Tests for the MassWeightedHilbertSpace implementation.
"""

import pytest
import numpy as np

from pygeoinf.hilbert_space import EuclideanSpace, MassWeightedHilbertSpace
from pygeoinf.linear_operators import LinearOperator


# All fixtures remain the same as they are needed for setup
@pytest.fixture(scope="module")
def dim() -> int:
    return 10


@pytest.fixture(scope="module")
def underlying_space(dim: int) -> EuclideanSpace:
    return EuclideanSpace(dim)


@pytest.fixture(scope="module")
def mass_matrix(dim: int) -> np.ndarray:
    A = np.random.randn(dim, dim)
    return A.T @ A + 1e-6 * np.eye(dim)


@pytest.fixture(scope="module")
def mass_operator(
    underlying_space: EuclideanSpace, mass_matrix: np.ndarray
) -> LinearOperator:
    return LinearOperator.from_matrix(underlying_space, underlying_space, mass_matrix)


@pytest.fixture(scope="module")
def inverse_mass_operator(
    underlying_space: EuclideanSpace, mass_matrix: np.ndarray
) -> LinearOperator:
    inverse_matrix = np.linalg.inv(mass_matrix)
    return LinearOperator.from_matrix(
        underlying_space, underlying_space, inverse_matrix
    )


@pytest.fixture(scope="module")
def mass_weighted_space(
    underlying_space: EuclideanSpace,
    mass_operator: LinearOperator,
    inverse_mass_operator: LinearOperator,
) -> MassWeightedHilbertSpace:
    return MassWeightedHilbertSpace(
        underlying_space, mass_operator, inverse_mass_operator
    )


def test_mass_weighted_axioms(mass_weighted_space: MassWeightedHilbertSpace):
    """
    Verifies that the MassWeightedHilbertSpace instance satisfies all axioms.
    """
    mass_weighted_space.check(n_checks=10)


def test_mass_weighted_dual_axioms(mass_weighted_space: MassWeightedHilbertSpace):
    """
    Verifies that the DUAL of the MassWeightedHilbertSpace also satisfies all axioms.
    """
    mass_weighted_space.dual.check(n_checks=10)


def test_inner_product_definition(
    mass_weighted_space: MassWeightedHilbertSpace,
    underlying_space: EuclideanSpace,
    mass_operator: LinearOperator,
):
    """
    Verifies the core definition of the mass-weighted inner product: (u, v)_Y = (M u, v)_X
    """
    u, v = mass_weighted_space.random(), mass_weighted_space.random()
    inner_product_Y = mass_weighted_space.inner_product(u, v)
    inner_product_X = underlying_space.inner_product(mass_operator(u), v)
    assert np.isclose(inner_product_Y, inner_product_X)
