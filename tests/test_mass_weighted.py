"""
Tests for the MassWeightedHilbertSpace implementation.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from pygeoinf.hilbert_space import (
    HilbertSpace,
    EuclideanSpace,
    MassWeightedHilbertSpace,
)
from pygeoinf.operators import LinearOperator
from .checks.hilbert_space import HilbertSpaceChecks


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture(scope="module")
def dim() -> int:
    """Provides the dimension for the test spaces."""
    return 10


@pytest.fixture(scope="module")
def underlying_space(dim: int) -> EuclideanSpace:
    """Provides the base space (a simple Euclidean space)."""
    return EuclideanSpace(dim)


@pytest.fixture(scope="module")
def mass_matrix(dim: int) -> np.ndarray:
    """
    Creates a random, symmetric, positive-definite mass matrix.
    """
    A = np.random.randn(dim, dim)
    M = A.T @ A + 1e-6 * np.eye(dim)
    return M


@pytest.fixture(scope="module")
def mass_operator(
    underlying_space: EuclideanSpace, mass_matrix: np.ndarray
) -> LinearOperator:
    """Creates a LinearOperator from the mass matrix."""
    return LinearOperator.from_matrix(underlying_space, underlying_space, mass_matrix)


@pytest.fixture(scope="module")
def inverse_mass_operator(
    underlying_space: EuclideanSpace, mass_matrix: np.ndarray
) -> LinearOperator:
    """Creates a LinearOperator from the inverse of the mass matrix."""
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
    """Creates the MassWeightedHilbertSpace instance for testing."""
    return MassWeightedHilbertSpace(
        underlying_space, mass_operator, inverse_mass_operator
    )


# =========================================================================
# Test Classes using the HilbertSpaceChecks contract
# =========================================================================


class TestMassWeightedHilbertSpace(HilbertSpaceChecks):
    """
    Runs the standard suite of Hilbert space checks on the primary
    MassWeightedHilbertSpace class.
    """

    @pytest.fixture
    def space(self, mass_weighted_space: MassWeightedHilbertSpace) -> HilbertSpace:
        """Provides the primary space to the test suite."""
        return mass_weighted_space


class TestMassWeightedHilbertSpaceDual(HilbertSpaceChecks):
    """
    Runs the standard suite of Hilbert space checks on the DUAL of the
    MassWeightedHilbertSpace class.
    """

    @pytest.fixture
    def space(self, mass_weighted_space: MassWeightedHilbertSpace) -> HilbertSpace:
        """Provides the DUAL space to the test suite."""
        return mass_weighted_space.dual


# =========================================================================
# Specific Tests for MassWeightedHilbertSpace
# =========================================================================


def test_inner_product_definition(
    mass_weighted_space: MassWeightedHilbertSpace,
    underlying_space: EuclideanSpace,
    mass_operator: LinearOperator,
):
    """
    Verifies the core mathematical definition of the mass-weighted inner product:
    (u, v)_Y = (M u, v)_X
    """
    u = mass_weighted_space.random()
    v = mass_weighted_space.random()

    # 1. Calculate the inner product directly in the mass-weighted space (LHS)
    inner_product_Y = mass_weighted_space.inner_product(u, v)

    # 2. Calculate the inner product using the underlying space definition (RHS)
    # Apply the mass operator to vector u
    Mu_vector = mass_operator(u)

    # Now, take the inner product in the underlying space X
    inner_product_X = underlying_space.inner_product(Mu_vector, v)

    # The two values should be close
    assert np.isclose(inner_product_Y, inner_product_X)
