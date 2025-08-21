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
from .checks.hilbert_space import HilbertSpaceChecks


# =========================================================================
# Fixtures for creating the test environment
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
    This is done by generating a random matrix A and computing M = A.T @ A,
    which guarantees the desired properties. A small identity matrix is
    added to ensure it is well-conditioned.
    """
    A = np.random.randn(dim, dim)
    M = A.T @ A + 1e-6 * np.eye(dim)
    return M


@pytest.fixture(scope="module")
def inverse_mass_matrix(mass_matrix: np.ndarray) -> np.ndarray:
    """Provides the inverse of the mass matrix."""
    return np.linalg.inv(mass_matrix)


@pytest.fixture(scope="module")
def mass_weighted_space(
    underlying_space: EuclideanSpace,
    mass_matrix: np.ndarray,
    inverse_mass_matrix: np.ndarray,
) -> MassWeightedHilbertSpace:
    """
    Creates the MassWeightedHilbertSpace instance for testing.
    This is module-scoped to avoid repeated instantiation.
    """
    # Using the refined __init__ that takes both M and M_inv
    return MassWeightedHilbertSpace(underlying_space, mass_matrix, inverse_mass_matrix)


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
    mass_matrix: np.ndarray,
):
    """
    Verifies the core mathematical definition of the mass-weighted inner product:
    (u, v)_Y = (M u, v)_X
    """
    # Get two random vectors from the space
    u = mass_weighted_space.random()
    v = mass_weighted_space.random()

    # 1. Calculate the inner product directly in the mass-weighted space (LHS)
    inner_product_Y = mass_weighted_space.inner_product(u, v)

    # 2. Calculate the inner product using the underlying space definition (RHS)
    # First, get the component representation of the vectors
    u_components = underlying_space.to_components(u)

    # Apply the mass matrix to the components of u
    Mu_components = mass_matrix @ u_components

    # Convert back to a vector in the underlying space
    Mu_vector = underlying_space.from_components(Mu_components)

    # Now, take the inner product in the underlying space X
    inner_product_X = underlying_space.inner_product(Mu_vector, v)

    # The two values should be close
    assert np.isclose(inner_product_Y, inner_product_X)
