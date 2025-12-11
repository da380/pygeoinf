"""
Tests for the EuclideanSpace implementation.
"""

import pytest
import numpy as np

from pygeoinf.hilbert_space import EuclideanSpace


@pytest.fixture(scope="module")
def euclidean_space() -> EuclideanSpace:
    """A module-scoped fixture for the EuclideanSpace instance."""
    return EuclideanSpace(dim=10)


def test_euclidean_axioms(euclidean_space: EuclideanSpace):
    """
    Verifies that the EuclideanSpace instance satisfies all Hilbert space axioms.
    """
    euclidean_space.check(n_checks=10)


def test_euclidean_dual_axioms(euclidean_space: EuclideanSpace):
    """
    Verifies that the DUAL of the EuclideanSpace also satisfies all axioms.
    """
    euclidean_space.dual.check(n_checks=10)


def test_subspace_projection_single_index(euclidean_space: EuclideanSpace):
    """
    Tests subspace_projection with a single index.
    """
    # Project onto the 3rd component (index 2)
    proj = euclidean_space.subspace_projection(2)
    
    # Check domain and codomain
    assert proj.domain == euclidean_space
    assert proj.codomain == EuclideanSpace(1)
    
    # Test forward mapping
    x = np.arange(10.0)
    result = proj(x)
    assert result.shape == (1,)
    assert result[0] == 2.0
    
    # Test adjoint mapping
    y = np.array([5.0])
    adjoint_result = proj.adjoint(y)
    assert adjoint_result.shape == (10,)
    assert adjoint_result[2] == 5.0
    assert np.sum(adjoint_result) == 5.0  # Only one non-zero entry


def test_subspace_projection_multiple_indices(euclidean_space: EuclideanSpace):
    """
    Tests subspace_projection with multiple indices.
    """
    # Project onto 1st, 4th, and 7th components (indices 0, 3, 6)
    indices = [0, 3, 6]
    proj = euclidean_space.subspace_projection(indices)
    
    # Check domain and codomain
    assert proj.domain == euclidean_space
    assert proj.codomain == EuclideanSpace(3)
    
    # Test forward mapping
    x = np.arange(10.0)
    result = proj(x)
    assert result.shape == (3,)
    assert np.array_equal(result, np.array([0.0, 3.0, 6.0]))
    
    # Test adjoint mapping
    y = np.array([1.0, 2.0, 3.0])
    adjoint_result = proj.adjoint(y)
    assert adjoint_result.shape == (10,)
    assert adjoint_result[0] == 1.0
    assert adjoint_result[3] == 2.0
    assert adjoint_result[6] == 3.0
    assert np.sum(adjoint_result) == 6.0


def test_subspace_projection_axioms(euclidean_space: EuclideanSpace):
    """
    Tests that subspace_projection satisfies linear operator axioms.
    """
    indices = [1, 4, 7, 9]
    proj = euclidean_space.subspace_projection(indices)
    
    # Check all linear operator axioms (linearity, adjoint identity, etc.)
    proj.check(n_checks=10)


def test_subspace_projection_out_of_bounds():
    """
    Tests that subspace_projection raises IndexError for out-of-bounds indices.
    """
    space = EuclideanSpace(5)
    
    # Index too large
    with pytest.raises(IndexError, match="out of range"):
        space.subspace_projection(5)
    
    # Index negative
    with pytest.raises(IndexError, match="out of range"):
        space.subspace_projection(-1)
    
    # Multiple indices with one out of bounds
    with pytest.raises(IndexError, match="out of range"):
        space.subspace_projection([0, 2, 5])


def test_subspace_projection_composition():
    """
    Tests composition of subspace projections.
    """
    space = EuclideanSpace(5)
    
    # First project onto indices [0, 2, 4]
    proj1 = space.subspace_projection([0, 2, 4])
    
    # Then project onto index 1 of the resulting 3D space (which is original index 2)
    proj2 = proj1.codomain.subspace_projection(1)
    
    # Compose
    composed = proj2 @ proj1
    
    # This should extract the 3rd component (index 2) of the original space
    x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    result = composed(x)
    
    assert result.shape == (1,)
    assert result[0] == 30.0

