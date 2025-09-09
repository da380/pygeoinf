"""
Tests for the EuclideanSpace implementation.
"""

import pytest

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
