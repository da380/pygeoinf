"""
Tests for the EuclideanSpace implementation.

This file demonstrates how to use the abstract HilbertSpaceChecks class
to verify the correctness of a concrete Hilbert space implementation and its
dual.
"""

import pytest
from pygeoinf.hilbert_space import EuclideanSpace, HilbertSpace
from .checks.hilbert_space import HilbertSpaceChecks


@pytest.fixture(scope="module")
def euclidean_space() -> EuclideanSpace:
    """
    A module-scoped fixture for the EuclideanSpace instance.
    This is created only once and shared between the test classes.
    """
    return EuclideanSpace(dim=10)


class TestEuclideanSpace(HilbertSpaceChecks):
    """
    Runs the standard suite of Hilbert space checks on the primary
    EuclideanSpace class.
    """

    @pytest.fixture
    def space(self, euclidean_space: EuclideanSpace) -> HilbertSpace:
        """Provides the primary space to the test suite."""
        return euclidean_space


class TestEuclideanSpaceDual(HilbertSpaceChecks):
    """
    Runs the standard suite of Hilbert space checks on the DUAL of the
    EuclideanSpace class.
    """

    @pytest.fixture
    def space(self, euclidean_space: EuclideanSpace) -> HilbertSpace:
        """Provides the DUAL space to the test suite."""
        return euclidean_space.dual
