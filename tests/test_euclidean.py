"""
Tests for the EuclideanSpace implementation.

This file demonstrates how to use the abstract HilbertSpaceChecks class
to verify the correctness of a concrete Hilbert space implementation.
"""

import pytest
from pygeoinf.hilbert_space import EuclideanSpace
from .checks.hilbert_space import HilbertSpaceChecks


# Define a pytest fixture that provides an instance of the concrete space.
# The fixture must be named `space` to match what the check class expects.
@pytest.fixture
def space() -> EuclideanSpace:
    """Provides a 10-dimensional EuclideanSpace instance for the tests."""
    return EuclideanSpace(dim=10)


# Create a test class that inherits from the abstract check class.
# Pytest will automatically discover and run all the test methods from
# HilbertSpaceChecks on the EuclideanSpace instance provided by the fixture.
class TestEuclideanSpace(HilbertSpaceChecks):
    """
    Runs the standard suite of Hilbert space checks on the EuclideanSpace class.
    """

    pass
