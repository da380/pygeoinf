"""
Tests for function spaces on a circle. 
"""

import pytest
import numpy as np
from pygeoinf.symmetric_space_new.circle import Lebesgue

from ..checks.hilbert_space import HilbertSpaceChecks

# from ..checks.linear_operator import LinearOperatorChecks


@pytest.fixture
def space() -> Lebesgue:
    """Provides a Lebesgue space instance on a circle for the tests."""
    # Using a relatively low kmax for efficiency in testing
    return Lebesgue(16)


class TestCircleLebesgue(HilbertSpaceChecks):
    """
    Runs the standard suite of Hilbert space checks on the circle.Lebesgue class.
    """

    # This is a new, specific test for the Circle Lebesgue space.
    def test_project_constant_function(self, space: Lebesgue):
        """
        Tests if projecting a constant function f(x) = c results in a
        vector where all elements are c.
        """
        # Define a constant function
        constant_value = 5.0
        constant_function = lambda theta: constant_value

        # Project the function onto the space
        projected_vector = space.project_function(constant_function)

        # Create the expected result: a numpy array of the same shape
        # filled with the constant value.
        expected_vector = np.full_like(projected_vector, constant_value)

        # Check that the projected vector is numerically close to the expected one.
        assert np.allclose(projected_vector, expected_vector)
