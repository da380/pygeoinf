"""
Tests for the Sobolev space implementation on a circle.
"""

import pytest
import numpy as np
from pygeoinf.symmetric_space.circle import Sobolev
from ..checks.hilbert_space import HilbertSpaceChecks
from ..checks.linear_operator import LinearOperatorChecks


@pytest.fixture
def space() -> Sobolev:
    """Provides a Sobolev space instance on a circle for the tests."""
    # Using a relatively low kmax for efficiency in testing
    return Sobolev(16, 1.0, 0.1)


class TestCircleSobolev(HilbertSpaceChecks):
    """
    Runs the standard suite of Hilbert space checks on the circle.Sobolev class.
    """

    # This is a new, specific test for the Circle Sobolev space.
    def test_project_constant_function(self, space: Sobolev):
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


# --- Tests for the Point Evaluation Operator ---
class TestPointEvaluationOperator(LinearOperatorChecks):
    """
    Runs the standard suite of linear operator checks on the
    point_evaluation_operator from the circle.Sobolev class.
    """

    @pytest.fixture
    def operator(self):
        space = Sobolev(16, 1.0, 0.1)
        observation_points = np.linspace(0, 2 * np.pi, 10)
        return space.point_evaluation_operator(observation_points)


# --- Tests for the Invariant Automorphism Operator ---
class TestInvariantAutomorphism(LinearOperatorChecks):
    """
    Runs the standard suite of linear operator checks on the
    invariant_automorphism operator from the circle.Sobolev class.
    """

    @pytest.fixture
    def operator(self):
        space = Sobolev(16, 1.0, 0.1)
        # Test with a simple smoothing function f(k^2) = 1 / (1 + k^2)
        return space.invariant_automorphism(lambda k_squared: 1.0 / (1.0 + k_squared))
