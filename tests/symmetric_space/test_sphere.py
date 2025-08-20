"""
Tests for the Sobolev space implementation on a sphere.
"""

import pytest
import numpy as np
from pygeoinf.symmetric_space.sphere import Sobolev
from ..checks.hilbert_space import HilbertSpaceChecks


# Define different sets of parameters to test the Sobolev class with.
# Each tuple contains (positional_args, keyword_args).
sobolev_parameter_sets = [
    # Default-like parameters
    ((8, 1.0, 0.1), {}),
    # Non-default radius
    ((8, 1.0, 0.1), {"radius": 2.0}),
    # Different grid type
    ((8, 1.0, 0.1), {"grid": "GLQ"}),
    # Test with vector_as_SHCoeffs
    ((8, 1.0, 0.1), {"vector_as_SHGrid": False}),
]


@pytest.fixture(params=sobolev_parameter_sets)
def space(request) -> Sobolev:
    """
    Provides parametrized Sobolev space instances on a sphere for the tests.
    The tests will run once for each tuple of parameters in the list above.
    """
    # request.param will be one of the (args, kwargs) tuples from the list
    args, kwargs = request.param
    return Sobolev(*args, **kwargs)


class TestSphereSobolev(HilbertSpaceChecks):
    """
    Runs the standard suite of Hilbert space checks on the sphere.Sobolev class
    for various different initialization parameters.
    """

    pass


class TestSphereSobolevSpecifics:
    """
    A suite for tests that are specific to the sphere.Sobolev implementation.
    """

    def test_l2_inner_product_constant_function(self):
        """
        Tests that the L2 inner product of a constant function with itself
        is equal to the analytical result: c^2 * surface_area.
        """
        lmax = 16
        radius = 2.0
        constant_value = 3.0

        # 1. Create an L2 space (order=0)
        l2_space = Sobolev(lmax, 0.0, 1.0, radius=radius)

        # 2. Create a vector representing the constant function
        constant_vector = l2_space.zero
        constant_vector.data[:, :] = constant_value

        # 3. Calculate the L2 inner product using the space's method
        inner_product_result = l2_space.inner_product(constant_vector, constant_vector)

        # 4. Calculate the analytical result
        surface_area = 4 * np.pi * radius**2
        analytical_result = constant_value**2 * surface_area

        # 5. The two results should be numerically very close
        assert np.isclose(inner_product_result, analytical_result)
