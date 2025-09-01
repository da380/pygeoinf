"""
Tests for function spaces on a circle. 
"""

import pytest
import numpy as np
from pygeoinf.symmetric_space.circle import Sobolev

from ..checks.hilbert_space import HilbertSpaceChecks


# A fixture for tests that only depend on the geometry (kmax, radius)
@pytest.fixture
def geometric_space() -> Sobolev:
    """Provides a Sobolev space with fixed order/scale for geometric tests."""
    return Sobolev(16, 2.0, 0.5, radius=1.0)


# A parameterized fixture for tests that depend on Sobolev parameters
@pytest.fixture(params=[(1.0, 0.5), (2.0, 1.0)])
def sobolev_space(request) -> Sobolev:
    """Provides a Sobolev space with varying order and scale."""
    order, scale = request.param
    return Sobolev(16, order, scale, radius=1.0)


class TestCircleSobolevGeometric(HilbertSpaceChecks):
    """
    Runs Hilbert space checks and tests that depend only on the underlying
    geometry of the Sobolev space.
    """

    @pytest.fixture
    def space(self, geometric_space: Sobolev) -> Sobolev:
        """Adapter fixture for the HilbertSpaceChecks."""
        return geometric_space

    def test_project_constant_function(self, space: Sobolev):
        """
        Tests if projecting a constant function f(x) = c works correctly.
        """
        constant_value = 5.0
        projected_vector = space.project_function(lambda theta: constant_value)
        expected_vector = np.full_like(projected_vector, constant_value)
        assert np.allclose(projected_vector, expected_vector)


class TestCircleSobolevSpecific:
    """
    Tests functionalities that are specific to the Sobolev nature of the space,
    particularly those dependent on order and scale.
    """

    def test_dirac_functional_property(self, sobolev_space: Sobolev):
        """
        Tests that <δ_p, f> = f(p) for different Sobolev parameters.
        """
        space = sobolev_space
        test_angle = np.pi / 4
        test_func = lambda theta: np.cos(theta) + 2 * np.sin(2 * theta)

        dirac_representation = space.dirac_representation(test_angle)
        func_vector = space.project_function(test_func)

        inner_product_result = space.inner_product(dirac_representation, func_vector)
        direct_evaluation_result = test_func(test_angle)

        assert np.isclose(inner_product_result, direct_evaluation_result)

    def test_dirac_order_requirement(self):
        """
        Tests that the dirac functional raises an error for Sobolev
        order <= 0.5, as required by theory.
        """
        # This test doesn't need a fixture as it creates its own specific instance
        low_order_space = Sobolev(16, 0.5, 1.0, radius=1.0)
        with pytest.raises(NotImplementedError):
            low_order_space.dirac(np.pi / 4)
