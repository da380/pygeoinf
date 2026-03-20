"""
Tests for function spaces on a circle.
"""

import pytest
import numpy as np

from pygeoinf.symmetric_space.circle import Sobolev


@pytest.mark.parametrize("kmax, radius", [(16, 1.0), (32, 2.5)])
def test_sobolev_axioms(kmax: int, radius: float):
    """
    Verifies that the Sobolev space instance satisfies all Hilbert space axioms
    by calling its internal self-check method.
    """
    # The Sobolev constructor takes kmax, order, and scale as positional args
    space = Sobolev(kmax, 2.0, 0.5, radius=radius)
    space.check(n_checks=5)


class TestCircleSobolevGeometric:
    """
    Tests that depend only on the underlying geometry of the Sobolev space.
    """

    def test_project_constant_function(self):
        """Tests if projecting a constant function f(x) = c works correctly."""
        space = Sobolev(16, 2.0, 0.5, radius=1.0)
        projected_vector = space.project_function(lambda theta: 5.0)
        expected_vector = np.full_like(projected_vector, 5.0)
        assert np.allclose(projected_vector, expected_vector)


class TestCircleSobolevSpecific:
    """
    Tests functionalities that are specific to the Sobolev nature of the space,
    particularly those dependent on order and scale.
    """

    @pytest.fixture(params=[(1.0, 0.5), (2.0, 1.0)])
    def sobolev_space(self, request) -> Sobolev:
        """Provides a Sobolev space with varying order and scale."""
        order, scale = request.param
        return Sobolev(16, order, scale, radius=1.0)

    def test_dirac_functional_property(self, sobolev_space: Sobolev):
        """Tests that <δ_p, f> = f(p) for different Sobolev parameters."""
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
        low_order_space = Sobolev(16, 0.5, 1.0, radius=1.0)
        with pytest.raises(NotImplementedError):
            low_order_space.dirac(np.pi / 4)

    def test_degree_and_with_degree(self, sobolev_space: Sobolev):
        """Tests the unified degree property and the with_degree factory."""
        space = sobolev_space
        assert space.degree == space.kmax

        target_degree = space.kmax + 4
        new_space = space.with_degree(target_degree)

        assert new_space.degree == target_degree
        assert new_space.radius == space.radius
        assert new_space.order == space.order
        assert new_space.scale == space.scale

    def test_degree_transfer_operator(self, sobolev_space: Sobolev):
        """Tests the degree transfer operator's axioms in a mass-weighted space."""
        space = sobolev_space
        op_up = space.degree_transfer_operator(space.kmax + 4)
        op_up.check(
            n_checks=5
        )  # Extremely important: checks adjoint with mass operators
