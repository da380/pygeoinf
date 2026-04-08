"""
Tests for the Sobolev function space Hˢ on a 2D Torus.
"""

import pytest
import numpy as np

from pygeoinf.symmetric_space.torus import Sobolev


@pytest.mark.parametrize("kmax, radius", [(4, 1.0), (6, 2.5)])
def test_sobolev_axioms(kmax: int, radius: float):
    """
    Verifies that the Sobolev space instance satisfies all Hilbert space axioms
    by calling its internal self-check method.
    """
    # Note: Order must be > 1.0 on a 2D manifold for continuous point evaluations
    space = Sobolev(kmax, 1.5, 0.5, radius_x=radius, radius_y=radius)
    space.check(n_checks=5)


class TestTorusSobolevGeometric:
    """
    Tests that depend only on the underlying geometry of the Sobolev space.
    """

    def test_project_constant_function(self):
        """Tests if projecting a constant function f(p) = c works correctly."""
        space = Sobolev(6, 1.5, 0.5, radius_x=1.0, radius_y=1.0)
        projected_vector = space.project_function(lambda p: 5.0)
        expected_vector = np.full_like(projected_vector, 5.0)
        assert np.allclose(projected_vector, expected_vector)


class TestTorusSobolevSpecific:
    """
    Tests functionalities specific to the Sobolev nature of the space,
    particularly those dependent on order and scale mass-weighting.
    """

    @pytest.fixture(params=[(1.5, 0.5), (2.0, 1.0)])
    def sobolev_space(self, request) -> Sobolev:
        """Provides a Sobolev space with varying order and scale."""
        order, scale = request.param
        return Sobolev(6, order, scale, radius_x=1.0, radius_y=1.0)

    def test_dirac_functional_property(self, sobolev_space: Sobolev):
        """
        Tests that applying the Dirac functional δ_p to a function f
        correctly evaluates the function at point p, i.e., <δ_p, f> = f(p).
        """
        space = sobolev_space
        test_point = (np.pi / 4, np.pi / 3)

        def test_func(p: tuple[float, float]) -> float:
            return np.cos(p[0]) + 2 * np.sin(2 * p[1])

        dirac_functional = space.dirac(test_point)
        func_vector = space.project_function(test_func)

        functional_evaluation = space.duality_product(dirac_functional, func_vector)
        direct_evaluation = test_func(test_point)

        assert np.isclose(functional_evaluation, direct_evaluation, rtol=1e-2)

    def test_dirac_order_requirement(self):
        """
        Tests that the dirac functional raises an error for Sobolev
        order <= 1.0, as required by theory for a 2D manifold.
        """
        # Removed safe=False
        low_order_space = Sobolev(6, 1.0, 1.0, radius_x=1.0, radius_y=1.0)
        with pytest.raises(NotImplementedError):
            low_order_space.dirac((np.pi, np.pi))

    def test_sobolev_coefficient_operators_axioms(self, sobolev_space: Sobolev):
        """
        Verifies that the Sobolev coefficient operators satisfy all
        LinearOperator axioms (including mass-weighted adjoints).
        """
        kmax = sobolev_space.kmax

        op_to = sobolev_space.to_coefficient_operator(kmax)
        op_to.check(n_checks=5)

        op_from = sobolev_space.from_coefficient_operator(kmax)
        op_from.check(n_checks=5)

    def test_spectral_projection_operator_axioms(self, sobolev_space: Sobolev):
        """
        Verifies that the Sobolev spectral projection satisfies LinearOperator
        axioms, meaning the adjoint correctly applies the inverse mass metric.
        """
        modes = [(0, 0), (1, 0), (0, 1), (1, 1)]
        op_proj = sobolev_space.spectral_projection_operator(modes)
        op_proj.check(n_checks=5)

    def test_degree_and_with_degree(self, sobolev_space: Sobolev):
        """Tests the unified degree property and the with_degree factory."""
        space = sobolev_space
        assert space.degree == space.kmax

        target_degree = space.kmax + 4
        new_space = space.with_degree(target_degree)

        assert new_space.degree == target_degree
        assert new_space.radius_x == space.radius_x
        assert new_space.radius_y == space.radius_y
        assert new_space.order == space.order
        assert new_space.scale == space.scale

    def test_degree_transfer_operator(self, sobolev_space: Sobolev):
        """Tests the degree transfer operator's axioms in a mass-weighted space."""
        op_up = sobolev_space.degree_transfer_operator(sobolev_space.kmax + 4)
        # This will fail if the metric_ratio scaling in the Lebesgue adjoint is broken!
        op_up.check(n_checks=5)


def test_factory_methods():
    """Tests the automatic truncation degree factories for Sobolev spaces."""
    # Increased kernel_scale to 1.0 for faster convergence in the test
    space = Sobolev.from_sobolev_kernel_prior(
        4.0, 1.0, 1.5, 0.5, radius_x=2.0, radius_y=2.0, min_degree=4, power_of_two=True
    )
    assert isinstance(space, Sobolev)
    assert space.order == 1.5
    assert space.scale == 0.5
    assert space.radius_x == 2.0
    assert space.kmax >= 4
    assert (space.kmax & (space.kmax - 1)) == 0  # Power of two check
