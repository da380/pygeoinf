"""
Tests for function spaces on a circle.
"""

import pytest
import numpy as np

from pygeoinf.symmetric_space.circle import Lebesgue


@pytest.mark.parametrize("kmax, radius", [(8, 1.0), (16, 2.5)])
def test_lebesgue_axioms(kmax: int, radius: float):
    """
    Verifies that the Lebesgue space instance satisfies all Hilbert space axioms
    by calling its internal self-check method.
    """
    # The Lebesgue constructor takes kmax as a positional argument
    space = Lebesgue(kmax, radius=radius)
    space.check(n_checks=5)


@pytest.mark.parametrize("kmax, radius", [(16, 1.0)])
class TestCircleLebesgueSpecifics:
    """
    Tests functionalities that are unique to the Lebesgue space on the circle.
    """

    @pytest.fixture
    def space(self, kmax: int, radius: float) -> Lebesgue:
        """Provides a Lebesgue space instance for the specific tests."""
        return Lebesgue(kmax, radius=radius)

    def test_project_constant_function(self, space: Lebesgue):
        """Tests if projecting a constant function f(x) = c works correctly."""
        projected_vector = space.project_function(lambda theta: 5.0)
        expected_vector = np.full_like(projected_vector, 5.0)
        assert np.allclose(projected_vector, expected_vector)

    def test_laplacian_eigenvalues(self, space: Lebesgue):
        """Tests the Laplacian eigenvalues against their analytical formula."""
        for k in range(space.kmax + 1):
            expected = (k / space.radius) ** 2
            assert np.isclose(space.laplacian_eigenvalue(k), expected)

    def test_transform_round_trip(self, space: Lebesgue):
        """Ensures that transforming to components and back is self-consistent."""
        original_vector = space.project_function(
            lambda theta: np.cos(3 * theta) + np.sin(theta)
        )
        components = space.to_components(original_vector)
        reconstructed_vector = space.from_components(components)
        assert np.allclose(original_vector, reconstructed_vector)

    def test_degree_and_with_degree(self, space: Lebesgue):
        """Tests the unified degree property and the with_degree factory."""
        assert space.degree == space.kmax

        target_degree = space.kmax + 4
        new_space = space.with_degree(target_degree)

        assert new_space.degree == target_degree
        assert new_space.radius == space.radius
        assert isinstance(new_space, Lebesgue)

    def test_degree_transfer_operator(self, space: Lebesgue):
        """Tests the degree transfer operator's axioms and round-trip behavior."""
        # Test Operator Axioms
        op_up = space.degree_transfer_operator(space.kmax + 4)
        op_up.check(n_checks=5)

        op_down = op_up.codomain.degree_transfer_operator(space.kmax)
        op_down.check(n_checks=5)

        # Test Round-Trip (Upsample then Downsample)
        u_orig = space.project_function(lambda th: np.cos(2 * th) + np.sin(th))
        u_upsampled = op_up(u_orig)
        u_recon = op_down(u_upsampled)

        assert np.allclose(u_orig, u_recon)
