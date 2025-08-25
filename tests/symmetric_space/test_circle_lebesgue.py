"""
Tests for function spaces on a circle. 
"""

import pytest
import numpy as np
from pygeoinf.symmetric_space.circle import Lebesgue

from ..checks.hilbert_space import HilbertSpaceChecks


@pytest.mark.parametrize("kmax, radius", [(8, 1.0), (16, 2.5)])
class TestCircleLebesgue(HilbertSpaceChecks):
    """
    Runs standard and specific checks on the circle.Lebesgue class.
    """

    @pytest.fixture
    def space(self, kmax, radius) -> Lebesgue:
        """Provides a Lebesgue space instance on a circle for the tests."""
        return Lebesgue(kmax, radius=radius)

    def test_project_constant_function(self, space: Lebesgue):
        """Tests if projecting a constant function f(x) = c works correctly."""
        constant_value = 5.0
        projected_vector = space.project_function(lambda theta: constant_value)
        expected_vector = np.full_like(projected_vector, constant_value)
        assert np.allclose(projected_vector, expected_vector)

    def test_project_sine_function(self, space: Lebesgue):
        """Tests if projecting a simple sine function works correctly."""
        projected_vector = space.project_function(lambda theta: np.sin(2 * theta))
        expected_vector = np.sin(2 * space.angles())
        assert np.allclose(projected_vector, expected_vector)

    def test_laplacian_eigenvalues(self, space: Lebesgue):
        """Tests the Laplacian eigenvalues against their analytical formula."""
        for k in range(space.kmax + 1):
            expected_eigenvalue = (k / space.radius) ** 2
            assert np.isclose(space.laplacian_eigenvalue(k), expected_eigenvalue)

    def test_transform_round_trip(self, space: Lebesgue):
        """Ensures that transforming to components and back is self-consistent."""
        original_vector = space.project_function(
            lambda theta: np.cos(3 * theta) + np.sin(theta)
        )
        components = space.to_components(original_vector)
        reconstructed_vector = space.from_components(components)
        assert np.allclose(original_vector, reconstructed_vector)

    def test_cosine_projection_components(self, space: Lebesgue):
        """
        Tests that projecting cos(theta) produces the correct, sparse
        component vector.
        """
        # Project cos(theta) onto the grid.
        cos_vector = space.project_function(np.cos)

        # Get the real component vector from the projection.
        components = space.to_components(cos_vector)

        # The function cos(theta) corresponds to the k=1 Fourier mode.
        # Its real coefficient should be non-zero. The component vector stores
        # real parts first, so this is at index 1.
        assert not np.isclose(components[1], 0.0)

        # Create a mask to check that all OTHER components are zero.
        # Set the expected non-zero component to zero in the mask.
        components[1] = 0.0
        assert np.allclose(components, 0.0)
