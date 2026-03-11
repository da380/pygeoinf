"""
Tests for Lebesgue function spaces on a line segment (interval).
"""

import pytest
import numpy as np

from pygeoinf.symmetric_space.line import Lebesgue


@pytest.mark.parametrize("kmax, a, b, c", [(8, 0.0, 1.0, 0.1), (16, -2.0, 2.0, 0.5)])
def test_lebesgue_axioms(kmax: int, a: float, b: float, c: float):
    """
    Verifies that the Lebesgue space instance satisfies all Hilbert space axioms
    by calling its internal self-check method.
    """
    space = Lebesgue(kmax, a=a, b=b, c=c)
    space.check(n_checks=5)


@pytest.mark.parametrize("kmax, a, b, c", [(16, 0.0, 2.0, 0.2)])
class TestLineLebesgueSpecifics:
    """
    Tests functionalities that are unique to the Lebesgue space on the line.
    """

    @pytest.fixture
    def space(self, kmax: int, a: float, b: float, c: float) -> Lebesgue:
        """Provides a Lebesgue space instance for the specific tests."""
        return Lebesgue(kmax, a=a, b=b, c=c)

    def test_project_constant_function(self, space: Lebesgue):
        """Tests if projecting a constant function f(x) = const works correctly."""
        projected_vector = space.project_function(lambda x: 5.0)
        expected_vector = np.full_like(projected_vector, 5.0)
        assert np.allclose(projected_vector, expected_vector)

    def test_laplacian_eigenvalues(self, space: Lebesgue):
        """
        Tests the Laplacian eigenvalues against their analytical formula
        for a periodic domain of length L = b - a + 2c.
        """
        length = space.b - space.a + 2 * space.c
        for k in range(space.kmax + 1):
            expected = (2 * np.pi * k / length) ** 2
            assert np.isclose(space.laplacian_eigenvalue(k), expected)

    def test_transform_round_trip(self, space: Lebesgue):
        """Ensures that transforming to components and back is self-consistent."""
        # Use a function that is periodic over the *padded* domain to avoid ringing
        length = space.b - space.a + 2 * space.c
        original_vector = space.project_function(
            lambda x: np.cos(2 * np.pi * 3 * x / length)
            + np.sin(2 * np.pi * x / length)
        )
        components = space.to_components(original_vector)
        reconstructed_vector = space.from_components(components)
        assert np.allclose(original_vector, reconstructed_vector)
