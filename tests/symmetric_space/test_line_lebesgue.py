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

    def test_project_constant_function(self, space):
        """
        Tests if projecting a constant function f(x) = const works correctly,
        specifically checking the raised cosine tapering in the padding regions.
        """
        constant_val = 5.0
        projected_vector = space.project_function(lambda x: constant_val)
        points = space.points()

        # 1. Check the physical interior [a, b] is exactly the constant
        interior_mask = (points >= space.a) & (points <= space.b)
        assert np.allclose(projected_vector[interior_mask], constant_val)

        # 2. Check the exterior boundaries (<= a-c and >= b+c) are exactly zero
        exterior_mask = (points <= space.a - space.c) | (points >= space.b + space.c)
        if np.any(exterior_mask):
            assert np.allclose(projected_vector[exterior_mask], 0.0)

        # 3. Check the padding transition regions are strictly between 0 and the constant
        taper_mask = (~interior_mask) & (~exterior_mask)
        if np.any(taper_mask):
            assert np.all(projected_vector[taper_mask] >= 0.0)
            assert np.all(projected_vector[taper_mask] <= constant_val)

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
