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

    def test_degree_and_with_degree(self, space: Lebesgue):
        """Tests the unified degree property and the with_degree factory."""
        assert space.degree == space.kmax

        target_degree = space.kmax + 4
        new_space = space.with_degree(target_degree)

        assert new_space.degree == target_degree
        assert new_space.a == space.a
        assert new_space.b == space.b
        assert new_space.c == space.c

    def test_degree_transfer_operator(self, space: Lebesgue):
        """Tests the degree transfer operator's axioms and round-trip behavior."""
        op_up = space.degree_transfer_operator(space.kmax + 4)
        op_up.check(n_checks=5)

        op_down = op_up.codomain.degree_transfer_operator(space.kmax)

        length = space.b - space.a + 2 * space.c
        u_orig = space.project_function(lambda x: np.cos(2 * np.pi * x / length))
        u_upsampled = op_up(u_orig)
        u_recon = op_down(u_upsampled)

        assert np.allclose(u_orig, u_recon)


def test_factory_methods():
    """Tests the automatic truncation degree factories for L2 spaces on a line."""
    space = Lebesgue.from_heat_kernel_prior(
        0.1, a=-1.0, b=3.0, c=0.5, min_degree=4, power_of_two=True
    )
    assert isinstance(space, Lebesgue)
    assert space.a == -1.0
    assert space.b == 3.0
    assert space.c == 0.5
    assert space.kmax >= 4
    assert (space.kmax & (space.kmax - 1)) == 0
