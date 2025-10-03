"""
Tests for the Lebesgue function space L² on a sphere.
"""

import pytest
import numpy as np
import pyshtools as sh

from pygeoinf.symmetric_space.sphere import Lebesgue


@pytest.mark.parametrize("lmax, radius, grid", [(8, 1.0, "DH"), (16, 6371.0, "GLQ")])
def test_lebesgue_axioms(lmax: int, radius: float, grid: str):
    """
    Verifies that the Lebesgue space instance satisfies all Hilbert space axioms
    by calling its internal self-check method.
    """
    space = Lebesgue(lmax, radius=radius, grid=grid)
    space.check(n_checks=5)


@pytest.mark.parametrize("lmax, radius, grid", [(16, 1.0, "DH")])
class TestSphereLebesgueSpecifics:
    """
    Tests functionalities that are unique to the Lebesgue space on the sphere,
    beyond the standard Hilbert space axioms.
    """

    @pytest.fixture
    def space(self, lmax: int, radius: float, grid: str) -> Lebesgue:
        """Provides a Lebesgue space instance for the specific tests."""
        return Lebesgue(lmax, radius=radius, grid=grid)

    def test_project_zonal_harmonic(self, space: Lebesgue):
        """
        Tests if projecting a simple spherical harmonic (Y_1,0, proportional
        to sin(latitude)) works correctly.
        """
        coeffs = sh.SHCoeffs.from_zeros(space.lmax, normalization="ortho", csphase=1)
        coeffs.set_coeffs(1.0, 1, 0)
        expected_grid = coeffs.expand(grid=space._grid_name())  #

        # Project the function sin(latitude)
        projected_grid = space.project_function(
            lambda point: np.sin(np.deg2rad(point[0]))
        )

        # Compare the normalized data arrays
        norm_expected = np.linalg.norm(expected_grid.data)
        norm_projected = np.linalg.norm(projected_grid.data)
        assert np.allclose(
            projected_grid.data / norm_projected,
            expected_grid.data / norm_expected,
            atol=1e-3,
        )

    def test_laplacian_eigenvalues(self, space: Lebesgue):
        """Tests the Laplacian eigenvalues against their analytical formula."""
        for l in range(space.lmax + 1):
            expected = l * (l + 1) / space.radius**2
            assert np.isclose(space.laplacian_eigenvalue((l, 0)), expected)

    def test_transform_round_trip(self, space: Lebesgue):
        """Ensures that transforming to components and back is self-consistent."""
        power_spectrum = np.ones(space.lmax + 1)
        coeffs = sh.SHCoeffs.from_random(
            power_spectrum, normalization="ortho", csphase=1
        )
        original_grid = coeffs.expand(grid=space._grid_name())  #

        components = space.to_components(original_grid)
        reconstructed_grid = space.from_components(components)

        assert np.allclose(original_grid.data, reconstructed_grid.data)
