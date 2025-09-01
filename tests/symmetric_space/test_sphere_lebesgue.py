"""
Tests for function spaces on a sphere. 
"""

import pytest
import numpy as np
import pyshtools as sh
from pygeoinf.symmetric_space.sphere import Lebesgue

from ..checks.hilbert_space import HilbertSpaceChecks


@pytest.mark.parametrize("lmax, radius, grid", [(8, 1.0, "DH"), (16, 6371.0, "GLQ")])
class TestSphereLebesgue(HilbertSpaceChecks):
    """
    Runs standard and specific checks on the sphere.Lebesgue class.
    """

    @pytest.fixture
    def space(self, lmax, radius, grid) -> Lebesgue:
        """Provides a Lebesgue space instance on a sphere for the tests."""
        return Lebesgue(lmax, radius=radius, grid=grid)

    def test_project_zonal_harmonic(self, space: Lebesgue):
        """
        Tests if projecting a simple spherical harmonic (Y_1,0, which is
        proportional to cos(colatitude)) works correctly.
        """
        coeffs = sh.SHCoeffs.from_zeros(space.lmax, normalization="ortho", csphase=1)
        coeffs.set_coeffs(1.0, 1, 0)
        expected_grid = coeffs.expand(grid=space.grid)

        projected_grid = space.project_function(
            lambda point: np.sin(np.deg2rad(point[0]))
        )
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
            expected_eigenvalue = l * (l + 1) / space.radius**2
            assert np.isclose(space.laplacian_eigenvalue((l, 0)), expected_eigenvalue)

    def test_transform_round_trip(self, space: Lebesgue):
        """Ensures that transforming to components and back is self-consistent."""
        power_spectrum = np.ones(space.lmax + 1)
        coeffs = sh.SHCoeffs.from_random(
            power_spectrum, normalization="ortho", csphase=1
        )
        original_grid = coeffs.expand(grid=space.grid)

        components = space.to_components(original_grid)
        reconstructed_grid = space.from_components(components)

        assert np.allclose(original_grid.data, reconstructed_grid.data)

    def test_y10_projection_components(self, space: Lebesgue):
        """
        Tests that projecting the Y(l=1, m=0) harmonic produces the correct,
        sparse component vector.
        """
        # The Y_1,0 harmonic is proportional to sin(latitude)
        y10_func = lambda point: np.sin(np.deg2rad(point[0]))
        y10_vector = space.project_function(y10_func)

        components = space.to_components(y10_vector)

        # In the real-packed format, the index for (l, m) is l*(l+1)/2 + m.
        # For (l=1, m=0), the index is 1*2/2 + 0 = 1.
        y10_index = 1

        # Check that this component is the only significant one.
        assert not np.isclose(components[y10_index], 0.0)

        # Check that all other components are zero.
        components[y10_index] = 0.0
        # Use a slightly higher tolerance for GLQ grids as projection is less exact
        assert np.allclose(components, 0.0, atol=1e-9)
