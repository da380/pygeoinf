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
        expected_grid = coeffs.expand(grid=space.grid_type)  #

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
        original_grid = coeffs.expand(grid=space.grid_type)  #

        components = space.to_components(original_grid)
        reconstructed_grid = space.from_components(components)

        assert np.allclose(original_grid.data, reconstructed_grid.data)

    def test_coefficient_operators_axioms(self, space: Lebesgue):
        """
        Verifies that the coefficient conversion operators satisfy all
        LinearOperator axioms (linearity, adjoint correctness) using the
        standard check() method.
        """
        # Test 'to_coefficient_operator'
        op_to = space.to_coefficient_operator(space.lmax)
        op_to.check(n_checks=5)

        # Test 'from_coefficient_operator'
        op_from = space.from_coefficient_operator(space.lmax)
        op_from.check(n_checks=5)

    def test_coefficient_operators_round_trip(self, space: Lebesgue):
        """
        Tests that the to_coefficient and from_coefficient operators are
        inverse to each other.
        """
        T_to = space.to_coefficient_operator(space.lmax)
        T_from = space.from_coefficient_operator(space.lmax)

        # 1. Test Grid -> Vector -> Grid
        # Create a random function on the sphere
        coeffs = sh.SHCoeffs.from_random(np.ones(space.lmax + 1), normalization="ortho")
        u = space.from_coefficients(coeffs)

        vec = T_to(u)
        u_recon = T_from(vec)

        assert np.allclose(u.data, u_recon.data)

        # 2. Test Vector -> Grid -> Vector
        vec_in = np.random.randn(T_to.codomain.dim)
        u_from = T_from(vec_in)
        vec_out = T_to(u_from)

        assert np.allclose(vec_in, vec_out)

    def test_degree_and_with_degree(self, space: Lebesgue):
        """Tests the unified degree property and the with_degree factory."""
        assert space.degree == space.lmax

        target_degree = space.lmax + 4
        new_space = space.with_degree(target_degree)

        assert new_space.degree == target_degree
        assert new_space.radius == space.radius
        assert new_space.grid == space.grid
        assert new_space.extend == space.extend

    def test_degree_transfer_operator(self, space: Lebesgue):
        """Tests the degree transfer operator's axioms and round-trip behavior."""
        op_up = space.degree_transfer_operator(space.lmax + 4)
        op_up.check(n_checks=5)

        op_down = op_up.codomain.degree_transfer_operator(space.lmax)

        # Test Round-Trip (Upsample then Downsample)
        coeffs = sh.SHCoeffs.from_random(np.ones(space.lmax + 1), normalization="ortho")
        u_orig = space.from_coefficients(coeffs)

        u_upsampled = op_up(u_orig)
        u_recon = op_down(u_upsampled)

        assert np.allclose(u_orig.data, u_recon.data)
