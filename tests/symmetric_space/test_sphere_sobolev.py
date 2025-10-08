"""
Tests for the Sobolev function space Hˢ on a sphere.
"""

import pytest
import numpy as np
import pyshtools as sh

from pygeoinf.symmetric_space.sphere import Sobolev


@pytest.mark.parametrize("lmax, radius, grid", [(8, 1.0, "DH"), (16, 6371.0, "GLQ")])
def test_sobolev_axioms(lmax: int, radius: float, grid: str):
    """
    Verifies that the Sobolev space instance satisfies all Hilbert space axioms
    by calling its internal self-check method.
    """
    # Sobolev parameters can be fixed for the generic axiom checks
    space = Sobolev(lmax, 2.0, 0.5, radius=radius, grid=grid)
    space.check(n_checks=5)


class TestSphereSobolevSpecifics:
    """
    Tests functionalities that are specific to the Sobolev nature of the space,
    particularly those dependent on order and scale.
    """

    @pytest.fixture(params=[(1.5, 0.5), (2.5, 0.8)])
    def sobolev_space(self, request) -> Sobolev:
        """Provides a Sobolev space with varying order and scale."""
        order, scale = request.param
        return Sobolev(16, order, scale, radius=1.0, grid="DH")

    def test_dirac_functional_property(self, sobolev_space: Sobolev):
        """
        Tests that applying the Dirac functional δ_p to a function f
        correctly evaluates the function at point p, i.e., <δ_p, f> = f(p).
        """
        space = sobolev_space
        test_point = (30.0, 60.0)  # 30°N, 60°E

        # A combination of Y(1,0) and Y(2,2), which is analytic everywhere.
        def test_func(point: tuple[float, float]) -> float:
            lat_rad, lon_rad = np.deg2rad(point[0]), np.deg2rad(point[1])
            return 2 * np.sin(lat_rad) + np.cos(lat_rad) ** 2 * np.cos(2 * lon_rad)

        dirac_functional = space.dirac(test_point)
        func_vector = space.project_function(test_func)

        functional_evaluation = space.duality_product(dirac_functional, func_vector)
        direct_evaluation = test_func(test_point)

        assert np.isclose(functional_evaluation, direct_evaluation, rtol=1e-2)

    def test_eigenfunction_norm(self, sobolev_space: Sobolev):
        """
        Tests that the Sobolev norm of a spherical harmonic eigenfunction
        matches its analytically calculated value.
        """
        space = sobolev_space
        l, m = 5, 2  # A non-trivial spherical harmonic

        coeffs = sh.SHCoeffs.from_zeros(space.lmax, normalization="ortho", csphase=1)
        coeffs.set_coeffs(1.0, l, m)
        harmonic_grid = space.from_coefficients(coeffs)  #

        numerical_norm = space.norm(harmonic_grid)

        eigenvalue = space.laplacian_eigenvalue((l, m))  #
        scaling_factor = np.sqrt(space.sobolev_function(eigenvalue))  #
        analytical_norm = space.radius * scaling_factor

        assert np.isclose(numerical_norm, analytical_norm)
