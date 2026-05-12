"""Tests for exact spherical-cap integral functionals on the sphere."""

import numpy as np
import pytest
from scipy.special import eval_legendre

from pygeoinf.linear_forms import LinearForm
from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev


def _cap_integral_y_l0(ell: int, alpha: float) -> float:
    r"""Exact north-pole cap integral of the orthonormal real $Y_{\ell 0}$."""
    cos_alpha = np.cos(alpha)
    if ell == 0:
        return np.sqrt(np.pi) * (1.0 - cos_alpha)
    return np.sqrt(np.pi / (2 * ell + 1)) * (
        eval_legendre(ell - 1, cos_alpha) - eval_legendre(ell + 1, cos_alpha)
    )


def test_spherical_cap_integral_matches_area_and_low_order_moments():
    """The cap integral should reproduce exact low-order moment formulas."""
    space = Lebesgue(12, radius=1.0, grid="DH")
    alpha = np.pi / 2.0
    cap_integral = space.spherical_cap_integral((90.0, 0.0), alpha)

    one = space.project_function(lambda _: 1.0)
    x_coord = space.project_function(
        lambda point: np.cos(np.deg2rad(point[0])) * np.cos(np.deg2rad(point[1]))
    )
    y_coord = space.project_function(
        lambda point: np.cos(np.deg2rad(point[0])) * np.sin(np.deg2rad(point[1]))
    )
    z_coord = space.project_function(lambda point: np.sin(np.deg2rad(point[0])))
    z_squared = space.project_function(lambda point: np.sin(np.deg2rad(point[0])) ** 2)

    np.testing.assert_allclose(cap_integral(one), 2.0 * np.pi, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(cap_integral(x_coord), 0.0, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(cap_integral(y_coord), 0.0, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(cap_integral(z_coord), np.pi, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        cap_integral(z_squared), 2.0 * np.pi / 3.0, rtol=1e-10, atol=1e-10
    )


def test_north_pole_cap_integral_matches_zonal_harmonic_formula():
    r"""North-pole cap coefficients should match exact $Y_{\ell 0}$ integrals."""
    space = Lebesgue(8, radius=1.0, grid="DH")
    alpha = 0.7
    cap_integral = space.spherical_cap_integral((90.0, 0.0), alpha)

    for ell in range(space.lmax + 1):
        zonal_index = space.index_to_integer((ell, 0))
        np.testing.assert_allclose(
            cap_integral.components[zonal_index],
            _cap_integral_y_l0(ell, alpha),
            rtol=1e-12,
            atol=1e-12,
        )

        for order in range(1, ell + 1):
            positive_index = space.index_to_integer((ell, order))
            negative_index = space.index_to_integer((ell, -order))
            np.testing.assert_allclose(
                cap_integral.components[positive_index], 0.0, rtol=0.0, atol=1e-12
            )
            np.testing.assert_allclose(
                cap_integral.components[negative_index], 0.0, rtol=0.0, atol=1e-12
            )


def test_geodesic_ball_integral_scales_with_sphere_area_element():
    """A geodesic ball integral should use physical surface area on $S_R^2$."""
    sphere_radius = 2.5
    space = Lebesgue(8, radius=sphere_radius, grid="DH")
    alpha = 0.6
    geodesic_radius = sphere_radius * alpha
    cap_integral = space.geodesic_ball_integral((90.0, 0.0), geodesic_radius)

    one = space.project_function(lambda _: 1.0)
    expected_area = sphere_radius**2 * 2.0 * np.pi * (1.0 - np.cos(alpha))

    np.testing.assert_allclose(cap_integral(one), expected_area, rtol=1e-10, atol=1e-10)


def test_sobolev_geodesic_ball_average_returns_domain_linear_form():
    """Sobolev spaces should expose cap averages with Sobolev-domain forms."""
    space = Sobolev(8, 1.5, 0.2, radius=1.0, grid="DH")
    cap_average = space.geodesic_ball_average((90.0, 0.0), np.pi / 2.0)
    constant = space.project_function(lambda _: 2.5)

    assert isinstance(cap_average, LinearForm)
    assert cap_average.domain == space
    np.testing.assert_allclose(cap_average(constant), 2.5, rtol=1e-10, atol=1e-10)


def test_spherical_cap_integral_rejects_invalid_radius():
    r"""Spherical-cap angular radii must lie in $[0, \pi]$."""
    space = Lebesgue(4, radius=1.0, grid="DH")

    with pytest.raises(ValueError, match="angular radius"):
        space.spherical_cap_integral((90.0, 0.0), -0.1)

    with pytest.raises(ValueError, match="angular radius"):
        space.spherical_cap_integral((90.0, 0.0), np.pi + 0.1)