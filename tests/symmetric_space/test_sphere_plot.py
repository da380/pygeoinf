"""
Tests for the plotting functions in the sphere module.
"""

import pytest
import matplotlib.pyplot as plt

# Safely import sphere modules, skipping tests if dependencies are missing
try:
    import pyshtools as sh
    import cartopy.crs as ccrs
    from cartopy.mpl.geoaxes import GeoAxes
    from pygeoinf.symmetric_space.sphere import (
        Lebesgue,
        plot,
        plot_geodesic,
        plot_geodesic_network,
    )
except ImportError:
    pytest.skip(
        "pyshtools or cartopy missing; skipping sphere plot tests",
        allow_module_level=True,
    )


@pytest.fixture
def sphere_data():
    """Provides a basic pyshtools SHGrid for testing."""
    space = Lebesgue(8, grid="DH")

    # Create a dummy harmonic grid (e.g., Y_1,0)
    coeffs = sh.SHCoeffs.from_zeros(space.lmax, normalization="ortho", csphase=1)
    coeffs.set_coeffs(1.0, 1, 0)
    u_grid = coeffs.expand(grid=space.grid_type)

    return space, u_grid


def test_plot_creates_geoaxes(sphere_data):
    """Tests the primary mapping function creates GeoAxes and returns the mappable."""
    _, u_grid = sphere_data

    # Test standard pcolormesh
    ax, im = plot(u_grid, gridlines=True, coasts=True)

    assert isinstance(ax, GeoAxes)
    assert im is not None
    # Ensure projection was applied (default is PlateCarree)
    assert isinstance(ax.projection, ccrs.PlateCarree)
    plt.close(ax.figure)


def test_plot_contour_and_symmetric(sphere_data):
    """Tests the mapping function with contouring and symmetric kwargs."""
    _, u_grid = sphere_data

    # Pass an existing axes
    fig, ax_in = plt.subplots(subplot_kw={"projection": ccrs.Mollweide()})

    ax_out, im = plot(u_grid, ax=ax_in, contour=True, symmetric=True)

    assert ax_out is ax_in
    assert im is not None
    plt.close(fig)


def test_plot_geodesic():
    """Tests plotting a single great-circle path."""
    p1 = (10.0, -20.0)  # Lat, Lon
    p2 = (40.0, 50.0)

    ax = plot_geodesic(p1, p2, color="blue")

    assert isinstance(ax, GeoAxes)
    assert len(ax.lines) > 0
    assert ax.lines[0].get_color() == "blue"
    plt.close(ax.figure)


def test_plot_geodesic_network():
    """Tests plotting a full source-receiver network."""
    paths = [((10.0, 20.0), (40.0, 50.0)), ((-10.0, -20.0), (30.0, 10.0))]

    ax = plot_geodesic_network(paths, alpha=0.7)

    assert isinstance(ax, GeoAxes)
    assert len(ax.lines) == 2  # Two geodesic paths
    assert len(ax.collections) == 3  # Coastlines + Sources + Receivers
    plt.close(ax.figure)
