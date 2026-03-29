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


# -------------------------------------------------- #
#             New State-Machine Tests                #
# -------------------------------------------------- #


def test_plot_uses_active_geoaxes(sphere_data):
    """Tests that plot() automatically grabs an active GeoAxes if ax=None."""
    _, u_grid = sphere_data

    fig = plt.figure()
    ax_active = fig.add_subplot(1, 1, 1, projection=ccrs.Mollweide())

    # We do not pass ax explicitly; the state machine should find ax_active
    ax_out, _ = plot(u_grid)

    assert ax_out is ax_active
    plt.close(fig)


def test_plot_avoids_cartesian_axis(sphere_data):
    """Tests that plot() spawns a new figure if the active axis is not a GeoAxes."""
    _, u_grid = sphere_data

    # Create a standard non-geographic Cartesian plot
    fig_cartesian, ax_standard = plt.subplots()
    ax_standard.plot([1, 2, 3], [1, 4, 9])

    # Call plot() without an explicit ax. It MUST ignore the active standard axis.
    ax_geo, _ = plot(u_grid)

    assert ax_geo is not ax_standard
    assert isinstance(ax_geo, GeoAxes)
    assert ax_geo.figure is not fig_cartesian

    plt.close(fig_cartesian)
    plt.close(ax_geo.figure)
