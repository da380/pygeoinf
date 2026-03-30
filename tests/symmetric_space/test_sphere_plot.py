"""
Tests for the modernized plotting functions in the sphere module.
"""

import pytest
import matplotlib.pyplot as plt

# Safely import sphere modules, skipping tests if dependencies (pyshtools/cartopy) are missing
try:
    import pyshtools as sh
    import cartopy.crs as ccrs
    from cartopy.mpl.geoaxes import GeoAxes
    from pygeoinf.symmetric_space.sphere import (
        Lebesgue,
        plot,
        create_map_figure,
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
    # Define a low-degree space for fast testing
    space = Lebesgue(8, grid="DH")

    # Create a dummy harmonic grid (e.g., Y_1,0)
    coeffs = sh.SHCoeffs.from_zeros(space.lmax, normalization="ortho", csphase=1)
    coeffs.set_coeffs(1.0, 1, 0)
    u_grid = coeffs.expand(grid=space.grid_type)

    return space, u_grid


# =============================================================================
# Modern Layout & Canvas Tests
# =============================================================================


def test_create_map_figure():
    """Tests the modern helper for creating a GeoAxes canvas."""
    projection = ccrs.Robinson()
    fig, ax = create_map_figure(figsize=(10, 5), projection=projection)

    assert isinstance(ax, GeoAxes)
    # Compare the instances directly instead of using isinstance on an instance
    assert ax.projection == projection
    # Alternatively, check against the class: isinstance(ax.projection, ccrs.Robinson)

    # Ensure modern constrained_layout is enabled
    assert fig.get_constrained_layout()

    plt.close(fig)


def test_plot_creates_geoaxes_by_default(sphere_data):
    """Tests that plot() creates its own GeoAxes if none are provided."""
    _, u_grid = sphere_data
    ax, im = plot(u_grid, gridlines=True, coasts=True)

    assert isinstance(ax, GeoAxes)
    assert im is not None
    # Default projection in pygeoinf is PlateCarree
    assert isinstance(ax.projection, ccrs.PlateCarree)
    plt.close(ax.figure)


def test_plot_avoids_cartesian_axis_bleed(sphere_data):
    """Tests that plot() ignores active non-geographic axes."""
    _, u_grid = sphere_data

    # Create a standard Cartesian plot
    fig_cart, ax_cart = plt.subplots()
    ax_cart.plot([0, 1], [0, 1])

    # Call plot without an ax; it must spawn a new figure rather than using ax_cart
    ax_geo, _ = plot(u_grid)

    assert ax_geo is not ax_cart
    assert isinstance(ax_geo, GeoAxes)
    assert ax_geo.figure is not fig_cart

    plt.close(fig_cart)
    plt.close(ax_geo.figure)


# =============================================================================
# Feature & Convenience Tests
# =============================================================================


def test_plot_with_native_colorbar(sphere_data):
    """Tests the integrated colorbar toggle in the base library."""
    _, u_grid = sphere_data
    ax, im = plot(
        u_grid,
        colorbar=True,
        colorbar_kwargs={"label": "EWT (mm)", "orientation": "vertical"},
    )

    fig = ax.get_figure()
    # Check that a second axis (the colorbar) was added
    assert len(fig.axes) > 1
    # Verify the label was applied
    assert fig.axes[-1].get_ylabel() == "EWT (mm)"

    plt.close(fig)


def test_plot_contour_and_symmetric(sphere_data):
    """Tests the mapping function with contouring and symmetric kwargs."""
    _, u_grid = sphere_data

    # Use a custom projection to ensure it's respected
    fig, ax_in = create_map_figure(projection=ccrs.Mollweide())
    ax_out, im = plot(u_grid, ax=ax_in, contour=True, symmetric=True)

    assert ax_out is ax_in
    # Verify data range is roughly symmetric
    assert abs(im.get_clim()[0] + im.get_clim()[1]) < 1e-10

    plt.close(fig)


# =============================================================================
# Geodesic & Network Tests
# =============================================================================


def test_plot_geodesic():
    """Tests plotting a single great-circle path."""
    p1, p2 = (10.0, -20.0), (40.0, 50.0)
    ax = plot_geodesic(p1, p2, color="blue", linewidth=3)

    assert isinstance(ax, GeoAxes)
    # Check that a line artist was created
    assert len(ax.lines) > 0
    assert ax.lines[-1].get_color() == "blue"
    plt.close(ax.figure)


def test_plot_geodesic_network():
    """Tests plotting a source-receiver network."""
    paths = [((10.0, 20.0), (40.0, 50.0)), ((-10.0, -20.0), (30.0, 10.0))]
    ax = plot_geodesic_network(paths, alpha=0.5)

    assert isinstance(ax, GeoAxes)
    # 2 geodesic paths + coastlines
    assert len(ax.lines) >= 2
    # Check that markers (sources/receivers) were added via scatter
    assert len(ax.collections) >= 2
    plt.close(ax.figure)
