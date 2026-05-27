"""
Tests for the plotting functions in the plane module.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from pygeoinf.symmetric_space.plane import (
    Lebesgue,
    plot,
    plot_points,
    plot_geodesic,
    plot_geodesic_network,
)


@pytest.fixture
def plane_data():
    """Provides a basic Lebesgue space and dummy data array for testing."""
    # Define a low-degree space with distinct bounding box
    space = Lebesgue(8, ax=0.0, bx=10.0, cx=2.0, ay=0.0, by=5.0, cy=1.0)

    # Create a dummy 2D function: u(x,y) = sin(x) + cos(y)
    X_flat, Y_flat = space.points()
    u_flat = np.sin(X_flat) + np.cos(Y_flat)
    u_grid = u_flat.reshape((2 * space.kmax, 2 * space.kmax))

    return space, u_grid


# =============================================================================
# Modern Layout & Canvas Tests
# =============================================================================


def test_plot_creates_axes_by_default(plane_data):
    """Tests that plot() creates its own Matplotlib Axes if none are provided."""
    space, u_grid = plane_data
    ax, im = plot(space, u_grid)

    assert isinstance(ax, Axes)
    assert im is not None
    assert ax.get_aspect() == 1.0

    plt.close(ax.figure)


def test_plot_domain_clipping(plane_data):
    """Tests that the 'full' keyword toggles between physical and padded bounds."""
    space, u_grid = plane_data

    # Standard plot (cropped)
    ax_cropped, _ = plot(space, u_grid, full=False)
    assert ax_cropped.get_xlim() == (space.bounds_x[0], space.bounds_x[1])
    assert ax_cropped.get_ylim() == (space.bounds_y[0], space.bounds_y[1])

    # Full plot (including padding)
    ax_full, _ = plot(space, u_grid, full=True)
    assert ax_full.get_xlim() == (
        space.bounds_x[0] - space.bounds_x[2],
        space.bounds_x[1] + space.bounds_x[2],
    )
    assert ax_full.get_ylim() == (
        space.bounds_y[0] - space.bounds_y[2],
        space.bounds_y[1] + space.bounds_y[2],
    )

    plt.close(ax_cropped.figure)
    plt.close(ax_full.figure)


def test_plot_avoids_cartesian_axis_bleed(plane_data):
    """Tests that plot() ignores active independent axes unless explicitly passed."""
    space, u_grid = plane_data

    fig_active, ax_active = plt.subplots()
    ax_active.plot([0, 1], [0, 1])

    ax_plane, _ = plot(space, u_grid)

    assert ax_plane is not ax_active
    assert ax_plane.figure is not fig_active

    plt.close(fig_active)
    plt.close(ax_plane.figure)


# =============================================================================
# Feature & Convenience Tests
# =============================================================================


def test_plot_with_native_colorbar(plane_data):
    """Tests the integrated colorbar toggle and dynamic attribute attachment."""
    space, u_grid = plane_data
    ax, im = plot(
        space,
        u_grid,
        colorbar=True,
        colorbar_kwargs={"label": "Amplitude"},
    )

    fig = ax.get_figure()
    assert len(fig.axes) > 1
    assert fig.axes[-1].get_xlabel() == "Amplitude"
    assert hasattr(im, "colorbar")

    plt.close(fig)


def test_plot_contour_and_symmetric(plane_data):
    """Tests the mapping function with contouring and symmetric kwargs."""
    space, u_grid = plane_data

    fig, ax_in = plt.subplots()
    ax_out, im = plot(space, u_grid, ax=ax_in, contour=True, symmetric=1.5)

    assert ax_out is ax_in

    # Verify data range is symmetric
    clim = im.get_clim()
    assert abs(clim[0] + clim[1]) < 1e-10

    plt.close(fig)


# =============================================================================
# Geodesic, Point & Network Tests
# =============================================================================


def test_plot_points():
    """Tests plotting point observations with attached colorbars."""
    points = [(2.0, 1.0), (8.0, 4.0)]
    data = [1.0, -1.0]

    ax, sc = plot_points(points, data=data, symmetric=True, colorbar=True)

    offsets = sc.get_offsets()
    assert np.allclose(offsets[0], [2.0, 1.0])
    assert hasattr(sc, "colorbar")

    plt.close(ax.figure)


def test_plot_geodesic():
    """Tests plotting a straight path on the 2D plane."""
    p1, p2 = (1.0, 1.0), (9.0, 4.0)
    ax, line = plot_geodesic(p1, p2, color="green")

    assert isinstance(ax, Axes)
    assert line is not None
    assert line.get_color() == "green"

    plt.close(ax.figure)


def test_plot_geodesic_network():
    """Tests plotting a source-receiver network on the Plane."""
    paths = [((1.0, 1.0), (9.0, 4.0)), ((2.0, 4.0), (8.0, 1.0))]
    ax, artists = plot_geodesic_network(paths, alpha=0.5)

    assert isinstance(ax, Axes)
    # 2 lines + 2 scatter collections
    assert len(artists) >= 2
    assert len(ax.collections) >= 2

    plt.close(ax.figure)
