"""
Tests for the plotting functions in the torus module.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from pygeoinf.symmetric_space.torus import (
    Lebesgue,
    plot,
    plot_points,
    plot_geodesic,
    plot_geodesic_network,
)


@pytest.fixture
def torus_data():
    """Provides a basic Lebesgue space and dummy data array for testing."""
    # Define a low-degree space for fast testing
    space = Lebesgue(8)

    # Create a dummy 2D function: u(x,y) = sin(x) * cos(y)
    X_flat, Y_flat = space.points()
    u_flat = np.sin(X_flat) * np.cos(Y_flat)
    u_grid = u_flat.reshape((2 * space.kmax, 2 * space.kmax))

    return space, u_grid


# =============================================================================
# Modern Layout & Canvas Tests
# =============================================================================


def test_plot_creates_axes_by_default(torus_data):
    """Tests that plot() creates its own Matplotlib Axes if none are provided."""
    space, u_grid = torus_data
    ax, im = plot(space, u_grid)

    assert isinstance(ax, Axes)
    assert im is not None

    # Verify the bounds are strictly set to the Torus domain [0, 2pi]
    assert ax.get_xlim() == (0, 2 * np.pi)
    assert ax.get_ylim() == (0, 2 * np.pi)
    assert ax.get_aspect() == 1.0  # Aspect "equal"

    plt.close(ax.figure)


def test_plot_avoids_cartesian_axis_bleed(torus_data):
    """Tests that plot() ignores active independent axes unless explicitly passed."""
    space, u_grid = torus_data

    # Create a standard Cartesian plot
    fig_active, ax_active = plt.subplots()
    ax_active.plot([0, 1], [0, 1])

    # Call plot without an ax; it must spawn a new figure rather than using ax_active
    ax_torus, _ = plot(space, u_grid)

    assert ax_torus is not ax_active
    assert ax_torus.figure is not fig_active

    plt.close(fig_active)
    plt.close(ax_torus.figure)


# =============================================================================
# Feature & Convenience Tests
# =============================================================================


def test_plot_with_native_colorbar(torus_data):
    """Tests the integrated colorbar toggle and dynamic attribute attachment."""
    space, u_grid = torus_data
    ax, im = plot(
        space,
        u_grid,
        colorbar=True,
        colorbar_kwargs={"label": "Amplitude", "orientation": "horizontal"},
    )

    fig = ax.get_figure()
    # Check that a second axis (the colorbar) was added
    assert len(fig.axes) > 1
    # Verify the label was applied (horizontal colorbars use xlabel)
    assert fig.axes[-1].get_xlabel() == "Amplitude"
    # Verify our custom dynamic attribute is attached
    assert hasattr(im, "colorbar")

    plt.close(fig)


def test_plot_contour_and_symmetric(torus_data):
    """Tests the mapping function with contouring and symmetric kwargs."""
    space, u_grid = torus_data

    # Pass in a predefined axis
    fig, ax_in = plt.subplots()
    ax_out, im = plot(space, u_grid, ax=ax_in, contour=True, symmetric=True)

    assert ax_out is ax_in
    # Verify data range is roughly symmetric
    clim = im.get_clim()
    assert abs(clim[0] + clim[1]) < 1e-10

    plt.close(fig)


# =============================================================================
# Geodesic, Point & Network Tests
# =============================================================================


def test_plot_points_wrapping():
    """Tests that observation points are properly wrapped onto the Torus domain."""
    # Provide points outside the [0, 2pi] bounds
    points = [(0.0, 0.0), (3 * np.pi, -np.pi / 2)]
    data = [1.0, -1.0]

    ax, sc = plot_points(points, data=data, symmetric=True, colorbar=True)

    # Fetch the actual plotted coordinates
    offsets = sc.get_offsets()

    # (0.0, 0.0) remains unchanged
    assert np.allclose(offsets[0], [0.0, 0.0])

    # (3*pi, -pi/2) should wrap to (pi, 1.5*pi)
    assert np.allclose(offsets[1], [np.pi, 1.5 * np.pi])

    # Ensure colorbar attached properly
    assert hasattr(sc, "colorbar")

    plt.close(ax.figure)


def test_plot_geodesic_wrap_handling():
    """Tests plotting a single path and verifies NaN injection for wrapped lines."""
    # A path that crosses the periodic boundary (e.g., from 0.1 to 2*pi - 0.1)
    p1, p2 = (0.1, 0.1), (2 * np.pi - 0.1, 2 * np.pi - 0.1)

    ax, line = plot_geodesic(p1, p2, color="green", linewidth=3)

    assert isinstance(ax, Axes)
    assert line is not None
    assert line.get_color() == "green"

    # To prevent diagonal streaks, the plotting function must inject NaNs where the path wraps
    xdata = line.get_xdata()
    assert np.any(np.isnan(xdata))

    plt.close(ax.figure)


def test_plot_geodesic_network():
    """Tests plotting a source-receiver network on the Torus."""
    paths = [
        ((0.5, 0.5), (1.5, 1.5)),
        (
            (3 * np.pi, 0.0),
            (0.0, 3 * np.pi),
        ),  # Will test the scatter coordinate wrapping too
    ]
    ax, artists = plot_geodesic_network(paths, alpha=0.5)

    assert isinstance(ax, Axes)
    # The artists list should contain the 2 lines and the 2 scatter collections
    assert len(artists) >= 2
    # Check that markers (sources/receivers) were added via scatter
    assert len(ax.collections) >= 2

    plt.close(ax.figure)
