"""
Tests for the plotting functions in the plane module.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from pygeoinf.symmetric_space.plane import Lebesgue, plot, plot_geodesic_network


@pytest.fixture
def plane_data():
    """Provides a basic space and spatial function array for testing."""
    # Using a non-symmetric rectangle with padding to test bounds correctly
    space = Lebesgue(6, ax=0.0, bx=2.0, cx=0.5, ay=-1.0, by=1.0, cy=0.2)

    # Generate a nice 2D wave function
    def test_func(p):
        return np.sin(p[0]) * np.cos(p[1])

    u = space.project_function(test_func)
    return space, u


def test_plot_creates_axes(plane_data):
    """Tests that plot() creates a new axes if none is provided."""
    space, u = plane_data
    ax, im = plot(space, u)

    assert isinstance(ax, Axes)
    assert im is not None
    assert ax.get_aspect() in (1.0, "equal")  # Equal aspect ratio should be enforced

    plt.close(ax.figure)


def test_plot_uses_existing_axes(plane_data):
    """Tests that plot() draws onto a user-provided axes."""
    space, u = plane_data
    fig, ax_in = plt.subplots()

    ax_out, im = plot(space, u, ax=ax_in, cmap="plasma")

    assert ax_out is ax_in
    assert im.get_cmap().name == "plasma"

    plt.close(fig)


def test_plot_features(plane_data):
    """Tests kwargs like contour, symmetric, and colorbar."""
    space, u = plane_data

    ax, im = plot(
        space, u, contour=True, symmetric=True, colorbar=True, contour_lines=True
    )

    fig = ax.get_figure()

    # Check colorbar creation (a second axis should be added to the figure)
    assert len(fig.axes) > 1

    # Check symmetric color limits
    clim = im.get_clim()
    assert abs(clim[0] + clim[1]) < 1e-10

    # Check contour lines (ax.collections should contain multiple paths)
    assert len(ax.collections) > 1

    plt.close(fig)


def test_plot_full_vs_cropped(plane_data):
    """Tests that the `full` flag correctly adjusts the x/y limits."""
    space, u = plane_data

    # 1. Cropped (Default) - should strictly match bounds
    ax_cropped, _ = plot(space, u, full=False)
    xlim = ax_cropped.get_xlim()
    ylim = ax_cropped.get_ylim()

    assert np.isclose(xlim[0], space.bounds_x[0])
    assert np.isclose(xlim[1], space.bounds_x[1])
    assert np.isclose(ylim[0], space.bounds_y[0])
    assert np.isclose(ylim[1], space.bounds_y[1])
    plt.close(ax_cropped.figure)

    # 2. Full - should extend out into the padding regions
    ax_full, _ = plot(space, u, full=True)
    xlim_full = ax_full.get_xlim()
    ylim_full = ax_full.get_ylim()

    assert xlim_full[0] < space.bounds_x[0]
    assert xlim_full[1] > space.bounds_x[1]
    assert ylim_full[0] < space.bounds_y[0]
    assert ylim_full[1] > space.bounds_y[1]
    plt.close(ax_full.figure)


def test_plot_geodesic_network():
    """Tests plotting a straight-line source-receiver network on the plane."""
    paths = [((0.5, 0.5), (1.5, 1.5)), ((-0.5, 0.5), (1.0, -1.0))]
    ax = plot_geodesic_network(paths, alpha=0.5)

    assert isinstance(ax, Axes)

    # Check that at least two path segments were drawn
    assert len(ax.lines) >= 2

    # Check that markers (sources/receivers) were added via scatter collections
    assert len(ax.collections) >= 2

    plt.close(ax.figure)
