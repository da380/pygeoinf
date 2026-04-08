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
)


@pytest.fixture
def torus_data():
    """Provides a basic space and spatial function array for testing."""
    space = Lebesgue(8, radius_x=1.0, radius_y=1.0)

    # Generate a nice wave function
    def test_func(p):
        return np.sin(p[0]) * np.cos(p[1])

    u = space.project_function(test_func)
    return space, u


def test_plot_creates_axes(torus_data):
    """Tests that plot() creates a new axes if none is provided."""
    space, u = torus_data
    ax, im = plot(space, u)

    assert isinstance(ax, Axes)
    assert im is not None
    assert ax.get_aspect() in (1.0, "equal")  # Equal aspect ratio should be enforced

    plt.close(ax.figure)


def test_plot_uses_existing_axes(torus_data):
    """Tests that plot() draws onto a user-provided axes."""
    space, u = torus_data
    fig, ax_in = plt.subplots()

    ax_out, im = plot(space, u, ax=ax_in, cmap="viridis")

    assert ax_out is ax_in
    assert im.get_cmap().name == "viridis"

    plt.close(fig)


def test_plot_features(torus_data):
    """Tests kwargs like contour, symmetric, and colorbar."""
    space, u = torus_data

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
