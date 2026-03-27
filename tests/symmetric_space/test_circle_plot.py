"""
Tests for the plotting functions in the circle module.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from pygeoinf.symmetric_space.circle import Lebesgue, plot, plot_error_bounds


@pytest.fixture
def circle_data():
    """Provides a basic space, function, and error bound for testing."""
    space = Lebesgue(16, radius=1.0)
    u = np.sin(space.points())
    u_bound = 0.1 * np.ones_like(u)
    return space, u, u_bound


def test_plot_creates_axes(circle_data):
    """Tests that plot() creates a new axes if none is provided."""
    space, u, _ = circle_data
    ax = plot(space, u)

    assert isinstance(ax, Axes)
    assert len(ax.lines) == 1  # Ensures a line was actually drawn
    plt.close(ax.figure)


def test_plot_uses_existing_axes(circle_data):
    """Tests that plot() draws onto a user-provided axes."""
    space, u, _ = circle_data
    fig, ax_in = plt.subplots()

    ax_out = plot(space, u, ax=ax_in, color="red")

    assert ax_out is ax_in
    assert len(ax_in.lines) == 1
    assert ax_in.lines[0].get_color() == "red"
    plt.close(fig)


def test_plot_error_bounds(circle_data):
    """Tests that plot_error_bounds() executes and creates a PolyCollection."""
    space, u, u_bound = circle_data
    ax = plot_error_bounds(space, u, u_bound)

    assert isinstance(ax, Axes)
    assert len(ax.collections) > 0  # fill_between creates a PolyCollection
    plt.close(ax.figure)
