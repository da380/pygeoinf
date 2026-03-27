"""
Tests for the plotting functions in the line module.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from pygeoinf.symmetric_space.line import Lebesgue, plot, plot_error_bounds


@pytest.fixture
def line_data():
    """Provides a basic space, function, and error bound for testing."""
    space = Lebesgue(16, a=0.0, b=1.0, c=0.1)
    u = np.cos(2 * np.pi * space.points())
    u_bound = 0.2 * np.ones_like(u)
    return space, u, u_bound


def test_plot_creates_axes(line_data):
    """Tests that plot() creates a new axes and respects the 'full' argument."""
    space, u, _ = line_data

    # Test with full=False (should crop x-limits to [a, b])
    ax1 = plot(space, u, full=False)
    assert isinstance(ax1, Axes)
    assert len(ax1.lines) == 1
    assert ax1.get_xlim() == (space.a, space.b)
    plt.close(ax1.figure)

    # Test with full=True (should let Matplotlib auto-scale to include padding)
    ax2 = plot(space, u, full=True)
    xlim_full = ax2.get_xlim()
    assert xlim_full[0] <= space.a - space.c
    assert xlim_full[1] >= space.b + space.c
    plt.close(ax2.figure)


def test_plot_uses_existing_axes(line_data):
    """Tests that plot() draws onto a user-provided axes."""
    space, u, _ = line_data
    fig, ax_in = plt.subplots()

    ax_out = plot(space, u, ax=ax_in, linestyle="--")

    assert ax_out is ax_in
    assert len(ax_in.lines) == 1
    assert ax_in.lines[0].get_linestyle() == "--"
    plt.close(fig)


def test_plot_error_bounds(line_data):
    """Tests that plot_error_bounds() executes and creates a PolyCollection."""
    space, u, u_bound = line_data
    ax = plot_error_bounds(space, u, u_bound)

    assert isinstance(ax, Axes)
    assert len(ax.collections) > 0
    plt.close(ax.figure)
