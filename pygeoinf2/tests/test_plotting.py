"""
The rendering layer.

Not much can be asserted about a picture, so these check the two things that
can be: that dispatch reaches the right renderer, and that the arithmetic done
*before* matplotlib sees anything is right — the colour limits and the seam
that a longitude grid leaves open.
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from pygeoinf2 import plotting  # noqa: E402
from pygeoinf2.symmetric_space import Interval, Lebesgue  # noqa: E402


class TestColourLimits:
    def test_it_takes_the_range_by_default(self):
        values = np.array([-1.0, 4.0])
        assert plotting.colour_limits(values) == (-1.0, 4.0)

    def test_symmetric_limits_put_zero_in_the_middle(self):
        values = np.array([-1.0, 4.0])
        assert plotting.colour_limits(values, symmetric=True) == (-4.0, 4.0)

    def test_explicit_limits_win(self):
        values = np.array([-1.0, 4.0])
        assert plotting.colour_limits(values, vmin=0.0, vmax=1.0) == (0.0, 1.0)

    def test_symmetric_applies_to_explicit_limits_too(self):
        values = np.array([0.0])
        assert plotting.colour_limits(values, vmin=-1.0, vmax=3.0, symmetric=True) == (
            -3.0,
            3.0,
        )


class TestBoxRenderer:
    def test_one_dimension_gives_a_line(self):
        X = Lebesgue((64,), lengths=(1.0,))
        ax, line = plotting.plot(X, X.project_function(lambda t: np.sin(6.0 * t)))
        assert line.get_xdata().size == 64

    def test_a_bounded_interval_dispatches_to_the_box_renderer(self):
        X = Interval(32, lower=0.0, upper=1.0)
        ax, line = plotting.plot(X, X.project_function(lambda t: t))
        assert line is not None

    def test_two_dimensions_give_a_mesh(self, rng):
        X = Lebesgue((16, 16), lengths=(1.0, 1.0))
        ax, mappable = plotting.plot(X, X.random(rng=rng), symmetric=True)
        low, high = mappable.get_clim()
        assert low == pytest.approx(-high)

    def test_three_dimensions_are_refused(self, rng):
        X = Lebesgue((8, 8, 8))
        with pytest.raises(NotImplementedError, match="3-dimensional"):
            plotting.plot(X, X.random(rng=rng))

    def test_a_wrong_shape_is_refused(self):
        X = Lebesgue((16,), lengths=(1.0,))
        with pytest.raises(ValueError, match="has shape"):
            plotting.plot(X, np.zeros(17))

    def test_subplots_makes_a_grid(self):
        X = Lebesgue((16,), lengths=(1.0,))
        figure, axes = plotting.subplots(X, rows=2, columns=3)
        assert axes.shape == (2, 3)


class TestSphereRenderer:
    def test_it_needs_a_registered_space(self):
        from pygeoinf2.algebra.spaces import EuclideanSpace

        with pytest.raises(NotImplementedError, match="No renderer"):
            plotting.plot(EuclideanSpace(3), np.zeros(3))

    def test_a_map_is_drawn_with_a_closed_seam(self):
        """Without the wrap a blank wedge appears down the dateline."""
        pytest.importorskip("cartopy")
        from pygeoinf2.symmetric_space.sphere import Lebesgue as SphereLebesgue

        X = SphereLebesgue(12)
        field = X.project_function(lambda p: np.cos(p[1]))
        ax, mappable = plotting.plot(X, field, colorbar=True)
        # pcolormesh was handed one more longitude than the grid holds
        assert mappable.get_array().size == X.grid_shape[0] * (X.grid_shape[1] + 1)
        assert mappable.colorbar is not None

    def test_symmetric_limits_reach_the_mappable(self):
        pytest.importorskip("cartopy")
        from pygeoinf2.symmetric_space.sphere import Lebesgue as SphereLebesgue

        X = SphereLebesgue(12)
        field = X.project_function(lambda p: 1.0 + np.cos(p[0]))
        ax, mappable = plotting.plot(X, field, symmetric=True)
        low, high = mappable.get_clim()
        assert low == pytest.approx(-high)

    def test_a_wrong_shape_is_refused(self):
        pytest.importorskip("cartopy")
        from pygeoinf2.symmetric_space.sphere import Lebesgue as SphereLebesgue

        X = SphereLebesgue(12)
        with pytest.raises(ValueError, match="has shape"):
            plotting.plot(X, np.zeros((3, 3)))
