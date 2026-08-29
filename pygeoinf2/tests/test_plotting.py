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
        """Without the wrap a blank wedge appears down the dateline.

        The mesh is given explicit cell edges rather than centres, and they run
        from -180 to +180 with the antimeridian cell drawn as its two halves.
        That is what closes the seam *and* keeps every cell on one side of
        cartopy's cut: a cell straddling the cut sends the whole mesh down a
        per-polygon path, which cost 1.16 s a map at lmax 128 against 30 ms.
        """
        pytest.importorskip("cartopy")
        from pygeoinf2.symmetric_space.sphere import Lebesgue as SphereLebesgue

        X = SphereLebesgue(12)
        rows, columns = X.grid_shape
        field = X.project_function(lambda p: np.cos(p[1]))
        ax, mappable = plotting.plot(X, field, colorbar=True)
        assert mappable.get_array().size == rows * (columns + 1)
        corners = mappable.get_coordinates()
        assert corners.shape == (rows + 1, columns + 2, 2)
        longitudes = corners[0, :, 0]
        assert longitudes[0] == pytest.approx(-180.0)
        assert longitudes[-1] == pytest.approx(180.0)
        assert np.all(np.diff(longitudes) >= 0.0)
        # Latitude edges stay on the sphere: an edge past the pole is not a
        # point any projection can place.
        latitudes = corners[:, 0, 1]
        assert latitudes.max() == pytest.approx(90.0)
        assert latitudes.min() >= -90.0
        assert mappable.colorbar is not None

    def test_the_seam_column_is_the_one_it_wraps(self):
        """The two half-cells at the edges of the map are the same column of
        the grid, so the picture is periodic across the join rather than merely
        continuous-looking."""
        pytest.importorskip("cartopy")
        from pygeoinf2.symmetric_space.sphere import Lebesgue as SphereLebesgue

        X = SphereLebesgue(8)
        field = X.project_function(lambda p: np.cos(np.radians(p[1])))
        ax, mappable = plotting.plot(X, field)
        drawn = mappable.get_array().reshape(X.grid_shape[0], -1)
        assert np.allclose(drawn[:, 0], drawn[:, -1])
        # and it is the column at longitude 180, the one the roll brought to
        # the front.
        values = X.grid_values(field)
        assert np.allclose(drawn[:, 0], values[:, X.grid_shape[1] // 2])

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


class TestDistributions:
    """Marginals and corner plots.

    Not much can be asserted about a picture, so the tests are about the
    arithmetic that happens before matplotlib sees anything: whether the
    moments are the right ones, whether the two routes to them agree, and
    whether the frame is opened wide enough to contain what it is meant to
    show.
    """

    @staticmethod
    def gaussian(space, rng, expectation=True):
        from pygeoinf2.algebra.operators import LinearOperator
        from pygeoinf2.numerics.functional_calculus import operator_sqrt
        from pygeoinf2.probability.gaussian import GaussianMeasure
        from pygeoinf2.traits import Traits

        root = rng.normal(size=(space.dim, space.dim))
        covariance = LinearOperator.from_matrix(
            space,
            space,
            root @ root.T + space.dim * np.identity(space.dim),
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
            form="galerkin",
        )
        return GaussianMeasure(
            space,
            covariance=covariance,
            covariance_factor=operator_sqrt(covariance),
            expectation=space.random(rng=rng) if expectation else None,
        )

    @pytest.fixture(params=["euclidean", "weighted", "dense-metric"])
    def measure(self, request, rng):
        from pygeoinf2.algebra.spaces import EuclideanSpace

        from .conftest import make_dense_metric_space, make_weighted_space

        space = {
            "euclidean": lambda: EuclideanSpace(4),
            "weighted": make_weighted_space,
            "dense-metric": make_dense_metric_space,
        }[request.param]()
        return space, self.gaussian(space, rng)

    def test_the_moments_are_the_components_covariance(self, measure):
        """``G^-1 C_gal G^-1``, not the covariance operator's component matrix
        — a different thing by 75% on the weighted space here, and the reason
        this asks the measure rather than reading a diagonal."""
        space, gaussian = measure
        mean, covariance, draws = plotting.moments(gaussian)
        assert draws is None
        gram = space.gram_matrix()
        galerkin = gaussian.covariance.matrix(form="galerkin")
        expected = np.linalg.solve(gram, np.linalg.solve(gram, galerkin).T)
        assert np.allclose(covariance, expected, atol=1e-10)
        assert np.allclose(mean, space.to_components(gaussian.expectation))

    @pytest.mark.slow
    def test_the_sampled_route_agrees_with_the_exact_one(self, measure):
        """On one measure, so the two must give the same answer. A linear
        push-forward of a Gaussian is Gaussian, but wrapping the operator as a
        general one hides that and forces the sampling branch."""
        from pygeoinf2.algebra.operators import LinearOperator, Operator
        from pygeoinf2.algebra.spaces import EuclideanSpace

        space, gaussian = measure
        target = EuclideanSpace(3)
        rng = np.random.default_rng(4)
        forward = LinearOperator.from_matrix(
            space, target, rng.normal(size=(3, space.dim)), form="galerkin"
        )
        exact_mean, exact_covariance, _ = plotting.moments(
            gaussian.push_forward(forward)
        )
        hidden = gaussian.push_forward(
            Operator.from_callables(space, target, lambda x: forward(x))
        )
        samples = 60000
        mean, covariance, draws = plotting.moments(
            hidden, samples=samples, rng=np.random.default_rng(5)
        )
        assert draws.shape == (samples, 3)
        # A covariance from n draws is good to about 1/sqrt(n), which here is
        # 0.4%; the tolerance is that, not a number chosen to pass.
        scale = np.abs(exact_covariance).max()
        tolerance = 10.0 / np.sqrt(samples)
        assert np.abs(mean - exact_mean).max() < tolerance * np.sqrt(scale)
        assert np.abs(covariance - exact_covariance).max() < tolerance * scale

    @pytest.mark.slow
    def test_a_non_gaussian_measure_is_drawn_from_draws(self, measure):
        from pygeoinf2.algebra.operators import Operator
        from pygeoinf2.algebra.spaces import EuclideanSpace

        space, gaussian = measure
        target = EuclideanSpace(3)

        def squash(x):
            c = space.to_components(x)
            return target.from_components(
                np.array([np.tanh(c[0]), c[1] ** 2, c[2] - c[0]])
            )

        pushed = gaussian.push_forward(Operator.from_callables(space, target, squash))
        assert pushed.can_sample
        axes = plotting.plot_corner(pushed, samples=4000, rng=np.random.default_rng(6))
        assert axes.shape == (3, 3)

    def test_the_frame_opens_to_contain_the_truth(self, measure):
        """A truth marker outside the frame says the fit is good by not being
        visible, so the window widens until it is inside."""
        space, gaussian = measure
        mean, covariance, _ = plotting.moments(gaussian)
        deviation = np.sqrt(np.diag(covariance))
        far = mean + 9.0 * deviation
        axes = plotting.plot_corner(gaussian, truth=far, width=3.0)
        for index in range(mean.size):
            lower, upper = axes[index, index].get_xlim()
            assert lower <= far[index] <= upper

    def test_the_corner_is_lower_triangular(self, measure):
        """Which is the point of the shape: every pair once, and the empty
        half says so."""
        space, gaussian = measure
        size = space.dim
        axes = plotting.plot_corner(gaussian)
        for row in range(size):
            for column in range(size):
                # `axison`, not `get_frame_on`: set_axis_off turns off the
                # axis, which leaves the frame flag alone.
                assert axes[row, column].axison == (column <= row), (row, column)

    def test_densities_put_priors_on_their_own_axis(self, measure, rng):
        space, gaussian = measure
        prior = self.gaussian(space, rng)
        alone = plotting.plot_densities(gaussian)
        assert not isinstance(alone, tuple)
        import matplotlib.pyplot as plt

        plt.close("all")
        both = plotting.plot_densities(gaussian, prior=prior)
        assert isinstance(both, tuple) and len(both) == 2
        plt.close("all")

    def test_what_cannot_be_drawn_is_refused(self, measure):
        space, gaussian = measure
        from pygeoinf2.algebra.spaces import EuclideanSpace

        with pytest.raises(ValueError, match="at least two components"):
            plotting.plot_corner(
                self.gaussian(EuclideanSpace(1), np.random.default_rng(1))
            )
        mean, _, _ = plotting.moments(gaussian)
        with pytest.raises(ValueError, match="true values for"):
            plotting.plot_corner(gaussian, truth=mean[:-1])
        with pytest.raises(IndexError, match="out of range"):
            plotting.plot_densities(gaussian, index=space.dim)


class TestDensityResolution:
    """The two defects in how a marginal is drawn."""

    def test_a_narrow_posterior_is_resolved_beside_a_wide_prior(self):
        """The case the twin axis exists for.

        A single grid over the union of the two windows resolves the prior and
        aliases the posterior: with a ratio of 1000 the spacing was about six
        posterior standard deviations, so the peak was missed and the drawn
        curve understated it by 16%. Raising the count does not fix it — the
        ratio is what defeats a shared grid — so each curve gets its own.
        """
        import matplotlib.pyplot as plt

        from pygeoinf2.algebra.spaces import EuclideanSpace
        from pygeoinf2.probability.gaussian import GaussianMeasure

        space = EuclideanSpace(1)
        for ratio in (10.0, 1000.0):
            deviation = 1.0 / ratio
            _, axis = plt.subplots()
            plotting.plot_densities(
                GaussianMeasure.from_standard_deviation(space, deviation),
                prior=GaussianMeasure.from_standard_deviation(space, 1.0),
                ax=axis,
            )
            peak = axis.get_lines()[0].get_ydata().max()
            exact = 1.0 / (deviation * np.sqrt(2.0 * np.pi))
            assert peak == pytest.approx(exact, rel=1e-3)
            plt.close("all")

    def test_the_corner_fill_means_the_same_thing_in_both_branches(self, rng):
        """``fill=True`` shaded a Mahalanobis distance in the Gaussian branch
        and a density in the sampled one, so one figure could be darkest at the
        mean and the other darkest away from it. Both now shade density."""
        import matplotlib.pyplot as plt

        from pygeoinf2.algebra.spaces import EuclideanSpace
        from pygeoinf2.probability.gaussian import GaussianMeasure

        space = EuclideanSpace(2)
        measure = GaussianMeasure.from_covariance_matrix(
            space, np.array([[1.0, 0.4], [0.4, 0.6]])
        )

        def centre_minus_corner(**kwargs):
            axes = plotting.plot_corner(measure, fill=True, **kwargs)
            panel = axes[1, 0]
            figure = panel.figure
            figure.canvas.draw()
            image = np.asarray(figure.canvas.buffer_rgba()).astype(float)
            box = panel.get_window_extent()
            height = image.shape[0]

            def luminance(fx, fy):
                px = int(box.x0 + fx * box.width)
                py = int(height - (box.y0 + fy * box.height))
                return image[py - 2 : py + 3, px - 2 : px + 3, :3].mean()

            value = luminance(0.5, 0.5) - luminance(0.06, 0.06)
            plt.close("all")
            return value

        exact = centre_minus_corner()
        sampled = centre_minus_corner(samples=4000, rng=rng)
        assert exact != pytest.approx(0.0, abs=5.0)
        assert (exact > 0.0) == (sampled > 0.0)


class TestPyslfpNeeds:
    """The keywords the review found every pyslfp call passing, and v2 not
    accepting. A caller who cannot title a plot has to reach past the return
    value to do it, and a corner plot with no legend has three unlabelled
    marks on it."""

    @pytest.fixture
    def measure(self, rng):
        import pygeoinf2 as gi
        from pygeoinf2.algebra.spaces import EuclideanSpace

        space = EuclideanSpace(3)
        root = rng.standard_normal((3, 3))
        return (
            gi.GaussianMeasure.from_covariance_matrix(
                space, root @ root.T + np.eye(3), form="components"
            ),
            gi.GaussianMeasure.from_standard_deviation(space, 5.0),
        )

    def test_the_corner_plot_takes_a_title(self, measure):
        from pygeoinf2.plotting.distributions import plot_corner

        posterior, prior = measure
        axes = plot_corner(posterior, prior=prior, title="a corner plot")
        assert axes[0, 0].get_figure()._suptitle.get_text() == "a corner plot"

    def test_the_corner_plot_has_a_legend(self, measure):
        """In the empty upper triangle, which costs no space."""
        from pygeoinf2.plotting.distributions import plot_corner

        posterior, prior = measure
        axes = plot_corner(
            posterior, prior=prior, truth=np.array([0.1, -0.2, 0.3])
        )
        legend = axes[0, 2].get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == [
            "posterior",
            "prior",
            "truth",
        ]

    def test_the_legend_names_only_what_was_drawn(self, measure):
        from pygeoinf2.plotting.distributions import plot_corner

        posterior, _ = measure
        axes = plot_corner(posterior)
        assert [
            text.get_text() for text in axes[0, 2].get_legend().get_texts()
        ] == ["posterior"]

    def test_it_can_be_turned_off(self, measure):
        from pygeoinf2.plotting.distributions import plot_corner

        posterior, _ = measure
        axes = plot_corner(posterior, legend=False)
        assert axes[0, 2].get_legend() is None

    def test_the_density_plot_takes_a_title(self, measure):
        from pygeoinf2.plotting.distributions import plot_densities

        posterior, prior = measure
        drawn = plot_densities(posterior, prior=prior, title="a density")
        axis = drawn[0] if isinstance(drawn, tuple) else drawn
        assert axis.get_title() == "a density"


class TestSphereMapOptions:
    """The field-plot keywords v1 had and v2 dropped."""

    @pytest.fixture
    def field(self):
        pytest.importorskip("cartopy")
        pytest.importorskip("pyshtools")
        from pygeoinf2.symmetric_space.sphere import Lebesgue

        space = Lebesgue(16)
        return space, space.project_function(
            lambda point: np.sin(np.radians(point[0]))
        )

    def test_a_map_extent_replaces_the_global_view(self, field):
        """Not both: ``set_global`` would undo the extent that was asked for,
        which is why this is a branch rather than an extra call."""
        from pygeoinf2.plotting import plot

        space, values = field
        axis, _ = plot(space, values, map_extent=(-30.0, 40.0, 20.0, 70.0))
        extent = axis.get_extent()
        whole, _ = plot(space, values)
        assert extent[1] - extent[0] < (whole.get_extent()[1] - whole.get_extent()[0])

    def test_contours_and_lines(self, field):
        from pygeoinf2.plotting import plot

        space, values = field
        axis, mappable = plot(
            space, values, contour=True, contour_lines=True, levels=8
        )
        assert hasattr(axis, "contour_set")
        assert mappable is not None

    def test_gridline_intervals_become_locators(self, field):
        """cartopy wants tick arrays; a caller thinks in intervals."""
        from pygeoinf2.plotting import plot

        space, values = field
        axis, _ = plot(
            space,
            values,
            gridlines=True,
            gridlines_kwargs={"lat_interval": 30.0, "lon_interval": 60.0},
        )
        assert axis.gridliner is not None

    def test_the_colourbar_takes_its_own_options(self, field):
        from pygeoinf2.plotting import plot

        space, values = field
        axis, mappable = plot(
            space,
            values,
            colorbar=True,
            colorbar_kwargs={"orientation": "horizontal"},
        )
        assert mappable.colorbar.orientation == "horizontal"

    def test_the_defaults_are_v1s(self, field):
        """v2 had flipped four of them with no reason recorded: the colour map
        from RdBu to viridis, the colourbar on, the graticule off, and the
        projection from PlateCarree to Robinson. A signed field on a
        sequential map reads as though it had a sign it does not have, so this
        is not only a matter of taste."""
        from pygeoinf2.plotting import plot, subplots

        space, values = field
        axis, mappable = plot(space, values)
        assert mappable.get_cmap().name == "RdBu"
        assert mappable.colorbar is None
        assert axis.gridliner is not None

        figure, fresh = subplots(space)
        import cartopy.crs as ccrs

        assert isinstance(fresh.projection, ccrs.PlateCarree)

    def test_a_label_asks_for_the_bar_it_goes_on(self, field):
        """The bar is off by default, so a label with no bar would be a
        silently dropped argument. An explicit ``colorbar=False`` still wins."""
        from pygeoinf2.plotting import plot

        space, values = field
        _, labelled = plot(space, values, colorbar_label="metres")
        assert labelled.colorbar is not None
        assert labelled.colorbar.ax.get_ylabel() == "metres"
        _, refused = plot(space, values, colorbar=False, colorbar_label="metres")
        assert refused.colorbar is None

    def test_a_title(self, field):
        from pygeoinf2.plotting import plot

        space, values = field
        axis, _ = plot(space, values, title="a map")
        assert axis.get_title() == "a map"


class TestErrorBounds:
    """A bound above and below is what an inference produces; a pair of lines
    reads as two estimates rather than one with an uncertainty."""

    @pytest.fixture
    def setting(self):
        from pygeoinf2.symmetric_space.circle import Lebesgue

        space = Lebesgue(64)
        middle = space.project_function(np.sin)
        return space, middle - 0.3, middle, middle + 0.3

    def test_it_shades_the_band(self, setting):
        from pygeoinf2.plotting import plot_error_bounds

        space, low, middle, high = setting
        axis, band = plot_error_bounds(space, low, high, centre=middle)
        assert band is not None
        assert hasattr(axis, "centre_line")

    def test_crossed_bounds_are_drawn_not_refused(self, setting):
        """A band that crosses over is what an inconsistent bound looks like,
        and refusing to draw it would hide the case worth seeing."""
        from pygeoinf2.plotting import plot_error_bounds

        space, low, _, high = setting
        axis, band = plot_error_bounds(space, high, low)
        assert band is not None

    def test_more_than_one_dimension_is_refused(self):
        from pygeoinf2.plotting import plot_error_bounds
        from pygeoinf2.symmetric_space.torus import Lebesgue

        space = Lebesgue((8, 8))
        with pytest.raises(ValueError, match="one-dimensional"):
            plot_error_bounds(space, np.zeros((8, 8)), np.ones((8, 8)))

    def test_a_wrong_shape_is_refused(self, setting):
        from pygeoinf2.plotting import plot_error_bounds

        space, low, _, high = setting
        with pytest.raises(ValueError, match="lower bound"):
            plot_error_bounds(space, np.zeros(5), high)
