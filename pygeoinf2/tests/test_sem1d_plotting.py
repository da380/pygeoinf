"""Drawing the spectral-element spaces: what is drawn is what the space says.

The pictures are checked through their numbers: the values handed to
matplotlib against the space's own point evaluation, which reaches them by
another route (DECISIONS.md D-107).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as pyplot  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

randomfield = pytest.importorskip("planetmodel.randomfield")
if not hasattr(randomfield, "SpectralBasis"):  # pragma: no cover
    pytest.skip("needs planetmodel 1.2 or later", allow_module_level=True)
pytest.importorskip("pyshtools")

from pygeoinf2 import plotting  # noqa: E402
from pygeoinf2.sem1d import ball as ball_module  # noqa: E402
from pygeoinf2.sem1d import interval as interval_module  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    pyplot.close("all")


@pytest.fixture
def ball():
    return ball_module.Sobolev(5, 2.0, 0.3, radial_modes=6, element_length=0.2)


@pytest.fixture
def annulus():
    return ball_module.Sobolev(
        5, 2.0, 0.2, inner_radius=0.5, radial_modes=6, element_length=0.2
    )


class TestInterval:
    def test_the_domain_alone_by_default(self, rng):
        space = interval_module.Sobolev(30, 1.0, 0.2, lower=-1.0, upper=2.0)
        x = space.random(rng=rng)
        ax, line = plotting.plot(space, x)
        assert np.array_equal(line.get_xdata(), space.interior_nodes)
        assert np.array_equal(line.get_ydata(), space.interior_values(x))
        assert ax.get_xlim() == (-1.0, 2.0)

    def test_the_padding_shaded_on_request(self, rng):
        space = interval_module.Sobolev(30, 1.0, 0.2, lower=-1.0, upper=2.0)
        ax, line = plotting.plot(space, space.random(rng=rng), padding=True)
        assert np.array_equal(line.get_xdata(), space.nodes)
        assert len(ax.patches) == 2

    def test_a_wrong_shape_is_refused(self):
        space = interval_module.Lebesgue(10)
        with pytest.raises(ValueError, match="shape"):
            plotting.plot(space, np.zeros(3))


class TestRadialProfile:
    def test_a_profile_is_a_line_against_radius(self, rng):
        from pygeoinf2.sem1d import radial as radial_module

        space = radial_module.Sobolev(20, 2.0, 0.2, inner_radius=0.5)
        x = space.random(rng=rng)
        ax, line = plotting.plot(space, x)
        assert np.array_equal(line.get_xdata(), space.interior_nodes)
        assert ax.get_xlim() == (0.5, 1.0)
        _, line = plotting.plot(space, x, padding=True)
        assert np.array_equal(line.get_xdata(), space.nodes)


class TestStations:
    def test_points_go_on_a_map_by_their_angles(self, ball):
        stations = [(1.0, 10.0, 20.0), (0.8, -40.0, 100.0), (1.0, 65.0, -150.0)]
        result = plotting.plot_points(ball, stations)
        ax = result[0] if isinstance(result, tuple) else result
        assert hasattr(ax, "projection")

    def test_a_section_shows_the_points_in_its_plane(self, ball, rng):
        first, second = (10.0, 20.0), (-40.0, 95.0)
        stations = [(1.0, *first), (0.7, *second), (1.0, 80.0, -120.0)]
        ax, _ = plotting.plot_section(
            ball, ball.random(rng=rng), through=(first, second)
        )
        scatter, drawn = plotting.plot_section_points(
            ball, stations, ax=ax, through=(first, second)
        )
        assert list(drawn) == [True, True, False]
        where = np.asarray(scatter.get_offsets())
        assert np.allclose(np.hypot(where[:, 0], where[:, 1]), [1.0, 0.7])
        # The same places the section's own values are read at.
        _, _, latitudes, longitudes = plotting.section_values(
            ball, ball.random(rng=rng), through=(first, second), angles=3600
        )
        nearest = np.argmin(np.hypot(latitudes - first[0], longitudes - first[1]))
        angle = 2.0 * np.pi * nearest / 3600
        assert np.allclose(where[0], [np.cos(angle), np.sin(angle)], atol=5e-3)

    def test_the_tolerance_is_an_angle_off_the_plane(self, ball):
        _, ax = pyplot.subplots()
        off = [(1.0, 0.0, 93.0), (1.0, 0.0, 100.0)]
        _, drawn = plotting.plot_section_points(ball, off, ax=ax, longitude=90.0)
        assert list(drawn) == [True, False]
        _, drawn = plotting.plot_section_points(
            ball, off, ax=ax, longitude=90.0, tolerance=15.0
        )
        assert list(drawn) == [True, True]


class TestShell:
    @pytest.mark.parametrize("which", ["ball", "annulus"])
    def test_a_shell_at_a_node_is_the_grid_there(self, which, request, rng):
        space = request.getfixturevalue(which)
        x = space.random(rng=rng)
        node = space.interior_radii.size // 2
        from pygeoinf2.plotting.sem1d import _on_a_shell

        shell = _on_a_shell(space, x, space.interior_radii[node])
        assert np.allclose(shell, space.interior_values(x)[node], atol=1e-11)

    def test_plot_is_the_shell_at_the_surface_on_a_map(self, ball, rng):
        x = ball.random(rng=rng)
        ax, mappable = plotting.plot(ball, x, symmetric=True)
        assert hasattr(ax, "projection")
        # The sphere's renderer repeats a column to close the seam at the
        # dateline, so the cells drawn are the grid's values and no others.
        drawn = np.unique(np.round(np.asarray(mappable.get_array()).ravel(), 9))
        surface = np.unique(np.round(ball.interior_values(x)[-1].ravel(), 9))
        assert np.array_equal(drawn, surface)
        low, high = mappable.get_clim()
        assert low == pytest.approx(-high)

    def test_subplots_give_map_axes(self, ball):
        _, axes = plotting.subplots(ball, columns=2)
        assert all(hasattr(ax, "projection") for ax in axes)

    def test_a_radius_outside_the_domain_is_refused(self, annulus, rng):
        with pytest.raises(ValueError):
            plotting.plot_shell(annulus, annulus.random(rng=rng), radius=0.2)


class TestSection:
    PLANES = [
        dict(longitude=30.0),
        dict(pole=(90.0, 0.0)),
        dict(pole=(25.0, -70.0)),
        dict(through=((10.0, 20.0), (-40.0, 95.0))),
    ]

    @pytest.mark.parametrize("plane", PLANES)
    @pytest.mark.parametrize("which", ["ball", "annulus"])
    def test_the_values_are_the_field_at_the_points_named(
        self, which, plane, request, rng
    ):
        space = request.getfixturevalue(which)
        x = space.random(rng=rng)
        values, radii, latitudes, longitudes = plotting.section_values(
            space, x, angles=24, **plane
        )
        assert values.shape == (space.interior_radii.size, 24)
        for column in range(0, 24, 5):
            points = [(r, latitudes[column], longitudes[column]) for r in radii]
            assert np.allclose(values[:, column], space.evaluate(x, points), atol=1e-10)

    def test_a_meridian_has_north_up_and_itself_on_the_right(self, ball, rng):
        _, _, latitudes, longitudes = plotting.section_values(
            ball, ball.random(rng=rng), longitude=30.0, angles=8
        )
        assert (latitudes[0], longitudes[0]) == pytest.approx((0.0, 30.0))
        assert latitudes[2] == pytest.approx(90.0)
        assert (latitudes[4], longitudes[4]) == pytest.approx((0.0, -150.0))

    def test_the_equator_has_greenwich_on_the_right(self, ball, rng):
        _, _, latitudes, longitudes = plotting.section_values(
            ball, ball.random(rng=rng), pole=(90.0, 0.0), angles=8
        )
        assert np.allclose(latitudes, 0.0, atol=1e-9)
        assert longitudes[0] == pytest.approx(0.0)
        assert longitudes[2] == pytest.approx(90.0)

    def test_two_points_are_met_in_order_anticlockwise(self, ball, rng):
        first, second = (10.0, 20.0), (10.0, 80.0)
        _, _, latitudes, longitudes = plotting.section_values(
            ball, ball.random(rng=rng), through=(first, second), angles=3600
        )
        where = [
            int(np.argmin(np.hypot(latitudes - lat, longitudes - lon)))
            for lat, lon in (first, second)
        ]
        assert 0 < (where[1] - where[0]) % 3600 < 1800

    def test_the_circle_passes_through_both_points(self, ball, rng):
        first, second = (10.0, 20.0), (-40.0, 95.0)
        _, _, latitudes, longitudes = plotting.section_values(
            ball, ball.random(rng=rng), through=(first, second), angles=3600
        )

        def unit(latitude, longitude):
            lat, lon = np.radians(latitude), np.radians(longitude)
            return np.array(
                [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
            )

        circle = np.stack([unit(a, b) for a, b in zip(latitudes, longitudes)])
        for point in (first, second):
            assert np.max(circle @ unit(*point)) > 1.0 - 1e-5

    def test_the_padding_is_drawn_on_request(self, ball, rng):
        x = ball.random(rng=rng)
        inside = plotting.section_values(ball, x, angles=12)
        padded = plotting.section_values(ball, x, angles=12, padding=True)
        assert np.array_equal(padded[1], ball.radii)
        assert np.allclose(padded[0][: inside[0].shape[0]], inside[0], atol=1e-12)

    def test_a_plane_is_named_once_and_properly(self, ball, rng):
        x = ball.random(rng=rng)
        with pytest.raises(ValueError, match="one of"):
            plotting.section_values(ball, x, longitude=0.0, pole=(90.0, 0.0))
        with pytest.raises(ValueError, match="great"):
            plotting.section_values(ball, x, through=((10.0, 20.0), (-10.0, 200.0)))
        with pytest.raises(ValueError, match="shape"):
            plotting.section_values(ball, np.zeros(4))

    @pytest.mark.parametrize("which", ["ball", "annulus"])
    def test_the_picture(self, which, request, rng):
        space = request.getfixturevalue(which)
        ax, mappable = plotting.plot_section(
            space,
            space.random(rng=rng),
            through=((10.0, 20.0), (-40.0, 95.0)),
            symmetric=True,
            colorbar_label="field",
            title="a section",
        )
        assert ax.get_aspect() == 1.0
        assert len(ax.lines) == (2 if space.inner_radius > 0.0 else 1)
        assert ax.get_title() == "a section"


class TestProfile:
    def test_the_line_is_the_field_along_the_ray(self, annulus, rng):
        x = annulus.random(rng=rng)
        ax, line = plotting.plot_profile(annulus, x, 35.0, -120.0, points=17)
        radii = np.linspace(0.5, 1.0, 17)
        assert np.allclose(line.get_xdata(), radii)
        assert np.allclose(
            line.get_ydata(), annulus.evaluate(x, [(r, 35.0, -120.0) for r in radii])
        )


class TestLayered:
    @pytest.fixture
    def profiles(self):
        from pygeoinf2.sem1d.layered import Layered

        return Layered.radial(
            [0.0, 0.35, 0.55, 1.0],
            [10, 8, 14],
            order=1.0,
            length_scale=[0.15, 0.1, 0.2],
        )

    @pytest.fixture
    def shells(self):
        from pygeoinf2.sem1d.layered import Layered

        return Layered.ball(
            [0.3, 0.6, 1.0],
            4,
            order=2.0,
            length_scale=0.2,
            radial_modes=5,
            element_length=0.1,
        )

    def test_a_profile_is_one_line_a_layer_with_a_break_at_each_jump(
        self, profiles, rng
    ):
        x = profiles.random(rng=rng)
        ax, _ = plotting.plot(profiles, x, label="a field")
        assert len(ax.lines) == 3
        assert len({line.get_color() for line in ax.lines}) == 1
        assert [line.get_label() for line in ax.lines].count("a field") == 1
        for line, layer, part in zip(ax.lines, profiles.layers, x):
            assert np.array_equal(line.get_xdata(), layer.interior_nodes)
            assert np.array_equal(line.get_ydata(), layer.interior_values(part))
        assert ax.get_xlim() == (0.0, 1.0)

    def test_a_section_draws_every_shell_on_one_scale(self, shells, rng):
        x = shells.random(rng=rng)
        ax, mappable = plotting.plot_section(shells, x, longitude=20.0, symmetric=True)
        meshes = [c for c in ax.collections if hasattr(c, "get_clim")]
        assert len(meshes) == 2
        assert len({mesh.get_clim() for mesh in meshes}) == 1
        assert len(ax.lines) == 3  # the outline of every boundary, once
        largest = max(
            np.abs(plotting.section_values(layer, part, longitude=20.0)[0]).max()
            for layer, part in zip(shells.layers, x)
        )
        assert mappable.get_clim()[1] == pytest.approx(largest)
        with pytest.raises(ValueError, match="padding"):
            plotting.plot_section(shells, x, padding=True)

    def test_a_profile_along_a_ray_breaks_at_the_interface(self, shells, rng):
        x = shells.random(rng=rng)
        ax, _ = plotting.plot_profile(shells, x, 10.0, 20.0)
        assert len(ax.lines) == 2
        below, above = ax.lines
        assert below.get_xdata()[-1] == pytest.approx(0.6)
        assert above.get_xdata()[0] == pytest.approx(0.6)
        point = [(0.6, 10.0, 20.0)]
        assert below.get_ydata()[-1] == pytest.approx(
            shells.evaluate(x, point, side="below")[0]
        )
        assert above.get_ydata()[0] == pytest.approx(
            shells.evaluate(x, point, side="above")[0]
        )

    def test_a_shell_at_an_interface_names_its_side(self, shells, rng):
        x = shells.random(rng=rng)
        ax, _ = plotting.plot(shells, x)
        assert hasattr(ax, "projection")
        plotting.plot_shell(shells, x, radius=0.45)
        with pytest.raises(ValueError, match="interface"):
            plotting.plot_shell(shells, x, radius=0.6)
        _, below = plotting.plot_shell(shells, x, radius=0.6, side="below")
        _, above = plotting.plot_shell(shells, x, radius=0.6, side="above")
        assert not np.allclose(below.get_array(), above.get_array())

    def test_stations_and_the_wrong_geometry(self, shells, profiles, rng):
        stations = [(1.0, 10.0, 20.0), (0.45, -40.0, 100.0)]
        plotting.plot_points(shells, stations)
        ax, _ = plotting.plot_section(shells, shells.random(rng=rng), longitude=20.0)
        _, drawn = plotting.plot_section_points(shells, stations, ax=ax, longitude=20.0)
        assert list(drawn) == [True, False]
        with pytest.raises(TypeError, match="balls"):
            plotting.plot_section(profiles, profiles.random(rng=rng))
