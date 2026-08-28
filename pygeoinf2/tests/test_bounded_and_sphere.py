"""Bounded domains and the sphere."""

import numpy as np
import pytest

from pygeoinf2 import Traits
from pygeoinf2.symmetric_space import Box, Interval, lift_formal_adjoint
from pygeoinf2.symmetric_space.box import Box as BoxClass
from pygeoinf2.testing import (
    check_coordinates,
    check_measure,
    check_operator,
    check_space,
    check_traits,
    check_white_noise,
)

pyshtools = pytest.importorskip("pyshtools")

from pygeoinf2.symmetric_space import sphere as sphere_module  # noqa: E402


class TestBoundedDomains:
    """v1 builds the line from the circle and the plane from the torus."""

    def test_the_padding_enlarges_the_periodic_domain(self):
        space = Interval(32, lower=0.0, upper=1.0, padding=0.25)
        assert space.domain_volume == pytest.approx(1.0)
        assert space.volume == pytest.approx(1.5)

    def test_the_default_padding_is_a_tenth(self):
        space = Interval(32, lower=0.0, upper=2.0)
        assert space.padding == (0.2,)

    @pytest.mark.parametrize(
        "space_factory",
        [
            lambda: Interval(32),
            lambda: Interval(32, order=2.0, length_scale=0.1),
            lambda: Box((16, 12), bounds=((0.0, 1.0), (0.0, 2.0))),
        ],
    )
    def test_the_space_axioms_hold(self, space_factory, rng):
        space = space_factory()
        check_space(space, rng=rng, rebuild=space_factory)
        check_coordinates(space, rng=rng)

    def test_a_function_rolls_off_across_the_padding(self, rng):
        """Not a step. A hard cutoff puts a discontinuity into a field the
        space represents by a truncated Fourier series, and a step rings: the
        integral of the constant one along a path spanning most of the domain
        came out 5% low. The padding now carries the boundary value tapered to
        zero."""
        space = Interval(32, lower=0.0, upper=1.0)
        field = space.project_function(lambda t: 1.0 + t**2)
        outside = field[~space.interior_mask]

        assert np.all(field[space.interior_mask] > 0.0)
        # Zero at the far edge of the padding, and monotone in between rather
        # than dropping to zero at once.
        assert np.all(outside >= 0.0)
        assert outside.min() == pytest.approx(0.0, abs=1e-12)
        assert outside.max() > 0.5

    def test_the_hard_cutoff_is_still_available(self):
        """Right when the function genuinely vanishes at the boundary, since
        then there is nothing to ring."""
        space = Interval(32, lower=0.0, upper=1.0)
        field = space.project_function(lambda t: 1.0 + t**2, taper=False)
        assert np.all(field[~space.interior_mask] == 0.0)

    def test_the_taper_removes_most_of_the_ringing(self):
        """Measured, because the point of it is a number.

        Measured *pointwise*, which is where Gibbs ringing actually lives: a
        path integral averages the overshoot and undershoot against each
        other, so it understates the ringing by an order of magnitude and is
        the wrong instrument for this. The earlier version of this test used
        one, and the large error it was reading turned out to be a coordinate
        bug in the NUFFT route on a padded box rather than ringing at all.
        With that fixed: 0.0508 against 0.0002, a factor of 250.
        """
        space = Interval(128, lower=0.0, upper=1.0)
        interior = [np.array([x]) for x in np.linspace(0.02, 0.98, 200)]

        tapered = space.project_function(lambda t: 1.0)
        hard = space.project_function(lambda t: 1.0, taper=False)

        assert np.abs(space.evaluate(hard, interior) - 1.0).max() > 0.04
        assert np.abs(space.evaluate(tapered, interior) - 1.0).max() < 0.001

        # And it still gets the integral right, to a two-hundredth of what
        # the hard cutoff manages: 8.1e-6 against 1.7e-3.
        start, end = np.array([0.05]), np.array([0.95])
        integral = space.path_integral_operator([(start, end)], count=40)
        exact = 0.9
        assert abs(integral(tapered)[0] - exact) < 1e-4
        assert abs(integral(hard)[0] - exact) > 1e-3

    def test_the_function_is_never_called_outside_the_domain(self):
        """It need not be defined there, which is why the padding is zeroed."""
        space = Interval(32, lower=0.0, upper=1.0)

        def only_inside(t):
            assert -1e-9 <= t <= 1.0 + 1e-9, f"called at {t}"
            return t

        space.project_function(only_inside)

    def test_evaluation_is_exact_at_grid_points(self, rng):
        space = Interval(64, lower=0.0, upper=1.0)
        field = space.project_function(lambda t: np.sin(np.pi * t))
        components = space.to_components(field)
        index = np.flatnonzero(space.interior_mask)[7]
        point = np.array([space.grid_axes[0][index]])
        assert components @ space.basis_at(point) == pytest.approx(
            field[index], abs=1e-10
        )

    def test_random_points_stay_inside_the_domain(self, rng):
        space = Interval(32, lower=2.0, upper=5.0)
        for _ in range(50):
            point = space.random_point(rng=rng)
            assert 2.0 <= point[0] <= 5.0

    def test_the_support_projection_is_an_orthogonal_projector(self, rng):
        space = Interval(32, lower=0.0, upper=1.0)
        projector = space.support_projection()
        assert Traits.SELF_ADJOINT & projector.traits
        assert Traits.IDEMPOTENT & projector.traits
        check_operator(projector, rng=rng)
        check_traits(projector, rng=rng)

    def test_on_a_sobolev_space_it_lifts_rather_than_refusing(self, rng):
        """Multiplying by a discontinuous mask does not commute with the
        metric, so it is not self-adjoint there and must not say it is. It
        used to raise and tell the caller to lift it by hand; doing the lift
        is the same operator without the extra step."""
        space = Interval(32, lower=0.0, upper=1.0, order=2.0, length_scale=0.1)
        projector = space.support_projection()

        assert projector.traits == Traits.NONE
        check_operator(projector, rng=rng)

    def test_the_lift_is_the_one_a_caller_would_have_written(self, rng):
        lebesgue = Interval(32, lower=0.0, upper=1.0)
        sobolev = Interval(32, lower=0.0, upper=1.0, order=2.0, length_scale=0.1)
        by_hand = lift_formal_adjoint(lebesgue.support_projection(), sobolev)

        field = sobolev.random(rng=rng)
        assert sobolev.norm(
            sobolev.subtract(sobolev.support_projection()(field), by_hand(field))
        ) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            (dict(bounds=((1.0, 0.0),)), "lower end first"),
            (dict(bounds=((0.0, 1.0), (0.0, 1.0))), "bounds for"),
            (dict(bounds=((0.0, 1.0),), padding=(-1.0,)), "non-negative"),
            (dict(bounds=((0.0, 1.0),), padding=(0.1, 0.2)), "paddings for"),
        ],
    )
    def test_bad_arguments_are_refused(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            BoxClass((8,), **kwargs)

    def test_a_two_dimensional_domain(self, rng):
        space = Box((16, 12), bounds=((0.0, 1.0), (-1.0, 1.0)), padding=0.1)
        assert space.spatial_dimension == 2
        assert space.domain_volume == pytest.approx(2.0)
        field = space.project_function(lambda p: p[0] + p[1])
        # The taper is a product over axes, so a corner of the padding is
        # damped by both and reaches zero at the outer edge.
        assert np.all(np.isfinite(field))
        assert space.grid_values(space.project_function(lambda p: 1.0))[0, 0] == (
            pytest.approx(0.0, abs=1e-12)
        )


class TestSphere:
    def test_the_dimension_is_the_number_of_harmonics(self):
        assert sphere_module.Lebesgue(8).dim == 81

    @pytest.mark.parametrize("lmax", [4, 8])
    def test_the_component_round_trip_is_exact(self, lmax, rng):
        space = sphere_module.Lebesgue(lmax)
        components = rng.normal(size=space.dim)
        field = space.from_components(components)
        assert np.allclose(space.to_components(field), components, atol=1e-12)

    def test_the_space_axioms_hold(self, rng):
        check_space(
            sphere_module.Lebesgue(6),
            rng=rng,
            rebuild=lambda: sphere_module.Lebesgue(6),
        )
        check_coordinates(sphere_module.Lebesgue(6), rng=rng)

    def test_the_sobolev_axioms_hold(self, rng):
        space = sphere_module.Sobolev(6, 2.0, 0.3)
        check_space(space, rng=rng, rebuild=lambda: sphere_module.Sobolev(6, 2.0, 0.3))
        check_coordinates(space, rng=rng)

    def test_basis_at_reproduces_the_field(self, rng):
        """Which pins the harmonic conventions: normalisation and phase."""
        space = sphere_module.Lebesgue(8)
        components = rng.normal(size=space.dim)
        field = space.from_components(components)
        for _ in range(6):
            i = int(rng.integers(0, space.grid_shape[0]))
            j = int(rng.integers(0, space.grid_shape[1]))
            point = space.to_latitude_degrees(
                np.array([space.colatitudes[i], space.longitudes[j]])
            )[0]
            assert components @ space.basis_at(point) == pytest.approx(
                space.grid_values(field)[i, j], abs=1e-10
            )

    def test_the_radius_enters_the_norm(self, rng):
        """The basis is orthonormal on the sphere of the given radius."""
        unit = sphere_module.Lebesgue(6, radius=1.0)
        larger = sphere_module.Lebesgue(6, radius=2.0)
        components = rng.normal(size=unit.dim)
        # Same components, same norm: the basis absorbs the radius.
        assert unit.squared_norm(unit.from_components(components)) == pytest.approx(
            larger.squared_norm(larger.from_components(components))
        )
        # But the same *field* has a larger norm on the larger sphere.
        assert larger.area == pytest.approx(4.0 * unit.area)

    def test_the_laplacian_has_the_analytic_spectrum(self):
        space = sphere_module.Lebesgue(6, radius=3.0)
        degrees = np.repeat(np.arange(7), 2 * np.arange(7) + 1)
        assert np.allclose(space.laplacian_eigenvalues, degrees * (degrees + 1.0) / 9.0)

    def test_it_annihilates_constants(self, rng):
        space = sphere_module.Lebesgue(6)
        constant = space.from_components(
            np.concatenate([[1.0], np.zeros(space.dim - 1)])
        )
        assert space.norm(space.laplacian(constant)) < 1e-10

    def test_white_noise_is_white(self, rng):
        check_white_noise(
            sphere_module.Sobolev(4, 2.0, 0.3), rng=rng, samples=6000, rtol=0.14
        )

    def test_an_invariant_measure(self, rng):
        space = sphere_module.Sobolev(4, 2.0, 0.3)
        check_measure(space.sobolev_measure(2.0, 0.3), rng=rng, samples=4000, rtol=0.15)

    def test_point_evaluation(self, rng):
        space = sphere_module.Sobolev(12, 2.0, 0.2)
        points = space.random_points(4, rng=rng)
        operator = space.point_evaluation_operator(points)
        check_operator(operator, rng=rng)

        # sin(latitude) is the degree-one zonal harmonic, so it is represented
        # exactly and the comparison is about the convention, not truncation.
        field = space.project_function(lambda p: np.sin(np.radians(p[0])))
        assert np.allclose(
            operator(field),
            [np.sin(np.radians(p[0])) for p in points],
            atol=1e-8,
        )

    def test_the_adjoint_returns_dirac_representers(self, rng):
        space = sphere_module.Sobolev(8, 2.0, 0.2)
        point = space.random_point(rng=rng)
        operator = space.point_evaluation_operator([point, space.random_point(rng=rng)])
        assert np.allclose(
            space.to_components(operator.adjoint(np.array([1.0, 0.0]))),
            space.to_components(space.dirac(point).representer),
        )

    def test_random_points_are_uniform_over_the_area(self, rng):
        """Uniform in sin(latitude), which is the classic thing to get wrong."""
        space = sphere_module.Lebesgue(4)
        sines = np.array(
            [np.sin(np.radians(space.random_point(rng=rng)[0])) for _ in range(4000)]
        )
        assert abs(sines.mean()) < 0.06
        # A uniform distribution on [-1, 1] has variance 1/3.
        assert sines.var() == pytest.approx(1.0 / 3.0, rel=0.1)

    def test_points_are_latitude_and_longitude_in_degrees(self, rng):
        """D-2. The convention every catalogue, every v1 script and pyshtools
        itself already use -- and the one whose absence would misplace a
        station rather than raise."""
        space = sphere_module.Lebesgue(8)
        assert space.reference_point == pytest.approx([90.0, 0.0])
        for point in space.random_points(200, rng=rng):
            assert -90.0 <= point[0] <= 90.0
            assert -180.0 <= point[1] <= 180.0

        # sin(latitude) is exactly the degree-one zonal harmonic, so it is
        # +1 at the north pole and -1 at the south with nothing to truncate.
        field = space.project_function(lambda p: np.sin(np.radians(p[0])))
        assert space.evaluate(field, [np.array([90.0, 0.0])])[0] == pytest.approx(
            1.0, abs=1e-8
        )
        assert space.evaluate(field, [np.array([-90.0, 0.0])])[0] == pytest.approx(
            -1.0, abs=1e-8
        )

    @pytest.mark.parametrize(
        "lmax, kwargs, message",
        [
            (-1, {}, "non-negative"),
            (4, dict(radius=0.0), "radius"),
            (4, dict(length_scale=0.0), "length_scale"),
        ],
    )
    def test_bad_arguments_are_refused(self, lmax, kwargs, message):
        with pytest.raises(ValueError, match=message):
            sphere_module.Sphere(lmax, **kwargs)

    def test_a_field_of_the_wrong_shape_is_refused(self):
        space = sphere_module.Lebesgue(4)
        with pytest.raises(ValueError, match="shape"):
            space.to_components(np.zeros((3, 3)))

    def test_a_point_needs_two_coordinates(self):
        with pytest.raises(ValueError, match="latitude"):
            sphere_module.Lebesgue(4).basis_at(np.array([0.1]))


class TestSphereVectorsAreGrids:
    """D-1: a field is an ``SHGrid``, not a bare array."""

    def test_a_field_is_an_shgrid(self, rng):
        space = sphere_module.Lebesgue(8)
        field = space.random(rng=rng)
        assert isinstance(field, pyshtools.SHGrid)
        # And so it can do what pyshtools fields do.
        assert field.expand() is not None
        assert space.grid_values(field).shape == space.grid_shape

    @pytest.mark.parametrize("sampling", [1, 2])
    def test_both_samplings_represent_the_same_functions(self, sampling, rng):
        """The grid is a choice about storage, not about the space: the
        components are the same numbers either way."""
        square = sphere_module.Sobolev(8, 2.0, 0.2, sampling=1)
        wide = sphere_module.Sobolev(8, 2.0, 0.2, sampling=2)
        assert square.dim == wide.dim
        assert square.grid_shape == (18, 18)
        assert wide.grid_shape == (18, 36)

        components = rng.normal(size=square.dim)
        assert square.to_components(
            square.from_components(components)
        ) == pytest.approx(components)
        assert wide.to_components(wide.from_components(components)) == pytest.approx(
            components
        )

    def test_the_default_is_the_square_grid(self):
        """v1's default, and half the memory of the wide one."""
        assert sphere_module.Lebesgue(8).sampling == 1

    def test_the_two_grids_are_different_spaces(self):
        """They hold different arrays, so a vector of one is not a vector of
        the other and the formal-adjoint lift must not move one silently."""
        assert not sphere_module.Lebesgue(8).shares_vectors_with(
            sphere_module.Lebesgue(8, sampling=2)
        )
        assert sphere_module.Lebesgue(8) != sphere_module.Lebesgue(8, sampling=2)

    def test_an_impossible_sampling_is_refused(self):
        with pytest.raises(ValueError, match="sampling is 1 or 2"):
            sphere_module.Lebesgue(8, sampling=3)

    def test_the_vector_operations_act_on_the_grid(self, rng):
        space = sphere_module.Lebesgue(8)
        first, second = space.random(rng=rng), space.random(rng=rng)
        before = space.grid_values(first).copy()

        total = space.axpy(2.0, second, space.copy(first))
        assert space.grid_values(total) == pytest.approx(
            before + 2.0 * space.grid_values(second)
        )
        # The copy really was one: the original is untouched.
        assert space.grid_values(first) == pytest.approx(before)


class TestGeometrySubmodules:
    """D-3: one submodule per geometry, exporting classes."""

    def test_the_sphere_spaces_are_classes(self):
        from pygeoinf2.symmetric_space.sphere import Lebesgue, Sobolev, Sphere

        space = sphere_module.Sobolev(8, 2.0, 0.2)
        assert isinstance(space, Sobolev)
        assert isinstance(space, Sphere)
        assert type(space).__name__ == "Sobolev"
        assert isinstance(Lebesgue(8), Sphere)

    @pytest.mark.parametrize(
        "module, build",
        [
            ("circle", lambda m: m.Sobolev(16, 2.0, 0.2)),
            ("torus", lambda m: m.Sobolev((8, 8), 2.0, 0.2)),
            ("line", lambda m: m.Sobolev(16, 2.0, 0.2)),
            (
                "plane",
                lambda m: m.Sobolev((8, 8), 2.0, 0.2, bounds=((0.0, 1.0), (0.0, 1.0))),
            ),
            (
                "box",
                lambda m: m.Sobolev((8,), 2.0, 0.2, bounds=((0.0, 1.0),)),
            ),
        ],
    )
    def test_each_geometry_names_itself(self, module, build):
        import importlib

        imported = importlib.import_module(f"pygeoinf2.symmetric_space.{module}")
        space = build(imported)
        assert isinstance(space, imported.Sobolev)
        assert space.dim > 0

    def test_a_torus_has_two_axes_and_says_so(self):
        from pygeoinf2.symmetric_space import torus

        with pytest.raises(ValueError, match="two axes"):
            torus.Lebesgue((4, 4, 4))
