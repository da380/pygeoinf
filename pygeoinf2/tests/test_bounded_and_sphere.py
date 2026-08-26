"""Bounded domains and the sphere."""

import numpy as np
import pytest

from pygeoinf2 import Traits
from pygeoinf2.spaces import Box, Interval, lift_formal_adjoint
from pygeoinf2.spaces.box import Box as BoxClass
from pygeoinf2.testing import (
    check_coordinates,
    check_measure,
    check_operator,
    check_space,
    check_traits,
    check_white_noise,
)

pyshtools = pytest.importorskip("pyshtools")

from pygeoinf2.spaces import sphere as sphere_module  # noqa: E402


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

    def test_a_function_vanishes_on_the_padding(self, rng):
        """The support assumption, made real rather than assumed."""
        space = Interval(32, lower=0.0, upper=1.0)
        field = space.project_function(lambda t: 1.0 + t**2)
        assert np.all(field[~space.interior_mask] == 0.0)
        assert np.all(field[space.interior_mask] > 0.0)

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

    def test_it_is_refused_on_a_sobolev_space(self):
        """Multiplying by a discontinuous mask does not commute with the metric."""
        space = Interval(32, order=2.0, length_scale=0.1)
        with pytest.raises(ValueError, match="Lebesgue"):
            space.support_projection()

    def test_it_can_be_lifted_instead(self, rng):
        """The lift gives the right adjoint and claims no symmetry."""
        lebesgue = Interval(32, lower=0.0, upper=1.0)
        sobolev = Interval(32, lower=0.0, upper=1.0, order=2.0, length_scale=0.1)
        lifted = lift_formal_adjoint(lebesgue.support_projection(), sobolev)
        assert lifted.traits == Traits.NONE
        check_operator(lifted, rng=rng)

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
        assert np.all(field[~space.interior_mask] == 0.0)


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
            point = np.array([space.colatitudes[i], space.longitudes[j]])
            assert components @ space.basis_at(point) == pytest.approx(
                field[i, j], abs=1e-10
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

        field = space.project_function(lambda p: np.cos(p[0]))
        assert np.allclose(operator(field), [np.cos(p[0]) for p in points], atol=1e-8)

    def test_the_adjoint_returns_dirac_representers(self, rng):
        space = sphere_module.Sobolev(8, 2.0, 0.2)
        point = space.random_point(rng=rng)
        operator = space.point_evaluation_operator([point, space.random_point(rng=rng)])
        assert np.allclose(
            space.to_components(operator.adjoint(np.array([1.0, 0.0]))),
            space.to_components(space.dirac(point).representer),
        )

    def test_random_points_are_uniform_over_the_area(self, rng):
        """Uniform in cos(colatitude), which is the classic thing to get wrong."""
        space = sphere_module.Lebesgue(4)
        cosines = np.array(
            [np.cos(space.random_point(rng=rng)[0]) for _ in range(4000)]
        )
        assert abs(cosines.mean()) < 0.06
        # A uniform distribution on [-1, 1] has variance 1/3.
        assert cosines.var() == pytest.approx(1.0 / 3.0, rel=0.1)

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
        with pytest.raises(ValueError, match="colatitude"):
            sphere_module.Lebesgue(4).basis_at(np.array([0.1]))
