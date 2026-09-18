"""Radial profiles: functions of radius under ``r^2 dr``.

Needs ``planetmodel`` 1.2. The independent routes (DECISIONS.md D-107) are the
ball, of which this space is the part of degree zero and with which it must
agree component for component; ``planetmodel``'s banded solve; and the closed
forms a constant length scale allows.
"""

import numpy as np
import pytest

randomfield = pytest.importorskip("planetmodel.randomfield")
if not hasattr(randomfield, "SpectralBasis"):  # pragma: no cover
    pytest.skip("needs planetmodel 1.2 or later", allow_module_level=True)

from pygeoinf2.sem1d.ball import Ball  # noqa: E402
from pygeoinf2.sem1d.radial import Lebesgue, Radial, Sobolev  # noqa: E402
from pygeoinf2.testing import (  # noqa: E402
    check_coordinates,
    check_measure,
    check_operator,
    check_space,
    check_traits,
    check_white_noise,
)


def whole(order, /, **options):
    options = {"modes": 30, "length_scale": 0.3, "element_length": 0.1, **options}
    return Radial(options.pop("modes"), order=order, **options).with_order(order)


def shell(order, /, **options):
    options = {"inner_radius": 0.5, "length_scale": 0.2, **options}
    return whole(order, **options)


GEOMETRIES = [whole, shell]


class TestTheSpace:
    @pytest.mark.parametrize("build", GEOMETRIES)
    @pytest.mark.parametrize("order", [0.0, 2.0])
    def test_the_axioms(self, build, order, rng):
        space = build(order)
        check_space(space, rng=rng, rebuild=lambda: build(order))
        check_coordinates(space, rng=rng)
        # The check compares basis directions, whose products with white noise
        # have the metric for their variance, against a fixed tolerance; a
        # Robin condition lifts the eigenvalues and the metric with them, so
        # the tolerance is put at four standard errors of the noisiest pair.
        samples = 20000
        noisiest = float(np.max(space.metric_values[:3]))
        check_white_noise(
            space,
            rng=rng,
            samples=samples,
            rtol=max(0.06, 4.0 * noisiest / np.sqrt(samples)),
        )

    def test_the_subclasses_are_named(self):
        sobolev = whole(2.0)
        assert type(sobolev) is Sobolev and type(sobolev.with_order(0.0)) is Lebesgue
        assert sobolev == Sobolev(30, 2.0, 0.3, element_length=0.1)
        assert sobolev.with_order(0.0) == Lebesgue(
            30, length_scale=0.3, element_length=0.1
        )
        assert whole(0.0) != shell(0.0)

    def test_a_whole_ball_is_padded_outwards_only(self):
        assert whole(0.0).padding == pytest.approx((0.0, 0.6))
        assert whole(0.0, boundary=None).padding == pytest.approx((0.0, 1.2))
        assert whole(0.0).nodes[0] == 0.0
        assert shell(0.0, padding=2.0).padding == pytest.approx((0.5, 2.0))
        with pytest.raises(ValueError, match="radii"):
            Radial(10, inner_radius=1.0, radius=0.5)

    def test_it_is_not_the_interval(self, rng):
        """Same nodes and the same field, and a different norm: ``r^2 dr`` is
        not ``dx``."""
        from pygeoinf2.sem1d.interval import Interval

        radial = shell(0.0, padding=0.3)
        line = Interval(
            30, lower=0.5, upper=1.0, length_scale=0.2, padding=0.3, element_length=0.1
        )
        assert np.allclose(radial.nodes, line.nodes)
        ones = np.ones(radial.nodes.size)
        assert radial.integral_functional()(ones) == pytest.approx(
            4.0 * np.pi * (1.0 - 0.125) / 3.0, rel=1e-3
        )
        assert line.integral_functional()(ones) == pytest.approx(0.5, rel=1e-3)

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_the_inner_product_is_the_volume_integral(self, build, rng):
        space = build(0.0)
        u, v = space.random(rng=rng), space.random(rng=rng)
        # planetmodel leaves the centre of a ball out of its mass, where the
        # weight r^2 is zero.
        mass = space.basis.family.mass(0)
        mass = np.concatenate((np.zeros(space.nodes.size - mass.size), mass))
        assert space.inner_product(u, v) == pytest.approx(
            4.0 * np.pi * np.sum(mass * u * v), rel=1e-10
        )

    def test_the_integral_of_one_is_the_volume(self):
        for build in GEOMETRIES:
            options = dict(element_length=0.25, padding=0.3)
            nodes = build(0.0, modes=1, **options).nodes.size
            space = build(0.0, modes=nodes - (build is whole), **options)
            constant = np.ones(space.nodes.size)
            assert space.integral_functional()(constant) == pytest.approx(
                space.domain_volume, rel=1e-8
            )

    @pytest.mark.parametrize("inner", [0.0, 0.5])
    def test_it_is_the_part_of_degree_zero_of_the_ball(self, inner, rng):
        """Component for component, and norm for norm."""
        options = dict(inner_radius=inner, length_scale=0.3, element_length=0.1)
        profile = Radial(9, **options)
        ball = Ball(3, radial_modes=9, **options)
        u = profile.random(rng=rng)
        field = u[:, None, None] + ball.zero()
        components = ball.to_components(field)
        zero = ball.basis.block(0, 0, 0)
        assert np.allclose(components[zero], profile.to_components(u), atol=1e-12)
        assert np.allclose(np.delete(components, np.arange(9)), 0.0, atol=1e-12)
        assert ball.norm(field) == pytest.approx(profile.norm(u), rel=1e-12)
        assert np.allclose(ball.eigenvalues[zero], profile.eigenvalues)

    def test_the_modes_are_even_in_the_radius(self):
        """``sin(k r) / r`` for a constant length scale: regular at the centre
        as a field of degree zero must be."""
        from scipy.special import spherical_jn

        space = whole(0.0, modes=12, element_length=0.05)
        radii = np.linspace(0.0, 1.0, 101)
        values = space.basis.evaluate(radii)
        for j in (1, 5, 11):
            k = np.sqrt(space.eigenvalues[j] - 1.0) / 0.3
            exact = spherical_jn(0, k * radii)
            scaled = exact * (values[:, j] @ exact) / (exact @ exact)
            assert np.allclose(
                values[:, j], scaled, atol=2e-3 * np.abs(values[:, j]).max()
            )


class TestBoundary:
    def test_robin_carries_the_curvature_of_each_end(self):
        """``0.7 / L -+ 1 / r`` at the ends of the mesh, none at the centre, and
        half the padding."""
        space = shell(0.0, length_scale=0.1, boundary="robin")
        assert space.padding == pytest.approx((0.2, 0.2))
        assert space.robin == pytest.approx((7.0 - 1.0 / 0.3, 7.0 + 1.0 / 1.2))
        ball = whole(0.0, length_scale=0.1, boundary="robin")
        assert ball.robin == pytest.approx((0.0, 7.0 + 1.0 / 1.2))
        assert whole(0.0, boundary=None).robin == (0.0, 0.0)
        assert whole(0.0, boundary=None) != whole(0.0)

    @pytest.mark.parametrize("inner", [0.0, 0.5])
    def test_two_length_scales_then_do_what_four_do_without_it(self, inner):
        """The prior variance on the domain, against a mesh padded so far that
        its ends cannot matter."""

        def variance(padding, boundary):
            space = Radial(
                180 if inner == 0.0 else 90,
                inner_radius=inner,
                length_scale=0.1,
                padding=padding,
                boundary=boundary,
                element_length=0.02,
            )
            return space.interior_values(
                space.pointwise_variance(space.eigenvalues**-3.0)
            )

        reference = variance(1.2, None)

        def error(padding, boundary):
            return np.abs(variance(padding, boundary) / reference - 1.0).max()

        assert error(0.2, "robin") < 2e-2
        assert error(0.2, None) > 1e-1
        assert error(0.2, "robin") < 2.0 * error(0.4, None)


class TestPoints:
    def test_the_centre_needs_three_halves_and_a_shell_one_half(self):
        assert whole(1.0).point_evaluation_order() == 1.5
        assert whole(1.0).point_evaluation_order(points=[0.3, 0.9]) == 0.5
        assert whole(1.0).point_evaluation_order(points=[0.0, 0.9]) == 1.5
        assert shell(1.0).point_evaluation_order() == 0.5
        whole(1.0).dirac(0.4)
        whole(1.0).point_evaluation_operator([0.2, 0.9])
        with pytest.raises(ValueError, match="above 1.5"):
            whole(1.0).dirac(0.0)
        with pytest.raises(ValueError, match="above 0.5"):
            whole(0.5).dirac(0.4)
        whole(1.75).dirac(0.0)
        whole(0.0).dirac(0.0, unsafe=True)

    def test_the_representer_says_the_same(self):
        """Its squared norm as modes are added: still growing at the centre at
        order 1.25 and settled at 1.75; settled away from it by 0.75."""

        def growth(order, point):
            norms = []
            for modes in (40, 80, 160):
                space = Radial(
                    modes, order=order, length_scale=0.2, element_length=0.01
                )
                norms.append(
                    space.squared_norm(space.dirac(point, unsafe=True).representer)
                )
            return norms[2] / norms[1]

        assert growth(1.25, 0.0) > 1.4
        assert growth(1.75, 0.0) < 1.15
        assert growth(0.75, 0.5) < 1.1

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_evaluation_and_its_adjoint(self, build, rng):
        space = build(2.0)
        radii = [space.random_point(rng=rng) for _ in range(6)]
        assert all(space.inner_radius <= r <= space.radius for r in radii)
        operator = space.point_evaluation_operator(radii)
        check_operator(operator, rng=rng)
        x = space.random(rng=rng)
        nodes = space.interior_nodes
        assert np.allclose(
            space.evaluate(x, nodes), space.interior_values(x), atol=1e-11
        )


class TestAlgebraAndMeasures:
    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_multiplication(self, build, rng):
        f = build(0.0).project_function(lambda r: 1.0 + 0.5 * r**2)
        on_lebesgue = build(0.0).multiplication_operator(f)
        check_operator(on_lebesgue, rng=rng)
        check_traits(on_lebesgue, rng=rng)
        check_operator(build(2.0).multiplication_operator(f), rng=rng)

    def test_the_sobolev_measure_is_the_banded_solve(self, rng):
        options = dict(
            inner_radius=0.5, length_scale=0.2, element_length=0.25, padding=0.3
        )
        nodes = Radial(1, **options).nodes.size
        space = Radial(nodes, **options)
        covariance = space.sobolev_measure(1.0).covariance
        v = rng.normal(size=nodes)
        assert np.allclose(covariance(v), space.basis.family.solve(0, v), atol=1e-10)

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_pointwise_std_is_what_the_covariance_says(self, build):
        """A profile even in ``r``, as one regular at the centre must be."""
        space = build(0.0, modes=60, element_length=0.05)
        target = 0.3 + 0.1 * space.nodes**2
        measure = space.sobolev_measure(2.5, pointwise_std=target)
        radii = np.linspace(space.inner_radius, space.radius, 6)
        evaluation = space.point_evaluation_operator(radii, unsafe=True)
        matrix = (evaluation @ measure.covariance @ evaluation.adjoint).matrix(
            form="components"
        )
        assert np.allclose(np.sqrt(np.diag(matrix)), 0.3 + 0.1 * radii**2, rtol=5e-3)

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_the_moments_match_the_samples(self, build, rng):
        space = build(2.0)
        check_measure(space.sobolev_measure(1.0, pointwise_std=0.3), rng=rng)
