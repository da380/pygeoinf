"""Piecewise-continuous fields: a direct sum of layers that are adjacent.

Needs ``planetmodel`` 1.2. The independent routes (DECISIONS.md D-107) are the
layers themselves, which a layered space must agree with piece by piece, the
plain direct sum it claims to be, and closed forms for a layered polynomial.
"""

import numpy as np
import pytest

randomfield = pytest.importorskip("planetmodel.randomfield")
if not hasattr(randomfield, "SpectralBasis"):  # pragma: no cover
    pytest.skip("needs planetmodel 1.2 or later", allow_module_level=True)

from pygeoinf2.algebra.direct_sum import DirectSum  # noqa: E402
from pygeoinf2.sem1d.ball import Ball  # noqa: E402
from pygeoinf2.sem1d.interval import Interval  # noqa: E402
from pygeoinf2.sem1d.layered import Layered  # noqa: E402
from pygeoinf2.sem1d.radial import Radial  # noqa: E402
from pygeoinf2.testing import (  # noqa: E402
    check_coordinates,
    check_measure,
    check_operator,
    check_space,
    check_traits,
    check_white_noise,
)

RADII = [0.0, 0.35, 0.55, 1.0]
# A density that jumps at both interfaces, even in r within each layer.
PIECES = [
    lambda r: 13.0 - 1.5 * r**2,
    lambda r: 12.0 - 8.0 * r**2,
    lambda r: 5.5 - 2.0 * r**2,
]
COEFFICIENTS = [(13.0, 1.5), (12.0, 8.0), (5.5, 2.0)]


def profiles(order=1.0, /, **options):
    options = {"length_scale": [0.15, 0.1, 0.2], **options}
    return Layered.radial(RADII, [12, 10, 20], order=order, **options)


def lines(order=1.0, /):
    return Layered.interval([-1.0, 0.0, 0.5, 2.0], 16, order=order, length_scale=0.2)


def balls(order=2.0, /, **options):
    options = {"length_scale": 0.3, "radial_modes": 5, "element_length": 0.2, **options}
    return Layered.ball([0.0, 0.5, 1.0], 3, order=order, **options)


GEOMETRIES = [profiles, lines, balls]


class TestTheSpace:
    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_the_axioms(self, build, rng):
        space = build()
        check_space(space, rng=rng, rebuild=build)
        check_coordinates(space, rng=rng)
        check_white_noise(space, rng=rng, samples=6000, rtol=0.14)

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_it_is_the_direct_sum_of_its_layers(self, build, rng):
        """Equal to it both ways round and hashing alike, so that what a block
        operator or a product measure builds lands here; and the same inner
        product, which is the claim's content."""
        space = build()
        plain = DirectSum(list(space.layers))
        assert space == plain and plain == space
        assert hash(space) == hash(plain)
        assert len({space, plain}) == 1
        x, y = space.random(rng=rng), space.random(rng=rng)
        assert space.inner_product(x, y) == pytest.approx(plain.inner_product(x, y))
        assert space != build(0.0)

    def test_the_structure(self):
        space = profiles()
        assert space.geometry is Radial and len(space) == 3
        assert space.layer_bounds == ((0.0, 0.35), (0.35, 0.55), (0.55, 1.0))
        assert space.interfaces == (0.35, 0.55)
        assert space.domain_volume == pytest.approx(4.0 * np.pi / 3.0)
        assert space.dim == 42 and space.columns(1) == slice(12, 22)
        assert lines().geometry is Interval and balls().geometry is Ball
        assert balls().spatial_dimension == 3

    def test_arguments_are_shared_or_given_layer_by_layer_as_a_list(self):
        space = profiles()
        assert [layer.dim for layer in space.layers] == [12, 10, 20]
        assert [layer.order for layer in space.layers] == [1.0, 1.0, 1.0]
        mixed = Layered.radial(RADII, 8, order=[0.0, 1.0, 2.0], padding=(0.1, 0.2))
        assert [layer.order for layer in mixed.layers] == [0.0, 1.0, 2.0]
        # A tuple is one value: the padding pair of every layer, the centre
        # taking none.
        assert mixed.layers[2].padding == pytest.approx((0.1, 0.2))
        assert mixed.layers[0].padding == pytest.approx((0.0, 0.2))
        with pytest.raises(ValueError, match="layer by layer"):
            Layered.radial(RADII, [8, 8])

    def test_layers_are_named_and_reached_by_name(self, rng):
        space = Layered.radial(RADII, 8, labels=["inner core", "outer core", "mantle"])
        x = space.random(rng=rng)
        assert space.component(x, "mantle") is x[2]
        assert space.columns("outer core") == slice(8, 16)

    def test_layers_are_built_and_handed_over_too(self):
        mantle = Radial(20, inner_radius=0.55, order=1.0, length_scale=0.2)
        core = Radial(12, radius=0.35, order=1.0, length_scale=0.15)
        space = Layered([core, mantle])
        assert space.interfaces == () and space.layer_bounds[1] == (0.55, 1.0)
        with pytest.raises(ValueError, match="overlap"):
            Layered([mantle, core])
        with pytest.raises(ValueError, match="overlap"):
            Layered([Radial(8, radius=0.6), mantle])
        with pytest.raises(TypeError, match="all"):
            Layered([core, Interval(8)])

    def test_with_order(self):
        space = profiles()
        assert [layer.order for layer in space.with_order(0.0).layers] == [0.0] * 3
        assert [layer.order for layer in space.with_order([0.0, 1.0, 2.0]).layers] == [
            0.0,
            1.0,
            2.0,
        ]
        assert space.with_order(1.0) == space


class TestPoints:
    def test_a_point_is_in_one_layer_or_on_an_interface(self):
        space = profiles()
        assert [space.layer_index(r) for r in (0.0, 0.2, 0.4, 0.9, 1.0)] == [
            0,
            0,
            1,
            2,
            2,
        ]
        assert space.layer_index(0.55, side="below") == 1
        assert space.layer_index(0.55, side="above") == 2
        with pytest.raises(ValueError, match="interface"):
            space.layer_index(0.55)
        with pytest.raises(ValueError, match="none of the layers"):
            space.layer_index(1.2)
        with pytest.raises(ValueError, match="side"):
            space.layer_index(0.2, side="left")
        assert balls().layer_index((0.7, 10.0, 20.0)) == 1

    def test_a_gap_is_in_no_layer(self):
        space = profiles()
        solid = Layered([space.layers[0], space.layers[2]])
        assert solid.interfaces == ()
        assert solid.layer_index(0.35) == 0 and solid.layer_index(0.55) == 1
        with pytest.raises(ValueError, match="none of the layers"):
            solid.layer_index(0.45)

    def test_a_function_that_jumps_is_given_piece_by_piece(self):
        space = profiles()
        density = space.project_function(PIECES)
        radii = [0.1, 0.3, 0.4, 0.5, 0.7, 1.0]
        exact = [PIECES[space.layer_index(r)](r) for r in radii]
        assert np.allclose(space.evaluate(density, radii), exact, atol=1e-5)
        for side, piece in (("below", PIECES[1]), ("above", PIECES[2])):
            assert space.evaluate(density, [0.55], side=side)[0] == pytest.approx(
                piece(0.55), abs=1e-5
            )
        with pytest.raises(ValueError, match="interface"):
            space.evaluate(density, [0.55])
        with pytest.raises(ValueError, match="layer by layer"):
            space.project_function(PIECES[:2])

    def test_one_function_serves_every_layer_when_it_does_not_jump(self):
        space = lines()
        field = space.project_function(np.cos)
        points = [-0.7, 0.0, 0.3, 0.5, 1.9]
        for side in ("below", "above"):
            assert np.allclose(
                space.evaluate(field, points, side=side), np.cos(points), atol=1e-6
            )

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_evaluation_is_the_layers_own(self, build, rng):
        space = build()
        x = space.random(rng=rng)
        points = [space.layers[i].random_point(rng=rng) for i in (0, 1, 1, 0)]
        values = space.evaluate(x, points)
        for value, point, i in zip(values, points, (0, 1, 1, 0)):
            assert value == pytest.approx(space.layers[i].evaluate(x[i], [point])[0])
        rows = space.basis_matrix(points)
        assert rows.shape == (4, space.dim)
        assert np.all(rows[0, space.columns(1)] == 0.0)
        assert np.allclose(rows @ space.to_components(x), values)

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_the_observation_operator_and_its_adjoint(self, build, rng):
        space = build()
        points = [space.layers[i].random_point(rng=rng) for i in (0, 1, 0, 1)]
        operator = space.point_evaluation_operator(points)
        check_operator(operator, rng=rng)
        x = space.random(rng=rng)
        assert np.allclose(operator(x), space.evaluate(x, points))
        assert operator(x)[1] == pytest.approx(space.dirac(points[1])(x))
        with pytest.raises(ValueError, match="one point"):
            space.point_evaluation_operator([])

    def test_each_layer_keeps_its_own_guard(self):
        """A rough layer refuses its points and lets the others' through."""
        space = Layered.radial(RADII, 8, order=[1.0, 0.0, 1.0])
        space.dirac(0.2)
        space.dirac(0.9)
        with pytest.raises(ValueError, match="above 0.5"):
            space.dirac(0.45)
        space.dirac(0.45, unsafe=True)
        # The centre is a point of space, and wants three halves.
        with pytest.raises(ValueError, match="above 1.5"):
            space.dirac(0.0)


class TestJumps:
    def test_the_jump_is_above_less_below(self, rng):
        space = profiles()
        density = space.project_function(PIECES)
        for k, radius in enumerate(space.interfaces):
            jump = space.jump_operator(k)
            check_operator(jump, rng=rng)
            assert jump(density)[0] == pytest.approx(
                PIECES[k + 1](radius) - PIECES[k](radius), abs=1e-5
            )
            x = space.random(rng=rng)
            assert jump(x)[0] == pytest.approx(
                space.evaluate(x, [radius], side="above")[0]
                - space.evaluate(x, [radius], side="below")[0]
            )
        with pytest.raises(IndexError):
            space.jump_operator(2)

    def test_in_balls_it_is_taken_at_directions(self, rng):
        space = balls()
        directions = [(10.0, 20.0), (-40.0, 100.0), (75.0, -60.0)]
        jump = space.jump_operator(0, directions=directions)
        check_operator(jump, rng=rng)
        x = space.random(rng=rng)
        points = [(0.5, lat, lon) for lat, lon in directions]
        assert np.allclose(
            jump(x),
            space.evaluate(x, points, side="above")
            - space.evaluate(x, points, side="below"),
        )
        with pytest.raises(ValueError, match="directions"):
            space.jump_operator(0)
        with pytest.raises(ValueError, match="directions"):
            profiles().jump_operator(0, directions=directions)

    def test_conditioning_on_no_jump_makes_the_medium_continuous_there(self, rng):
        """The layers are independent under the prior, which is what a
        discontinuity means; conditioned on a vanishing jump at one interface
        every draw is continuous across it, and still jumps at the other."""
        space = profiles()
        prior = space.sobolev_measure(1.5, pointwise_std=0.3)
        welded = prior.condition(space.jump_operator(1), np.zeros(1))
        jumps = np.array(
            [
                [space.jump_operator(k)(x)[0] for k in (0, 1)]
                for x in (welded.sample(rng=rng) for _ in range(40))
            ]
        )
        assert np.abs(jumps[:, 1]).max() < 1e-8
        assert jumps[:, 0].std() > 0.2
        free = np.array(
            [space.jump_operator(1)(prior.sample(rng=rng))[0] for _ in range(40)]
        )
        assert free.std() > 0.2


class TestAlgebraAndMeasures:
    def test_the_integral_is_over_the_whole_medium(self):
        space = profiles()
        density = space.project_function(PIECES)
        exact = sum(
            4.0
            * np.pi
            * (
                (a * r2**3 / 3.0 - b * r2**5 / 5.0)
                - (a * r1**3 / 3.0 - b * r1**5 / 5.0)
            )
            for (a, b), r1, r2 in zip(COEFFICIENTS, RADII[:-1], RADII[1:])
        )
        assert space.integral_functional()(density) == pytest.approx(exact, rel=1e-7)

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_products_are_taken_layer_by_layer(self, build, rng):
        space = build()
        x, y = space.random(rng=rng), space.random(rng=rng)
        product = space.multiply(x, y)
        for i, layer in enumerate(space.layers):
            assert np.array_equal(product[i], layer.multiply(x[i], y[i]))
        operator = space.multiplication_operator(x)
        check_operator(operator, rng=rng)
        assert np.allclose(
            space.to_components(operator(y)), space.to_components(product)
        )
        on_lebesgue = space.with_order(0.0).multiplication_operator(x)
        check_traits(on_lebesgue, rng=rng)
        truncated = space.truncate(product)
        assert np.allclose(space.to_components(truncated), space.to_components(product))

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_the_prior_lives_on_the_space_with_independent_layers(self, build, rng):
        space = build()
        measure = space.sobolev_measure(2.5, pointwise_std=0.3)
        assert measure.domain == space
        check_measure(measure, rng=rng, samples=3000, rtol=0.2)
        a = space.layers[0].random_point(rng=rng)
        b = space.layers[1].random_point(rng=rng)
        look = space.point_evaluation_operator([a, b], unsafe=True)
        covariance = (look @ measure.covariance @ look.adjoint).matrix(
            form="components"
        )
        assert covariance[0, 1] == pytest.approx(0.0, abs=1e-12)

    def test_the_prior_is_given_layer_by_layer(self):
        space = profiles()
        expectation = space.project_function(PIECES)
        measure = space.sobolev_measure(
            [1.5, 2.0, 1.5], expectation=expectation, pointwise_std=[0.2, 0.3, 0.4]
        )
        assert np.allclose(measure.expectation[2], expectation[2])
        radii = [0.2, 0.45, 0.8]
        look = space.point_evaluation_operator(radii)
        covariance = (look @ measure.covariance @ look.adjoint).matrix(
            form="components"
        )
        assert np.allclose(np.sqrt(np.diag(covariance)), [0.2, 0.3, 0.4], rtol=2e-2)
        fields = tuple(
            0.1 + 0.0 * layer.zero() + 0.1 * (i + 1)
            for i, layer in enumerate(space.layers)
        )
        by_field = space.sobolev_measure(1.5, pointwise_std=fields)
        covariance = (look @ by_field.covariance @ look.adjoint).matrix(
            form="components"
        )
        assert np.allclose(np.sqrt(np.diag(covariance)), [0.2, 0.3, 0.4], rtol=2e-2)
        with pytest.raises(ValueError, match="layer by layer"):
            space.sobolev_measure([1.5, 1.5])
