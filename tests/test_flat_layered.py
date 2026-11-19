"""Flat layers stacked in depth: the layered strip and the layered slab.

The independent routes are the layers themselves, which a layered space must
agree with piece by piece; a layered space of one layer against the plain
strip or slab; the plain direct sum it claims to be; functions that jump by a
known amount, projected layer by layer and read back; closed-form integrals;
and the shared class's own reading of the sides, along the coordinate, as the
negative control of the reading in depth.
"""

import numpy as np
import pytest
from planetmodel.fourier import PaddedDomain

from pygeoinf import disk, slab, spectral_element, strip
from pygeoinf.algebra.direct_sum import DirectSum
from pygeoinf.slab import Slab
from pygeoinf.spectral_element import base as spectral_base
from pygeoinf.strip import Strip
from pygeoinf.testing import (
    check_coordinates,
    check_measure,
    check_operator,
    check_space,
    check_traits,
    check_white_noise,
)
from pygeoinf.traits import Traits

DEPTHS = [0.0, 0.3, 0.6, 1.2]

# A field of (depth, x) that jumps at both interfaces of the strip, periodic
# over its period of 2 and a polynomial in depth within each layer.
STRIP_PIECES = [
    lambda p: 2.0 + p[0] + 0.5 * np.cos(np.pi * p[1]),
    lambda p: -1.0 + 0.5 * p[0] ** 2 + 0.3 * np.sin(np.pi * p[1]),
    lambda p: 1.5 - p[0] + 0.2 * np.cos(2.0 * np.pi * p[1]),
]
# Their integrals in depth over each layer: the lateral terms integrate to
# zero over the period.
STRIP_DEPTH_INTEGRALS = [
    2.0 * 0.3 + 0.3**2 / 2.0,
    -1.0 * 0.3 + 0.5 * (0.6**3 - 0.3**3) / 3.0,
    1.5 * 0.6 - (1.2**2 - 0.6**2) / 2.0,
]

# A field of (depth, x, y) that jumps at the slab's one interface, periodic
# over its periods of 2 and 1.5.
SLAB_PIECES = [
    lambda p: 1.0 + p[0] ** 2 + 0.4 * np.cos(np.pi * p[1]),
    lambda p: 3.0 - p[0] + 0.6 * np.sin(4.0 * np.pi * p[2] / 3.0),
]


def strips(order=2.0, /, **options):
    """Three strips over the whole period of 2, an even grid, the length
    scale given layer by layer."""
    settings = dict(lateral=2.0, length_scale=[0.2, 0.3, 0.4], element_length=0.15)
    settings.update(options)
    return strip.Layered(DEPTHS, 12, order=order, **settings)


def bounded_strips(order=2.0, /, **options):
    """Two strips bounded across, inside a wider period, the first below zero
    depth."""
    settings = dict(
        lateral=PaddedDomain([(0.5, 2.0)], padding=0.6),
        length_scale=0.3,
        element_length=0.3,
    )
    settings.update(options)
    return strip.Layered([0.2, 0.5, 1.0], 11, order=order, **settings)


def slabs(order=2.0, /, **options):
    """Two slabs over the whole period, one axis even and one odd, the mesh
    given layer by layer."""
    settings = dict(lateral=(2.0, 1.5), length_scale=0.4, element_length=[0.15, 0.3])
    settings.update(options)
    return slab.Layered(DEPTHS[:3], (6, 5), order=order, **settings)


def bounded_slabs(order=2.0, /, **options):
    """Two slabs bounded along ``x`` and periodic along ``y``."""
    settings = dict(
        lateral=PaddedDomain([(0.0, 1.0), (-1.0, 0.5)], padding=(0.5, 0.0)),
        length_scale=0.4,
        element_length=0.4,
    )
    settings.update(options)
    return slab.Layered([0.0, 0.4, 1.0], (5, 6), order=order, **settings)


GEOMETRIES = [strips, bounded_strips, slabs, bounded_slabs]


def random_points(space, layers, rng, /):
    """A random point of each named layer's domain proper."""
    return [space.layers[i].random_point(rng=rng) for i in layers]


def lateral_positions(space, count, rng, /):
    """Random lateral positions in the domain proper: ``x`` along a strip,
    ``(x, y)`` pairs in a slab."""
    bounds = space.layers[0].lateral_domain.bounds
    columns = [rng.uniform(low, high, count) for low, high in bounds]
    return columns[0] if len(columns) == 1 else np.column_stack(columns)


def grid_positions(space, /):
    """Every point of the lateral grid, as lateral positions."""
    axes = space.layers[0].lateral_axes
    if len(axes) == 1:
        return axes[0]
    return np.column_stack([a.ravel() for a in np.meshgrid(*axes, indexing="ij")])


def at_depth(depth, positions, /):
    """Points ``(depth, x)`` or ``(depth, x, y)`` at the lateral positions."""
    lateral = np.asarray(positions, dtype=float).reshape(len(positions), -1)
    return [(depth, *row) for row in lateral]


class TestTheSpace:
    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_the_axioms(self, build, rng):
        def small():
            return build(depth_modes=2)

        space = small()
        check_space(space, rng=rng, rebuild=small)
        check_coordinates(space, rng=rng)

    @pytest.mark.parametrize("build", [strips, slabs])
    def test_white_noise_has_identity_covariance(self, build, rng):
        """The layers' own white noise, side by side: the tolerance is six
        standard errors at this many samples."""
        check_white_noise(build(depth_modes=2), rng=rng, samples=2000, rtol=0.24)

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

    @pytest.mark.parametrize(
        "module, shape, lateral",
        [(strip, 12, 2.0), (slab, (6, 5), (2.0, 1.5))],
    )
    def test_one_layer_is_the_plain_space(self, module, shape, lateral, rng):
        """Built from two boundaries, a layered space is the one layer: its
        inner product, components, values, rows, integral, observation
        operators and prior are the plain space's own."""
        layer = module.Sobolev(shape, 2.0, 0.3, lateral=lateral, top=0.2, bottom=1.0)
        space = module.Layered(
            [0.2, 1.0], shape, lateral=lateral, order=2.0, length_scale=0.3
        )
        assert len(space) == 1 and space.layers[0] == layer
        assert space.interfaces == ()
        x, y = layer.random(rng=rng), layer.random(rng=rng)
        assert space.inner_product((x,), (y,)) == pytest.approx(
            layer.inner_product(x, y), rel=1e-12
        )
        assert np.allclose(space.to_components((x,)), layer.to_components(x))
        points = [layer.random_point(rng=rng) for _ in range(6)]
        assert np.allclose(space.evaluate((x,), points), layer.evaluate(x, points))
        assert np.allclose(space.basis_matrix(points), layer.basis_matrix(points))
        assert space.integral_functional()((x,)) == pytest.approx(
            layer.integral_functional()(x), rel=1e-12
        )
        for route in ("dense", "local", "nufft"):
            assert np.allclose(
                space.point_evaluation_operator(points, route=route)((x,)),
                layer.point_evaluation_operator(points, route=route)(x),
                rtol=0.0,
                atol=1e-12 * np.abs(layer.evaluate(x, points)).max(),
            )
        got = space.sobolev_measure(2.5, pointwise_std=0.3).covariance((x,))[0]
        want = layer.sobolev_measure(2.5, pointwise_std=0.3).covariance(x)
        assert layer.norm(layer.subtract(got, want)) < 1e-12 * layer.norm(want)

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_each_geometry_has_its_own_class_on_the_shared_one(self, build):
        space = build()
        assert isinstance(space, spectral_element.Layered)
        assert type(space.with_order(0.0)) is type(space)
        handed = type(space).from_layers(space.layers, labels=space.labels)
        assert type(handed) is type(space) and handed == space

    def test_flat_layers_go_to_their_own_class_and_no_other(self):
        """Only the flat classes name the sides in depth and take jumps at
        lateral positions, so the shared class and the other geometry's
        refuse flat layers by name; and a flat class takes its own layers
        alone."""
        across, boxed = strips().layers, slabs().layers
        for layers, name in ((across, "strip.Layered"), (boxed, "slab.Layered")):
            with pytest.raises(TypeError, match=name):
                spectral_element.Layered(layers)
            with pytest.raises(TypeError, match=name):
                spectral_element.Layered.from_layers(layers)
        with pytest.raises(TypeError, match="slab.Layered"):
            strip.Layered.from_layers(boxed)
        with pytest.raises(TypeError, match="strip.Layered"):
            slab.Layered.from_layers(across)
        rings = disk.Layered([0.0, 0.5, 1.0], 3, order=2.0, length_scale=0.3)
        with pytest.raises(TypeError, match="takes Strips"):
            strip.Layered.from_layers(rings.layers)
        with pytest.raises(TypeError, match="all"):
            strip.Layered.from_layers([across[0], boxed[1]])

    def test_the_structure(self):
        space = strips()
        assert space.geometry is Strip and len(space) == 3
        assert space.layer_bounds == ((0.0, 0.3), (0.3, 0.6), (0.6, 1.2))
        assert space.interfaces == (0.3, 0.6)
        assert space.domain_volume == pytest.approx(2.0 * 1.2)
        assert space.spatial_dimension == 2
        dims = [layer.dim for layer in space.layers]
        assert space.dim == sum(dims)
        assert space.columns(1) == slice(dims[0], dims[0] + dims[1])
        assert repr(space).startswith("Layered(Strip: [0, 0.3], [0.3, 0.6]")
        box = bounded_slabs()
        assert box.geometry is Slab and box.spatial_dimension == 3
        assert box.domain_volume == pytest.approx(1.0 * 1.5 * 1.0)
        assert bounded_strips().domain_volume == pytest.approx(1.5 * 0.8)

    def test_the_lateral_grid_and_box_are_shared_and_the_rest_given_by_layer(self):
        space = strips([0.0, 2.0, 2.5], depth_modes=[3, 4, 5], padding=(0.1, 0.4))
        assert [layer.order for layer in space.layers] == [0.0, 2.0, 2.5]
        assert [int(layer.basis.nmodes.max()) for layer in space.layers] == [3, 4, 5]
        assert [layer.length_scale for layer in space.layers] == [0.2, 0.3, 0.4]
        # A tuple is one value: the padding pair of every layer.
        assert all(layer.padding == (0.1, 0.4) for layer in space.layers)
        assert {layer.lateral_shape for layer in space.layers} == {(12,)}
        assert all(
            layer.lateral_domain == PaddedDomain([(0.0, 2.0)]) for layer in space.layers
        )
        # The shape and the box of a slab are one value each, a list too.
        box = slab.Layered([0.0, 0.5, 1.0], [6, 5], lateral=[2.0, 1.5])
        assert {layer.lateral_shape for layer in box.layers} == {(6, 5)}
        assert {layer.lateral_domain.lengths for layer in box.layers} == {(2.0, 1.5)}
        with pytest.raises(ValueError, match="layer by layer"):
            strips(depth_modes=[3, 4])

    def test_layers_built_by_hand_share_one_lateral_grid_and_box(self):
        """Refused by name otherwise: the shape, or the bounds and padding."""
        options = dict(order=2.0, length_scale=0.3, element_length=0.3)
        top = Strip(12, lateral=2.0, bottom=0.4, **options).with_order(2.0)
        base = Strip(12, lateral=2.0, top=0.4, **options).with_order(2.0)
        assert strip.Layered.from_layers([top, base]) == strip.Layered(
            [0.0, 0.4, 1.0], 12, lateral=2.0, **options
        )
        finer = Strip(13, lateral=2.0, top=0.4, **options)
        with pytest.raises(ValueError, match="lateral grid"):
            strip.Layered.from_layers([top, finer])
        wider = Strip(12, lateral=3.0, top=0.4, **options)
        with pytest.raises(ValueError, match="lateral box"):
            strip.Layered.from_layers([top, wider])
        bounded = Strip(
            12, lateral=PaddedDomain([(0.0, 2.0)], padding=0.5), top=0.4, **options
        )
        with pytest.raises(ValueError, match="lateral box"):
            strip.Layered.from_layers([top, bounded])
        with pytest.raises(ValueError, match="overlap"):
            strip.Layered.from_layers([base, top])

    def test_a_gap_is_in_no_layer(self):
        options = dict(order=2.0, lateral=2.0, length_scale=0.3, element_length=0.3)
        top = Strip(12, bottom=0.3, **options)
        base = Strip(12, top=0.5, **options)
        space = strip.Layered.from_layers([top, base])
        assert space.interfaces == ()
        assert space.layer_index((0.3, 1.0)) == 0 and space.layer_index((0.5, 1.0)) == 1
        with pytest.raises(ValueError, match="none of the layers"):
            space.layer_index((0.4, 1.0))
        with pytest.raises(IndexError):
            space.jump_operator(0, positions=[1.0])

    def test_layers_are_named_and_reached_by_name(self, rng):
        space = strips(labels=["sediment", "crust", "mantle"])
        x = space.random(rng=rng)
        assert space.component(x, "mantle") is x[2]
        assert space.columns("crust") == space.columns(1)
        assert space.with_order(0.0).labels == space.labels

    def test_with_order(self):
        space = strips()
        assert [layer.order for layer in space.with_order(0.0).layers] == [0.0] * 3
        mixed = space.with_order([0.0, 2.0, 2.5])
        assert [layer.order for layer in mixed.layers] == [0.0, 2.0, 2.5]
        assert space.with_order(2.0) == space
        assert all(
            new.basis is old.basis
            for new, old in zip(space.with_order(0.0).layers, space.layers)
        )


class TestPoints:
    def test_above_is_shallower_and_below_deeper(self):
        """At an interface, above is the layer of smaller depth, which comes
        first; the shared class's own reading, along the coordinate, is the
        other way round, and is what the flat class turns over."""
        space = strips()
        depths = [0.0, 0.1, 0.45, 0.9, 1.2]
        assert [space.layer_index((z, 0.5)) for z in depths] == [0, 0, 1, 2, 2]
        assert space.layer_index((0.3, 0.5), side="above") == 0
        assert space.layer_index((0.3, 0.5), side="below") == 1
        assert space.layer_index((0.6, 0.5), side="above") == 1
        assert space.layer_index((0.6, 0.5), side="below") == 2
        along = spectral_element.Layered.layer_index
        assert along(space, (0.3, 0.5), side="above") == 1
        assert along(space, (0.3, 0.5), side="below") == 0
        with pytest.raises(ValueError, match="interface"):
            space.layer_index((0.3, 0.5))
        for depth in (-0.1, 1.3):
            with pytest.raises(ValueError, match="none of the layers"):
                space.layer_index((depth, 0.5))
        with pytest.raises(ValueError, match="side"):
            space.layer_index((0.1, 0.5), side="deeper")

    @pytest.mark.parametrize(
        "build, pieces",
        [(strips, STRIP_PIECES), (slabs, SLAB_PIECES)],
    )
    def test_a_function_that_jumps_is_given_piece_by_piece(self, build, pieces, rng):
        """Read back inside each layer and on each side of an interface,
        above taking the shallower piece."""
        space = build(depth_modes=12)
        field = space.project_function(pieces)
        points = random_points(space, [0, 1, len(pieces) - 1, 0], rng)
        exact = [pieces[space.layer_index(p)](p) for p in points]
        assert np.allclose(space.evaluate(field, points), exact, atol=1e-8)
        for k, depth in enumerate(space.interfaces):
            on = at_depth(depth, lateral_positions(space, 3, rng))
            for side, piece in (("above", pieces[k]), ("below", pieces[k + 1])):
                assert np.allclose(
                    space.evaluate(field, on, side=side),
                    [piece(p) for p in on],
                    atol=1e-8,
                )
            with pytest.raises(ValueError, match="interface"):
                space.evaluate(field, on)
        with pytest.raises(ValueError, match="layer by layer"):
            space.project_function(pieces[:1])

    def test_one_function_serves_every_layer_when_it_does_not_jump(self):
        space = strips(depth_modes=12)

        def smooth(p):
            return np.exp(-p[0]) * np.cos(np.pi * p[1] + 0.3)

        field = space.project_function(smooth)
        points = at_depth(0.3, [0.2, 1.1]) + at_depth(0.6, [0.7]) + [(1.0, 1.9)]
        for side in ("above", "below"):
            assert np.allclose(
                space.evaluate(field, points, side=side),
                [smooth(p) for p in points],
                atol=1e-8,
            )

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_evaluation_is_the_layers_own(self, build, rng):
        space = build()
        x = space.random(rng=rng)
        which = [0, 1, 1, 0]
        points = random_points(space, which, rng)
        values = space.evaluate(x, points)
        for value, point, i in zip(values, points, which):
            assert value == pytest.approx(space.layers[i].evaluate(x[i], [point])[0])
        rows = space.basis_matrix(points)
        assert rows.shape == (4, space.dim)
        assert np.all(rows[0, space.columns(1)] == 0.0)
        assert np.allclose(rows @ space.to_components(x), values)

    @pytest.mark.parametrize("build", GEOMETRIES)
    @pytest.mark.parametrize("route", ["dense", "local", "nufft"])
    def test_every_route_and_its_adjoint(self, build, route, rng):
        """Each route against the rows of the basis matrix, points on an
        interface among them, and each adjoint against its action: to
        rounding where the lateral sums are direct, to the accuracy asked
        of the non-uniform transforms where they are not."""
        space = build()
        depth = space.interfaces[0]
        points = random_points(space, [0, 1, 0, 1, 1], rng)
        points += at_depth(depth, lateral_positions(space, 2, rng))
        for side in ("above", "below"):
            operator = space.point_evaluation_operator(points, side=side, route=route)
            check_operator(operator, rng=rng, rtol=1e-9)
            x = space.random(rng=rng)
            exact = space.basis_matrix(points, side=side) @ space.to_components(x)
            tolerance = 1e-9 if route == "nufft" else 1e-12
            assert np.allclose(
                operator(x), exact, rtol=0.0, atol=tolerance * np.abs(exact).max()
            )
            assert np.allclose(
                space.evaluate(x, points, side=side),
                exact,
                rtol=0.0,
                atol=1e-12 * np.abs(exact).max(),
            )
            assert operator(x)[-1] == pytest.approx(
                space.dirac(points[-1], side=side)(x), rel=1e-8
            )
        with pytest.raises(ValueError, match="interface"):
            space.point_evaluation_operator(points, route=route)
        with pytest.raises(ValueError, match="At least one"):
            space.point_evaluation_operator([], route=route)

    @pytest.mark.parametrize("build", [strips, slabs])
    def test_auto_is_dense_within_the_limit_and_nufft_beyond(
        self, build, rng, monkeypatch
    ):
        space = build()
        points = random_points(space, [0, 1] * 10, rng)
        x = space.random(rng=rng)
        within = space.point_evaluation_operator(points)
        assert within.known_matrix(form="galerkin") is not None
        monkeypatch.setattr(
            spectral_base, "_DENSE_EVALUATION_LIMIT", 8 * len(points) * space.dim - 1
        )
        beyond = space.point_evaluation_operator(points, epsilon=1e-6)
        assert beyond.known_matrix(form="galerkin") is None
        named = space.point_evaluation_operator(points, route="nufft", epsilon=1e-6)
        assert np.array_equal(beyond(x), named(x))
        scale = np.abs(within(x)).max()
        assert 1e-13 < np.abs(beyond(x) - within(x)).max() / scale < 1e-4
        with pytest.raises(ValueError, match="route"):
            space.point_evaluation_operator(points, route="fast")

    def test_each_layer_keeps_its_own_guard(self):
        """A rough layer refuses its points by every route, and the jump at
        its interface, and lets the other's through."""
        space = strips([2.0, 0.0, 2.0])
        space.dirac((0.1, 0.5))
        space.dirac((0.3, 0.5), side="above")
        with pytest.raises(ValueError, match="above 1"):
            space.dirac((0.45, 0.5))
        with pytest.raises(ValueError, match="above 1"):
            space.dirac((0.3, 0.5), side="below")
        for route in ("auto", "dense", "local", "nufft"):
            with pytest.raises(ValueError, match="above 1"):
                space.point_evaluation_operator([(0.1, 0.5), (0.45, 0.5)], route=route)
            space.point_evaluation_operator(
                [(0.1, 0.5), (0.45, 0.5)], route=route, unsafe=True
            )
        with pytest.raises(ValueError, match="above 1"):
            space.jump_operator(0, positions=[0.5])
        space.jump_operator(0, positions=[0.5], unsafe=True)

    def test_a_periodic_axis_wraps_and_a_bounded_one_refuses(self, rng):
        space = strips()
        x = space.random(rng=rng)
        assert space.evaluate(x, [(0.4, -0.3)]) == pytest.approx(
            space.evaluate(x, [(0.4, 1.7)]), abs=1e-12
        )
        bounded = bounded_strips()
        for route in ("dense", "local", "nufft"):
            with pytest.raises(ValueError, match="lateral axis 0"):
                bounded.point_evaluation_operator([(0.6, 2.3)], route=route)
        with pytest.raises(ValueError, match="lateral axis 0"):
            bounded.jump_operator(0, positions=[2.3])
        box = bounded_slabs()
        with pytest.raises(ValueError, match="lateral axis 0"):
            box.jump_operator(0, positions=[(1.2, 0.0)])
        y = box.random(rng=rng)
        assert box.evaluate(y, [(0.7, 0.5, 0.75)]) == pytest.approx(
            box.evaluate(y, [(0.7, 0.5, -0.75)]), abs=1e-12
        )


class TestJumps:
    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_the_jump_is_above_less_below(self, build, rng):
        """The value on the shallower side less the value on the deeper, at
        the interface's depth and the positions given."""
        space = build()
        positions = lateral_positions(space, 4, rng)
        x = space.random(rng=rng)
        for k, depth in enumerate(space.interfaces):
            jump = space.jump_operator(k, positions=positions)
            check_operator(jump, rng=rng)
            check_traits(jump, rng=rng)
            on = at_depth(depth, positions)
            above = space.evaluate(x, on, side="above")
            below = space.evaluate(x, on, side="below")
            assert np.allclose(
                jump(x), above - below, rtol=0.0, atol=1e-12 * np.abs(above).max()
            )
            # Above is the shallower layer: the layer listed first.
            shallower = space.layers[k].evaluate(x[k], on)
            assert np.allclose(above, shallower, atol=1e-12 * np.abs(above).max())

    @pytest.mark.parametrize(
        "build, pieces",
        [(strips, STRIP_PIECES), (slabs, SLAB_PIECES)],
    )
    def test_the_jump_of_a_known_discontinuity(self, build, pieces, rng):
        """A function that jumps by a known amount: the shallower piece less
        the deeper at the interface, which is minus the jump along the
        coordinate."""
        space = build(depth_modes=12)
        field = space.project_function(pieces)
        positions = lateral_positions(space, 5, rng)
        for k, depth in enumerate(space.interfaces):
            on = at_depth(depth, positions)
            exact = np.array([pieces[k](p) - pieces[k + 1](p) for p in on])
            jump = space.jump_operator(k, positions=positions)(field)
            assert np.allclose(jump, exact, atol=1e-8)
            assert np.abs(exact).min() > 0.1

    def test_the_positions_are_required_and_shaped(self, rng):
        space, box = strips(), slabs()
        x = space.random(rng=rng)
        assert np.allclose(
            space.jump_operator(0, positions=[0.2, 1.4])(x),
            space.jump_operator(0, positions=[(0.2,), (1.4,)])(x),
        )
        for layered, positions, empty in (
            (space, [0.2], []),
            (box, [(0.2, 0.3)], np.empty((0, 2))),
        ):
            with pytest.raises(ValueError, match="positions"):
                layered.jump_operator(0)
            with pytest.raises(ValueError, match="At least one position"):
                layered.jump_operator(0, positions=empty)
            with pytest.raises(IndexError):
                layered.jump_operator(len(layered.interfaces), positions=positions)
        with pytest.raises(ValueError, match="pairs"):
            box.jump_operator(0, positions=[0.2, 0.3, 0.4])
        with pytest.raises(ValueError, match="pairs"):
            box.jump_operator(0, positions=[(0.2, 0.3, 0.4)])

    @pytest.mark.parametrize("build", [strips, slabs])
    def test_it_claims_to_be_onto_only_while_it_can_be(self, build, rng):
        """No more positions than lateral Fourier functions, and it claims
        to be onto, which the rank confirms; more, and it claims nothing."""
        space = build()
        count = space.layers[0].transform.size
        within = space.jump_operator(0, positions=grid_positions(space))
        assert within.codomain.dim == count
        assert Traits.SURJECTIVE in within.traits
        check_traits(within, rng=rng)
        beyond = space.jump_operator(
            0, positions=lateral_positions(space, count + 1, rng)
        )
        assert Traits.SURJECTIVE not in beyond.traits
        assert np.linalg.matrix_rank(beyond.matrix(form="galerkin")) == count

    @pytest.mark.parametrize("build", [strips, slabs])
    def test_conditioning_on_no_jump_makes_the_medium_continuous_there(
        self, build, rng
    ):
        """The layers are independent under the prior, which is what a
        discontinuity means; conditioned on a vanishing jump at some
        positions every draw is continuous across the interface there, and
        still jumps far from them and at any other interface."""
        space = build()
        prior = space.sobolev_measure(2.0, pointwise_std=0.3)
        # Three positions in one half of the period, three in the other,
        # several length scales away.
        positions, elsewhere = [0.2, 0.3, 0.4], [1.2, 1.4, 1.6]
        if space.spatial_dimension == 3:
            positions = list(zip(positions, [0.2, 0.3, 0.2]))
            elsewhere = list(zip(elsewhere, [0.9, 1.1, 0.8]))
        last = len(space.interfaces) - 1
        welded = prior.condition(
            space.jump_operator(last, positions=positions), np.zeros(3)
        )
        draws = [welded.sample(rng=rng) for _ in range(30)]
        there = np.array(
            [space.jump_operator(last, positions=positions)(x) for x in draws]
        )
        assert np.abs(there).max() < 1e-8
        between = np.array(
            [space.jump_operator(last, positions=elsewhere)(x) for x in draws]
        )
        assert between.std() > 0.1
        if last:
            other = np.array(
                [space.jump_operator(0, positions=positions)(x) for x in draws]
            )
            assert other.std() > 0.2
        free = np.array(
            [
                space.jump_operator(last, positions=positions)(prior.sample(rng=rng))
                for _ in range(30)
            ]
        )
        assert free.std() > 0.2

    @pytest.mark.parametrize("build", [strips, slabs])
    def test_welded_at_every_grid_point_the_interface_is_continuous_across(
        self, build, rng
    ):
        """The layers share the lateral grid, so the jump is a lateral
        Fourier series with one term per grid point: conditioned to vanish
        at every point of the grid, it vanishes at any position."""
        space = build()
        prior = space.sobolev_measure(2.0, pointwise_std=0.3)
        jump = space.jump_operator(0, positions=grid_positions(space))
        welded = prior.condition(jump, np.zeros(jump.codomain.dim))
        anywhere = space.jump_operator(0, positions=lateral_positions(space, 8, rng))
        jumps = np.array([anywhere(welded.sample(rng=rng)) for _ in range(20)])
        assert np.abs(jumps).max() < 1e-7


class TestAlgebraAndMeasures:
    def test_the_integral_is_over_the_whole_medium(self, rng):
        """In closed form for a function that jumps, and as the sum of the
        layers' own integrals for any field."""
        space = strips(depth_modes=12)
        field = space.project_function(STRIP_PIECES)
        exact = 2.0 * sum(STRIP_DEPTH_INTEGRALS)
        integral = space.integral_functional()
        assert integral(field) == pytest.approx(exact, rel=1e-10)
        one = space.project_function(lambda p: 1.0)
        assert integral(one) == pytest.approx(space.domain_volume, rel=1e-10)
        for build in GEOMETRIES:
            layered = build()
            x = layered.random(rng=rng)
            by_layer = sum(
                layer.integral_functional()(part)
                for layer, part in zip(layered.layers, x)
            )
            assert layered.integral_functional()(x) == pytest.approx(
                by_layer, rel=1e-12
            )

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
        """No covariance between points of two layers, and at each point the
        variance of its layer's own calibrated prior: to rounding, and to
        the few per cent to which a calibration holds off the grid's points
        under the default truncation."""
        space = build()
        deviations = [0.2, 0.4] + [0.3] * (len(space) - 2)
        measure = space.sobolev_measure(2.5, pointwise_std=deviations)
        assert measure.domain is space
        points = random_points(space, [0, 1], rng)
        look = space.point_evaluation_operator(points)
        covariance = (look @ measure.covariance @ look.adjoint).matrix(
            form="components"
        )
        assert covariance[0, 1] == pytest.approx(0.0, abs=1e-12)
        for i, point in enumerate(points):
            layer = space.layers[i]
            own = layer.sobolev_measure(2.5, pointwise_std=deviations[i])
            alone = layer.point_evaluation_operator([point])
            variance = (alone @ own.covariance @ alone.adjoint).matrix(
                form="components"
            )[0, 0]
            assert covariance[i, i] == pytest.approx(variance, rel=1e-10)
        assert np.allclose(np.sqrt(np.diag(covariance)), [0.2, 0.4], rtol=5e-2)

    @pytest.mark.parametrize("build", [strips, slabs])
    def test_the_moments_match_the_samples(self, build, rng):
        space = build(depth_modes=2)
        deviations = [0.2, 0.4] + [0.3] * (len(space) - 2)
        check_measure(
            space.sobolev_measure(2.5, pointwise_std=deviations),
            rng=rng,
            samples=2000,
            rtol=0.2,
        )

    @pytest.mark.parametrize("build", [strips, bounded_slabs])
    def test_the_prior_is_each_layers_own(self, build, rng):
        """Every argument given once or layer by layer, each layer's
        covariance the one its own ``sobolev_measure`` builds."""
        space = build()
        count = len(space)
        orders = [2.0, 2.5, 3.0][:count]
        amplitudes = [0.5, 1.0, 2.0][:count]
        scales = [0.25, None, 0.5][:count]
        deviations = [0.2, None, 0.3][:count]
        measure = space.sobolev_measure(
            orders,
            amplitude=amplitudes,
            length_scale=scales,
            lateral_length_scale=0.35,
            pointwise_std=deviations,
        )
        x = space.random(rng=rng)
        got = measure.covariance(x)
        for i, layer in enumerate(space.layers):
            want = layer.sobolev_measure(
                orders[i],
                amplitude=amplitudes[i],
                length_scale=scales[i],
                lateral_length_scale=0.35,
                pointwise_std=deviations[i],
            ).covariance(x[i])
            gap = layer.norm(layer.subtract(got[i], want)) / layer.norm(want)
            assert gap < 1e-12
        with pytest.raises(ValueError, match="layer by layer"):
            space.sobolev_measure([2.0] * (count + 1))
        with pytest.raises(ValueError, match="layer by layer"):
            space.sobolev_measure(2.0, lateral_length_scale=[0.3] * (count + 1))


# ------------------------------------------------------------------------- #
#                                 Plotting                                  #
# ------------------------------------------------------------------------- #


def drawn(mesh, /):
    """The coordinates and values of a gouraud mesh: ``(rows, columns, 2)``
    and ``(rows, columns)``."""
    where = np.asarray(mesh.get_coordinates())
    return where, np.asarray(mesh.get_array()).reshape(where.shape[:2])


def meshes(ax, /):
    """The color meshes on the axes, in the order drawn."""
    return [c for c in ax.collections if hasattr(c, "get_clim")]


def grid_index(layer, axis, coordinates, /):
    """Which point of the lateral grid along an axis each coordinate is, one
    period on read as the period's first point."""
    grid = layer.lateral_axes[axis]
    offset = (np.asarray(coordinates)[:, None] - grid[None, :]) / (
        layer.lateral_domain.lengths[axis]
    )
    gaps = np.abs(offset - np.round(offset))
    index = np.argmin(gaps, axis=1)
    assert np.max(gaps[np.arange(index.size), index]) < 1e-12
    return index


def bound_columns(layer, axis, coordinates, /):
    """Which of the coordinates drawn along a lateral axis are the bounds of
    a bounded domain, and not points of the grid. No bound of these
    fixtures is a point of the grid, so along a bounded axis they are the
    first coordinate and the last, and they are the bounds."""
    coordinates = np.asarray(coordinates)
    edges = np.zeros(coordinates.size, dtype=bool)
    if layer.lateral_domain.padding[axis] > 0.0:
        bounds = layer.lateral_domain.bounds[axis]
        assert tuple(coordinates[[0, -1]]) == tuple(bounds)
        gaps = np.abs(layer.lateral_axes[axis][:, None] - np.asarray(bounds)[None, :])
        assert np.min(gaps) > 1e-3
        edges[[0, -1]] = True
    return edges


def interior_rows(layer, /):
    """The rows of a layer's grid array that are its domain's nodes in depth."""
    lateral = tuple(range(1, layer.interior_mask.ndim))
    return np.flatnonzero(layer.interior_mask.any(axis=lateral))


def jumping(space, /):
    """A field that jumps at every interface, by a known amount."""
    if space.spatial_dimension == 2:
        return space.project_function(STRIP_PIECES[: len(space.layers)])
    return space.project_function(SLAB_PIECES[: len(space.layers)])


class TestPlotting:
    """Each layer drawn from its own values on one color scale, so that the
    jump at an interface is drawn as one: the two rows at the interface's
    depth hold the two sides of it, which differ by what the jump operator
    says, a second route."""

    @pytest.fixture(autouse=True)
    def _headless(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot

        yield
        pyplot.close("all")

    @pytest.mark.parametrize("build", [strips, bounded_strips])
    def test_a_layered_strip_is_drawn_layer_by_layer(self, build):
        """Each layer from its own grid values, and across a bounded domain
        a column at each bound evaluated from the layer's own values: the
        jump at an interface is as sharp in those columns as in the rest,
        the jump operator's there too."""
        from pygeoinf import plotting

        space = build()
        x = jumping(space)
        ax, mappable = plotting.plot(space, x, colorbar_label="field")
        drawn_meshes = meshes(ax)
        assert len(drawn_meshes) == len(space.layers)
        assert len({mesh.get_clim() for mesh in drawn_meshes}) == 1
        assert mappable is drawn_meshes[-1]
        pictures = [drawn(mesh) for mesh in drawn_meshes]
        for (where, values), layer, part in zip(pictures, space.layers, x):
            depths, across = where[:, 0, 1], where[0, :, 0]
            assert np.array_equal(depths, layer.interior_depths)
            edges = bound_columns(layer, 0, across)
            columns = grid_index(layer, 0, across[~edges])
            expected = part[np.ix_(interior_rows(layer), columns)]
            assert np.array_equal(values[:, ~edges], expected)
            assert edges.any() == (build is bounded_strips)
            if edges.any():
                points = [(depth, a) for depth in depths for a in across[edges]]
                evaluated = layer.evaluate(part, points).reshape(depths.size, -1)
                scale = np.max(np.abs(evaluated))
                assert np.max(np.abs(values[:, edges] - evaluated)) <= 1e-12 * scale
        for i, depth in enumerate(space.interfaces):
            (upper, above), (lower, below) = pictures[i], pictures[i + 1]
            assert upper[-1, 0, 1] == lower[0, 0, 1] == pytest.approx(depth)
            positions = upper[-1, :, 0]
            jump = space.jump_operator(i, positions=positions)(x)
            scale = np.max(np.abs(jump))
            assert np.max(np.abs(above[-1] - below[0] - jump)) <= 1e-12 * scale
            # Not smeared: the whole jump is there, at the interface, the
            # columns at the bounds of a bounded domain included.
            assert np.min(np.abs(jump)) > 0.5
            points = at_depth(depth, positions)
            sides = [space.evaluate(x, points, side=s) for s in ("above", "below")]
            scale = np.max(np.abs(sides))
            assert np.max(np.abs(above[-1] - sides[0])) <= 1e-12 * scale
            assert np.max(np.abs(below[0] - sides[1])) <= 1e-12 * scale
        # Depth downward, over the medium, with every boundary marked once.
        ends = space.layer_bounds
        assert ax.yaxis_inverted() and ax.get_ylim() == (ends[-1][1], ends[0][0])
        across = sorted(
            line.get_ydata()[0]
            for line in ax.lines
            if line.get_ydata()[0] == line.get_ydata()[1]
        )
        assert across == sorted({end for pair in ends for end in pair})

    def test_the_interfaces_are_marked_on_request(self):
        from pygeoinf import plotting

        space = strips()
        x = jumping(space)
        ax, _ = plotting.plot(space, x, outline=False)
        assert not ax.lines
        with pytest.raises(ValueError, match="padding"):
            plotting.plot(space, x, padding=True)
        with pytest.raises(ValueError, match="parts"):
            plotting.plot(space, x[:2])

    @pytest.mark.parametrize("build", [slabs, bounded_slabs])
    def test_a_section_of_layered_slabs_is_sharp_at_the_interface(self, build):
        """Along x at a y of the grid: each layer evaluated on that line, the
        grid's values there to rounding, at the bounds of a bounded domain
        the layer's field there, and the jump between the two rows at the
        interface the jump operator's."""
        from pygeoinf import plotting

        space = build()
        x = jumping(space)
        first = space.layers[0]
        j = 2
        ax, _ = plotting.plot_section(space, x, y=first.y[j], symmetric=True)
        drawn_meshes = meshes(ax)
        assert len(drawn_meshes) == 2
        assert len({mesh.get_clim() for mesh in drawn_meshes}) == 1
        pictures = [drawn(mesh) for mesh in drawn_meshes]
        for (where, values), layer, part in zip(pictures, space.layers, x):
            depths, across = where[:, 0, 1], where[0, :, 0]
            assert np.array_equal(depths, layer.interior_depths)
            edges = bound_columns(layer, 0, across)
            columns = grid_index(layer, 0, across[~edges])
            expected = part[:, :, j][np.ix_(interior_rows(layer), columns)]
            scale = np.max(np.abs(expected))
            assert np.max(np.abs(values[:, ~edges] - expected)) <= 1e-12 * scale
            assert edges.any() == (build is bounded_slabs)
            if edges.any():
                points = [(d, a, first.y[j]) for d in depths for a in across[edges]]
                evaluated = layer.evaluate(part, points).reshape(depths.size, -1)
                scale = np.max(np.abs(evaluated))
                assert np.max(np.abs(values[:, edges] - evaluated)) <= 1e-12 * scale
        (upper, above), (lower, below) = pictures
        (depth,) = space.interfaces
        assert upper[-1, 0, 1] == lower[0, 0, 1] == pytest.approx(depth)
        positions = np.column_stack(
            [upper[-1, :, 0], np.full(upper.shape[1], first.y[j])]
        )
        jump = space.jump_operator(0, positions=positions)(x)
        scale = np.max(np.abs(jump))
        assert np.max(np.abs(above[-1] - below[0] - jump)) <= 1e-12 * scale
        assert np.min(np.abs(jump)) > 0.3
        across = {
            line.get_ydata()[0]
            for line in ax.lines
            if line.get_ydata()[0] == line.get_ydata()[1]
        }
        assert across == {end for pair in space.layer_bounds for end in pair}

    @pytest.mark.parametrize("build", [slabs, bounded_slabs])
    def test_a_map_at_an_interface_takes_the_side_named(self, build):
        """Above is the shallower layer: its map at the interface is that
        layer's base on its grid, below is the deeper layer's top; with no
        side named the map is refused, and with none given it is the top of
        the medium."""
        from pygeoinf import plotting

        space = build()
        x = jumping(space)
        (depth,) = space.interfaces
        with pytest.raises(ValueError, match="interface"):
            plotting.plot(space, x, depth=depth)
        rows = {
            "above": (0, interior_rows(space.layers[0])[-1]),
            "below": (1, interior_rows(space.layers[1])[0]),
        }
        maps = {}
        for side, (i, row) in rows.items():
            _, mesh = plotting.plot(space, x, depth=depth, side=side)
            where, values = drawn(mesh)
            layer = space.layers[i]
            # Bounded along x or not at all: the bounds are the first and last
            # columns, evaluated on the side named.
            edges = bound_columns(layer, 0, where[:, 0, 0])
            index = np.ix_(
                grid_index(layer, 0, where[~edges, 0, 0]),
                grid_index(layer, 1, where[0, :, 1]),
            )
            expected = x[i][row][index]
            scale = np.max(np.abs(expected))
            assert np.max(np.abs(values[~edges] - expected)) <= 1e-12 * scale
            if edges.any():
                points = at_depth(depth, where[edges].reshape(-1, 2))
                evaluated = space.evaluate(x, points, side=side)
                scale = np.max(np.abs(evaluated))
                difference = values[edges].ravel() - evaluated
                assert np.max(np.abs(difference)) <= 1e-12 * scale
            maps[side] = values
        assert np.min(np.abs(maps["above"] - maps["below"])) > 0.3
        _, mesh = plotting.plot(space, x)
        where, values = drawn(mesh)
        top = x[0][interior_rows(space.layers[0])[0]]
        layer = space.layers[0]
        on = ~bound_columns(layer, 0, where[:, 0, 0])
        index = np.ix_(
            grid_index(layer, 0, where[on, 0, 0]), grid_index(layer, 1, where[0, :, 1])
        )
        assert np.max(np.abs(values[on] - top[index])) <= 1e-12 * np.max(np.abs(top))

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_a_profile_breaks_at_every_jump(self, build, rng):
        from pygeoinf import plotting

        space = build()
        x = space.random(rng=rng)
        position = np.atleast_1d(lateral_positions(space, 1, rng)[0])
        ax, _ = plotting.plot_profile(space, x, *position, label="a field")
        assert len(ax.lines) == len(space.layers)
        assert len({line.get_color() for line in ax.lines}) == 1
        assert [line.get_label() for line in ax.lines].count("a field") == 1
        for upper, lower, depth in zip(ax.lines[:-1], ax.lines[1:], space.interfaces):
            assert upper.get_ydata()[-1] == pytest.approx(depth)
            assert lower.get_ydata()[0] == pytest.approx(depth)
            point = [(depth, *position)]
            assert upper.get_xdata()[-1] == pytest.approx(
                space.evaluate(x, point, side="above")[0]
            )
            assert lower.get_xdata()[0] == pytest.approx(
                space.evaluate(x, point, side="below")[0]
            )
        ends = space.layer_bounds
        assert ax.get_ylim() == (ends[-1][1], ends[0][0])

    def test_points_and_axes(self):
        from pygeoinf import plotting

        ax, collection = plotting.plot_points(strips(), [(0.4, 2.5)])
        assert np.allclose(collection.get_offsets(), [[0.5, 0.4]])
        assert ax.yaxis_inverted()
        _, collection = plotting.plot_points(slabs(), [(0.4, 0.5, 2.0)])
        assert np.allclose(collection.get_offsets(), [[0.5, 0.5]])
        figure, axes = plotting.subplots(slabs(), columns=2)
        assert len(axes) == 2 and not hasattr(axes[0], "projection")

    def test_a_balls_own_pictures_are_refused(self):
        """A shell and a planet model are a ball's: a layered slab has
        neither, and is refused naming what has them."""
        from pygeoinf import plotting

        space = slabs()
        x = jumping(space)
        with pytest.raises(TypeError, match="balls"):
            plotting.plot_shell(space, x)
        with pytest.raises(TypeError, match="balls"):
            plotting.planet_model(space, x)
