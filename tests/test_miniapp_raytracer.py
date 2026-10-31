"""
The ray tracer the mini-apps share keeps its answers.

``miniapps/raytracer`` traces rays in a planet whose velocity depends on
radius alone, piece by piece between given radii, between ends at any
radius. Checked here against closed forms in the flattened linear gradient,
where every ray is an arc of a circle, and in two flattened linear layers
with a jump in velocity between them: interfaces where the velocity does not
jump change nothing but put nodes where the ray crosses them; a source at
depth gives the circle through its two ends, every node on it and the time
along the arc; a real discontinuity gives the closed form of the two layers
on both sides of it, and the integral over the travel time across it, each
side with its own slowness, is second order in the spacing; and a ray the
tracer does not trace -- reflected at the jump, reaching a distance more than
once, or leaving the source upwards -- is refused rather than answered.

The closed form's time between two points at depth is the tracer's, with the
turning point between them or not. Along a traced ray, dynamic ray tracing
gives the Hessians of the travel time across it from either end: out of the
ray's plane they are the closed forms of a radial medium; in the plane a
combination of the integrated quantities is conserved; along a chord in a
constant velocity they are ``1 / (v l)`` and ``1 / (v (L - l))``; in the
flattened gradient their sums are the curvature of the closed form's detour
time through a point off the ray, in three dimensions; and traced the other
way the two swap. Needs only NumPy and SciPy.
"""

from __future__ import annotations

import importlib
import pathlib
import sys

import numpy as np
import pytest

TRACER = pathlib.Path(__file__).resolve().parents[1] / "miniapps" / "raytracer"


@pytest.fixture(scope="module")
def rays():
    """The shared tracer, imported with its directory on the path for the
    duration, as the apps import it."""
    if not (TRACER / "raytracer.py").exists():
        pytest.skip(f"The shared tracer is not at {TRACER}.")
    sys.path.insert(0, str(TRACER))
    try:
        return importlib.import_module("raytracer")
    finally:
        sys.path.remove(str(TRACER))


#: The flattened linear gradient ``v(r) = r (1 - b ln r)`` with ``v(0.55) =
#: 1.8``, as in the tomography app's tests.
FLAT_V0 = 1.0
FLAT_GRADIENT = (1.8 / 0.55 - 1.0) / -np.log(0.55)
#: Interfaces at which the velocity does not jump.
CUTS = [(0.5,), (0.4, 0.7, 0.9), (0.3, 0.6, 0.8, 0.95, 0.99)]


@pytest.fixture(scope="module")
def flattened(rays):
    velocity = rays.flattened_linear_gradient(FLAT_V0, FLAT_GRADIENT)
    whole = rays.RadialProfile(velocity, bottom=0.05)
    return whole, rays.CircularArcs(FLAT_V0, FLAT_GRADIENT)


def _split(rays, whole, cut):
    return rays.RadialProfile(whole.velocity, bottom=whole.bottom, discontinuities=cut)


# ------------------------------------------------------------------------- #
#                  Interfaces where nothing happens                         #
# ------------------------------------------------------------------------- #


@pytest.mark.parametrize("cut", CUTS)
@pytest.mark.parametrize("source", [None, 0.9, 0.75])
@pytest.mark.parametrize("distance", [5.0, 20.0, 45.0, 80.0, 120.0, 150.0])
def test_interfaces_where_nothing_jumps_change_nothing(
    rays, flattened, cut, source, distance
):
    """Split at radii where the velocity is smooth, the profile gives the
    rays of the profile in one piece: the same ray parameter, time and
    length, to the rounding the turning point amplifies; every node on the
    closed form's circle, with one at each crossing of a cut, on the way
    down from the source and on the way up to the receiver; and the
    integrals along it and over its travel time as accurate as the unsplit
    ray's."""
    whole, arcs = flattened
    split = _split(rays, whole, cut)
    options = dict(source_radius=source)
    try:
        expected = whole.ray_parameter(distance, **options)
    except ValueError:
        # From a source this deep no ray turns before reaching that far.
        with pytest.raises(ValueError, match="nearest"):
            split.ray_parameter(distance, **options)
        return
    p = split.ray_parameter(distance, **options)
    # The largest differences are 5e-11 in p and 4e-10 in the time and the
    # length, from rays turning near an interface.
    assert p == pytest.approx(expected, rel=5e-10)
    length = split.path_length(p, **options)
    assert length == pytest.approx(whole.path_length(expected, **options), rel=4e-9)
    ray, reference = (
        profile.trace(30.0, 30.0 + distance, **options) for profile in (split, whole)
    )
    assert ray.travel_time == pytest.approx(reference.travel_time, rel=4e-9)
    depth = 0.0 if source is None else -np.log(source)
    _, (xc, zc, radius), time = arcs.from_depth(np.radians(distance), depth)
    x, z = np.radians(ray.nodes[:, 1] - 30.0), -np.log(ray.nodes[:, 0])
    # The largest distance off the circle is 8.7e-11.
    assert np.abs(np.hypot(x - xc, z - zc) - radius).max() < 1e-9
    turning = split.turning_radius(p, **options)
    ends = (1.0 if source is None else source, 1.0)
    for radius in cut:
        crossings = sum(turning < radius < end for end in ends)
        assert np.count_nonzero(ray.nodes[1:-1, 0] == radius) == crossings
    # At 301 nodes the trapezoid rule along the polyline is out by at most
    # 6.4e-7 in the length and, over the travel time, 1.56e-5 against the
    # closed form, the unsplit ray's 1.55e-5.
    assert ray.weights.sum() == pytest.approx(length, rel=1.5e-6)
    for traced in (ray, reference):
        assert traced.time_weights.sum() == pytest.approx(time, rel=3e-5)


def test_one_velocity_for_each_piece_is_one_velocity(rays, flattened):
    """The same function given once for every piece, or once for each, is
    the same profile, to the last bit."""
    whole, _ = flattened
    cut = CUTS[1]
    once = _split(rays, whole, cut)
    each = rays.RadialProfile(
        [whole.velocity] * (len(cut) + 1), bottom=whole.bottom, discontinuities=cut
    )
    for source in (None, 0.8):
        a = once.trace(10.0, 100.0, source_radius=source)
        b = each.trace(10.0, 100.0, source_radius=source)
        assert np.array_equal(a.nodes, b.nodes)
        assert a.travel_time == b.travel_time
    radii = np.linspace(0.05, 1.0, 41)
    assert np.array_equal(each.eta(radii), whole.eta(radii))


# ------------------------------------------------------------------------- #
#                         A source at depth                                 #
# ------------------------------------------------------------------------- #


@pytest.mark.parametrize("cut", [(), *CUTS])
@pytest.mark.parametrize("source", [0.95, 0.85, 0.7])
@pytest.mark.parametrize("distance", [45.0, 80.0, 120.0])
def test_a_source_at_depth_is_the_circle_through_its_ends(
    rays, flattened, cut, source, distance
):
    """From a source at depth to a receiver on the surface, the ray of the
    flattened gradient is the arc through the two of the circle centered at
    the height ``v0 / g`` above the surface: every node lies on it, and the
    time is the time along the arc."""
    whole, arcs = flattened
    profile = _split(rays, whole, cut)
    reach, depth = np.radians(distance), -np.log(source)
    p_exact, (xc, zc, radius), time = arcs.from_depth(reach, depth)
    assert 0.0 < xc < reach
    p = profile.ray_parameter(distance, source_radius=source)
    # The largest differences are 1.1e-10 in p, 3.3e-11 in the time, 4.2e-11
    # in the turning radius and 7.6e-11 off the circle.
    assert p == pytest.approx(p_exact, rel=1e-9)
    assert profile.travel_time(p, source_radius=source) == pytest.approx(
        time, rel=5e-10
    )
    assert profile.turning_radius(p, source_radius=source) == pytest.approx(
        np.exp(-(zc + radius)), abs=5e-10
    )
    ray = profile.trace(30.0, 30.0 + distance, source_radius=source)
    x = np.radians(ray.nodes[:, 1] - 30.0)
    z = -np.log(ray.nodes[:, 0])
    assert np.abs(np.hypot(x - xc, z - zc) - radius).max() < 1e-9
    assert ray.travel_time == pytest.approx(time, rel=5e-10)
    assert np.array_equal(ray.nodes[0], [source, 30.0])
    assert ray.nodes[-1] == pytest.approx([1.0, 30.0 + distance], abs=5e-8)
    # The deepest node is near the bottom of the circle, between the ends.
    deepest = np.argmin(ray.nodes[:, 0])
    assert 0 < deepest < len(ray.nodes) - 1
    assert x[deepest] == pytest.approx(xc, abs=2.0 * np.diff(x).max())


def test_the_ray_is_the_same_either_way_along_it(rays, flattened):
    """A source at depth and a receiver on the surface, or the other way
    round: the same time, and the same nodes in the opposite order."""
    whole, _ = flattened
    profile = _split(rays, whole, CUTS[1])
    out = profile.trace(20.0, 95.0, source_radius=0.8)
    back = profile.trace(95.0, 20.0, receiver_radius=0.8)
    assert back.travel_time == pytest.approx(out.travel_time, rel=1e-12)
    # Each end's angle is the other's plus the distance the root find
    # reaches, which is the one asked for to 2e-9 degrees.
    assert np.allclose(back.nodes[::-1], out.nodes, rtol=0.0, atol=2e-8)


def test_both_ends_at_depth(rays, flattened):
    """Two ends at depth: the legs from the turning point to each, against
    the circle through both, centered at the height ``v0 / g``."""
    whole, arcs = flattened
    profile = _split(rays, whole, CUTS[2])
    ends, distance = (0.9, 0.8), 70.0
    p = profile.ray_parameter(distance, source_radius=ends[0], receiver_radius=ends[1])
    # The circle through (0, z1) and (X, z2) with its center at z = -v0 / g.
    height = FLAT_V0 / FLAT_GRADIENT
    (z1, z2), reach = -np.log(ends), np.radians(distance)
    xc = (reach**2 + (z2 + height) ** 2 - (z1 + height) ** 2) / (2.0 * reach)
    radius = np.hypot(xc, z1 + height)
    assert p == pytest.approx(1.0 / (radius * FLAT_GRADIENT), rel=1e-9)
    time = sum(np.arccosh(1.0 / (p * (FLAT_V0 + FLAT_GRADIENT * z))) for z in (z1, z2))
    ray = profile.trace(0.0, distance, source_radius=ends[0], receiver_radius=ends[1])
    assert ray.travel_time == pytest.approx(time / FLAT_GRADIENT, rel=5e-10)
    assert ray.nodes[0, 0] == ends[0] and ray.nodes[-1, 0] == ends[1]


def test_a_ray_that_does_not_turn_is_refused(rays, flattened):
    """Near enough, a ray from a source at depth leaves it upwards and never
    turns; it is not traced, and an end must lie within the profile."""
    whole, _ = flattened
    with pytest.raises(ValueError, match="nearest"):
        whole.ray_parameter(10.0, source_radius=0.75)
    with pytest.raises(ValueError, match="above the bottom"):
        whole.trace(0.0, 30.0, source_radius=1.2)


# ------------------------------------------------------------------------- #
#                   A real discontinuity: two flat layers                   #
# ------------------------------------------------------------------------- #

#: Two flat layers, ``v = a + g (z - z_top)`` in each, ``z = -ln r``, the
#: velocity jumping up at the interface, flattened onto the disk.
INTERFACE = 0.7
LAYERS = ((1.0, 2.0, 0.0), (2.0, 2.5, -np.log(INTERFACE)))
BOTTOM = 0.2


def _flat_layer(a, g, top):
    def velocity(r):
        r = np.asarray(r, dtype=float)
        return r * (a + g * (-np.log(r) - top))

    return velocity


@pytest.fixture(scope="module")
def layered(rays):
    upper, lower = (_flat_layer(*layer) for layer in LAYERS)
    return rays.RadialProfile(
        [lower, upper], bottom=BOTTOM, discontinuities=(INTERFACE,)
    )


def _two_layers(p, source_depth=0.0):
    """The closed form of a ray that turns in the lower layer: across a layer
    from depth ``z1`` down to ``z2`` it sweeps ``(sqrt(1 - p^2 v1^2) - sqrt(1
    - p^2 v2^2)) / (p g)`` and takes ``(arccosh(1 / (p v1)) - arccosh(1 / (p
    v2))) / g``, the second terms zero at the turning point."""
    (a1, g1, _), (a2, g2, depth) = LAYERS
    turning = depth + (1.0 / p - a2) / g2

    def across(a, g, z1, z2, top):
        v1, v2 = a + g * (z1 - top), a + g * (z2 - top)
        reach = np.sqrt(1.0 - (p * v1) ** 2) - np.sqrt(max(1.0 - (p * v2) ** 2, 0.0))
        time = np.arccosh(1.0 / (p * v1)) - np.arccosh(max(1.0 / (p * v2), 1.0))
        return np.array([reach / (p * g), time / g])

    lower = across(a2, g2, depth, turning, depth)
    legs = [lower + across(a1, g1, z, depth, 0.0) for z in (source_depth, 0.0)]
    return (*sum(legs), np.exp(-turning))


@pytest.mark.parametrize("fraction", [0.02, 0.25, 0.5, 0.75, 0.98])
@pytest.mark.parametrize("source", [None, 0.85])
def test_a_discontinuity_is_the_closed_form_of_its_two_layers(
    layered, fraction, source
):
    """A ray that turns below the jump crosses it into the upper layer: each
    layer's velocity on its own side, the distance, the time and the turning
    radius those of the two layers' closed forms."""
    low, high = (float(layered._eta(0, r)) for r in (BOTTOM, INTERFACE))
    p = low + fraction * (high - low)
    depth = 0.0 if source is None else -np.log(source)
    distance, time, turning = _two_layers(p, depth)
    # The largest differences are 9e-11 in the distance and 5e-11 in the
    # time, from the ray that turns just below the interface.
    assert layered.distance(p, source_radius=source) == pytest.approx(
        distance, rel=1e-9
    )
    assert layered.travel_time(p, source_radius=source) == pytest.approx(time, rel=1e-9)
    assert layered.turning_radius(p, source_radius=source) == pytest.approx(
        turning, abs=1e-12
    )


@pytest.mark.parametrize("distance", [60.0, 90.0, 120.0])
@pytest.mark.parametrize("source", [None, 0.85])
def test_the_time_across_a_discontinuity_is_second_order(layered, distance, source):
    """A ray that turns below the jump has a node at each crossing of it,
    and the integral over its travel time takes each side's slowness on its
    own: summed, its weights are the closed form's travel time to second
    order in the spacing, though the slowness jumps by a fifth. The error
    falls by 3.97 to 4.05 at each doubling of the nodes, from at most
    8.8e-6 at 201 nodes; with one side's slowness at both crossings it is
    1.8e-4 at 801, and falls only as the spacing."""
    branch = (BOTTOM, INTERFACE)
    p = layered.ray_parameter(distance, source_radius=source, turning_between=branch)
    _, time, _ = _two_layers(p, 0.0 if source is None else -np.log(source))
    errors = []
    for nodes in (201, 801):
        ray = layered.trace(
            0.0,
            distance,
            nodes=nodes,
            source_radius=source,
            turning_between=branch,
        )
        assert len(ray.nodes) == nodes
        assert np.count_nonzero(ray.nodes[:, 0] == INTERFACE) == 2
        errors.append(abs(ray.time_weights.sum() / time - 1.0))
    assert errors[0] < 2e-5
    assert errors[1] < errors[0] / 14.0


def test_a_reflected_ray_is_refused(layered):
    """A ray parameter in the jump of ``r / v`` at the interface belongs to
    a ray reflected there, which is not traced."""
    below, above = (float(layered._eta(piece, INTERFACE)) for piece in (0, 1))
    assert below < above
    with pytest.raises(ValueError, match="reflected"):
        layered.turning_radius(0.5 * (below + above))


def test_a_distance_reached_twice_is_refused_unless_a_branch_is_chosen(layered):
    """Past the jump the rays that turn just below it come back to the
    surface nearer than those that turn just above it, so some distances
    are reached by a ray turning in each layer: refused, unless the radii
    the ray must turn between choose one."""
    with pytest.raises(ValueError, match="2 rays"):
        layered.ray_parameter(60.0)
    deep = layered.ray_parameter(60.0, turning_between=(BOTTOM, INTERFACE))
    shallow = layered.ray_parameter(60.0, turning_between=(INTERFACE, 1.0))
    assert layered.turning_radius(deep) < INTERFACE < layered.turning_radius(shallow)
    for p in (deep, shallow):
        assert np.degrees(layered.distance(p)) == pytest.approx(60.0, rel=1e-10)
    assert layered.travel_time(deep) == pytest.approx(_two_layers(deep)[1], rel=1e-9)
    ray = layered.trace(0.0, 60.0, turning_between=(BOTTOM, INTERFACE))
    assert ray.nodes[:, 0].min() == pytest.approx(
        layered.turning_radius(deep), abs=1e-3
    )
    with pytest.raises(ValueError, match="between 0.2 and 0.7"):
        layered.ray_parameter(30.0, turning_between=(BOTTOM, INTERFACE))


# ------------------------------------------------------------------------- #
#                The two-point time between two depths                      #
# ------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "ends, distance",
    [((0.9, 0.8), 70.0), ((1.0, 0.85), 45.0), ((0.7, 0.95), 100.0), ((1.0, 1.0), 30.0)],
)
def test_the_time_between_two_depths_with_the_turning_point_between(
    flattened, ends, distance
):
    """Two ends at depth, the ray turning between them: the closed form's
    ray parameter and time are the tracer's, and with one end on the surface
    they are :meth:`from_depth`'s."""
    whole, arcs = flattened
    depths = -np.log(ends)
    p, (xc, _, _), time = arcs.between_depths(np.radians(distance), *depths)
    assert 0.0 < xc < np.radians(distance)
    ray = whole.trace(0.0, distance, source_radius=ends[0], receiver_radius=ends[1])
    # The largest differences are 1.1e-11 in p and 9.4e-12 in the time.
    assert p == pytest.approx(ray.ray_parameter, rel=1e-9)
    assert time == pytest.approx(ray.travel_time, rel=5e-10)
    if ends[1] == 1.0:
        p_once, _, time_once = arcs.from_depth(np.radians(distance), depths[0])
        assert p == pytest.approx(p_once, rel=1e-14)
        assert time == pytest.approx(time_once, rel=1e-14)


@pytest.mark.parametrize("source", [None, 0.85])
def test_the_time_between_two_depths_on_one_leg(flattened, source):
    """Two points on the same leg of a ray, the turning point not between
    them: the closed form's time between them is the difference of the
    tracer's times from each out to the surface along the same ray, and its
    ray parameter is the ray's."""
    whole, arcs = flattened
    ray = whole.trace(0.0, 90.0, source_radius=source)
    p = ray.ray_parameter
    count = len(ray.nodes)
    deepest = int(np.argmin(ray.nodes[:, 0]))
    pairs = [(2, deepest - 5), (deepest // 2, deepest - 1)]
    pairs += [(deepest + 5, count - 3), (deepest + 1, (deepest + count) // 2)]
    for a, b in pairs:
        (ra, ta), (rb, tb) = ray.nodes[a], ray.nodes[b]
        reach = np.radians(tb - ta)
        p_closed, (xc, _, _), time = arcs.between_depths(
            reach, -np.log(ra), -np.log(rb)
        )
        assert not 0.0 < xc < reach
        # From a point to the surface on the way down, or from the turning
        # point to it on the way up.
        out = [
            whole.travel_time(p, source_radius=r)
            if k < deepest
            else whole.travel_time(p) / 2.0
            - whole.travel_time(p, source_radius=r, receiver_radius=r) / 2.0
            for k, r in ((a, ra), (b, rb))
        ]
        # The largest differences are 2.4e-14 in the time and 5.6e-16 in p.
        assert time == pytest.approx(out[0] - out[1], rel=1e-12)
        assert p_closed == pytest.approx(p, rel=1e-12)


# ------------------------------------------------------------------------- #
#                         Dynamic ray tracing                               #
# ------------------------------------------------------------------------- #

#: The tomography app's mantle: ``v = 1 + g (1 - r)`` from 1 at the surface to
#: 1.8 at the core, at radius 0.55.
MANTLE_GRADIENT = (1.8 - 1.0) / (1.0 - 0.55)
#: A constant velocity, in which every ray is a chord.
CHORD_VELOCITY = 1.3


@pytest.fixture(scope="module")
def dynamic(rays):
    """Profiles with the velocity's first two derivatives given: the app's
    mantle, the flattened linear gradient, and a constant velocity."""

    def constant(r):
        return np.full_like(np.asarray(r, dtype=float), CHORD_VELOCITY)

    def zero(r):
        return np.zeros_like(np.asarray(r, dtype=float))

    return {
        "mantle": rays.RadialProfile(
            rays.linear_gradient(1.0, MANTLE_GRADIENT),
            bottom=0.55,
            derivatives=rays.linear_gradient_derivatives(1.0, MANTLE_GRADIENT),
        ),
        "flattened": rays.RadialProfile(
            rays.flattened_linear_gradient(FLAT_V0, FLAT_GRADIENT),
            bottom=0.05,
            derivatives=rays.flattened_linear_gradient_derivatives(
                FLAT_V0, FLAT_GRADIENT
            ),
        ),
        "chord": rays.RadialProfile(constant, bottom=0.05, derivatives=(zero, zero)),
    }


#: Rays of each profile, ``(distance, source radius, receiver radius)``, the
#: surface where a radius is ``None``.
DYNAMIC_RAYS = [
    ("mantle", 20.0, None, None),
    ("mantle", 45.0, None, None),
    ("mantle", 75.0, None, None),
    ("mantle", 45.0, 0.9, None),
    ("mantle", 60.0, None, 0.85),
    ("flattened", 20.0, None, None),
    ("flattened", 80.0, None, None),
    ("flattened", 150.0, None, None),
    ("flattened", 80.0, 0.85, None),
    ("flattened", 70.0, 0.9, 0.8),
]


def _traced(dynamic, name, distance, source, receiver):
    profile = dynamic[name]
    ray = profile.trace(
        10.0, 10.0 + distance, source_radius=source, receiver_radius=receiver
    )
    return profile, ray, profile.dynamic_ray(ray)


def _swept(ray):
    """The angle swept from the source at each node, in radians."""
    return np.radians(np.abs(ray.nodes[:, 1] - ray.nodes[0, 1]))


def _out_of_plane_sum(ray):
    """``p sin(Delta) / (r^2 sin(phi) sin(Delta - phi))`` at each node
    between the ends."""
    phi, r = _swept(ray)[1:-1], ray.nodes[1:-1, 0]
    reach = np.radians(abs(ray.nodes[-1, 1] - ray.nodes[0, 1]))
    return (
        ray.ray_parameter * np.sin(reach) / (r**2 * np.sin(phi) * np.sin(reach - phi))
    )


def _in_plane_changes(first, ray, traced):
    """How far ``r cos(i) P1 + (p^2 v' / r - 1 / v) Q1`` from the source,
    and the same with ``cos(i)`` of the opposite sign from the receiver,
    move from their values at their ends, at most, given ``v'``."""
    p, r, v = ray.ray_parameter, ray.nodes[:, 0], traced.velocity
    cosine = np.cos(traced.incidence)
    across = p**2 * first(r) / r - 1.0 / v
    forward = r * cosine * traced.forward_p[:, 0] + across * traced.forward_q[:, 0]
    backward = -r * cosine * traced.backward_p[:, 0] + across * traced.backward_q[:, 0]
    assert cosine[0] < 0.0 < cosine[-1]
    return (
        np.abs(forward - r[0] * cosine[0]).max(),
        np.abs(backward + r[-1] * cosine[-1]).max(),
    )


@pytest.mark.parametrize("name, distance, source, receiver", DYNAMIC_RAYS)
def test_the_out_of_plane_hessians_are_the_closed_forms(
    dynamic, name, distance, source, receiver
):
    """Out of the ray's plane both Hessians have closed forms in a radial
    medium: their sum ``p sin(Delta) / (r^2 sin(phi) sin(Delta - phi))`` at
    every node between the ends, ``phi`` the angle swept from the source, and
    the one from the source ``sin(i + phi) / (v r sin(phi))`` at every node
    after it, from a source on the surface or at depth."""
    _, ray, traced = _traced(dynamic, name, distance, source, receiver)
    # The largest relative difference is 2.0e-11, the most near the turning
    # point, where the traced ray itself is that accurate.
    closed = _out_of_plane_sum(ray)
    assert np.allclose(traced.sums[1:-1, 1], closed, rtol=2e-10, atol=0.0)
    phi, r = _swept(ray)[1:], ray.nodes[1:, 0]
    v, l = traced.velocity[1:], traced.arc_length[1:]
    forward = np.sin(traced.incidence[1:] + phi) / (v * r * np.sin(phi))
    # It changes sign, so the difference is measured against 1 / (v l), the
    # size it has near the source: at most 5.1e-11 of it.
    assert np.abs((traced.forward[1:, 1] - forward) * v * l).max() < 5e-10


@pytest.mark.parametrize("name, distance, source, receiver", DYNAMIC_RAYS)
def test_the_in_plane_combination_is_conserved(
    dynamic, name, distance, source, receiver
):
    """In the ray's plane there is no closed form, but ``r cos(i) P1 + (p^2
    v' / r - 1 / v) Q1`` keeps its value at the source, ``r_s cos(i_s)``, all
    along the ray; and from the receiver, along the reversed ray, the same
    with ``cos(i)`` of the opposite sign."""
    profile, ray, traced = _traced(dynamic, name, distance, source, receiver)
    # The largest change is 5.1e-11, at the turning point, where the traced
    # ray itself is that accurate.
    changes = _in_plane_changes(profile.derivatives[0], ray, traced)
    assert max(changes) < 5e-10


@pytest.mark.parametrize(
    "distance, source", [(40.0, None), (90.0, None), (150.0, None), (60.0, 0.8)]
)
def test_a_chord_in_a_constant_velocity(dynamic, distance, source):
    """In a constant velocity the ray is the chord, of length ``L``, along
    which ``M' = 1 / (v l)`` and ``M'' = 1 / (v (L - l))`` in and out of the
    plane, so that both sums are ``L / (v l (L - l))``, ``2 / (v sin(Delta /
    2))`` at the middle of a chord from the surface to the surface; and the
    frame's tangent is the chord's direction all along it."""
    _, ray, traced = _traced(dynamic, "chord", distance, source, None)
    a, b = traced.frame.point[0], traced.frame.point[-1]
    chord = np.linalg.norm(b - a)
    l, inner = traced.arc_length, slice(1, -1)
    assert l[-1] == pytest.approx(chord, rel=1e-12)
    v = CHORD_VELOCITY
    expected = chord / (v * l[inner] * (chord - l[inner]))
    # The largest relative differences are 1.4e-13 on Linux; the tolerances
    # leave two decades for other platforms' libraries, whose rounding near
    # the turning point the Hessians amplify.
    for i in range(2):
        assert np.allclose(
            traced.forward[1:, i], 1.0 / (v * l[1:]), rtol=1e-11, atol=0.0
        )
        assert np.allclose(
            traced.backward[:-1, i], 1.0 / (v * (chord - l[:-1])), rtol=1e-11, atol=0.0
        )
        assert np.allclose(traced.sums[inner, i], expected, rtol=1e-11, atol=0.0)
    assert np.all(np.isinf(traced.sums[[0, -1]]))
    if source is None:
        middle = len(l) // 2
        assert l[middle] == pytest.approx(chord / 2.0, rel=1e-12)
        # The traced chord is 2 sin(Delta / 2) long to 7.5e-13 on Linux and
        # about 1e-12 on macOS.
        assert traced.sums[middle] == pytest.approx(
            2.0 / (v * np.sin(np.radians(distance) / 2.0)), rel=1e-10
        )
    # The nodes are off the chord by at most 4.7e-12, and the tangent off its
    # direction by 1.4e-11, near the turning point.
    assert np.allclose(traced.frame.tangent, (b - a) / chord, rtol=0.0, atol=1e-10)


def _time_between(arcs, x, y):
    """The closed-form time between points of the flattened gradient's ball
    in three dimensions, along the ray in the plane through them and the
    center."""
    angle = np.arctan2(np.linalg.norm(np.cross(x, y), axis=-1), np.sum(x * y, axis=-1))
    depths = (-np.log(np.linalg.norm(z, axis=-1)) for z in (x, y))
    return arcs.between_depths(angle, *depths)[2]


@pytest.mark.parametrize(
    "distance, source",
    [(30.0, None), (90.0, None), (140.0, None), (70.0, 0.85), (45.0, 0.9)],
)
def test_the_sums_are_the_curvature_of_the_detour_time(
    dynamic, flattened, distance, source
):
    """The detour through a point near the ray, ``T_xs + T_xr - T_rs``, from
    the closed form in three dimensions alone, is ``(A1 q1^2 + A2 q2^2) / 2``
    to second order: its second differences along each of the frame's two
    normals, extrapolated to zero step, are the two sums, at points from a
    tenth of the way along the ray to nine tenths."""
    _, arcs = flattened
    _, ray, traced = _traced(dynamic, "flattened", distance, source, None)
    frame, count = traced.frame, len(ray.nodes)
    s, r = frame.point[0], frame.point[-1]
    direct = _time_between(arcs, s, r)
    assert direct == pytest.approx(ray.travel_time, rel=1e-9)
    nodes = np.round(np.array([0.1, 0.25, 0.5, 0.75, 0.9]) * (count - 1)).astype(int)
    for axis in range(2):
        sums = traced.sums[nodes, axis]
        # A step a hundredth of the width over which the detour grows by
        # its own size, and half of it; the second differences' error is
        # that step squared, which the two remove.
        curvatures = []
        for fraction in (1.0, 0.5):
            step = fraction * 1e-2 / (traced.velocity[nodes] * sums)
            offsets = np.zeros((count, 2))
            offsets[nodes] = np.column_stack([step, -step])
            q1, q2 = (offsets, 0.0 * offsets) if axis == 0 else (0.0 * offsets, offsets)
            x = frame.place(q1, q2)[nodes]
            detour = _time_between(arcs, s, x) + _time_between(arcs, x, r) - direct
            curvatures.append(detour.sum(axis=1) / step**2)
        extrapolated = (4.0 * curvatures[1] - curvatures[0]) / 3.0
        # The extrapolated second differences agree to 1.4e-9, the step's
        # error and the rounding in the times balanced; unextrapolated, at
        # the best of steps from 1e-4 to 3e-3 of the width, to about 2e-7.
        assert np.allclose(extrapolated, sums, rtol=1e-8, atol=0.0)


@pytest.mark.parametrize(
    "name, distance, source",
    [
        ("mantle", 45.0, None),
        ("mantle", 60.0, 0.85),
        ("flattened", 80.0, None),
        ("flattened", 120.0, 0.7),
    ],
)
def test_the_reversed_ray_swaps_the_two_hessians(dynamic, name, distance, source):
    """Traced the other way, from the receiver to the source, a ray's Hessian
    from its new source is the one from its old receiver, node for node, and
    the sums are the same; the frame's normal in the plane is the same, and
    its tangent and its normal to the plane are reversed."""
    profile = dynamic[name]
    out = profile.trace(10.0, 10.0 + distance, source_radius=source)
    back = profile.trace(10.0 + distance, 10.0, receiver_radius=source)
    assert np.allclose(back.nodes[::-1], out.nodes, rtol=0.0, atol=2e-8)
    there, home = profile.dynamic_ray(out), profile.dynamic_ray(back)
    # The largest relative differences are 2.7e-12, from the two root finds,
    # and 1.1e-11 in the frame.
    options = dict(rtol=2e-11, atol=0.0)
    assert np.allclose(home.forward[::-1][:-1], there.backward[:-1], **options)
    assert np.allclose(home.backward[::-1][1:], there.forward[1:], **options)
    assert np.allclose(home.sums[::-1][1:-1], there.sums[1:-1], **options)
    flipped, options = home.frame, dict(rtol=0.0, atol=1e-10)
    assert np.allclose(flipped.in_plane[::-1], there.frame.in_plane, **options)
    assert np.allclose(flipped.tangent[::-1], -there.frame.tangent, **options)
    assert np.array_equal(flipped.out_of_plane, -there.frame.out_of_plane)


@pytest.mark.parametrize("name, distance, source, receiver", DYNAMIC_RAYS[:5])
def test_the_frame_is_right_handed_and_follows_the_ray(
    dynamic, name, distance, source, receiver
):
    """The frame is orthonormal and right-handed, its tangent along the ray
    between neighboring nodes, its normal in the plane on the side of
    increasing radius, and :meth:`Frame.place` puts a point at the offsets
    asked for, one offset or many at each node."""
    _, ray, traced = _traced(dynamic, name, distance, source, receiver)
    frame = traced.frame
    k, e1, e2 = frame.tangent, frame.in_plane, frame.out_of_plane
    # Orthonormal and right-handed to 4.4e-16.
    options = dict(rtol=0.0, atol=1e-14)
    for a, b in ((k, k), (e1, e1), (e2, e2)):
        assert np.allclose(np.sum(a * b, axis=1), 1.0, **options)
    for a, b in ((k, e1), (k, e2), (e1, e2)):
        assert np.allclose(np.sum(a * b, axis=1), 0.0, **options)
    assert np.allclose(np.cross(e1, e2), k, **options)
    assert np.all(np.sum(e1 * frame.point, axis=1) > 0.0)
    radii, angles = ray.nodes[:, 0], np.radians(ray.nodes[:, 1])
    assert np.allclose(frame.point[:, 0], radii * np.cos(angles), **options)
    assert np.allclose(frame.point[:, 1], radii * np.sin(angles), **options)
    # Each chord between neighbors is along the mean of their tangents to
    # second order in the spacing: one minus the cosine between them is at
    # most 8.1e-13 here.
    chords = np.diff(frame.point, axis=0)
    chords /= np.linalg.norm(chords, axis=1)[:, None]
    middle = k[1:] + k[:-1]
    middle /= np.linalg.norm(middle, axis=1)[:, None]
    assert np.all(1.0 - np.sum(chords * middle, axis=1) < 1e-10)
    q1 = np.linspace(-0.01, 0.01, 3)
    many = frame.place(np.tile(q1, (len(radii), 1)), 0.02)
    assert many.shape == (len(radii), 3, 3)
    assert np.allclose(many[:, 2], frame.point + 0.01 * e1 + 0.02 * e2, **options)
    assert np.array_equal(frame.place(0.0, 0.0), frame.point)
    # Out of the plane by q2, a point's distance from the center grows as
    # the hypotenuse.
    out = frame.place(0.0, 0.05)
    assert np.allclose(np.linalg.norm(out, axis=1), np.hypot(radii, 0.05), **options)


def test_the_derivatives_default_to_the_series(rays, dynamic):
    """Without the velocity's derivatives, those of its Chebyshev series on
    each piece are taken: the linear gradient's exactly, the flattened
    gradient's to 3.3e-10 (3.2e-9 on macOS) and 6.3e-7 of a second
    derivative that reaches 76 at the bottom, which moves the sums along its
    rays by at most 1.9e-12 on Linux."""
    for name, bottom in (("mantle", 0.55), ("flattened", 0.05)):
        given = dynamic[name]
        series = rays.RadialProfile(given.velocity, bottom=bottom)
        radii = np.linspace(bottom, 1.0, 401)
        first, second = series._derivatives(0)
        exact = [f(radii) for f in given.derivatives]
        assert np.abs(first(radii) - exact[0]).max() < 3e-8
        assert np.abs(second(radii) - exact[1]).max() < 6e-5
        for distance, source in ((20.0, None), (60.0, None), (45.0, 0.9)):
            ray = given.trace(0.0, distance, source_radius=source)
            sums = (profile.dynamic_ray(ray).sums for profile in (given, series))
            assert np.allclose(*sums, rtol=1e-9, atol=0.0)


def test_one_pair_of_derivatives_for_each_piece(rays, dynamic):
    """Through interfaces where nothing jumps, the ray integrated stretch by
    stretch between its crossings: the derivatives given once for every
    piece or once for each are the same profile to the last bit, and those
    of the series on each piece agree with them; and the out-of-plane sum is
    its closed form and the in-plane combination is conserved, across every
    crossing."""
    whole = dynamic["flattened"]
    cut = CUTS[1]
    pair = whole.derivatives
    pieces = len(cut) + 1
    once, each, series = (
        rays.RadialProfile(
            velocity, bottom=0.05, discontinuities=cut, derivatives=derivatives
        )
        for velocity, derivatives in (
            (whole.velocity, pair),
            ([whole.velocity] * pieces, [pair] * pieces),
            (whole.velocity, None),
        )
    )
    for source in (None, 0.8):
        ray = once.trace(10.0, 100.0, source_radius=source)
        assert np.count_nonzero(np.isin(ray.nodes[1:-1, 0], cut)) >= 3
        traced = once.dynamic_ray(ray)
        assert np.array_equal(each.dynamic_ray(ray).sums, traced.sums)
        # The largest relative difference is 2.8e-13 on Linux and above 1e-11
        # on macOS; the tolerance leaves room for either.
        assert np.allclose(
            series.dynamic_ray(ray).sums, traced.sums, rtol=1e-9, atol=0.0
        )
        closed = _out_of_plane_sum(ray)
        assert np.allclose(traced.sums[1:-1, 1], closed, rtol=2e-10, atol=0.0)
        assert max(_in_plane_changes(pair[0], ray, traced)) < 5e-10
    with pytest.raises(ValueError, match="pair"):
        rays.RadialProfile(
            whole.velocity, bottom=0.05, discontinuities=cut, derivatives=[pair] * 2
        )


def test_dynamic_tracing_refuses_what_it_cannot_do(rays, dynamic, layered):
    """Refused: a ray that carries no ray parameter; a ray that crosses a
    jump in the velocity, where the Hessians jump by conditions not applied
    here, though one that turns above the jump is traced; and a ray along
    which a sum changes sign, at a caustic, made here by giving a second
    derivative far from the velocity's own, which focuses the rays in their
    plane until they cross."""
    mantle = dynamic["mantle"]
    ray = mantle.trace(0.0, 45.0)
    bare = rays.Ray(ray.nodes, ray.weights, ray.travel_time)
    with pytest.raises(ValueError, match="ray parameter"):
        mantle.dynamic_ray(bare)
    deep = layered.trace(0.0, 90.0, turning_between=(BOTTOM, INTERFACE))
    with pytest.raises(ValueError, match="jumps"):
        layered.dynamic_ray(deep)
    shallow = layered.trace(0.0, 20.0, turning_between=(INTERFACE, 1.0))
    assert np.all(layered.dynamic_ray(shallow).sums[1:-1] > 0.0)

    def focusing(r):
        return np.full_like(np.asarray(r, dtype=float), 200.0)

    focused = rays.RadialProfile(
        mantle.velocity,
        bottom=0.55,
        derivatives=(mantle.derivatives[0], focusing),
    )
    with pytest.raises(ValueError, match="caustic"):
        focused.dynamic_ray(focused.trace(0.0, 45.0))
