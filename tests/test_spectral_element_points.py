"""Point evaluation on the spectral-element spaces, assembled and matrix-free.

The assembled route is the oracle: its rows are the basis at the points, by
``planetmodel``'s harmonics and the element polynomials of the modes, while
the matrix-free route interpolates nodal values within the points' elements
and sums its own harmonics. The two share the components and nothing else,
and must agree forward and adjoint to rounding on every geometry; the
matrix-free one must also pass the adjoint check on its own, against each
space's Sobolev measure.

Needs ``planetmodel`` 1.4.
"""

import numpy as np
import pytest

randomfield = pytest.importorskip("planetmodel.randomfield")

from planetmodel import harmonics  # noqa: E402

from pygeoinf import ball, disk, interval  # noqa: E402
from pygeoinf.interval import radial  # noqa: E402
from pygeoinf.spectral_element import _points, base  # noqa: E402
from pygeoinf.testing import check_operator  # noqa: E402

TOLERANCE = 1e-12


def line():
    return interval.Sobolev(
        24, 1.0, lambda x: 0.15 + 0.1 * x, lower=-0.5, upper=1.0, ngll=6
    )


def whole_profile():
    return radial.Sobolev(16, 2.0, 0.25)


def shell_profile():
    return radial.Sobolev(16, 1.0, 0.2, inner_radius=0.4)


def whole_ball():
    return ball.Sobolev(5, 2.0, 0.3, element_length=0.3)


def shell_ball():
    return ball.Sobolev(6, 2.0, 0.2, inner_radius=0.5, element_length=0.2)


def whole_disk():
    return disk.Sobolev(6, 2.0, 0.3, element_length=0.3)


def ring_disk():
    return disk.Sobolev(7, 1.5, 0.2, inner_radius=0.5, element_length=0.2)


def lines():
    return interval.Layered(
        [-1.0, 0.0, 0.5, 2.0], [14, 10, 16], order=1.0, length_scale=[0.2, 0.15, 0.3]
    )


def profiles():
    return radial.Layered([0.0, 0.4, 1.0], 12, order=2.0, length_scale=0.2)


def balls():
    return ball.Layered(
        [0.0, 0.5, 1.0], [3, 4], order=2.0, length_scale=0.3, element_length=0.2
    )


def disks():
    return disk.Layered(
        [0.0, 0.5, 1.0], [4, 5], order=2.0, length_scale=0.25, element_length=0.2
    )


def axis_points(space, rng, /):
    """Points of an interval or a profile, its ends and, on a whole profile,
    the center and a point in the first element among them."""
    lower, upper = space.bounds
    points = list(rng.uniform(lower, upper, 30)) + [lower, upper]
    return points + ([1e-3] if lower == 0.0 else [])


def ball_points(space, rng, /):
    """Points of a ball or a shell, the poles, both ends of the radius, a
    longitude past a turn and, in a whole ball, the center among them."""
    lower, upper = space.bounds
    points = [space.random_point(rng=rng) for _ in range(30)]
    points += [(lower, 90.0, 0.0), (upper, -90.0, 40.0), (upper, 10.0, 400.0)]
    return points + ([(0.0, 20.0, 30.0), (1e-3, -45.0, 10.0)] if lower == 0.0 else [])


def disk_points(space, rng, /):
    """Points of a disk or a ring, as the ball's."""
    lower, upper = space.bounds
    points = [space.random_point(rng=rng) for _ in range(30)]
    points += [(lower, 0.0), (upper, -170.0), (upper, 400.0)]
    return points + ([(0.0, 20.0), (1e-3, 10.0)] if lower == 0.0 else [])


def layered_points(space, rng, /):
    """Points of each layer, the interfaces included, and the outer ends."""
    points = []
    for lower, upper in space.layer_bounds:
        points += [lower, upper, *rng.uniform(lower, upper, 12)]
    if space.spatial_dimension == 1 or space.geometry is radial.Radial:
        return points
    if space.geometry is ball.Ball:
        return [
            (r, float(rng.uniform(-90.0, 90.0)), float(rng.uniform(0.0, 360.0)))
            for r in points
        ]
    return [(r, float(rng.uniform(0.0, 360.0))) for r in points]


SINGLE = [
    (line, axis_points),
    (whole_profile, axis_points),
    (shell_profile, axis_points),
    (whole_ball, ball_points),
    (shell_ball, ball_points),
    (whole_disk, disk_points),
    (ring_disk, disk_points),
]
LAYERED = [
    (lines, layered_points),
    (profiles, layered_points),
    (balls, layered_points),
    (disks, layered_points),
]
CASES = SINGLE + LAYERED
IDS = [build.__name__ for build, _ in CASES]


def operators(space, points, /):
    """The dense and the matrix-free evaluation at the points; a point on an
    interface of a layered space is taken from below."""
    options = {"side": "below"} if hasattr(space, "layers") else {}
    return (
        space.point_evaluation_operator(points, route="dense", **options),
        space.point_evaluation_operator(points, route="local", **options),
    )


def relative(a, b, /):
    return float(np.abs(a - b).max() / np.abs(b).max())


@pytest.mark.parametrize(("build", "where"), CASES, ids=IDS)
class TestTheLocalFormIsTheDenseOne:
    def test_forward(self, build, where, rng):
        space = build()
        dense, local = operators(space, where(space, rng))
        measure = space.sobolev_measure(2.0)
        for x in (space.random(rng=rng), measure.sample(rng=rng)):
            assert relative(local(x), dense(x)) < TOLERANCE
        c = rng.standard_normal(space.dim)
        assert (
            relative(local.components_action()(c), dense.components_action()(c))
            < TOLERANCE
        )

    def test_adjoint(self, build, where, rng):
        space = build()
        points = where(space, rng)
        dense, local = operators(space, points)
        for _ in range(3):
            y = rng.standard_normal(len(points))
            assert (
                relative(
                    space.to_components(local.adjoint(y)),
                    space.to_components(dense.adjoint(y)),
                )
                < TOLERANCE
            )
            assert (
                relative(
                    local.components_adjoint_action()(y),
                    dense.components_adjoint_action()(y),
                )
                < TOLERANCE
            )

    def test_the_adjoint_identity_under_the_sobolev_measure(self, build, where, rng):
        space = build()
        _, local = operators(space, where(space, rng))
        check_operator(local, rng=rng, measure=space.sobolev_measure(2.0))

    def test_a_field_off_the_span_is_read_through_its_components(
        self, build, where, rng
    ):
        """A pointwise product is left on the grid, off the span: both routes
        read it through its components, so they agree there too, and the
        evaluation composed with a multiplication keeps an exact adjoint."""
        space = build()
        dense, local = operators(space, where(space, rng))
        u, v = space.random(rng=rng), space.random(rng=rng)
        product = space.multiply(u, v)
        assert relative(local(product), dense(product)) < TOLERANCE
        check_operator(
            local @ space.multiplication_operator(u),
            rng=rng,
            measure=space.sobolev_measure(2.0),
        )


@pytest.mark.parametrize(
    ("build", "where"), SINGLE, ids=[b.__name__ for b, _ in SINGLE]
)
def test_evaluate_is_the_dense_rows(build, where, rng):
    space = build()
    points = where(space, rng)
    x = space.random(rng=rng)
    expected = space.basis_matrix(points) @ space.to_components(x)
    assert relative(space.evaluate(x, points), expected) < TOLERANCE


@pytest.mark.parametrize("build", [whole_ball, ring_disk])
def test_chunks_and_a_recomputed_table_change_nothing(build, rng, monkeypatch):
    """Chunks of a few points, so that a chunk ends inside an element and
    an element spans chunks, and the angular functions recomputed per chunk
    rather than kept."""
    space = build()
    points = [space.random_point(rng=rng) for _ in range(60)]
    dense = space.point_evaluation_operator(points, route="dense")
    harmonics_count = (
        (space.lmax + 1) ** 2 if hasattr(space, "lmax") else 2 * space.mmax + 1
    )
    monkeypatch.setattr(_points, "_CHUNK_ENTRIES", 7 * harmonics_count)
    monkeypatch.setattr(_points, "_TABLE_BUDGET", 0)
    local = space.point_evaluation_operator(points, route="local")
    c = rng.standard_normal(space.dim)
    y = rng.standard_normal(len(points))
    assert (
        relative(local.components_action()(c), dense.components_action()(c)) < TOLERANCE
    )
    assert (
        relative(
            local.components_adjoint_action()(y), dense.components_adjoint_action()(y)
        )
        < TOLERANCE
    )


@pytest.mark.parametrize(("build", "where"), CASES, ids=IDS)
def test_auto_is_dense_within_the_limit_and_local_beyond(
    build, where, rng, monkeypatch
):
    space = build()
    points = where(space, rng)
    options = {"side": "below"} if hasattr(space, "layers") else {}
    within = space.point_evaluation_operator(points, **options)
    assert within.known_matrix(form="galerkin") is not None
    monkeypatch.setattr(
        base, "_DENSE_EVALUATION_LIMIT", 8 * len(points) * space.dim - 1
    )
    beyond = space.point_evaluation_operator(points, **options)
    assert beyond.known_matrix(form="galerkin") is None
    x = space.random(rng=rng)
    assert relative(beyond(x), within(x)) < TOLERANCE


class TestRefusals:
    @pytest.mark.parametrize("route", ["auto", "dense", "local"])
    @pytest.mark.parametrize(
        ("build", "outside"),
        [
            (line, 1.5),
            (shell_profile, 0.3),
            (whole_ball, (1.2, 0.0, 0.0)),
            (shell_ball, (0.3, 0.0, 0.0)),
            (whole_disk, (1.1, 0.0)),
            (ring_disk, (0.2, 0.0)),
            (lines, 2.5),
            (balls, (1.5, 0.0, 0.0)),
        ],
    )
    def test_a_point_outside_the_domain(self, build, outside, route):
        space = build()
        with pytest.raises(ValueError):
            space.point_evaluation_operator([outside], route=route)

    @pytest.mark.parametrize("route", ["auto", "dense", "local"])
    @pytest.mark.parametrize(
        ("build", "point", "threshold"),
        [
            (lambda: line().with_order(0.5), 0.2, "0.5"),
            (lambda: whole_profile().with_order(1.5), 0.0, "1.5"),
            (lambda: whole_ball().with_order(1.5), (0.5, 0.0, 0.0), "1.5"),
            (lambda: whole_disk().with_order(1.0), (0.5, 0.0), "1"),
            (lambda: lines().with_order(0.5), 0.2, "0.5"),
        ],
    )
    def test_an_order_too_low(self, build, point, threshold, route):
        space = build()
        with pytest.raises(ValueError, match=f"above {threshold}"):
            space.point_evaluation_operator([point], route=route)
        space.point_evaluation_operator([point], route=route, unsafe=True)

    @pytest.mark.parametrize("build", [line, whole_ball, lines])
    def test_no_points_and_an_unknown_route(self, build, rng):
        space = build()
        with pytest.raises(ValueError, match="one point"):
            space.point_evaluation_operator([], route="local")
        point = space.random_point(rng=rng) if hasattr(space, "random_point") else 0.1
        with pytest.raises(ValueError, match="route"):
            space.point_evaluation_operator([point], route="sparse")

    def test_a_latitude_past_a_pole(self):
        space = whole_ball()
        for route in ("dense", "local"):
            with pytest.raises(ValueError):
                space.point_evaluation_operator([(0.5, 91.0, 0.0)], route=route)


@pytest.mark.parametrize("lmax", [0, 1, 2, 7, 40])
def test_the_harmonic_table_is_planetmodels(lmax, rng):
    colatitudes = np.concatenate(([0.0, np.pi, 0.5 * np.pi], rng.uniform(0, np.pi, 50)))
    longitudes = rng.uniform(-np.pi, 3.0 * np.pi, colatitudes.size)
    parts, degrees, orders = harmonics.packing(lmax)
    expected = harmonics.real_harmonics(lmax, colatitudes, longitudes)[
        parts, degrees, orders
    ]
    table = _points.harmonic_table(lmax, colatitudes, longitudes)
    assert np.abs(table - expected).max() < 1e-13 * np.abs(expected).max()
