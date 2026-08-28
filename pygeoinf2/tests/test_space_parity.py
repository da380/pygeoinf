"""Every geometry answers the same questions.

The test whose absence let the box regressions through. ``SymmetricSpace``
declares its geometric primitives non-abstract and raising, so a subclass that
does not implement one still constructs and fails only when something
downstream reaches for it — which for ``geodesic_distance`` meant
``path_average_operator``, ``covariance_function`` and
``geodesic_ball_average_operator`` were all unavailable on every box, while a
suite of a thousand tests passed and the catalogue said "Ported".

So this calls each method on each geometry, rather than asking whether the
attribute exists: on the base class it always does.

See REVIEW.md, appendix Y, Consider-30.
"""

import numpy as np
import pytest

from pygeoinf2.symmetric_space import box as box_module
from pygeoinf2.symmetric_space import circle, line, plane, torus


def build_geometries():
    """One space per geometry, all of the same Sobolev order."""
    # Resolved enough that the numerical identities below mean something: on
    # an eight-point interval the band limit is four, and the taper's roll-off
    # is most of the domain.
    geometries = {
        "circle": lambda: circle.Sobolev(64, 2.0, 0.3),
        "torus": lambda: torus.Sobolev((16, 16), 2.0, 0.3),
        "line": lambda: line.Sobolev(64, 2.0, 0.3),
        "plane": lambda: plane.Sobolev(
            (16, 16), 2.0, 0.3, bounds=((0.0, 1.0), (0.0, 1.0))
        ),
        "box": lambda: box_module.Sobolev((64,), 2.0, 0.3, bounds=((0.0, 1.0),)),
    }
    try:
        import pyshtools  # noqa: F401

        from pygeoinf2.symmetric_space.sphere import Sobolev as SphereSobolev

        geometries["sphere"] = lambda: SphereSobolev(8, 2.0, 0.3)
    except ImportError:  # pragma: no cover - exercised only without pyshtools
        pass
    return geometries


GEOMETRIES = build_geometries()


@pytest.fixture(params=sorted(GEOMETRIES))
def geometry(request):
    return request.param, GEOMETRIES[request.param]()


class TestEveryGeometryHasTheGeometry:
    """The primitives, and the operators that are built on them."""

    def test_it_reports_its_shape(self, geometry):
        _, space = geometry
        assert space.spatial_dimension >= 1
        assert space.dim > 0
        assert np.isfinite(space.gaussian_curvature)
        assert space.degrees.shape == (space.dim,)

    def test_it_produces_points(self, geometry, rng):
        _, space = geometry
        assert space.reference_point is not None
        assert len(space.random_points(4, rng=rng)) == 4

    def test_it_measures_distances(self, geometry, rng):
        _, space = geometry
        first, second = space.random_point(rng=rng), space.random_point(rng=rng)

        assert space.geodesic_distance(first, first) == pytest.approx(0.0, abs=1e-12)
        assert space.geodesic_distance(first, second) == pytest.approx(
            space.geodesic_distance(second, first)
        )
        assert space.geodesic_distance(first, second) > 0.0

    def test_its_quadrature_weights_sum_to_the_distance(self, geometry, rng):
        """The contract the weights are defined by, and what makes integrating
        the constant one give the arc length."""
        _, space = geometry
        first, second = space.random_point(rng=rng), space.random_point(rng=rng)
        _, weights = space.geodesic_quadrature(first, second, count=8)
        assert float(np.sum(weights)) == pytest.approx(
            space.geodesic_distance(first, second)
        )

    def test_it_walks_a_known_distance(self, geometry, rng):
        _, space = geometry
        start = space.random_point(rng=rng)
        step = 0.05 * space.geodesic_distance(start, space.random_point(rng=rng))
        if step <= 0.0:
            pytest.skip("degenerate step")
        (arrived,) = space.walk_from(start, np.array([step]))
        assert space.geodesic_distance(start, arrived) == pytest.approx(step, rel=1e-6)

    def test_it_builds_the_operators_that_need_geodesics(self, geometry, rng):
        """The four that were unreachable on every box, because each one asks
        for a primitive the base class only declared."""
        _, space = geometry
        first, second = space.random_point(rng=rng), space.random_point(rng=rng)
        radius = 0.1 * space.geodesic_distance(first, second)

        assert space.path_integral_operator([(first, second)], count=6) is not None
        assert space.path_average_operator([(first, second)], count=6) is not None
        assert (
            space.geodesic_ball_average_operator([first], radius, count=20) is not None
        )
        assert space.covariance_function(
            space.sobolev_measure(2.0, 0.3), np.array([0.0, radius])
        ).shape == (2,)

    def test_a_path_integral_of_one_is_its_length(self, geometry, rng):
        """Across every geometry, which is the point: the same identity, the
        same code path, six domains."""
        _, space = geometry
        lebesgue = space.with_order(0.0)
        one = lebesgue.project_function(lambda point: 1.0)

        # Kept well inside a bounded domain, where the padding's roll-off has
        # died away; on a periodic one it makes no difference.
        first, second = space.random_point(rng=rng), space.random_point(rng=rng)
        middle = space.geodesic_quadrature(first, second, count=3)[0]
        start, end = middle[0], middle[-1]

        operator = lebesgue.path_integral_operator([(start, end)], count=40)
        assert operator(one)[0] == pytest.approx(
            lebesgue.geodesic_distance(start, end), rel=2e-2
        )
