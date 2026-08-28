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


class TestNeighbourSearchAndClustering:
    """On the base, so every geometry has them, and by KD-tree."""

    def test_the_pairs_are_the_pairs(self, geometry, rng):
        """Against a brute-force sweep, which is what v1 does on the base and
        what v2 did on the sphere -- correct, and 216 MB of differences at
        n = 3000."""
        _, space = geometry
        points = space.random_points(30, rng=rng)
        scale = 1.01 * space.geodesic_distance(points[0], points[1])

        rows, columns, distances = space.pairs_within_distance(
            points, scale, with_distances=True
        )
        expected = {
            (i, j)
            for i in range(30)
            for j in range(30)
            if space.geodesic_distance(points[i], points[j]) <= scale
        }
        assert set(zip(rows.tolist(), columns.tolist())) == expected

        # And the reported separations are the geodesic ones. On a periodic
        # domain the tree wraps, so measuring the pairs afterwards has to wrap
        # too -- otherwise it reports the long way round.
        reference = np.array(
            [space.geodesic_distance(points[i], points[j]) for i, j in zip(rows, columns)]
        )
        assert distances == pytest.approx(reference)

    def test_the_pattern_is_symmetric_and_has_a_diagonal(self, geometry, rng):
        """It is the sparsity pattern of a symmetric matrix, which is what a
        localised covariance is assembled into."""
        _, space = geometry
        points = space.random_points(20, rng=rng)
        scale = 1.01 * space.geodesic_distance(points[0], points[1])
        rows, columns = space.pairs_within_distance(points, scale)
        pairs = set(zip(rows.tolist(), columns.tolist()))

        assert all((j, i) in pairs for i, j in pairs)
        assert all((i, i) in pairs for i in range(20))

    def test_clustering_by_count_gives_that_many(self, geometry, rng):
        """The mode that sizes preconditioner blocks to a budget, and the one
        v2 dropped in favour of a greedy rule seeded by the lowest remaining
        index -- which is not stable under reordering the points."""
        _, space = geometry
        points = space.random_points(20, rng=rng)
        clusters = space.cluster_points(points, count=4)

        assert len(clusters) == 4
        assert sorted(i for cluster in clusters for i in cluster) == list(range(20))

    def test_clustering_by_radius_bounds_the_width(self, geometry, rng):
        """Complete linkage, so the *widest* separation within a cluster is
        what the radius caps."""
        _, space = geometry
        points = space.random_points(20, rng=rng)
        scale = 0.5 * space.geodesic_distance(points[0], points[1])
        for cluster in space.cluster_points(points, radius=scale):
            for i in cluster:
                for j in cluster:
                    assert space.geodesic_distance(points[i], points[j]) <= scale * 1.001

    def test_exactly_one_criterion_is_needed(self, geometry, rng):
        _, space = geometry
        points = space.random_points(5, rng=rng)
        with pytest.raises(ValueError, match="exactly one"):
            space.cluster_points(points)
        with pytest.raises(ValueError, match="exactly one"):
            space.cluster_points(points, radius=1.0, count=2)


class TestResolutionTransfer:
    """Moving a field between grids, on every geometry that has more than one."""

    def test_prolonging_then_restricting_is_the_identity(self, geometry, rng):
        """Nothing is lost going up and coming back: the finer grid holds every
        mode of the coarser one."""
        from pygeoinf2.testing import check_operator

        name, space = geometry
        finer = space.with_degree(space.degrees.max() + 2)

        up = space.degree_transfer_operator(finer)
        down = finer.degree_transfer_operator(space)
        check_operator(up, rng=rng)

        field = space.random(rng=rng)
        recovered = down(up(field))
        assert space.norm(space.subtract(recovered, field)) < 1e-10 * space.norm(field)


class TestPowerPerDegree:
    """A spectrum written per degree, on every geometry.

    The multiplicity is the whole of the method, and on a box it is not the
    ``2l + 1`` of the sphere: degrees there are ``floor(|k|)``, so the counts
    are irregular and some degrees can be missing altogether.
    """

    def test_each_degree_holds_the_power_it_was_given(self, geometry):
        _, space = geometry
        degrees = space.degrees
        power = (1.0 + np.arange(degrees.max() + 1)) ** -3.0
        eigenvalues = space.power_measure(power).covariance.eigenvalues

        for degree in np.unique(degrees):
            held = eigenvalues[degrees == degree].sum()
            assert held == pytest.approx(power[degree])

    def test_the_multiplicities_are_the_multiplicities(self, geometry):
        """Tabulated in one pass rather than swept per component -- the
        comprehension this replaced was O(dim^2), and 6.8 s at lmax 511."""
        _, space = geometry
        degrees = space.degrees
        counts = np.bincount(degrees)[degrees]
        expected = np.array([np.count_nonzero(degrees == d) for d in degrees])
        assert np.array_equal(counts, expected)
