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


class TestCoefficientAccess:
    """The public way out to coefficients, and the operators either side."""

    def test_coefficients_round_trip(self, geometry, rng):
        """An SHCoeffs on the sphere, a complex rfftn array on a box. Either
        way it is the object the rest of the world takes, and reading it back
        in returns the same field."""
        _, space = geometry
        field = space.random(rng=rng)
        recovered = space.from_coefficients(space.to_coefficients(field))
        assert space.norm(space.subtract(recovered, field)) < 1e-12 * space.norm(field)

    def test_the_power_spectrum_is_the_power(self, geometry, rng):
        """Summing over the degrees gives the whole squared L2 norm, which is
        what says nothing was double-counted or dropped between them."""
        _, space = geometry
        field = space.random(rng=rng)
        components = space.to_components(field)
        assert space.power_spectrum(field).sum() == pytest.approx(
            np.sum(components**2)
        )

    def test_a_lebesgue_measures_power_is_the_power_it_was_given(self, geometry, rng):
        """On L2 the two meet: draws from ``power_measure(p)`` have spectrum
        ``p``. This is the scale check -- an error of a factor of ``2l + 1``,
        which is the multiplicity the method exists to divide out, would show
        at any degree above zero."""
        _, base = geometry
        space = base.with_order(0.0)
        degrees = space.degrees
        measure = space.power_measure(np.full(degrees.max() + 1, 4.0))

        drawn = np.mean(
            [space.power_spectrum(measure.sample(rng=rng)) for _ in range(400)],
            axis=0,
        )
        present = np.unique(degrees)
        assert np.all(drawn[present] > 2.0)
        assert np.all(drawn[present] < 8.0)

    def test_on_a_sobolev_space_the_metric_shows(self, geometry, rng):
        """And it shows by a *known* factor, not an unexplained one. The
        eigenvalues an invariant measure is given are the covariance operator's
        in that space's metric, so a draw's coefficients carry
        ``eigenvalue / gram``. Checked exactly rather than by sampling."""
        _, space = geometry
        degrees = space.degrees
        measure = space.power_measure(np.full(degrees.max() + 1, 4.0))

        expected = np.bincount(
            degrees,
            weights=measure.covariance.eigenvalues / space.apply_gram(np.ones(space.dim)),
        )
        drawn = np.mean(
            [space.power_spectrum(measure.sample(rng=rng)) for _ in range(400)],
            axis=0,
        )
        present = np.unique(degrees)
        assert drawn[present] == pytest.approx(expected[present], rel=0.35)

    def test_analysis_and_synthesis_invert_each_other(self, geometry, rng):
        """And are *not* each other's adjoint on anything but L2 -- which is
        why there are two of them, rather than one and its adjoint. v1 wrote
        the difference into the adjoint by hand as a power of the radius."""
        from pygeoinf2.testing import check_operator

        _, space = geometry
        analysis = space.coefficient_operator(lmax=3, lmin=1)
        synthesis = space.from_coefficient_operator(lmax=3, lmin=1)
        check_operator(analysis, rng=rng)
        check_operator(synthesis, rng=rng)

        coefficients = rng.standard_normal(synthesis.domain.dim)
        assert analysis(synthesis(coefficients)) == pytest.approx(coefficients)

    def test_synthesis_is_not_the_adjoint_of_analysis(self, geometry, rng):
        """Stated as a test because it is the trap: on a Sobolev space the
        adjoint carries the metric, so using it as synthesis is wrong by the
        Sobolev symbol."""
        _, space = geometry
        analysis = space.coefficient_operator(lmax=3, lmin=1)
        synthesis = space.from_coefficient_operator(lmax=3, lmin=1)

        coefficients = rng.standard_normal(synthesis.domain.dim)
        theirs = analysis.adjoint(coefficients)
        mine = synthesis(coefficients)
        assert space.norm(space.subtract(theirs, mine)) > 1e-8 * space.norm(mine)

    def test_a_band_outside_the_space_is_refused(self, geometry):
        _, space = geometry
        with pytest.raises(ValueError, match="lmin <= lmax"):
            space.coefficient_operator(lmax=space.degrees.max() + 1)
        with pytest.raises(ValueError, match="lmin <= lmax"):
            space.from_coefficient_operator(lmin=3, lmax=2)


class TestPowerSpectrumAgainstPyshtools:
    """v1 got a spectrum by drawing samples and calling SHCoeffs.spectrum.
    This is the same number, on this library's scale."""

    def test_it_is_pyshtools_spectrum_up_to_the_orthonormal_scale(self, rng):
        sh = pytest.importorskip("pyshtools")
        from pygeoinf2.symmetric_space.sphere import Sobolev

        space = Sobolev(8, 2.0, 0.3, radius=2.0)
        field = space.random(rng=rng)

        mine = space.power_spectrum(field)
        theirs = space.to_coefficients(field).spectrum(convention="power")
        area = 4.0 * np.pi * space.radius**2
        assert mine == pytest.approx(area * theirs)


class TestTransformOptions:
    """The NUFFT's accuracy and thread count, reachable from the operator."""

    def test_the_route_does_not_change_the_answer(self, geometry, rng):
        """Whichever side of the crossover, and whatever the thread count.
        The thresholds were retuned, so this is what says the retuning was a
        change of speed and not of result."""
        _, space = geometry
        field = space.random(rng=rng)
        points = space.random_points(300, rng=rng)

        reference = space.evaluate(field, points)
        for nthreads in (1, 2, 0):
            assert space.evaluate(field, points, nthreads=nthreads) == pytest.approx(
                reference, rel=1e-7, abs=1e-9
            )

    def test_a_loose_accuracy_is_looser(self, geometry, rng):
        """eps reaches the transform: asking for less gets less. On the direct
        route there is no transform to ask, and it is ignored -- which is why
        this only asserts a bound, not a difference."""
        _, space = geometry
        field = space.random(rng=rng)
        points = space.random_points(300, rng=rng)

        exact = space.basis_matrix(points) @ space.to_components(field)
        loose = space.evaluate(field, points, eps=1e-4)
        assert loose == pytest.approx(exact, rel=1e-3, abs=1e-4 * np.abs(exact).max())

    def test_the_operator_passes_them_down(self, geometry, rng):
        _, space = geometry
        field = space.random(rng=rng)
        points = space.random_points(300, rng=rng)

        operator = space.point_evaluation_operator(points, nthreads=2, eps=1e-8)
        assert operator(field) == pytest.approx(
            space.evaluate(field, points), rel=1e-6, abs=1e-8
        )
        assert operator.adjoint(operator(field)) is not None

    def test_the_defaults_are_left_to_the_geometry(self, geometry, rng):
        """Passing nothing must not impose a common eps: a sphere's default is
        1e-10 and a box's 1e-12, and the operator has no business flattening
        that."""
        _, space = geometry
        points = space.random_points(10, rng=rng)
        assert space._transform_options(None, None) == {}
        assert space._transform_options(1e-6, None) == {"eps": 1e-6}
        assert space._transform_options(None, 3) == {"nthreads": 3}


class TestBothEvaluationRoutesAgree:
    """The direct sum and the non-uniform FFT are the same map.

    They have to be, since which one runs is a performance threshold the
    caller never sees. On a padded Box they were not: the NUFFT route took
    the point's own coordinate where the direct route took it relative to the
    enclosing grid, so the field came out displaced by exactly the padding.
    It went unseen because the old crossover needed 512 points to reach it.
    """

    def test_the_two_routes_agree(self, geometry, rng):
        _, space = geometry
        field = space.random(rng=rng)
        points = space.random_points(400, rng=rng)

        direct = space.basis_matrix(points) @ space.to_components(field)
        assert space.evaluate(field, points) == pytest.approx(
            direct, rel=1e-8, abs=1e-8 * np.abs(direct).max()
        )

    def test_the_two_adjoint_routes_agree(self, geometry, rng):
        _, space = geometry
        points = space.random_points(400, rng=rng)
        weights = rng.standard_normal(400)

        direct = space.basis_matrix(points).T @ weights
        assert space.accumulate(weights, points) == pytest.approx(
            direct, rel=1e-8, abs=1e-8 * np.abs(direct).max()
        )

    def test_it_is_the_padding_that_was_wrong(self, rng):
        """Named directly, because the general test above only says the two
        disagree and not why. A Box with no padding never showed the bug."""
        from pygeoinf2.symmetric_space import box as box_module

        space = box_module.Sobolev((64,), 2.0, 0.3, bounds=((5.0, 6.0),), padding=0.25)
        assert space.grid_axes[0][0] == pytest.approx(4.75)

        field = space.random(rng=rng)
        points = space.random_points(400, rng=rng)
        direct = space.basis_matrix(points) @ space.to_components(field)
        assert space.evaluate(field, points) == pytest.approx(
            direct, rel=1e-8, abs=1e-8 * np.abs(direct).max()
        )


class TestSharedTablesAndSmallFixes:
    """A cluster of Should items from the review, each with its own reason."""

    def test_a_truncation_degree_does_not_depend_on_tie_order(self, geometry):
        """The power in a degree is summed before the degrees are walked. On a
        box the components of one degree have different eigenvalues and so
        different power, and cumulating the sorted components instead left the
        answer depending on how the sort broke the ties."""
        _, space = geometry
        symbol = lambda eigenvalues: (1.0 + 0.05 * eigenvalues) ** -4.0

        degree = space.estimate_truncation_degree(symbol)
        power = np.bincount(space.degrees, weights=symbol(space.laplacian_eigenvalues))
        held = power[: degree + 1].sum() / power.sum()

        assert held >= 1.0 - 1e-3
        assert power[:degree].sum() / power.sum() < 1.0 - 1e-3


class TestSphereTables:
    """Shared between spaces of the same truncation, and still correct."""

    def test_they_are_what_the_loops_built(self):
        """Vectorised, so this is the check that the vectorisation is exact."""
        pytest.importorskip("pyshtools")
        from pyshtools.legendre import PlmIndex

        from pygeoinf2.symmetric_space.sphere import _legendre_indices_for, _packing_for

        lmax = 12
        parts, degrees, orders = _packing_for(lmax)

        expected = []
        for degree in range(lmax + 1):
            expected += [(0, degree, order) for order in range(degree + 1)]
            expected += [(1, degree, order) for order in range(1, degree + 1)]
        assert list(zip(parts.tolist(), degrees.tolist(), orders.tolist())) == expected

        assert _legendre_indices_for(lmax).tolist() == [
            PlmIndex(degree, order) for degree, order in zip(degrees, orders)
        ]

    def test_changing_the_order_reuses_them(self):
        """with_order makes a new space, and multiplication_operator makes one
        on every call. Rebuilding these there was 231 ms at lmax 256."""
        pytest.importorskip("pyshtools")
        from pygeoinf2.symmetric_space.sphere import Sobolev

        space = Sobolev(16, 2.0, 0.3)
        other = space.with_order(0.0)

        assert other._packing[0] is space._packing[0]
        assert other._legendre_indices is space._legendre_indices
        assert other._quadrature is space._quadrature

    def test_they_are_read_only(self):
        """Shared, so a caller must not be able to edit everyone's copy."""
        pytest.importorskip("pyshtools")
        from pygeoinf2.symmetric_space.sphere import Sobolev

        space = Sobolev(8, 2.0, 0.3)
        for table in (*space._packing, space._legendre_indices, space._quadrature):
            with pytest.raises(ValueError):
                table[0] = 0

    def test_a_short_arc_keeps_its_precision(self):
        """arccos of a dot product loses half its digits for nearby points,
        which is where the short paths are: at 1e-8 degrees apart it returns
        zero, a relative error of one. atan2 of the cross and dot products is
        exact there, as geodesic_distance already knew."""
        pytest.importorskip("pyshtools")
        from pygeoinf2.symmetric_space.sphere import Sobolev

        space = Sobolev(8, 2.0, 0.3)
        for separation in (1e-2, 1e-6, 1e-8):
            start, end = np.array([0.0, 0.0]), np.array([0.0, separation])
            _, weights = space.geodesic_quadrature(start, end, count=5)
            assert weights.sum() == pytest.approx(
                space.radius * np.radians(separation), rel=1e-12
            )


class TestCorrelatedMeasureAccessors:
    """Reading a joint measure back, which v1 could and v2 could not."""

    @staticmethod
    def _measure(space):
        variances = [
            space.sobolev_symbol(-2.0, 0.2),
            space.heat_symbol(0.14),
        ]
        return space.correlated_measure_from_correlations(
            variances, np.array([[1.0, 0.5], [0.5, 1.0]]), labels=("a", "b")
        )

    def test_a_marginal_is_the_invariant_measure_it_was_built_from(self, geometry):
        """And keeps its diagonal structure while doing it. Reading the block
        off the covariance's dense component matrix -- the only way there was
        -- costs a dim-by-dim assembly to look at one summand."""
        _, space = geometry
        measure = self._measure(space)

        marginal = measure.marginal(0)
        assert marginal.covariance.domain is space
        assert marginal.covariance.eigenvalues == pytest.approx(
            space.sobolev_symbol(-2.0, 0.2)
        )

    def test_a_marginal_can_still_be_sampled(self, geometry, rng):
        """A diagonal block carries its own factor. Sandwiching the covariance
        between projections instead gives a composition, which does not."""
        _, space = geometry
        marginal = self._measure(space).marginal(1)
        assert marginal.can_sample
        assert marginal.sample(rng=rng) is not None

    def test_labels_work_as_well_as_positions(self, geometry):
        _, space = geometry
        measure = self._measure(space)
        assert measure.marginal("b").covariance.eigenvalues == pytest.approx(
            measure.marginal(1).covariance.eigenvalues
        )

    def test_the_correlations_come_back(self, geometry):
        """Wherever there is a correlation to report. A heat symbol underflows
        to zero at the shortest wavelengths a circle carries, and there the
        convention below applies instead."""
        _, space = geometry
        live = (
            space.sobolev_symbol(-2.0, 0.2) * space.heat_symbol(0.14)
        ) > 0.0
        correlations = space.spectral_correlations(self._measure(space))
        assert correlations[live] == pytest.approx(0.5)
        assert correlations[~live] == pytest.approx(0.0)

    def test_they_come_back_scale_by_scale(self, geometry):
        """The point of the construction: two fields may agree at long
        wavelengths and not at short ones, which one number cannot say."""
        _, space = geometry
        variance = space.sobolev_symbol(-2.0, 0.2)
        wanted = np.linspace(0.9, -0.9, space.dim)

        sigma = np.zeros((space.dim, 2, 2))
        sigma[:, 0, 0] = sigma[:, 1, 1] = variance
        sigma[:, 0, 1] = sigma[:, 1, 0] = wanted * variance

        measure = space.correlated_measure(sigma)
        live = variance > 0.0
        assert space.spectral_correlations(measure)[live] == pytest.approx(
            wanted[live]
        )

    def test_a_mode_with_no_variance_has_no_correlation(self, geometry):
        """0/0, and v1's convention is zero -- there is nothing to report."""
        _, space = geometry
        variance = space.sobolev_symbol(-2.0, 0.2).copy()
        variance[:3] = 0.0

        sigma = np.zeros((space.dim, 2, 2))
        sigma[:, 0, 0] = sigma[:, 1, 1] = sigma[:, 0, 1] = sigma[:, 1, 0] = variance

        assert space.spectral_correlations(space.correlated_measure(sigma))[
            :3
        ] == pytest.approx(0.0)

    def test_the_cross_covariance_is_the_off_diagonal_block(self, geometry):
        _, space = geometry
        measure = self._measure(space)

        cross = measure.cross_covariance(0, 1)
        assert cross.domain is space and cross.codomain is space
        assert cross.eigenvalues == pytest.approx(
            0.5
            * np.sqrt(space.sobolev_symbol(-2.0, 0.2) * space.heat_symbol(0.14))
        )

    def test_they_are_refused_off_a_direct_sum(self, geometry):
        _, space = geometry
        measure = space.invariant_measure(space.heat_symbol(0.14))
        with pytest.raises(ValueError, match="direct sum"):
            measure.marginal(0)
        with pytest.raises(ValueError, match="direct sum"):
            measure.cross_covariance(0, 1)


class TestTruncationDegreeFor:
    """Choosing lmax from the prior, before there is a space to ask."""

    def test_it_gives_v1s_answer(self):
        """The rule is v1's, so the number is v1's: order 2, length scale 0.2
        and rtol 1e-8 gave 354 there."""
        pytest.importorskip("pyshtools")
        from pygeoinf2.symmetric_space.sphere import Sphere

        assert Sphere.truncation_degree_for(2.0, 0.2) == 354

    def test_it_needs_no_space(self):
        """Static, because the answer is what you pass to the constructor."""
        pytest.importorskip("pyshtools")
        from pygeoinf2.symmetric_space.sphere import Sobolev, Sphere

        degree = Sphere.truncation_degree_for(3.0, 0.3)
        assert Sobolev(degree, 3.0, 0.3).lmax == degree

    def test_the_length_scale_is_read_against_the_radius(self):
        pytest.importorskip("pyshtools")
        from pygeoinf2.symmetric_space.sphere import Sphere

        assert Sphere.truncation_degree_for(
            2.0, 1000.0, radius=6371.0
        ) == Sphere.truncation_degree_for(2.0, 1000.0 / 6371.0)

    def test_a_power_of_two_is_available(self):
        pytest.importorskip("pyshtools")
        from pygeoinf2.symmetric_space.sphere import Sphere

        plain = Sphere.truncation_degree_for(2.0, 0.2)
        rounded = Sphere.truncation_degree_for(2.0, 0.2, power_of_two=True)
        assert rounded >= plain
        assert rounded & (rounded - 1) == 0

    def test_a_slower_spectrum_needs_more_degrees(self):
        pytest.importorskip("pyshtools")
        from pygeoinf2.symmetric_space.sphere import Sphere

        assert Sphere.truncation_degree_for(1.5, 0.2) > Sphere.truncation_degree_for(
            3.0, 0.2
        )

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            (dict(order=1.0, length_scale=0.2), "order"),
            (dict(order=2.0, length_scale=0.2, rtol=0.0), "tolerance"),
            (dict(order=2.0, length_scale=-1.0), "positive"),
        ],
    )
    def test_it_refuses_what_it_cannot_answer(self, kwargs, message):
        pytest.importorskip("pyshtools")
        from pygeoinf2.symmetric_space.sphere import Sphere

        order = kwargs.pop("order")
        length_scale = kwargs.pop("length_scale")
        with pytest.raises(ValueError, match=message):
            Sphere.truncation_degree_for(order, length_scale, **kwargs)


class TestTheGeometryIsRequired:
    """A space cannot be built without the geometry it claims to have.

    Which is the structural version of what this file tests by example: the
    review found that every geometric method existed on the sphere alone, and
    a base class that raises NotImplementedError lets that happen quietly. An
    abstract method makes it a construction error instead.
    """

    def test_an_incomplete_space_cannot_be_constructed(self):
        from pygeoinf2.symmetric_space.base import SymmetricSpace

        class Incomplete(SymmetricSpace):
            pass

        with pytest.raises(TypeError, match="abstract"):
            Incomplete()

    @pytest.mark.parametrize(
        "name",
        [
            "degrees",
            "reference_point",
            "walk_from",
            "spatial_dimension",
            "gaussian_curvature",
            "geodesic_distance",
            "geodesic_quadrature",
            "geodesic_ball_quadrature",
            "project_function",
            "random_point",
        ],
    )
    def test_each_geometric_primitive_is_required(self, name):
        from pygeoinf2.symmetric_space.base import SymmetricSpace

        assert name in SymmetricSpace.__abstractmethods__

    def test_every_concrete_space_supplies_them(self, geometry):
        """And they are the space's own, not inherited stubs."""
        from pygeoinf2.symmetric_space.base import SymmetricSpace

        _, space = geometry
        for name in SymmetricSpace.__abstractmethods__:
            assert getattr(type(space), name, None) is not getattr(
                SymmetricSpace, name, None
            )
