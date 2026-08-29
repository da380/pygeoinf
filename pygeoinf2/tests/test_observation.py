"""
The observation layer: the operators that connect a space to real data.

Everything here is a linear operator built from *derivative components*, so
every test ends up asking the same question in a different setting — does the
metric enter exactly once, in the adjoint? Where an operator has two routes to
the same answer, the two are compared rather than one being trusted.

See DESIGN.md sections 20.1 and 20.5.
"""

import numpy as np
import pytest

from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.testing import check_operator

pyshtools = pytest.importorskip("pyshtools")

from pygeoinf2.symmetric_space import Sobolev as BoxSobolev  # noqa: E402
from pygeoinf2.symmetric_space.sphere import Lebesgue, Sobolev  # noqa: E402

RADIUS = 2.0


@pytest.fixture
def space():
    return Sobolev(24, 2.0, 0.2, radius=RADIUS)


@pytest.fixture
def lebesgue():
    return Lebesgue(24, radius=RADIUS)


class TestEvaluation:
    def test_the_fast_route_matches_the_generic_one(self, space, rng):
        """The sphere expands once; the base class sums the basis per point."""
        points = space.random_points(9, rng=rng)
        x = space.random(rng=rng)
        components = space.to_components(x)
        generic = np.array(
            [float(np.dot(space.basis_at(point), components)) for point in points]
        )
        assert np.allclose(space.evaluate(x, points), generic)

    def test_point_evaluation_is_matrix_free_and_assembles(self, space, rng):
        points = space.random_points(5, rng=rng)
        A = space.point_evaluation_operator(points)
        check_operator(A, rng=rng)

        assembled = A.assembled()
        x, y = space.random(rng=rng), np.arange(1.0, 6.0)
        assert np.allclose(A(x), assembled(x))
        assert np.allclose(
            space.to_components(A.adjoint(y)),
            space.to_components(assembled.adjoint(y)),
        )

    def test_the_adjoint_returns_a_sum_of_representers(self, space, rng):
        points = space.random_points(3, rng=rng)
        A = space.point_evaluation_operator(points)
        weights = np.array([2.0, -1.0, 0.5])
        expected = space.representer(
            sum(w * space.basis_at(p) for w, p in zip(weights, points))
        )
        assert np.allclose(
            space.to_components(A.adjoint(weights)), space.to_components(expected)
        )


class TestGeodesics:
    def test_pole_to_equator_is_a_quarter_circumference(self, space):
        pole = np.array([90.0, 0.0])
        equator = np.array([0.0, 74.5])
        assert space.geodesic_distance(pole, equator) == pytest.approx(
            np.pi * RADIUS / 2.0
        )

    def test_distance_is_symmetric_and_vanishes(self, space, rng):
        a, b = space.random_point(rng=rng), space.random_point(rng=rng)
        assert space.geodesic_distance(a, b) == pytest.approx(
            space.geodesic_distance(b, a)
        )
        assert space.geodesic_distance(a, a) == pytest.approx(0.0, abs=1e-12)

    def test_arc_weights_sum_to_the_arc_length(self, space, rng):
        a, b = space.random_point(rng=rng), space.random_point(rng=rng)
        _, weights = space.geodesic_quadrature(a, b, count=12)
        assert weights.sum() == pytest.approx(space.geodesic_distance(a, b))

    def test_arc_nodes_lie_on_the_sphere_and_between_the_ends(self, space, rng):
        a, b = space.random_point(rng=rng), space.random_point(rng=rng)
        separation = space.geodesic_distance(a, b)
        nodes, _ = space.geodesic_quadrature(a, b, count=8)
        for node in nodes:
            along = space.geodesic_distance(a, node) + space.geodesic_distance(node, b)
            assert along == pytest.approx(separation)

    def test_close_points_keep_their_precision(self, space):
        """The negative control for atan2 over arccos.

        The cosine is flat near zero separation, so ``acos(u . v)`` throws away
        half its digits exactly where a localisation radius needs them.
        """
        pole = np.array([90.0, 0.0])
        separation = 1.0e-6  # radians of arc
        nearby = np.array([90.0 - np.degrees(separation), 0.0])
        exact = RADIUS * separation
        assert space.geodesic_distance(pole, nearby) == pytest.approx(exact, rel=1e-12)

        first, second = space._to_vector(pole), space._to_vector(nearby)
        by_arccos = RADIUS * np.arccos(np.clip(np.dot(first, second), -1.0, 1.0))
        assert abs(by_arccos - exact) > 1.0e-6 * exact

    def test_antipodal_endpoints_are_refused(self, space):
        pole = np.array([90.0, 0.0])
        other = np.array([-90.0, 0.0])
        with pytest.raises(ValueError, match="antipodal"):
            space.geodesic_quadrature(pole, other, count=4)

    def test_ball_weights_sum_to_the_cap_area(self, space, rng):
        centre = space.random_point(rng=rng)
        radius = 0.3 * RADIUS
        _, weights = space.geodesic_ball_quadrature(centre, radius, count=200)
        exact = 2.0 * np.pi * RADIUS**2 * (1.0 - np.cos(radius / RADIUS))
        assert weights.sum() == pytest.approx(exact, rel=1e-12)

    def test_ball_nodes_lie_inside_the_ball(self, space, rng):
        """The rule is now built as one array rather than ring by ring, so
        this pins that the rings and their azimuths still line up."""
        centre = space.random_point(rng=rng)
        radius = 0.3 * RADIUS
        nodes, weights = space.geodesic_ball_quadrature(centre, radius, count=200)
        assert len(nodes) == weights.size == 200
        distances = np.array(
            [space.geodesic_distance(centre, node) for node in nodes]
        )
        assert distances.max() <= radius + 1e-12
        assert distances.max() > 0.5 * radius

    def test_a_gauss_rule_is_computed_once(self):
        """REVIEW2 4.2.6: `leggauss` solves an eigenproblem, and every path
        asks for the same few counts -- 0.8 s for 2000 of them."""
        from pygeoinf2.symmetric_space.base import _gauss_legendre

        abscissae, weights = _gauss_legendre(12)
        again = _gauss_legendre(12)
        assert again[0] is abscissae and again[1] is weights
        reference = np.polynomial.legendre.leggauss(12)
        assert np.array_equal(abscissae, reference[0])
        assert np.array_equal(weights, reference[1])
        with pytest.raises(ValueError):
            abscissae[0] = 0.0


class TestAverages:
    """Constant fields are the calibration: an average of one must be one."""

    def test_the_cap_average_of_one_is_one(self, lebesgue, rng):
        one = lebesgue.project_function(lambda point: 1.0)
        centre = lebesgue.random_point(rng=rng)
        assert lebesgue.spherical_cap_average(centre, 0.15)(one) == pytest.approx(1.0)

    def test_the_cap_integral_of_one_is_the_area(self, lebesgue, rng):
        one = lebesgue.project_function(lambda point: 1.0)
        centre = lebesgue.random_point(rng=rng)
        angular = 8.6  # degrees, as every angle on a sphere now is
        area = 2.0 * np.pi * RADIUS**2 * (1.0 - np.cos(np.radians(angular)))
        assert lebesgue.spherical_cap_integral(centre, angular)(one) == pytest.approx(
            area
        )

    def test_the_closed_form_agrees_with_the_rotated_indicator(self, lebesgue, rng):
        """REVIEW2 4.2.5. The components used to come from
        ``SHCoeffs.from_cap``, which builds the cap at the pole and rotates it
        -- 8.5 ms a centre at lmax 128, against 0.2 ms for the addition
        theorem. This is the check that they are the same components."""
        from pyshtools import SHCoeffs

        from pygeoinf2.symmetric_space.sphere import _NO_CONDON_SHORTLEY

        centres = lebesgue.random_points(5, rng=rng)
        for angular in (2.0, 37.0, 90.0, 172.0):
            rotated = []
            for centre in centres:
                cap = SHCoeffs.from_cap(
                    angular,
                    lebesgue.lmax,
                    clat=float(centre[0]),
                    clon=float(centre[1]),
                    normalization="ortho",
                    csphase=_NO_CONDON_SHORTLEY,
                    kind="real",
                    degrees=True,
                )
                parts, degrees, orders = lebesgue._packing
                coefficients = cap.to_array(lmax=lebesgue.lmax)[parts, degrees, orders]
                fraction = 0.5 * (1.0 - np.cos(np.radians(angular)))
                rotated.append(
                    coefficients
                    / (lebesgue.radius * 4.0 * np.pi)
                    * lebesgue.area
                    * fraction
                )
            closed = lebesgue.cap_integral_components(centres, angular)
            assert np.allclose(closed, np.stack(rotated), atol=1e-10)

    def test_the_closed_form_normalises_by_the_cap_area(self, lebesgue, rng):
        centres = lebesgue.random_points(3, rng=rng)
        angular = 24.0
        area = (
            2.0 * np.pi * lebesgue.radius**2 * (1.0 - np.cos(np.radians(angular)))
        )
        integrals = lebesgue.cap_integral_components(centres, angular)
        averages = lebesgue.cap_integral_components(centres, angular, normalise=True)
        assert np.allclose(averages * area, integrals)

    def test_a_cap_of_zero_area_has_no_average(self, lebesgue):
        assert np.allclose(lebesgue.cap_integral_components([[0.0, 0.0]], 0.0), 0.0)
        with pytest.raises(ValueError, match="no average"):
            lebesgue.cap_integral_components([[0.0, 0.0]], 0.0, normalise=True)
        with pytest.raises(ValueError, match=r"\[0, 180\]"):
            lebesgue.cap_integral_components([[0.0, 0.0]], 181.0)

    def test_exact_and_quadrature_cap_averages_agree(self, lebesgue, rng):
        """The whole reason for the exact route is that it is cheaper."""
        centre = lebesgue.random_point(rng=rng)
        radius = 0.2 * RADIUS
        field = lebesgue.random(rng=rng)
        exact = lebesgue.geodesic_ball_average_operator([centre], radius)
        quadrature = lebesgue.geodesic_ball_average_operator(
            [centre], radius, count=4000
        )
        assert exact(field)[0] == pytest.approx(quadrature(field)[0], rel=1e-4)

    def test_the_path_average_of_one_is_one(self, lebesgue, rng):
        one = lebesgue.project_function(lambda point: 1.0)
        a, b = lebesgue.random_point(rng=rng), lebesgue.random_point(rng=rng)
        A = lebesgue.path_average_operator([(a, b)], count=12)
        assert A(one)[0] == pytest.approx(1.0)

    def test_the_path_integral_of_one_is_the_arc_length(self, lebesgue, rng):
        """The integral is its own method now, rather than a keyword on the
        average: a travel time is an integral along a ray, and v1 computes
        exactly this while calling it an average."""
        one = lebesgue.project_function(lambda point: 1.0)
        a, b = lebesgue.random_point(rng=rng), lebesgue.random_point(rng=rng)
        A = lebesgue.path_integral_operator([(a, b)], count=12)
        assert A(one)[0] == pytest.approx(lebesgue.geodesic_distance(a, b))

    def test_the_node_count_follows_the_length_scale(self, rng):
        """Left to itself, the quadrature resolves the field it is integrating:
        two nodes per length scale, as v1 does. A fixed count is
        under-resolved for a long path on a short-scale space."""
        from pygeoinf2.symmetric_space.sphere import Sobolev

        short = Sobolev(24, 2.0, 0.1, radius=2.0)
        long = Sobolev(24, 2.0, 0.8, radius=2.0)
        a, b = np.array([10.0, 20.0]), np.array([-30.0, 80.0])
        one = short.with_order(0.0).project_function(lambda p: 1.0)

        # Both still integrate a constant exactly; the finer one simply uses
        # more nodes to do it.
        for space in (short, long):
            assert space.path_integral_operator([(a, b)])(one)[0] == pytest.approx(
                space.geodesic_distance(a, b)
            )
        arc = short.geodesic_distance(a, b)
        assert int(np.ceil(2.0 * arc / short.length_scale)) > int(
            np.ceil(2.0 * arc / long.length_scale)
        )

    def test_a_weight_along_the_path_is_applied(self, lebesgue, rng):
        """For a slowness that varies along the ray for reasons other than the
        field being solved for."""
        one = lebesgue.project_function(lambda point: 1.0)
        a, b = lebesgue.random_point(rng=rng), lebesgue.random_point(rng=rng)
        plain = lebesgue.path_integral_operator([(a, b)], count=12)
        doubled = lebesgue.path_integral_operator(
            [(a, b)], count=12, weight=lambda point: 2.0
        )
        assert doubled(one)[0] == pytest.approx(2.0 * plain(one)[0])

    def test_the_path_average_is_the_quadrature_sum(self, space, rng):
        """The W E factorisation must agree with doing it by hand."""
        a, b = space.random_point(rng=rng), space.random_point(rng=rng)
        nodes, weights = space.geodesic_quadrature(a, b, count=10)
        field = space.random(rng=rng)
        by_hand = float(np.dot(weights, space.evaluate(field, nodes))) / weights.sum()
        A = space.path_average_operator([(a, b)], count=10)
        assert A(field)[0] == pytest.approx(by_hand)

    def test_the_averaging_operators_have_working_adjoints(self, space, rng):
        a, b = space.random_point(rng=rng), space.random_point(rng=rng)
        centre = space.random_point(rng=rng)
        check_operator(space.path_average_operator([(a, b)], count=8), rng=rng)
        check_operator(
            space.geodesic_ball_average_operator([centre], 0.2 * RADIUS), rng=rng
        )
        check_operator(
            space.geodesic_ball_average_operator([centre], 0.2 * RADIUS, count=120),
            rng=rng,
        )

    def test_an_empty_set_of_paths_is_refused(self, space):
        with pytest.raises(ValueError, match="At least one path"):
            space.path_average_operator([])


class TestCoefficients:
    def test_the_operator_reads_off_the_components(self, space, rng):
        A = space.coefficient_operator(lmax=3)
        x = space.random(rng=rng)
        degrees = space._packing[1]
        assert np.allclose(A(x), space.to_components(x)[degrees <= 3])

    def test_a_degree_band_selects_the_right_count(self, space):
        A = space.coefficient_operator(lmin=2, lmax=4)
        assert A.codomain.dim == 5 + 7 + 9

    def test_the_adjoint_is_a_representer(self, space, rng):
        A = space.coefficient_operator(lmax=1)
        check_operator(A, rng=rng)
        y = np.array([1.0, 0.0, 0.0, 0.0])
        representer = space.to_components(A.adjoint(y))
        assert representer[0] == pytest.approx(1.0 / space.metric_values[0])
        assert np.allclose(representer[1:], 0.0)

    def test_degrees_outside_the_space_are_refused(self, space):
        with pytest.raises(ValueError, match="Degrees must satisfy"):
            space.coefficient_operator(lmax=space.lmax + 1)


class TestResolution:
    def test_truncation_keeps_the_low_degrees(self, space, rng):
        coarse = space.with_degree(8)
        P = space.degree_transfer_operator(coarse)
        x = space.random(rng=rng)
        kept = space._packing[1] <= 8
        assert np.allclose(coarse.to_components(P(x)), space.to_components(x)[kept])

    def test_prolongation_pads_with_zeros(self, space, rng):
        fine = space.with_degree(32)
        P = space.degree_transfer_operator(fine)
        x = space.random(rng=rng)
        components = fine.to_components(P(x))
        assert np.allclose(components[: space.dim], space.to_components(x))
        assert np.allclose(components[space.dim :], 0.0)

    def test_the_transfer_operators_have_working_adjoints(self, space, rng):
        check_operator(space.degree_transfer_operator(space.with_degree(8)), rng=rng)
        check_operator(space.degree_transfer_operator(space.with_degree(32)), rng=rng)

    def test_truncation_and_prolongation_are_adjoint(self, space, rng):
        """True because the two spaces share a metric on their common degrees.

        Derived rather than asserted: the adjoint comes from the derivative
        components, so it stays right when the metrics differ and this identity
        does not.
        """
        coarse = space.with_degree(8)
        restrict = space.degree_transfer_operator(coarse)
        prolong = coarse.degree_transfer_operator(space)
        y = coarse.random(rng=rng)
        assert np.allclose(
            space.to_components(restrict.adjoint(y)),
            space.to_components(prolong(y)),
        )

    def test_a_different_radius_is_refused(self, space):
        with pytest.raises(ValueError, match="common radius"):
            space.degree_transfer_operator(Sobolev(8, 2.0, 0.2, radius=1.0))


class TestAcquisitionGeometry:
    def test_the_station_table_loads(self, space):
        stations = space.stations()
        assert len(stations) > 100
        for point in stations:
            assert -90.0 <= point[0] <= 90.0
            assert -180.0 <= point[1] <= 180.0

    def test_a_named_station_lands_where_it_should(self, space):
        """AAK is in Kyrgyzstan: 42.6 N, 74.5 E."""
        first = space.stations()[0]
        # Straight out of the table: a point is what the catalogue holds.
        assert first[0] == pytest.approx(42.6375)
        assert first[1] == pytest.approx(74.4942)

    def test_the_catalogue_filters_by_magnitude(self, space):
        assert len(space.earthquakes(minimum_magnitude=6.0)) < len(space.earthquakes())

    def test_a_subsample_is_without_replacement(self, space, rng):
        points = space.stations(count=20, rng=rng)
        assert len({tuple(point) for point in points}) == 20

    def test_asking_for_too_many_is_refused(self, space, rng):
        with pytest.raises(ValueError, match="from a table of"):
            space.stations(count=100000, rng=rng)

    def test_pairs_within_distance_finds_the_diagonal_and_no_more(self, space, rng):
        points = space.random_points(30, rng=rng)
        rows, columns = space.pairs_within_distance(points, 0.0)
        assert np.array_equal(rows, np.arange(30))
        assert np.array_equal(columns, np.arange(30))

    def test_pairs_within_distance_is_symmetric(self, space, rng):
        points = space.random_points(25, rng=rng)
        rows, columns = space.pairs_within_distance(points, 0.6 * RADIUS)
        found = set(zip(rows.tolist(), columns.tolist()))
        assert all((j, i) in found for i, j in found)


class TestPointwiseVariance:
    """The parameterisation a modeller actually has an opinion about."""

    def test_it_matches_the_covariance_of_a_dirac(self, space):
        """Independent computation: (C u, u) with u the Dirac's representer."""
        variances = space.sobolev_symbol(-2.0, 0.15)
        measure = space.invariant_measure(variances)
        representer = space.dirac(space.reference_point).representer
        direct = space.inner_product(measure.covariance(representer), representer)
        assert space.pointwise_variance(variances) == pytest.approx(direct)

    def test_it_is_the_same_at_every_point(self, space, rng):
        """Homogeneity, which is why naming one reference point is honest."""
        variances = space.sobolev_symbol(-2.0, 0.15)
        elsewhere = space.dirac(space.random_point(rng=rng)).representer
        direct = space.inner_product(
            space.invariant_measure(variances).covariance(elsewhere), elsewhere
        )
        assert space.pointwise_variance(variances) == pytest.approx(direct)

    @pytest.mark.parametrize("shape", [(6, 4), (7, 5)])
    def test_it_is_the_same_at_every_grid_point_of_a_box(self, shape):
        """The homogeneity the docstring now claims, and the one that holds:
        the basis is orthonormal against the grid's own quadrature, so the sum
        of its squares is the reciprocal of the cell weight at every sample.

        A flat spectrum on a coarse grid, which is the worst case there is.
        """
        import itertools

        from pygeoinf2.symmetric_space.fourier import Lebesgue as PeriodicLebesgue

        X = PeriodicLebesgue(shape, lengths=(1.0, 2.0))
        variances = np.ones(X.dim)
        expected = X.pointwise_variance(variances)
        for point in itertools.product(*X.grid_axes):
            basis = X.basis_at(np.array(point))
            assert np.isclose(
                float(np.sum(variances * basis**2 / X.metric_values)), expected
            )

    def test_between_the_grid_points_an_even_axis_is_the_exception(self, rng):
        """The claim the docstring used to make and could not keep, pinned
        both ways round. An even axis holds a Nyquist cosine whose sine the
        grid cannot see, so the basis is orthonormal on the grid and not
        isotropic off it, and the interpolated variance dips between samples.
        With every axis odd there is no such mode and the value is constant
        everywhere.
        """
        from pygeoinf2.symmetric_space.fourier import Lebesgue as PeriodicLebesgue

        def spread(shape):
            X = PeriodicLebesgue(shape, lengths=(1.0, 2.0))
            variances = np.ones(X.dim)
            between = [
                float(
                    np.sum(
                        variances
                        * X.basis_at(X.random_point(rng=rng)) ** 2
                        / X.metric_values
                    )
                )
                for _ in range(50)
            ]
            return X.pointwise_variance(variances), min(between)

        odd_value, odd_lowest = spread((7, 5))
        assert odd_lowest == pytest.approx(odd_value)

        even_value, even_lowest = spread((6, 4))
        assert even_lowest < 0.95 * even_value

    def test_dropping_the_metric_would_give_a_different_answer(self, space):
        """The negative control for the 1/g factor in pointwise_variance.

        On a Lebesgue space the metric is the identity and the mistake is
        invisible, which is exactly why it needs pinning on a Sobolev one.
        """
        variances = space.sobolev_symbol(-2.0, 0.15)
        basis = space.basis_at(space.reference_point)
        naive = float(np.dot(variances, basis**2))
        assert not np.isclose(naive, space.pointwise_variance(variances))

    def test_calibration_hits_the_requested_standard_deviation(self, space):
        measure = space.sobolev_measure(2.0, 0.15, pointwise_std=0.05)
        representer = space.dirac(space.reference_point).representer
        variance = space.inner_product(measure.covariance(representer), representer)
        assert np.sqrt(variance) == pytest.approx(0.05)

    def test_calibration_shows_up_in_samples(self, space, rng):
        measure = space.sobolev_measure(2.0, 0.15, pointwise_std=0.05)
        point = [space.reference_point]
        draws = np.array(
            [space.evaluate(measure.sample(rng=rng), point)[0] for _ in range(600)]
        )
        assert draws.std() == pytest.approx(0.05, rel=0.15)

    def test_calibration_leaves_the_spectrum_shape_alone(self, space):
        """Only the amplitude moves, so the correlation length is untouched."""
        plain = space.sobolev_measure(2.0, 0.15)
        scaled = space.sobolev_measure(2.0, 0.15, pointwise_std=0.05)
        first = plain.covariance.eigenvalues
        second = scaled.covariance.eigenvalues
        ratios = second[first > 0] / first[first > 0]
        assert np.allclose(ratios, ratios[0])

    def test_it_works_on_a_periodic_box_too(self):
        """Nothing in the calculation is spherical."""
        box = BoxSobolev((64,), 2.0, 0.05, lengths=(1.0,))
        measure = box.heat_measure(0.032, pointwise_std=3.0)
        representer = box.dirac(box.reference_point).representer
        variance = box.inner_product(measure.covariance(representer), representer)
        assert np.sqrt(variance) == pytest.approx(3.0)

    def test_a_non_positive_standard_deviation_is_refused(self, space):
        with pytest.raises(ValueError, match="must be positive"):
            space.sobolev_measure(2.0, 0.15, pointwise_std=0.0)


class TestWeightOperator:
    """The sparse half of the W E factorisation."""

    def test_it_is_its_own_transpose_adjoint(self, rng):
        from pygeoinf2.symmetric_space.base import _weight_matrix, _weight_operator

        sparse = _weight_matrix(
            2, 4, [0, 0, 1, 1], [0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0]
        )
        W = _weight_operator(sparse)
        check_operator(W, rng=rng)
        assert isinstance(W, LinearOperator)
        assert W.domain == EuclideanSpace(4)
        assert np.allclose(W(np.array([1.0, 1.0, 1.0, 1.0])), [3.0, 7.0])

    def test_the_dense_route_never_densifies_the_weights(self, space, rng):
        """REVIEW2 4.2.6. `weights.matrix()` built a (paths, nodes) array that
        holds one entry per node: 1.59 s against 0.018 s at 2000 paths."""
        paths = list(
            zip(space.random_points(6, rng=rng), space.random_points(6, rng=rng))
        )
        field = space.random(rng=rng)
        free = space.path_integral_operator(paths, count=8)
        dense = space.path_integral_operator(paths, count=8, dense=True)
        assert np.allclose(free(field), dense(field))
        assert np.allclose(
            space.to_components(free.adjoint(np.arange(1.0, 7.0))),
            space.to_components(dense.adjoint(np.arange(1.0, 7.0))),
        )


class TestPointConvention:
    """D-2: points are ``(latitude, longitude)`` in degrees, everywhere."""

    def test_the_converters_invert_each_other(self, space, rng):
        points = np.array(space.random_points(50, rng=rng))
        back = space.to_latitude_degrees(space.to_colatitude_radians(points))
        assert back == pytest.approx(points)

    def test_a_catalogue_point_is_the_catalogue_s_numbers(self, space):
        """The loaders used to convert privately, so a caller who read the same
        file themselves got different answers from the same numbers."""
        first = space.stations()[0]
        assert first[0] == pytest.approx(42.6375)
        assert first[1] == pytest.approx(74.4942)

    def test_a_field_is_sampled_where_the_point_says(self, space):
        """sin(latitude) is exactly the degree-one zonal harmonic, so this is
        about the convention and not about truncation."""
        field = space.project_function(lambda p: np.sin(np.radians(p[0])))
        for latitude in (-90.0, -45.0, 0.0, 45.0, 90.0):
            value = space.evaluate(field, [np.array([latitude, 13.0])])[0]
            assert value == pytest.approx(np.sin(np.radians(latitude)), abs=1e-8)

    def test_an_angle_is_degrees_and_a_distance_is_not(self, space):
        """A cap's half-angle is an angle; a ball's radius is a length in the
        units of the sphere's radius. Mixing them was how the ball average
        first disagreed with the exact cap."""
        one = space.with_order(0.0).project_function(lambda p: 1.0)
        centre = np.array([12.0, 34.0])
        lebesgue = space.with_order(0.0)

        half_angle = 8.0  # degrees
        by_angle = lebesgue.spherical_cap_integral(centre, half_angle)(one)
        expected = 2.0 * np.pi * RADIUS**2 * (1.0 - np.cos(np.radians(half_angle)))
        assert by_angle == pytest.approx(expected)

        # The same cap, reached through the physical radius.
        distance = np.radians(half_angle) * RADIUS
        by_distance = lebesgue.geodesic_ball_average_operator([centre], distance)
        assert by_distance(one)[0] == pytest.approx(1.0, rel=1e-6)
