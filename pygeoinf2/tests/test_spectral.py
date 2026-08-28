"""
Degree-wise operators on a symmetric space, and the statistics of a measure.

The operators here all have a closed form in the spectral basis, so most of
these tests compare against one. The measure statistics have two routes — a
spectral one for a diagonal covariance and a dense one for anything — and the
point of testing them is that the two agree, since only the second is obviously
right and only the first is affordable.

See DESIGN.md sections 20.5 (S) and 21.2 (P).
"""

import numpy as np
import pytest

from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.symmetric_space import Lebesgue as BoxLebesgue
from pygeoinf2.symmetric_space import Sobolev as BoxSobolev
from pygeoinf2.testing import check_operator, check_traits
from pygeoinf2.traits import Traits

from .conftest import make_weighted_space

pyshtools = pytest.importorskip("pyshtools")

from pygeoinf2.symmetric_space.sphere import Lebesgue, Sobolev  # noqa: E402


class TestDegrees:
    def test_the_sphere_reports_harmonic_degrees(self):
        X = Lebesgue(4)
        assert X.degrees[0] == 0
        assert list(X.degrees[:4]) == [0, 1, 1, 1]
        for degree in range(5):
            assert X.degree_multiplicity(degree) == 2 * degree + 1

    def test_a_box_reports_wavenumber_magnitudes(self):
        X = BoxLebesgue((16,), lengths=(1.0,))
        assert X.degrees.min() == 0
        assert X.degrees.max() == 8


class TestSpectralOperators:
    def test_an_explicit_symbol_becomes_a_diagonal_operator(self, rng):
        X = Sobolev(6, 2.0, 0.2)
        values = 1.0 / (2.0 * X.degrees + 1.0)
        operator = X.spectral_operator(values)
        check_operator(operator, rng=rng)
        assert np.allclose(operator.eigenvalues, values)

    def test_a_wrong_length_is_refused(self):
        X = Lebesgue(4)
        with pytest.raises(ValueError, match="values"):
            X.spectral_operator(np.ones(3))

    def test_the_band_projection_is_a_projection(self, rng):
        X = Sobolev(8, 2.0, 0.2)
        projection = X.spectral_projection_operator(lmin=2, lmax=5)
        check_operator(projection, rng=rng)
        check_traits(projection, rng=rng)
        assert Traits.IDEMPOTENT & projection.traits
        x = X.random(rng=rng)
        assert np.allclose(projection(projection(x)), projection(x))

    def test_the_band_projection_keeps_exactly_its_band(self, rng):
        X = Sobolev(8, 2.0, 0.2)
        projection = X.spectral_projection_operator(lmin=2, lmax=5)
        components = X.to_components(projection(X.random(rng=rng)))
        outside = (X.degrees < 2) | (X.degrees > 5)
        assert np.allclose(components[outside], 0.0)
        assert not np.allclose(components[~outside], 0.0)

    def test_it_complements_the_coefficient_operator(self, rng):
        """One stays in the space; the other maps out of it."""
        X = Sobolev(6, 2.0, 0.2)
        inside = X.spectral_projection_operator(lmax=3)
        outward = X.coefficient_operator(lmax=3)
        assert inside.codomain == X
        assert outward.codomain.dim == 16
        x = X.random(rng=rng)
        assert np.allclose(X.to_components(inside(x))[X.degrees <= 3], outward(x))


class TestOrderInclusion:
    def test_it_acts_as_the_identity(self, rng):
        X = Sobolev(6, 2.0, 0.2)
        inclusion = X.order_inclusion_operator(X.with_order(1.0))
        check_operator(inclusion, rng=rng)
        x = X.random(rng=rng)
        assert np.allclose(inclusion(x), x)

    def test_its_adjoint_is_not_the_identity(self, rng):
        """Which is the whole content: the metrics differ, so the adjoint does.

        Reading a function in a different Sobolev order is a relabelling of the
        vector and a genuine change to every inner product it takes part in.
        """
        X = Sobolev(6, 2.0, 0.2)
        target = X.with_order(1.0)
        inclusion = X.order_inclusion_operator(target)
        y = target.random(rng=rng)
        assert not np.allclose(inclusion.adjoint(y), y)

    def test_mismatched_dimensions_are_refused(self):
        X = Lebesgue(6)
        with pytest.raises(ValueError, match="matching dimensions"):
            X.order_inclusion_operator(Lebesgue(3))


class TestL2Products:
    def test_the_rows_are_l2_inner_products(self, rng):
        X = Sobolev(6, 2.0, 0.2)
        base = X.with_order(0.0)
        fields = [X.random(rng=rng) for _ in range(3)]
        operator = X.l2_products_operator(fields)
        check_operator(operator, rng=rng)
        x = X.random(rng=rng)
        assert np.allclose(
            operator(x), [base.inner_product(field, x) for field in fields]
        )

    def test_it_means_the_same_at_every_order(self, rng):
        """The L2 products, not this space's -- so the order does not enter."""
        base = Lebesgue(6)
        fields = [base.random(rng=rng) for _ in range(2)]
        x = base.random(rng=rng)
        first = base.l2_products_operator(fields)(x)
        second = Sobolev(6, 2.0, 0.2).l2_products_operator(fields)(x)
        assert np.allclose(first, second)

    def test_an_empty_set_is_refused(self):
        with pytest.raises(ValueError, match="At least one field"):
            Lebesgue(4).l2_products_operator([])


class TestTruncationDegree:
    def test_a_steeper_spectrum_needs_fewer_degrees(self):
        X = Sobolev(24, 2.0, 0.2)
        shallow = X.estimate_truncation_degree(lambda lam: (1.0 + 0.5**2 * lam) ** -1.5)
        steep = X.estimate_truncation_degree(lambda lam: (1.0 + 0.5**2 * lam) ** -6.0)
        assert steep < shallow

    def test_a_tighter_tolerance_needs_more_degrees(self):
        X = Sobolev(24, 2.0, 0.2)

        def symbol(eigenvalues):
            return (1.0 + 0.5**2 * eigenvalues) ** -3.0

        loose = X.estimate_truncation_degree(symbol, tolerance=1e-1)
        tight = X.estimate_truncation_degree(symbol, tolerance=1e-4)
        assert tight >= loose

    def test_a_tolerance_outside_the_unit_interval_is_refused(self):
        with pytest.raises(ValueError, match="tolerance"):
            Lebesgue(4).estimate_truncation_degree(lambda lam: lam, tolerance=2.0)


class TestMeasureStatistics:
    @pytest.fixture
    def pair(self, rng):
        X = make_weighted_space()
        size = X.dim
        first = rng.normal(size=(size, size))
        second = rng.normal(size=(size, size))
        return (
            X,
            GaussianMeasure.from_covariance_matrix(
                X, first @ first.T + size * np.identity(size)
            ),
            GaussianMeasure.from_covariance_matrix(
                X, second @ second.T + size * np.identity(size)
            ),
        )

    def test_kl_matches_the_dense_formula(self, pair, rng):
        space, first, second = pair
        gram = space.gram_matrix()
        p = first.covariance.matrix(form="galerkin")
        q = second.covariance.matrix(form="galerkin")
        shift = space.to_components(
            space.subtract(second.expectation, first.expectation)
        )
        weighted = gram @ shift
        reference = 0.5 * (
            np.trace(np.linalg.solve(q, p))
            + weighted @ np.linalg.solve(q, weighted)
            - space.dim
            + np.linalg.slogdet(q)[1]
            - np.linalg.slogdet(p)[1]
        )
        assert first.kl_divergence(second) == pytest.approx(reference)

    def test_kl_of_a_measure_from_itself_vanishes(self, pair):
        _, first, _ = pair
        assert first.kl_divergence(first) == pytest.approx(0.0, abs=1e-9)

    def test_kl_is_not_symmetric(self, pair):
        _, first, second = pair
        assert first.kl_divergence(second) != pytest.approx(second.kl_divergence(first))

    def test_the_spectral_route_agrees_with_the_dense_one(self):
        """The reason the fast path exists is that it is affordable, not that
        it is different."""
        X = Sobolev(6, 2.0, 0.2)
        first = X.sobolev_measure(2.0, 0.2)
        second = X.heat_measure(0.02)
        assert first.kl_divergence(second) == pytest.approx(
            first.kl_divergence(second, method="dense")
        )

    @pytest.mark.slow
    def test_the_stochastic_route_agrees_with_the_dense_one(self, pair):
        """Nothing is formed on this route, so it is the one that survives a
        space too large to hold two covariance matrices. It is checked in
        units of its own standard error, which is the only tolerance a
        Hutchinson estimate has."""
        _, first, second = pair
        exact = first.kl_divergence(second, method="dense")
        estimate = first.kl_divergence_estimate(
            second,
            method="stochastic",
            samples=6000,
            rng=np.random.default_rng(7),
            max_iterations=60,
            rtol=1e-10,
        )
        assert estimate.standard_error > 0.0
        assert abs(estimate.value - exact) < 4.0 * estimate.standard_error

    def test_the_exact_routes_report_no_error(self, pair):
        """So a caller can treat all three uniformly and still see which it
        got, which is the whole reason an Estimate comes back."""
        _, first, second = pair
        for method in ("dense",):
            assert (
                first.kl_divergence_estimate(second, method=method).standard_error
                == 0.0
            )

    @pytest.mark.slow
    def test_auto_takes_the_spectral_route_when_it_can(self):
        X = Sobolev(6, 2.0, 0.2)
        first = X.sobolev_measure(2.0, 0.2)
        second = X.heat_measure(0.02)
        assert first.kl_divergence_estimate(second).standard_error == 0.0
        forced = first.kl_divergence_estimate(
            second,
            method="stochastic",
            samples=4000,
            rng=np.random.default_rng(9),
            max_iterations=60,
            rtol=1e-10,
        )
        exact = first.kl_divergence(second)
        assert abs(forced.value - exact) < 4.0 * forced.standard_error

    def test_auto_refuses_rather_than_going_stochastic_unasked(self, pair):
        """The stochastic route has to be named.

        It is the only inexact route, and on the spectra this library produces
        it has returned -88.6 +/- 21.7 for a divergence of zero. Silently
        selecting it turns a wrong answer into the default one, so ``"auto"``
        raises and says what the alternatives are.
        """
        _, first, second = pair
        with pytest.raises(ValueError, match="method='stochastic'"):
            first.kl_divergence(second, dense_limit=0)

        forced = first.kl_divergence_estimate(
            second, method="stochastic", dense_limit=0, rng=np.random.default_rng(3)
        )
        assert forced.standard_error > 0.0

    def test_a_bad_method_is_refused_and_so_is_a_route_that_does_not_apply(self, pair):
        _, first, second = pair
        with pytest.raises(ValueError, match="'auto', 'spectral'"):
            first.kl_divergence(second, method="lanczos")
        with pytest.raises(ValueError, match="both covariances diagonal"):
            first.kl_divergence(second, method="spectral")

    def test_measures_on_different_spaces_are_refused(self, pair):
        space, first, _ = pair
        other = GaussianMeasure.from_standard_deviation(EuclideanSpace(2), 1.0)
        with pytest.raises(ValueError, match="same space"):
            first.kl_divergence(other)

    def test_the_nuclear_norm_is_the_total_variance(self):
        X = Sobolev(6, 2.0, 0.2)
        measure = X.sobolev_measure(2.0, 0.2)
        assert measure.nuclear_norm() == pytest.approx(
            float(np.sum(measure.covariance.eigenvalues))
        )
        assert measure.nuclear_norm() == pytest.approx(
            measure.nuclear_norm(method="dense")
        )

    def test_the_norms_are_basis_independent(self, pair):
        """A trace is the *component* matrix's, not the Galerkin matrix's.

        The two differ by a factor of the metric, and on a weighted space they
        are visibly different numbers -- so this is the check that the norm
        means what it says rather than what is convenient to compute.
        """
        space, first, _ = pair
        components = first.covariance.matrix(form="components")
        galerkin = first.covariance.matrix(form="galerkin")
        assert not np.isclose(np.trace(components), np.trace(galerkin))
        assert first.nuclear_norm() == pytest.approx(np.trace(components))

    def test_the_hilbert_schmidt_norm_agrees_both_ways(self):
        X = Sobolev(6, 2.0, 0.2)
        measure = X.sobolev_measure(2.0, 0.2)
        assert measure.hilbert_schmidt_norm() == pytest.approx(
            measure.hilbert_schmidt_norm(method="dense")
        )

    def test_directional_statistics(self, pair, rng):
        space, first, _ = pair
        u, v = space.random(rng=rng), space.random(rng=rng)
        assert first.directional_variance(u) == pytest.approx(
            space.inner_product(first.covariance(u), u)
        )
        assert first.directional_covariance(u, v) == pytest.approx(
            first.directional_covariance(v, u)
        )
        assert first.directional_variance(u) > 0.0


class TestTwoPointCovariance:
    def test_at_its_anchor_it_is_the_pointwise_variance(self):
        X = Sobolev(12, 2.0, 0.2)
        symbol = X.sobolev_symbol(-2.0, 0.2)
        measure = X.invariant_measure(symbol)
        anchor = X.reference_point
        field = measure.two_point_covariance(anchor)
        assert X.evaluate(field, [anchor])[0] == pytest.approx(
            X.pointwise_variance(symbol)
        )

    def test_it_agrees_with_a_pair_of_diracs(self, rng):
        """The definition, done the expensive way."""
        X = Sobolev(10, 2.0, 0.2)
        measure = X.sobolev_measure(2.0, 0.2)
        anchor, other = X.reference_point, X.random_point(rng=rng)
        field = measure.two_point_covariance(anchor)
        pair = measure.directional_covariance(
            X.dirac(anchor).representer, X.dirac(other).representer
        )
        assert X.evaluate(field, [other])[0] == pytest.approx(pair)

    def test_it_decays_with_distance(self, rng):
        X = Sobolev(24, 2.0, 0.05)
        field = X.heat_measure(0.002).two_point_covariance(X.reference_point)
        near = X.evaluate(field, [np.array([0.05, 0.0])])[0]
        far = X.evaluate(field, [np.array([1.5, 0.0])])[0]
        assert near > far

    def test_a_space_without_points_is_refused(self, rng):
        measure = GaussianMeasure.from_standard_deviation(EuclideanSpace(3), 1.0)
        with pytest.raises(TypeError, match="evaluation functional"):
            measure.two_point_covariance(0)


class TestBoxSpectral:
    def test_a_box_supports_the_same_operators(self, rng):
        X = BoxSobolev((32,), 2.0, 0.05, lengths=(1.0,))
        check_operator(X.spectral_projection_operator(lmax=4), rng=rng)
        check_operator(X.order_inclusion_operator(X.with_order(1.0)), rng=rng)
        check_operator(X.l2_products_operator([X.random(rng=rng)]), rng=rng)


class TestNormCalibration:
    def test_norm_std_hits_the_total_size(self):
        X = Sobolev(16, 2.0, 0.2)
        measure = X.sobolev_measure(2.0, 0.2, norm_std=3.0)
        # E||x||^2 is the trace of the covariance
        assert np.sqrt(measure.nuclear_norm()) == pytest.approx(3.0)

    def test_norm_std_shows_up_in_samples(self, rng):
        X = Sobolev(12, 2.0, 0.2)
        measure = X.sobolev_measure(2.0, 0.2, norm_std=3.0)
        draws = [measure.sample(rng=rng) for _ in range(400)]
        root_mean_square = np.sqrt(np.mean([X.squared_norm(x) for x in draws]))
        assert root_mean_square == pytest.approx(3.0, rel=0.1)

    def test_the_two_calibrations_are_alternatives(self):
        X = Sobolev(8, 2.0, 0.2)
        with pytest.raises(ValueError, match="not both"):
            X.sobolev_measure(2.0, 0.2, norm_std=1.0, pointwise_std=1.0)

    def test_a_non_positive_norm_is_refused(self):
        X = Sobolev(8, 2.0, 0.2)
        with pytest.raises(ValueError, match="must be positive"):
            X.heat_measure(0.01, norm_std=0.0)


class TestPowerMeasure:
    def test_each_degree_holds_the_power_it_was_given(self):
        """The spectrum a modeller writes down is per degree, not per mode.

        The two differ by the multiplicity, which is the whole of the method.
        """
        X = Lebesgue(12)
        power = np.array([(1.0 + degree) ** -3.0 for degree in range(X.lmax + 1)])
        measure = X.power_measure(power)
        eigenvalues = measure.covariance.eigenvalues
        for degree in (0, 2, 5, 12):
            held = eigenvalues[X.degrees == degree].sum()
            assert held == pytest.approx(power[degree])

    def test_a_callable_spectrum_works_too(self):
        X = Lebesgue(8)
        measure = X.power_measure(lambda degree: (1.0 + degree) ** -2.0)
        eigenvalues = measure.covariance.eigenvalues
        assert eigenvalues[X.degrees == 3].sum() == pytest.approx(4.0**-2.0)

    def test_too_short_a_spectrum_is_refused(self):
        X = Lebesgue(8)
        with pytest.raises(ValueError, match="degree"):
            X.power_measure(np.ones(3))


class TestCovarianceFunction:
    def test_it_starts_at_the_pointwise_variance(self):
        X = Sobolev(24, 2.0, 0.05)
        symbol = X.sobolev_symbol(-2.0, 0.05)
        measure = X.invariant_measure(symbol)
        values = X.covariance_function(measure, np.array([0.0, 0.1]))
        assert values[0] == pytest.approx(X.pointwise_variance(symbol))

    def test_it_falls_away_from_the_origin(self):
        X = Sobolev(48, 2.0, 0.05)
        values = X.covariance_function(
            X.heat_measure(0.002), np.array([0.0, 0.05, 0.2, 0.6])
        )
        assert values[0] > values[1] > values[2]

    def test_a_longer_correlation_length_decays_more_slowly(self):
        X = Sobolev(48, 2.0, 0.05)
        distance = np.array([0.3])
        short = X.covariance_function(X.heat_measure(0.002), distance)
        long = X.covariance_function(X.heat_measure(0.02), distance)
        assert long[0] / X.pointwise_variance(X.heat_symbol(0.02)) > short[
            0
        ] / X.pointwise_variance(X.heat_symbol(0.002))
