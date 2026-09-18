"""
Degree-wise operators on a symmetric space, and the statistics of a measure.

The operators here all have a closed form in the spectral basis, so most of
these tests compare against one. The measure statistics have two routes — a
spectral one for a diagonal covariance and a dense one for anything — and the
point of testing them is that the two agree, since only the second is obviously
right and only the first is affordable.
"""

import numpy as np
import pytest

from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.symmetric_space import Lebesgue as BoxLebesgue
from pygeoinf2.symmetric_space import Sobolev as BoxSobolev
from pygeoinf2.testing import check_operator, check_traits
from pygeoinf2.traits import Traits

from .conftest import make_dense_metric_space, make_weighted_space, values

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


class TestSpectralLabels:
    """The packing made public, vectorized: v1's ``indices``,
    ``index_to_integer`` and ``integer_to_index`` as arrays of labels and
    one method placing a label."""

    def test_the_sphere_labels_every_component_once(self):
        X = Lebesgue(6)
        degrees, orders = X.degrees, X.orders
        assert degrees.shape == orders.shape == (X.dim,)
        assert np.all(np.abs(orders) <= degrees)
        labels = set(zip(degrees.tolist(), orders.tolist()))
        assert len(labels) == X.dim
        # The round trip, one label at a time and all at once.
        for i in range(X.dim):
            assert X.component_of(degrees[i], orders[i]) == i
        assert np.array_equal(X.component_of(degrees, orders), np.arange(X.dim))
        # v1's convention: cosines first within a degree, then sines.
        assert X.component_of(2, 0) == 4
        assert X.component_of(2, 2) == 6
        assert X.component_of(2, -1) == 7
        with pytest.raises(ValueError, match="label"):
            X.component_of(3, 4)
        with pytest.raises(ValueError, match="label"):
            X.component_of(7, 0)

    def test_a_symbol_written_against_the_orders(self, rng):
        """A zonal projection, which no function of the degree can express."""
        X = Lebesgue(6)
        zonal = X.spectral_operator(np.where(X.orders == 0, 1.0, 0.0))
        x = X.random(rng=rng)
        c = X.to_components(zonal(x))
        assert np.allclose(c[X.orders != 0], 0.0)
        assert np.allclose(c[X.orders == 0], X.to_components(x)[X.orders == 0])
        # The zonal coefficient of degree three is where component_of says.
        assert c[X.component_of(3, 0)] == pytest.approx(
            X.to_components(x)[X.component_of(3, 0)]
        )

    @pytest.mark.parametrize("shape", [(8,), (6, 8), (4, 6, 8)])
    def test_a_box_labels_every_component_once(self, shape):
        X = BoxLebesgue(shape, lengths=tuple(float(n) for n in shape))
        wavevectors, phases = X.wavevectors, X.phases
        assert wavevectors.shape == (len(shape), X.dim)
        assert phases.shape == (X.dim,)
        labels = set(zip(map(tuple, wavevectors.T.tolist()), phases.tolist()))
        assert len(labels) == X.dim
        assert np.array_equal(
            X.component_of(wavevectors, phase=phases), np.arange(X.dim)
        )
        for i in range(0, X.dim, 7):
            assert X.component_of(wavevectors[:, i], phase=phases[i]) == i
        # The degree is the magnitude, rounded down.
        assert np.array_equal(
            X.degrees,
            np.floor(np.sqrt((wavevectors.astype(float) ** 2).sum(axis=0))).astype(int),
        )
        # The constant mode is self-conjugate: it has no sine.
        zero = np.zeros(len(shape), dtype=int)
        assert X.phases[X.component_of(zero)] == 0
        with pytest.raises(ValueError, match="self-conjugate"):
            X.component_of(zero, phase=1)
        with pytest.raises(ValueError, match="wavevector on this box"):
            X.component_of(np.zeros(len(shape) + 1, dtype=int))

    def test_named_coefficients_as_a_property_operator(self, rng):
        """The application: a linear operator from a field to a chosen set
        of its coefficients, named by label, in the order asked for."""
        X = Lebesgue(5)
        positions = X.component_of([2, 2, 4], [0, -1, 3])
        pick = X.coefficient_operator(components=positions)
        assert pick.codomain.dim == 3
        x = X.random(rng=rng)
        assert np.allclose(pick(x), X.to_components(x)[positions])
        check_operator(pick, rng=rng)
        # Synthesis puts them back where they belong, and nowhere else.
        place = X.from_coefficient_operator(components=positions)
        c = np.array([1.0, -2.0, 0.5])
        back = X.to_components(place(c))
        assert np.allclose(back[positions], c)
        assert np.allclose(np.delete(back, positions), 0.0)
        check_operator(place, rng=rng)
        with pytest.raises(ValueError, match="not both"):
            X.coefficient_operator(lmax=2, components=positions)
        with pytest.raises(ValueError, match="repeat"):
            X.coefficient_operator(components=[1, 1])
        with pytest.raises(ValueError, match="lie in"):
            X.coefficient_operator(components=[X.dim])
        # On a box, by wavevector.
        Y = BoxLebesgue((8, 8), lengths=(1.0, 1.0))
        modes = np.array([[1, 0], [0, 2], [3, 3]]).T
        where = Y.component_of(modes, phase=[0, 1, 0])
        y = Y.random(rng=rng)
        assert np.allclose(
            Y.coefficient_operator(components=where)(y), Y.to_components(y)[where]
        )

    def test_a_box_symbol_written_against_the_wavevectors(self, rng):
        """An anisotropic filter, keeping modes that vary along one axis only."""
        X = BoxLebesgue((8, 8), lengths=(1.0, 1.0))
        along_x = X.spectral_operator(np.where(X.wavevectors[1] == 0, 1.0, 0.0))
        x = X.random(rng=rng)
        c = X.to_components(along_x(x))
        assert np.allclose(c[X.wavevectors[1] != 0], 0.0)
        assert np.allclose(
            c[X.wavevectors[1] == 0], X.to_components(x)[X.wavevectors[1] == 0]
        )


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
        assert np.allclose(*values(X, projection(projection(x)), projection(x)))

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
        assert np.allclose(*values(X, inclusion(x), x))

    def test_its_adjoint_is_not_the_identity(self, rng):
        """Which is the whole content: the metrics differ, so the adjoint does.

        Reading a function in a different Sobolev order is a relabeling of the
        vector and a genuine change to every inner product it takes part in.
        """
        X = Sobolev(6, 2.0, 0.2)
        target = X.with_order(1.0)
        inclusion = X.order_inclusion_operator(target)
        y = target.random(rng=rng)
        assert not np.allclose(*values(X, inclusion.adjoint(y), y))

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
    @pytest.fixture(
        params=[make_weighted_space, make_dense_metric_space],
        ids=["weighted", "dense-metric"],
    )
    def metric(self, request):
        """A KL divergence is metric-sensitive throughout -- the trace term,
        the quadratic form and the log-determinant each involve the Gram --
        and a diagonal one cannot tell a correct implementation from one that
        works in components."""
        return request.param()

    @pytest.fixture
    def pair(self, rng, metric):
        X = metric
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
        second = X.heat_measure(0.14)
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
        second = X.heat_measure(0.14)
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

    def test_a_correlated_measures_norms_come_from_its_slices(self, monkeypatch):
        """One small matrix per mode: the trace is the sum of their traces and
        the Hilbert-Schmidt norm the root of the sum of their squared
        Frobenius norms, in O(dim n^2), as v1 had. The port assembled the
        dense (n dim)^2 block matrix for both."""
        from pygeoinf2.algebra.operators import LinearOperator

        X = Sobolev(6, 2.0, 0.2)
        base = np.array([[1.0, 0.5], [0.5, 2.0]])
        decay = 1.0 / (1.0 + np.arange(X.dim))
        slices = decay[:, None, None] * base[None, :, :]
        measure = X.correlated_measure(slices)
        dense_nuclear = measure.nuclear_norm(method="dense")
        dense_hs = measure.hilbert_schmidt_norm(method="dense")

        monkeypatch.setattr(
            LinearOperator,
            "matrix",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("dense")),
        )
        assert measure.nuclear_norm() == pytest.approx(dense_nuclear)
        assert measure.nuclear_norm() == pytest.approx(np.einsum("kii->", slices))
        assert measure.hilbert_schmidt_norm() == pytest.approx(dense_hs)

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
        field = X.heat_measure(0.045).two_point_covariance(X.reference_point)
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
            X.heat_measure(0.1, norm_std=0.0)


class TestPowerMeasure:
    def test_each_degree_holds_the_power_it_was_given(self):
        """The spectrum a modeler writes down is per degree, not per mode.

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
            X.heat_measure(0.045), np.array([0.0, 0.05, 0.2, 0.6])
        )
        assert values[0] > values[1] > values[2]

    def test_a_longer_correlation_length_decays_more_slowly(self):
        X = Sobolev(48, 2.0, 0.05)
        distance = np.array([0.3])
        short = X.covariance_function(X.heat_measure(0.045), distance)
        long = X.covariance_function(X.heat_measure(0.14), distance)
        assert long[0] / X.pointwise_variance(X.heat_symbol(0.14)) > short[
            0
        ] / X.pointwise_variance(X.heat_symbol(0.045))


class TestInvariantAlgebraDrawsInOneTransform:
    """v1's invariant measure class carried a Karhunen-Loeve sampler and lost
    it whenever the algebra rebuilt the measure; the one-transform draw now
    lives on any diagonal factor, so scaled, summed and marginal measures
    all draw in one synthesis, and a sum touching a zero variance keeps its
    factor."""

    @staticmethod
    def syntheses(space, measure, rng, draws=3):
        """How many syntheses a draw costs, counted on the space's class."""
        count = {"n": 0}
        cls = type(space)
        original = cls.from_components

        def counting(self, c):
            count["n"] += 1
            return original(self, c)

        cls.from_components = counting
        try:
            for _ in range(draws):
                measure.sample(rng=rng)
        finally:
            cls.from_components = original
        return count["n"] / draws

    def test_the_algebra_keeps_the_one_transform_draw(self, rng):
        X = Sobolev(8, 1.5, 0.3)
        prior = X.invariant_measure(lambda k: 1.0 / (1.0 + k) ** 2)
        noise = X.invariant_measure(np.full(X.dim, 0.01))
        assert self.syntheses(X, prior, rng) == 1
        assert self.syntheses(X, 3.0 * prior, rng) == 1
        assert self.syntheses(X, prior / 2.0, rng) == 1
        assert self.syntheses(X, prior + noise, rng) == 1
        assert self.syntheses(X, prior - noise, rng) == 1
        assert self.syntheses(X, prior.translate(X.random(rng=rng)), rng) == 1

    def test_the_algebra_keeps_the_covariance_it_claims(self, rng):
        from pygeoinf2.testing import check_measure

        X = Sobolev(6, 1.0, 0.3)
        prior = X.invariant_measure(lambda k: 1.0 / (1.0 + k) ** 2)
        noise = X.invariant_measure(np.full(X.dim, 0.05))
        for measure in (2.0 * prior, prior + noise, prior - noise):
            check_measure(measure, rng=rng, samples=3000)

    def test_a_sum_with_a_zero_variance_keeps_its_factor(self, rng):
        """A band-limited prior plus anything: v2 refused both the factor
        and the precision when any summed variance was zero, so the sum
        could neither draw in one transform nor say it had no density. A
        square root exists for any non-negative spectrum; only the precision
        needs a positive one."""
        X = Lebesgue(6)
        band = X.invariant_measure(np.where(X.degrees <= 2, 1.0, 0.0))
        low = X.invariant_measure(np.where(X.degrees <= 1, 0.5, 0.0))
        total = band + low
        assert total.covariance_factor is not None
        assert self.syntheses(X, total, rng) == 1
        assert total.precision is None
        draw = total.sample(rng=rng)
        assert np.allclose(X.to_components(draw)[X.degrees > 2], 0.0)

    def test_a_marginal_keeps_factor_and_precision(self, rng):
        X = Lebesgue(4)
        joint = X.correlated_measure_from_correlations(
            np.stack([1.0 / (1.0 + X.degrees), 2.0 / (1.0 + X.degrees)]),
            np.array([[1.0, 0.3], [0.3, 1.0]]),
        )
        first = joint.marginal(0)
        assert first.covariance_factor is not None and first.precision is not None
        assert self.syntheses(X, first, rng) == 1
        assert np.isfinite(first.log_density(X.random(rng=rng)))


class TestWeakenedEllipsoid:
    """v1's fractional credible geometry, between the credible ellipsoid and
    the ambient ball."""

    def test_the_ends_are_the_two_hardenings(self, rng):
        from pygeoinf2.geometry.convex import Ball, Ellipsoid

        X = Sobolev(6, 1.5, 0.3)
        measure = X.invariant_measure(lambda k: 1.0 / (1.0 + k) ** 2)
        full = measure.weakened_ellipsoid(level=0.9, power=1.0)
        exact = measure.credible_set(level=0.9)
        assert isinstance(full, Ellipsoid)
        for _ in range(10):
            x = measure.sample(rng=rng)
            assert full.contains(x) == exact.contains(x)
        ball = measure.weakened_ellipsoid(level=0.9, power=0.0)
        assert isinstance(ball, Ball)
        assert ball.radius == pytest.approx(measure.ambient_ball(level=0.9).radius)

    def test_it_carries_its_level_and_nests(self, rng):
        """Coverage by Monte Carlo at three powers, and the weakened set is
        neither the ellipsoid nor the ball but between them in shape."""
        X = Sobolev(6, 1.5, 0.3)
        measure = X.invariant_measure(lambda k: 1.0 / (1.0 + k) ** 2)
        draws = measure.samples(3000, rng=rng)
        for power in (0.25, 0.5, 0.75):
            weakened = measure.weakened_ellipsoid(level=0.9, power=power)
            assert weakened.has_support_function
            covered = np.mean([weakened.contains(x) for x in draws])
            assert covered == pytest.approx(0.9, abs=0.025)

    def test_a_general_covariance_goes_through_the_calculus(self, rng):
        """A dense, non-diagonal covariance on a weighted space: the spectrum
        by the dense route and the fractional powers by Lanczos, and the
        coverage still holds."""
        from pygeoinf2.probability.gaussian import GaussianMeasure

        X = make_weighted_space()
        raw = rng.normal(size=(X.dim, X.dim))
        # A symmetric positive definite Galerkin matrix is a covariance on
        # any metric; the same numbers as components would not be.
        matrix = raw @ raw.T + X.dim * np.eye(X.dim)
        measure = GaussianMeasure.from_covariance_matrix(X, matrix, form="galerkin")
        weakened = measure.weakened_ellipsoid(level=0.9, power=0.5)
        draws = measure.samples(3000, rng=rng)
        covered = np.mean([weakened.contains(x) for x in draws])
        assert covered == pytest.approx(0.9, abs=0.03)
        assert weakened.has_support_function

    def test_a_bad_power_or_level_is_refused(self):
        X = Lebesgue(4)
        measure = X.invariant_measure(np.ones(X.dim))
        with pytest.raises(ValueError, match="power"):
            measure.weakened_ellipsoid(power=1.5)
        with pytest.raises(ValueError, match="level"):
            measure.weakened_ellipsoid(level=1.0, power=0.5)


class TestSampledPointwiseVariance:
    """v1's Monte Carlo pointwise variance on any module, against the exact
    invariant answer on a symmetric space."""

    def test_it_matches_the_exact_answer(self, rng):
        X = Lebesgue(6)
        variances = 1.0 / (1.0 + X.degrees) ** 2
        measure = X.invariant_measure(variances)
        exact = X.pointwise_variance(variances)
        grid = lambda field: np.asarray(getattr(field, "data", field))  # noqa: E731
        field = measure.sample_pointwise_variance(1500, rng=rng)
        assert np.mean(grid(field)) == pytest.approx(exact, rel=0.08)
        std = measure.sample_pointwise_std(1500, rng=rng)
        assert np.mean(grid(std) ** 2) == pytest.approx(exact, rel=0.15)

    def test_a_space_without_a_product_is_refused(self, rng):
        from pygeoinf2.probability.gaussian import GaussianMeasure

        X = make_weighted_space()
        measure = GaussianMeasure.from_standard_deviation(X, 1.0)
        with pytest.raises(TypeError, match="pointwise product"):
            measure.sample_pointwise_variance(3, rng=rng)
        with pytest.raises(ValueError, match="one draw"):
            Lebesgue(3).invariant_measure(np.ones(16)).sample_pointwise_variance(0)


class TestAdaptiveDiagonals:
    """The deflated diagonal and the pointwise variance on it pass the
    tolerance on to the estimator underneath."""

    def test_the_deflated_diagonal_passes_the_tolerance_on(self, rng, monkeypatch):
        from pygeoinf2.numerics import randomized

        X = EuclideanSpace(16)
        raw = rng.normal(size=(16, 16))
        A = LinearOperator.from_matrix(
            X,
            X,
            raw @ raw.T + np.eye(16),
            form="components",
            traits=Traits.POSITIVE_DEFINITE,
        )
        seen = []
        real = randomized.random_diagonal

        def spy(operator, **kwargs):
            seen.append(dict(kwargs))
            return real(operator, **kwargs)

        monkeypatch.setattr(randomized, "random_diagonal", spy)
        randomized.deflated_diagonal(
            A, rank=3, samples=10, rtol=1e-2, max_samples=300, rng=rng
        )
        assert seen[-1]["rtol"] == 1e-2 and seen[-1]["max_samples"] == 300
        randomized.deflated_diagonal(A, rank=0, samples=10, rtol=5e-3, rng=rng)
        assert seen[-1]["rtol"] == 5e-3

    def test_a_tolerance_takes_the_sampled_route_for_the_pointwise_variance(
        self, rng, monkeypatch
    ):
        """``rtol`` alone selects the sampled route, with the default first
        batch, where ``samples=None`` alone means exact."""
        from pygeoinf2.numerics import randomized

        X = Sobolev(6, 2.0, 0.3)
        measure = X.invariant_measure(lambda k: 1.0 / (1.0 + k) ** 2)
        points = [X.random_point(rng=rng) for _ in range(3)]
        exact = X.pointwise_variance_at(measure, points)
        seen = {}
        real = randomized.deflated_diagonal

        def spy(operator, **kwargs):
            seen.update(kwargs)
            return real(operator, **kwargs)

        monkeypatch.setattr(randomized, "deflated_diagonal", spy)
        sampled = X.pointwise_variance_at(
            measure, points, rtol=1e-2, max_samples=4000, rng=rng
        )
        assert (
            seen["rtol"] == 1e-2
            and seen["samples"] == 20
            and seen["max_samples"] == 4000
        )
        assert np.allclose(sampled, exact, rtol=0.2)


class TestPriorWeightedProbes:
    """v1's ``measure=`` on the range finder: probes drawn from a prior find
    the range the prior lets the data see, which white noise finds badly.
    David remembered this working far better on function spaces; measured,
    it does (DECISIONS.md D-53)."""

    def test_the_prior_weighted_range_captures_what_the_data_can_see(self, rng):
        from pygeoinf2.numerics.randomized import random_range

        X = BoxLebesgue((64,), lengths=(1.0,))
        prior = X.invariant_measure(lambda k: 1.0 / (1.0 + k) ** 2)
        D = EuclideanSpace(40)
        forward = LinearOperator.from_matrix(
            X, D, rng.normal(size=(40, X.dim)) / 8.0, form="components"
        )
        # The composition the prior-weighted range finds: A L, with L the
        # prior's factor. Its unresolved fraction is what is compared.
        seen = forward @ prior.covariance_factor
        matrix = seen.matrix(form="components")

        def unresolved(basis):
            Q = np.column_stack([D.to_components(v) for v in basis])
            residual = matrix - Q @ (Q.T @ matrix)
            return np.linalg.norm(residual, 2) / np.linalg.norm(matrix, 2)

        white = np.mean(
            [
                unresolved(random_range(forward, rank=4, oversampling=2, rng=rng))
                for _ in range(5)
            ]
        )
        weighted = np.mean(
            [
                unresolved(
                    random_range(
                        forward, rank=4, oversampling=2, measure=prior, rng=rng
                    )
                )
                for _ in range(5)
            ]
        )
        assert weighted < 0.5 * white
        # And it is the range finder's option on every factorization.
        from pygeoinf2.numerics.randomized import random_svd

        assert random_svd(forward, rank=4, measure=prior, rng=rng) is not None
        with pytest.raises(ValueError, match="probe measure"):
            random_range(
                forward,
                rank=4,
                measure=X.invariant_measure(np.ones(X.dim)).push_forward(forward),
                rng=rng,
            )
