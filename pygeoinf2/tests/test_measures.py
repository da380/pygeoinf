"""Operations on measures: adjustments, quadratic forms, the hardenings,
conditioning, the trapezoid route to Imhof, quantile sampling and the
correlated measures."""

import warnings

import numpy as np
import pytest

from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.numerics.solvers import (
    CholeskySolver,
)
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.testing import check_operator
from pygeoinf2.traits import Traits

from .conftest import (
    make_dense_metric_space,
    make_weighted_space,
)


class TestMeasureAdjustments:
    @pytest.fixture
    def measure(self, rng):
        """A measure with a covariance and deliberately no precision.

        Built from the factor rather than from ``from_covariance_matrix``,
        which now attaches a precision of its own whenever the matrix it is
        given is nonsingular -- and this class is about what happens when one
        is missing.
        """
        X = EuclideanSpace(6)
        root = rng.normal(size=(6, 6))
        factor = LinearOperator.from_matrix(EuclideanSpace(6), X, root, form="galerkin")
        return (X, root @ root.T, GaussianMeasure(X, covariance_factor=factor))

    def test_a_measure_without_a_precision_says_so(self, measure, rng):
        """Rather than falling back to a dense solve behind a quadratic form."""
        space, _, without = measure
        with pytest.raises(NotImplementedError, match="no precision"):
            without.mahalanobis_squared(space.random(rng=rng))

    def test_rescaling_hits_the_requested_deviation(self, measure, rng):
        space, _, original = measure
        direction = space.random(rng=rng)
        rescaled = original.rescale_directional_variance(direction, 2.5)
        assert np.sqrt(rescaled.directional_variance(direction)) == pytest.approx(2.5)

    def test_rescaling_moves_every_direction_together(self, measure, rng):
        """A recalibration, not a change of shape."""
        space, _, original = measure
        first, second = space.random(rng=rng), space.random(rng=rng)
        rescaled = original.rescale_directional_variance(first, 2.5)
        ratio = rescaled.directional_variance(first) / original.directional_variance(
            first
        )
        assert rescaled.directional_variance(second) == pytest.approx(
            ratio * original.directional_variance(second)
        )

    def test_a_zero_variance_direction_is_refused(self, measure):
        _, _, original = measure
        with pytest.raises(ValueError, match="not positive"):
            original.rescale_directional_variance(original.domain.zero(), 1.0)

    def test_a_non_positive_target_is_refused(self, measure):
        """Zero would be a point mass and a negative deviation is not one."""
        space, _, original = measure
        direction = space.basis_vector(0)
        for target in (0.0, -1.0):
            with pytest.raises(ValueError, match="must be positive"):
                original.rescale_directional_variance(direction, target)

    def test_the_covariance_form_is_required(self):
        """No default: a Galerkin matrix read as components is a silent
        error of one Gram factor."""
        space = make_weighted_space()
        with pytest.raises(TypeError):
            GaussianMeasure.from_covariance_matrix(space, np.identity(space.dim))

    def test_the_sparse_covariance_is_an_operator_built_without_the_matrix(
        self, measure, monkeypatch
    ):
        """The approximation probes columns and never holds the dense matrix,
        so there is no global threshold and no O(N^3) definiteness check on
        an assembled array. And the result is an operator, not
        a measure: a thresholded covariance need not be definite, and a
        sparse one cannot be sampled without a sparse Cholesky."""
        from scipy.sparse import issparse

        space, matrix, original = measure

        def refuse(*args, **kwargs):
            raise AssertionError("the dense matrix was formed")

        monkeypatch.setattr(LinearOperator, "matrix", refuse)
        sparse = original.sparse_covariance(threshold=1e-12)
        assert isinstance(sparse, LinearOperator)
        assert not hasattr(sparse, "sample")
        assert issparse(sparse._stored)
        monkeypatch.undo()
        assert np.allclose(sparse.matrix(form="galerkin"), matrix)

    def test_the_correlation_criterion_is_the_scale_free_one(self):
        """An entry that is a large fraction of the *smaller* variance but a
        small correlation: a correlation criterion drops it; the
        preconditioner's per-column magnitude test keeps it, since in the
        small-variance column it dominates the diagonal, and the symmetrized
        pattern then carries it to the other column too."""
        from pygeoinf2.numerics.preconditioners import sparse_approximation

        space = EuclideanSpace(3)
        covariance = np.array([[1.0, 1e-3, 0.0], [1e-3, 1e-4, 0.0], [0.0, 0.0, 1.0]])
        operator = LinearOperator.from_matrix(
            space,
            space,
            covariance,
            traits=Traits.POSITIVE_SEMIDEFINITE,
            form="galerkin",
        )
        # correlation 1e-3 / sqrt(1e-4) == 0.1, below the threshold of 0.5;
        # against column 1's own diagonal, 1e-3 >= 0.5 * 1e-4.
        by_correlation = sparse_approximation(operator, threshold=0.5)
        by_column = sparse_approximation(operator, threshold=0.5, criterion="column")
        assert by_correlation.matrix(form="galerkin")[0, 1] == 0.0
        assert by_column.matrix(form="galerkin")[0, 1] == pytest.approx(1e-3)
        assert by_correlation.matrix(form="galerkin")[1, 1] == pytest.approx(1e-4)

    def test_the_cap_keeps_the_diagonal_and_the_largest(self, rng):
        from pygeoinf2.numerics.preconditioners import sparse_approximation

        space = make_dense_metric_space(6)
        root = rng.normal(size=(6, 6))
        operator = LinearOperator.from_matrix(
            space,
            space,
            root @ root.T,
            traits=Traits.POSITIVE_DEFINITE,
            form="galerkin",
        )
        capped = sparse_approximation(operator, threshold=0.0, max_per_column=2)
        stored = capped._stored.toarray()
        assert np.all(np.diag(stored) != 0.0)
        assert np.allclose(stored, stored.T)
        check_operator(capped, rng=rng)

    def test_a_supplied_diagonal_is_used(self, rng):
        """For an operator too large to probe twice, an estimate from
        random_diagonal stands in for the exact one."""
        from pygeoinf2.numerics.preconditioners import sparse_approximation

        space = EuclideanSpace(5)
        root = rng.normal(size=(5, 5))
        operator = LinearOperator.from_matrix(
            space,
            space,
            root @ root.T,
            traits=Traits.POSITIVE_DEFINITE,
            form="galerkin",
        )
        # A diagonal so large that every correlation falls below threshold:
        # only the diagonal survives.
        huge = np.full(5, 1e12)
        alone = sparse_approximation(operator, threshold=0.5, diagonal=huge)
        assert alone._stored.nnz == 5
        with pytest.raises(ValueError, match="shape"):
            sparse_approximation(operator, diagonal=np.ones(4))

    def test_the_options_are_validated(self, measure):
        _, _, original = measure
        with pytest.raises(ValueError, match="non-negative"):
            original.sparse_covariance(threshold=-1.0)
        with pytest.raises(ValueError, match="max_per_column"):
            original.sparse_covariance(max_per_column=0)
        with pytest.raises(ValueError, match="criterion"):
            original.sparse_covariance(criterion="sideways")


class TestQuadraticForms:
    """The distribution of ``sum w_i Z_i^2``, which has no closed form."""

    def test_equal_weights_are_an_ordinary_chi_square(self):
        from scipy.stats import chi2

        from pygeoinf2.numerics.quadratic_forms import weighted_chi2_quantile

        for count in (1, 3, 7):
            for level in (0.5, 0.95, 0.99):
                assert weighted_chi2_quantile(np.ones(count), level) == pytest.approx(
                    float(chi2.ppf(level, count))
                )

    def test_a_common_scale_factors_out(self):
        from pygeoinf2.numerics.quadratic_forms import weighted_chi2_quantile

        weights = np.array([2.0, 2.0, 2.0])
        assert weighted_chi2_quantile(weights, 0.9) == pytest.approx(
            2.0 * weighted_chi2_quantile(np.ones(3), 0.9)
        )

    def test_imhof_matches_monte_carlo_for_unequal_weights(self, rng):
        from pygeoinf2.numerics.quadratic_forms import weighted_chi2_cdf

        weights = np.array([5.0, 2.0, 1.0, 0.5, 0.1])
        for value in (2.0, 8.0, 20.0):
            exact = weighted_chi2_cdf(weights, value)
            sampled = weighted_chi2_cdf(
                weights, value, method="monte_carlo", samples=200_000, rng=rng
            )
            assert exact == pytest.approx(sampled, abs=0.005)

    def test_the_quantile_inverts_the_distribution(self):
        from pygeoinf2.numerics.quadratic_forms import (
            weighted_chi2_cdf,
            weighted_chi2_quantile,
        )

        weights = np.array([4.0, 1.0, 0.25])
        for level in (0.1, 0.5, 0.95):
            assert weighted_chi2_cdf(
                weights, weighted_chi2_quantile(weights, level)
            ) == pytest.approx(level, abs=1e-6)

    def test_nonsense_input_is_refused(self):
        from pygeoinf2.numerics.quadratic_forms import (
            weighted_chi2_cdf,
            weighted_chi2_quantile,
        )

        with pytest.raises(ValueError, match="non-negative"):
            weighted_chi2_cdf(np.array([-1.0]), 1.0)
        with pytest.raises(ValueError, match="[Aa]t least one weight"):
            weighted_chi2_cdf(np.zeros(3), 1.0)
        with pytest.raises(ValueError, match="probability"):
            weighted_chi2_quantile(np.ones(2), 1.5)


class TestHardening:
    """Turning a measure into a set, which is never canonical."""

    @pytest.fixture
    def measures(self, rng):
        pairs = []
        # A dense metric as well as diagonal ones. Every metric bug found in
        # this library so far has been an expression that is right when G is
        # diagonal and wrong when it is not, and a fixture of Euclidean and
        # weighted spaces cannot tell the two apart.
        for space in (
            EuclideanSpace(4),
            make_weighted_space(),
            make_dense_metric_space(),
        ):
            root = rng.normal(size=(space.dim, space.dim))
            pairs.append(
                (
                    space,
                    GaussianMeasure.from_covariance_matrix(
                        space,
                        root @ root.T + space.dim * np.identity(space.dim),
                        form="galerkin",
                    ),
                )
            )
        return pairs

    def test_a_credible_set_covers_what_it_claims(self, measures, rng):
        """It did not, on a weighted space, until the precision was built from
        ``G C_gal^-1 G`` rather than from ``C_gal^-1`` -- which agree only on an
        orthonormal basis, and there it covered 46% of its nominal 90%."""
        for space, measure in measures:
            for level in (0.5, 0.9):
                region = measure.credible_set(level=level)
                covered = np.mean(
                    [region.contains(measure.sample(rng=rng)) for _ in range(3000)]
                )
                assert covered == pytest.approx(level, abs=0.03)

    def test_a_measure_without_a_precision_gets_one_through_a_solver(
        self, rng, monkeypatch
    ):
        """No dense inverse: the covariance is inverted by conjugate gradients
        as an operator in the space's metric, so nothing is assembled and the
        set still covers what it claims on every metric. A direct solver,
        passed by name, factorizes once and gives the same region."""

        for space in (
            EuclideanSpace(4),
            make_weighted_space(),
            make_dense_metric_space(),
        ):
            root = rng.normal(size=(space.dim, space.dim)) + 2.0 * np.identity(
                space.dim
            )
            factor = LinearOperator.from_matrix(
                EuclideanSpace(space.dim), space, root, form="components"
            )
            measure = GaussianMeasure(space, covariance_factor=factor)
            assert measure.precision is None

            with monkeypatch.context() as patched:
                patched.setattr(
                    LinearOperator,
                    "matrix",
                    lambda *a, **k: (_ for _ in ()).throw(AssertionError("dense")),
                )
                region = measure.credible_set(level=0.9)
                covered = np.mean(
                    [region.contains(measure.sample(rng=rng)) for _ in range(2000)]
                )
            assert covered == pytest.approx(0.9, abs=0.03)

            direct = measure.credible_set(level=0.9, solver=CholeskySolver())
            for _ in range(20):
                point = space.add(
                    measure.expectation, space.scale(2.0, space.white_noise(rng=rng))
                )
                assert direct.contains(point) == region.contains(point)

    def test_an_ambient_ball_covers_what_it_claims(self, measures, rng):
        for space, measure in measures:
            ball = measure.ambient_ball(level=0.9)
            covered = np.mean(
                [ball.contains(measure.sample(rng=rng)) for _ in range(3000)]
            )
            assert covered == pytest.approx(0.9, abs=0.03)

    def test_the_two_hardenings_are_different_regions(self, measures, rng):
        """Which is what "not canonical" means, made concrete.

        Both carry the same probability and they are not the same region.
        Which way any containment runs depends on how anisotropic the measure
        is — for a strongly anisotropic one the ball simply swallows the
        ellipsoid, and for a mild one each reaches somewhere the other does
        not. What is always true is that they differ, and choosing between them
        is choosing what to say rather than computing one thing two ways.
        """
        for space, measure in measures:
            ball = measure.ambient_ball(level=0.9)
            region = measure.credible_set(level=0.9)
            differing = 0
            for _ in range(600):
                point = space.add(
                    measure.expectation,
                    space.scale(3.0, space.white_noise(rng=rng)),
                )
                differing += ball.contains(point) != region.contains(point)
            assert differing > 0

    def test_the_scipy_view_is_the_covariance_of_the_components(self, measures):
        """``G^-1 C_gal G^-1``, exactly, and symmetric.

        Asserted against the formula rather than against samples: this was
        wrong by 18% on a dense metric and the sampling check that was here
        allowed 35%, so it could not have failed. The expression applied
        ``solve_gram`` to a *matrix* and relied on broadcasting, which divides
        row-wise on a space with a diagonal metric and genuinely solves on one
        without — right in the first case, and not even symmetric in the
        second.
        """
        for space, measure in measures:
            gram = space.gram_matrix()
            galerkin = measure.covariance.matrix(form="galerkin")
            expected = np.linalg.solve(gram, np.linalg.solve(gram, galerkin).T)
            frozen = measure.as_multivariate_normal()
            assert np.allclose(frozen.cov, expected, atol=1e-10)
            assert np.allclose(frozen.cov, frozen.cov.T, atol=1e-12)

    def test_the_scipy_view_has_the_sampled_covariance(self, measures, rng):
        """Independent corroboration of the above, from draws rather than
        algebra. Loose, because 5000 samples is loose — which is why it is the
        corroboration and not the test."""
        for space, measure in measures:
            frozen = measure.as_multivariate_normal()
            draws = np.array(
                [space.to_components(measure.sample(rng=rng)) for _ in range(5000)]
            )
            assert np.allclose(
                frozen.cov, np.cov(draws.T), atol=0.35 * np.abs(frozen.cov).max()
            )


class TestConditioning:
    def test_an_exact_constraint_is_satisfied(self, rng):
        space = make_weighted_space()
        root = rng.normal(size=(space.dim, space.dim))
        measure = GaussianMeasure.from_covariance_matrix(
            space, root @ root.T + space.dim * np.identity(space.dim), form="galerkin"
        )
        operator = LinearOperator.from_matrix(
            space, EuclideanSpace(2), rng.normal(size=(2, space.dim)), form="galerkin"
        )
        value = np.array([0.5, -1.0])
        assert np.allclose(
            operator(measure.condition(operator, value).expectation), value
        )

    def test_a_noisy_constraint_is_the_bayesian_posterior(self, rng):
        """Two routes to one answer: a statement about a measure, and a
        statement about an inverse problem."""
        from pygeoinf2.inference import LinearForwardProblem, LinearGaussianInversion

        space = make_weighted_space()
        root = rng.normal(size=(space.dim, space.dim))
        prior = GaussianMeasure.from_covariance_matrix(
            space, root @ root.T + space.dim * np.identity(space.dim), form="galerkin"
        )
        data_space = EuclideanSpace(2)
        operator = LinearOperator.from_matrix(
            space, data_space, rng.normal(size=(2, space.dim)), form="galerkin"
        )
        noise = GaussianMeasure.from_standard_deviation(data_space, 0.3)
        value = np.array([0.5, -1.0])

        conditioned = prior.condition(operator, value, noise=noise)
        posterior = LinearGaussianInversion(
            LinearForwardProblem(operator, error=noise), prior
        )(value)
        assert np.allclose(
            space.to_components(conditioned.expectation),
            space.to_components(posterior.expectation),
        )
        assert np.allclose(
            conditioned.covariance.matrix(form="components"),
            posterior.covariance.matrix(form="components"),
        )

    def test_conditioning_shrinks_the_total_variance(self, rng):
        space = make_weighted_space()
        root = rng.normal(size=(space.dim, space.dim))
        measure = GaussianMeasure.from_covariance_matrix(
            space, root @ root.T + space.dim * np.identity(space.dim), form="galerkin"
        )
        operator = LinearOperator.from_matrix(
            space, EuclideanSpace(2), rng.normal(size=(2, space.dim)), form="galerkin"
        )
        conditioned = measure.condition(operator, np.array([0.5, -1.0]))
        assert conditioned.nuclear_norm() < measure.nuclear_norm()


class TestImhofOnATrapezoid:
    """Adaptive quadrature on a scalar integrand exhausted its subdivisions,
    warned, and took 106 ms at three weights -- the slowest-decaying case.
    A vectorized trapezoid rule is both faster and quieter."""

    def test_it_is_exact_where_there_is_a_closed_form(self):
        """Equal weights make it a scaled chi-squared. Perturbed by 1e-12 so
        the equal-weight short circuit does not fire and the real quadrature
        runs."""
        from scipy.stats import chi2

        from pygeoinf2.numerics.quadratic_forms import weighted_chi2_cdf

        weights = np.full(7, 2.0) + np.linspace(0.0, 1e-12, 7)
        for value in (5.0, 14.0, 30.0):
            assert weighted_chi2_cdf(weights, value, method="imhof") == pytest.approx(
                chi2.cdf(value / 2.0, 7), abs=1e-9
            )

    def test_it_does_not_warn(self, rng):
        """The old route hit its subdivision limit and said so."""

        from pygeoinf2.numerics.quadratic_forms import weighted_chi2_cdf

        weights = rng.uniform(0.5, 4.0, 3)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            weighted_chi2_cdf(weights, float(weights.sum()), method="imhof")

    def test_it_agrees_with_sampling(self, rng):
        from pygeoinf2.numerics.quadratic_forms import weighted_chi2_cdf

        weights = rng.uniform(0.5, 4.0, 12)
        value = float(weights.sum())
        assert weighted_chi2_cdf(weights, value, method="imhof") == pytest.approx(
            weighted_chi2_cdf(
                weights,
                value,
                method="monte_carlo",
                samples=200_000,
                rng=np.random.default_rng(1),
            ),
            abs=5e-3,
        )


class TestQuantileSampling:
    """The quantile is drawn with an ``rng``; a ``monte_carlo`` method that
    reached a root find on a function returning a different value each call
    would have no root to find."""

    def test_the_sampled_quantile_matches_the_computed_one(self, rng):
        from pygeoinf2.numerics.quadratic_forms import weighted_chi2_quantile

        weights = rng.uniform(0.5, 4.0, 10)
        computed = weighted_chi2_quantile(weights, 0.95)
        sampled = weighted_chi2_quantile(
            weights,
            0.95,
            method="monte_carlo",
            samples=400_000,
            rng=np.random.default_rng(1),
        )
        assert sampled == pytest.approx(computed, rel=5e-3)

    def test_it_is_reproducible(self, rng):
        from pygeoinf2.numerics.quadratic_forms import weighted_chi2_quantile

        weights = rng.uniform(0.5, 4.0, 6)
        first, second = (
            weighted_chi2_quantile(
                weights,
                0.9,
                method="monte_carlo",
                samples=20_000,
                rng=np.random.default_rng(7),
            )
            for _ in range(2)
        )
        assert first == second


class TestCorrelatedMeasures:
    """Several fields on one domain, correlated scale by scale."""

    @staticmethod
    def space():
        pytest.importorskip("pyshtools")
        from pygeoinf2.symmetric_space.sphere import Sobolev

        return Sobolev(10, 2.0, 0.2)

    @pytest.mark.parametrize("correlation", [0.0, 0.7, -0.9])
    def test_the_requested_correlation_appears_in_samples(self, correlation, rng):
        X = self.space()
        first = X.sobolev_symbol(-2.0, 0.2)
        second = X.heat_symbol(0.14)
        measure = X.correlated_measure_from_correlations(
            [first, second],
            np.array([[1.0, correlation], [correlation, 1.0]]),
            labels=("u", "v"),
        )
        draws = [measure.sample(rng=rng) for _ in range(3000)]
        u = np.array([X.to_components(draw[0]) for draw in draws])
        v = np.array([X.to_components(draw[1]) for draw in draws])
        empirical = np.corrcoef(u[:, 5], v[:, 5])[0, 1]
        assert empirical == pytest.approx(correlation, abs=0.06)

    def test_the_marginals_are_the_invariant_measures(self, rng):
        X = self.space()
        first = X.sobolev_symbol(-2.0, 0.2)
        measure = X.correlated_measure_from_correlations(
            [first, X.heat_symbol(0.14)], np.array([[1.0, 0.5], [0.5, 1.0]])
        )
        block = measure.covariance.matrix(form="components")[: X.dim, : X.dim]
        assert np.allclose(np.diag(block), first)

    def test_sampling_comes_from_the_factor(self, rng):
        """An extended Karhunen-Loeve expansion, without writing one."""
        X = self.space()
        measure = X.correlated_measure_from_correlations(
            [X.heat_symbol(0.14), X.heat_symbol(0.14)],
            np.array([[1.0, 0.8], [0.8, 1.0]]),
        )
        assert measure.can_sample
        draw = measure.sample(rng=rng)
        assert len(draw) == 2

    def test_a_correlation_beyond_one_is_refused(self):
        X = self.space()
        with pytest.raises(ValueError, match="positive semidefinite"):
            X.correlated_measure_from_correlations(
                [X.heat_symbol(0.14), X.heat_symbol(0.14)],
                np.array([[1.0, 1.5], [1.5, 1.0]]),
            )

    def test_a_correlation_may_vary_with_scale(self, rng):
        """The point of the construction, as against one number times two
        marginals."""
        X = self.space()
        varying = np.zeros((X.dim, 2, 2))
        varying[:, 0, 0] = varying[:, 1, 1] = 1.0
        varying[:, 0, 1] = varying[:, 1, 0] = np.where(X.degrees < 3, 0.9, 0.0)
        measure = X.correlated_measure_from_correlations(
            [X.heat_symbol(0.14), X.heat_symbol(0.14)], varying
        )
        components = measure.covariance.matrix(form="components")
        cross = np.diag(components[: X.dim, X.dim :])
        assert np.all(cross[X.degrees < 3] > 0.0)
        assert np.allclose(cross[X.degrees >= 3], 0.0)

    def test_a_non_symmetric_slice_is_refused(self):
        X = self.space()
        sigma = np.zeros((X.dim, 2, 2))
        sigma[:, 0, 0] = sigma[:, 1, 1] = 1.0
        sigma[:, 0, 1] = 0.5
        with pytest.raises(ValueError, match="symmetric"):
            X.correlated_measure(sigma)

    def test_a_wrong_shape_is_refused(self):
        X = self.space()
        with pytest.raises(ValueError, match="Expected shape"):
            X.correlated_measure(np.zeros((3, 2, 2)))


class TestAMeasureOnAFunctionSpaceIsTraceClass:
    """An identity covariance, or a Sobolev measure of order at or below half
    the dimension its eigenvalues grow in, is not a Gaussian measure on a
    function space; the library refuses both, with an escape for a test."""

    def test_the_identity_is_refused_on_a_function_space(self):
        from pygeoinf2.algebra.spaces import EuclideanSpace
        from pygeoinf2.symmetric_space.fourier import Sobolev

        box = Sobolev((8, 8), 2.0, 0.3)
        with pytest.raises(ValueError, match="not trace class"):
            GaussianMeasure.from_standard_deviation(box, 1.0)
        assert (
            GaussianMeasure.from_standard_deviation(box, 1.0, unsafe=True).domain == box
        )
        assert (
            GaussianMeasure.from_standard_deviation(EuclideanSpace(3), 1.0).domain.dim
            == 3
        )

    def test_the_order_must_exceed_half_the_dimension(self):
        from pygeoinf2.ball import Sobolev as Ball
        from pygeoinf2.disk import Sobolev as Disk
        from pygeoinf2.interval.radial import Radial
        from pygeoinf2.symmetric_space.fourier import Sobolev as Box

        box = Box((8, 8), 2.0, 0.3)
        with pytest.raises(ValueError, match="trace class"):
            box.sobolev_measure(1.0, 0.3)
        assert box.sobolev_measure(1.0, 0.3, unsafe=True).domain == box
        assert box.sobolev_measure(1.5, 0.3).domain == box
        disk = Disk(4, 2.0, 0.3)
        with pytest.raises(ValueError, match="trace class"):
            disk.sobolev_measure(1.0)
        assert disk.sobolev_measure(1.5).domain == disk
        ball = Ball(3, 2.0, 0.3)
        with pytest.raises(ValueError, match="trace class"):
            ball.sobolev_measure(1.5)
        assert ball.sobolev_measure(2.0).domain == ball
        # A radial profile's eigenvalues grow as on a line, whatever the
        # volume measure it carries.
        radial = Radial(8, radius=1.0, inner_radius=0.5, order=2.0, length_scale=0.3)
        assert radial.sobolev_measure(1.0).domain == radial
        with pytest.raises(ValueError, match="trace class"):
            radial.sobolev_measure(0.5)


def _identity_routes():
    """Every route to a multiple of the identity covariance on a function
    space, each a callable of ``unsafe``."""
    from pygeoinf2.algebra.diagonal import DiagonalLinearOperator
    from pygeoinf2.interval import Sobolev as Interval
    from pygeoinf2.symmetric_space import circle, sphere

    X = circle.Sobolev(16, 1.0, 0.3)
    S = sphere.Sobolev(8, 2.0, 0.3)
    line = Interval(12, 1.0, 0.2)
    identity = LinearOperator.identity(X)
    constant = DiagonalLinearOperator(X, np.full(X.dim, 0.5))
    slices = np.broadcast_to(np.diag([1.0, 2.0]), (X.dim, 2, 2)).copy()
    return {
        "from_standard_deviation": lambda unsafe: (
            GaussianMeasure.from_standard_deviation(X, 1.0, unsafe=unsafe)
        ),
        "from_standard_deviations": lambda unsafe: (
            GaussianMeasure.from_standard_deviations(
                X, np.full(X.dim, 2.0), unsafe=unsafe
            )
        ),
        "covariance": lambda unsafe: GaussianMeasure(
            X, covariance=identity, unsafe=unsafe
        ),
        "scaled covariance": lambda unsafe: GaussianMeasure(
            X, covariance=3.0 * identity, unsafe=unsafe
        ),
        "covariance factor": lambda unsafe: GaussianMeasure(
            X, covariance_factor=constant, unsafe=unsafe
        ),
        "precision": lambda unsafe: GaussianMeasure(
            X, precision=identity, unsafe=unsafe
        ),
        "precision factor": lambda unsafe: GaussianMeasure(
            X, precision_factor=constant, unsafe=unsafe
        ),
        "component matrix": lambda unsafe: GaussianMeasure.from_covariance_matrix(
            X, np.eye(X.dim), form="components", unsafe=unsafe
        ),
        "galerkin matrix": lambda unsafe: GaussianMeasure.from_covariance_matrix(
            X, 2.0 * X.gram_matrix(), form="galerkin", unsafe=unsafe
        ),
        "invariant": lambda unsafe: X.invariant_measure(
            np.ones(X.dim), pointwise_std=0.8, unsafe=unsafe
        ),
        "heat at zero": lambda unsafe: X.heat_measure(0.0, unsafe=unsafe),
        "correlated": lambda unsafe: X.correlated_measure(slices, unsafe=unsafe),
        "correlations": lambda unsafe: X.correlated_measure_from_correlations(
            [np.ones(X.dim), X.heat_symbol(0.3)], np.eye(2), unsafe=unsafe
        ),
        "flat power": lambda unsafe: S.power_measure(
            lambda degree: np.ones_like(degree), unsafe=unsafe
        ),
        "white power": lambda unsafe: S.power_measure(
            lambda degree: 2.0 * degree + 1.0, unsafe=unsafe
        ),
        "spectral": lambda unsafe: line.spectral_measure(
            np.ones(line.dim), unsafe=unsafe
        ),
        "sobolev at order zero": lambda unsafe: X.sobolev_measure(
            0.0, 0.3, unsafe=unsafe
        ),
    }


class TestTheIdentityIsRefusedByEveryRoute:
    """The literal identity, in whatever form it reaches the library, is
    refused on a function space without ``unsafe=True``, with one exception
    class and one message; past that, trace class is the caller's to ensure."""

    @pytest.mark.parametrize("route", list(_identity_routes()))
    def test_refused_and_built_with_unsafe(self, route):
        build = _identity_routes()[route]
        with pytest.raises(ValueError, match="trace class"):
            build(False)
        assert build(True) is not None

    def test_a_decaying_spectrum_on_each_route_is_accepted(self):
        from pygeoinf2.interval import Sobolev as Interval
        from pygeoinf2.symmetric_space import circle, sphere

        X = circle.Sobolev(16, 1.0, 0.3)
        S = sphere.Sobolev(8, 2.0, 0.3)
        line = Interval(12, 1.0, 0.2)
        decaying = X.sobolev_symbol(-1.0, 0.3)
        GaussianMeasure.from_standard_deviations(X, np.sqrt(decaying))
        X.invariant_measure(decaying, pointwise_std=0.8)
        X.heat_measure(0.1)
        X.correlated_measure_from_correlations(
            [decaying, X.heat_symbol(0.3)], np.eye(2)
        )
        S.power_measure(lambda degree: (1.0 + degree) ** -3.0)
        line.spectral_measure(line.eigenvalues**-1.0)
        GaussianMeasure.from_covariance_matrix(X, np.diag(decaying), form="components")

    def test_a_finite_dimensional_space_is_untouched(self):
        E = EuclideanSpace(4)
        GaussianMeasure.from_standard_deviation(E, 1.0)
        GaussianMeasure.from_standard_deviations(E, np.ones(4))
        GaussianMeasure(E, covariance=LinearOperator.identity(E))
        GaussianMeasure.from_covariance_matrix(E, np.eye(4), form="components")

    def test_the_algebra_on_an_unsafe_measure_is_not_judged_again(self, rng):
        from pygeoinf2.symmetric_space import circle

        X = circle.Sobolev(16, 1.0, 0.3)
        white = GaussianMeasure.from_standard_deviation(X, 1.0, unsafe=True)
        for derived in (
            white.translate(X.random(rng=rng)),
            2.0 * white,
            white + white,
            GaussianMeasure.from_product([white, white]).marginal(0),
            white.with_dense_covariance(),
        ):
            assert derived.can_sample
