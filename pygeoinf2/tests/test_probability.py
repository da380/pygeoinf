"""Measures: sampling, moments, pushforward, and the white-noise correction."""

import warnings

import numpy as np
import pytest

from pygeoinf2.algebra.operators import AffineOperator, LinearOperator, Operator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.probability import (
    GaussianMeasure,
    PushForwardMeasure,
)
from pygeoinf2.testing import check_measure, check_traits
from pygeoinf2.traits import Traits

from .conftest import make_dense_metric_space, make_weighted_space
from .doubles import OpaqueSpace, StrictSpace

SAMPLES = 30000


def spd(rng, n):
    root = rng.normal(size=(n, n))
    return root @ root.T + n * np.identity(n)


class TestConstruction:
    def test_a_factor_gives_a_covariance_with_traits_for_free(self, rng):
        """L L* is recognized as positive semidefinite by the palindrome rule."""
        X, Y = EuclideanSpace(3), EuclideanSpace(5)
        factor = LinearOperator.from_matrix(
            X, Y, rng.normal(size=(5, 3)), form="components"
        )
        mu = GaussianMeasure(Y, covariance_factor=factor)
        assert Traits.SELF_ADJOINT & mu.covariance.traits
        assert Traits.POSITIVE_SEMIDEFINITE & mu.covariance.traits

    def test_an_invertible_factor_gives_definiteness(self, rng):
        X = make_weighted_space()
        mu = GaussianMeasure.from_standard_deviation(X, 2.0)
        assert Traits.POSITIVE_DEFINITE & mu.covariance.traits

    def test_an_unstructured_covariance_is_refused(self, rng):
        X = EuclideanSpace(4)
        unstructured = LinearOperator.from_matrix(X, X, spd(rng, 4), form="components")
        with pytest.raises(ValueError, match="must claim"):
            GaussianMeasure(X, covariance=unstructured)

    def test_the_message_points_at_the_remedy(self, rng):
        X = EuclideanSpace(4)
        unstructured = LinearOperator.from_matrix(X, X, spd(rng, 4), form="components")
        with pytest.raises(ValueError, match="check_traits"):
            GaussianMeasure(X, covariance=unstructured)

    def test_something_must_be_supplied(self):
        with pytest.raises(ValueError, match="needs a covariance"):
            GaussianMeasure(EuclideanSpace(3))

    def test_a_covariance_on_the_wrong_space_is_refused(self, rng):
        X, Y = EuclideanSpace(3), EuclideanSpace(4)
        wrong = LinearOperator.from_matrix(
            Y, Y, spd(rng, 4), traits=Traits.POSITIVE_SEMIDEFINITE, form="components"
        )
        with pytest.raises(ValueError, match="operator on"):
            GaussianMeasure(X, covariance=wrong)

    @pytest.mark.parametrize(
        "build",
        [lambda: EuclideanSpace(3), make_weighted_space, make_dense_metric_space],
    )
    def test_a_component_covariance_matrix_round_trips(self, build, rng):
        """The Galerkin matrix of a component-form covariance is ``G C_c``.

        Only a non-diagonal Gram tells this apart from ``C_c G``: on a
        diagonal metric the two agree, which is why the transposed form here
        went unnoticed. The input must be a genuine component matrix, so it is
        built as ``G^-1 S`` from a symmetric ``S``.
        """
        X = build()
        galerkin = spd(rng, X.dim)
        components = np.column_stack([X.solve_gram(c) for c in galerkin.T])

        mu = GaussianMeasure.from_covariance_matrix(X, components, form="components")

        assert mu.covariance.matrix(form="components") == pytest.approx(components)
        assert mu.covariance.matrix(form="galerkin") == pytest.approx(galerkin)

    @pytest.mark.parametrize(
        "build",
        [lambda: EuclideanSpace(3), make_weighted_space, make_dense_metric_space],
    )
    def test_the_normalizing_constant_uses_the_component_determinant(self, build, rng):
        """``det C_c``, not ``det(G C_c)``: the metric's own determinant is not
        part of the measure, and the density is with respect to the space's
        volume measure."""
        X = build()
        galerkin = spd(rng, X.dim)
        components = np.column_stack([X.solve_gram(c) for c in galerkin.T])
        mu = GaussianMeasure.from_covariance_matrix(X, components, form="components")

        _, logdet = np.linalg.slogdet(components)
        expected = -0.5 * X.dim * np.log(2.0 * np.pi) - 0.5 * logdet
        assert mu.log_normalizing_constant() == pytest.approx(expected)

    @pytest.mark.parametrize(
        "build",
        [lambda: EuclideanSpace(3), make_weighted_space, make_dense_metric_space],
    )
    def test_a_covariance_matrix_brings_its_precision(self, build, rng):
        """The inverse factor comes with the covariance matrix; without it
        every measure built this way would have no density.

        The closed form carries the metric twice: the precision's component
        matrix is ``C_c^-1``, so the factor's is ``R^-1 G`` and not ``R^-1``.
        """
        X = build()
        galerkin = spd(rng, X.dim)
        mu = GaussianMeasure.from_covariance_matrix(X, galerkin, form="galerkin")
        components = np.column_stack([X.solve_gram(c) for c in galerkin.T])

        assert mu.precision is not None
        assert mu.precision.matrix(form="components") == pytest.approx(
            np.linalg.inv(components)
        )
        x = X.random(rng=rng)
        coordinates = X.to_components(x)
        expected = (
            coordinates @ X.gram_matrix() @ np.linalg.inv(components) @ coordinates
        )
        assert mu.mahalanobis_squared(x) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "build",
        [lambda: EuclideanSpace(4), make_weighted_space, make_dense_metric_space],
    )
    def test_a_singular_covariance_is_accepted(self, build, rng):
        """A covariance is positive *semi*definite, and the Cholesky route
        refuses every degenerate one; an eigendecomposition accepts them, and
        the measure is still samplable, and still has no
        density, which is the truth about a degenerate Gaussian."""
        X = build()
        rotation, _ = np.linalg.qr(rng.standard_normal((X.dim, X.dim)))
        spectrum = np.zeros(X.dim)
        spectrum[: X.dim - 2] = np.arange(1, X.dim - 1, dtype=float)
        galerkin = rotation @ np.diag(spectrum) @ rotation.T

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            mu = GaussianMeasure.from_covariance_matrix(X, galerkin, form="galerkin")

        assert mu.covariance.matrix(form="galerkin") == pytest.approx(
            galerkin, abs=1e-12
        )
        assert mu.can_sample
        assert mu.precision is None
        with pytest.raises(NotImplementedError, match="no precision"):
            mu.mahalanobis_squared(X.random(rng=rng))

    def test_a_slightly_negative_eigenvalue_is_clipped_with_a_warning(self, rng):
        """What a semidefinite matrix assembled in floating point looks like."""
        X = make_dense_metric_space(5)
        rotation, _ = np.linalg.qr(rng.standard_normal((5, 5)))
        spectrum = np.array([3.0, 2.0, 1.0, -1e-17, -2e-17])
        galerkin = rotation @ np.diag(spectrum) @ rotation.T

        with pytest.warns(UserWarning, match="small negative eigenvalues"):
            mu = GaussianMeasure.from_covariance_matrix(X, galerkin, form="galerkin")
        assert mu.covariance.matrix(form="galerkin") == pytest.approx(
            galerkin, abs=1e-12
        )

    def test_a_genuinely_indefinite_matrix_is_still_refused(self, rng):
        X = make_weighted_space()
        rotation, _ = np.linalg.qr(rng.standard_normal((X.dim, X.dim)))
        spectrum = np.ones(X.dim)
        spectrum[-1] = -1.0
        galerkin = rotation @ np.diag(spectrum) @ rotation.T

        with pytest.raises(ValueError, match="not positive"):
            GaussianMeasure.from_covariance_matrix(X, galerkin, form="galerkin")

    def test_the_singular_sample_stays_in_the_range(self, rng):
        """The check that the accepted factor is the right one: every draw
        lies in the covariance's range, and the sample covariance of the
        components is ``G^-1 C_gal G^-1``."""
        X = make_dense_metric_space(4)
        gram = X.gram_matrix()
        rotation, _ = np.linalg.qr(rng.standard_normal((4, 4)))
        galerkin = rotation @ np.diag([4.0, 2.0, 0.0, 0.0]) @ rotation.T

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            mu = GaussianMeasure.from_covariance_matrix(X, galerkin, form="galerkin")

        draws = np.array([X.to_components(mu.sample(rng=rng)) for _ in range(4000)])
        inverse = np.linalg.inv(gram)
        expected = inverse @ galerkin @ inverse
        # Six standard errors of a covariance entry at this many draws: the
        # factor's eigenvector signs differ between LAPACK builds, so one
        # seed is a different realization on another platform.
        tolerance = 6.0 * np.sqrt(2.0 / len(draws)) * np.abs(expected).max()
        assert np.cov(draws.T) == pytest.approx(expected, abs=tolerance)
        # every draw is in the range: the two null directions carry nothing.
        # The residue is the square root of eigh's own error on a zero
        # eigenvalue, so 1e-16 in the spectrum is 1e-8 in the factor.
        null = rotation[:, 2:]
        assert np.max(np.abs(draws @ gram @ null)) < 1e-6

    def test_the_normalizing_constant_is_exact_for_a_diagonal_covariance(self):
        """The diagonal route is taken before any retraiting, so it stays exact."""
        X = make_weighted_space()
        mu = GaussianMeasure.from_standard_deviation(X, 2.0)
        expected = -0.5 * X.dim * np.log(2.0 * np.pi) - X.dim * np.log(2.0)
        assert mu.log_normalizing_constant() == pytest.approx(expected)


class TestMomentsMatchSamples:
    @pytest.mark.parametrize(
        "build",
        [lambda: EuclideanSpace(4), make_weighted_space, make_dense_metric_space],
    )
    def test_isotropic(self, build, rng):
        mu = GaussianMeasure.from_standard_deviation(build(), 1.5)
        check_measure(mu, rng=rng, samples=SAMPLES)

    def test_with_a_nontrivial_factor(self, rng):
        X = make_weighted_space()
        E = EuclideanSpace(X.dim)
        factor = LinearOperator.from_matrix(
            E, X, rng.normal(size=(X.dim, X.dim)), form="components"
        )
        mu = GaussianMeasure(X, covariance_factor=factor)
        check_measure(mu, rng=rng, samples=SAMPLES)

    def test_with_a_nonzero_mean(self, rng):
        X = make_weighted_space()
        mean = X.random(rng=rng)
        mu = GaussianMeasure.from_standard_deviation(X, 1.2, expectation=mean)
        check_measure(mu, rng=rng, samples=SAMPLES)

    def test_from_a_covariance_matrix(self, rng):
        X = make_weighted_space()
        mu = GaussianMeasure.from_covariance_matrix(X, spd(rng, X.dim), form="galerkin")
        check_measure(mu, rng=rng, samples=SAMPLES)

    def test_from_samples(self, rng):
        """The empirical construction is coordinate-free."""
        X = make_weighted_space()
        source = GaussianMeasure.from_standard_deviation(X, 2.0)
        draws = source.samples(4000, rng=rng)
        mu = GaussianMeasure.from_samples(X, draws)

        assert X.norm(X.subtract(mu.expectation, X.mean(draws))) < 1e-10
        for i in range(X.dim):
            u = X.basis_vector(i)
            empirical = np.mean(
                [X.inner_product(X.subtract(x, mu.expectation), u) ** 2 for x in draws]
            )
            assert X.inner_product(mu.covariance(u), u) == pytest.approx(
                empirical, rel=1e-3
            )

    def test_from_samples_needs_two(self, rng):
        X = EuclideanSpace(3)
        with pytest.raises(ValueError, match="two samples"):
            GaussianMeasure.from_samples(X, [X.random(rng=rng)])


class TestWhiteNoiseCorrection:
    """White noise drawn in the space's own inner product, at the point where
    it actually bites: sampling."""

    def test_isotropic_covariance_is_the_identity_on_the_space(self, rng):
        """sigma^2 I on the space, not sigma^2 G in components."""
        X = make_weighted_space()
        sigma = 1.5
        mu = GaussianMeasure.from_standard_deviation(X, sigma)
        draws = mu.samples(SAMPLES, rng=rng)

        for i in range(X.dim):
            u = X.basis_vector(i)
            empirical = np.mean([X.inner_product(x, u) ** 2 for x in draws])
            # Identity covariance means E[(X,u)^2] == sigma^2 (u,u) ...
            assert empirical == pytest.approx(
                sigma**2 * X.inner_product(u, u), rel=0.08
            )
            # ... and NOT sigma^2 (u,u)^2, which is what a component-space draw
            # would give. On this space the two differ by the metric value.
            wrong = sigma**2 * X.inner_product(u, u) ** 2
            if not np.isclose(X.inner_product(u, u), 1.0):
                assert not np.isclose(empirical, wrong, rtol=0.2)

    def test_the_check_catches_a_component_space_draw(self, rng):
        X = make_weighted_space()

        class ComponentNoiseMeasure(GaussianMeasure):
            def sample(self, *, rng=None):
                rng = np.random.default_rng() if rng is None else rng
                return X.from_components(rng.standard_normal(X.dim))

        mu = ComponentNoiseMeasure(X, covariance=LinearOperator.identity(X))
        with pytest.raises(AssertionError, match="sample covariance"):
            check_measure(mu, rng=rng, samples=8000)


class TestPushForward:
    def test_a_linear_map_keeps_it_gaussian(self, rng):
        X, Y = make_weighted_space(), EuclideanSpace(3)
        A = LinearOperator.from_matrix(
            X, Y, rng.normal(size=(3, X.dim)), form="components"
        )
        mu = GaussianMeasure.from_standard_deviation(X, 1.3)
        nu = A @ mu
        assert isinstance(nu, GaussianMeasure)
        check_measure(nu, rng=rng, samples=SAMPLES)

    def test_the_pushforward_covariance_is_recognized_as_semidefinite(self, rng):
        """A C A*, with nothing asserted."""
        X, Y = make_weighted_space(), EuclideanSpace(3)
        A = LinearOperator.from_matrix(
            X, Y, rng.normal(size=(3, X.dim)), form="components"
        )
        nu = A @ GaussianMeasure.from_standard_deviation(X, 1.3)
        assert Traits.POSITIVE_SEMIDEFINITE & nu.covariance.traits
        check_traits(nu.covariance, rng=rng)

    def test_an_affine_map_shifts_the_mean(self, rng):
        X, Y = EuclideanSpace(4), EuclideanSpace(3)
        A = LinearOperator.from_matrix(X, Y, rng.normal(size=(3, 4)), form="components")
        b = Y.random(rng=rng)
        mu = GaussianMeasure.from_standard_deviation(X, 1.0)
        nu = AffineOperator(A, b) @ mu
        assert isinstance(nu, GaussianMeasure)
        assert np.allclose(nu.expectation, b)
        check_measure(nu, rng=rng, samples=SAMPLES)

    def test_a_nonlinear_map_is_still_samplable(self, rng):
        """No closed density, but the samples are what nonlinear work needs."""
        X, Y = EuclideanSpace(3), EuclideanSpace(2)

        def value(x):
            return np.array([float(x @ x), float(x[0])])

        F = Operator.from_callables(X, Y, value)
        nu = F @ GaussianMeasure.from_standard_deviation(X, 1.0)

        assert isinstance(nu, PushForwardMeasure)
        assert not nu.has_covariance and not nu.has_expectation
        draws = nu.samples(2000, rng=rng)
        # E[|x|^2] == 3 for a standard normal on R^3.
        assert np.mean([d[0] for d in draws]) == pytest.approx(3.0, rel=0.1)

    def test_domain_mismatch_is_refused(self, rng):
        X, Y = EuclideanSpace(4), EuclideanSpace(3)
        A = LinearOperator.from_matrix(Y, Y, np.identity(3), form="components")
        mu = GaussianMeasure.from_standard_deviation(X, 1.0)
        with pytest.raises(ValueError, match="domain"):
            A @ mu


class TestAlgebra:
    def test_independent_sum(self, rng):
        X = make_weighted_space()
        mu = GaussianMeasure.from_standard_deviation(X, 1.0)
        nu = GaussianMeasure.from_standard_deviation(X, 2.0)
        total = mu + nu
        assert isinstance(total, GaussianMeasure)
        # Variances add: 1 + 4 == 5.
        u = X.basis_vector(0)
        assert X.inner_product(total.covariance(u), u) == pytest.approx(
            5.0 * X.inner_product(u, u)
        )
        check_measure(total, rng=rng, samples=SAMPLES)

    def test_scaling_squares_the_covariance(self, rng):
        X = make_weighted_space()
        mu = GaussianMeasure.from_standard_deviation(X, 1.0)
        scaled = 3.0 * mu
        u = X.basis_vector(1)
        assert X.inner_product(scaled.covariance(u), u) == pytest.approx(
            9.0 * X.inner_product(u, u)
        )
        check_measure(scaled, rng=rng, samples=SAMPLES)

    def test_translation(self, rng):
        X = make_weighted_space()
        shift = X.random(rng=rng)
        mu = GaussianMeasure.from_standard_deviation(X, 1.0).translate(shift)
        assert np.allclose(X.to_components(mu.expectation), X.to_components(shift))

    def test_a_subclass_stays_in_its_class(self, rng):
        """The specialization protocol, for measures."""
        X = EuclideanSpace(4)

        class Tagged(GaussianMeasure):
            def _rebuild(self, domain, **kwargs):
                return Tagged(domain, **kwargs)

        mu = Tagged(X, covariance_factor=2.0 * LinearOperator.identity(X))
        assert isinstance(3.0 * mu, Tagged)
        assert isinstance(mu + mu, Tagged)
        A = LinearOperator.from_matrix(X, X, np.identity(4), form="components")
        assert isinstance(A @ mu, Tagged)


class TestDensities:
    def test_log_density_and_gradient(self, rng):
        X = make_weighted_space()
        mu = GaussianMeasure.from_standard_deviation(X, 2.0)
        assert mu.has_log_density and mu.has_grad_log_density

        x = X.random(rng=rng)
        # log p(x) == -|x|^2 / (2 sigma^2), up to a constant.
        assert mu.log_density(x) == pytest.approx(-0.5 * X.squared_norm(x) / 4.0)
        # The gradient is a VECTOR, and equals -x/sigma^2.
        assert np.allclose(
            X.to_components(mu.grad_log_density(x)), X.to_components(X.scale(-0.25, x))
        )

    def test_the_gradient_really_is_a_gradient(self, rng):
        """Finite-difference the log density along a direction."""
        X = make_weighted_space()
        mu = GaussianMeasure.from_standard_deviation(
            X, 1.7, expectation=X.random(rng=rng)
        )
        x = X.random(rng=rng)
        gradient = mu.grad_log_density(x)

        step = 1e-6
        for _ in range(3):
            d = X.random(rng=rng)
            forward = mu.log_density(X.axpy(step, d, X.copy(x)))
            backward = mu.log_density(X.axpy(-step, d, X.copy(x)))
            numerical = (forward - backward) / (2.0 * step)
            assert X.inner_product(gradient, d) == pytest.approx(numerical, rel=1e-5)

    def test_absent_precision_is_reported_clearly(self, rng):
        X = EuclideanSpace(4)
        mu = GaussianMeasure(
            X,
            covariance=LinearOperator.from_matrix(
                X,
                X,
                spd(rng, 4),
                traits=Traits.POSITIVE_SEMIDEFINITE,
                form="components",
            ),
        )
        assert not mu.has_log_density
        with pytest.raises(NotImplementedError, match="no precision"):
            mu.log_density(X.random(rng=rng))


class TestCoordinateFree:
    def test_sampling_and_moments_without_components(self, rng):
        X = OpaqueSpace(np.array([1.0, 4.0, 9.0]))
        mu = GaussianMeasure.from_standard_deviation(X, 1.4)
        check_measure(mu, rng=rng, samples=20000, rtol=0.1)

    def test_from_samples_is_coordinate_free(self, rng):
        """Building the measure touches no component map.

        Note the samples themselves must come from somewhere: white noise on a
        CoordinateSpace is inherently a coordinate operation, since G^-1/2 is a
        statement about a basis. A genuinely coordinate-free space supplies its
        own white_noise, as OpaqueSpace does above.
        """
        base = make_weighted_space()
        strict = StrictSpace(base)
        draws = [strict.random(rng=rng) for _ in range(500)]

        mu = GaussianMeasure.from_samples(strict, draws)
        u = strict.random(rng=rng)
        assert strict.inner_product(mu.covariance(u), u) > 0.0
        total = draws[0]
        for draw in draws[1:]:
            total = strict.add(total, draw)
        mean = strict.scale(1.0 / len(draws), total)
        assert strict.norm(strict.subtract(mu.expectation, mean)) < 1e-10 * strict.norm(
            mean
        )

    def test_sampling_needs_a_factor(self, rng):
        X = EuclideanSpace(4)
        mu = GaussianMeasure(
            X,
            covariance=LinearOperator.from_matrix(
                X,
                X,
                spd(rng, 4),
                traits=Traits.POSITIVE_SEMIDEFINITE,
                form="components",
            ),
        )
        assert not mu.can_sample
        with pytest.raises(NotImplementedError, match="no factor"):
            mu.sample(rng=rng)


class TestReproducibility:
    def test_the_same_seed_gives_the_same_samples(self):
        """Draws come from the generator passed in, never from the global state."""
        X = make_weighted_space()
        mu = GaussianMeasure.from_standard_deviation(X, 1.0)
        first = mu.sample(rng=np.random.default_rng(42))
        second = mu.sample(rng=np.random.default_rng(42))
        assert np.allclose(X.to_components(first), X.to_components(second))


class TestPrecisionSurvivesTheAlgebra:
    """No operation drops the precision, translation included."""

    @pytest.fixture(params=["euclidean", "weighted", "dense"])
    def measure(self, request):
        from pygeoinf2.algebra.diagonal import DiagonalLinearOperator

        values = np.linspace(1.0, 3.0, 4)
        if request.param == "dense":
            # A diagonal of components is not self-adjoint on a dense metric;
            # a diagonal Galerkin matrix is, and its inverse is G D^-1 G.
            space = make_dense_metric_space(4)
            gram = space.gram_matrix()
            traits = Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
            return space, GaussianMeasure(
                space,
                covariance=LinearOperator.from_matrix(
                    space, space, np.diag(values), form="galerkin", traits=traits
                ),
                precision=LinearOperator.from_matrix(
                    space,
                    space,
                    gram @ np.diag(1.0 / values) @ gram,
                    form="galerkin",
                    traits=traits,
                ),
            )
        space = (
            EuclideanSpace(4) if request.param == "euclidean" else make_weighted_space()
        )
        return space, GaussianMeasure(
            space,
            covariance=DiagonalLinearOperator(space, values),
            precision=DiagonalLinearOperator(space, 1.0 / values),
        )

    def operations(self, space, mu, rng):
        from pygeoinf2.algebra.diagonal import DiagonalLinearOperator

        diagonal = DiagonalLinearOperator(space, np.linspace(0.5, 2.0, space.dim))
        return [
            ("scaled", 2.0 * mu),
            ("divided", mu / 2.0),
            ("summed", mu + mu),
            ("translated", mu.translate(space.random(rng=rng))),
            ("mapped", mu.affine_map(diagonal)),
        ]

    def test_the_precision_survives_and_inverts_the_covariance(self, measure, rng):
        """Not merely present: the right operator. ``C -> alpha^2 C`` means
        ``P -> P / alpha^2``, and getting the power wrong would be invisible to
        a test that only asked whether a precision existed."""
        from pygeoinf2.algebra.diagonal import DiagonalLinearOperator

        space, mu = measure
        for name, derived in self.operations(space, mu, rng):
            if name == "summed" and not isinstance(
                mu.covariance, DiagonalLinearOperator
            ):
                # Two diagonal covariances add to a diagonal one whose
                # inverse is closed-form; a sum of general operators has no
                # precision without a solve, and claims none.
                assert derived.precision is None
                continue
            assert derived.precision is not None, name
            probe = space.random(rng=rng)
            residual = space.subtract(
                derived.precision(derived.covariance(probe)), probe
            )
            assert space.norm(residual) < 1e-10 * space.norm(probe), name

    def test_a_translated_measure_still_has_a_density(self, measure, rng):
        """A shift does not touch the covariance, so losing the density under
        one was pure loss — and translation is the commonest of the three."""
        space, mu = measure
        shift = space.random(rng=rng)
        moved = mu.translate(shift)
        point = space.random(rng=rng)

        assert moved.has_log_density
        assert moved.log_density(point) == pytest.approx(
            mu.log_density(space.subtract(point, shift))
        )

    def test_a_sum_of_diagonal_measures_keeps_one_draw(self, measure, rng):
        """``sqrt(a + b)`` is a diagonal factor, so the sum needs one white
        noise draw rather than one per summand. A fact about diagonal
        covariances: a sum of general ones draws once per summand."""
        from pygeoinf2.algebra.diagonal import DiagonalLinearOperator

        space, mu = measure
        if not isinstance(mu.covariance, DiagonalLinearOperator):
            pytest.skip("the one-draw sum is a diagonal fast path")
        total = mu + mu
        assert total.covariance_factor is not None
        check_measure(total, rng=rng, samples=SAMPLES)

    def test_a_product_carries_a_block_precision(self):
        """What the Woodbury data form needs: without it the preconditioner
        inverts Q by conjugate gradients on every application."""
        first = GaussianMeasure.from_standard_deviation(EuclideanSpace(3), 2.0)
        second = GaussianMeasure.from_standard_deviation(make_weighted_space(), 0.5)
        product = GaussianMeasure.from_product([first, second])

        assert product.precision is not None
        probe = product.domain.random()
        residual = product.domain.subtract(
            product.precision(product.covariance(probe)), probe
        )
        assert product.domain.norm(residual) < 1e-10 * product.domain.norm(probe)


class TestStochasticNorms:
    """The ``"stochastic"`` option is a Hutchinson trace; forming the dense
    component matrix and returning the exact answer is the one thing a
    matrix-free estimator is supposed not to do."""

    @pytest.fixture
    def measure(self, rng):
        X = make_dense_metric_space(20)
        matrix = rng.standard_normal((20, 20))
        galerkin = matrix @ matrix.T + 0.5 * np.identity(20)
        components = np.linalg.solve(X.gram_matrix(), galerkin)
        return GaussianMeasure.from_covariance_matrix(
            X, galerkin, form="galerkin"
        ), components

    def test_the_nuclear_norm_is_a_trace_of_the_component_matrix(self, measure, rng):
        """Not of the Galerkin one, which carries an extra factor of ``G``.
        The probes are white noise *on the space*, which is what makes the
        expectation the operator's own trace."""
        mu, components = measure
        exact = np.trace(components)
        estimate = mu.nuclear_norm(method="stochastic", samples=4000, rng=rng)
        assert estimate == pytest.approx(exact, rel=0.05)
        assert mu.nuclear_norm(method="dense") == pytest.approx(exact)

    def test_the_hilbert_schmidt_norm_estimates_the_trace_of_the_square(
        self, measure, rng
    ):
        mu, components = measure
        exact = np.sqrt(np.sum(components * components.T))
        estimate = mu.hilbert_schmidt_norm(method="stochastic", samples=4000, rng=rng)
        assert estimate == pytest.approx(exact, rel=0.05)

    def test_it_does_not_form_the_matrix(self, measure, rng):
        """The point of the option. A covariance that refuses to be assembled
        still has an estimable trace."""
        mu, components = measure
        space = mu.domain
        applications = []

        def apply(x):
            applications.append(x)
            return mu.covariance(x)

        refuses = LinearOperator.self_adjoint(
            space, apply, traits=Traits.POSITIVE_DEFINITE
        )
        opaque = GaussianMeasure(space, covariance=refuses)
        estimate = opaque.nuclear_norm(method="stochastic", samples=200, rng=rng)
        assert estimate == pytest.approx(np.trace(components), rel=0.2)
        assert len(applications) == 200

    def test_the_exact_norms_never_form_the_matrix(self, measure, rng, monkeypatch):
        """The exact route sums a probed diagonal in linear memory and never
        forms the dense component matrix. A covariance that
        refuses to be assembled still has exact norms: N applications for the
        trace and 2N for the trace of the square."""
        mu, components = measure
        space = mu.domain
        applications = []

        def apply(x):
            applications.append(x)
            return mu.covariance(x)

        opaque = GaussianMeasure(
            space,
            covariance=LinearOperator.self_adjoint(
                space, apply, traits=Traits.POSITIVE_DEFINITE
            ),
        )
        monkeypatch.setattr(
            LinearOperator,
            "matrix",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("dense")),
        )
        assert opaque.nuclear_norm() == pytest.approx(np.trace(components))
        assert len(applications) == space.dim
        applications.clear()
        assert opaque.hilbert_schmidt_norm() == pytest.approx(
            np.sqrt(np.sum(components * components.T))
        )
        assert len(applications) == 2 * space.dim

    def test_a_stored_matrix_is_read_not_reassembled(self, measure, monkeypatch):
        """A covariance built from a matrix already holds it; the norms read
        it rather than probing or forming a second copy."""
        mu, components = measure
        monkeypatch.setattr(
            LinearOperator,
            "matrix",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("dense")),
        )
        assert mu.nuclear_norm() == pytest.approx(np.trace(components))
        assert mu.hilbert_schmidt_norm() == pytest.approx(
            np.sqrt(np.sum(components * components.T))
        )

    def test_a_tolerance_stops_when_it_is_met(self, measure, rng):
        mu, components = measure
        estimate = mu.nuclear_norm(method="stochastic", samples=50, rtol=0.02, rng=rng)
        assert estimate == pytest.approx(np.trace(components), rel=0.1)

    def test_an_unknown_method_is_refused(self, measure):
        mu, _ = measure
        with pytest.raises(ValueError, match="Unknown method"):
            mu.nuclear_norm(method="magic")
        with pytest.raises(ValueError, match="Unknown method"):
            mu.hilbert_schmidt_norm(method="magic")


class TestStochasticDivergence:
    """A route that estimated three things and combined them: the trace
    of ``C_o^-1 C_s`` and two log-determinants, each with its own probes. Two
    of those are large and nearly equal whenever the measures are close, so the
    answer was a small difference of large noisy numbers -- ``KL(mu || mu)``
    came back as ``369 +/- 35``, and the other way round as ``-351 +/- 33``.

    It is now one trace of ``g(M) == M - 1 - log M`` with ``M`` similar to
    ``C_o^-1 C_s``, which is non-negative and vanishes at ``M == I``.
    """

    @staticmethod
    def build(space, seed, /, *, shift=0.0):
        matrix = np.random.default_rng(seed).standard_normal((space.dim, space.dim))
        galerkin = matrix @ matrix.T + 0.5 * np.identity(space.dim)
        expectation = None
        if shift:
            expectation = space.from_components(shift * np.ones(space.dim))
        return GaussianMeasure.from_covariance_matrix(
            space, galerkin, form="galerkin", expectation=expectation
        )

    @pytest.fixture(
        params=[
            lambda: EuclideanSpace(8),
            make_weighted_space,
            make_dense_metric_space,
        ],
        ids=["euclidean", "weighted", "dense-metric"],
    )
    def space(self, request):
        return request.param()

    def test_a_measure_from_itself_is_zero_probe_by_probe(self, space):
        """Not zero on average: zero in every probe, because ``M`` is the
        identity and ``g`` has a double zero there. The old route needed the
        cancellation to happen between three separate estimates and it did
        not."""
        mu = self.build(space, 1)
        estimate = mu.kl_divergence_estimate(
            mu, method="stochastic", samples=8, rng=np.random.default_rng(0)
        )
        assert abs(estimate.value) < 1e-8
        assert estimate.standard_error < 1e-8

    def test_it_agrees_with_the_dense_route(self, space):
        mu = self.build(space, 1)
        nu = self.build(space, 2, shift=0.3)
        exact = mu.kl_divergence(nu, method="dense")
        estimate = mu.kl_divergence_estimate(
            nu, method="stochastic", samples=4000, rng=np.random.default_rng(5)
        )
        assert abs(estimate.value - exact) < 4.0 * estimate.standard_error

    def test_it_is_never_negative_and_never_less_certain_than_it_is_large(self, space):
        """A divergence cannot be negative, and this estimator cannot report
        one: every probe returns ``(x, g(M) x) >= 0``. The same
        non-negativity bounds the standard error by the mean, since
        ``sum x_i^2 <= (sum x_i)^2`` for non-negative ``x``."""
        mu = self.build(space, 1)
        for seed in range(3, 9):
            nu = self.build(space, seed)
            estimate = mu.kl_divergence_estimate(
                nu, method="stochastic", samples=4, rng=np.random.default_rng(seed)
            )
            assert estimate.value >= 0.0
            assert estimate.standard_error <= estimate.value + 1e-12

    def test_a_solver_that_does_not_invert_is_refused(self, rng):
        """A Krylov solver stops on the residual, which on an ill-conditioned
        covariance says almost nothing about the error. The old route trusted
        it: CG at ``rtol=1e-8`` on a covariance of condition 4e10 gave a trace
        of 217 where the truth was 289."""
        from pygeoinf2.numerics.solvers import CGSolver

        X = EuclideanSpace(60)
        spectrum = np.geomspace(1.0, 1e-12, 60)
        rotation, _ = np.linalg.qr(rng.standard_normal((60, 60)))
        covariance = LinearOperator.from_matrix(
            X,
            X,
            rotation @ np.diag(spectrum) @ rotation.T,
            form="components",
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
        )
        factor = LinearOperator.from_matrix(
            EuclideanSpace(60),
            X,
            rotation @ np.diag(np.sqrt(spectrum)),
            form="components",
        )
        reference = GaussianMeasure(X, covariance=covariance)
        mine = GaussianMeasure(X, covariance_factor=factor)

        with pytest.raises(ValueError, match="round trip"):
            mine.kl_divergence_estimate(
                reference,
                method="stochastic",
                samples=4,
                solver=CGSolver(rtol=1e-6, strict=False),
                rng=rng,
            )

    def test_an_unearned_definiteness_claim_is_not_made_for_the_caller(self, rng):
        """The old route wrote ``with_traits(POSITIVE_DEFINITE)`` on both
        covariances so that CG and Lanczos would accept them. That is the
        caller's claim to make."""
        X = EuclideanSpace(6)
        matrix = rng.standard_normal((6, 6))
        semidefinite = LinearOperator.from_matrix(
            X,
            X,
            matrix @ matrix.T,
            form="components",
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_SEMIDEFINITE,
        )
        reference = GaussianMeasure(X, covariance=semidefinite)
        mine = self.build(X, 1)

        with pytest.raises(ValueError, match="POSITIVE_DEFINITE"):
            mine.kl_divergence_estimate(reference, method="stochastic", samples=4)

    def test_it_says_so_when_it_has_no_square_root(self, rng):
        """``C_o^-1 C_s`` is not self-adjoint; the estimator needs it written
        around a factor of one side or the other."""
        X = EuclideanSpace(6)
        matrix = rng.standard_normal((6, 6))
        definite = LinearOperator.from_matrix(
            X,
            X,
            matrix @ matrix.T + np.identity(6),
            form="components",
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
        )
        bare = GaussianMeasure(X, covariance=definite)
        reference = GaussianMeasure(
            X,
            covariance=definite,
            precision=LinearOperator.from_matrix(
                X,
                X,
                np.linalg.inv(matrix @ matrix.T + np.identity(6)),
                form="components",
                traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
            ),
        )

        with pytest.raises(ValueError, match="square root"):
            bare.kl_divergence_estimate(reference, method="stochastic", samples=4)


class TestConditioningAppliesTheCovarianceTwice:
    """``(I - C A* N^-1 A) C`` rather than ``C - C A* N^-1 A C``: the same
    operator, one fewer application of ``C`` per action."""

    def test_two_applications_and_the_same_answer(self, rng):
        from pygeoinf2.numerics.solvers import CholeskySolver

        X = make_dense_metric_space(5)
        target = EuclideanSpace(2)
        gram = X.gram_matrix()
        galerkin = spd(rng, 5)
        base = GaussianMeasure.from_covariance_matrix(X, galerkin, form="galerkin")

        tally = []

        def counted(x):
            tally.append(1)
            return base.covariance(x)

        covariance = LinearOperator.self_adjoint(
            X, counted, traits=Traits.POSITIVE_DEFINITE
        )
        measure = GaussianMeasure(X, covariance=covariance)
        operator = LinearOperator.from_matrix(
            X, target, rng.normal(size=(2, 5)), form="galerkin"
        )
        noise = GaussianMeasure.from_standard_deviation(target, 0.3)
        conditioned = measure.condition(
            operator, target.random(rng=rng), noise=noise, solver=CholeskySolver()
        )

        tally.clear()
        conditioned.covariance(X.random(rng=rng))
        assert len(tally) == 2

        components = np.linalg.solve(gram, galerkin)
        forward = operator.matrix(form="components")
        adjoint = np.linalg.solve(gram, forward.T)
        middle = forward @ components @ adjoint + 0.09 * np.identity(2)
        expected = components - components @ adjoint @ np.linalg.solve(
            middle, forward @ components
        )
        assert conditioned.covariance.matrix(form="components") == pytest.approx(
            expected, abs=1e-10
        )


class TestTheAmbientBall:
    """A dense component matrix and a non-symmetric eigendecomposition,
    whatever the covariance, would be cubic, and this is reached on every
    Backus route with a Gaussian error. A randomized spectrum and a sampling
    radius are the alternatives, and a diagonal covariance costs nothing at
    all."""

    def test_a_diagonal_covariance_needs_no_decomposition(self, rng):
        """Exact, and the spectrum is already in hand. Measured at dimension
        8000: 0.48 s of eigenvalues against 16 us."""
        X = EuclideanSpace(60)
        deviations = np.linspace(0.5, 2.0, 60)
        mu = GaussianMeasure.from_standard_deviations(X, deviations)

        from pygeoinf2.numerics.quadratic_forms import weighted_chi2_quantile

        expected = np.sqrt(weighted_chi2_quantile(deviations**2, 0.9))
        assert mu.ambient_ball(level=0.9).radius == pytest.approx(expected)

    def test_every_route_gives_the_same_ball(self, rng):
        """On a dense metric, where the operator's eigenvalues are those of
        ``G^-1 C_gal`` and not of any symmetric matrix in sight."""
        X = make_dense_metric_space(20)
        galerkin = spd(rng, 20)
        mu = GaussianMeasure.from_covariance_matrix(X, galerkin, form="galerkin")

        exact = np.linalg.eigvals(np.linalg.solve(X.gram_matrix(), galerkin)).real
        from pygeoinf2.numerics.quadratic_forms import weighted_chi2_quantile

        expected = np.sqrt(weighted_chi2_quantile(np.clip(exact, 0.0, None), 0.9))

        dense = mu.ambient_ball(level=0.9, method="dense")
        assert dense.radius == pytest.approx(expected, rel=1e-8)
        spectral = mu.ambient_ball(
            level=0.9, method="spectral", rank=20, rng=np.random.default_rng(1)
        )
        assert spectral.radius == pytest.approx(expected, rel=1e-6)
        sampled = mu.ambient_ball(
            level=0.9, method="sampling", samples=40000, rng=np.random.default_rng(2)
        )
        assert sampled.radius == pytest.approx(expected, rel=0.05)

    def test_a_truncated_spectrum_is_a_lower_bound(self, rng):
        """Which is the honest reading of a randomized route: the tail it
        drops is positive, so the ball it gives is too small."""
        X = make_dense_metric_space(20)
        mu = GaussianMeasure.from_covariance_matrix(X, spd(rng, 20), form="galerkin")
        full = mu.ambient_ball(level=0.9, method="dense").radius
        truncated = mu.ambient_ball(
            level=0.9, method="spectral", rank=5, rng=np.random.default_rng(1)
        ).radius
        assert truncated < full

    def test_the_ball_covers_what_it_claims(self, rng):
        X = make_dense_metric_space(12)
        mu = GaussianMeasure.from_covariance_matrix(X, spd(rng, 12), form="galerkin")
        ball = mu.ambient_ball(level=0.9)
        draws = mu.samples(8000, rng=rng)
        assert np.mean([ball.contains(x) for x in draws]) == pytest.approx(
            0.9, abs=0.02
        )

    def test_auto_says_what_it_needs_when_it_cannot_afford_a_route(self):
        """A large space, a covariance that cannot be sampled and no rank: the
        only exact route is cubic and the message says which of the three
        things to supply."""
        from pygeoinf2.algebra.operators import LinearOperator as _Operator

        X = EuclideanSpace(4000)
        covariance = _Operator.self_adjoint(
            X, lambda v: v, traits=Traits.POSITIVE_DEFINITE
        )
        mu = GaussianMeasure(X, covariance=covariance)
        with pytest.raises(ValueError, match="No affordable route"):
            mu.ambient_ball(level=0.9)
        # ...and it is affordable once told how
        assert mu.ambient_ball(level=0.9, method="spectral", rank=8).radius > 0.0

    def test_a_long_spectrum_takes_the_matched_quantile(self):
        """Imhof's inversion sums over every weight at every quadrature point:
        0.8 s at 500 weights, 10.4 s at 2000. A sum of that many comparable
        chi-squares is within 3e-5 of its moment-matched one, in 1 ms."""
        weights = np.linspace(0.25, 4.0, 2000)
        assert GaussianMeasure._quantile_method_for(weights) == "matched"
        # a spectrum carried by a handful of modes is not, and keeps Imhof
        dominated = np.concatenate([[1e6, 1e6], np.full(2000, 1e-8)])
        assert GaussianMeasure._quantile_method_for(dominated) == "imhof"
        assert GaussianMeasure._quantile_method_for(np.ones(50)) == "imhof"


class TestDenseCovariance:
    """``with_dense_covariance(n_jobs=)``. Every pyslfp plotting script calls
    it before drawing a variance
    field: the covariance of a posterior is a graph of operators, and once it
    is going to be applied ``dim`` times it is cheaper assembled."""

    @pytest.fixture
    def opaque(self, rng):
        """A measure whose covariance is a callable: no factor, no precision,
        nothing to read a matrix off."""
        X = make_dense_metric_space(6)
        galerkin = spd(rng, 6)
        components = np.linalg.solve(X.gram_matrix(), galerkin)

        def apply(vector):
            return X.from_components(components @ X.to_components(vector))

        covariance = LinearOperator.self_adjoint(
            X, apply, traits=Traits.POSITIVE_DEFINITE
        )
        mean = X.random(rng=rng)
        return (
            X,
            components,
            GaussianMeasure(X, covariance=covariance, expectation=mean),
        )

    def test_it_assembles_the_same_law(self, opaque, rng):
        X, components, measure = opaque
        assert not measure.can_sample
        assert measure.precision is None

        dense = measure.with_dense_covariance(n_jobs=2)

        assert dense.covariance.matrix(form="components") == pytest.approx(components)
        assert X.norm(X.subtract(dense.expectation, measure.expectation)) < 1e-14

    def test_the_result_can_be_sampled_and_has_a_density(self, opaque, rng):
        """Which the original could do neither of: the factorization that
        assembling makes possible supplies both."""
        X, components, measure = opaque
        dense = measure.with_dense_covariance()

        assert dense.can_sample
        assert dense.precision.matrix(form="components") == pytest.approx(
            np.linalg.inv(components)
        )
        draws = np.array([X.to_components(dense.sample(rng=rng)) for _ in range(20000)])
        inverse = np.linalg.inv(X.gram_matrix())
        expected = components @ inverse
        scale = float(np.max(np.abs(expected)))
        assert np.cov(draws.T) == pytest.approx(expected, abs=0.05 * scale)

    def test_the_assembly_happens_once(self, opaque):
        X, _, measure = opaque
        tally = []
        counting = LinearOperator.self_adjoint(
            X,
            lambda v: (tally.append(1), measure.covariance(v))[1],
            traits=Traits.POSITIVE_DEFINITE,
        )
        wrapped = GaussianMeasure(X, covariance=counting)

        first = wrapped.with_dense_covariance()
        assert len(tally) == X.dim
        assert wrapped.with_dense_covariance(n_jobs=4) is first
        assert len(tally) == X.dim

    def test_a_measure_without_a_covariance_says_so(self):
        from pygeoinf2.algebra.diagonal import DiagonalLinearOperator

        space = EuclideanSpace(4)
        measure = GaussianMeasure(
            space, precision=DiagonalLinearOperator(space, np.ones(4))
        )
        with pytest.raises(ValueError, match="covariance"):
            measure.with_dense_covariance()


class TestPrecisionOnlyMeasures:
    """``covariance is None`` is legal, so it must fail legibly."""

    def test_a_two_point_covariance_needs_the_covariance(self):
        from pygeoinf2.symmetric_space import Sobolev

        space = Sobolev((16,), 2.0, 0.3)
        measure = GaussianMeasure(space, precision=LinearOperator.identity(space))
        with pytest.raises(ValueError, match="two-point covariance needs"):
            measure.two_point_covariance(0.5)

    @pytest.fixture
    def precision_only(self):
        from pygeoinf2.algebra.diagonal import DiagonalLinearOperator

        space = EuclideanSpace(4)
        return space, GaussianMeasure(
            space, precision=DiagonalLinearOperator(space, np.linspace(1.0, 2.0, 4))
        )

    @pytest.mark.parametrize(
        "operation",
        ["nuclear_norm", "hilbert_schmidt_norm", "ambient_ball", "directional"],
    )
    def test_what_needs_a_covariance_says_so(self, precision_only, operation):
        """These raised ``TypeError: unsupported operand type(s) for *:
        'NoneType'`` from several frames down."""
        space, measure = precision_only
        calls = {
            "nuclear_norm": lambda: measure.nuclear_norm(),
            "hilbert_schmidt_norm": lambda: measure.hilbert_schmidt_norm(),
            "ambient_ball": lambda: measure.ambient_ball(),
            "directional": lambda: measure.directional_variance(space.basis_vector(0)),
        }
        with pytest.raises(ValueError, match="covariance"):
            calls[operation]()

    def test_what_needs_only_a_precision_works(self, precision_only, rng):
        """An ellipsoid is defined by a precision, and a density needs one:
        neither has any business asking for the covariance."""
        space, measure = precision_only
        point = space.random(rng=rng)

        assert measure.log_density(point) < 0.0
        region = measure.credible_set(level=0.9)
        assert region.contains(space.zero())

    def test_the_algebra_keeps_it_usable(self, precision_only, rng):
        space, measure = precision_only
        shift = space.random(rng=rng)
        for derived in (2.0 * measure, measure.translate(shift)):
            assert derived.precision is not None
            assert derived.has_log_density


class TestStandardDeviations:
    @pytest.mark.parametrize("build", [lambda: EuclideanSpace(4), make_weighted_space])
    def test_a_deviation_per_direction(self, build, rng):
        space = build()
        deviations = np.array([0.5, 1.0, 2.0, 3.0])
        mu = GaussianMeasure.from_standard_deviations(space, deviations)

        assert mu.can_sample
        assert mu.has_log_density
        for index, deviation in enumerate(deviations):
            direction = space.basis_vector(index)
            # (C u, u) with C == diag(sigma^2) as an operator, so on a
            # weighted space the metric enters once, not twice.
            assert mu.directional_variance(direction) == pytest.approx(
                deviation**2 * space.squared_norm(direction), rel=1e-12
            )
        check_measure(mu, rng=rng, samples=SAMPLES)

    def test_a_non_diagonal_metric_is_refused(self):
        """``diag(sigma)`` is not self-adjoint there, so ``L L*`` is not
        ``diag(sigma^2)``: on a dense metric the construction would give a
        covariance with off-diagonal entries and the wrong variances, and it
        says what to use instead."""
        space = make_dense_metric_space(4)
        with pytest.raises(ValueError, match="from_covariance_matrix"):
            GaussianMeasure.from_standard_deviations(
                space, np.array([1.0, 2.0, 3.0, 4.0])
            )

    def test_the_singular_form_refuses_an_array(self):
        with pytest.raises(ValueError, match="from_standard_deviations"):
            GaussianMeasure.from_standard_deviation(
                EuclideanSpace(3), np.array([1.0, 2.0, 3.0])
            )

    def test_bad_input_is_refused(self):
        space = EuclideanSpace(3)
        with pytest.raises(ValueError, match="Expected 3"):
            GaussianMeasure.from_standard_deviations(space, np.ones(2))
        with pytest.raises(ValueError, match="must be positive"):
            GaussianMeasure.from_standard_deviations(space, np.array([1.0, 0.0, 1.0]))


class TestConditioning:
    """The conditioned measure must be samplable: pyslfp conditions every prior."""

    @pytest.fixture(params=["euclidean", "weighted", "dense"])
    def setting(self, request, rng):
        space = {
            "euclidean": lambda: EuclideanSpace(5),
            "weighted": make_weighted_space,
            "dense": lambda: make_dense_metric_space(5),
        }[request.param]()
        constraint = EuclideanSpace(2)
        operator = LinearOperator.from_matrix(
            space, constraint, rng.normal(size=(2, space.dim)), form="components"
        )
        return space, operator, GaussianMeasure.from_standard_deviation(space, 1.2)

    def test_an_exactly_conditioned_measure_can_be_sampled(self, setting, rng):
        """By the Matheron rule. Without it the conditioned prior could not
        generate synthetic data, and every posterior built on it lost its
        sampler too."""
        space, operator, prior = setting
        posterior = prior.condition(operator, np.zeros(2))

        assert posterior.can_sample
        draws = [posterior.sample(rng=rng) for _ in range(200)]
        for draw in draws:
            assert np.abs(operator(draw)).max() < 1e-8

    @pytest.mark.slow
    def test_the_draws_have_the_stated_moments(self, setting, rng):
        space, operator, prior = setting
        noise = GaussianMeasure.from_standard_deviation(operator.codomain, 0.3)
        for posterior in (
            prior.condition(operator, np.zeros(2)),
            prior.condition(operator, np.array([0.4, -0.2]), noise=noise),
        ):
            check_measure(posterior, rng=rng, samples=60000)

    def test_the_solve_can_be_chosen(self, setting, rng):
        """It was a dense ``np.linalg.inv`` with no way past it."""
        from pygeoinf2.numerics.solvers import CholeskySolver

        space, operator, prior = setting
        direct = prior.condition(operator, np.zeros(2), solver=CholeskySolver())
        default = prior.condition(operator, np.zeros(2))
        probe = space.random(rng=rng)
        assert space.norm(
            space.subtract(direct.covariance(probe), default.covariance(probe))
        ) < 1e-8 * space.norm(probe)


class TestParallelLoops:
    """``n_jobs`` at the loops *around* operators, never inside them."""

    def test_draws_do_not_depend_on_the_job_count(self, rng):
        """One stream per draw is spawned from the parent whether or not the
        loop is parallel, so the same seed gives the same draws at any
        ``n_jobs`` -- and the workers never share a stream."""
        space = make_dense_metric_space(6)
        measure = GaussianMeasure.from_standard_deviation(space, 1.5)

        serial = measure.samples(12, rng=np.random.default_rng(3))
        parallel = measure.samples(12, rng=np.random.default_rng(3), n_jobs=2)
        assert len(parallel) == 12
        for x, y in zip(serial, parallel):
            assert np.allclose(x, y)
        assert not np.allclose(serial[0], serial[1])

    def test_parallel_draws_have_the_right_law(self, rng):
        space = make_weighted_space()
        measure = GaussianMeasure.from_standard_deviation(space, 1.5)
        draws = measure.samples(2000, rng=np.random.default_rng(3), n_jobs=2)
        for index in range(space.dim):
            direction = space.basis_vector(index)
            empirical = np.mean([space.inner_product(x, direction) ** 2 for x in draws])
            assert empirical == pytest.approx(
                1.5**2 * space.inner_product(direction, direction), rel=0.12
            )

    def test_a_parallel_matrix_is_the_serial_one(self, rng):
        space = make_dense_metric_space()
        codomain = EuclideanSpace(4)
        matrix = rng.normal(size=(4, space.dim))
        operator = LinearOperator.from_callables(
            space,
            codomain,
            lambda x: codomain.from_components(matrix @ space.to_components(x)),
            adjoint=lambda y: space.from_components(
                space.solve_gram(
                    matrix.T @ codomain.apply_gram(codomain.to_components(y))
                )
            ),
        )
        for form in ("components", "galerkin"):
            for by in ("columns", "rows"):
                assert operator.matrix(form=form, by=by) == pytest.approx(
                    operator.matrix(form=form, by=by, n_jobs=2)
                )

    def test_a_trace_estimate_does_not_depend_on_the_job_count(self, rng):
        from pygeoinf2.numerics.randomized import random_trace

        space = make_dense_metric_space(5)
        operator = LinearOperator.from_matrix(
            space, space, np.diag([1.0, 2.0, 3.0, 4.0, 5.0]), form="components"
        )
        serial = random_trace(operator, samples=8, rng=np.random.default_rng(2))
        parallel = random_trace(
            operator, samples=8, rng=np.random.default_rng(2), n_jobs=2
        )
        assert serial.value == pytest.approx(parallel.value)
        assert serial.standard_error == pytest.approx(parallel.standard_error)
        adaptive = random_trace(
            operator, samples=8, rtol=0.5, rng=np.random.default_rng(2), n_jobs=2
        )
        assert adaptive.samples >= 8

    def test_the_process_backend_works_on_a_sphere(self, rng):
        """The only safe backend there: the pyshtools transforms crash the
        interpreter when called from two Python threads at once."""
        pytest.importorskip("pyshtools")
        from pygeoinf2.symmetric_space.sphere import Sobolev

        space = Sobolev(10, 2.0, 0.3)
        measure = space.sobolev_measure(2.0, 0.3, pointwise_std=1.5)
        points = space.random_points(6, rng=rng)

        serial = space.pointwise_variance_at(measure, points)
        parallel = space.pointwise_variance_at(measure, points, n_jobs=2)
        assert parallel == pytest.approx(serial)

        draws = measure.samples(4, rng=np.random.default_rng(1), n_jobs=2)
        assert len(draws) == 4

    def test_the_job_count_is_validated(self):
        from pygeoinf2.parallel import resolve_jobs

        assert resolve_jobs(None) == 1
        assert resolve_jobs(1) == 1
        assert resolve_jobs(-1) >= 1
        with pytest.raises(ValueError, match="positive count"):
            resolve_jobs(0)
        with pytest.raises(ValueError, match="positive count"):
            resolve_jobs(-2)

    def test_all_cores_means_the_cores_this_process_may_use(self):
        """``-1`` follows the affinity mask, not the machine: in a scheduler
        allocation ``os.cpu_count`` reports the whole node."""
        import os

        from pygeoinf2.parallel import resolve_jobs

        if not hasattr(os, "sched_setaffinity"):
            pytest.skip("no affinity control on this platform")
        original = os.sched_getaffinity(0)
        if len(original) < 2:
            pytest.skip("one core only")
        try:
            os.sched_setaffinity(0, set(sorted(original)[:1]))
            assert resolve_jobs(-1) == 1
        finally:
            os.sched_setaffinity(0, original)

    def test_workers_get_one_thread_each(self, monkeypatch):
        """joblib's own default hands each worker cores // n_jobs OpenMP and
        BLAS threads, which for this library's transforms is a measured loss;
        the loop caps them at one. An exported variable, or a
        ``parallel_config`` context, wins over that -- so the variables are
        cleared here first, or a runner that exports them tests its own
        environment instead."""
        import os

        from joblib import parallel_config

        from pygeoinf2.parallel import parallel_map

        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)

        def threads(_):
            return os.environ.get("OMP_NUM_THREADS"), os.environ.get(
                "OPENBLAS_NUM_THREADS"
            )

        assert set(parallel_map(threads, range(4), n_jobs=2)) == {("1", "1")}
        with parallel_config(backend="loky", inner_max_num_threads=2):
            assert set(parallel_map(threads, range(4), n_jobs=2)) == {("2", "2")}

    def test_a_nested_loop_runs_serially_in_its_worker(self):
        """joblib would otherwise turn the inner request into threads inside
        the worker -- on a sphere, a crash. So a forwarded ``n_jobs`` can never
        produce threads."""
        import os
        import threading

        from pygeoinf2.parallel import parallel_map

        def inner(_):
            return set(
                parallel_map(
                    lambda j: (os.getpid(), threading.current_thread().name),
                    range(3),
                    n_jobs=3,
                )
            )

        for where in parallel_map(inner, range(2), n_jobs=2):
            assert len(where) == 1
            ((pid, thread),) = where
            assert pid != os.getpid()
            assert thread == "MainThread"

    def test_a_context_chooses_the_backend(self):
        """The loops pass only ``n_jobs``; everything else is joblib's
        ``parallel_config``, which is how a threading backend -- safe for
        NumPy-bound work -- or a cluster is chosen."""
        import threading

        from joblib import parallel_config

        from pygeoinf2.parallel import parallel_map

        with parallel_config(backend="threading"):
            kinds = set(
                parallel_map(
                    lambda i: type(threading.current_thread()).__name__,
                    range(4),
                    n_jobs=2,
                )
            )
        assert kinds == {"DummyProcess"}


class TestAMarginalsFactorMatchesItsCovariance:
    """``marginal`` gives a diagonal block its square root as a covariance
    factor, so that the marginal can still be sampled. On a non-diagonal metric
    that root is not one: ``L`` with component matrix ``diag(r)`` has adjoint
    ``G^-1 diag(r) G``, so ``L L*`` is ``diag(r) G^-1 diag(r) G`` and equals
    ``diag(r^2)`` only when the two commute. The sampler is what would be
    wrong, and nothing would say so.
    """

    @staticmethod
    def build(space, values):
        from pygeoinf2.algebra.diagonal import DiagonalLinearOperator

        return DiagonalLinearOperator(space, values).with_traits(
            Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
        )

    @staticmethod
    def joint(first, second, blocks):
        from pygeoinf2.algebra.direct_sum import BlockLinearOperator, DirectSum
        from pygeoinf2.algebra.operators import LinearOperator as _Operator

        total = DirectSum([first, second])
        zero_upper = _Operator.zero(second, codomain=first)
        zero_lower = _Operator.zero(first, codomain=second)
        covariance = BlockLinearOperator(
            [[blocks[0], zero_upper], [zero_lower, blocks[1]]]
        ).with_traits(Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE)
        return total, GaussianMeasure(total, covariance=covariance)

    def test_on_a_diagonal_metric_the_factor_is_kept(self):
        first, second = make_weighted_space(), make_weighted_space()
        values = np.linspace(1.0, 2.0, first.dim)
        total, measure = self.joint(
            first,
            second,
            [self.build(first, values), self.build(second, values)],
        )
        marginal = measure.marginal(0)

        assert marginal.can_sample
        factor = marginal.covariance_factor
        assert (factor @ factor.adjoint).matrix(form="components") == pytest.approx(
            np.diag(values)
        )

    def test_on_a_dense_metric_it_is_not_offered(self):
        """Because there it is not a factor of anything."""
        first, second = make_dense_metric_space(4), make_dense_metric_space(4)
        values = np.linspace(1.0, 2.0, 4)
        total, measure = self.joint(
            first,
            second,
            [self.build(first, values), self.build(second, values)],
        )
        marginal = measure.marginal(0)

        assert marginal.covariance.matrix(form="components") == pytest.approx(
            np.diag(values)
        )
        assert marginal.covariance_factor is None
        # ...and the reason: the root would not have been one
        from pygeoinf2.algebra.diagonal import DiagonalLinearOperator

        root = DiagonalLinearOperator(first, np.sqrt(values))
        product = (root @ root.adjoint).matrix(form="components")
        assert not np.allclose(product, np.diag(values))


class TestSaddlepoint:
    """The Lugannani-Rice approximation to the weighted chi-square, beside
    Imhof."""

    def test_it_converges_to_the_exact_chi_square(self):
        """On equal weights the error falls like one over the number of
        terms; the public routine handles that case exactly, so this goes
        through the approximation itself."""
        from scipy.stats import chi2

        from pygeoinf2.numerics.quadratic_forms import _saddlepoint

        previous = None
        for n in (4, 16, 64, 256):
            worst = max(
                abs(_saddlepoint(np.ones(n), chi2.ppf(level, n)) - level) / (1 - level)
                for level in (0.5, 0.9, 0.99, 0.999)
            )
            if previous is not None:
                assert worst < previous / 3.0
            previous = worst
        assert previous < 1e-5

    def test_it_agrees_with_imhof_and_is_far_cheaper(self):
        from pygeoinf2.numerics.quadratic_forms import (
            weighted_chi2_cdf,
            weighted_chi2_quantile,
        )

        weights = 1.0 / np.arange(1, 65) ** 2
        for level in (0.9, 0.99, 0.999):
            value = weighted_chi2_quantile(weights, level, method="imhof")
            assert weighted_chi2_cdf(
                weights, value, method="saddlepoint"
            ) == pytest.approx(level, abs=2e-3)
        # And the quantile by the saddlepoint is a root find on it.
        assert weighted_chi2_quantile(
            weights, 0.95, method="saddlepoint"
        ) == pytest.approx(
            weighted_chi2_quantile(weights, 0.95, method="imhof"), rel=5e-3
        )

    def test_the_mean_and_the_extremes_are_handled(self):
        from pygeoinf2.numerics.quadratic_forms import weighted_chi2_cdf

        weights = np.array([3.0, 1.0, 0.5])
        at_mean = weighted_chi2_cdf(weights, float(weights.sum()), method="saddlepoint")
        assert 0.4 < at_mean < 0.7
        assert weighted_chi2_cdf(weights, 1e-9, method="saddlepoint") == pytest.approx(
            0.0, abs=1e-6
        )
        assert weighted_chi2_cdf(weights, 1e3, method="saddlepoint") == pytest.approx(
            1.0, abs=1e-9
        )
        assert weighted_chi2_cdf(weights, 0.0, method="saddlepoint") == 0.0


class TestTheFunctionSpaceMarkerGatesDensities:
    """On a space that discretises functions, a density against the space's
    own volume measure is an artefact of the truncation, so the density
    family refuses without ``unsafe=True``, as point evaluation refuses
    below the order that admits it. On an
    intrinsically finite space nothing changes."""

    @staticmethod
    def function_space():
        from pygeoinf2.symmetric_space import Sobolev

        return Sobolev((16,), 2.0, 0.3)

    def test_the_marker_and_its_propagation(self):
        from pygeoinf2.algebra.diagonal import DiagonalLinearOperator
        from pygeoinf2.algebra.direct_sum import DirectSum
        from pygeoinf2.algebra.spaces import MassWeightedSpace

        space = self.function_space()
        finite = EuclideanSpace(3)
        assert space.is_function_space
        assert not finite.is_function_space
        assert DirectSum([space, finite]).is_function_space
        assert not DirectSum([finite, finite]).is_function_space
        mass = DiagonalLinearOperator(finite, np.array([1.0, 2.0, 3.0]))
        assert not MassWeightedSpace(finite, mass).is_function_space

    @staticmethod
    def _identity_mass(space):
        return LinearOperator.identity(space).with_traits(
            Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
        )

    def test_a_space_without_coordinates_carries_the_marker(self, rng):
        """The marker is on ``HilbertSpace`` itself. It was on
        ``CoordinateSpace``, so every density on a backend's coordinate-free
        space raised ``AttributeError``, and so did a direct sum or a
        mass-weighted space built over one."""
        from pygeoinf2.algebra.direct_sum import DirectSum
        from pygeoinf2.algebra.spaces import MassWeightedSpace

        opaque = OpaqueSpace(np.array([1.0, 2.0, 3.0]))
        assert not opaque.is_function_space
        assert not DirectSum([opaque, EuclideanSpace(2)]).is_function_space
        assert DirectSum([self.function_space(), opaque]).is_function_space
        weighted = MassWeightedSpace(opaque, self._identity_mass(opaque))
        assert not weighted.is_function_space

        identity = self._identity_mass(opaque)
        measure = GaussianMeasure(opaque, covariance=identity, precision=identity)
        x = opaque.random(rng=rng)
        assert measure.log_density(x) == pytest.approx(
            -0.5 * measure.mahalanobis_squared(x)
        )

    def test_a_mass_weighted_space_takes_its_base_marker_without_coordinates(self):
        """The delegation is on ``MassWeightedSpace`` itself, not only on its
        coordinate variant, so a coordinate-free base that discretises
        functions is still one after reweighting."""
        from pygeoinf2.algebra.spaces import MassWeightedSpace

        class OpaqueFunctions(OpaqueSpace):
            @property
            def is_function_space(self) -> bool:
                return True

        base = OpaqueFunctions(np.array([1.0, 2.0]))
        weighted = MassWeightedSpace(base, self._identity_mass(base))
        assert type(weighted) is MassWeightedSpace
        assert weighted.is_function_space

    def test_the_density_family_refuses_and_unsafe_proceeds(self, rng):
        space = self.function_space()
        measure = space.sobolev_measure(2.0, 0.3)
        x = measure.sample(rng=rng)

        for call in (
            lambda **k: measure.log_density(x, **k),
            lambda **k: measure.grad_log_density(x, **k),
            lambda **k: measure.mahalanobis_squared(x, **k),
            lambda **k: measure.log_normalizing_constant(**k),
            lambda **k: measure.credible_set(level=0.9, **k),
            lambda **k: measure.weakened_ellipsoid(level=0.9, power=1.0, **k),
        ):
            with pytest.raises(ValueError, match="discretises a function space"):
                call()
            call(unsafe=True)

    def test_a_weakened_ellipsoid_below_power_one_is_not_a_density(self, rng):
        """``power < 1`` is covariance geometry, not the distribution's own
        credible set, and it stays open -- the brief gates ``power == 1``."""
        space = self.function_space()
        measure = space.sobolev_measure(2.0, 0.3)
        measure.weakened_ellipsoid(level=0.9, power=0.5)

    def test_a_finite_space_is_untouched(self, rng):
        space = make_dense_metric_space(4)
        measure = GaussianMeasure.from_standard_deviation(space, 1.5)
        x = measure.sample(rng=rng)
        assert measure.log_density(x) == pytest.approx(
            -0.5 * measure.mahalanobis_squared(x)
        )
        measure.credible_set(level=0.9)

    def test_the_mixture_is_gated_too(self, rng):
        from pygeoinf2.probability import GaussianMixture

        space = self.function_space()
        components = [
            space.sobolev_measure(2.0, 0.3),
            space.sobolev_measure(2.0, 0.5),
        ]
        mixture = GaussianMixture(components, weights=np.array([0.5, 0.5]))
        x = mixture.sample(rng=rng)
        with pytest.raises(ValueError, match="discretises a function space"):
            mixture.log_density(x)
        with pytest.raises(ValueError, match="discretises a function space"):
            mixture.marginal_probabilities(x)
        probabilities = mixture.marginal_probabilities(x, unsafe=True)
        assert probabilities.sum() == pytest.approx(1.0)


def test_an_identity_covariance_samples_without_a_factor(rng):
    """The identity is its own factor, so a white measure can be drawn from."""
    from pygeoinf2.algebra.operators import LinearOperator
    from pygeoinf2.probability.gaussian import GaussianMeasure

    from .conftest import make_dense_metric_space

    space = make_dense_metric_space(4)
    identity = LinearOperator.identity(space)
    measure = GaussianMeasure(space, covariance=identity, precision=identity)
    assert measure.can_sample
    draw = measure.sample(rng=rng)
    assert space.to_components(draw).shape == (space.dim,)


class TestHardeningsAreGatedOnFunctionSpaces:
    """The credible ellipsoid is a Mahalanobis set, so on a space that
    discretises functions it is refused by name without ``unsafe=True``, as
    the density family is; the ambient ball, a norm bound, is not gated,
    and the weakened ellipsoid is gated only where it is the credible
    set."""

    @staticmethod
    def measure():
        from pygeoinf2.symmetric_space.circle import Sobolev

        space = Sobolev(16, 2.0, 0.2)
        assert space.is_function_space
        return space, GaussianMeasure.from_standard_deviation(space, 1.0)

    def test_the_credible_set_is_refused_without_unsafe(self):
        space, mu = self.measure()
        with pytest.raises(ValueError, match="refused on"):
            mu.credible_set(level=0.9)
        ellipsoid = mu.credible_set(level=0.9, unsafe=True)
        assert ellipsoid.contains(space.zero())

    def test_the_weakened_ellipsoid_is_gated_only_at_power_one(self):
        """At ``power == 1`` it is the credible set and refused like it; at
        a power below one the radius is a quantile that survives
        refinement, and no flag is needed."""
        space, mu = self.measure()
        with pytest.raises(ValueError, match="refused on"):
            mu.weakened_ellipsoid(power=1.0, level=0.9)
        assert mu.weakened_ellipsoid(power=1.0, level=0.9, unsafe=True).contains(
            space.zero()
        )
        assert mu.weakened_ellipsoid(power=0.5, level=0.9).contains(space.zero())

    def test_the_ambient_ball_needs_no_unsafe(self):
        space, mu = self.measure()
        ball = mu.ambient_ball(level=0.9)
        assert ball.contains(space.zero())
