"""Measures: sampling, moments, pushforward, and the white-noise correction."""

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
        """L L* is recognised as positive semidefinite by the palindrome rule."""
        X, Y = EuclideanSpace(3), EuclideanSpace(5)
        factor = LinearOperator.from_component_matrix(X, Y, rng.normal(size=(5, 3)))
        mu = GaussianMeasure(Y, covariance_factor=factor)
        assert Traits.SELF_ADJOINT & mu.covariance.traits
        assert Traits.POSITIVE_SEMIDEFINITE & mu.covariance.traits

    def test_an_invertible_factor_gives_definiteness(self, rng):
        X = make_weighted_space()
        mu = GaussianMeasure.from_standard_deviation(X, 2.0)
        assert Traits.POSITIVE_DEFINITE & mu.covariance.traits

    def test_an_unstructured_covariance_is_refused(self, rng):
        X = EuclideanSpace(4)
        unstructured = LinearOperator.from_component_matrix(X, X, spd(rng, 4))
        with pytest.raises(ValueError, match="must claim"):
            GaussianMeasure(X, covariance=unstructured)

    def test_the_message_points_at_the_remedy(self, rng):
        X = EuclideanSpace(4)
        unstructured = LinearOperator.from_component_matrix(X, X, spd(rng, 4))
        with pytest.raises(ValueError, match="check_traits"):
            GaussianMeasure(X, covariance=unstructured)

    def test_something_must_be_supplied(self):
        with pytest.raises(ValueError, match="needs a covariance"):
            GaussianMeasure(EuclideanSpace(3))

    def test_a_covariance_on_the_wrong_space_is_refused(self, rng):
        X, Y = EuclideanSpace(3), EuclideanSpace(4)
        wrong = LinearOperator.from_component_matrix(
            Y, Y, spd(rng, 4), traits=Traits.POSITIVE_SEMIDEFINITE
        )
        with pytest.raises(ValueError, match="operator on"):
            GaussianMeasure(X, covariance=wrong)


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
        factor = LinearOperator.from_component_matrix(
            E, X, rng.normal(size=(X.dim, X.dim))
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
        mu = GaussianMeasure.from_covariance_matrix(X, spd(rng, X.dim))
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
    """DESIGN.md section 9, at the point where it actually bites: sampling."""

    def test_isotropic_covariance_is_the_identity_on_the_space(self, rng):
        """v1 produces sigma^2 G here, not sigma^2 I."""
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

        class V1StyleMeasure(GaussianMeasure):
            def sample(self, *, rng=None):
                rng = np.random.default_rng() if rng is None else rng
                return X.from_components(rng.standard_normal(X.dim))

        mu = V1StyleMeasure(X, covariance=LinearOperator.identity(X))
        with pytest.raises(AssertionError, match="sample covariance"):
            check_measure(mu, rng=rng, samples=8000)


class TestPushForward:
    def test_a_linear_map_keeps_it_gaussian(self, rng):
        X, Y = make_weighted_space(), EuclideanSpace(3)
        A = LinearOperator.from_component_matrix(X, Y, rng.normal(size=(3, X.dim)))
        mu = GaussianMeasure.from_standard_deviation(X, 1.3)
        nu = A @ mu
        assert isinstance(nu, GaussianMeasure)
        check_measure(nu, rng=rng, samples=SAMPLES)

    def test_the_pushforward_covariance_is_recognised_as_semidefinite(self, rng):
        """A C A*, with nothing asserted."""
        X, Y = make_weighted_space(), EuclideanSpace(3)
        A = LinearOperator.from_component_matrix(X, Y, rng.normal(size=(3, X.dim)))
        nu = A @ GaussianMeasure.from_standard_deviation(X, 1.3)
        assert Traits.POSITIVE_SEMIDEFINITE & nu.covariance.traits
        check_traits(nu.covariance, rng=rng)

    def test_an_affine_map_shifts_the_mean(self, rng):
        X, Y = EuclideanSpace(4), EuclideanSpace(3)
        A = LinearOperator.from_component_matrix(X, Y, rng.normal(size=(3, 4)))
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
        A = LinearOperator.from_component_matrix(Y, Y, np.identity(3))
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
        """The specialisation protocol, for measures."""
        X = EuclideanSpace(4)

        class Tagged(GaussianMeasure):
            def _rebuild(self, domain, **kwargs):
                return Tagged(domain, **kwargs)

        mu = Tagged(X, covariance_factor=2.0 * LinearOperator.identity(X))
        assert isinstance(3.0 * mu, Tagged)
        assert isinstance(mu + mu, Tagged)
        A = LinearOperator.from_component_matrix(X, X, np.identity(4))
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
            covariance=LinearOperator.from_component_matrix(
                X, X, spd(rng, 4), traits=Traits.POSITIVE_SEMIDEFINITE
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
        assert strict.norm(mu.expectation) >= 0.0

    def test_sampling_needs_a_factor(self, rng):
        X = EuclideanSpace(4)
        mu = GaussianMeasure(
            X,
            covariance=LinearOperator.from_component_matrix(
                X, X, spd(rng, 4), traits=Traits.POSITIVE_SEMIDEFINITE
            ),
        )
        assert not mu.can_sample
        with pytest.raises(NotImplementedError, match="no factor"):
            mu.sample(rng=rng)


class TestReproducibility:
    def test_the_same_seed_gives_the_same_samples(self):
        """v1 draws from the legacy global state, so nothing is reproducible."""
        X = make_weighted_space()
        mu = GaussianMeasure.from_standard_deviation(X, 1.0)
        first = mu.sample(rng=np.random.default_rng(42))
        second = mu.sample(rng=np.random.default_rng(42))
        assert np.allclose(X.to_components(first), X.to_components(second))
