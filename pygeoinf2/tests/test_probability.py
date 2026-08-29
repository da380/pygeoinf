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
        """L L* is recognised as positive semidefinite by the palindrome rule."""
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
    def test_the_normalising_constant_uses_the_component_determinant(self, build, rng):
        """``det C_c``, not ``det(G C_c)``: the metric's own determinant is not
        part of the measure, and the density is with respect to the space's
        volume measure."""
        X = build()
        galerkin = spd(rng, X.dim)
        components = np.column_stack([X.solve_gram(c) for c in galerkin.T])
        mu = GaussianMeasure.from_covariance_matrix(X, components, form="components")

        _, logdet = np.linalg.slogdet(components)
        expected = -0.5 * X.dim * np.log(2.0 * np.pi) - 0.5 * logdet
        assert mu.log_normalising_constant() == pytest.approx(expected)

    @pytest.mark.parametrize(
        "build",
        [lambda: EuclideanSpace(3), make_weighted_space, make_dense_metric_space],
    )
    def test_a_covariance_matrix_brings_its_precision(self, build, rng):
        """v1 attached the inverse factor and v2 had stopped, which left every
        measure built this way without a density.

        The closed form carries the metric twice: the precision's component
        matrix is ``C_c^-1``, so the factor's is ``R^-1 G`` and not ``R^-1``.
        """
        X = build()
        galerkin = spd(rng, X.dim)
        mu = GaussianMeasure.from_covariance_matrix(X, galerkin)
        components = np.column_stack([X.solve_gram(c) for c in galerkin.T])

        assert mu.precision is not None
        assert mu.precision.matrix(form="components") == pytest.approx(
            np.linalg.inv(components)
        )
        x = X.random(rng=rng)
        coordinates = X.to_components(x)
        expected = coordinates @ X.gram_matrix() @ np.linalg.inv(components) @ coordinates
        assert mu.mahalanobis_squared(x) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "build",
        [lambda: EuclideanSpace(4), make_weighted_space, make_dense_metric_space],
    )
    def test_a_singular_covariance_is_accepted(self, build, rng):
        """A covariance is positive *semi*definite, and the Cholesky route
        refused every degenerate one. v1 took an eigendecomposition and so
        accepted them; the measure is still samplable, and still has no
        density, which is the truth about a degenerate Gaussian."""
        X = build()
        rotation, _ = np.linalg.qr(rng.standard_normal((X.dim, X.dim)))
        spectrum = np.zeros(X.dim)
        spectrum[: X.dim - 2] = np.arange(1, X.dim - 1, dtype=float)
        galerkin = rotation @ np.diag(spectrum) @ rotation.T

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            mu = GaussianMeasure.from_covariance_matrix(X, galerkin)

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
            mu = GaussianMeasure.from_covariance_matrix(X, galerkin)
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
            GaussianMeasure.from_covariance_matrix(X, galerkin)

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
            mu = GaussianMeasure.from_covariance_matrix(X, galerkin)

        draws = np.array([X.to_components(mu.sample(rng=rng)) for _ in range(4000)])
        inverse = np.linalg.inv(gram)
        expected = inverse @ galerkin @ inverse
        assert np.cov(draws.T) == pytest.approx(expected, abs=0.2)
        # every draw is in the range: the two null directions carry nothing.
        # The residue is the square root of eigh's own error on a zero
        # eigenvalue, so 1e-16 in the spectrum is 1e-8 in the factor.
        null = rotation[:, 2:]
        assert np.max(np.abs(draws @ gram @ null)) < 1e-6

    def test_the_normalising_constant_is_exact_for_a_diagonal_covariance(self):
        """The diagonal route is taken before any retraiting, so it stays exact."""
        X = make_weighted_space()
        mu = GaussianMeasure.from_standard_deviation(X, 2.0)
        expected = -0.5 * X.dim * np.log(2.0 * np.pi) - X.dim * np.log(2.0)
        assert mu.log_normalising_constant() == pytest.approx(expected)


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
        A = LinearOperator.from_matrix(
            X, Y, rng.normal(size=(3, X.dim)), form="components"
        )
        mu = GaussianMeasure.from_standard_deviation(X, 1.3)
        nu = A @ mu
        assert isinstance(nu, GaussianMeasure)
        check_measure(nu, rng=rng, samples=SAMPLES)

    def test_the_pushforward_covariance_is_recognised_as_semidefinite(self, rng):
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
        """The specialisation protocol, for measures."""
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
        assert strict.norm(mu.expectation) >= 0.0

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
        """v1 draws from the legacy global state, so nothing is reproducible."""
        X = make_weighted_space()
        mu = GaussianMeasure.from_standard_deviation(X, 1.0)
        first = mu.sample(rng=np.random.default_rng(42))
        second = mu.sample(rng=np.random.default_rng(42))
        assert np.allclose(X.to_components(first), X.to_components(second))


class TestPrecisionSurvivesTheAlgebra:
    """Every operation used to drop the precision, including translation."""

    @pytest.fixture(params=["euclidean", "weighted"])
    def measure(self, request):
        from pygeoinf2.algebra.diagonal import DiagonalLinearOperator

        space = (
            EuclideanSpace(4) if request.param == "euclidean" else make_weighted_space()
        )
        values = np.linspace(1.0, 3.0, space.dim)
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
        space, mu = measure
        for name, derived in self.operations(space, mu, rng):
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
        noise draw rather than one per summand."""
        space, mu = measure
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
    """The ``"stochastic"`` option used to form the dense component matrix and
    return the exact answer, which is the one thing a matrix-free estimator is
    supposed not to do. It is now a Hutchinson trace, as v1's was."""

    @pytest.fixture
    def measure(self, rng):
        X = make_dense_metric_space(20)
        matrix = rng.standard_normal((20, 20))
        galerkin = matrix @ matrix.T + 0.5 * np.identity(20)
        components = np.linalg.solve(X.gram_matrix(), galerkin)
        return GaussianMeasure.from_covariance_matrix(X, galerkin), components

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


class TestPrecisionOnlyMeasures:
    """``covariance is None`` is legal, so it must fail legibly."""

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

    @pytest.fixture(params=["euclidean", "weighted"])
    def setting(self, request, rng):
        space = (
            EuclideanSpace(5) if request.param == "euclidean" else make_weighted_space()
        )
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
    """D-6: ``n_jobs`` at the loops *around* operators, never inside them."""

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
        from pygeoinf2.numerics.randomised import random_trace

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

    @pytest.mark.slow
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

    def test_workers_get_one_thread_each(self):
        """joblib's own default hands each worker cores // n_jobs OpenMP and
        BLAS threads, which for this library's transforms is a measured loss;
        the loop caps them at one. An exported variable, or a
        ``parallel_config`` context, wins over that."""
        import os

        from joblib import parallel_config

        from pygeoinf2.parallel import parallel_map

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
            (pid, thread), = where
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
