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

    def test_parallel_draws_have_the_right_law(self, rng):
        """Not the same numbers as a serial run at the same seed -- each worker
        gets its own spawned stream, so the draws are independent rather than
        identical. Reproducible, and not a repeat."""
        space = make_weighted_space()
        measure = GaussianMeasure.from_standard_deviation(space, 1.5)

        serial = measure.samples(2000, rng=np.random.default_rng(3))
        parallel = measure.samples(
            2000, rng=np.random.default_rng(3), n_jobs=2, backend="threading"
        )
        assert len(parallel) == 2000
        assert not np.allclose(serial[0], parallel[0])

        for draws in (serial, parallel):
            components = np.array([space.to_components(x) for x in draws])
            for index in range(space.dim):
                direction = space.basis_vector(index)
                empirical = np.mean(
                    [space.inner_product(x, direction) ** 2 for x in draws]
                )
                assert empirical == pytest.approx(
                    1.5**2 * space.inner_product(direction, direction), rel=0.12
                )
            assert components.shape == (2000, space.dim)

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
                    operator.matrix(form=form, by=by, n_jobs=2, backend="threading")
                )

    def test_the_job_count_is_validated(self):
        from pygeoinf2.parallel import resolve_jobs

        assert resolve_jobs(None) == 1
        assert resolve_jobs(1) == 1
        assert resolve_jobs(-1) >= 1
        with pytest.raises(ValueError, match="positive count"):
            resolve_jobs(0)
        with pytest.raises(ValueError, match="positive count"):
            resolve_jobs(-2)

    def test_it_stays_serial_without_joblib(self, monkeypatch):
        """The dependency is optional: with one job nothing is imported, and
        with joblib missing the loop still runs."""
        import builtins

        from pygeoinf2.parallel import parallel_map

        real_import = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name == "joblib":
                raise ImportError("no joblib")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse)
        assert parallel_map(lambda i: i * i, range(4), n_jobs=2) == [0, 1, 4, 9]
