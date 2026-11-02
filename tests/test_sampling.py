"""Function-space samplers, layer by layer, against a posterior known exactly.

A linear problem with a Gaussian prior has a Gaussian posterior in closed
form, so every sampler can be held to it: each layer is checked for what its
construction promises (the reference invariant at any step, a block's energy
the Hamiltonian's, a transition density the proposal's), and each assembled
sampler for the posterior mean within its own Monte Carlo error. Everything
runs on a dense metric, where a component standing in for a vector, or a
representer for a derivative, would show.
"""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

import pygeoinf as gi
from pygeoinf.algebra.operators import EvaluationError, LinearOperator, Operator
from pygeoinf.algebra.spaces import EuclideanSpace
from pygeoinf.geometry import Ball
from pygeoinf.inference import (
    ForwardProblem,
    GaussianReference,
    LinearForwardProblem,
    LinearGaussianInversion,
    MixtureReference,
)
from pygeoinf.inference.estimators import MeasureEstimator
from pygeoinf.inference.sampling import (
    HMC,
    MALA,
    PCN,
    SMC,
    Chains,
    GeometricHMC,
    GeometricMALA,
    Hamiltonian,
    LabelJump,
    MetropolisHastings,
    MetropolizedBlock,
    MomentumRefresh,
    Overdamped,
    PreconditionedHMC,
    PreconditionedOU,
    PriorOU,
    PriorRotation,
    SampledPosterior,
    SplittingScheme,
    State,
    StepSizeAdaptation,
    Target,
    Tempering,
    Underdamped,
    WeightedEnsemble,
)
from pygeoinf.inference.laplace import MaximumAPosteriori
from pygeoinf.inference.sampling.flows import (
    Flow,
    crank_nicolson_delta,
    pcn_beta,
    time_from_crank_nicolson,
    time_from_pcn_beta,
)
from pygeoinf.numerics.chain_diagnostics import integrated_autocorrelation_time
from pygeoinf.symmetric_space import circle
from pygeoinf.testing import check_scheme

from .conftest import make_dense_metric_space


class Setting:
    """A linear problem on a dense metric, its exact posterior, and a probe."""

    def __init__(self):
        rng = np.random.default_rng(3)
        space = make_dense_metric_space(4)
        data_space = EuclideanSpace(3)
        matrix = rng.standard_normal((3, 4))
        self.forward = LinearOperator.from_matrix(space, data_space, matrix, form="components")
        self.prior = gi.GaussianMeasure.from_standard_deviation(space, 1.0)
        self.error = gi.GaussianMeasure.from_standard_deviation(data_space, 0.5)
        self.problem = LinearForwardProblem(self.forward, error=self.error)
        self.data = self.forward(self.prior.sample(rng=rng)) + self.error.sample(rng=rng)
        self.inversion = LinearGaussianInversion(self.problem, self.prior)
        self.posterior = self.inversion(self.data)
        self.space = space
        probe = space.basis_vector(0)
        self.exact = space.inner_product(self.posterior.expectation, probe)
        self.record = LinearOperator.from_callables(
            space,
            EuclideanSpace(1),
            lambda x: np.array([space.inner_product(x, probe)]),
            adjoint=lambda c: c[0] * probe,
        )

    def target(self, reference=None):
        return Target(self.problem, self.data, self.prior if reference is None else reference)

    def components_covariance(self, measure):
        """The covariance in components, ``C_c`` with ``C x == C_c G x``."""
        space = self.space
        columns = np.array([measure.covariance(space.basis_vector(i)) for i in range(4)]).T
        return columns @ np.linalg.inv(space.gram_matrix())


@pytest.fixture(scope="module")
def setting():
    return Setting()


def assert_recovers_the_mean(posterior, exact, *, bound=4.0):
    estimate = posterior.scalar_means()[1]
    assert abs(estimate.value - exact) < bound * estimate.standard_error
    assert posterior.diagnostics().rhat.max() < 1.05


class TestTheTarget:
    def test_its_value_is_the_likelihood_plus_the_reference_density(self, setting, rng):
        reference = setting.inversion.reference(setting.data, rank=2, rng=rng)
        target = setting.target(reference)
        u = setting.prior.sample(rng=rng)
        phi = setting.problem.potential(setting.data)(u)
        assert target.value(State(u)) == pytest.approx(
            phi + reference.relative_log_density(u), rel=1e-12
        )
        assert target.with_temperature(2.0).value(State(u)) == pytest.approx(
            phi / 2.0 + reference.relative_log_density(u), rel=1e-12
        )

    def test_its_gradient_is_the_residuals(self, setting, rng):
        reference = setting.inversion.reference(setting.data, rank=2, rng=rng)
        target = setting.target(reference)
        u = setting.prior.sample(rng=rng)
        assert np.allclose(
            target.gradient(State(u)), target.residual_functional().gradient(u), atol=1e-12
        )
        assert np.allclose(
            target.preconditioned_gradient(State(u)),
            reference.covariance(target.gradient(State(u))),
            atol=1e-12,
        )

    def test_one_evaluation_serves_value_gradient_and_every_temperature(self, setting, rng):
        calls = []
        space, data_space = setting.space, setting.forward.codomain
        matrix = np.array([setting.forward(space.basis_vector(i)) for i in range(4)]).T

        def value(x):
            calls.append("value")
            return matrix @ x

        forward = Operator.from_callables(
            space,
            data_space,
            value,
            derivative=lambda x: LinearOperator.from_matrix(
                space, data_space, matrix, form="components"
            ),
        )
        target = Target(ForwardProblem(forward, error=setting.error), setting.data, setting.prior)
        state = State(setting.prior.sample(rng=rng))
        target.gradient(state)
        target.value(state)
        target.with_temperature(3.0).value(state)
        assert calls == ["value"]

    def test_a_point_outside_the_constraint_is_infinite_unevaluated(self, setting, rng):
        target = Target(
            setting.problem,
            setting.data,
            setting.prior,
            constraint=Ball(setting.space, radius=1e-3),
        )
        state = State(setting.space.from_components(np.ones(4)))
        assert target.value(state) == float("inf")

    def test_a_failed_forward_is_infinite(self, setting, rng):
        def value(x):
            raise EvaluationError("the solver diverged")

        forward = Operator.from_callables(setting.space, setting.forward.codomain, value)
        target = Target(ForwardProblem(forward, error=setting.error), setting.data, setting.prior)
        assert target.value(State(setting.prior.sample(rng=rng))) == float("inf")

    def test_the_reference_alone_has_no_potential(self, setting, rng):
        target = Target.from_reference(setting.prior)
        state = State(setting.prior.sample(rng=rng))
        assert target.value(state) == 0.0
        assert setting.space.norm(target.gradient(state)) == 0.0

    def test_a_reference_on_another_space_is_refused(self, setting):
        with pytest.raises(ValueError, match="model space"):
            Target(setting.problem, setting.data, gi.GaussianMeasure.from_standard_deviation(
                EuclideanSpace(4), 1.0
            ))


class TestTheFlows:
    def test_each_flow_turns_time_into_its_own_angle(self, setting):
        target = setting.target()
        assert PriorRotation(target).angle(0.3) == pytest.approx(0.3)
        assert np.cos(PriorOU(target).angle(0.3)) == pytest.approx(np.exp(-0.3))
        refresh = MomentumRefresh(target, friction=2.0)
        assert np.cos(refresh.angle(0.3)) == pytest.approx(np.exp(-0.6))
        assert MomentumRefresh(target).angle(0.3) == pytest.approx(np.pi / 2)

    def test_a_rotation_is_undone_by_a_flip(self, setting, rng):
        dynamics = Hamiltonian(setting.target())
        state = dynamics.initial_state(setting.prior.sample(rng=rng), rng=rng)
        rotation = dynamics.flows()["A"]
        there, change = rotation.apply(state, 0.7)
        back, _ = rotation.apply(dynamics.flip(there), 0.7)
        assert change == 0.0
        assert np.allclose(back.position, state.position, atol=1e-12)
        assert np.allclose(back.velocity, setting.space.negative(state.velocity), atol=1e-12)

    def test_a_gradient_flow_on_a_target_without_one_is_refused(self, setting):
        forward = Operator.from_callables(
            setting.space, setting.forward.codomain, lambda x: setting.forward(x)
        )
        target = Target(ForwardProblem(forward, error=setting.error), setting.data, setting.prior)
        PreconditionedOU(target)
        with pytest.raises(ValueError, match="gradient"):
            Overdamped(target)


class TestTheScheme:
    def test_any_word_is_accepted(self, setting):
        assert not SplittingScheme(Hamiltonian(setting.target()), "OBAB", step=0.2).is_palindrome
        assert not SplittingScheme(Overdamped(setting.target()), "BO", step=0.2).is_palindrome

    def test_a_letter_met_twice_takes_half_the_step_each_time(self, setting):
        scheme = SplittingScheme(Underdamped(setting.target(), friction=1.0), "OBABO", step=0.2)
        assert scheme.word == (("O", 0.5), ("B", 0.5), ("A", 1.0), ("B", 0.5), ("O", 0.5))
        assert scheme.is_palindrome

    @pytest.mark.parametrize(
        "word, keywords, match",
        [
            ("OXAB", {}, "letters"),
            ([("B", 0.5), ("A", 1.0)], {}, "sum to one"),
            ("", {}, "at least one"),
            ("BAB", {"repeat": 0}, "whole number"),
        ],
    )
    def test_a_malformed_word_is_refused(self, setting, word, keywords, match):
        with pytest.raises(ValueError, match=match):
            SplittingScheme(Hamiltonian(setting.target()), word, step=0.2, **keywords)

    @pytest.mark.parametrize("step", [0.0, -1.0, float("inf")])
    def test_a_step_that_is_not_positive_and_finite_is_refused(self, setting, step):
        with pytest.raises(ValueError, match="positive and finite"):
            SplittingScheme(Hamiltonian(setting.target()), "BAB", step=step)

    def test_a_new_step_is_a_new_scheme(self, setting):
        scheme = SplittingScheme(Hamiltonian(setting.target()), "BAB", step=0.2)
        assert scheme.with_step(0.4).step == 0.4
        assert scheme.step == 0.2

    @pytest.mark.parametrize(
        "build",
        [
            lambda t: SplittingScheme(Hamiltonian(t), "OBAB", step=1.3),
            lambda t: SplittingScheme(Hamiltonian(t), "BAB", step=0.3, repeat=3),
            lambda t: SplittingScheme(Underdamped(t, friction=0.5), "OBABO", step=2.0),
            lambda t: SplittingScheme(PreconditionedOU(t), "O", step=2.0),
        ],
        ids=["hmc", "block", "underdamped", "pcn"],
    )
    def test_the_reference_is_invariant_at_a_large_step(self, setting, build, rng):
        check_scheme(build(setting.target()), rng=rng, samples=3000)

    def test_a_word_that_does_not_preserve_the_reference_is_caught(self, setting, rng):
        class Shrink(Flow):
            touches = frozenset({"position"})

            def apply(self, state, time, /, *, rng=None):
                space = self.target.model_space
                return state.moved(space.scale(0.5, state.position), state.velocity), 0.0

        class Broken(Hamiltonian):
            def _build_flows(self):
                return {**super()._build_flows(), "S": Shrink(self._target)}

            def with_target(self, target, /):
                return Broken(target)

        with pytest.raises(AssertionError):
            check_scheme(SplittingScheme(Broken(setting.target()), "SAS", step=0.5), rng=rng)


class TestTheBlockCorrection:
    def test_the_energy_is_the_hamiltonians(self, setting, rng):
        """The flows' energy changes add up to the finite-dimensional
        ``H == Phi + (|u|^2_{C^-1} + |v|^2_{C^-1}) / 2`` difference."""
        target = setting.target()
        dynamics = Hamiltonian(target)
        block = SplittingScheme(dynamics, "BAB", step=0.3, repeat=5)
        precision = setting.prior.precision
        space = setting.space

        def hamiltonian(state):
            u, v = state.position, state.velocity
            return target.value(state) + 0.5 * (
                space.inner_product(precision(u), u) + space.inner_product(precision(v), v)
            )

        state = dynamics.initial_state(setting.prior.sample(rng=rng), rng=rng)
        proposal, energy = block.apply(state)
        assert target.value(proposal) - target.value(state) + energy == pytest.approx(
            hamiltonian(proposal) - hamiltonian(state), rel=1e-10
        )

    def test_a_rejection_keeps_the_position_and_flips_the_velocity(self, setting, rng):
        """With no refresh in the word, the flipped velocity is known."""
        dynamics = Hamiltonian(setting.target())
        scheme = SplittingScheme(dynamics, "BAB", step=5.0)
        kernel = MetropolizedBlock(scheme.block("BAB", repeat=20)).within(scheme)
        state = kernel.initial_state(setting.prior.sample(rng=rng), rng=rng)
        for _ in range(20):
            step = kernel.step(state, rng=rng)
            if not step.info.accepted:
                break
        else:
            pytest.fail("a step this large should be rejected")
        assert np.array_equal(step.state.position, state.position)
        assert np.array_equal(step.state.velocity, setting.space.negative(state.velocity))

    def test_a_block_that_is_not_a_palindrome_is_refused(self, setting):
        scheme = SplittingScheme(Hamiltonian(setting.target()), "BA", step=0.2)
        with pytest.raises(ValueError, match="palindrome"):
            MetropolizedBlock(scheme)

    def test_a_block_with_a_stochastic_letter_is_refused(self, setting):
        scheme = SplittingScheme(Hamiltonian(setting.target()), "OBAB", step=0.2)
        with pytest.raises(ValueError, match="deterministic"):
            MetropolizedBlock(scheme)

    def test_a_dynamics_without_a_velocity_is_refused(self, setting):
        scheme = SplittingScheme(Overdamped(setting.target()), "BOB", step=0.2)
        with pytest.raises(ValueError, match="velocity"):
            MetropolizedBlock(scheme)

    def test_the_block_must_sit_in_the_scheme(self, setting):
        dynamics = Hamiltonian(setting.target())
        block = MetropolizedBlock(SplittingScheme(dynamics, "BAB", step=0.2))
        with pytest.raises(ValueError, match="does not appear"):
            block.within(SplittingScheme(dynamics, "OABA", step=0.2))
        with pytest.raises(ValueError, match="one dynamics"):
            block.within(SplittingScheme(Hamiltonian(setting.target()), "OBAB", step=0.2))
        with pytest.raises(ValueError, match="one step"):
            block.within(SplittingScheme(dynamics, "OBAB", step=0.3))


class TestMetropolisHastings:
    def test_the_mala_ratio_is_the_direct_one(self, setting, rng):
        """Against the finite-dimensional densities: the posterior's, and the
        proposal's ``N(c (u - h K grad Phi), b^2 C)``."""
        target = setting.target()
        scheme = SplittingScheme(Overdamped(target), "BO", step=0.05)
        covariance = setting.components_covariance(setting.prior)
        potential = setting.problem.potential(setting.data)
        c, b = np.cos(PriorOU(target).angle(0.05)), np.sin(PriorOU(target).angle(0.05))

        def log_posterior(u):
            return -potential(u) + multivariate_normal(np.zeros(4), covariance).logpdf(u)

        def log_proposal(u, v):
            drift = target.preconditioned_gradient(State(u))
            return multivariate_normal(c * (u - 0.05 * drift), b * b * covariance).logpdf(v)

        for _ in range(3):
            state = State(setting.prior.sample(rng=rng))
            proposal, _ = scheme.apply(state, rng=rng)
            u, v = state.position, proposal.position
            library = target.value(state) - target.value(proposal) + scheme.log_transition_ratio(
                state, proposal
            )
            direct = log_posterior(v) + log_proposal(v, u) - log_posterior(u) - log_proposal(u, v)
            assert library == pytest.approx(direct, rel=1e-8, abs=1e-10)

    def test_a_reference_preserving_word_has_a_unit_ratio(self, setting, rng):
        scheme = SplittingScheme(PreconditionedOU(setting.target()), "O", step=0.3)
        state = State(setting.prior.sample(rng=rng))
        assert scheme.log_transition_ratio(state, scheme.apply(state, rng=rng)[0]) == 0.0

    def test_a_word_of_unknown_density_is_refused(self, setting):
        with pytest.raises(ValueError):
            MetropolisHastings(SplittingScheme(Overdamped(setting.target()), "OB", step=0.2))

    def test_a_dynamics_with_a_velocity_is_refused(self, setting):
        with pytest.raises(ValueError):
            MetropolisHastings(SplittingScheme(Hamiltonian(setting.target()), "OBAB", step=0.2))


class TestTheChains:
    def kernel(self, setting, step=0.3):
        return MetropolisHastings(
            SplittingScheme(PreconditionedOU(setting.target()), "O", step=step)
        )

    def test_hmc_by_hand_samples_the_posterior(self, setting):
        dynamics = Hamiltonian(setting.target())
        scheme = SplittingScheme(dynamics, "OBAB", step=0.3)
        kernel = MetropolizedBlock(scheme.block("BAB", repeat=5)).within(scheme)
        posterior = Chains(2)(
            kernel,
            start=setting.prior,
            steps=2000,
            burn_in=500,
            adapt=StepSizeAdaptation(acceptance=0.7),
            record=setting.record,
            rng=np.random.default_rng(7),
        )
        assert isinstance(posterior, SampledPosterior)
        assert_recovers_the_mean(posterior, setting.exact)

    def test_a_result_does_not_depend_on_the_workers(self, setting):
        run = lambda n_jobs: Chains(2)(
            self.kernel(setting), start=setting.prior, steps=200,
            rng=np.random.default_rng(5), n_jobs=n_jobs,
        )
        serial, parallel = run(None), run(2)
        for a, b in zip(serial.chains, parallel.chains):
            assert np.array_equal(a.potential, b.potential)

    def test_the_thinned_states_are_stored_and_drawn(self, setting, rng):
        posterior = Chains(2)(
            self.kernel(setting), start=setting.prior, steps=100, thin=25, rng=rng
        )
        assert len(posterior.stored_positions) == 8
        drawn = posterior.sample_thinned(rng=rng)
        assert any(drawn is x for x in posterior.stored_positions)

    def test_without_thinning_nothing_is_stored(self, setting, rng):
        posterior = Chains(1)(self.kernel(setting), start=setting.prior, steps=20, rng=rng)
        with pytest.raises(ValueError, match="thin"):
            posterior.sample_thinned(rng=rng)


    def test_the_records_are_named_and_shaped(self, setting, rng):
        posterior = Chains(3)(
            self.kernel(setting), start=setting.prior, steps=40,
            record=setting.record, record_names=["first"], rng=rng,
        )
        assert posterior.scalars().shape == (2, 3, 40)
        assert posterior.diagnostics().names == ("potential", "first")
        assert 0.0 <= posterior.acceptance_rate <= 1.0

    def test_the_callback_sees_every_step(self, setting, rng):
        seen = []
        Chains(2)(
            self.kernel(setting), start=setting.prior, steps=10, burn_in=5, rng=rng,
            callback=lambda chain, iteration, step: seen.append((chain, iteration)),
        )
        assert sorted(seen) == [(c, i) for c in range(2) for i in range(15)]

    def test_a_start_of_infinite_potential_is_refused(self, setting, rng):
        target = Target(
            setting.problem, setting.data, setting.prior,
            constraint=Ball(setting.space, radius=1e-3),
        )
        kernel = MetropolisHastings(SplittingScheme(PreconditionedOU(target), "O", step=0.3))
        with pytest.raises(ValueError):
            Chains(1)(kernel, start=setting.space.from_components(np.ones(4)), steps=5, rng=rng)

    @pytest.mark.parametrize("keywords", [{"steps": 0}, {"steps": 5, "burn_in": -1},
                                          {"steps": 5, "thin": 0}])
    def test_bad_lengths_are_refused(self, setting, rng, keywords):
        with pytest.raises(ValueError):
            Chains(1)(self.kernel(setting), start=setting.prior, rng=rng, **keywords)


class TestTheAdaptation:
    def test_the_step_grows_when_it_accepts_too_often_and_shrinks_otherwise(self):
        adaptation = StepSizeAdaptation(acceptance=0.5)
        assert adaptation.update(0.1, 0, 0.9) > 0.1
        assert adaptation.update(0.1, 0, 0.1) < 0.1

    def test_the_step_stops_at_its_maximum(self):
        adaptation = StepSizeAdaptation(acceptance=0.5, maximum=0.2)
        assert adaptation.update(0.19, 0, 1.0) == 0.2

    @pytest.mark.parametrize(
        "keywords",
        [{"acceptance": 1.0}, {"acceptance": 0.5, "rate": 0.0},
         {"acceptance": 0.5, "decay": 0.5}, {"acceptance": 0.5, "maximum": 0.0}],
    )
    def test_bad_settings_are_refused(self, keywords):
        with pytest.raises(ValueError):
            StepSizeAdaptation(**keywords)


class TestTempering:
    def test_it_estimates_the_evidence(self, setting):
        target = setting.target()
        mutation = MetropolisHastings(SplittingScheme(PreconditionedOU(target), "O", step=0.05))
        ensemble = Tempering(1000, mutation=mutation, mutation_steps=20)(
            target, rng=np.random.default_rng(0)
        )
        assert isinstance(ensemble, WeightedEnsemble)
        assert ensemble.log_normalizer.value == pytest.approx(
            setting.posterior.log_normalizer.value, abs=0.15
        )
        assert ensemble.temperatures[-1] == 1.0
        assert np.all(np.diff(ensemble.temperatures) > 0.0)
        assert ensemble.weights.sum() == pytest.approx(1.0)

    def test_a_reference_other_than_the_prior_is_refused(self, setting, rng):
        target = setting.target(setting.inversion.reference(setting.data))
        mutation = MetropolisHastings(SplittingScheme(PreconditionedOU(target), "O", step=0.05))
        with pytest.raises(ValueError, match="prior"):
            Tempering(10, mutation=mutation)(target, rng=rng)

    @pytest.mark.parametrize(
        "keywords",
        [{"particles": 1}, {"schedule": "linear"}, {"ess_fraction": 1.0},
         {"mutation_steps": 0}],
    )
    def test_bad_settings_are_refused(self, setting, keywords):
        mutation = MetropolisHastings(
            SplittingScheme(PreconditionedOU(setting.target()), "O", step=0.05)
        )
        particles = keywords.pop("particles", 10)
        with pytest.raises(ValueError):
            Tempering(particles, mutation=mutation, **keywords)


class TestThePresets:
    @pytest.mark.parametrize(
        "build, steps",
        [
            (lambda s: PCN(s.problem, s.prior, step=0.3), 8000),
            (lambda s: MALA(s.problem, s.prior, step=0.3), 8000),
            (lambda s: HMC(s.problem, s.prior, step=0.3, length=5), 2000),
            (lambda s: PreconditionedHMC(s.problem, s.prior, step=0.3), 4000),
            (lambda s: GeometricMALA(s.problem, s.inversion.reference(s.data), step=0.5), 2000),
            (
                lambda s: GeometricHMC(
                    s.problem, s.inversion.reference(s.data), step=0.4, length=4
                ),
                1000,
            ),
        ],
        ids=["pcn", "mala", "hmc", "preconditioned-hmc", "geometric-mala", "geometric-hmc"],
    )
    def test_each_samples_the_posterior(self, setting, build, steps):
        """The slow samplers on the prior's reference run longer to reach the
        same Monte Carlo error; the geometric ones, about the exact posterior,
        draw nearly independent states."""
        sampler = build(setting)
        posterior = sampler(
            setting.data, steps=steps, chains=2, burn_in=800, record=setting.record,
            rng=np.random.default_rng(7),
        )
        assert_recovers_the_mean(posterior, setting.exact)

    def test_a_preset_is_its_hand_assembled_kernel(self, setting):
        kernel = HMC(setting.problem, setting.prior, step=0.2, length=3).kernel(setting.data)
        assert isinstance(kernel.dynamics, Hamiltonian)
        assert kernel.block.repeat == 3
        assert kernel.step_size == 0.2

    def test_an_exact_reference_stops_the_step_at_a_quarter_turn(self, setting):
        sampler = GeometricHMC(
            setting.problem, setting.inversion.reference(setting.data), step=0.1, length=4
        )
        posterior = sampler(setting.data, steps=10, chains=1, burn_in=200,
                            rng=np.random.default_rng(1))
        assert posterior.chains[0].step_size == pytest.approx(np.pi / 8)

    def test_a_gradient_sampler_needs_a_derivative(self, setting):
        forward = Operator.from_callables(
            setting.space, setting.forward.codomain, lambda x: setting.forward(x)
        )
        problem = ForwardProblem(forward, error=setting.error)
        PCN(problem, setting.prior, step=0.1)
        with pytest.raises(ValueError, match="gradient"):
            MALA(problem, setting.prior, step=0.1)

    def test_a_geometric_preset_takes_a_reference(self, setting):
        with pytest.raises(TypeError, match="GaussianReference"):
            GeometricHMC(setting.problem, setting.prior, step=0.1, length=2)

    def test_a_reference_over_another_prior_is_refused(self, setting):
        other = GaussianReference.from_prior(
            gi.GaussianMeasure.from_standard_deviation(setting.space, 2.0)
        )
        with pytest.raises(ValueError, match="prior"):
            HMC(setting.problem, setting.prior, step=0.1, length=2, reference=other)

    def test_a_preset_is_a_measure_estimator(self, setting):
        sampler = PCN(setting.problem, setting.prior, step=0.1)
        assert isinstance(sampler, MeasureEstimator)
        assert sampler.data_space == setting.problem.data_space
        assert sampler.push_forward(setting.record).property_space == setting.record.codomain

    def test_smc_estimates_the_evidence(self, setting):
        ensemble = SMC(PCN(setting.problem, setting.prior, step=0.05), particles=1000,
                       mutation_steps=20)(setting.data, rng=np.random.default_rng(0))
        assert ensemble.log_normalizer.value == pytest.approx(
            setting.posterior.log_normalizer.value, abs=0.15
        )

    def test_smc_starts_from_the_prior(self, setting):
        sampler = PCN(setting.problem, setting.prior, step=0.05,
                      reference=setting.inversion.reference(setting.data))
        with pytest.raises(ValueError, match="prior"):
            SMC(sampler)


class TestTheConverters:
    def test_pcn_beta_is_the_step_s_sine(self, setting):
        h = time_from_pcn_beta(0.3)
        assert np.sin(PriorOU(setting.target()).angle(h)) == pytest.approx(0.3)
        assert pcn_beta(h) == pytest.approx(0.3)

    def test_crank_nicolson_delta_is_the_step_s_contraction(self, setting):
        h = time_from_crank_nicolson(0.5)
        assert np.cos(PriorOU(setting.target()).angle(h)) == pytest.approx(1.5 / 2.5)
        assert crank_nicolson_delta(h) == pytest.approx(0.5)

    @pytest.mark.parametrize(
        "convert, value",
        [(time_from_pcn_beta, 1.0), (time_from_crank_nicolson, 2.0), (pcn_beta, 0.0),
         (crank_nicolson_delta, -1.0)],
    )
    def test_values_out_of_range_are_refused(self, convert, value):
        with pytest.raises(ValueError):
            convert(value)


def make_mixture(setting, rng):
    """Two components on a two-dimensional chart, apart along it."""
    chart = [setting.space.random(rng=rng) for _ in range(2)]
    return MixtureReference(
        setting.prior,
        chart,
        np.array([0.4, 0.6]),
        np.array([[0.5, 0.0], [-0.5, 0.3]]),
        np.array([np.diag([0.2, 0.3]), np.diag([0.4, 0.1])]),
    )


class TestOverAMixtureReference:
    def test_the_reference_as_a_target_has_no_residual(self, setting, rng):
        """``Phi_ref == 0``: its target is the reference itself, not the prior."""
        mixture = make_mixture(setting, rng)
        target = Target.from_reference(mixture)
        state = Hamiltonian(target).initial_state(mixture.sample(rng=rng), rng=rng)
        assert target.value(state) == pytest.approx(0.0, abs=1e-12)
        assert setting.space.norm(target.gradient(state)) < 1e-12

    @pytest.mark.parametrize(
        "build",
        [
            lambda t: SplittingScheme(PreconditionedOU(t), "LO", step=0.7),
            lambda t: SplittingScheme(Overdamped(t), "LBO", step=0.7),
            lambda t: SplittingScheme(Hamiltonian(t), "LOBAB", step=0.7),
            lambda t: SplittingScheme(Underdamped(t, friction=1.0), "LOBAB", step=0.7),
        ],
        ids=["pcn", "mala", "hmc", "underdamped"],
    )
    def test_every_word_leaves_the_mixture_invariant(self, setting, build, rng):
        check_scheme(build(setting.target(make_mixture(setting, rng))), rng=rng, samples=3000)

    def test_a_state_starts_with_a_component(self, setting, rng):
        mixture = make_mixture(setting, rng)
        state = Hamiltonian(setting.target(mixture)).initial_state(
            mixture.components[1].expectation, rng=rng
        )
        assert state.label == 1
        assert state.velocity is not None

    def test_a_word_without_the_label_jump_is_refused(self, setting, rng):
        target = setting.target(make_mixture(setting, rng))
        with pytest.raises(ValueError, match="label jump"):
            MetropolisHastings(SplittingScheme(PreconditionedOU(target), "O", step=0.3))
        scheme = SplittingScheme(Hamiltonian(target), "OBAB", step=0.3)
        with pytest.raises(ValueError, match="label jump"):
            MetropolizedBlock(scheme.block("BAB")).within(scheme)

    def test_a_label_jump_needs_a_mixture(self, setting):
        with pytest.raises(ValueError, match="mixture"):
            LabelJump(setting.target())

    @pytest.mark.parametrize(
        "build, steps",
        [
            (lambda s, r: PCN(s.problem, s.prior, step=0.3, reference=r), 6000),
            (lambda s, r: MALA(s.problem, s.prior, step=0.1, reference=r), 4000),
            (lambda s, r: GeometricHMC(s.problem, r, step=0.3, length=3), 2000),
        ],
        ids=["pcn", "mala", "hmc"],
    )
    def test_each_preset_samples_the_posterior(self, setting, build, steps, rng):
        """Any mixture over the prior leaves a chain exact; this one is not
        fitted to the posterior at all."""
        posterior = build(setting, make_mixture(setting, rng))(
            setting.data, steps=steps, chains=2, burn_in=500, record=setting.record,
            rng=np.random.default_rng(7),
        )
        assert_recovers_the_mean(posterior, setting.exact)


@pytest.fixture(scope="module")
def recorded_run(setting):
    """An HMC run recording every component, its states stored."""
    space = setting.space
    record = LinearOperator.from_matrix(space, EuclideanSpace(4), np.eye(4), form="components")
    return HMC(setting.problem, setting.prior, step=0.3, length=5)(
        setting.data, steps=3000, chains=2, burn_in=500, record=record, thin=50,
        rng=np.random.default_rng(11),
    )


@pytest.fixture(scope="module")
def pcn_run(setting):
    """A pCN run recording the probe."""
    return PCN(setting.problem, setting.prior, step=0.3)(
        setting.data, steps=2000, chains=2, burn_in=500, record=setting.record,
        rng=np.random.default_rng(5),
    )


class TestTheFittedReference:
    def test_a_gaussian_fit_is_close_to_the_posterior(self, setting, recorded_run):
        """Charted on every component, the fit is the posterior to the
        Monte Carlo error."""
        reference = recorded_run.fitted_reference()
        assert isinstance(reference, GaussianReference)
        assert np.allclose(reference.expectation, setting.posterior.expectation, atol=0.1)
        fitted = setting.components_covariance(reference)
        exact = setting.components_covariance(setting.posterior)
        assert np.allclose(fitted, exact, atol=0.1 * np.abs(exact).max())

    def test_a_mixture_fit_has_the_components_asked_for(self, recorded_run, rng):
        reference = recorded_run.fitted_reference(components=2, rng=rng)
        assert isinstance(reference, MixtureReference)
        assert len(reference.components) == 2

    def test_another_chart_is_read_off_the_stored_states(self, setting, recorded_run):
        reference = recorded_run.fitted_reference(chart=setting.record)
        assert reference.rank == 1

    def test_a_chart_is_needed(self, setting, rng):
        kernel = MetropolisHastings(
            SplittingScheme(PreconditionedOU(setting.target()), "O", step=0.3)
        )
        posterior = Chains(1)(kernel, start=setting.prior, steps=20, rng=rng)
        with pytest.raises(ValueError, match="record="):
            posterior.fitted_reference()
        with pytest.raises(ValueError, match="thin="):
            posterior.fitted_reference(chart=setting.record)

    def test_a_weighted_ensemble_fits_one_too(self, setting):
        ensemble = SMC(PCN(setting.problem, setting.prior, step=0.05), particles=500,
                       mutation_steps=10)(setting.data, rng=np.random.default_rng(0))
        reference = ensemble.fitted_reference(setting.record)
        assert reference.rank == 1
        assert setting.space.inner_product(
            reference.expectation, setting.space.basis_vector(0)
        ) == pytest.approx(setting.exact, abs=0.15)


class TestFreshDraws:
    def test_draws_are_from_the_posterior(self, setting, pcn_run):
        values = np.array([
            setting.space.inner_product(x, setting.space.basis_vector(0))
            for x in pcn_run.samples(120, rng=np.random.default_rng(1))
        ])
        error = values.std() / np.sqrt(values.size)
        assert abs(values.mean() - setting.exact) < 4 * error
        assert pcn_run.steps_per_draw >= 1

    def test_draws_do_not_depend_on_the_workers(self, pcn_run):
        serial = pcn_run.samples(4, rng=np.random.default_rng(2))
        parallel = pcn_run.samples(4, rng=np.random.default_rng(2), n_jobs=2)
        assert all(np.array_equal(a, b) for a, b in zip(serial, parallel))

    def test_a_draw_reference_can_be_set(self, setting, pcn_run, rng):
        mixture = pcn_run.fitted_reference(components=2, rng=rng)
        other = pcn_run.with_draw_reference(mixture)
        assert other.draw_reference is mixture
        assert pcn_run.draw_reference is not mixture
        other.sample(rng=rng)

    def test_a_draw_reference_over_another_prior_is_refused(self, setting, pcn_run):
        prior = gi.GaussianMeasure.from_standard_deviation(setting.space, 2.0)
        with pytest.raises(ValueError, match="prior"):
            pcn_run.with_draw_reference(GaussianReference.from_prior(prior))


class TestAPosteriorWithTwoModes:
    """``d == m^2 + e``: a gradient sampler stays in the well it starts in;
    over a mixture fitted to a pCN run it jumps between them."""

    def test_a_fitted_mixture_moves_hmc_between_the_modes(self):
        space, data_space = EuclideanSpace(1), EuclideanSpace(1)
        forward = Operator.from_callables(
            space, data_space, lambda m: m**2,
            derivative=lambda m: LinearOperator.from_matrix(
                space, data_space, np.array([[2.0 * m[0]]]), form="components"
            ),
        )
        prior = gi.GaussianMeasure.from_standard_deviation(space, 1.0)
        problem = ForwardProblem(
            forward, error=gi.GaussianMeasure.from_standard_deviation(data_space, 0.3)
        )
        data = np.array([1.0])
        record = LinearOperator.from_matrix(space, EuclideanSpace(1), np.eye(1), form="components")
        rng = np.random.default_rng(37)
        run = PCN(problem, prior, step=0.1)(
            data, steps=1000, chains=4, burn_in=300, record=record, rng=rng
        )
        mixture = run.fitted_reference(components=2, rng=rng)
        posterior = GeometricHMC(problem, mixture, step=0.3, length=3)(
            data, steps=1000, chains=4, burn_in=300, record=record, rng=rng
        )
        m = posterior.scalars()[1]
        crossings = int(np.sum(np.diff(np.sign(m), axis=1) != 0))
        assert crossings > 500
        assert np.mean(m > 0) == pytest.approx(0.5, abs=0.05)
        assert posterior.diagnostics().rhat[1] < 1.02


class MeshSetting:
    """Attenuation along arcs of a circle, ``d_i == exp(-(a_i, m))``, on grids
    of any size, with one data set made on the finest and shared by all."""

    arcs = [(np.array([a]), np.array([a + 1.2])) for a in np.linspace(0.0, 2 * np.pi, 7)[:-1]]
    sigma = 0.05
    levels = (32, 128, 512)

    def __init__(self):
        rng = np.random.default_rng(38)
        _, prior, problem = self.build(self.levels[-1])
        truth = prior.sample(rng=rng)
        self.data = problem.forward_operator(truth) + self.sigma * rng.standard_normal(6)
        self.grids = {}
        for points in self.levels:
            space, prior, problem = self.build(points)
            start = MaximumAPosteriori(problem, prior).solve(self.data).minimizer
            self.grids[points] = (space, prior, problem, start)

    def build(self, points):
        space = circle.Sobolev(points, 1.0, 0.3)
        prior = space.sobolev_measure(2.0, 0.3)
        paths = space.path_integral_operator(self.arcs, dense=True)
        data_space = paths.codomain

        def derivative(m):
            weights = LinearOperator.from_matrix(
                data_space, data_space, np.diag(-np.exp(-paths(m))), form="components"
            )
            return weights @ paths

        forward = Operator.from_callables(
            space, data_space, lambda m: np.exp(-paths(m)), derivative=derivative
        )
        error = gi.GaussianMeasure.from_standard_deviation(data_space, self.sigma)
        return space, prior, ForwardProblem(forward, error=error)


def mean_acceptance(probabilities):
    """The mean acceptance probability and its Monte Carlo standard error."""
    probabilities = np.asarray(probabilities)
    tau = integrated_autocorrelation_time(probabilities)
    return probabilities.mean(), probabilities.std() * np.sqrt(tau / probabilities.size)


@pytest.fixture(scope="module")
def mesh_setting():
    return MeshSetting()


@pytest.mark.slow
class TestTheMeshDoesNotMatter:
    """The function-space guarantee, measured: at a fixed step, a sampler's
    acceptance rate is the same on every grid of a nonlinear problem, and
    tempering takes the same stages. A random walk with the same proposal size
    is the control that shows the check can fail: its acceptance falls as the
    grid is refined."""

    @pytest.mark.parametrize(
        "build",
        [
            lambda problem, prior: PCN(problem, prior, step=0.005, acceptance=None),
            lambda problem, prior: MALA(problem, prior, step=0.005, acceptance=None),
            lambda problem, prior: HMC(problem, prior, step=0.08, length=3, acceptance=None),
        ],
        ids=["pcn", "mala", "hmc"],
    )
    def test_the_acceptance_rate_is_flat_under_refinement(self, mesh_setting, build):
        rates = []
        for space, prior, problem, start in mesh_setting.grids.values():
            posterior = build(problem, prior)(
                mesh_setting.data, steps=400, chains=2, burn_in=100, start=start,
                rng=np.random.default_rng(1),
            )
            rates.append(mean_acceptance(np.stack([c.acceptance for c in posterior.chains])))
        (coarse, coarse_error) = rates[0]
        assert 0.1 < coarse < 0.95
        for rate, error in rates[1:]:
            assert abs(rate - coarse) < 3.0 * np.hypot(error, coarse_error)

    def test_tempering_takes_the_same_path_under_refinement(self, mesh_setting):
        """The invariant for SMC is the stages it takes. Each stage raises
        beta as far as the effective sample size allows, which is the
        information the data add, a property of the posterior and not of the
        grid. So the number of stages, and where each early stage stops, must
        agree across grids. (The effective size before each resampling is set
        by construction and tests nothing.) The final stage may split in two
        when the bisection lands just short of one, and the evidence, from 300
        particles, is checked only to its own scatter."""
        paths = []
        evidence = []
        for space, prior, problem, start in mesh_setting.grids.values():
            ensemble = SMC(PCN(problem, prior, step=0.005), particles=300, mutation_steps=10)(
                mesh_setting.data, rng=np.random.default_rng(0)
            )
            paths.append(ensemble.temperatures)
            evidence.append(ensemble.log_normalizer.value)
        coarse = paths[0]
        assert coarse.size >= 5
        for path, log_z in zip(paths[1:], evidence[1:]):
            assert abs(path.size - coarse.size) <= 1
            assert np.all(np.abs(np.log(path[:6]) - np.log(coarse[:6])) < 0.3)
            assert abs(log_z - evidence[0]) < 1.5

    def test_a_random_walk_is_not(self, mesh_setting):
        """``m + s xi`` with ``xi`` a prior draw and ``s`` pCN's proposal size
        at ``h == 0.005``: the same proposal less pCN's contraction."""
        step = np.sqrt(1.0 - np.exp(-0.01))
        rates = []
        for space, prior, problem, start in mesh_setting.grids.values():
            potential = problem.potential(mesh_setting.data)

            def log_density(m):
                return -potential(m) - 0.5 * space.inner_product(prior.precision(m), m)

            generator = np.random.default_rng(2)
            current, value = start, log_density(start)
            probabilities = []
            for _ in range(600):
                proposal = space.axpy(step, prior.sample(rng=generator), space.copy(current))
                proposed = log_density(proposal)
                probabilities.append(np.exp(min(0.0, proposed - value)))
                if np.log(generator.random()) < proposed - value:
                    current, value = proposal, proposed
            rates.append(mean_acceptance(probabilities)[0])
        assert rates[0] > 0.15
        assert rates[-1] < 0.3 * rates[0]
