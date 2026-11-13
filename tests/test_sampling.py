"""Function-space samplers, layer by layer, against a posterior known exactly.

A linear problem with a Gaussian prior has a Gaussian posterior in closed
form, so every sampler can be held to it: each layer is checked for what its
construction promises (the reference invariant at any step, a block's energy
the Hamiltonian's, a transition density the proposal's), and each move, run
by the Monte Carlo estimator, for the posterior mean within its own Monte
Carlo error. Everything
runs on a dense metric, where a component standing in for a vector, or a
representer for a derivative, would show.
"""

import ast
import pathlib
import warnings

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

import pygeoinf as gi
from pygeoinf.algebra.operators import EvaluationError, LinearOperator, Operator
from pygeoinf.algebra.spaces import EuclideanSpace
from pygeoinf.geometry import Ball
from pygeoinf.inference import (
    ForwardProblem,
    LaplaceApproximation,
    LinearForwardProblem,
    LinearGaussianInversion,
    MonteCarloEstimator,
)
from pygeoinf.inference.estimators import MeasureEstimator
from pygeoinf.probability import GaussianReference, MixtureReference
from pygeoinf.sampling import (
    HMC,
    MALA,
    PCN,
    SMC,
    Blockwise,
    Chains,
    GeneralizedHMC,
    Gibbs,
    GradientDrift,
    Hamiltonian,
    LabelJump,
    LikelihoodKick,
    MetropolisHastings,
    MetropolizedBlock,
    MomentumRefresh,
    Overdamped,
    PreconditionedOU,
    PriorOU,
    PriorRotation,
    SampledPosterior,
    SplittingScheme,
    State,
    StepSizeAdaptation,
    Sweep,
    Target,
    Underdamped,
    WeightedEnsemble,
)
from pygeoinf.inference.laplace import MaximumAPosteriori
from pygeoinf.sampling.flows import (
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
        self.forward = LinearOperator.from_matrix(
            space, data_space, matrix, form="components"
        )
        self.prior = gi.GaussianMeasure.from_standard_deviation(space, 1.0)
        self.error = gi.GaussianMeasure.from_standard_deviation(data_space, 0.5)
        self.problem = LinearForwardProblem(self.forward, error=self.error)
        self.data = self.forward(self.prior.sample(rng=rng)) + self.error.sample(
            rng=rng
        )
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
        return Target(
            self.problem.potential(self.data),
            self.prior if reference is None else reference,
        )

    def components_covariance(self, measure):
        """The covariance in components, ``C_c`` with ``C x == C_c G x``."""
        space = self.space
        columns = np.array(
            [measure.covariance(space.basis_vector(i)) for i in range(4)]
        ).T
        return columns @ np.linalg.inv(space.gram_matrix())


@pytest.fixture(scope="module")
def setting():
    return Setting()


def reference_named(setting, name, rng):
    """The prior, whose covariance is the identity and whose mean is zero, or
    a rank-2 Gaussian with neither.

    The rank-2 Gaussian is fitted to the posterior of other data under twice
    the noise. Fitted to this posterior it would match it exactly along the
    two directions where its covariance differs from the prior's, and
    ``grad Phi_ref`` would be constant there: a drift preconditioned by the
    prior's covariance in place of the reference's would then change the
    transition ratio by a constant that cancels, and pass unseen."""
    if name == "prior":
        return setting.prior
    error = gi.GaussianMeasure.from_standard_deviation(setting.forward.codomain, 1.0)
    inversion = LinearGaussianInversion(
        LinearForwardProblem(setting.forward, error=error), setting.prior
    )
    return inversion.reference(setting.data + 1.0, rank=2, rng=rng)


def sample(
    setting,
    move,
    /,
    *,
    reference=None,
    constraint=None,
    record=None,
    start=None,
    **budget,
):
    """The setting's posterior sampled by a move, through the estimator, its
    probe recorded unless another map is given."""
    estimator = MonteCarloEstimator(
        setting.problem,
        setting.prior,
        sampler=Chains(move, **budget),
        reference=reference,
        constraint=constraint,
        record=setting.record if record is None else record,
    )
    return estimator(setting.data, start=start)


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
            target.gradient(State(u)),
            target.residual_functional().gradient(u),
            atol=1e-12,
        )
        assert np.allclose(
            target.preconditioned_gradient(State(u)),
            reference.covariance(target.gradient(State(u))),
            atol=1e-12,
        )

    def test_one_evaluation_serves_value_gradient_and_every_temperature(
        self, setting, rng
    ):
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
        target = Target(
            ForwardProblem(forward, error=setting.error).potential(setting.data),
            setting.prior,
        )
        state = State(setting.prior.sample(rng=rng))
        target.gradient(state)
        target.value(state)
        target.with_temperature(3.0).value(state)
        assert calls == ["value"]

    def test_a_point_outside_the_constraint_is_infinite_unevaluated(self, setting, rng):
        target = Target(
            setting.problem.potential(setting.data),
            setting.prior,
            constraint=Ball(setting.space, radius=1e-3),
        )
        state = State(setting.space.from_components(np.ones(4)))
        assert target.value(state) == float("inf")

    def test_a_failed_forward_is_infinite(self, setting, rng):
        def value(x):
            raise EvaluationError("the solver diverged")

        forward = Operator.from_callables(
            setting.space, setting.forward.codomain, value
        )
        target = Target(
            ForwardProblem(forward, error=setting.error).potential(setting.data),
            setting.prior,
        )
        assert target.value(State(setting.prior.sample(rng=rng))) == float("inf")

    def test_the_reference_alone_has_no_potential(self, setting, rng):
        target = Target.from_reference(setting.prior)
        state = State(setting.prior.sample(rng=rng))
        assert target.value(state) == 0.0
        assert setting.space.norm(target.gradient(state)) == 0.0

    def test_a_reference_on_another_space_is_refused(self, setting):
        with pytest.raises(ValueError, match="domain"):
            Target(
                setting.problem.potential(setting.data),
                gi.GaussianMeasure.from_standard_deviation(EuclideanSpace(4), 1.0),
            )

    def test_a_local_reference_is_built_over_the_prior(self, setting, rng):
        reference = setting.inversion.reference(setting.data, rank=2, rng=rng)
        with pytest.raises(ValueError, match="prior"):
            Target(
                setting.problem.potential(setting.data),
                reference,
                local=lambda u: reference,
            )


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
        assert np.allclose(
            back.velocity, setting.space.negative(state.velocity), atol=1e-12
        )

    def test_a_gradient_flow_on_a_target_without_one_is_refused(self, setting):
        forward = Operator.from_callables(
            setting.space, setting.forward.codomain, lambda x: setting.forward(x)
        )
        target = Target(
            ForwardProblem(forward, error=setting.error).potential(setting.data),
            setting.prior,
        )
        PreconditionedOU(target)
        with pytest.raises(ValueError, match="gradient"):
            Overdamped(target)


class CountingSetting:
    """The linear setting's problem behind a forward that counts its calls."""

    def __init__(self, setting):
        self.calls = {"value": 0, "derivative": 0}

        def value(x):
            self.calls["value"] += 1
            return setting.forward(x)

        def derivative(x):
            self.calls["derivative"] += 1
            return setting.forward

        forward = Operator.from_callables(
            setting.space, setting.forward.codomain, value, derivative=derivative
        )
        self.problem = ForwardProblem(forward, error=setting.error)
        self.setting = setting

    def target(self, reference=None):
        s = self.setting
        return Target(
            self.problem.potential(s.data), s.prior if reference is None else reference
        )

    def reset(self):
        self.calls.update(value=0, derivative=0)


class TestTheForwardIsEvaluatedOncePerPosition:
    """The value and the gradient at a position come from one forward
    evaluation, whichever of them is asked for first."""

    @pytest.mark.parametrize(
        "flow", [LikelihoodKick, GradientDrift], ids=["kick", "drift"]
    )
    def test_a_kick_or_a_drift(self, setting, flow, rng):
        counting = CountingSetting(setting)
        target = counting.target()
        state = Hamiltonian(target).initial_state(
            setting.prior.sample(rng=rng), rng=rng
        )
        counting.reset()
        flow(target).apply(state, 0.1)
        assert counting.calls == {"value": 1, "derivative": 1}

    @pytest.mark.parametrize(
        "move, value, derivative",
        [
            (PCN(step=0.3, acceptance=None), 1, 0),
            (MALA(step=0.1, acceptance=None), 1, 1),
            (HMC(step=0.2, length=3, acceptance=None), 3, 3),
            (GeneralizedHMC(step=0.2, length=3, acceptance=None), 3, 3),
        ],
        ids=["pcn", "mala", "hmc", "generalized-hmc"],
    )
    def test_each_step_of_a_move(self, setting, move, value, derivative, rng):
        counting = CountingSetting(setting)
        kernel = move.kernel(counting.target())
        state = kernel.initial_state(setting.prior.sample(rng=rng), rng=rng)
        state = kernel.step(state, rng=rng).state
        counting.reset()
        steps = 20
        for _ in range(steps):
            state = kernel.step(state, rng=rng).state
        assert counting.calls == {
            "value": value * steps,
            "derivative": derivative * steps,
        }

    def test_an_accepted_label_jump_before_a_kick(self, setting, rng):
        counting = CountingSetting(setting)
        target = counting.target(make_mixture(setting, rng))
        jump = Hamiltonian(target).flows()["L"]
        kick = LikelihoodKick(target)
        for _ in range(200):
            state = Hamiltonian(target).initial_state(
                target.reference.sample(rng=rng), rng=rng
            )
            target.value(state, gradient=True)
            counting.reset()
            jumped, _ = jump.apply(state, 0.0, rng=rng)
            if jumped.label != state.label:
                break
        else:
            pytest.fail("No label jump was accepted in 200 tries.")
        kick.apply(jumped, 0.1)
        assert counting.calls == {"value": 1, "derivative": 1}


class TestTheScheme:
    def test_any_word_is_accepted(self, setting):
        assert not SplittingScheme(
            Hamiltonian(setting.target()), "OBAB", step=0.2
        ).is_palindrome
        assert not SplittingScheme(
            Overdamped(setting.target()), "BO", step=0.2
        ).is_palindrome

    def test_a_letter_met_twice_takes_half_the_step_each_time(self, setting):
        scheme = SplittingScheme(
            Underdamped(setting.target(), friction=1.0), "OBABO", step=0.2
        )
        assert scheme.word == (
            ("O", 0.5),
            ("B", 0.5),
            ("A", 1.0),
            ("B", 0.5),
            ("O", 0.5),
        )
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

    def test_its_probes_can_be_drawn_from_a_measure(self, setting, rng):
        scheme = SplittingScheme(Hamiltonian(setting.target()), "OBAB", step=0.3)
        check_scheme(scheme, rng=rng, measure=setting.posterior, samples=2000)
        elsewhere = gi.GaussianMeasure.from_standard_deviation(EuclideanSpace(4), 1.0)
        with pytest.raises(ValueError, match="lives on"):
            check_scheme(scheme, rng=rng, measure=elsewhere)

    def test_a_word_that_does_not_preserve_the_reference_is_caught(self, setting, rng):
        class Shrink(Flow):
            def apply(self, state, time, /, *, rng=None):
                space = self.target.model_space
                return state.moved(
                    space.scale(0.5, state.position), state.velocity
                ), 0.0

        class Broken(Hamiltonian):
            def _build_flows(self):
                return {**super()._build_flows(), "S": Shrink(self._target)}

            def with_target(self, target, /):
                return Broken(target)

        with pytest.raises(AssertionError):
            check_scheme(
                SplittingScheme(Broken(setting.target()), "SAS", step=0.5), rng=rng
            )


class TestTheBlockCorrection:
    @pytest.mark.parametrize("reference", ["prior", "rank 2"])
    def test_the_energy_is_the_hamiltonians(self, setting, reference, rng):
        """The flows' energy changes add up to the finite-dimensional
        ``H == Phi + |u|^2_{C^-1} / 2 + |v|^2_{K^-1} / 2`` difference, with
        ``C`` the prior's covariance and ``K`` the reference's, the velocity's
        law, written from their densities in components."""
        target = setting.target(reference_named(setting, reference, rng))
        dynamics = Hamiltonian(target)
        block = SplittingScheme(dynamics, "BAB", step=0.3, repeat=5)
        potential = setting.problem.potential(setting.data)
        prior = multivariate_normal(
            np.zeros(4), setting.components_covariance(setting.prior)
        )
        velocity = multivariate_normal(
            np.zeros(4), setting.components_covariance(target.reference)
        )

        def hamiltonian(state):
            u, v = state.position, state.velocity
            return potential(u) - prior.logpdf(u) - velocity.logpdf(v)

        state = dynamics.initial_state(setting.prior.sample(rng=rng), rng=rng)
        proposal, energy = block.apply(state)
        assert target.value(proposal) - target.value(state) + energy == pytest.approx(
            hamiltonian(proposal) - hamiltonian(state), rel=1e-10, abs=1e-8
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
        assert np.array_equal(
            step.state.velocity, setting.space.negative(state.velocity)
        )

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
            block.within(
                SplittingScheme(Hamiltonian(setting.target()), "OBAB", step=0.2)
            )
        with pytest.raises(ValueError, match="one step"):
            block.within(SplittingScheme(dynamics, "OBAB", step=0.3))


class TestMetropolisHastings:
    @pytest.mark.parametrize("reference", ["prior", "rank 2"])
    def test_the_mala_ratio_is_the_direct_one(self, setting, reference, rng):
        """Against the finite-dimensional densities: the posterior's, and the
        proposal's ``N(m + c (u - h K grad Phi_ref - m), b^2 K)``, about the
        reference ``N(m, K)``. The drift is written out in components, with
        ``grad Phi_ref == grad Phi - K^-1 (u - m) + C^-1 u``, rather than read
        from the target."""
        target = setting.target(reference_named(setting, reference, rng))
        scheme = SplittingScheme(Overdamped(target), "BO", step=0.05)
        prior_covariance = setting.components_covariance(setting.prior)
        covariance = setting.components_covariance(target.reference)
        mean = target.reference.expectation
        potential = setting.problem.potential(setting.data)
        matrix = setting.forward.matrix(form="components")
        precision = 1.0 / 0.5**2
        c, b = np.cos(PriorOU(target).angle(0.05)), np.sin(PriorOU(target).angle(0.05))

        def log_posterior(u):
            return -potential(u) + multivariate_normal(
                np.zeros(4), prior_covariance
            ).logpdf(u)

        def drift(u):
            gradient = precision * matrix.T @ (matrix @ u - setting.data)
            return (
                covariance @ gradient
                - (u - mean)
                + covariance @ np.linalg.solve(prior_covariance, u)
            )

        def log_proposal(u, v):
            center = mean + c * (u - 0.05 * drift(u) - mean)
            return multivariate_normal(center, b * b * covariance).logpdf(v)

        for _ in range(3):
            state = State(setting.prior.sample(rng=rng))
            proposal, _ = scheme.apply(state, rng=rng)
            u, v = state.position, proposal.position
            library = (
                target.value(state)
                - target.value(proposal)
                + scheme.log_transition_ratio(state, proposal)
            )
            direct = (
                log_posterior(v)
                + log_proposal(v, u)
                - log_posterior(u)
                - log_proposal(u, v)
            )
            assert library == pytest.approx(direct, rel=1e-8, abs=1e-10)

    def test_a_reference_preserving_word_has_a_unit_ratio(self, setting, rng):
        scheme = SplittingScheme(PreconditionedOU(setting.target()), "O", step=0.3)
        state = State(setting.prior.sample(rng=rng))
        assert (
            scheme.log_transition_ratio(state, scheme.apply(state, rng=rng)[0]) == 0.0
        )

    def test_a_word_of_unknown_density_is_refused(self, setting):
        with pytest.raises(ValueError):
            MetropolisHastings(
                SplittingScheme(Overdamped(setting.target()), "OB", step=0.2)
            )

    def test_a_dynamics_with_a_velocity_is_refused(self, setting):
        with pytest.raises(ValueError):
            MetropolisHastings(
                SplittingScheme(Hamiltonian(setting.target()), "OBAB", step=0.2)
            )


class TestTheChains:
    """Chains of a kernel built by hand, which runs at its own step on the
    target it is called on."""

    def kernel(self, setting, step=0.3):
        return MetropolisHastings(
            SplittingScheme(PreconditionedOU(setting.target()), "O", step=step)
        )

    def test_hmc_by_hand_samples_the_posterior(self, setting):
        target = setting.target()
        dynamics = Hamiltonian(target)
        scheme = SplittingScheme(dynamics, "OBAB", step=0.1)
        kernel = MetropolizedBlock(scheme.block("BAB", repeat=5)).within(scheme)
        posterior = Chains(
            kernel, steps=2000, burn_in=500, chains=2, rng=np.random.default_rng(7)
        )(target, start=setting.prior, record=setting.record)
        assert isinstance(posterior, SampledPosterior)
        assert_recovers_the_mean(posterior, setting.exact)

    def test_a_kernel_runs_on_the_target_it_is_called_on(self, setting, rng):
        kernel = self.kernel(setting)
        tempered = kernel.target.with_temperature(2.0)
        posterior = Chains(kernel, steps=5, chains=1, rng=rng)(tempered)
        assert posterior.kernel.target is tempered
        assert posterior.kernel.step_size == kernel.step_size

    def test_a_result_does_not_depend_on_the_workers(self, setting):
        kernel = self.kernel(setting)
        run = lambda n_jobs: Chains(
            kernel, steps=200, chains=2, rng=np.random.default_rng(5), n_jobs=n_jobs
        )(kernel.target, start=setting.prior)
        serial, parallel = run(None), run(2)
        for a, b in zip(serial.chains, parallel.chains):
            assert np.array_equal(a.potential, b.potential)

    def test_the_thinned_states_are_stored_and_drawn(self, setting, rng):
        kernel = self.kernel(setting)
        posterior = Chains(kernel, steps=100, chains=2, thin=25, rng=rng)(
            kernel.target, start=setting.prior
        )
        assert len(posterior.stored_positions) == 8
        drawn = posterior.sample_thinned(rng=rng)
        assert any(drawn is x for x in posterior.stored_positions)

    def test_without_thinning_nothing_is_stored(self, setting, rng):
        kernel = self.kernel(setting)
        posterior = Chains(kernel, steps=20, chains=1, rng=rng)(kernel.target)
        with pytest.raises(ValueError, match="thin"):
            posterior.sample_thinned(rng=rng)

    def test_the_records_are_named_and_shaped(self, setting, rng):
        kernel = self.kernel(setting)
        posterior = Chains(kernel, steps=40, chains=3, rng=rng)(
            kernel.target, record=setting.record, record_names=["first"]
        )
        assert posterior.scalars().shape == (2, 3, 40)
        assert posterior.diagnostics().names == ("potential", "first")
        assert 0.0 <= posterior.acceptance_rate <= 1.0

    def test_a_record_may_be_nonlinear(self, setting, rng):
        space = setting.space
        squared = Operator.from_callables(
            space, EuclideanSpace(1), lambda x: np.array([space.squared_norm(x)])
        )
        kernel = self.kernel(setting)
        posterior = Chains(kernel, steps=30, chains=2, thin=1, rng=rng)(
            kernel.target, record=squared, record_names=["squared norm"]
        )
        recorded = posterior.scalars()[1]
        stored = np.array([space.squared_norm(x) for x in posterior.stored_positions])
        assert recorded.shape == (2, 30)
        np.testing.assert_allclose(recorded.ravel(), stored)
        assert posterior.record is squared
        with pytest.raises(ValueError, match="nonlinear"):
            posterior.fitted_reference()

    def test_a_record_off_the_model_space_is_refused(self, setting, rng):
        kernel = self.kernel(setting)
        other = LinearOperator.from_matrix(
            EuclideanSpace(4), EuclideanSpace(1), np.ones((1, 4)), form="components"
        )
        with pytest.raises(ValueError, match="model space"):
            Chains(kernel, steps=5, rng=rng)(kernel.target, record=other)

    def test_the_callback_sees_every_step(self, setting, rng):
        seen = []
        kernel = self.kernel(setting)
        Chains(
            kernel,
            steps=10,
            burn_in=5,
            chains=2,
            rng=rng,
            callback=lambda chain, iteration, step: seen.append((chain, iteration)),
        )(kernel.target)
        assert sorted(seen) == [(c, i) for c in range(2) for i in range(15)]

    def test_a_start_of_infinite_potential_is_refused(self, setting, rng):
        target = Target(
            setting.problem.potential(setting.data),
            setting.prior,
            constraint=Ball(setting.space, radius=1e-3),
        )
        kernel = MetropolisHastings(
            SplittingScheme(PreconditionedOU(target), "O", step=0.3)
        )
        with pytest.raises(ValueError):
            Chains(kernel, steps=5, chains=1, rng=rng)(
                target, start=setting.space.from_components(np.ones(4))
            )

    @pytest.mark.parametrize(
        "keywords",
        [
            {"steps": 0},
            {"steps": 5, "burn_in": -1},
            {"steps": 5, "thin": 0},
            {"steps": 5, "chains": 0},
        ],
    )
    def test_bad_lengths_are_refused(self, setting, keywords):
        with pytest.raises(ValueError):
            Chains(self.kernel(setting), **keywords)


MOVES_ON_A_BOUNDED_POSTERIOR = pytest.mark.parametrize(
    "move",
    [
        PCN(step=0.3, acceptance=None),
        MALA(step=0.1, acceptance=None),
        HMC(step=0.2, length=3, acceptance=None),
    ],
    ids=["pcn", "mala", "hmc"],
)


def visited(problem, prior, move, data, start, *, constraint=None):
    """Every state a short run visits, burn-in included, and how many of its
    proposals had no finite potential."""
    positions, infinite = [], []

    def callback(chain, iteration, step):
        positions.append(step.state.position)
        infinite.append(step.info.energy_error == float("inf"))

    sampler = Chains(
        move,
        steps=300,
        chains=2,
        burn_in=50,
        rng=np.random.default_rng(4),
        callback=callback,
    )
    MonteCarloEstimator(problem, prior, sampler=sampler, constraint=constraint)(
        data, start=start
    )
    return positions, sum(infinite)


class TestAnInfinitePotentialInARunningChain:
    """A proposal with no finite potential, outside the constraint or refused
    by the forward, is rejected and the chain runs on: pCN and MALA meet it
    in the Metropolis-Hastings ratio, HMC in a kick partway along a
    trajectory. Each region holds about half the posterior's mass, so many
    proposals fall outside it."""

    @MOVES_ON_A_BOUNDED_POSTERIOR
    def test_a_constraint_is_never_left(self, setting, move):
        center = setting.posterior.expectation
        ball = Ball(setting.space, radius=0.85, center=center)
        positions, infinite = visited(
            setting.problem, setting.prior, move, setting.data, center, constraint=ball
        )
        distances = [setting.space.norm(x - center) for x in positions]
        assert max(distances) <= 0.85
        assert infinite > 0

    @MOVES_ON_A_BOUNDED_POSTERIOR
    def test_a_refused_proposal_is_rejected(self, setting, move, rng):
        threshold = setting.exact

        def value(x):
            if setting.record(x)[0] > threshold:
                raise EvaluationError("the solver diverged")
            return setting.forward(x)

        forward = Operator.from_callables(
            setting.space,
            setting.forward.codomain,
            value,
            derivative=lambda x: setting.forward,
        )
        problem = ForwardProblem(forward, error=setting.error)
        start = next(
            x
            for x in (setting.posterior.sample(rng=rng) for _ in range(100))
            if setting.record(x)[0] < threshold
        )
        positions, infinite = visited(problem, setting.prior, move, setting.data, start)
        assert max(setting.record(x)[0] for x in positions) <= threshold
        assert infinite > 0


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
        [
            {"acceptance": 1.0},
            {"acceptance": 0.5, "rate": 0.0},
            {"acceptance": 0.5, "decay": 0.5},
            {"acceptance": 0.5, "maximum": 0.0},
        ],
    )
    def test_bad_settings_are_refused(self, keywords):
        with pytest.raises(ValueError):
            StepSizeAdaptation(**keywords)

    def test_a_move_adapts_its_step_during_burn_in(self, setting):
        posterior = sample(
            setting,
            HMC(step=0.01, length=3, acceptance=0.7),
            steps=50,
            burn_in=300,
            chains=2,
            rng=np.random.default_rng(3),
        )
        assert all(chain.step_size > 0.05 for chain in posterior.chains)

    def test_a_step_adapted_to_its_maximum_is_warned_of(self, setting):
        """About the exact posterior every step is accepted, so the step grows
        until it is stopped; the warning names the caller's line."""
        with pytest.warns(UserWarning, match="about its largest") as caught:
            sample(
                setting,
                MALA(step=1.0),
                reference=setting.inversion.reference(setting.data),
                steps=4,
                chains=2,
                burn_in=200,
                rng=np.random.default_rng(1),
            )
        assert caught[0].filename == __file__

    @pytest.mark.parametrize(
        "move, burn_in",
        [
            (MALA(step=1.0, acceptance=None), 200),
            (MALA(step=1.0), 0),
            (HMC(step=0.1, length=4), 200),
        ],
        ids=["fixed-step", "no-burn-in", "hmc-about-a-reference-at-its-quarter-turn"],
    )
    def test_a_step_not_adapted_or_aimed_there_is_not_warned_of(
        self, setting, move, burn_in
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            sample(
                setting,
                move,
                reference=setting.inversion.reference(setting.data),
                steps=4,
                chains=2,
                burn_in=burn_in,
                rng=np.random.default_rng(1),
            )

    def test_hmc_over_the_prior_is_warned_of_at_its_quarter_turn(self):
        """Over the prior a quarter turn is no aim: the data's force is what
        a trajectory is for."""
        target = Target.from_reference(
            gi.GaussianMeasure.from_standard_deviation(make_dense_metric_space(4), 1.0)
        )
        assert HMC(step=0.1, length=4).adaptation(target).warns
        assert PCN(step=0.1).adaptation(target).warns


class TestTempering:
    """SMC on a target directly, its mutation a kernel built by hand."""

    def mutation(self, target, step=0.05):
        return MetropolisHastings(
            SplittingScheme(PreconditionedOU(target), "O", step=step)
        )

    def test_it_estimates_the_evidence(self, setting):
        target = setting.target()
        ensemble = SMC(
            self.mutation(target),
            particles=1000,
            mutation_steps=20,
            rng=np.random.default_rng(0),
        )(target)
        assert isinstance(ensemble, WeightedEnsemble)
        assert ensemble.log_normalizer.value == pytest.approx(
            setting.posterior.log_normalizer.value, abs=0.15
        )
        assert ensemble.temperatures[-1] == 1.0
        assert np.all(np.diff(ensemble.temperatures) > 0.0)
        assert ensemble.weights.sum() == pytest.approx(1.0)

    def test_a_reference_other_than_the_prior_is_refused(self, setting, rng):
        target = setting.target(setting.inversion.reference(setting.data))
        with pytest.raises(ValueError, match="prior"):
            SMC(self.mutation(target), particles=10, rng=rng)(target)

    def test_a_start_or_a_record_is_refused(self, setting, rng):
        target = setting.target()
        smc = SMC(self.mutation(target), particles=10, rng=rng)
        with pytest.raises(ValueError, match="prior"):
            smc(target, start=setting.prior)
        with pytest.raises(ValueError, match="forward"):
            smc(target, record=setting.record)

    @pytest.mark.parametrize(
        "keywords",
        [{"particles": 1}, {"ess_fraction": 1.0}, {"mutation_steps": 0}],
    )
    def test_bad_settings_are_refused(self, setting, keywords):
        with pytest.raises(ValueError):
            SMC(self.mutation(setting.target()), **keywords)


class TestTheMoves:
    @pytest.mark.parametrize(
        "move, reference, steps",
        [
            (PCN(step=0.3), None, 8000),
            (MALA(step=0.3), None, 8000),
            (HMC(step=0.3, length=5), None, 2000),
            (GeneralizedHMC(step=0.3), None, 4000),
            (MALA(step=0.5), "exact", 2000),
            (HMC(step=0.4, length=4), "exact", 1000),
        ],
        ids=[
            "pcn",
            "mala",
            "hmc",
            "generalized-hmc",
            "mala-about-the-posterior",
            "hmc-about-the-posterior",
        ],
    )
    def test_each_samples_the_posterior(self, setting, move, reference, steps):
        """The moves about the prior run longer to reach the same Monte Carlo
        error; about the exact posterior, they draw nearly independent
        states."""
        posterior = sample(
            setting,
            move,
            reference=setting.inversion if reference == "exact" else None,
            steps=steps,
            chains=2,
            burn_in=800,
            rng=np.random.default_rng(7),
        )
        assert_recovers_the_mean(posterior, setting.exact)

    def test_a_move_is_its_hand_assembled_kernel(self, setting):
        kernel = HMC(step=0.2, length=3).kernel(setting.target())
        assert isinstance(kernel.dynamics, Hamiltonian)
        assert kernel.block.repeat == 3
        assert kernel.step_size == 0.2
        assert (
            HMC(step=0.2, length=3).kernel(setting.target(), step=0.4).step_size == 0.4
        )

    def test_an_exact_reference_stops_the_step_at_a_quarter_turn(self, setting):
        posterior = sample(
            setting,
            HMC(step=0.1, length=4),
            reference=setting.inversion.reference(setting.data),
            steps=10,
            chains=1,
            burn_in=200,
            rng=np.random.default_rng(1),
        )
        assert posterior.chains[0].step_size == pytest.approx(np.pi / 8)

    @pytest.mark.parametrize(
        "build",
        [
            lambda: PCN(step=0.0),
            lambda: PCN(step=0.1, acceptance=1.0),
            lambda: HMC(step=0.1, length=0),
            lambda: GeneralizedHMC(step=0.1, friction=0.0),
        ],
        ids=["step", "acceptance", "length", "friction"],
    )
    def test_bad_settings_are_refused(self, build):
        with pytest.raises(ValueError):
            build()


class TestTheEstimator:
    def test_it_is_a_measure_estimator(self, setting):
        estimator = MonteCarloEstimator(
            setting.problem, setting.prior, sampler=Chains(PCN(step=0.1), steps=10)
        )
        assert isinstance(estimator, MeasureEstimator)
        assert estimator.data_space == setting.problem.data_space
        assert estimator.property_space == setting.problem.model_space
        assert (
            estimator.push_forward(setting.record).property_space
            == setting.record.codomain
        )

    def test_its_target_is_the_problem_s_against_the_reference(self, setting, rng):
        reference = setting.inversion.reference(setting.data, rank=2, rng=rng)
        estimator = MonteCarloEstimator(
            setting.problem,
            setting.prior,
            sampler=Chains(PCN(step=0.1), steps=10),
            reference=reference,
        )
        target = estimator.target(setting.data)
        u = setting.prior.sample(rng=rng)
        assert target.reference is reference
        assert target.value(State(u)) == pytest.approx(
            setting.target(reference).value(State(u)), rel=1e-12
        )

    def test_an_estimator_as_reference_is_fitted_to_the_data(self, setting):
        estimator = MonteCarloEstimator(
            setting.problem,
            setting.prior,
            sampler=Chains(PCN(step=0.1), steps=10),
            reference=setting.inversion,
        )
        built = estimator.target(setting.data).reference
        x = setting.space.basis_vector(1)
        assert np.allclose(built.expectation, setting.posterior.expectation)
        assert np.allclose(built.covariance(x), setting.posterior.covariance(x))

    def test_a_gradient_move_needs_a_derivative(self, setting):
        forward = Operator.from_callables(
            setting.space, setting.forward.codomain, lambda x: setting.forward(x)
        )
        problem = ForwardProblem(forward, error=setting.error)
        MonteCarloEstimator(
            problem, setting.prior, sampler=Chains(PCN(step=0.1), steps=5)
        )
        with pytest.raises(ValueError, match="gradient"):
            MonteCarloEstimator(
                problem, setting.prior, sampler=Chains(MALA(step=0.1), steps=5)
            )

    def test_a_reference_over_another_prior_is_refused(self, setting):
        other = GaussianReference.from_prior(
            gi.GaussianMeasure.from_standard_deviation(setting.space, 2.0)
        )
        with pytest.raises(ValueError, match="prior"):
            MonteCarloEstimator(
                setting.problem,
                setting.prior,
                sampler=Chains(HMC(step=0.1, length=2), steps=5),
                reference=other,
            )

    def test_a_reference_of_another_kind_is_refused(self, setting):
        with pytest.raises(TypeError, match="GaussianReference"):
            MonteCarloEstimator(
                setting.problem,
                setting.prior,
                sampler=Chains(PCN(step=0.1), steps=5),
                reference=gi.GaussianMeasure.from_standard_deviation(
                    setting.space, 2.0
                ),
            )

    def test_a_move_is_not_a_sampler(self, setting):
        with pytest.raises(TypeError, match="Chains"):
            MonteCarloEstimator(setting.problem, setting.prior, sampler=PCN(step=0.1))

    def test_smc_estimates_the_evidence(self, setting):
        ensemble = MonteCarloEstimator(
            setting.problem,
            setting.prior,
            sampler=SMC(
                PCN(step=0.05),
                particles=1000,
                mutation_steps=20,
                rng=np.random.default_rng(0),
            ),
        )(setting.data)
        assert ensemble.log_normalizer.value == pytest.approx(
            setting.posterior.log_normalizer.value, abs=0.15
        )

    def test_smc_starts_from_the_prior(self, setting):
        estimator = MonteCarloEstimator(
            setting.problem,
            setting.prior,
            sampler=SMC(PCN(step=0.05), particles=10),
            reference=setting.inversion.reference(setting.data),
        )
        with pytest.raises(ValueError, match="prior"):
            estimator(setting.data)

    def test_smc_with_an_hmc_mutation(self, setting):
        """The evidence, and the posterior mean of the probe by the weighted
        particles, within four of its standard errors at the ensemble's
        effective sample size. The population is smaller than the pCN test's,
        to keep the test short, and the evidence is held more loosely."""
        ensemble = MonteCarloEstimator(
            setting.problem,
            setting.prior,
            sampler=SMC(
                HMC(step=0.2, length=3),
                particles=400,
                mutation_steps=5,
                rng=np.random.default_rng(0),
            ),
        )(setting.data)
        assert ensemble.log_normalizer.value == pytest.approx(
            setting.posterior.log_normalizer.value, abs=0.3
        )
        probe = setting.space.basis_vector(0)
        mean = ensemble.weighted_mean(lambda x: setting.space.inner_product(x, probe))
        deviation = np.sqrt(
            setting.space.inner_product(setting.posterior.covariance(probe), probe)
        )
        error = deviation / np.sqrt(ensemble.effective_sample_size)
        assert abs(mean - setting.exact) < 4.0 * error

    def test_a_weighted_mean_weighs_each_particle(self, setting):
        """Against the sum written out from the particles and their weights."""
        ensemble = MonteCarloEstimator(
            setting.problem,
            setting.prior,
            sampler=SMC(
                PCN(step=0.05),
                particles=200,
                mutation_steps=2,
                rng=np.random.default_rng(1),
            ),
        )(setting.data)
        values = np.array([setting.record(x)[0] for x in ensemble.particles])
        assert ensemble.weighted_mean(lambda x: setting.record(x)[0]) == pytest.approx(
            ensemble.weights @ values, rel=1e-12
        )
        assert ensemble.weighted_mean(lambda x: 1.0) == pytest.approx(1.0)


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
        [
            (time_from_pcn_beta, 1.0),
            (time_from_crank_nicolson, 2.0),
            (pcn_beta, 0.0),
            (crank_nicolson_delta, -1.0),
        ],
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
        check_scheme(
            build(setting.target(make_mixture(setting, rng))), rng=rng, samples=3000
        )

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
        "move, steps",
        [
            (PCN(step=0.3), 6000),
            (MALA(step=0.1), 4000),
            (HMC(step=0.3, length=3), 2000),
        ],
        ids=["pcn", "mala", "hmc"],
    )
    def test_each_move_samples_the_posterior(self, setting, move, steps, rng):
        """Any mixture over the prior leaves a chain exact; this one is not
        fitted to the posterior at all."""
        posterior = sample(
            setting,
            move,
            reference=make_mixture(setting, rng),
            steps=steps,
            chains=2,
            burn_in=500,
            rng=np.random.default_rng(7),
        )
        assert_recovers_the_mean(posterior, setting.exact)

    def test_a_label_jump_is_accepted_as_the_densities_say(self, setting, rng):
        """Each jump's decision against the finite-dimensional acceptance
        ``pi(u') nu(u) / (pi(u) nu(u'))``, with ``pi`` the posterior's
        density and ``nu`` the mixture's, both in components; the jump's
        generator is replayed for the component it proposes and the uniform
        it accepts on."""
        mixture = make_mixture(setting, rng)
        target = setting.target(mixture)
        dynamics = Hamiltonian(target)
        jump = dynamics.flows()["L"]
        potential = setting.problem.potential(setting.data)
        prior = multivariate_normal(
            np.zeros(4), setting.components_covariance(setting.prior)
        )
        components = [
            multivariate_normal(
                component.expectation, setting.components_covariance(component)
            )
            for component in mixture.components
        ]

        def log_posterior_over_mixture(u):
            mixture_density = logsumexp(
                [
                    np.log(weight) + component.logpdf(u)
                    for weight, component in zip(mixture.weights, components)
                ]
            )
            return -potential(u) + prior.logpdf(u) - mixture_density

        decided, accepted = 0, 0
        for seed in range(300):
            state = dynamics.initial_state(mixture.sample(rng=rng), rng=rng)
            result, _ = jump.apply(state, 0.0, rng=np.random.default_rng(seed))
            replay = np.random.default_rng(seed)
            destination = int(replay.choice(len(mixture.weights), p=mixture.weights))
            if destination == state.label:
                assert result is state
                continue
            proposal = mixture.transport(state.position, state.label, destination)
            log_alpha = log_posterior_over_mixture(
                proposal
            ) - log_posterior_over_mixture(state.position)
            log_uniform = np.log(replay.random())
            if abs(log_uniform - log_alpha) < 1e-9:
                continue
            if log_uniform < log_alpha:
                assert result.label == destination
                assert np.allclose(result.position, proposal, atol=1e-12)
                accepted += 1
            else:
                assert result is state
            decided += 1
        assert decided > 50
        assert 0 < accepted < decided


@pytest.fixture(scope="module")
def recorded_run(setting):
    """An HMC run recording every component, its states stored."""
    space = setting.space
    record = LinearOperator.from_matrix(
        space, EuclideanSpace(4), np.eye(4), form="components"
    )
    return sample(
        setting,
        HMC(step=0.3, length=5),
        record=record,
        steps=3000,
        chains=2,
        burn_in=500,
        thin=50,
        rng=np.random.default_rng(11),
    )


@pytest.fixture(scope="module")
def pcn_run(setting):
    """A pCN run recording the probe."""
    return sample(
        setting,
        PCN(step=0.3),
        steps=2000,
        chains=2,
        burn_in=500,
        rng=np.random.default_rng(5),
    )


class TestTheFittedReference:
    def test_a_gaussian_fit_is_close_to_the_posterior(self, setting, recorded_run):
        """Charted on every component, the fit is the posterior to the
        Monte Carlo error."""
        reference = recorded_run.fitted_reference()
        assert isinstance(reference, GaussianReference)
        assert np.allclose(
            reference.expectation, setting.posterior.expectation, atol=0.1
        )
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
        posterior = Chains(kernel, steps=20, chains=1, rng=rng)(kernel.target)
        with pytest.raises(ValueError, match="record="):
            posterior.fitted_reference()
        with pytest.raises(ValueError, match="thin="):
            posterior.fitted_reference(chart=setting.record)

    def test_a_weighted_ensemble_fits_one_too(self, setting):
        ensemble = MonteCarloEstimator(
            setting.problem,
            setting.prior,
            sampler=SMC(
                PCN(step=0.05),
                particles=500,
                mutation_steps=10,
                rng=np.random.default_rng(0),
            ),
        )(setting.data)
        reference = ensemble.fitted_reference(setting.record)
        assert reference.rank == 1
        assert setting.space.inner_product(
            reference.expectation, setting.space.basis_vector(0)
        ) == pytest.approx(setting.exact, abs=0.15)


class TestFreshDraws:
    def test_draws_are_from_the_posterior(self, setting, pcn_run):
        values = np.array(
            [
                setting.space.inner_product(x, setting.space.basis_vector(0))
                for x in pcn_run.samples(120, rng=np.random.default_rng(1))
            ]
        )
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
            space,
            data_space,
            lambda m: m**2,
            derivative=lambda m: LinearOperator.from_matrix(
                space, data_space, np.array([[2.0 * m[0]]]), form="components"
            ),
        )
        prior = gi.GaussianMeasure.from_standard_deviation(space, 1.0)
        problem = ForwardProblem(
            forward, error=gi.GaussianMeasure.from_standard_deviation(data_space, 0.3)
        )
        data = np.array([1.0])
        record = LinearOperator.from_matrix(
            space, EuclideanSpace(1), np.eye(1), form="components"
        )
        rng = np.random.default_rng(37)

        def run(move, reference=None):
            sampler = Chains(move, steps=1000, chains=4, burn_in=300, rng=rng)
            return MonteCarloEstimator(
                problem, prior, sampler=sampler, reference=reference, record=record
            )(data)

        mixture = run(PCN(step=0.1)).fitted_reference(components=2, rng=rng)
        posterior = run(HMC(step=0.3, length=3), mixture)
        m = posterior.scalars()[1]
        crossings = int(np.sum(np.diff(np.sign(m), axis=1) != 0))
        assert crossings > 500
        assert np.mean(m > 0) == pytest.approx(0.5, abs=0.05)
        assert posterior.diagnostics().rhat[1] < 1.02


class NonlinearSetting:
    """``exp(B m)`` on a dense metric, its Laplace approximation, and the
    posterior mean of a probe by self-normalized importance sampling from the
    Laplace reference at the mode, with its standard error."""

    def __init__(self):
        rng = np.random.default_rng(4)
        space = make_dense_metric_space(4)
        data_space = EuclideanSpace(3)
        matrix = 0.5 * rng.standard_normal((3, 4))
        forward = Operator.from_callables(
            space,
            data_space,
            lambda m: np.exp(matrix @ m),
            derivative=lambda m: LinearOperator.from_matrix(
                space,
                data_space,
                np.diag(np.exp(matrix @ m)) @ matrix,
                form="components",
            ),
        )
        self.space = space
        self.prior = gi.GaussianMeasure.from_standard_deviation(space, 0.5)
        self.error = gi.GaussianMeasure.from_standard_deviation(data_space, 0.1)
        self.problem = ForwardProblem(forward, error=self.error)
        self.data = forward(self.prior.sample(rng=rng)) + self.error.sample(rng=rng)
        self.laplace = LaplaceApproximation.from_problem(
            self.problem, self.prior, gauss_newton=True
        )
        probe = space.basis_vector(0)
        self.record = LinearOperator.from_callables(
            space,
            EuclideanSpace(1),
            lambda x: np.array([space.inner_product(x, probe)]),
            adjoint=lambda c: c[0] * probe,
        )
        mode = self.laplace.solve(self.data).model
        proposal = self.laplace.reference(self.data, at=mode)
        potential = self.problem.potential(self.data)
        draws = [proposal.sample(rng=rng) for _ in range(20000)]
        log_weights = np.array(
            [-potential(x) - proposal.relative_log_density(x) for x in draws]
        )
        weights = np.exp(log_weights - log_weights.max())
        weights /= weights.sum()
        values = np.array([space.inner_product(x, probe) for x in draws])
        self.exact = float(weights @ values)
        self.exact_error = float(
            np.sqrt(np.sum(weights**2 * (values - self.exact) ** 2))
        )


@pytest.fixture(scope="module")
def nonlinear_setting():
    return NonlinearSetting()


def local_mala(s, /, **budget):
    """The estimator running a local MALA about the setting's Laplace
    approximation."""
    move = MALA(step=budget.pop("step", 1.0), local=True)
    return MonteCarloEstimator(
        s.problem,
        s.prior,
        sampler=Chains(move, **budget),
        reference=s.laplace,
        record=s.record,
    )


class TestTheLocalMALA:
    """MALA about the Laplace Gaussian at each state: the proposal is that
    Gaussian's Ornstein-Uhlenbeck step, and the acceptance compares the
    pair laws of the references at the state and at the proposal."""

    def test_the_ratio_is_the_direct_one(self, nonlinear_setting, rng):
        """Against the finite-dimensional densities: the posterior's, and
        each proposal's ``N(m_x + c (x - m_x), s^2 K_x)``."""
        s = nonlinear_setting
        space, prior, laplace = s.space, s.prior, s.laplace
        target = local_mala(s, steps=1).target(s.data)
        scheme = SplittingScheme(PreconditionedOU(target), "O", step=0.4)
        kernel = MetropolisHastings(scheme)
        c = np.exp(-0.4)
        gram_inverse = np.linalg.inv(space.gram_matrix())

        def covariance(measure):
            columns = np.array(
                [measure.covariance(space.basis_vector(i)) for i in range(4)]
            )
            return columns.T @ gram_inverse

        potential = s.problem.potential(s.data)
        prior_covariance = covariance(prior)

        def log_posterior(x):
            return -potential(x) + multivariate_normal(
                prior.expectation, prior_covariance
            ).logpdf(x)

        def log_proposal(x, y):
            local = laplace.reference(s.data, at=x)
            mean = local.expectation
            return multivariate_normal(
                mean + c * (x - mean), (1 - c * c) * covariance(local)
            ).logpdf(y)

        for _ in range(3):
            state = State(prior.sample(rng=rng))
            proposal, _ = scheme.apply(state, rng=rng)
            x, y = state.position, proposal.position
            library = (
                target.value(state)
                - target.value(proposal)
                + kernel._local_ratio(state, proposal)
            )
            direct = (
                log_posterior(y)
                + log_proposal(y, x)
                - log_posterior(x)
                - log_proposal(x, y)
            )
            assert library == pytest.approx(direct, rel=1e-8, abs=1e-8)

    def test_on_a_linear_problem_every_proposal_is_accepted(self, setting):
        """The local Gaussian is then the posterior at every state, so the
        sampler draws from it exactly, at any step."""
        laplace = LaplaceApproximation.from_problem(
            setting.problem, setting.prior, gauss_newton=True
        )
        posterior = sample(
            setting,
            MALA(step=3.0, acceptance=None, local=True),
            reference=laplace,
            steps=100,
            chains=1,
            rng=np.random.default_rng(3),
        )
        assert posterior.chains[0].acceptance.min() == pytest.approx(1.0, abs=1e-10)

    def test_it_samples_a_nonlinear_posterior(self, nonlinear_setting):
        s = nonlinear_setting
        posterior = local_mala(
            s, steps=1500, chains=2, burn_in=300, rng=np.random.default_rng(2)
        )(s.data)
        estimate = posterior.scalar_means()[1]
        assert abs(estimate.value - s.exact) < 4.0 * np.hypot(
            estimate.standard_error, s.exact_error
        )
        assert posterior.diagnostics().rhat.max() < 1.05

    def test_the_move_says_it_is_local(self):
        assert MALA(step=1.0, local=True).is_local
        assert Chains(MALA(step=1.0, local=True), steps=1).is_local
        assert not MALA(step=1.0).is_local

    def test_only_the_ou_word_is_offered(self, nonlinear_setting):
        s = nonlinear_setting
        target = local_mala(s, steps=1).target(s.data)
        with pytest.raises(ValueError, match="'O'"):
            MetropolisHastings(
                SplittingScheme(PreconditionedOU(target), "OO", step=0.4)
            )
        with pytest.raises(ValueError, match="Ornstein-Uhlenbeck"):
            Overdamped(target)
        with pytest.raises(ValueError, match="Ornstein-Uhlenbeck"):
            Hamiltonian(target)

    def test_tempering_refuses_it(self, nonlinear_setting, rng):
        s = nonlinear_setting
        move = MALA(step=1.0, local=True)
        estimator = MonteCarloEstimator(
            s.problem, s.prior, sampler=SMC(move, particles=10), reference=s.laplace
        )
        with pytest.raises(ValueError, match="prior"):
            estimator(s.data)
        target = estimator.target(s.data)
        with pytest.raises(ValueError, match="prior"):
            SMC(move.kernel(target), particles=10, rng=rng)(target)

    def test_it_needs_a_laplace_approximation(self, nonlinear_setting, setting):
        s = nonlinear_setting
        sampler = Chains(MALA(step=1.0, local=True), steps=5)
        with pytest.raises(ValueError, match="LaplaceApproximation"):
            MonteCarloEstimator(s.problem, s.prior, sampler=sampler)
        with pytest.raises(ValueError, match="prior"):
            MonteCarloEstimator(
                setting.problem, setting.prior, sampler=sampler, reference=s.laplace
            )
        with pytest.raises(ValueError, match="follows the state"):
            MALA(step=1.0, local=True).kernel(setting.target())


class GibbsSetting:
    """Two fields on a direct sum, tied in the prior through their noises and
    both seen by every datum, so the exact posterior couples them."""

    def __init__(self):
        rng = np.random.default_rng(1)
        field = make_dense_metric_space(4)
        self.prior = gi.GaussianMeasure.from_product(
            [
                gi.GaussianMeasure.from_standard_deviation(field, 1.0),
                gi.GaussianMeasure.from_standard_deviation(field, 0.5),
            ],
            correlation=np.array([[1.0, 0.6], [0.6, 1.0]]),
            labels=("a", "b"),
        )
        self.space = space = self.prior.domain
        data_space = EuclideanSpace(5)
        self.matrix = rng.standard_normal((5, 8))
        self.forward = LinearOperator.from_matrix(
            space, data_space, self.matrix, form="components"
        )
        self.error = gi.GaussianMeasure.from_standard_deviation(data_space, 0.3)
        self.problem = LinearForwardProblem(self.forward, error=self.error)
        self.data = self.forward(self.prior.sample(rng=rng)) + self.error.sample(
            rng=rng
        )
        self.posterior = LinearGaussianInversion(self.problem, self.prior)(self.data)
        probes = (field.basis_vector(0), field.basis_vector(1))
        self.record = LinearOperator.from_callables(
            space,
            EuclideanSpace(2),
            lambda x: np.array(
                [
                    field.inner_product(x[0], probes[0]),
                    field.inner_product(x[1], probes[1]),
                ]
            ),
            adjoint=lambda c: (c[0] * probes[0], c[1] * probes[1]),
        )
        self.exact = self.record(self.posterior.expectation)

    def target(self, problem=None):
        return Target((problem or self.problem).potential(self.data), self.prior)


@pytest.fixture(scope="module")
def gibbs_setting():
    return GibbsSetting()


def two_block_gibbs():
    """PCN on one field and HMC on the other, from one starting step."""
    return Gibbs({"a": PCN(step=0.3), "b": HMC(step=0.3, length=3)})


def gibbs_sample(s, move, /, **budget):
    """The Gibbs setting's posterior sampled by a move, its two probes
    recorded."""
    sampler = Chains(move, **budget)
    return MonteCarloEstimator(s.problem, s.prior, sampler=sampler, record=s.record)(
        s.data
    )


@pytest.fixture(scope="module")
def gibbs_run(gibbs_setting):
    """A short run of :func:`two_block_gibbs`: long enough for each block's
    step to adapt, too short to hold its means to the exact ones."""
    return gibbs_sample(
        gibbs_setting,
        two_block_gibbs(),
        steps=1000,
        chains=2,
        burn_in=500,
        rng=np.random.default_rng(2),
    )


def assert_recovers_both_means(posterior, exact):
    """Both recorded means within four of their Monte Carlo standard errors.

    No R-hat is asserted: over two chains of a few thousand correlated steps
    it is too noisy a statistic to hold to a threshold, and the means are
    what the run is for."""
    estimates = posterior.scalar_means()
    for k in range(2):
        estimate = estimates[k + 1]
        assert abs(estimate.value - exact[k]) < 4.0 * estimate.standard_error


class TestABlock:
    def test_it_is_the_whole_target_with_the_others_held(self, gibbs_setting, rng):
        s = gibbs_setting
        target = s.target()
        state = State(s.prior.sample(rng=rng))
        block = target.block("b", state)
        inner = State(state.position[1])
        assert block.value(inner) == pytest.approx(target.value(state), rel=1e-12)
        assert np.allclose(block.gradient(inner), target.gradient(state)[1], atol=1e-12)

    def test_its_reference_is_the_prior_s_conditional(self, gibbs_setting, rng):
        s = gibbs_setting
        state = State(s.prior.sample(rng=rng))
        reference = s.target().block(0, state).reference
        conditional = s.prior.conditional(0, state.position)
        field = conditional.domain
        assert np.allclose(reference.prior.expectation, conditional.expectation)
        x = field.random(rng=rng)
        assert np.allclose(reference.prior.covariance(x), conditional.covariance(x))

    def test_a_reference_other_than_the_prior_is_refused(self, gibbs_setting, rng):
        s = gibbs_setting
        reference = LinearGaussianInversion(s.problem, s.prior).reference(
            s.data, rank=2, rng=rng
        )
        target = Target(s.problem.potential(s.data), reference)
        with pytest.raises(ValueError, match="conditional"):
            target.block(0, State(s.prior.sample(rng=rng)))

    def test_a_model_off_a_direct_sum_is_refused(self, setting, rng):
        with pytest.raises(ValueError, match="direct sum"):
            setting.target().block(0, State(setting.prior.sample(rng=rng)))


class TestTheSweep:
    def test_the_move_is_a_sweep_of_blockwise_dynamics(self, gibbs_setting):
        s = gibbs_setting
        kernel = Gibbs(
            {"b": MALA(step=0.2), "a": PCN(step=0.3)}, order="symmetric"
        ).kernel(s.target())
        assert isinstance(kernel, Sweep) and isinstance(kernel.dynamics, Blockwise)
        assert kernel.step_sizes == (0.3, 0.2)
        flows = [flow.index for flow, _ in kernel.scheme.flows]
        assert flows == flows[::-1] and set(flows) == {0, 1}

    @pytest.mark.parametrize("order", ["systematic", "symmetric"])
    def test_it_leaves_the_reference_invariant(self, gibbs_setting, order, rng):
        s = gibbs_setting
        kernel = Gibbs(
            {"a": PCN(step=0.5), "b": HMC(step=0.5, length=3)}, order=order
        ).kernel(s.target())
        check_scheme(kernel.scheme, rng=rng, samples=2000)

    def test_a_sweep_evaluates_the_forward_once_per_proposal(self, gibbs_setting, rng):
        """The block starts from the likelihood the whole state holds, and
        hands its own back."""
        s = gibbs_setting
        calls = []

        def value(x):
            calls.append(1)
            return s.forward(x)

        forward = Operator.from_callables(
            s.space, s.forward.codomain, value, derivative=lambda x: s.forward
        )
        problem = ForwardProblem(forward, error=s.error)
        kernel = Gibbs(
            {"a": PCN(step=0.1), "b": PCN(step=0.1)}, repeat={"b": 2}
        ).kernel(s.target(problem))
        state = kernel.initial_state(s.prior.sample(rng=rng), rng=rng)
        kernel.target.value(state)
        calls.clear()
        kernel.step(state, rng=rng)
        assert len(calls) == 3

    def test_a_letter_that_does_not_preserve_the_target_is_refused(self, setting):
        with pytest.raises(ValueError, match="preserve"):
            Sweep(SplittingScheme(PreconditionedOU(setting.target()), "O", step=0.5))


class TestGibbs:
    @pytest.mark.slow
    def test_it_samples_the_posterior(self, gibbs_setting):
        posterior = gibbs_sample(
            gibbs_setting,
            two_block_gibbs(),
            steps=3000,
            chains=2,
            burn_in=500,
            rng=np.random.default_rng(2),
        )
        assert_recovers_both_means(posterior, gibbs_setting.exact)

    def test_each_block_adapts_to_its_own_acceptance(self, gibbs_run):
        """PCN aims at a quarter, HMC at two thirds, from the same step."""
        steps = gibbs_run.chains[0].step_sizes
        assert steps[0] != steps[1]
        rates = gibbs_run.block_acceptance_rates
        assert rates.shape == (2,)
        assert abs(rates[0] - 0.25) < 0.1
        assert abs(rates[1] - 0.65) < 0.15

    @pytest.mark.slow
    def test_the_symmetric_scan_samples_the_posterior(self, gibbs_setting):
        move = Gibbs(
            {0: MALA(step=0.2), 1: PCN(step=0.3)},
            order="symmetric",
            repeat={0: 4, 1: 4},
        )
        posterior = gibbs_sample(
            gibbs_setting,
            move,
            steps=1500,
            chains=2,
            burn_in=500,
            rng=np.random.default_rng(3),
        )
        assert_recovers_both_means(posterior, gibbs_setting.exact)

    def test_fresh_draws_are_whole_models(self, gibbs_setting, gibbs_run):
        draws = gibbs_run.samples(4, rng=np.random.default_rng(5))
        assert len(draws) == 4
        assert all(np.all(np.isfinite(gibbs_setting.record(x))) for x in draws)

    @pytest.mark.parametrize(
        "blocks, match",
        [
            ({"a": PCN(step=0.1)}, "needs a block"),
            ({"a": PCN(step=0.1), 0: PCN(step=0.1)}, "two blocks"),
        ],
        ids=["missing", "duplicate"],
    )
    def test_blocks_that_do_not_cover_the_summands_once_are_refused(
        self, gibbs_setting, blocks, match
    ):
        with pytest.raises(ValueError, match=match):
            Gibbs(blocks).kernel(gibbs_setting.target())

    @pytest.mark.parametrize(
        "blocks, keywords, error, match",
        [
            ({"a": PCN, "b": PCN(step=0.1)}, {}, TypeError, "move with one step"),
            (
                {"a": Gibbs({"x": PCN(step=0.1)}), "b": PCN(step=0.1)},
                {},
                TypeError,
                "move with one step",
            ),
            (
                {"a": MALA(step=0.1, local=True), "b": PCN(step=0.1)},
                {},
                ValueError,
                "conditional",
            ),
            (
                {"a": PCN(step=0.1), "b": PCN(step=0.1)},
                {"order": "random"},
                ValueError,
                "scan",
            ),
            (
                {"a": PCN(step=0.1), "b": PCN(step=0.1)},
                {"repeat": {"a": 0}},
                ValueError,
                "at least one",
            ),
        ],
        ids=["not-a-move", "nested", "local", "unknown-order", "no-repeat"],
    )
    def test_bad_blocks_are_refused(self, blocks, keywords, error, match):
        with pytest.raises(error, match=match):
            Gibbs(blocks, **keywords)

    def test_a_model_off_a_direct_sum_is_refused(self, setting):
        with pytest.raises(ValueError, match="direct sum"):
            Gibbs({0: PCN(step=0.1)}).kernel(setting.target())

    @pytest.mark.parametrize(
        "blocks, repeat, order, moved",
        [
            (
                {"a": PCN(step=0.05), "b": HMC(step=0.1, length=3)},
                {"a": 3, "b": 2},
                "systematic",
                0.5,
            ),
            ({0: MALA(step=0.05), 1: PCN(step=0.05)}, None, "symmetric", 0.25),
        ],
        ids=["systematic", "symmetric"],
    )
    def test_one_sweep_from_the_posterior_stays_there(
        self, gibbs_setting, blocks, repeat, order, moved
    ):
        """From exact posterior draws, one sweep changes neither the mean nor
        the variance: invariance with no autocorrelation to estimate. The
        steps are small, so many blocks move."""
        s = gibbs_setting
        kernel = Gibbs(blocks, order=order, repeat=repeat).kernel(s.target())
        rng = np.random.default_rng(11)
        before, after = [], []
        for _ in range(2000):
            x = s.posterior.sample(rng=rng)
            state = kernel.step(kernel.initial_state(x, rng=rng), rng=rng).state
            before.append(s.record(x))
            after.append(s.record(state.position))
        before, after = np.array(before), np.array(after)
        difference = after - before
        assert np.mean(np.abs(difference) > 0) > moved
        z = difference.mean(axis=0) / (difference.std(axis=0) / np.sqrt(len(after)))
        assert np.all(np.abs(z) < 4.0)
        assert np.allclose(after.var(axis=0) / before.var(axis=0), 1.0, atol=0.06)


class MeshSetting:
    """Attenuation along arcs of a circle, ``d_i == exp(-(a_i, m))``, on grids
    of any size, with one data set made on the finest and shared by all."""

    arcs = [
        (np.array([a]), np.array([a + 1.2]))
        for a in np.linspace(0.0, 2 * np.pi, 7)[:-1]
    ]
    sigma = 0.05
    levels = (32, 128, 512)

    def __init__(self):
        rng = np.random.default_rng(38)
        _, prior, problem = self.build(self.levels[-1])
        truth = prior.sample(rng=rng)
        self.data = problem.forward_operator(truth) + self.sigma * rng.standard_normal(
            6
        )
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
        "move",
        [
            PCN(step=0.005, acceptance=None),
            MALA(step=0.005, acceptance=None),
            HMC(step=0.08, length=3, acceptance=None),
            MALA(step=1.0, acceptance=None, local=True),
        ],
        ids=["pcn", "mala", "hmc", "local-mala"],
    )
    def test_the_acceptance_rate_is_flat_under_refinement(self, mesh_setting, move):
        rates = []
        for space, prior, problem, start in mesh_setting.grids.values():
            reference = None
            if move.is_local:
                reference = LaplaceApproximation.from_problem(
                    problem, prior, gauss_newton=True
                )
            sampler = Chains(
                move, steps=400, chains=2, burn_in=100, rng=np.random.default_rng(1)
            )
            posterior = MonteCarloEstimator(
                problem, prior, sampler=sampler, reference=reference
            )(mesh_setting.data, start=start)
            rates.append(
                mean_acceptance(np.stack([c.acceptance for c in posterior.chains]))
            )
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
            sampler = SMC(
                PCN(step=0.005),
                particles=300,
                mutation_steps=10,
                rng=np.random.default_rng(0),
            )
            ensemble = MonteCarloEstimator(problem, prior, sampler=sampler)(
                mesh_setting.data
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
                proposal = space.axpy(
                    step, prior.sample(rng=generator), space.copy(current)
                )
                proposed = log_density(proposal)
                probabilities.append(np.exp(min(0.0, proposed - value)))
                if np.log(generator.random()) < proposed - value:
                    current, value = proposal, proposed
            rates.append(mean_acceptance(probabilities)[0])
        assert rates[0] > 0.15
        assert rates[-1] < 0.3 * rates[0]


def test_sampling_imports_nothing_from_inference():
    """The engine sits below the estimators: a target is a potential and a
    reference, and a sampler is called on a target, so nothing in
    ``pygeoinf.sampling`` reaches into ``pygeoinf.inference``."""
    package = pathlib.Path(gi.sampling.__file__).parent
    offenders = []
    for path in sorted(package.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                if node.level == 2 and node.module.split(".")[0] == "inference":
                    offenders.append(f"{path.name}: from ..{node.module}")
                if node.module.startswith("pygeoinf.inference"):
                    offenders.append(f"{path.name}: from {node.module}")
    assert offenders == []
