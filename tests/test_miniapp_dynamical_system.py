"""
The dynamical-system mini-app keeps working.

``miniapps/dynamical_system`` builds the forward operator of a parameter
estimation problem from four pieces: a parameter map, a solution operator
that integrates a rule and differentiates the differential equation, a
lift of the integrator's table to functions of time, and an observation
operator. The rule's derivatives, the three routes to the solution's
derivative, the lift and the whole operator are tested on a short record;
then the part runs once at the quick scale through its script's entry
point, and what it shows is asserted (marked slow).
"""

from __future__ import annotations

import dataclasses
import importlib
import pathlib
import sys

import numpy as np
import pytest

from pygeoinf.testing import (
    check_derivative,
    check_operator,
    check_second_derivative,
)

MINIAPP = pathlib.Path(__file__).resolve().parents[1] / "miniapps" / "dynamical_system"
# The app's modules, in the package ``src``, and its part scripts, which
# import each other and the package by their plain names. Other mini-apps
# use the same names.
MODULES = (
    "src.system",
    "src.pendulum",
    "src.trajectory",
    "src.spaces",
    "src.parameters",
    "src.observation",
    "src.sampling",
    "src.problem",
    "src.common",
)
SCRIPTS = ("ladder", "point", "bayesian", "forcing", "run_all")
ADJOINT = "the adjoint satisfies (A x, y) == (x, A* y)"


def _forget() -> None:
    """Drop the app's names, and any other app's of the same names."""
    for name in list(sys.modules):
        if name == "src" or name.startswith("src.") or name in SCRIPTS:
            del sys.modules[name]


def _module(name: str):
    """One of the mini-app's modules or scripts, imported by name with the
    app's directory on the path for the duration."""
    sys.path.insert(0, str(MINIAPP))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(MINIAPP))


@pytest.fixture(scope="module")
def app():
    """The mini-app's modules and scripts, keyed by their last names."""
    if not (MINIAPP / "src" / "trajectory.py").exists():
        pytest.skip(f"The mini-app is not at {MINIAPP}.")
    _forget()
    modules = {name.split(".")[-1]: _module(name) for name in MODULES + SCRIPTS}
    yield modules
    _forget()


@pytest.fixture(scope="module")
def small(app):
    """A short record on a coarse mesh, from a gentle release."""
    scale = dataclasses.replace(
        app["problem"].QUICK, duration=1.5, element_length=0.15, n_times=6
    )
    return app["problem"].build(scale)


PARAMETERS = np.array([1.0, 0.8, 1.0, 0.7])


class TestTheRule:
    def test_the_complex_step_jacobians_agree_with_differences(self, app, rng):
        system = app["pendulum"].double_pendulum()
        state = rng.normal(size=4)
        in_state, in_parameters = system.jacobians(state, PARAMETERS)
        step = 1e-6
        for k in range(4):
            ahead, behind = state.copy(), state.copy()
            ahead[k] += step
            behind[k] -= step
            difference = (
                system.rate(ahead, PARAMETERS) - system.rate(behind, PARAMETERS)
            ) / (2.0 * step)
            assert np.allclose(in_state[:, k], difference, rtol=1e-7, atol=1e-8)
            ahead, behind = PARAMETERS.copy(), PARAMETERS.copy()
            ahead[k] += step
            behind[k] -= step
            difference = (system.rate(state, ahead) - system.rate(state, behind)) / (
                2.0 * step
            )
            assert np.allclose(in_parameters[:, k], difference, rtol=1e-7, atol=1e-8)

    def test_the_second_derivatives_agree_with_differences_of_the_first(self, app, rng):
        system = app["pendulum"].double_pendulum()
        state, dx, dp = rng.normal(size=4), rng.normal(size=4), 0.1 * rng.normal(size=4)
        found = system.second_directional(state, PARAMETERS, dx, dp)
        step = 1e-4
        ahead = system.jacobians(state + step * dx, PARAMETERS + step * dp)
        behind = system.jacobians(state - step * dx, PARAMETERS - step * dp)
        for a, b, c in zip(found, ahead, behind):
            assert np.allclose(a, (b - c) / (2.0 * step), rtol=1e-5, atol=1e-6)

    def test_the_unforced_motion_keeps_its_energy(self, app, small):
        """An independent check of the rule: nothing in it was told the energy."""
        state = small.solver.solve(PARAMETERS)
        energies = app["pendulum"].energy(state(small.grid.nodes), PARAMETERS)
        assert np.ptp(energies) <= 1e-6 * abs(energies[0])

    def test_the_masses_enter_through_their_ratio_alone(self, small):
        scaled = PARAMETERS * np.array([3.0, 3.0, 1.0, 1.0])
        first = small.solver.sample(small.solver.solve(PARAMETERS))
        second = small.solver.sample(small.solver.solve(scaled))
        assert np.abs(first - second).max() <= 1e-6
        longer = PARAMETERS * np.array([1.0, 1.0, 1.1, 1.0])
        third = small.solver.sample(small.solver.solve(longer))
        assert np.abs(first - third).max() >= 1e-2


class TestTheSolution:
    def test_three_routes_to_one_derivative(self, small, rng):
        """The tangent equation, the sensitivities, and finite differences."""
        solver = small.solver
        state = solver.solve(PARAMETERS)
        change = 0.1 * rng.normal(size=4)
        tangent = solver.sample(solver.tangent(state, PARAMETERS, change))
        jacobian = solver.sensitivities(state, PARAMETERS)
        step = 1e-5
        difference = (
            solver.sample(solver.solve(PARAMETERS + step * change))
            - solver.sample(solver.solve(PARAMETERS - step * change))
        ) / (2.0 * step)
        size = np.abs(tangent).max()
        assert np.abs(jacobian @ change - tangent).max() <= 1e-6 * size
        assert np.abs(difference - tangent).max() <= 1e-5 * size

    def test_the_adjoint_pairs_as_the_tangent_does(self, small, rng):
        """With a weight at every node, so every node is a stop."""
        solver = small.solver
        state = solver.solve(PARAMETERS)
        change = rng.normal(size=4)
        weights = rng.normal(size=(4, small.grid.nodes.size))
        tangent = solver.sample(solver.tangent(state, PARAMETERS, change))
        in_parameters, in_forcing = solver.adjoint(
            state, PARAMETERS, weights, forcing=False
        )
        assert in_forcing is None
        assert change @ in_parameters == pytest.approx(
            np.sum(weights * tangent), rel=1e-6
        )
        # The control: a weight left out is a pairing that fails.
        weights[:, 5] = 0.0
        short = solver.adjoint(state, PARAMETERS, weights, forcing=False)[0]
        assert abs(change @ (short - in_parameters)) > 1e-6 * abs(
            change @ in_parameters
        )

    def test_the_forcing_s_adjoint_pairs_as_the_tangent_does(self, small):
        """Weights at every node, and the weights a set of readings hands back."""
        solver, nodes = small.forced_solver, small.grid.nodes
        forcing = np.stack([np.sin(3.0 * nodes), np.cos(2.0 * nodes)])
        change = np.stack([np.cos(4.0 * nodes), np.sin(5.0 * nodes)])
        state = solver.solve(PARAMETERS, forcing)
        tangent = solver.sample(
            solver.tangent(state, PARAMETERS, 0.0 * PARAMETERS, change)
        )
        dense = np.stack([np.sin((k + 1.0) * nodes) for k in range(4)])
        sparse = small.lift.adjoint(small.observe.adjoint(small.data))
        assert np.mean(np.abs(sparse) > 1e-12 * np.abs(sparse).max()) < 0.6
        for weights in (dense, sparse):
            in_forcing = solver.adjoint(state, PARAMETERS, weights)[1]
            pairing = np.sum(weights * tangent)
            assert np.sum(change * in_forcing) == pytest.approx(pairing, rel=1e-5)
        # The control: one node's share left out is a pairing that fails.
        in_forcing[:, 7] = 0.0
        assert abs(np.sum(change * in_forcing) - pairing) > 1e-4 * abs(pairing)

    def test_the_second_derivative_and_its_adjoint(self, small, rng):
        """Against differences of the first tangent, then the pairing."""
        solver = small.solver
        state = solver.solve(PARAMETERS)
        change, other = 0.3 * rng.normal(size=4), 0.3 * rng.normal(size=4)
        first = solver.tangent(state, PARAMETERS, change)
        bent = solver.sample(
            solver.second_tangent(state, first, PARAMETERS, change, other)
        )
        step = 1e-5

        def tangent_at(parameters):
            return solver.sample(
                solver.tangent(solver.solve(parameters), parameters, other)
            )

        difference = (
            tangent_at(PARAMETERS + step * change)
            - tangent_at(PARAMETERS - step * change)
        ) / (2.0 * step)
        assert np.abs(difference - bent).max() <= 1e-4 * np.abs(bent).max()
        weights = rng.normal(size=bent.shape)
        in_parameters, _ = solver.second_adjoint(
            state, first, PARAMETERS, change, weights, forcing=False
        )
        assert other @ in_parameters == pytest.approx(np.sum(weights * bent), rel=1e-5)

    def test_a_table_of_jacobians_gives_the_same_derivative_less_exactly(
        self, app, small
    ):
        """Two routes to the Jacobians along the state: the rule, and a table of it."""
        scale = dataclasses.replace(small.scale, table=16)
        tabled = app["problem"].build(scale)
        exact = small.forward_by_adjoint.at(small.truth).derivative.matrix(
            form="components"
        )
        found = tabled.forward_by_adjoint.at(tabled.truth).derivative.matrix(
            form="components"
        )
        defect = np.abs(found - exact).max() / np.abs(exact).max()
        assert 0.0 < defect <= 1e-4
        # The table is made once for a state: one call of the rule serves both.
        before = tabled.system.calls
        state = tabled.solver.solve(PARAMETERS)
        solves = tabled.system.calls - before
        tabled.solver.adjoint(
            state, PARAMETERS, np.ones((4, tabled.grid.nodes.size)), forcing=False
        )
        tabled.solver.tangent(state, PARAMETERS, np.ones(4))
        assert tabled.system.calls - before - solves == 1

    def test_the_sensitivity_route_refuses_a_forcing(self, app, small):
        with pytest.raises(ValueError, match="without forcing"):
            app["trajectory"].solution_operator(
                small.forced_solver, forced=True, route="sensitivity"
            )


class TestTheSpaces:
    def test_a_table_is_the_function_s_nodal_values(self, app, small, rng):
        space = small.space
        up, down = app["spaces"].from_table(space), app["spaces"].to_table(space)
        table = rng.normal(size=small.grid.nodes.size)
        assert np.allclose(down(up(table)), table, atol=1e-10)
        assert np.allclose(
            space.evaluate(up(table), list(small.grid.nodes)), table, atol=1e-10
        )
        for operator in (up, down, small.lift, small.observe):
            assert check_operator(operator, rng=rng).worst <= 1e-9

    def test_an_observation_is_the_trajectory_at_its_times(self, small):
        state = small.solver.solve(PARAMETERS)
        records = small.observe(small.lift(small.solver.sample(state)))
        for record, index, times in zip(records, small.scale.observed, small.times):
            assert np.allclose(record, state(times)[index], atol=5e-3)

    def test_the_parameter_map(self, app, rng):
        constraints = ["positive", (0.5, 2.0), "free", "positive"]
        mapping = app["parameters"].ParameterMap(constraints)
        point = rng.normal(size=4)
        parameters = mapping.parameters(point)
        assert parameters[0] > 0.0 and 0.5 < parameters[1] < 2.0 and parameters[3] > 0.0
        assert parameters[2] == point[2]
        assert np.allclose(mapping.coordinates(parameters), point)
        operator = mapping.operator()
        check_derivative(operator, point, rng=rng)
        check_second_derivative(operator, point, rng=rng)
        with pytest.raises(ValueError, match="constraint"):
            app["parameters"].ParameterMap(["negative"])


class TestTheInstruments:
    def test_a_combination_of_components_is_read_as_one(self, app, small):
        times = [0.2, 0.7, 1.1]
        observation = app["observation"]
        both = observation.point_readings(small.trajectory, [(0, times), (1, times)])
        gap = observation.point_readings(
            small.trajectory, [([1.0, -1.0, 0.0, 0.0], times)]
        )
        trajectory = small.lift(small.solver.sample(small.solver.solve(PARAMETERS)))
        first, second = both(trajectory)
        assert np.allclose(gap(trajectory)[0], first - second, atol=1e-12)

    def test_a_window_s_mean_is_the_integral_over_its_length(self, app, small, rng):
        observation = app["observation"]
        windows = [(0.1, 0.4), (0.5, 1.3)]
        averages = observation.window_averages(small.trajectory, [(1, windows)])
        nodes = small.grid.nodes
        table = np.zeros((4, nodes.size))
        table[1] = np.sin(2.0 * nodes)
        found = averages(small.lift(table))[0]
        exact = [
            (np.cos(2.0 * a) - np.cos(2.0 * b)) / (2.0 * (b - a)) for a, b in windows
        ]
        assert np.allclose(found, exact, atol=1e-4)
        assert check_operator(averages, rng=rng).worst <= 1e-9

    def test_a_function_of_the_state_and_the_parameters(self, app, small, rng):
        observation, pendulum = app["observation"], app["pendulum"]
        times = [0.3, 0.6, 1.2]
        at_times = observation.states_at(small.trajectory, times)
        position = observation.pointwise_observation(
            pendulum.lower_bob, at_times.codomain, small.parameter_map.space, 2
        )
        state = small.solver.solve(PARAMETERS)
        states = at_times(small.lift(small.solver.sample(state)))
        found = position((states, PARAMETERS))
        exact = pendulum.lower_bob(state(np.array(times)), PARAMETERS[:, None])
        assert np.allclose(found, exact, atol=5e-3)
        point = (states, PARAMETERS)
        check_derivative(position, point, rng=rng)
        check_second_derivative(position, point, rng=rng)
        assert check_operator(position.derivative(point), rng=rng).worst <= 1e-12

    @pytest.mark.parametrize("instrument", ["averages", "bob"])
    def test_each_instrument_completes_the_forward_operator(self, app, instrument, rng):
        scale = dataclasses.replace(
            app["problem"].QUICK,
            duration=1.5,
            element_length=0.15,
            n_times=5,
            instrument=instrument,
        )
        problem = app["problem"].build(scale)
        assert problem.forward.has_second_derivative
        assert (problem.observe is None) == (instrument == "bob")
        check_derivative(problem.forward, problem.truth, rng=rng, rtol=1e-4)
        derivative = problem.forward_by_adjoint.at(problem.truth).derivative
        assert check_operator(derivative, rng=rng, rtol=1e-5).defects[ADJOINT] <= 1e-6
        point = (problem.truth, problem.forcing_space.zero())
        space = problem.data_space
        unforced = problem.forward(problem.truth)
        assert (
            space.norm(space.subtract(problem.forced_forward(point), unforced)) <= 1e-6
        )

    def test_an_unknown_instrument_is_refused(self, app):
        scale = dataclasses.replace(app["problem"].QUICK, instrument="sonar")
        with pytest.raises(ValueError, match="instrument"):
            app["problem"].build(scale)


class TestTheForwardOperator:
    def test_the_two_routes_agree_and_pass_the_checks(self, small, rng):
        truth = small.truth
        first = small.forward.at(truth).derivative
        second = small.forward_by_adjoint.at(truth).derivative
        check_derivative(small.forward, truth, rng=rng, rtol=1e-4)
        check_derivative(small.forward_by_adjoint, truth, rng=rng, rtol=1e-4)
        assert check_operator(first, rng=rng).defects[ADJOINT] <= 1e-12
        assert check_operator(second, rng=rng, rtol=1e-5).defects[ADJOINT] <= 1e-6
        a, b = first.matrix(form="components"), second.matrix(form="components")
        assert np.abs(a - b).max() <= 1e-6 * np.abs(a).max()

    def test_the_data_are_the_clean_readings_plus_noise(self, small):
        space = small.observe.codomain
        misfit = space.norm(space.subtract(small.data, small.clean))
        expected = small.scale.noise_std * np.sqrt(space.dim)
        assert 0.3 * expected <= misfit <= 2.0 * expected

    def test_with_a_forcing(self, small, rng):
        point = (small.truth, small.forcing_prior.sample(rng=rng))
        still = (small.truth, small.forcing_space.zero())
        unforced = small.forward(small.truth)
        space = small.observe.codomain
        # No forcing is the unforced problem; a forcing moves the readings.
        assert space.norm(space.subtract(small.forced_forward(still), unforced)) <= 1e-6
        assert space.norm(space.subtract(small.forced_forward(point), unforced)) >= 1e-3
        derivative = small.forced_forward.at(point).derivative
        gradient = derivative.adjoint(small.data)
        assert (
            gradient[0].shape == (4,) and gradient[1].shape == small.forcing_space.shape
        )


class TestTheSamplers:
    def test_a_scale_names_its_move(self, app):
        sampling, quick = app["sampling"], app["problem"].QUICK
        move, step = sampling.move_for(quick, 2.0)
        assert type(move).__name__ == "PCN" and step == 2.0
        named = dataclasses.replace(
            quick, sampler="hmc", sampler_step=0.2, sampler_length=3
        )
        move, step = sampling.move_for(named, 2.0)
        assert type(move).__name__ == "HMC" and move.length == 3 and step == 0.2
        for name, local in (("mala", False), ("local", True)):
            move, step = sampling.move_for(
                dataclasses.replace(quick, sampler=name), 2.0
            )
            assert type(move).__name__ == "MALA" and move.is_local == local
            assert step == sampling.STEPS[name]
        with pytest.raises(ValueError, match="sampler"):
            sampling.move_for(dataclasses.replace(quick, sampler="gibbs"), 2.0)

    @pytest.mark.parametrize("name", ["pcn", "mala", "local"])
    def test_each_move_samples_about_the_laplace_gaussian(self, app, small, name):
        """A few steps of each: the chains start in the posterior and stay there."""
        from pygeoinf import ForwardProblem, LaplaceApproximation, LinearOperator

        scale = dataclasses.replace(
            small.scale, sampler=name, chains=2, burn_in=2, steps=12
        )
        problem = ForwardProblem(small.forward, error=small.noise)
        prior = small.parameter_prior
        laplace = LaplaceApproximation.from_problem(problem, prior, gauss_newton=True)
        result = laplace.solve(small.data)
        posterior, line = app["sampling"].sample(
            problem,
            prior,
            small.data,
            scale,
            at_mode=laplace,
            start=result.measure,
            record=LinearOperator.identity(prior.domain),
            record_names=list(small.parameter_names),
            pcn_step=2.0,
            rng=np.random.default_rng(3),
        )
        assert line.startswith(name) and posterior.acceptance_rate >= 0.3
        values = posterior.scalars()[1:].reshape(4, -1)
        std = np.sqrt(np.diag(result.measure.covariance.matrix(form="components")))
        assert np.all(np.abs(values.mean(axis=1) - result.model) <= 3.0 * std)


def test_every_field_of_the_scale_is_an_option(app, tmp_path):
    arguments = app["common"].parse(
        "description",
        quick=app["problem"].QUICK,
        full=app["problem"].FULL,
        argv=[
            "--quick",
            "--duration",
            "2.5",
            "--release",
            "0.4",
            "0.1",
            "--observed",
            "1",
            "--outdir",
            str(tmp_path),
        ],
    )
    scale = arguments.scale
    assert scale.name == "quick" and scale.duration == 2.5
    assert scale.release == (0.4, 0.1) and scale.observed == (1,)
    assert scale.n_times == app["problem"].QUICK.n_times
    problem = app["problem"].build(scale)
    assert len(problem.data) == 1


@pytest.fixture(scope="module")
def results(app, tmp_path_factory):
    """Every part at the quick scale, through the run-all script."""
    outdir = tmp_path_factory.mktemp("dynamical_system")
    # Short chains: the samplers are checked for agreement, not for precision.
    scale = dataclasses.replace(app["problem"].QUICK, chains=2, burn_in=20, steps=100)
    found = app["run_all"].run_all(scale, outdir=outdir)
    found["outdir"] = outdir
    return found


@pytest.mark.slow
def test_a_the_ladder(results):
    a = results["a"]
    assert a["representation"]["release"][0] <= 1e-3
    assert all(defect <= 1e-9 for defect in a["exact"].values())
    slope, adjoint, _ = a["routes"]["sensitivities"]
    assert slope <= 1e-4 and adjoint <= 1e-12
    slope, adjoint, _ = a["routes"]["adjoint equation"]
    assert slope <= 1e-4 and adjoint <= 1e-6
    assert a["agreement"] <= 1e-6
    assert a["forced"][0] <= 1e-4 and a["forced"][1] <= 1e-5
    for bend, adjoint, _ in a["curvature"].values():
        assert bend <= 1e-2 and adjoint <= 1e-4
    assert a["growth"]["lively"] > 2.0 * a["growth"]["release"]
    assert (results["outdir"] / "A_forward_operator.png").exists()


@pytest.mark.slow
def test_b_the_point_answer(results):
    b = results["b"]
    problem = results["problem"]
    exact, gauss = b["searches"]["exact Hessian"], b["searches"]["Gauss-Newton"]
    assert exact["converged"] and gauss["converged"]
    assert b["apart"] <= 1e-5
    size = problem.data_space.dim
    assert 0.4 * size <= exact["chi_squared"] <= 2.5 * size
    truth = np.array(problem.scale.true_parameters)
    # The lengths and the masses' ratio are found; the masses' size is the prior's.
    assert np.allclose(exact["parameters"][2:], truth[2:], rtol=2e-2)
    ratio = exact["parameters"][1] / exact["parameters"][0]
    assert ratio == pytest.approx(truth[1] / truth[0], rel=3e-2)
    assert exact["parameters"][0] * exact["parameters"][1] == pytest.approx(
        1.0, abs=1e-4
    )
    assert b["moved"]["the masses' common size"] <= 1e-4
    assert abs(b["bias"][0]) > 5e-2 and abs(b["bias"][2]) < 1e-3
    assert b["valleys"]["release"] == 1
    assert (results["outdir"] / "B_point.png").exists()


@pytest.mark.slow
def test_c_the_measure_answer(results):
    c = results["c"]
    prior_std = results["problem"].scale.prior_std
    exact, gauss = c["laplace"]["exact Hessian"], c["laplace"]["Gauss-Newton"]
    assert np.allclose(exact["std"], gauss["std"], rtol=5e-2)
    # Along the masses' common size the posterior is the prior.
    assert c["directions"]["the masses' common size"] == pytest.approx(
        prior_std, rel=1e-3
    )
    assert c["directions"]["their ratio"] <= 0.1 * prior_std
    assert np.all(exact["std"][2:] <= 0.05 * prior_std)
    assert np.all(c["inside"] <= 4.0)
    sampled = c["sampled"]
    assert sampled["acceptance"] >= 0.5
    assert np.all(sampled["rhat"] <= 1.1)
    assert np.allclose(sampled["std"], exact["std"], rtol=0.25)
    assert np.all(np.abs(sampled["mean"] - exact["mean"]) <= 0.3 * exact["std"])
    for name in ("C_laplace.png", "C_sampling.png"):
        assert (results["outdir"] / name).exists()


@pytest.mark.slow
def test_d_the_forcing(results):
    d = results["d"]
    assert d["shift"] > d["noise"]
    ignored, unknown, sampled = d["ignored"], d["unknown"], d["sampled"]
    # Ignored, the forcing leaves the lengths confidently wrong.
    assert np.all(ignored["away"][2:] > 3.0)
    assert ignored["chi_squared"] > 1.5 * unknown["chi_squared"]
    # Unknown, it widens their posterior and the truth is inside it.
    assert np.all(unknown["std"][2:] > 3.0 * ignored["std"][2:])
    assert np.all(unknown["away"] <= 3.0)
    assert sampled["acceptance"] >= 0.3
    assert np.allclose(sampled["std"], unknown["std"], rtol=0.5)
    assert d["forcing"]["covered"] >= 0.7
    for name in ("ignored", "unknown", "sampled"):
        assert (results["outdir"] / f"D_forcing_{name}.png").exists()
