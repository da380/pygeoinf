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
    "src.problem",
    "src.common",
)
SCRIPTS = ("ladder", "run_all")
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
    found = app["run_all"].run_all(app["problem"].QUICK, outdir=outdir)
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
    assert a["growth"]["lively"] > 2.0 * a["growth"]["release"]
    assert (results["outdir"] / "A_forward_operator.png").exists()
