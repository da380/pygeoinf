"""
The planet-density mini-app keeps working.

``pygeoinf2/miniapps/planet_density`` infers the Earth's radial density from
its mass and moment of inertia on a three-layer space, under priors of
increasing strength and with the two boundary radii as further unknowns.
The quick scale of every part is run once here, each part through its own
script's entry point, and what must hold is asserted: the functionals are
the integrals they claim to be, two numbers constrain the core's and the
mantle's mean densities through the mass and leave the jump at the
core-mantle boundary alone, a tight prior about a reference the data reject
shows that rejection in every kind of answer, the evidence can weigh two
references only when the prior lets them differ, and the boundary
derivatives agree with finite differences.

Needs planetmodel, for PREM and the layered space; without it the tests skip
and say so.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("planetmodel")

MINIAPP = pathlib.Path(__file__).resolve().parents[1] / "miniapps" / "planet_density"

pytestmark = pytest.mark.slow

# Every module of the app, in an order in which each one's own imports are
# already loaded. Other mini-apps use some of the same plain names.
MODULES = (
    "reference",
    "common",
    "problem",
    "setup",
    "uninformative",
    "tight",
    "mixture",
    "boundaries",
)


def _module(name: str):
    """One of the mini-app's modules, imported by path.

    The scripts import each other by their plain names, so the directory
    goes on the path for the duration of the import; a module is registered
    before it runs so that its dataclasses can resolve their annotations.
    """
    path = MINIAPP / f"{name}.py"
    if not path.exists():
        pytest.skip(f"The mini-app is not at {MINIAPP}.")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(MINIAPP))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(MINIAPP))
    return module


@pytest.fixture(scope="module")
def app():
    """The mini-app's modules, keyed by name."""
    # A module of the same plain name left by another mini-app is not ours.
    for name in MODULES:
        sys.modules.pop(name, None)
    modules = {name: _module(name) for name in MODULES}
    yield modules
    for name in MODULES:
        sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def results(app, tmp_path_factory):
    """Every part at the quick scale, once, each through its own entry point."""
    outdir = tmp_path_factory.mktemp("planet_density")
    scale, seed = app["problem"].QUICK, 2026
    problem = app["setup"].run(scale, seed=seed, outdir=outdir)
    results = {"a": problem}
    results["b"] = app["uninformative"].run(
        scale, seed=seed, outdir=outdir, problem=problem
    )
    results["c"] = app["tight"].run(
        scale, seed=seed, outdir=outdir, problem=problem, loose=results["b"]
    )
    results["d"] = app["mixture"].run(scale, seed=seed, outdir=outdir, problem=problem)
    results["e"] = app["boundaries"].run(
        scale, seed=seed, outdir=outdir, problem=problem
    )
    return results, outdir


def test_the_functionals_are_the_integrals(app, results):
    problem = results[0]["a"]
    space = problem.space
    volume = app["problem"].volume_integral(space, np.ones_like)
    exact = space.integral_functional().derivative_components
    assert np.max(np.abs(volume - exact)) < 1e-10 * np.max(np.abs(exact))
    # The volume of the medium, and the mass and moment of a uniform density
    # against their closed forms, to the accuracy of the constant's fit to
    # the modes kept.
    one = space.project_function(lambda r: 1.0)
    radius = app["reference"].R_SURFACE
    assert volume @ space.to_components(one) == pytest.approx(
        4.0 * np.pi * radius**3 / 3.0, rel=1e-5
    )
    mass, moment = problem.forward(one)
    to_mass, to_moment = app["problem"].TO_MASS, app["problem"].TO_MOMENT
    assert mass == pytest.approx(to_mass * 4.0 * np.pi * radius**3 / 3.0, rel=1e-5)
    assert moment == pytest.approx(to_moment * 8.0 * np.pi * radius**5 / 15.0, rel=1e-5)


def test_the_reference_is_light_and_prem_fits(app, results):
    problem = results[0]["a"]
    data = app["problem"]
    # The integration settled, and starts at PREM's uppermost-mantle density.
    assert problem.reference.iterations < 100
    assert problem.reference(app["reference"].R_SURFACE) == pytest.approx(
        3.3807, abs=1e-3
    )
    # Without the transition zone's jumps the reference is several percent
    # short of the mass; the jump at 660 km makes up part of it.
    deficit = problem.reference.mass / data.MASS - 1.0
    assert -0.12 < deficit < -0.05
    assert problem.reference.mass < problem.reference_660.mass < data.MASS
    # The space's functionals agree with the profile's own integrals.
    mass, moment = problem.forward(problem.reference_model)
    assert mass * data.MASS_UNIT == pytest.approx(problem.reference.mass, rel=1e-5)
    assert moment * data.MOMENT_UNIT == pytest.approx(
        problem.reference.moment, rel=1e-5
    )
    # PREM, projected onto the quick scale's modes, fits the data, and keeps
    # its mean densities; its jump at the core-mantle boundary is rounded by
    # the mantle's few modes, which also carry PREM's other discontinuities.
    assert np.all(np.abs(problem.misfit(problem.prem_model)) < 4.0)
    projected = problem.properties(problem.prem_model)
    assert projected[:2] == pytest.approx(problem.prem_properties[:2], abs=0.01)
    assert projected[2] == pytest.approx(problem.prem_properties[2], abs=0.1)


def test_two_numbers_constrain_the_means_and_not_the_jump(results):
    problem, part = results[0]["a"], results[0]["b"]
    assert np.all(np.abs(part["misfit"]) < 1.0)
    # The mass ties the two means together.
    assert part["correlation"] < -0.99
    assert part["deviation"][1] < 0.3 * part["prior_deviation"][1]
    assert part["deviation"][2] > 0.95 * part["prior_deviation"][2]
    inside = (
        np.abs(part["center"] - problem.prem_properties) <= 1.96 * part["deviation"]
    )
    assert np.all(inside)
    # The set side: not empty, brackets that hold PREM and are wider than
    # the posterior's intervals, certified to a small gap.
    assert not part["empty"]
    low, high = part["brackets"][:, 0], part["brackets"][:, 1]
    assert np.all(low <= problem.prem_properties)
    assert np.all(problem.prem_properties <= high)
    assert np.all(high - low > 2 * 1.96 * part["deviation"])
    assert np.all(part["gaps"] < 0.01 * (high - low))


def test_a_tight_prior_about_a_rejected_reference_shows_it(app, results):
    problem, loose, part = results[0]["a"], results[0]["b"], results[0]["c"]
    # The posterior had to travel many prior deviations to meet the data.
    assert np.all(part["shift"][:2] > 5.0)
    assert part["log_bayes_factor"] > 50.0
    # Its intervals are narrower than the loose prior's, and miss PREM.
    assert np.all(part["deviation"] <= loose["deviation"])
    outside = (
        np.abs(part["center"] - problem.prem_properties) > 1.96 * part["deviation"]
    )
    assert outside[:2].all()
    # No model within the tight prior's ball fits the data.
    assert part["empty"]
    assert part["reach"] > 1.0
    # The minimum-norm change meets the discrepancy level, and its bias were
    # PREM the truth exceeds the noise it propagates.
    point = part["minimum_norm"]
    assert point["chi_squared"] == pytest.approx(
        problem.problem.critical_chi_squared(level=app["problem"].LEVEL), rel=1e-2
    )
    assert np.any(np.abs(point["bias"]) > point["propagated"])


def test_the_evidence_weighs_the_mass_not_the_jump(results):
    part = results[0]["d"]
    for strength in ("loose", "tight"):
        assert part[strength]["weights"].sum() == pytest.approx(1.0)
    # Under the loose prior the data cannot tell the references apart.
    assert np.all(np.abs(part["loose"]["weights"] - 0.5) < 0.2)
    # Under the tight one the heavier reference takes the weight, and it is
    # the one the data lie nearer to.
    assert part["tight"]["weights"][1] > 0.99
    assert part["tight"]["distances"][1] < part["tight"]["distances"][0]


def test_the_boundaries(results):
    part = results[0]["e"]
    assert part["worst_defect"] < 1e-6
    assert part["converged"]
    # The mode pushes the core-mantle boundary out, the one move that adds
    # mass cheaply; the data barely narrow either displacement.
    assert part["center"][4] > 0.0
    assert part["deviation"][3] > 0.95 * 20.0
    assert part["deviation"][4] < 20.0
    # Occam meets the discrepancy level, by moving the boundaries far beyond
    # anything the prior admits.
    assert part["occam_converged"]
    assert part["occam_chi_squared"] == pytest.approx(part["critical"], rel=1e-3)
    assert np.max(np.abs(part["occam_shift"])) > 10 * 20.0


def test_the_figures_were_written(results):
    outdir = results[1]
    for name in ("setup", "uninformative", "tight", "mixture", "boundaries"):
        assert (outdir / f"{name}.png").exists(), name
