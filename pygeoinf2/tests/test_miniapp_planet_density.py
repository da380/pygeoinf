"""
The planet-density mini-app keeps working.

``pygeoinf2/miniapps/planet_density`` infers the Earth's radial density from
its mass and moment of inertia on a space of PREM's own layers. PREM in the
space is the reference; the truth is the reference plus a draw from the
tight prior about it, and the data are the truth's. The point, measure and
set answers come under the tight prior and under a deliberately wide one,
and the two boundary radii are then further unknowns. The quick scale of
every part is run once here, each part through its own script's entry
point, and what must hold is asserted: the functionals are the integrals
they claim to be, the reference is PREM's, the data reject the reference and
fit the truth, the point answer's error is its bias and its propagated
noise, under the tight prior the truth lies inside the brackets and within
two posterior standard deviations, the wide answers are wider, the MAP
model's boundary displacements recover the truth's to within two Laplace
standard deviations, and Occam's model in two norms meets the level both
times with displacements further apart than those deviations.

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
    "answers",
    "setup",
    "tight",
    "wide",
    "radii",
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
    scale, seed = app["problem"].QUICK, app["common"].SEED
    problem = app["setup"].run(scale, seed=seed, outdir=outdir)
    results = {"a": problem}
    for key, part in (("b", "tight"), ("c", "wide"), ("d", "boundaries")):
        results[key] = app[part].run(scale, seed=seed, outdir=outdir, problem=problem)
    return results, outdir


def test_the_functionals_are_the_integrals(app, results):
    problem = results[0]["a"]
    space = problem.space
    volume = app["problem"].volume_integral(space, np.ones_like)
    exact = space.integral_functional().derivative_components
    assert np.max(np.abs(volume - exact)) < 1e-10 * np.max(np.abs(exact))
    # The volume of the medium, and the mass and moment of a uniform density
    # against their closed forms, to the accuracy of the constant's fit to
    # the basis functions kept.
    one = space.project_function(lambda r: 1.0)
    radius = app["reference"].R_SURFACE
    assert volume @ space.to_components(one) == pytest.approx(
        4.0 * np.pi * radius**3 / 3.0, rel=1e-5
    )
    mass, moment = problem.forward(one)
    to_mass, to_moment = app["problem"].TO_MASS, app["problem"].TO_MOMENT
    assert mass == pytest.approx(to_mass * 4.0 * np.pi * radius**3 / 3.0, rel=1e-5)
    assert moment == pytest.approx(to_moment * 8.0 * np.pi * radius**5 / 15.0, rel=1e-5)


def test_the_reference_is_prem_and_the_data_the_truths(app, results):
    problem = results[0]["a"]
    data = app["problem"]
    # The reference holds PREM's mass, moment and properties: every layer
    # but the crust holds PREM's density to the truncation's accuracy.
    own = app["reference"].prem_summary()
    mass, moment = problem.forward(problem.reference)
    assert mass * data.MASS_UNIT == pytest.approx(own["mass"], rel=1e-4)
    assert moment * data.MOMENT_UNIT == pytest.approx(own["moment"], rel=1e-4)
    expected = [own["core"], own["mantle"], own["jump"]]
    assert problem.properties(problem.reference) == pytest.approx(expected, abs=0.02)
    # The data fit the truth and reject the reference.
    critical = problem.problem.critical_chi_squared(level=data.LEVEL)
    assert problem.problem.chi_squared(problem.truth, problem.data) < critical
    assert problem.problem.chi_squared(problem.reference, problem.data) > critical


def test_the_point_answer(results):
    problem, part = results[0]["a"], results[0]["b"]["point"]
    # The reference is rejected, so the smallest change to it moves, to the
    # discrepancy level.
    assert not part["saturated"]
    critical = problem.problem.critical_chi_squared(level=0.95)
    assert part["chi_squared"] == pytest.approx(critical, rel=1e-3)
    reference = problem.properties(problem.reference)
    assert np.max(np.abs(part["estimate"] - reference)) > 0.005
    # Its error is its bias at the truth and the noise it propagates, and
    # the bias is the larger part of it.
    error = part["estimate"] - problem.true_properties
    assert np.all(np.abs(error - part["bias"]) < 4.0 * part["propagated"])
    assert np.any(np.abs(part["bias"]) > 3.0 * part["propagated"])


def test_the_tight_prior_holds_the_truth(results):
    problem, part = results[0]["a"], results[0]["b"]
    measure, sets = part["measure"], part["set"]
    truth = problem.true_properties
    assert np.all(np.abs(measure["center"] - truth) <= 2.0 * measure["deviation"])
    assert np.all(measure["deviation"] <= measure["prior_deviation"] * (1 + 1e-9))
    # The mass ties the two means together, and narrows the mantle's most.
    assert measure["correlation"] < -0.5
    assert measure["deviation"][1] < 0.3 * measure["prior_deviation"][1]
    assert sets["truth_in_ball"]
    low, high = sets["brackets"][:, 0], sets["brackets"][:, 1]
    assert np.all((low <= truth) & (truth <= high))
    # The brackets lie within what the ball admits alone, certified to a
    # small gap.
    assert np.all(sets["bounds"][:, 0] <= low + 1e-9)
    assert np.all(high <= sets["bounds"][:, 1] + 1e-9)
    assert np.all(sets["gaps"] < 0.01 * (high - low))


def test_the_wide_answers_are_wider(results):
    problem, tight, wide = results[0]["a"], results[0]["b"], results[0]["c"]
    truth = problem.true_properties
    # The point is the same: the minimum-norm change sees the prior's shape,
    # not its size.
    for key in ("estimate", "bias", "propagated"):
        assert np.allclose(wide["point"][key], tight["point"][key], atol=1e-4)
    measure = wide["measure"]
    assert np.all(measure["deviation"] > tight["measure"]["deviation"])
    assert np.all(np.abs(measure["center"] - truth) <= 2.0 * measure["deviation"])
    # Two numbers cut a slice: the means are tied along the line of
    # constant mass, and the jump at the core-mantle boundary is untouched.
    assert measure["correlation"] < -0.99
    assert measure["deviation"][2] > 0.95 * measure["prior_deviation"][2]
    widths = np.diff(wide["set"]["brackets"], axis=1)
    assert np.all(widths > np.diff(tight["set"]["brackets"], axis=1))
    low, high = wide["set"]["brackets"][:, 0], wide["set"]["brackets"][:, 1]
    assert np.all((low <= truth) & (truth <= high))
    # The data prefer the prior the truth was drawn from.
    assert wide["log_bayes_factor"] > 0.0


def test_the_boundaries(results):
    part = results[0]["d"]
    assert part["worst_defect"] < 1e-6
    assert part["converged"]
    # The MAP model's displacements recover the truth's to within two
    # Laplace standard deviations, which the data narrow at the core-mantle
    # boundary and leave at the prior's at the inner-core boundary.
    shifts, deviations = part["center"][3:], part["deviation"][3:]
    assert np.all(np.abs(shifts - part["truth"][3:]) < 2.0 * deviations)
    assert deviations[1] < 0.9 * part["prior_deviation"][4]
    assert deviations[0] > 0.95 * part["prior_deviation"][3]
    # Occam's model meets the level in the model space's norm and in the
    # prior's, each is the smaller change in its own norm, and the two put
    # the boundaries further apart than the data can place either.
    space, prior = part["occam"]["space"], part["occam"]["prior"]
    for answer in (space, prior):
        assert answer["converged"]
        assert answer["chi_squared"] == pytest.approx(part["critical"], rel=1e-3)
    assert space["space_norm"] < prior["space_norm"]
    assert prior["prior_norm"] < space["prior_norm"]
    assert np.all(np.abs(space["shifts"] - prior["shifts"]) > deviations)


def test_the_figures_were_written(results):
    outdir = results[1]
    names = ["setup", "boundaries"] + [
        f"{prior}_{answer}"
        for prior in ("tight", "wide")
        for answer in ("point", "measure", "set")
    ]
    for name in names:
        assert (outdir / f"{name}.png").exists(), name
