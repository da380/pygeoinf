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
standard deviations, and Occam's model meets the level in two norms, each
answer the smaller change in its own norm.

Needs planetmodel, for PREM and the layered space; without it the tests skip
and say so.
"""

from __future__ import annotations

import dataclasses
import importlib
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("planetmodel")

MINIAPP = pathlib.Path(__file__).resolve().parents[1] / "miniapps" / "planet_density"

pytestmark = pytest.mark.slow

# The app's modules, in the package ``src``, and its part scripts, which
# import the package by its plain name. Other mini-apps use the same names.
MODULES = ("src.reference", "src.common", "src.problem", "src.answers", "src.radii")
SCRIPTS = ("setup", "tight", "wide", "boundaries")


def _forget() -> None:
    """Drop the app's names, and any other app's of the same names."""
    for name in list(sys.modules):
        if name == "src" or name.startswith("src.") or name in SCRIPTS:
            del sys.modules[name]


def _module(name: str):
    """One of the mini-app's modules or scripts, imported by name with the
    app's directory on the path for the duration, as a script run there
    would find it."""
    if not (MINIAPP / f"{name.replace('.', '/')}.py").exists():
        pytest.skip(f"The mini-app is not at {MINIAPP}.")
    sys.path.insert(0, str(MINIAPP))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(MINIAPP))


@pytest.fixture(scope="module")
def app():
    """The mini-app's modules and scripts, keyed by their last names."""
    _forget()
    modules = {name.split(".")[-1]: _module(name) for name in MODULES + SCRIPTS}
    yield modules
    _forget()


@pytest.fixture(scope="module")
def scale(app, tmp_path_factory):
    """The quick preset, built through the scripts' own command line with
    options given at their preset values."""
    problem = app["problem"]
    quick = problem.QUICK
    arguments = app["common"].parse(
        "the quick scale",
        quick=quick,
        full=problem.FULL,
        argv=[
            "--quick",
            "--true-shifts",
            *map(str, quick.true_shifts),
            "--basis-functions",
            *map(str, quick.basis_functions),
            "--outdir",
            str(tmp_path_factory.mktemp("planet_density_options")),
        ],
    )
    assert arguments.scale == quick
    return arguments.scale


@pytest.fixture(scope="module")
def results(app, scale, tmp_path_factory):
    """Every part at the quick scale, once, each through its own entry point."""
    outdir = tmp_path_factory.mktemp("planet_density")
    problem = app["setup"].run(scale, outdir=outdir)
    results = {"a": problem}
    for key, part in (("b", "tight"), ("c", "wide"), ("d", "boundaries")):
        results[key] = app[part].run(scale, outdir=outdir, problem=problem)
    return results, outdir


def test_every_field_of_the_scale_is_an_option(app, tmp_path):
    problem = app["problem"]
    arguments = app["common"].parse(
        "",
        quick=problem.QUICK,
        full=problem.FULL,
        argv=[
            "--wide-std",
            "1",
            "--true-shifts",
            "-20",
            "-60",
            "--outdir",
            str(tmp_path),
        ],
    )
    assert arguments.scale == dataclasses.replace(
        problem.FULL, wide_std=1.0, true_shifts=(-20.0, -60.0)
    )


def test_the_functionals_are_the_integrals(app, results):
    problem = results[0]["a"]
    space = problem.space
    volume = app["problem"].volume_integral(space, np.ones_like)
    exact = space.integral_functional().derivative_components
    assert np.max(np.abs(volume - exact)) < 1e-10 * np.max(np.abs(exact))
    # The volume of the medium, a ball of radius one in Earth radii, and the
    # mass and moment of a uniform density of 1 g/cm^3, 1000 kg/m^3, in SI
    # against their closed forms, to the accuracy of the constant's fit to
    # the basis functions kept.
    one = space.project_function(lambda r: 1.0)
    assert volume @ space.to_components(one) == pytest.approx(
        4.0 * np.pi / 3.0, rel=1e-5
    )
    mass, moment = problem.forward(one)
    meters = 1e3 * app["reference"].R_SURFACE
    units = app["problem"]
    assert mass * units.MASS_UNIT == pytest.approx(
        1e3 * 4.0 * np.pi * meters**3 / 3.0, rel=1e-5
    )
    assert moment * units.MOMENT_UNIT == pytest.approx(
        1e3 * 8.0 * np.pi * meters**5 / 15.0, rel=1e-5
    )


def test_the_reference_is_prem_and_the_data_the_truths(app, scale, results):
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
    # With radius in Earth radii the norms are of order one: PREM's is about
    # 20, the truth's departure from it below one.
    space = problem.space
    assert 10.0 < space.norm(problem.reference) < 40.0
    assert space.norm(space.subtract(problem.truth, problem.reference)) < 1.0
    # The data fit the truth and reject the reference.
    critical = problem.problem.critical_chi_squared(level=scale.level)
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
    assert sets["truth_in_ball"] and sets["gauge"] <= 1.0
    low, high = sets["brackets"][:, 0], sets["brackets"][:, 1]
    assert np.all((low <= truth) & (truth <= high))
    # The ball about the reference admits as much either way of the
    # reference's properties; the brackets lie within that, certified to a
    # small gap.
    reference = problem.properties(problem.reference)
    assert np.allclose(sets["bounds"].mean(axis=1), reference)
    assert sets["radius"] > 0.0
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
    # The operator is bilinear, so central differences at a step of one are
    # exact for it and the defects are rounding.
    assert part["worst_defect"] < 1e-10
    assert part["converged"]
    # The MAP model's displacements recover the truth's to within two
    # Laplace standard deviations, which the data narrow at the core-mantle
    # boundary and leave at the prior's at the inner-core boundary.
    shifts, deviations = part["center"][3:], part["deviation"][3:]
    assert np.all(np.abs(shifts - part["truth"][3:]) < 2.0 * deviations)
    assert deviations[1] < 0.9 * part["prior_deviation"][4]
    assert deviations[0] > 0.95 * part["prior_deviation"][3]
    # Occam's model meets the level in the model space's norm and in the
    # prior's, and each is the smaller change in its own norm. In the model
    # space's norm a unit of displacement costs far more than a density
    # change with the same effect on the data, and the boundaries stay
    # within a few kilometers of PREM's.
    space, prior = part["occam"]["space"], part["occam"]["prior"]
    for answer in (space, prior):
        assert answer["converged"]
        assert answer["chi_squared"] == pytest.approx(part["critical"], rel=1e-3)
    assert space["space_norm"] < prior["space_norm"]
    assert prior["prior_norm"] < space["prior_norm"]
    assert np.all(np.abs(space["shifts"]) < 0.05 * part["prior_deviation"][3:])


def test_the_figures_were_written(results):
    outdir = results[1]
    names = ["setup", "boundaries"] + [
        f"{prior}_{answer}"
        for prior in ("tight", "wide")
        for answer in ("point", "measure", "set")
    ]
    for name in names:
        assert (outdir / f"{name}.png").exists(), name
