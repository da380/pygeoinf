"""
The point-data mini-app keeps working.

``pygeoinf2/miniapps/point_data`` observes a field on the sphere at ocean points
only and answers the same questions three ways. The quick scale of every part
is run once here, each part through its own script's entry point, and what
must hold is asserted: the preconditioned solves agree, the point estimate
meets the discrepancy principle, the posterior standard deviation returns
towards the prior's under the continents and falls well below it under the
oceans, and the truth lies in most of the posterior intervals and in the
feasible brackets.

Needs cartopy, shapely and pyshtools, and the Natural Earth land polygons,
which cartopy downloads into its cache on first use; without any of those
the tests skip and say so.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("pyshtools")
pytest.importorskip("cartopy")
pytest.importorskip("shapely")

MINIAPP = pathlib.Path(__file__).resolve().parents[1] / "miniapps" / "point_data"

pytestmark = pytest.mark.slow


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
    modules = {
        name: _module(name)
        for name in (
            "common",
            "solvers",
            "ocean_mask",
            "problem",
            "truth",
            "solve",
            "point",
            "bayesian",
            "sets",
            "nonstationary",
        )
    }
    try:
        modules["ocean_mask"].land_polygons()
    except Exception as error:  # noqa: BLE001 - whatever cartopy raises offline
        pytest.skip(f"The Natural Earth land polygons are not available: {error}")
    yield modules
    # The scripts import each other by their plain names, which were
    # registered for the duration; nothing after this module may see them.
    for name in modules:
        sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def results(app, tmp_path_factory):
    """Every part at the quick scale, once, each through its own entry point."""
    outdir = tmp_path_factory.mktemp("point_data")
    scale, seed = app["problem"].QUICK, 2026
    problem = app["truth"].run(scale, seed=seed, outdir=outdir)
    results = {"a": problem}
    results["b"] = app["solve"].run(scale, seed=seed, outdir=outdir, problem=problem)
    results["c"] = app["point"].run(scale, seed=seed, outdir=outdir, problem=problem)
    results["d"] = app["bayesian"].run(scale, seed=seed, outdir=outdir, problem=problem)
    results["e"] = app["sets"].run(scale, seed=seed, outdir=outdir, problem=problem)
    results["f"] = app["nonstationary"].run(
        scale, seed=seed, outdir=outdir, problem=problem, bayesian=results["d"]
    )
    return results, outdir


def test_the_setup_is_over_the_sea(app, results):
    setup = results[0]["a"]
    mask = app["ocean_mask"]
    assert len(setup.points) == app["problem"].QUICK.n_data
    assert not np.any(mask.on_land(setup.land, setup.points[:, 0], setup.points[:, 1]))
    # The grid mask sees both land and sea, in about the real proportion.
    fraction = np.average(setup.on_land, weights=mask.area_weights(setup.space))
    assert 0.2 < fraction < 0.4


def test_the_preconditioned_solve_agrees_with_the_plain_one(results):
    part = results[0]["b"]
    plain, woodbury = part["rows"]
    assert plain["converged"] and woodbury["converged"]
    assert part["agreement"] < 1e-5
    assert woodbury["iterations"] < plain["iterations"]


def test_the_point_answers(app, results):
    setup, part = results[0]["a"], results[0]["c"]
    problem = setup.problem
    assert problem.chi_squared(part["smallest"], setup.data) == pytest.approx(
        problem.critical_chi_squared(level=app["problem"].LEVEL), rel=1e-2
    )
    assert setup.space.norm(part["smallest"]) < setup.space.norm(setup.truth)
    # A feature under the sea is seen; one under Africa is seen less.
    assert part["peaks"]["North Pacific"] > part["peaks"]["Africa"]
    assert part["peaks"]["North Pacific"] > 0.5


def test_the_posterior_std_returns_to_the_prior_under_the_continents(app, results):
    part = results[0]["d"]
    exact = part["exact_std"]
    prior_std = app["problem"].PRIOR_STD
    assert exact["central Asia"] > 0.6 * prior_std
    assert exact["North Pacific"] < 0.3 * prior_std
    assert exact["South Atlantic"] < 0.3 * prior_std
    assert part["std_land"] > part["std_sea"]
    assert np.all(part["std_values"] <= 1.2 * prior_std)


def test_the_truth_lies_in_most_posterior_intervals(results):
    setup, part = results[0]["a"], results[0]["d"]
    inside = np.abs(part["center"] - setup.true_averages) <= 1.96 * part["deviation"]
    assert inside.sum() >= 2


def test_the_feasible_brackets_hold_the_truth(results):
    setup, bayesian, part = results[0]["a"], results[0]["d"], results[0]["e"]
    low, high = part["brackets"][:, 0], part["brackets"][:, 1]
    assert np.all(low - 1e-6 <= setup.true_averages)
    assert np.all(setup.true_averages <= high + 1e-6)
    assert part["inside"]
    assert np.all(part["gaps"] < 1e-2)
    # A norm bound admits more than the Gaussian prior does.
    assert np.all(high - low > 2 * 1.96 * bayesian["deviation"])
    # The data narrow the bound of the ball alone, least where they are
    # fewest: under Africa the bracket is nearly the ball's whole range.
    alone = part["alone"]
    assert np.all(alone[:, 0] <= low + 1e-9) and np.all(high <= alone[:, 1] + 1e-9)
    narrowed = (high - low) / (alone[:, 1] - alone[:, 0])
    assert narrowed[2] > max(narrowed[0], narrowed[1])


def test_the_non_stationary_prior_moves_the_std_with_its_amplitude(results):
    before, after = results[0]["d"]["exact_std"], results[0]["f"]["exact_std"]
    assert after["South Atlantic"] > before["South Atlantic"]
    assert after["North Pacific"] < before["North Pacific"]


def test_the_figures_were_written(results):
    outdir = results[1]
    for name in ("truth", "solve", "point", "posterior", "brackets", "nonstationary"):
        assert (outdir / f"{name}.png").exists(), name
