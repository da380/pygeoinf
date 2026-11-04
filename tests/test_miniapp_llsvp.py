"""
The LLSVP mini-app keeps working.

``miniapps/llsvp`` samples the base of the mantle: three coupled thermal fields
on a thin shell, a level set whose positive part is the provinces, and the
chemical contrasts inside them, from synthetic travel times and normal-mode
coefficients. The small scale of the parts built so far is run here, each part
through its own script's entry point, and what must hold is asserted: the forward
map's proofs, the coupled prior's correlation, every sampler against the exact
posterior of the thermal-only problem wherever its chains have mixed, and on the
whole problem Gibbs moving towards the truth and the answers recovering the
chemistry.

Needs cartopy, shapely and pyshtools, and the Natural Earth land polygons, which
cartopy downloads into its cache on first use; without any of those the tests
skip and say so.
"""

from __future__ import annotations

import dataclasses
import importlib
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("pyshtools")
pytest.importorskip("cartopy")
pytest.importorskip("shapely")

MINIAPP = pathlib.Path(__file__).resolve().parents[1] / "miniapps" / "llsvp"

pytestmark = pytest.mark.slow


# The app's modules, in the package ``src``, and its part scripts, which
# import each other and the package by their plain names. Other mini-apps
# use the same names.
MODULES = ("src.common", "src.shell", "src.prior", "src.forward", "src.record")
SCRIPTS = ("synthetic", "linear", "proofs", "gibbs", "answers")
# Shorter chains than the preset's, for the test's sake: what is asserted of
# them allows for the errors they carry.
SHORT = (
    "--budget",
    "1000",
    "--geometric-steps",
    "30",
    "--geometric-burn-in",
    "10",
    "--ladder-steps",
    "40",
    "--gibbs-steps",
    "150",
    "--gibbs-burn-in",
    "150",
    "--compare-steps",
    "30",
    "--compare-burn-in",
    "20",
    "--level-ladder",
    "1",
    "5",
)


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
    try:
        modules["shell"].land_polygons()
    except Exception as error:  # noqa: BLE001 - whatever cartopy raises offline
        _forget()
        pytest.skip(f"The Natural Earth land polygons are not available: {error}")
    yield modules
    _forget()


@pytest.fixture(scope="module")
def scale(app, tmp_path_factory):
    """The quick preset with shorter chains, built through the scripts' own
    command line."""
    shell = app["shell"]
    arguments = app["common"].parse(
        "the quick scale",
        quick=shell.QUICK,
        full=shell.FULL,
        argv=[
            "--quick",
            *SHORT,
            "--outdir",
            str(tmp_path_factory.mktemp("llsvp_options")),
        ],
    )
    return arguments.scale


@pytest.fixture(scope="module")
def results(app, scale, tmp_path_factory):
    """The parts at the quick scale, once, each through its own entry point."""
    outdir = tmp_path_factory.mktemp("llsvp")
    # The chains run in worker processes, which import the app's modules by
    # their plain names from the path they are given.
    sys.path.insert(0, str(MINIAPP))
    try:
        results = {"a": app["proofs"].run(scale, outdir=outdir)}
        results["b"] = app["synthetic"].run(scale, outdir=outdir)
        state = results["b"]["setup"]
        results["c"] = app["linear"].run(scale, outdir=outdir, state=state)
        results["d"] = app["gibbs"].run(scale, outdir=outdir, state=state)
        results["f"] = app["answers"].run(
            scale, outdir=outdir, state=state, gibbs=results["d"]
        )
    finally:
        sys.path.remove(str(MINIAPP))
    return results, outdir


def test_every_field_of_the_scale_is_an_option(app, tmp_path):
    shell = app["shell"]
    arguments = app["common"].parse(
        "",
        quick=shell.QUICK,
        full=shell.FULL,
        argv=[
            "--quick",
            "--noise-s",
            "0.5",
            "--thermal-std",
            "2.0",
            "1.0",
            "0.5",
            "--n-jobs",
            "none",
            "--outdir",
            str(tmp_path),
        ],
    )
    assert arguments.scale == dataclasses.replace(
        shell.QUICK, noise_s=0.5, thermal_std=(2.0, 1.0, 0.5), n_jobs=None
    )


def test_the_network_is_a_function_of_the_seed(app):
    shell = app["shell"]
    first, second = shell.network(shell.PROOF), shell.network(shell.PROOF)
    assert np.array_equal(first.s_caps, second.s_caps)
    assert np.array_equal(first.events, second.events)
    assert len(first.s_arcs) + len(first.p_arcs) == shell.PROOF.arcs
    other = shell.network(dataclasses.replace(shell.PROOF, seed=1))
    assert not np.array_equal(first.events, other.events)


def test_the_diffracted_arcs_have_their_descents_taken_off(app):
    shell = app["shell"]
    scale = shell.PROOF
    grid = shell.network(scale)
    starts = shell.to_vector([start for start, _ in grid.s_arcs])
    ends = shell.to_vector([end for _, end in grid.s_arcs])
    lengths = np.degrees(shell.separation(starts, ends))
    low, high = scale.arc_range
    assert lengths.min() >= low - 2.0 * scale.descent - 1e-6
    assert lengths.max() <= high - 2.0 * scale.descent + 1e-6


def test_the_proofs_hold(results):
    part = results[0]["a"]
    for run in (part["proof"], part["scale"]):
        assert run["reports"]["check_operator(T)"].worst < 1e-12
        assert run["quadrature"]["arcs"] < 1e-9
        assert run["quadrature"]["caps"] < 2e-2 * run["quadrature"]["field_std"]
        assert run["hand"]["arcs"] < 1e-3
        for found, exact in run["hand"]["coefficients"].values():
            assert found == pytest.approx(exact, abs=5e-3)
        assert run["identities"]["linearity"] < 1e-12
        assert run["identities"]["flip"] < 1e-11
        assert run["identities"]["derivative"].worst < 1e-12
        assert run["identities"]["along"] < 1e-11
        assert run["identities"]["across"] < 1e-11
        assert run["identities"]["level_set_gradient"] == 0.0


def test_the_closed_form_conditionals_are_the_schur_complements(results):
    for gaps in results[0]["a"]["conditionals"].values():
        assert gaps["mean"] < 1e-10
        assert gaps["covariance"] < 1e-10
        assert 0.0 < gaps["reduction"] < 1.0


def test_the_prior_couples_the_fields_and_the_data_see_the_chemistry(results, app):
    part = results[0]["b"]
    assert np.max(np.abs(part["correlation"] - part["target"])) < 0.05
    assert part["shares"].min() < part["share"] < part["shares"].max()
    data_size = part["setup"].data.size
    assert abs(part["chi_truth"] - data_size) < 5.0 * np.sqrt(2.0 * data_size)
    # Without the chemistry the truth's thermal part fits far worse.
    assert part["chi_thermal"] > 3.0 * data_size


def test_every_mixed_sampler_agrees_with_the_exact_posterior(results):
    rows = results[0]["c"]["weak_rows"]["rows"]
    mixed = 0
    for row in rows.values():
        # Chains started from exact draws stay near the answer however slowly
        # they mix; their own error bars can be believed once they have mixed.
        assert np.all(np.abs(row["offsets"]) < 2.5)
        if row["rhat"] < 1.1:
            mixed += 1
            assert np.all(np.abs(row["z"]) < 5.0)
    assert mixed >= 1


def test_a_reference_buys_what_the_data_take_away(results):
    part = results[0]["c"]
    weak, given = part["weak_rows"]["rows"], part["given_rows"]["rows"]
    # Stronger data shrink the prior reference's step, but not the reference's
    # own, which is fitted to them.
    assert given["PCN"]["step"] < 0.2 * weak["PCN"]["step"]
    assert given["HMC"]["step"] < 0.5 * weak["HMC"]["step"]
    assert given["GeometricHMC"]["step"] > 0.15
    assert given["GeometricHMC"]["ess_rate"] > given["HMC"]["ess_rate"]


def test_the_samplers_do_not_notice_the_refinement(results):
    steps = results[0]["c"]["ladder"]
    for name in ("PCN", "MALA", "HMC"):
        low, high = steps[name]["step"][0], steps[name]["step"][-1]
        assert 0.5 < high / low < 2.0
    for name in ("PCN", "MALA", "HMC"):
        accepted = steps[name]["acceptance"]
        assert abs(accepted[-1] - accepted[0]) < 0.3
    # A plain random walk of pCN's proposal size accepts next to nothing.
    assert steps["random walk"]["acceptance"][-1] < 0.05


def test_a_chain_records_the_nonlinear_scalars_of_its_states(app, results):
    """What was recorded at the last kept step is what the stored state says."""
    state, main = results[0]["b"]["setup"], results[0]["d"]["main"]
    prior, record = app["prior"], app["record"]
    chain = main.chains[0]
    last = record.summary(state.space, prior.ungrouped(chain.positions[-1]))
    kept = chain.steps - chain.steps % state.scale.gibbs_thin - 1
    np.testing.assert_allclose(chain.records[kept], last)


def test_gibbs_moves_the_contrast_towards_the_truth(results):
    """Burn-in carries the shear contrast from the guess, which the thermal fields
    have taken most of, most of the way to the truth."""
    part = results[0]["d"]
    guess = part["start"][2][0]
    truth = results[0]["b"]["setup"].truth[2][0]
    reached = part["main"].scalars()[1].mean()
    assert abs(reached - truth) < abs(guess - truth)
    # Both blocks adapted to a target they could reach.
    rates = part["main"].block_acceptance_rates
    assert 0.3 < rates[0] < 0.95
    assert 0.02 < rates[1] < 0.6


def test_moving_the_coupled_block_together_moves_the_contrast_further(results):
    rows = results[0]["d"]["rows"]
    together, apart = rows["5 pCN"]["rates"], rows["three"]["rates"]
    assert together[0] > apart[0]


def test_the_answers_recover_the_chemistry(results):
    part = results[0]["f"]
    # Dense piles, as in the truth, wherever the chains have got to.
    assert part["dense"] > 0.9
    assert part["rows"]["ratio"]["mean"] > part["thermal"]
    # A point inside the true provinces is far likelier in the sampled ones.
    assert part["agreement"] > 2.0 * part["outside"]


def test_the_figures_were_written(results):
    outdir = results[1]
    for name in ("network", "prior", "truth", "samplers", "gibbs", "answers"):
        assert (outdir / f"{name}.png").exists(), name
