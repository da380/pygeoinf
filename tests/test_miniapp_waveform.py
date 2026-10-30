"""
The waveform mini-app keeps working.

``miniapps/waveform`` wraps a C++ acoustic finite-difference solver,
its linearization and its first and second adjoints, bound with nanobind, as a
library ``Operator`` with both derivatives, and runs the nonlinear estimators
on it. Here the adjoints are proved on the small grid -- the explicit
transpose of the stencil, the fused kernels against the unfused ones, the
library's check ladder, dot tests and central differences falling as the
square of the step -- what the sponge sends back is measured at the quick
scale against a grid too big to reflect, and the quick scale of parts B to E
is run through the scripts' entry points, with what must hold of each
asserted.

The extension is built in a temporary directory when it is not already built
into the app's ``src/``. Without cmake and a working compiler the tests skip and
say so. Part F, the grid split across MPI ranks, runs on two ranks in a copy of
the app built with MPI, and skips unless an MPI compiler wrapper builds it:
``MPICXX`` names one, or ``mpicxx`` on the path is tried.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.machinery
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pytest

from pygeoinf import EvaluationError, MaximumAPosteriori, Occam
from pygeoinf.testing import check_traits

MINIAPP = pathlib.Path(__file__).resolve().parents[1] / "miniapps" / "waveform"
# The app's modules, in the package ``src``, and its part scripts, which
# import each other and the package by their plain names. Other mini-apps
# use the same names.
MODULES = ("src.survey", "src.wave_operator", "src.common", "src.inversion")
SCRIPTS = ("proofs", "synthetic", "newton", "occam", "laplace")
# What the app's build is made from, in its cpp/.
BUILD = ("acoustic.cpp", "CMakeLists.txt", "build.py")

pytestmark = pytest.mark.slow


def _importable_from(directory: pathlib.Path) -> bool:
    """Whether ``acoustic`` imports from *directory* alone."""
    return (
        importlib.machinery.PathFinder.find_spec("acoustic", [str(directory)])
        is not None
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
    sys.path.insert(0, str(MINIAPP))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(MINIAPP))


def _copy_build(app: pathlib.Path, /) -> None:
    """The build's files into a copy of the app, in its ``cpp/``, with an empty
    ``src/`` for the build to put the modules in."""
    (app / "cpp").mkdir(parents=True, exist_ok=True)
    (app / "src").mkdir(exist_ok=True)
    for name in BUILD:
        shutil.copy(MINIAPP / "cpp" / name, app / "cpp" / name)


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    """The mini-app's modules and scripts, keyed by their last names, with the
    extension built if need be."""
    if not (MINIAPP / "cpp" / "acoustic.cpp").exists():
        pytest.skip(f"The mini-app is not at {MINIAPP}.")
    pytest.importorskip("nanobind")
    if shutil.which("cmake") is None:
        pytest.skip("cmake is not on the path.")
    added = None
    if not _importable_from(MINIAPP / "src"):
        scratch = tmp_path_factory.mktemp("waveform")
        _copy_build(scratch)
        result = subprocess.run(
            [sys.executable, "cpp/build.py"],
            cwd=scratch,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not _importable_from(scratch / "src"):
            tail = "\n".join((result.stderr or result.stdout).splitlines()[-8:])
            pytest.skip(
                f"The acoustic extension did not build on this platform:\n{tail}"
            )
        added = str(scratch / "src")
        sys.path.insert(0, added)
    _forget()
    modules = {name.split(".")[-1]: _module(name) for name in MODULES + SCRIPTS}
    yield modules
    _forget()
    if added is not None:
        sys.path.remove(added)


@pytest.fixture(scope="module")
def proof(app):
    """The proof grid: its experiment and its operator."""
    survey = app["survey"]
    experiment = survey.build(survey.PROOF)
    wave = app["wave_operator"].WaveformOperator(experiment.survey, experiment.space)
    return experiment, wave


class TestTheProofs:
    def test_the_stencil_is_transposed_explicitly(self, proof):
        experiment, _ = proof
        kernels = experiment.survey.model
        rng = np.random.default_rng(0)
        u = rng.normal(size=experiment.survey.shape)
        for transpose in (False, True):
            fast = kernels.laplacian(u, transpose)
            literal = kernels.laplacian(u, transpose, reference=True)
            assert np.linalg.norm(fast - literal) < 1e-14 * np.linalg.norm(literal)
        # The free surface makes it unsymmetric, so the transpose matters.
        asymmetry = np.linalg.norm(kernels.laplacian(u) - kernels.laplacian(u, True))
        assert asymmetry > 1e-2 * np.linalg.norm(kernels.laplacian(u))

    def test_the_dot_tests_hold_to_rounding(self, app, proof):
        experiment, wave = proof
        own = app["proofs"].own_defects(experiment, wave, np.random.default_rng(1))
        for name, defect in own["defects"].items():
            if "not symmetric" in name:
                continue
            assert defect < 1e-13, name
        # Central differences fall as the square of the step until rounding.
        steps = {step: (first, second) for step, first, second in own["taylor"]}
        for column in (0, 1):
            ratio = steps[1e-2][column] / steps[1e-3][column]
            assert 50 < ratio < 200

    def test_the_library_check_ladder(self, app, proof):
        experiment, wave = proof
        reports = app["proofs"].library_ladder(
            experiment, wave, np.random.default_rng(2)
        )
        assert reports["check_operator(F'(m))"].worst < 1e-12
        assert reports["check_operator(F''(m)[d, .])"].worst < 1e-12
        assert reports["check_derivative(F)"].worst < 1e-7
        assert reports["check_second_derivative(F)"].worst < 1e-7
        assert reports["check_gradient(MAP objective)"].worst < 1e-6
        assert reports["check_traits(exact Hessian)"].worst < 1e-10

    def test_the_gauss_newton_hessians_are_symmetric(self, app, proof):
        experiment, wave = proof
        from pygeoinf import GaussianMeasure
        from pygeoinf.inference import ForwardProblem

        noise = GaussianMeasure.from_standard_deviation(wave.data_space, 0.01)
        problem = ForwardProblem(wave.first_order, error=noise)
        data = problem.synthetic_data(experiment.truth, rng=np.random.default_rng(3))
        point = experiment.prior.sample(rng=np.random.default_rng(4))
        estimator = MaximumAPosteriori(problem, experiment.prior, gauss_newton=True)
        hessian = estimator.objective(data).hessian(point)
        check_traits(hessian, rng=np.random.default_rng(5), measure=experiment.probes)
        occam = Occam(problem, reference=experiment.background, gauss_newton=True)
        constraint = occam.misfit_constraint(data).hessian(point)
        check_traits(
            constraint, rng=np.random.default_rng(6), measure=experiment.probes
        )

    def test_an_unstable_model_is_refused_and_the_cache_is_reused(self, proof):
        experiment, wave = proof
        too_fast = experiment.background.copy()
        too_fast[10, 10] = 1e-3
        with pytest.raises(EvaluationError):
            wave.value(too_fast)
        negative = experiment.background.copy()
        negative[5, 5] = -1.0
        with pytest.raises(EvaluationError):
            wave.value(negative)
        before = wave.calls["forward"]
        first = wave.value(experiment.truth)
        second = wave.value(experiment.truth.copy())
        assert wave.calls["forward"] == before + experiment.survey.n_shots
        np.testing.assert_array_equal(first, second)


def test_the_sponge_sends_back_little(app):
    """Against a grid padded by 200 cells, at the quick scale."""
    sponge = app["proofs"].reflection(app["survey"].QUICK)
    assert sponge["data"] < 0.01
    assert sponge["interior"] < 0.03


@pytest.fixture(scope="module")
def scale(app, tmp_path_factory):
    """The quick preset, built through the scripts' own command line with
    options given at their preset values."""
    survey = app["survey"]
    quick = survey.QUICK
    arguments = app["common"].parse(
        "the quick scale",
        quick=quick,
        full=survey.FULL,
        argv=[
            "--quick",
            "--noise-fraction",
            str(quick.noise_fraction),
            "--shots",
            *map(str, quick.shots),
            "--outdir",
            str(tmp_path_factory.mktemp("waveform_options")),
        ],
    )
    assert arguments.scale == quick
    return arguments.scale


def test_every_field_of_the_scale_is_an_option(app, tmp_path):
    survey, inversion = app["survey"], app["inversion"]
    arguments = app["common"].parse(
        "",
        quick=survey.QUICK,
        full=survey.FULL,
        argv=[
            "--quick",
            "--noise-fraction",
            "0.1",
            "--n-jobs",
            "none",
            "--outdir",
            str(tmp_path),
        ],
    )
    changed = dataclasses.replace(survey.QUICK, noise_fraction=0.1, n_jobs=None)
    assert arguments.scale == changed
    # Part C's saved modes are never read back under another problem.
    assert inversion.modes_path(tmp_path, survey.QUICK).name == "modes_quick_2026.npz"
    assert inversion.modes_path(tmp_path, changed) != inversion.modes_path(
        tmp_path, survey.QUICK
    )


@pytest.fixture(scope="module")
def results(app, scale, tmp_path_factory):
    """Parts B to E at the quick scale, once, each through its own entry point."""
    outdir = tmp_path_factory.mktemp("waveform_quick")
    results = {"b": app["synthetic"].run(scale, outdir=outdir)}
    state = results["b"]["setup"]
    results["c"] = app["newton"].run(scale, outdir=outdir, state=state)
    mode = results["c"]["runs"]["polish"]["model"]
    results["d"] = app["occam"].run(scale, outdir=outdir, state=state, mode=mode)
    results["e"] = app["laplace"].run(scale, outdir=outdir, state=state, mode=mode)
    return results, outdir


def test_the_data_see_the_anomalies(results):
    part = results[0]["b"]
    n = part["setup"].wave.data_space.dim
    assert abs(part["misfit"]["truth"] - n) < 5 * np.sqrt(2 * n)
    assert part["misfit"]["background"] > 1.2 * n


def test_the_map_recovers_the_anomalies(results):
    part = results[0]["c"]
    runs = part["runs"]
    truth_misfit = results[0]["b"]["misfit"]["truth"]
    # Over the whole section, which takes in the noise's imprint along its
    # sides and bottom as well as the anomalies.
    assert runs["gauss-newton"]["error"] < 0.65
    assert runs["polish"]["error"] < 0.65
    for run in runs.values():
        assert run["chi2"] < truth_misfit
    # The exact Hessian polishes: two Newton steps with tight inner solves
    # take the gradient down by orders of magnitude.
    last_gauss_newton = runs["gauss-newton"]["log"][-1][1]
    assert runs["polish"]["log"][-1][1] < 1e-2 * last_gauss_newton
    # The modes of the two Hessians agree.
    state = part["setup"]
    space = state.experiment.space
    gap = space.norm(
        space.subtract(runs["gauss-newton"]["model"], runs["polish"]["model"])
    )
    assert gap < 0.05 * space.norm(
        space.subtract(runs["polish"]["model"], state.experiment.background)
    )


def test_occam_meets_its_level_with_less_structure(results):
    part = results[0]["d"]
    assert part["result"].converged
    assert part["chi2"] == pytest.approx(part["target"], rel=1e-3)
    assert part["distance"]["Occam"] < part["distance"]["MAP mode"]
    assert part["distance"]["Occam"] < part["distance"]["truth"]


def test_the_laplace_gaussian_holds_the_truth(results):
    part = results[0]["e"]
    assert np.all(part["inside"])
    assert np.all(part["std"] < 0.35 * part["prior_std"])
    assert np.all(part["settling"] > 0.98)


def test_the_figures_were_written(results):
    outdir = results[1]
    for name in ("data", "map", "occam", "laplace"):
        assert (outdir / f"{name}.png").exists(), name


def _mpi_tools() -> tuple[str, str] | None:
    """An MPI compiler wrapper and its launcher, or None.

    ``MPICXX`` names the wrapper if it is set, otherwise ``mpicxx`` on the
    path; the launcher is ``MPIEXEC`` if set, otherwise the ``mpiexec`` beside
    the wrapper, otherwise the one on the path.
    """
    import os

    wrapper = os.environ.get("MPICXX") or shutil.which("mpicxx")
    if wrapper is None:
        return None
    beside = pathlib.Path(wrapper).resolve().parent / "mpiexec"
    launcher = os.environ.get("MPIEXEC") or (
        str(beside) if beside.exists() else shutil.which("mpiexec")
    )
    if launcher is None:
        return None
    return wrapper, launcher


def test_the_grid_split_across_ranks_gives_the_serial_answers(tmp_path):
    """Part F on two ranks: the operator, the checks and the MAP, bit for bit."""
    import json
    import os

    if not (MINIAPP / "mpi_run.py").exists():
        pytest.skip(f"The mini-app is not at {MINIAPP}.")
    pytest.importorskip("nanobind")
    if shutil.which("cmake") is None:
        pytest.skip("cmake is not on the path.")
    tools = _mpi_tools()
    if tools is None:
        pytest.skip(
            "No MPI compiler wrapper and launcher: set MPICXX, or put mpicxx on the path."
        )
    wrapper, launcher = tools
    # A copy of the app, built with MPI, so that the scripts import the
    # modules built here and not whatever sits in the app's src/.
    _copy_build(tmp_path)
    for folder in (MINIAPP, MINIAPP / "src"):
        for path in folder.glob("*.py"):
            shutil.copy(path, tmp_path / path.relative_to(MINIAPP))
    environment = dict(os.environ, MPICXX=wrapper, MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable, "cpp/build.py", "--mpi"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        lines = (result.stdout + result.stderr).splitlines()
        errors = [line for line in lines if "rror" in line and "Traceback" not in line]
        tail = "\n".join((errors or lines)[:8])
        pytest.skip(f"acoustic_mpi did not build with {wrapper}:\n{tail}")
    outdir = tmp_path / "out"
    run = subprocess.run(
        [
            launcher,
            "-n",
            "2",
            sys.executable,
            "mpi_run.py",
            "--quick",
            "--outdir",
            str(outdir),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    numbers = json.loads((outdir / "mpi_quick.json").read_text())
    assert numbers["ranks"] == 2
    for grid in ("proof", "grid"):
        assert max(numbers[grid]["agreement"].values()) == 0.0
    ladder = numbers["proof"]["ladder"]
    assert ladder["check_operator(F'(m))"] < 1e-12
    assert ladder["check_operator(F''(m)[d, .])"] < 1e-12
    assert numbers["grid"]["error"] < 0.65
