"""
The waveform mini-app keeps working.

``pygeoinf2/miniapps/waveform`` wraps a C++ acoustic finite-difference solver,
its linearization and its first and second adjoints, bound with nanobind, as a
library ``Operator`` with both derivatives, and runs the nonlinear estimators
on it. Here the adjoints are proved on the small grid -- the explicit
transpose of the stencil, the fused kernels against the unfused ones, the
library's check ladder, dot tests and central differences falling as the
square of the step -- and the quick scale of parts B to D is run through the
scripts' entry points, with what must hold of each asserted.

The extension is built in a temporary directory when it is not already built
next to its sources. Without cmake and a working compiler the tests skip and
say so. Part E, the grid split across MPI ranks, runs on two ranks in a copy of
the app built with MPI, and skips unless an MPI compiler wrapper builds it:
``MPICXX`` names one, or ``mpicxx`` on the path is tried.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pytest

from pygeoinf2 import EvaluationError, MaximumAPosteriori, Occam
from pygeoinf2.testing import check_traits

MINIAPP = pathlib.Path(__file__).resolve().parents[1] / "miniapps" / "waveform"
MODULES = (
    "survey",
    "wave_operator",
    "common",
    "inversion",
    "proofs",
    "synthetic",
    "newton",
    "laplace",
    "occam",
)
SEED = 2026

pytestmark = pytest.mark.slow


def _importable_from(directory: pathlib.Path) -> bool:
    """Whether ``acoustic`` imports from *directory* alone."""
    return (
        importlib.machinery.PathFinder.find_spec("acoustic", [str(directory)])
        is not None
    )


def _module(name: str):
    """One of the mini-app's modules, imported by path.

    The scripts import each other by their plain names, so the directory goes
    on the path for the duration of the import; a module is registered before
    it runs so that its dataclasses can resolve their annotations.
    """
    path = MINIAPP / f"{name}.py"
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
def app(tmp_path_factory):
    """The mini-app's modules, keyed by name, with the extension built if need be."""
    if not (MINIAPP / "acoustic.cpp").exists():
        pytest.skip(f"The mini-app is not at {MINIAPP}.")
    pytest.importorskip("nanobind")
    if shutil.which("cmake") is None:
        pytest.skip("cmake is not on the path.")
    added = None
    if not _importable_from(MINIAPP):
        scratch = tmp_path_factory.mktemp("waveform")
        for name in ("acoustic.cpp", "CMakeLists.txt", "build.py"):
            shutil.copy(MINIAPP / name, scratch / name)
        result = subprocess.run(
            [sys.executable, "build.py"],
            cwd=scratch,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not _importable_from(scratch):
            tail = "\n".join((result.stderr or result.stdout).splitlines()[-8:])
            pytest.skip(
                f"The acoustic extension did not build on this platform:\n{tail}"
            )
        added = str(scratch)
        sys.path.insert(0, added)
    # Other mini-apps use some of the same plain names; none may leak in.
    saved = {name: sys.modules.pop(name) for name in MODULES if name in sys.modules}
    modules = {name: _module(name) for name in MODULES}
    yield modules
    for name in MODULES:
        sys.modules.pop(name, None)
    sys.modules.update(saved)
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
        from pygeoinf2 import GaussianMeasure
        from pygeoinf2.inference import ForwardProblem

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


@pytest.fixture(scope="module")
def results(app, tmp_path_factory):
    """Parts B to D at the quick scale, once, each through its own entry point."""
    outdir = tmp_path_factory.mktemp("waveform_quick")
    scale = app["survey"].QUICK
    results = {"b": app["synthetic"].run(scale, seed=SEED, outdir=outdir)}
    state = results["b"]["setup"]
    results["c"] = app["newton"].run(scale, seed=SEED, outdir=outdir, state=state)
    mode = results["c"]["runs"]["polish"]["model"]
    results["d"] = app["laplace"].run(
        scale, seed=SEED, outdir=outdir, state=state, mode=mode
    )
    results["occam"] = app["occam"].run(
        scale, seed=SEED, outdir=outdir, state=state, mode=mode
    )
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
    assert runs["gauss-newton"]["error"] < 0.4
    assert runs["polish"]["error"] < 0.4
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


def test_the_laplace_gaussian_holds_the_truth(results):
    part = results[0]["d"]
    assert np.all(part["inside"])
    assert np.all(part["std"] < 0.3 * part["prior_std"])
    assert np.all(part["settling"] > 0.98)


def test_occam_meets_its_level_with_less_structure(results):
    part = results[0]["occam"]
    assert part["result"].converged
    assert part["chi2"] == pytest.approx(part["target"], rel=1e-3)
    assert part["distance"]["Occam"] < part["distance"]["MAP mode"]
    assert part["distance"]["Occam"] < part["distance"]["truth"]


def test_the_figures_were_written(results):
    outdir = results[1]
    for name in ("data", "map", "laplace", "occam"):
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
    """Part E on two ranks: the operator, the checks and the MAP, bit for bit."""
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
    # modules built here and not whatever sits beside the sources.
    for path in MINIAPP.iterdir():
        if path.suffix in (".py", ".cpp") or path.name == "CMakeLists.txt":
            shutil.copy(path, tmp_path / path.name)
    environment = dict(os.environ, MPICXX=wrapper, MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable, "build.py", "--mpi"],
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
    assert numbers["grid"]["error"] < 0.4
