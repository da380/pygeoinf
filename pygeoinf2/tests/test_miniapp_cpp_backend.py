"""
The C++ vector backend mini-app keeps working.

``pygeoinf2/miniapps/cpp_backend`` wraps a C++ vector type with nanobind and exposes
it as a ``HilbertSpace`` whose vectors have no buffer, no indexing and no
arithmetic. Running the check ladder and the estimators over it is the
regression test that the library defers every vector calculation to the
backend: any reach for a component raises instead of passing by luck.

The extension is built in a temporary directory when it is not already
built next to its sources. Without cmake and a working compiler the tests skip
and say so.
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

from pygeoinf2.algebra.spaces import CoordinateSpace
from pygeoinf2.inference import (
    BackusGilbertParker,
    LinearForwardProblem,
    LinearGaussianInversion,
    MinimumNorm,
)
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.testing import (
    check_operator,
    check_space,
    check_traits,
    check_white_noise,
)

MINIAPP = pathlib.Path(__file__).resolve().parents[1] / "miniapps" / "cpp_backend"


def _importable_from(directory: pathlib.Path) -> bool:
    """Whether ``cppvec`` imports from *directory* alone."""
    spec = importlib.machinery.PathFinder.find_spec("cppvec", [str(directory)])
    return spec is not None


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    """The mini-app's ``space`` module, with the extension built if need be."""
    if not (MINIAPP / "space.py").exists():
        pytest.skip(f"The mini-app is not at {MINIAPP}.")
    pytest.importorskip("nanobind")
    if shutil.which("cmake") is None:
        pytest.skip("cmake is not on the path.")
    if not _importable_from(MINIAPP):
        scratch = tmp_path_factory.mktemp("cpp_backend")
        for name in ("cppvec.cpp", "CMakeLists.txt", "build.py"):
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
            pytest.skip(f"The cppvec extension did not build on this platform:\n{tail}")
        sys.path.insert(0, str(scratch))
    spec = importlib.util.spec_from_file_location(
        "cpp_backend_space", MINIAPP / "space.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def problem(backend):
    """A small instance of the mini-app's problem: space, prior, operators, truth and data."""
    rng = np.random.default_rng(7)
    space = backend.CppSpace(60)
    prior = backend.matern_measure(space, length_scale=0.1)
    points = np.linspace(0.08, 0.92, 8).tolist()
    evaluate = backend.point_evaluation_operator(space, points)
    averages = backend.local_average_operator(space, [(0.2, 0.4), (0.6, 0.9)])
    noise = GaussianMeasure.from_standard_deviation(evaluate.codomain, 0.1)
    forward = LinearForwardProblem(evaluate, error=noise)
    truth, data = forward.synthetic_model_and_data(prior, rng=rng)
    return {
        "space": space,
        "prior": prior,
        "evaluate": evaluate,
        "averages": averages,
        "laplacian": backend.laplacian_operator(space),
        "problem": forward,
        "truth": truth,
        "data": data,
    }


class TestTheVectorIsOpaque:
    def test_the_vector_has_no_array_face(self, backend):
        vector = backend.CppSpace(4).zero()
        for name in (
            "__array__",
            "__getitem__",
            "__len__",
            "__add__",
            "__mul__",
            "__iter__",
        ):
            assert not hasattr(vector, name), name
        with pytest.raises(TypeError):
            memoryview(vector)
        with pytest.raises(TypeError):
            vector + vector  # noqa: B018 - the point is that it raises

    def test_the_space_has_no_coordinates(self, backend):
        space = backend.CppSpace(4)
        assert not isinstance(space, CoordinateSpace)
        assert not hasattr(space, "to_components")


class TestTheCheckLadder:
    def test_space(self, backend, problem, rng):
        space = problem["space"]
        report = check_space(
            space,
            rng=rng,
            measure=problem["prior"],
            rebuild=lambda: backend.CppSpace(60),
        )
        assert report.worst < 1e-12

    def test_white_noise(self, problem, rng):
        report = check_white_noise(problem["space"], rng=rng, measure=problem["prior"])
        assert report.worst < 0.06

    @pytest.mark.parametrize("name", ["evaluate", "averages", "laplacian"])
    def test_operators(self, problem, rng, name):
        operator = problem[name]
        assert check_operator(operator, rng=rng, measure=problem["prior"]).worst < 1e-9
        assert check_traits(operator, rng=rng, measure=problem["prior"]).worst < 1e-9

    def test_prior(self, problem, rng):
        prior = problem["prior"]
        for operator in (prior.covariance, prior.covariance_factor):
            assert check_operator(operator, rng=rng, measure=prior).worst < 1e-9
            assert check_traits(operator, rng=rng, measure=prior).worst < 1e-9


class TestTheEstimators:
    def test_linear_gaussian_inversion(self, problem, rng):
        space, forward = problem["space"], problem["problem"]
        posterior = LinearGaussianInversion(forward, problem["prior"])(problem["data"])
        mean = posterior.expectation
        assert type(mean).__name__ == "Vector"
        assert forward.chi_squared(mean, problem["data"]) < forward.chi_squared(
            space.zero(), problem["data"]
        )
        # The posterior at the observation points is tighter than the prior.
        at_points = posterior.affine_map(problem["evaluate"]).covariance.diagonals()[0]
        before = (
            problem["prior"].affine_map(problem["evaluate"]).covariance.diagonals()[0]
        )
        assert np.all(at_points < before)
        assert np.all(at_points < 0.1**2)
        for draw in posterior.samples(2, rng=rng):
            assert type(draw).__name__ == "Vector"

    def test_minimum_norm_meets_the_discrepancy_principle(self, problem):
        from scipy.stats import chi2

        forward = problem["problem"]
        estimator = MinimumNorm(forward, level=0.95)
        estimate = estimator(problem["data"])
        target = chi2.ppf(0.95, forward.data_space.dim)
        assert forward.chi_squared(estimate, problem["data"]) == pytest.approx(
            target, rel=1e-4
        )
        assert problem["space"].norm(estimate) < problem["space"].norm(problem["truth"])

    def test_feasible_set_brackets_the_truth(self, problem, rng):
        prior, forward = problem["prior"], problem["problem"]
        ball = prior.ambient_ball(level=0.95, rng=rng, samples=2000)
        feasible = BackusGilbertParker(forward, problem["averages"], ball, level=0.95)(
            problem["data"]
        )
        values = problem["averages"].codomain.to_components(
            problem["averages"](problem["truth"])
        )
        space = problem["averages"].codomain
        for index in range(2):
            direction = space.basis_vector(index)
            _, high = feasible.support_bracket(direction)
            _, low = feasible.support_bracket(space.scale(-1.0, direction))
            assert -low - 1e-6 <= values[index] <= high + 1e-6


def test_the_library_asked_only_for_the_contract(problem):
    """Every backend call made so far is one of the members the space defines."""
    allowed = {
        "dim",
        "_key",
        "zero",
        "copy",
        "inner_product",
        "axpy",
        "scale_inplace",
        "random",
        "white_noise",
        "mass_solve",
        "matern.smooth",
        "laplacian.value",
        "point_evaluation.value",
        "point_evaluation.adjoint",
        "local_average.value",
        "local_average.adjoint",
    }
    assert set(problem["space"].calls) <= allowed
    assert problem["space"].calls["inner_product"] > 0
