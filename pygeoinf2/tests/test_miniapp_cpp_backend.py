"""
The C++ vector backend mini-app keeps working.

``pygeoinf2/miniapps/cpp_backend`` wraps a C++ vector type with nanobind and
exposes it as ``H^1`` on an interval, a ``HilbertSpace`` whose vectors have
no buffer, no indexing and no arithmetic. Running the check ladder and the
estimators over it is the regression test that the library defers every
vector calculation to the backend: any reach for a component raises instead
of passing by luck. The space, its operators, the prior on it and the three
answers are tested on a small mesh; then every part runs once at the quick
scale, built through the scripts' own command line, each through its own
script's entry point, and what it shows is asserted (marked slow).

The extension is built in a temporary directory when it is not already
built into the app's ``src/``. Without cmake and a working compiler the
tests skip and say so.
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
from scipy.stats import chi2

from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.algebra.spaces import CoordinateSpace
from pygeoinf2.inference import (
    BackusGilbertParker,
    LinearForwardProblem,
    LinearGaussianInversion,
    MinimumNorm,
)
from pygeoinf2.numerics import SpectralFamilySolver
from pygeoinf2.numerics.quadratic_forms import weighted_chi2_quantile
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.testing import (
    check_measure,
    check_operator,
    check_space,
    check_traits,
    check_white_noise,
)

MINIAPP = pathlib.Path(__file__).resolve().parents[1] / "miniapps" / "cpp_backend"
# The app's modules, in the package ``src``, and its part scripts, which
# import each other and the package by their plain names. Other mini-apps
# use the same names.
MODULES = ("src.space", "src.problem", "src.common")
SCRIPTS = ("ladder", "point", "bayesian", "sets", "run_all")
# What the build is made from: the C++ sources in ``cpp/`` and the driver.
SOURCES = ("cppvec.cpp", "CMakeLists.txt")
# Every name the space tallies: its members and the operators' callables.
CONTRACT = {
    "dim",
    "_key",
    "zero",
    "copy",
    "inner_product",
    "axpy",
    "scale_inplace",
    "random",
    "white_noise",
    "riesz",
    "point_evaluation.value",
    "point_evaluation.adjoint",
    "local_average.value",
    "local_average.adjoint",
    "prior.covariance",
    "prior.factor",
    "prior.factor.adjoint",
    "prior.precision",
    "prior.whiten",
    "prior.whiten.adjoint",
    "l2_form",
    "l2_form.inverse",
}


def _importable_from(directory: pathlib.Path) -> bool:
    """Whether ``cppvec`` imports from *directory* alone."""
    spec = importlib.machinery.PathFinder.find_spec("cppvec", [str(directory)])
    return spec is not None


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


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    """The mini-app's modules and scripts, keyed by their last names, with
    the extension built if need be."""
    if not (MINIAPP / "cpp" / "cppvec.cpp").exists():
        pytest.skip(f"The mini-app is not at {MINIAPP}.")
    pytest.importorskip("nanobind")
    if shutil.which("cmake") is None:
        pytest.skip("cmake is not on the path.")
    added = None
    if not _importable_from(MINIAPP / "src"):
        scratch = tmp_path_factory.mktemp("cpp_backend")
        (scratch / "cpp").mkdir()
        (scratch / "src").mkdir()
        for name in SOURCES:
            shutil.copy(MINIAPP / "cpp" / name, scratch / "cpp" / name)
        shutil.copy(MINIAPP / "build.py", scratch / "build.py")
        result = subprocess.run(
            [sys.executable, "build.py"],
            cwd=scratch,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not _importable_from(scratch / "src"):
            tail = "\n".join((result.stderr or result.stdout).splitlines()[-8:])
            pytest.skip(f"The cppvec extension did not build on this platform:\n{tail}")
        added = str(scratch / "src")
        sys.path.insert(0, added)
    _forget()
    modules = {name.split(".")[-1]: _module(name) for name in MODULES + SCRIPTS}
    yield modules
    _forget()
    if added is not None:
        sys.path.remove(added)


@pytest.fixture(scope="module")
def small(app):
    """A small instance of the problem: space, prior, operators, truth and data."""
    backend = app["space"]
    space = backend.CppSpace(60, 0.1)
    prior = backend.matern_measure(space, length_scale=0.1, pointwise_std=1.0)
    points = np.linspace(0.08, 0.92, 8).tolist()
    evaluate = backend.point_evaluation_operator(space, points)
    averages = backend.local_average_operator(space, [(0.2, 0.4), (0.6, 0.9)])
    noise = GaussianMeasure.from_standard_deviation(evaluate.codomain, 0.1)
    forward = LinearForwardProblem(evaluate, error=noise)
    truth, data = forward.synthetic_model_and_data(prior, rng=np.random.default_rng(7))
    return {
        "space": space,
        "prior": prior,
        "evaluate": evaluate,
        "averages": averages,
        "noise": noise,
        "problem": forward,
        "truth": truth,
        "data": data,
    }


def _matrices(n_elements: int) -> tuple[np.ndarray, np.ndarray]:
    """The mass and stiffness matrices of the mesh, assembled here in NumPy."""
    h = 1.0 / n_elements
    mass = np.zeros((n_elements + 1, n_elements + 1))
    stiffness = np.zeros_like(mass)
    for e in range(n_elements):
        block = slice(e, e + 2)
        mass[block, block] += h / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
        stiffness[block, block] += np.array([[1.0, -1.0], [-1.0, 1.0]]) / h
    return mass, stiffness


class TestTheVectorIsOpaque:
    def test_the_vector_has_no_array_face(self, app):
        vector = app["space"].CppSpace(4, 0.1).zero()
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

    def test_the_space_has_no_coordinates(self, app):
        space = app["space"].CppSpace(4, 0.1)
        assert not isinstance(space, CoordinateSpace)
        assert not hasattr(space, "to_components")


class TestTheSpace:
    def test_the_axioms_hold(self, app, small, rng):
        report = check_space(
            small["space"],
            rng=rng,
            measure=small["prior"],
            rebuild=lambda: app["space"].CppSpace(60, 0.1),
        )
        assert report.worst < 1e-12

    def test_white_noise_is_white(self, small, rng):
        report = check_white_noise(small["space"], rng=rng, measure=small["prior"])
        assert report.worst < 0.06

    @pytest.mark.parametrize("length", [0.1, 0.0])
    def test_the_inner_product_is_h1(self, app, rng, length):
        space = app["space"].CppSpace(12, length)
        mass, stiffness = _matrices(12)
        x, y = space.random(rng=rng), space.random(rng=rng)
        expected = space.values(x) @ (mass + length**2 * stiffness) @ space.values(y)
        assert space.inner_product(x, y) == pytest.approx(expected, rel=1e-12)

    def test_the_length_is_part_of_the_space(self, app):
        CppSpace = app["space"].CppSpace
        assert CppSpace(10, 0.1) == CppSpace(10, 0.1)
        assert CppSpace(10, 0.1) != CppSpace(10, 0.0)
        assert CppSpace(10, 0.1) != CppSpace(20, 0.1)

    def test_a_value_at_a_point_is_bounded_on_h1_and_not_on_l2(self, app):
        norm = app["ladder"].evaluation_norm
        h1 = [norm(n, 0.08)[0] for n in (100, 200, 400)]
        l2 = [norm(n, 0.0)[0] for n in (100, 200, 400)]
        # The representer's norm settles at that of the continuous Green's
        # function, 1 / sqrt(2 l) in the middle, and grows as the root of the
        # number of elements without the derivative in the norm.
        assert h1 == pytest.approx([1.0 / np.sqrt(2 * 0.08)] * 3, rel=1e-3)
        assert l2[1] / l2[0] == pytest.approx(np.sqrt(2.0), rel=0.02)
        assert l2[2] / l2[1] == pytest.approx(np.sqrt(2.0), rel=0.02)

    def test_nonsense_is_refused(self, app):
        backend = app["space"]
        with pytest.raises(ValueError):
            backend.CppSpace(0, 0.1)
        with pytest.raises(ValueError):
            backend.CppSpace(10, -0.1)
        space = backend.CppSpace(10, 0.1)
        with pytest.raises(ValueError):
            backend.point_evaluation_operator(space, [1.5])
        with pytest.raises(ValueError):
            backend.local_average_operator(space, [(0.5, 0.2)])
        with pytest.raises(ValueError):
            backend.matern_measure(space, length_scale=0.0, pointwise_std=1.0)


class TestTheOperators:
    @pytest.mark.parametrize("name", ["evaluate", "averages"])
    def test_the_adjoint_holds(self, small, rng, name):
        report = check_operator(small[name], rng=rng, measure=small["prior"])
        assert report.worst < 1e-10

    def test_the_adjoint_is_the_representer_in_h1(self, app, small):
        # The representer of a value is G^-1 c, not M^-1 c: its pairing with
        # a model in the space's inner product is the value.
        space, evaluate = small["space"], small["evaluate"]
        representer = evaluate.adjoint(evaluate.codomain.basis_vector(3))
        assert space.inner_product(representer, small["truth"]) == pytest.approx(
            evaluate.codomain.to_components(evaluate(small["truth"]))[3], rel=1e-12
        )


class TestThePrior:
    @pytest.mark.parametrize(
        "name", ["covariance", "covariance_factor", "precision", "precision_factor"]
    )
    def test_its_operators_hold_their_adjoints(self, small, rng, name):
        prior = small["prior"]
        report = check_operator(getattr(prior, name), rng=rng, measure=prior)
        assert report.worst < 1e-10

    @pytest.mark.parametrize("name", ["covariance", "precision"])
    def test_its_operators_are_definite(self, small, rng, name):
        prior = small["prior"]
        assert check_traits(getattr(prior, name), rng=rng, measure=prior).worst < 1e-10

    def test_its_samples_have_its_covariance(self, small, rng):
        check_measure(small["prior"], rng=rng, samples=4000, rtol=0.15)

    def test_its_pieces_agree(self, small, rng):
        space, prior = small["space"], small["prior"]
        x = prior.sample(rng=rng)

        def defect(a, b):
            return space.norm(space.subtract(a, b)) / space.norm(b)

        factor, whiten = prior.covariance_factor, prior.precision_factor
        assert defect(factor(factor.adjoint(x)), prior.covariance(x)) < 1e-12
        assert defect(prior.precision(prior.covariance(x)), x) < 1e-10
        assert defect(whiten(factor(x)), x) < 1e-12
        assert defect(whiten.adjoint(whiten(x)), prior.precision(x)) < 1e-12

    def test_its_pointwise_std_is_the_one_asked_for(self, app, small):
        backend, space, prior = app["space"], small["space"], small["prior"]
        at = backend.point_evaluation_operator(space, [0.5, 0.25, 0.0])
        std = np.sqrt(prior.affine_map(at).covariance.diagonals()[0])
        assert std[0] == pytest.approx(1.0, rel=1e-10)
        assert std[1] == pytest.approx(1.0, rel=0.02)
        # The natural end reflects the field and doubles its variance.
        assert std[2] == pytest.approx(np.sqrt(2.0), rel=0.02)


class TestTheAnswers:
    def test_the_posterior(self, small, rng):
        space, forward, prior = small["space"], small["problem"], small["prior"]
        posterior = LinearGaussianInversion(forward, prior)(small["data"])
        mean = posterior.expectation
        assert type(mean).__name__ == "Vector"
        assert forward.chi_squared(mean, small["data"]) < forward.chi_squared(
            space.zero(), small["data"]
        )
        at_points = posterior.affine_map(small["evaluate"]).covariance.diagonals()[0]
        before = prior.affine_map(small["evaluate"]).covariance.diagonals()[0]
        assert np.all(at_points < before)
        assert np.all(at_points < 0.1**2)
        for draw in posterior.samples(2, rng=rng):
            assert type(draw).__name__ == "Vector"

    def test_the_minimum_norm_model_in_the_priors_norm(self, small):
        forward, prior = small["problem"], small["prior"]
        factor = prior.covariance_factor
        estimator = MinimumNorm(forward.parameterized(factor), level=0.95)
        change = estimator(small["data"])
        estimate = factor(change)
        target = chi2.ppf(0.95, forward.data_space.dim)
        assert forward.chi_squared(estimate, small["data"]) == pytest.approx(
            target, rel=1e-4
        )
        space = small["space"]
        truth_norm = space.norm(prior.precision_factor(small["truth"]))
        assert space.norm(change) < truth_norm

    @pytest.mark.parametrize("l2", [False, True])
    def test_the_feasible_set_brackets_the_truth(self, app, small, rng, l2):
        prior_set = app["sets"].hardened(small["prior"], 0.95, rng, l2=l2)
        assert prior_set.contains(small["truth"])
        noise_ball = small["noise"].ambient_ball(level=0.95)
        feasible = BackusGilbertParker(
            small["problem"],
            small["averages"],
            prior_set,
            noise=noise_ball,
            solver=SpectralFamilySolver(),
        )(small["data"])
        P = small["averages"].codomain
        values = P.to_components(small["averages"](small["truth"]))
        for index in range(2):
            direction = P.basis_vector(index)
            _, high = feasible.support_bracket(direction)
            _, low = feasible.support_bracket(P.scale(-1.0, direction))
            assert -low - 1e-6 <= values[index] <= high + 1e-6

    def test_the_library_asked_only_for_the_contract(self, small):
        """Every backend call made so far on the small problem is one of the
        members the space defines or one of its operators' callables."""
        calls = small["space"].calls
        assert set(calls) <= CONTRACT
        assert calls["inner_product"] > 0 and calls["prior.covariance"] > 0


class TestTheL2Ball:
    """The prior's ball in the ``L^2`` norm, a constraint set of ``H^1`` given
    by the operator ``B = G^-1 M`` and its inverse."""

    def test_the_form_is_the_mass_inner_product(self, app, rng):
        space = app["space"].CppSpace(12, 0.1)
        form, inverse = app["space"].l2_form(space)
        mass, _ = _matrices(12)
        x, y = space.random(rng=rng), space.random(rng=rng)
        expected = space.values(x) @ mass @ space.values(y)
        assert space.inner_product(form(x), y) == pytest.approx(expected, rel=1e-12)
        back = space.subtract(form(inverse(x)), x)
        assert space.norm(back) < 1e-12 * space.norm(x)

    @pytest.mark.parametrize("which", [0, 1])
    def test_it_and_its_inverse_hold_their_claims(self, app, small, rng, which):
        operator = app["space"].l2_form(small["space"])[which]
        prior = small["prior"]
        assert check_operator(operator, rng=rng, measure=prior).worst < 1e-10
        assert check_traits(operator, rng=rng, measure=prior).worst < 1e-10

    def test_an_operator_without_the_traits_is_refused(self, app, small):
        space = small["space"]
        form, _ = app["space"].l2_form(space)
        plain = LinearOperator.from_callables(space, space, form, adjoint=form)
        with pytest.raises(ValueError, match="claim"):
            small["prior"].ambient_ball(
                level=0.95, operator=plain, method="randomized", rank=space.dim
            )

    @pytest.mark.parametrize("l2", [False, True])
    def test_the_radius_is_the_quantile_of_the_weights(self, app, small, rng, l2):
        """Against the weights formed in NumPy: the eigenvalues of the nodal
        covariance seen through the mass matrix, or through the Gram matrix
        for the space's own ball."""
        backend, space, prior = app["space"], small["space"], small["prior"]
        n = space.dim
        # The nodal covariance: the operator is Sigma G, and G^-1 is the
        # Riesz map.
        columns = []
        for i in range(n):
            unit = space.interpolate(np.eye(n)[i])
            columns.append(space.values(prior.covariance(space.riesz(unit))))
        sigma = np.array(columns).T
        mass, stiffness = _matrices(n - 1)
        metric = mass if l2 else mass + 0.1**2 * stiffness
        root = np.linalg.cholesky(metric)
        weights = np.linalg.eigvalsh(root.T @ sigma @ root)
        expected = np.sqrt(weighted_chi2_quantile(weights, 0.95, method="imhof"))
        prior_set = app["sets"].hardened(prior, 0.95, rng, l2=l2)
        assert prior_set.radius == pytest.approx(expected, rel=1e-8)
        if l2:
            # The precision is the form over the radius squared, along any
            # vector.
            form, _ = backend.l2_form(space)
            u = small["truth"]
            ratio = space.inner_product(form(u), u) / space.inner_product(
                prior_set.precision(u), u
            )
            assert np.sqrt(ratio) == pytest.approx(prior_set.radius, rel=1e-10)

    def test_at_length_zero_it_is_the_ball(self, app, rng):
        backend = app["space"]
        space = backend.CppSpace(40, 0.0)
        prior = backend.matern_measure(space, length_scale=0.1, pointwise_std=1.0)
        ball = app["sets"].hardened(prior, 0.95, rng)
        ellipsoid = app["sets"].hardened(prior, 0.95, rng, l2=True)
        average = backend.local_average_operator(space, [(0.2, 0.4)])
        direction = average.adjoint(average.codomain.basis_vector(0))
        assert ellipsoid.support_function()(direction) == pytest.approx(
            ball.support_function()(direction), rel=1e-8
        )


def test_every_field_of_the_scale_is_an_option(app, tmp_path):
    problem = app["problem"]
    arguments = app["common"].parse(
        "",
        quick=problem.QUICK,
        full=problem.FULL,
        argv=[
            "--space-length",
            "0.05",
            "--n-points",
            "20",
            "--prior-length",
            "0.1",
            "--outdir",
            str(tmp_path),
        ],
    )
    assert arguments.scale == dataclasses.replace(
        problem.FULL, space_length=0.05, n_points=20, prior_length=0.1
    )
    assert arguments.outdir == tmp_path


# --------------------------------------------------------------------- #
#                  Every part at the quick scale, by script              #
# --------------------------------------------------------------------- #


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
            "--noise-std",
            str(quick.noise_std),
            "--n-elements",
            str(quick.n_elements),
            "--outdir",
            str(tmp_path_factory.mktemp("cpp_backend_options")),
        ],
    )
    assert arguments.scale == quick
    return arguments.scale


@pytest.fixture(scope="module")
def results(app, scale, tmp_path_factory):
    """Every part at the quick scale, once, each through its own entry point,
    each building the problem from the seed as a part run alone does."""
    outdir = tmp_path_factory.mktemp("cpp_backend")
    parts = {"a": "ladder", "b": "point", "c": "bayesian", "d": "sets"}
    return {
        part: app[name].run(scale, outdir=outdir) for part, name in parts.items()
    }, outdir


@pytest.mark.slow
def test_a_the_ladder(results):
    part = results[0]["a"]
    h1 = [row[1] for row in part["norms"]]
    l2 = [row[2] for row in part["norms"]]
    assert max(h1[1:]) - min(h1[1:]) < 1e-2 * h1[-1]
    # The root of an eightfold refinement, 2.8.
    assert l2 == sorted(l2) and l2[-1] > 2.5 * l2[1]
    for name, report in part["reports"].items():
        if name not in ("white noise", "measure"):
            assert report.worst < 1e-10, name
    assert all(defect < 1e-10 for defect in part["inverses"].values())
    assert set(part["calls"]) <= CONTRACT


@pytest.mark.slow
def test_b_the_point_answer(results):
    part = results[0]["b"]
    assert part["chi_squared"] == pytest.approx(part["critical"], rel=1e-3)
    assert part["norm"] < part["truth_norm"]
    # What the estimate makes of the noise is within its propagated spread.
    noise_part = part["estimate"] - part["true_averages"] - part["bias"]
    assert np.all(np.abs(noise_part) < 4.0 * part["propagated"])
    assert set(part["calls"]) <= CONTRACT


@pytest.mark.slow
def test_c_the_bayesian_answer(results):
    part = results[0]["c"]
    assert np.all(part["after"] <= part["before"] + 1e-12)
    assert part["after"][part["observed"]].max() < 0.3
    assert np.all(part["deviation"] < part["prior_deviation"])
    inside = np.abs(part["center"] - part["true_averages"]) <= 4.0 * part["deviation"]
    assert inside.all()
    assert set(part["calls"]) <= CONTRACT


@pytest.mark.slow
def test_d_the_set_answer(app, results):
    part = results[0]["d"]
    sets = app["sets"]
    assert all(report.worst < 1e-10 for report in part["checks"].values())
    truth = part["true_averages"]
    narrowed = {}
    for name in (sets.H1, sets.L2):
        answer = part["answers"][name]
        low, high = answer["brackets"][:, 0], answer["brackets"][:, 1]
        assert np.all(low <= truth) and np.all(truth <= high), name
        assert answer["inside"] and answer["gauge"] <= 1.0, name
        assert np.all(answer["gaps"] < 1e-3), name
        alone = answer["alone"]
        assert np.allclose(alone[:, 0], -alone[:, 1]), name
        narrowed[name] = (high - low) / (alone[:, 1] - alone[:, 0])
        assert np.all(narrowed[name] < 1.0), name
    # The L^2 ball admits more than the H^1 ball for the narrower average and
    # less for the wider one; the point data narrow it far less, a value at a
    # point not being bounded in L^2.
    h1, l2 = part["answers"][sets.H1], part["answers"][sets.L2]
    assert l2["radius"] < h1["radius"]
    assert l2["alone"][0, 1] > h1["alone"][0, 1]
    assert l2["alone"][1, 1] < h1["alone"][1, 1]
    assert np.all(narrowed[sets.L2] > 2.0 * narrowed[sets.H1])
    # Both bounds alone settle under a fourfold refinement.
    refinement = np.array(part["refinement"])[:, 1:]
    spread = refinement.max(axis=0) - refinement.min(axis=0)
    assert np.all(spread < 0.02 * refinement.max(axis=0))
    assert set(part["calls"]) <= CONTRACT


@pytest.mark.slow
def test_the_figures_were_written(results):
    outdir = results[1]
    for name in ("ladder", "point", "posterior", "brackets"):
        assert (outdir / f"{name}.png").exists(), name
