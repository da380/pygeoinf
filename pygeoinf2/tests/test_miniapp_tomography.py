"""
The ray-tomography mini-app keeps working.

``pygeoinf2/miniapps/tomography`` traces rays through a radially layered disk
with its own ray tracer, wraps them as a path-integral operator on the
library's layered disk and answers the same questions every way the library
offers. Checked here: the ray tracer against the closed form of the circular
rays of a flattened linear gradient; the forward operator against the
composition of point evaluation with the path weights, and its adjoint by the
library's check; the shooting tracer of the bent rays against the radial
one, and the bent forward's derivative against finite differences; and the
quick scale of every part, through its script's entry point, with what must
hold. Needs planetmodel, which the disk and the
ball are built on; without it the tests skip.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("planetmodel")

MINIAPP = pathlib.Path(__file__).resolve().parents[1] / "miniapps" / "tomography"

pytestmark = pytest.mark.slow

#: The app's modules, which import each other by their plain names.
NAMES = (
    "rays",
    "common",
    "problem",
    "setup",
    "bayesian",
    "point",
    "sets",
    "ball",
    "bent",
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
    if (
        name in sys.modules
        and pathlib.Path(getattr(sys.modules[name], "__file__", "") or "").parent
        == MINIAPP
    ):
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
    # Another app's modules of the same plain names must not stand in for these.
    saved = {name: sys.modules.pop(name) for name in NAMES if name in sys.modules}
    modules = {name: _module(name) for name in NAMES}
    yield modules
    for name in NAMES:
        sys.modules.pop(name, None)
    sys.modules.update(saved)


# ------------------------------------------------------------------------- #
#                        The ray tracer, against the closed form            #
# ------------------------------------------------------------------------- #

#: The flattened linear gradient ``v(r) = r (1 - b ln r)`` with ``v(0.55) =
#: 1.8``: the same velocity contrast as the app's mantle.
FLAT_V0 = 1.0
FLAT_GRADIENT = (1.8 / 0.55 - 1.0) / -np.log(0.55)


@pytest.fixture(scope="module")
def flattened(app):
    rays = app["rays"]
    profile = rays.RadialProfile(
        rays.flattened_linear_gradient(FLAT_V0, FLAT_GRADIENT), bottom=0.05
    )
    return profile, rays.CircularArcs(FLAT_V0, FLAT_GRADIENT)


@pytest.mark.parametrize("distance", [5.0, 20.0, 45.0, 80.0, 120.0, 150.0])
def test_the_two_point_ray_is_the_closed_form(flattened, distance):
    profile, arcs = flattened
    reach = np.radians(distance)
    p, exact = profile.ray_parameter(distance), arcs.ray_parameter(reach)
    assert p == pytest.approx(exact, rel=1e-10)
    assert profile.travel_time(p) == pytest.approx(arcs.travel_time(exact), rel=1e-10)
    assert profile.turning_radius(p) == pytest.approx(
        np.exp(-arcs.turning_depth(exact)), rel=1e-10
    )
    # Every node lies on the ray's circle in the flattened medium.
    ray = profile.trace(30.0, 30.0 + distance)
    x = np.radians(ray.nodes[:, 1] - 30.0)
    z = -np.log(ray.nodes[:, 0])
    xc, zc, radius = arcs.circle(exact, reach)
    assert np.abs(np.hypot(x - xc, z - zc) - radius).max() < 1e-9
    assert ray.travel_time == pytest.approx(arcs.travel_time(exact), rel=1e-10)
    assert ray.nodes[0] == pytest.approx([1.0, 30.0])
    assert ray.nodes[-1] == pytest.approx([1.0, 30.0 + distance])


def test_the_path_weights_converge_at_second_order(flattened):
    """The travel time as the path integral of the slowness over the nodes:
    the trapezoid rule's error falls ninefold when the nodes triple."""
    profile, arcs = flattened
    exact = arcs.travel_time(arcs.ray_parameter(np.radians(90.0)))
    errors = []
    for count in (101, 301, 901):
        ray = profile.trace(200.0, 110.0, nodes=count)
        quadrature = ray.weights @ (1.0 / profile.velocity(ray.nodes[:, 0]))
        errors.append(abs(quadrature - exact) / exact)
    assert errors[-1] < 1e-5
    assert errors[0] / errors[1] == pytest.approx(9.0, rel=0.1)
    assert errors[1] / errors[2] == pytest.approx(9.0, rel=0.1)


def test_the_ray_goes_the_short_way_round(flattened):
    profile, _ = flattened
    ray = profile.trace(350.0, 30.0)
    assert ray.nodes[-1, 1] == pytest.approx(390.0)
    assert np.all(np.diff(ray.nodes[:, 1]) > 0.0)
    back = profile.trace(30.0, 350.0)
    assert back.nodes[-1, 1] == pytest.approx(-10.0)
    assert back.travel_time == pytest.approx(ray.travel_time, rel=1e-12)


def test_the_mantle_rays_never_reach_the_core(app):
    problem = app["problem"]
    profile = problem.MANTLE
    low, high = profile.parameter_range
    distances = [profile.distance(p) for p in np.linspace(low, high, 200)[1:-1]]
    assert np.all(np.diff(distances) < 0.0)
    assert profile.grazing_distance > problem.MAX_DISTANCE
    deepest = profile.trace(0.0, problem.MAX_DISTANCE)
    assert deepest.nodes[:, 0].min() > problem.CORE_RADIUS
    with pytest.raises(ValueError, match="reaches"):
        profile.ray_parameter(profile.grazing_distance + 1.0)


# ------------------------------------------------------------------------- #
#                         The forward operator                              #
# ------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def quick_problem(app):
    problem = app["problem"]
    return problem.build(scale=problem.QUICK, quiet=True)


@pytest.fixture(scope="module")
def ball_problem(app):
    problem = app["problem"]
    return problem.build(scale=problem.BALL, quiet=True)


@pytest.mark.parametrize("which", ["quick_problem", "ball_problem"])
def test_the_forward_is_point_evaluation_composed_with_the_weights(app, request, which):
    built = request.getfixturevalue(which)
    composed = app["problem"].composed_operator(built.scale, built.space, built.rays)
    rows = built.forward.matrix(form="galerkin")
    assert (
        np.abs(composed.matrix(form="galerkin") - rows).max()
        < 1e-12 * np.abs(rows).max()
    )


@pytest.mark.parametrize("which", ["quick_problem", "ball_problem"])
def test_the_adjoint_is_the_librarys(request, which):
    from pygeoinf2.testing import check_operator

    built = request.getfixturevalue(which)
    report = check_operator(
        built.forward, rng=np.random.default_rng(7), measure=built.prior, rtol=1e-12
    )
    assert report.worst < 1e-12


def test_the_reference_slowness_gives_the_reference_times(app, quick_problem):
    """The path integral of ``1 / v`` is the travel time, and of one the
    length: the operator integrates what it is given along the right paths."""
    problem = app["problem"]
    space = quick_problem.space
    slowness = space.project_function(
        [
            lambda p: 1.0 / problem.CORE_VELOCITY,
            lambda p: 1.0 / float(problem.MANTLE.velocity(p[0])),
        ]
    )
    times = np.array([ray.travel_time for ray in quick_problem.rays])
    assert np.allclose(quick_problem.forward(slowness), times, rtol=1e-4)
    one = space.project_function(lambda p: 1.0)
    lengths = [
        problem.MANTLE.path_length(
            problem.MANTLE.ray_parameter(abs((r - s + 180.0) % 360.0 - 180.0))
        )
        for s, r in quick_problem.pairs
    ]
    assert np.allclose(quick_problem.forward(one), lengths, rtol=1e-4)


def test_no_ray_is_counted_twice(app, quick_problem):
    ends = {frozenset(pair) for pair in quick_problem.pairs}
    assert len(ends) == len(quick_problem.pairs)
    matrix = quick_problem.forward.matrix(form="galerkin")
    assert np.linalg.matrix_rank(matrix) == len(quick_problem.pairs)


def test_an_unseen_direction_is_reached_by_an_explicit_model(app, quick_problem):
    """The core's mean, which no ray sees: the smallest model fitting the data
    plus the property's representer, filled out to the ball, is feasible and
    reaches the certified outer value."""
    from pygeoinf2 import BackusGilbertParker, CholeskySolver

    sets, level = app["sets"], app["problem"].LEVEL
    built = quick_problem
    ball = built.prior.ambient_ball(level=level)
    noise_ball = built.problem.error_measure.ambient_ball(level=level)
    smallest = sets.smallest_fitting_model(built)
    feasible = BackusGilbertParker(
        built.problem, built.properties, ball, noise=noise_ball, solver=CholeskySolver()
    )(built.data)
    core = built.properties.codomain.basis_vector(2)
    value, norm, misfit = sets.unseen_end(built, ball, noise_ball, smallest, core)
    assert norm <= 1.0 + 1e-9 and misfit <= 1.0 + 1e-9
    assert value == pytest.approx(feasible.support(core), rel=1e-5)
    seen = built.properties.codomain.basis_vector(0)
    assert sets.unseen_end(built, ball, noise_ball, smallest, seen) is None


# ------------------------------------------------------------------------- #
#                      Every part at the quick scale                        #
# ------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def results(app, tmp_path_factory):
    """Every part at the quick scale, once, each through its own entry point."""
    outdir = tmp_path_factory.mktemp("tomography")
    scale, seed = app["problem"].QUICK, 2026
    problem = app["setup"].run(scale, seed=seed, outdir=outdir)
    results = {"a": problem}
    results["b"] = app["bayesian"].run(scale, seed=seed, outdir=outdir, problem=problem)
    results["c"] = app["point"].run(scale, seed=seed, outdir=outdir, problem=problem)
    results["d"] = app["sets"].run(
        scale,
        seed=seed,
        outdir=outdir,
        problem=problem,
        bayesian=results["b"],
        point=results["c"],
    )
    return results, outdir


def test_the_posterior_std_is_smaller_where_the_coverage_is(app, results):
    part = results[0]["b"]
    exact, prior = part["exact_std"], part["prior_std"]
    assert exact["well covered"] < exact["poorly covered"] < exact["core"]
    assert exact["under the stations"] < 0.5 * prior["under the stations"]
    # The core is a layer of its own, which no ray enters.
    assert exact["core"] == pytest.approx(prior["core"], rel=1e-10)
    # The sampled field agrees with the exact values to its standard error.
    samples = app["problem"].QUICK.std_samples
    for name, value in exact.items():
        assert part["sampled_std"][name] == pytest.approx(
            value, rel=4.0 / np.sqrt(2.0 * samples)
        )


def test_the_core_posterior_is_its_prior(results):
    part = results[0]["b"]
    assert part["center"][2] == pytest.approx(0.0, abs=1e-12)
    assert part["deviation"][2] == pytest.approx(part["prior_deviation"][2], rel=1e-10)
    assert np.all(part["deviation"][:2] < part["prior_deviation"][:2])
    assert part["deviation"][0] < part["deviation"][1]


def test_the_truth_lies_in_most_posterior_intervals(results):
    setup, part = results[0]["a"], results[0]["b"]
    inside = np.abs(part["center"] - setup.true_properties) <= 1.96 * part["deviation"]
    assert inside.sum() >= 2


def test_the_point_answers(app, results):
    setup, part = results[0]["a"], results[0]["c"]
    problem = setup.problem
    assert problem.chi_squared(part["smallest"], setup.data) == pytest.approx(
        problem.critical_chi_squared(level=app["problem"].LEVEL), rel=1e-2
    )
    # A bump where the rays cross is seen; one where they do not is seen less.
    assert part["peaks"]["well covered"] > part["peaks"]["poorly covered"]
    # The core is invisible: the estimate is zero there and so is the noise
    # it propagates, and its whole error is bias.
    assert part["estimate"][2] == pytest.approx(0.0, abs=1e-12)
    assert part["propagated"][2] == pytest.approx(0.0, abs=1e-12)
    assert part["bias"][2] == pytest.approx(-setup.true_properties[2], rel=1e-8)


def test_the_feasible_brackets_hold_the_truth(results):
    setup, bayesian, part = results[0]["a"], results[0]["b"], results[0]["d"]
    low, high = part["brackets"][:, 0], part["brackets"][:, 1]
    assert np.all(low - 1e-6 <= setup.true_properties)
    assert np.all(setup.true_properties <= high + 1e-6)
    assert part["inside"]
    assert np.all(
        part["gaps"] < 1e-3 * np.maximum(1.0, np.abs(part["brackets"]).max(1))
    )
    # A norm bound admits more than the Gaussian prior does.
    assert np.all(high - low > 2 * 1.96 * bayesian["deviation"])


def test_the_figures_were_written(results):
    outdir = results[1]
    for name in ("setup", "posterior", "point", "answers"):
        assert (outdir / f"{name}.png").exists(), name


# ------------------------------------------------------------------------- #
#                               The ball                                    #
# ------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def ball_results(app, tmp_path_factory):
    outdir = tmp_path_factory.mktemp("tomography_ball")
    return app["ball"].run(app["problem"].BALL, seed=2026, outdir=outdir), outdir


def test_the_ball_reads_the_same_structure(ball_results):
    results = ball_results[0]
    exact = results["b"]["exact_std"]
    assert exact["well covered"] < exact["poorly covered"] < exact["core"]
    assert results["b"]["deviation"][2] == pytest.approx(
        results["b"]["prior_deviation"][2], rel=1e-10
    )
    assert (
        results["c"]["peaks"]["well covered"] > results["c"]["peaks"]["poorly covered"]
    )
    setup, part = results["a"], results["d"]
    low, high = part["brackets"][:, 0], part["brackets"][:, 1]
    assert np.all(low - 1e-6 <= setup.true_properties)
    assert np.all(setup.true_properties <= high + 1e-6)
    assert part["inside"]


def test_the_ball_figures_were_written(ball_results):
    outdir = ball_results[1]
    for name in ("setup", "posterior", "point", "answers"):
        assert (outdir / f"ball_{name}.png").exists(), name


# ------------------------------------------------------------------------- #
#                               Bent rays                                   #
# ------------------------------------------------------------------------- #


def test_shooting_in_the_reference_disk_is_the_radial_tracer(app):
    """The ray equations integrated in the unperturbed disk, the two-point
    rays found by shooting, against the radial tracer's rays."""
    rays, problem = app["rays"], app["problem"]
    g = problem.GRADIENT

    def radial(x, y):
        r = np.hypot(x, y)
        v = problem.SURFACE_VELOCITY + g * (problem.RADIUS - r)
        return 1.0 / v, g / v**2 * x / r, g / v**2 * y / r

    pairs = [(10.0, 35.0), (200.0, 140.0), (300.0, 368.0)]
    distances = [abs((r - s + 180.0) % 360.0 - 180.0) for s, r in pairs]
    start = np.arcsin([problem.MANTLE.ray_parameter(d) for d in distances]) + 0.01
    bent, angles, _ = rays.bend(radial, pairs, start, bottom=0.56, step=2.5e-3)
    for ray, (source, receiver) in zip(bent, pairs):
        reference = problem.MANTLE.trace(source, receiver, nodes=401)
        assert ray.travel_time == pytest.approx(reference.travel_time, rel=1e-9)
        assert np.allclose(ray.nodes, reference.nodes, atol=1e-8)
        assert ray.weights.sum() == pytest.approx(reference.weights.sum(), rel=1e-6)


@pytest.fixture(scope="module")
def bent_results(app, tmp_path_factory):
    outdir = tmp_path_factory.mktemp("tomography_bent")
    bent = app["bent"]
    return bent.run(bent.BENT_QUICK, seed=2026, outdir=outdir), outdir


def test_the_bent_forward_starts_where_the_linear_one_does(bent_results):
    results = bent_results[0]
    assert np.abs(results["zero"]).max() < 1e-6
    assert results["shift_at_zero"] < 1e-8


def test_the_derivative_is_the_path_integral_along_the_bent_ray(app, bent_results):
    """Central differences converge on it as the square of the step, down to
    a floor the integrator sets."""
    defects = bent_results[0]["defects"]
    steps = app["bent"].STEPS
    for values in defects.values():
        assert values[0] / values[1] == pytest.approx(
            (steps[0] / steps[1]) ** 2, rel=0.3
        )
        assert min(values) < 1e-4
    assert min(defects["reference"]) < 1e-5


def test_bending_shortens_every_time_and_the_mode_fits_it(bent_results):
    results = bent_results[0]
    # Fermat: the bent ray is the quickest path, so no time exceeds the
    # straight-model prediction.
    assert np.all(results["difference"] <= 1e-8)
    assert results["shifts"].max() > 0.01
    assert results["result"].converged
    assert results["mode_chi_squared"] < results["linear_chi_squared"]
    assert 0.0 < results["change"] < 0.5
    assert (bent_results[1] / "bent.png").exists()
