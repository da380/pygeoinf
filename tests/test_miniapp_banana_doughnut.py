"""
The banana-doughnut mini-app keeps working.

``miniapps/banana_doughnut`` sets the two linear functionals of a traveltime
side by side on one model space, ``dln c`` on the mantle annulus in ``H^2``:
ray theory's, point evaluation along the ray traced in the reference mantle,
and finite-frequency theory's, a quadrature of the paraxial kernel in
ray-centered coordinates about that ray, composed with point evaluation in
the same way. Checked here, at the quick scale: the quadrature in the detour
time against the Gaussian identity that reduces the one functional to the
other; the two functionals on models constant across the banana; the
finite-frequency functional closing on the ray's as the period falls; both
through the library's operator check under a prior, composed against
assembled; the weight lost where the banana leaves the mantle, below the
core-mantle boundary and above the surface apart; the full scale's
quadrature on the quick space; the priors, each with lengths of its own on
the one space, through the library's measure checks; every field of the
scale as an option; and part A's figures. The dynamic ray tracing behind
the kernel is checked with the tracer, in ``test_miniapp_raytracer.py``.
Needs planetmodel, which the ball is built on; without it the tests skip.
"""

from __future__ import annotations

import dataclasses
import importlib
import pathlib
import sys
import time

import numpy as np
import pytest

pytest.importorskip("planetmodel")

MINIAPP = pathlib.Path(__file__).resolve().parents[1] / "miniapps" / "banana_doughnut"

pytestmark = pytest.mark.slow

#: The app's modules, in the package ``src``, and its part scripts, which
#: import each other and the package by their plain names. Other mini-apps
#: use the same names.
MODULES = (
    "src.rays",
    "src.common",
    "src.planet",
    "src.pulse",
    "src.functionals",
    "src.problem",
)
SCRIPTS = ("setup",)

#: Dominant periods falling by halves from the quick scale's band, for the
#: functionals' approach to each other.
FALLING = (0.024, 0.012, 0.006, 0.003)


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
    the list options given at their preset values."""
    problem = app["problem"]
    quick = problem.QUICK
    arguments = app["common"].parse(
        "the quick scale",
        quick=quick,
        full=problem.FULL,
        argv=[
            "--quick",
            "--prior-lengths",
            *map(str, quick.prior_lengths),
            "--prior-pairs",
            *(f"{radial},{lateral}" for radial, lateral in quick.prior_pairs),
            "--outdir",
            str(tmp_path_factory.mktemp("banana_doughnut_options")),
        ],
    )
    assert arguments.scale == quick
    return arguments.scale


@pytest.fixture(scope="module")
def problem(app, scale):
    return app["problem"].build(scale=scale, quiet=True)


@pytest.fixture(scope="module")
def falling(app, problem):
    """The quadratures of the quick ray's finite-frequency functional in
    bands of falling period, at the quick scale's counts."""
    functionals, pulse = app["functionals"], app["pulse"]
    path = problem.paths[0]
    return [
        functionals.quadrature(path, pulse.Band(period), problem.scale.rule)
        for period in FALLING
    ]


def test_every_field_of_the_scale_is_an_option(app, tmp_path, capsys):
    problem, common = app["problem"], app["common"]
    arguments = common.parse(
        "",
        quick=problem.QUICK,
        full=problem.FULL,
        argv=[
            "--quick",
            "--prior-lengths",
            "0.1",
            "--prior-pairs",
            "0.05,0.2",
            "0.1,0.4",
            "--lmax",
            "12",
            "--outdir",
            str(tmp_path),
        ],
    )
    expected = dataclasses.replace(
        problem.QUICK,
        prior_lengths=(0.1,),
        prior_pairs=((0.05, 0.2), (0.1, 0.4)),
        lmax=12,
    )
    assert arguments.scale == expected
    assert arguments.scale.prior_scales == [(0.1, 0.1), (0.05, 0.2), (0.1, 0.4)]
    # The full preset is the default.
    default = common.parse(
        "", quick=problem.QUICK, full=problem.FULL, argv=["--outdir", str(tmp_path)]
    )
    assert default.scale == problem.FULL
    # A pair is two values with a comma between them.
    with pytest.raises(SystemExit):
        common.parse(
            "",
            quick=problem.QUICK,
            full=problem.FULL,
            argv=["--prior-pairs", "0.05", "--outdir", str(tmp_path)],
        )
    capsys.readouterr()
    with pytest.raises(SystemExit):
        common.parse("", quick=problem.QUICK, full=problem.FULL, argv=["--help"])
    shown = " ".join(capsys.readouterr().out.split())
    for field in dataclasses.fields(problem.Scale):
        if field.name not in common.FIXED:
            assert "--" + field.name.replace("_", "-") in shown, field.name
    assert "0.05,0.2 0.05,0.5" in shown


# ------------------------------------------------------------------------- #
#                 The finite-frequency functional against the ray's         #
# ------------------------------------------------------------------------- #


@pytest.mark.parametrize("period", [0.0015, 0.006, 0.024, 0.096])
def test_the_gaussian_identity_holds_in_every_band(app, scale, period):
    """``int G du = 1`` by the quadrature in the detour time: ``-2.476e-8``
    in every band, since the quadrature of every band is one scaled
    quadrature of one universal function, and the same at twice the nodes
    to 2e-14, so that what remains is the tail cut off, not the count."""
    band = app["pulse"].Band(period)
    options = {"cutoff": scale.cutoff, "frequencies": scale.frequency_nodes}
    defect = band.gaussian_defect(scale.detour_nodes, **options)
    assert abs(defect) < 1e-7
    assert defect == pytest.approx(
        band.gaussian_defect(2 * scale.detour_nodes, **options), abs=1e-12
    )


def test_on_a_constant_the_two_functionals_agree(app, problem):
    """A model constant across the banana is the reduction to ray theory.

    Asked of the constant itself at every point, the surface not clipped,
    the finite-frequency functional is the ray's times the Gaussian
    identity's sum, to rounding (measured 6e-16). Through the space it
    drops the points above the surface, and misses the ray's by the weight
    clipped, to what the projected constant's own departure from one makes
    of it at degree 16: 9e-6 measured, 5e-5 allowed.
    """
    functionals = app["functionals"]
    path, band = problem.paths[0], problem.bands[0]
    quadrature = problem.quadratures[0][0]
    scale = problem.scale
    defect = band.gaussian_defect(
        scale.detour_nodes, cutoff=scale.cutoff, frequencies=scale.frequency_nodes
    )
    exact = quadrature.integrate(lambda p: np.ones(len(p)), clip=False)
    ray = functionals.ray_weights(path).sum()
    assert exact / ray - 1.0 == pytest.approx(defect, abs=1e-13)

    one = problem.space.project_function(lambda p: 1.0)
    through = problem.finite_frequency_operator(0)(one)[0]
    along = problem.ray_operator()(one)[0]
    assert through / along - 1.0 == pytest.approx(-quadrature.clipped, abs=5e-5)


def test_on_a_radial_model_of_long_wavelength_they_close_as_the_period_falls(
    app, problem, falling
):
    """``cos(2 pi (1 - r) / 0.9)``, asked at every point exactly and nothing
    clipped: what separates the two functionals is the model's curvature
    across the banana, which falls as the Fresnel zone's area, as ``T0``.
    The gap over the travel time measured 6.5e-6, 1.6e-6, 4.2e-7, 1.2e-7 as
    ``T0`` halves, falling by 4.0, 3.9 and 3.6."""
    functionals = app["functionals"]
    path = problem.paths[0]

    def radial(points: np.ndarray) -> np.ndarray:
        return np.cos(2.0 * np.pi * (1.0 - points[:, 0]) / 0.9)

    ray = functionals.ray_weights(path) @ radial(functionals.ray_points(path))
    gaps = np.array(
        [
            abs(quadrature.integrate(radial, clip=False) - ray) / path.travel_time
            for quadrature in falling
        ]
    )
    assert gaps[0] < 2e-5
    assert np.all(gaps[:-1] / gaps[1:] > 3.0)


def test_on_a_smooth_draw_they_close_as_the_period_falls(app, problem, falling):
    """A draw from the smoothest prior, through the space: the gap between
    the two functionals over the prior's standard deviation of the ray
    delay. The weight clipped at the surface sets it, at first order and
    unevenly between bands, and the paraxial approach to the ray beneath at
    second; both fall with the period. Over twenty draws the gap measured
    at most 5.6e-3 in the quick band and 1.6e-5 at an eighth of its period,
    and fell at least twelvefold between them; this one 4.3e-3 and 1.1e-5.
    """
    functionals = app["functionals"]
    space, scale = problem.space, problem.scale
    prior = problem.prior(max(scale.prior_lengths))
    ray = problem.ray_operator()
    spread = np.sqrt(
        (ray @ prior.covariance @ ray.adjoint).matrix(form="components")[0, 0]
    )
    model = prior.sample(rng=np.random.default_rng(scale.seed))
    delay = ray(model)[0]
    gaps = np.array(
        [
            abs(
                functionals.composed_operator(
                    space, [(quadrature.lifted, quadrature.kept_weights)]
                )(model)[0]
                - delay
            )
            / spread
            for quadrature in falling
        ]
    )
    assert gaps[0] < 2e-2
    assert gaps[-1] < 1e-4
    assert gaps[-1] < gaps[0] / 4.0


# ------------------------------------------------------------------------- #
#                         The functionals as operators                      #
# ------------------------------------------------------------------------- #


def _operator(problem, which: str, assembled: bool):
    if which == "ray":
        return problem.ray_operator(assembled=assembled)
    return problem.finite_frequency_operator(0, assembled=assembled)


@pytest.mark.parametrize("assembled", [False, True])
@pytest.mark.parametrize("which", ["ray", "finite frequency"])
def test_the_adjoint_is_the_librarys(problem, which, assembled):
    """Under the anisotropic prior, whose draws are the most varied; the
    worst defect of either functional either way measured 1.5e-13 over
    every prior of the quick scale."""
    from pygeoinf.testing import check_operator

    measure = problem.prior(*problem.scale.prior_pairs[0])
    report = check_operator(
        _operator(problem, which, assembled),
        rng=np.random.default_rng(7),
        measure=measure,
        rtol=1e-12,
    )
    assert report.worst < 1e-12


@pytest.mark.parametrize("which", ["ray", "finite frequency"])
def test_the_composed_and_assembled_functionals_agree(problem, which):
    composed = _operator(problem, which, False).matrix(form="galerkin")
    rows = _operator(problem, which, True).matrix(form="galerkin")
    assert composed.shape == (len(problem.paths), problem.space.dim)
    assert np.abs(composed - rows).max() <= 1e-12 * np.abs(rows).max()


def test_the_weight_dropped_above_the_surface_shrinks_with_the_band(problem, falling):
    """The paraxial kernel knows nothing of the free surface, and the points
    of the quadrature above it are dropped with their weight: the
    approximation. The signed fraction, by which the functional misses on a
    constant, cancels between rings of either sign and moves with the nodes
    along the ray; the absolute one is steady, 1.05% in the quick band, and
    falls with the period: 1.05%, 0.34%, 0.061%, 0.0007%. The quick ray
    turns 0.27 above the core-mantle boundary, and in these bands no point
    reaches below it."""
    quadrature = problem.quadratures[0][0]
    assert quadrature.kept.sum() == len(quadrature.lifted)
    assert abs(quadrature.surface_clipped) <= quadrature.surface_clipped_absolute < 0.03
    for q in falling:
        assert q.core_clipped == 0.0 and q.core_clipped_absolute == 0.0
        assert q.surface_clipped == pytest.approx(q.clipped, abs=1e-15)
    absolute = np.array([q.surface_clipped_absolute for q in falling])
    assert np.all(np.diff(absolute) < 0.0)
    assert absolute[-1] < 1e-4


def test_the_two_boundaries_are_reported_apart(app, problem):
    """In the full scale's widest band the quick ray's doughnut reaches
    below the core-mantle boundary, where the functional is zero by
    physics, as well as above the surface: 0.021% and 10.3% of the absolute
    weight. The two fractions are reported apart and add up to the
    total."""
    functionals, pulse = app["functionals"], app["pulse"]
    full = app["problem"].FULL
    quadrature = functionals.quadrature(
        problem.paths[0], pulse.Band(max(full.periods)), problem.scale.rule
    )
    assert 0.0 < quadrature.core_clipped_absolute < quadrature.surface_clipped_absolute
    assert quadrature.core_clipped + quadrature.surface_clipped == pytest.approx(
        quadrature.clipped, abs=1e-15
    )
    assert (
        quadrature.core_clipped_absolute + quadrature.surface_clipped_absolute
    ) == pytest.approx(quadrature.clipped_absolute, abs=1e-15)
    radii = quadrature.radii
    assert np.array_equal(
        ~quadrature.kept,
        functionals.above_surface(radii) | functionals.below_core(radii),
    )


def test_the_full_scale_functional_on_the_quick_space(app, problem):
    """The full scale's quadrature, the most points of any functional, in
    its narrowest band, on the quick space: every node inside the ends with
    its detours and angles, the ends one point each; it applies in 0.14 s
    measured, and its adjoint is the library's (measured 4e-14)."""
    from pygeoinf.testing import check_operator

    functionals, planet, pulse = app["functionals"], app["planet"], app["pulse"]
    full = app["problem"].FULL
    path = planet.trace(problem.paths[0].distance, nodes=full.nodes)
    quadrature = functionals.quadrature(path, pulse.Band(min(full.periods)), full.rule)
    count = (full.nodes - 2) * full.detour_nodes * full.angle_nodes + 2
    assert quadrature.points.shape == (count, 3)
    assert quadrature.kept.mean() > 0.95
    operator = functionals.composed_operator(
        problem.space, [(quadrature.lifted, quadrature.kept_weights)]
    )
    measure = problem.prior(*problem.scale.prior_pairs[0])
    model = measure.sample(rng=np.random.default_rng(3))
    started = time.perf_counter()
    operator(model)
    assert time.perf_counter() - started < 10.0
    report = check_operator(
        operator, rng=np.random.default_rng(7), measure=measure, rtol=1e-12
    )
    assert report.worst < 1e-12


def test_the_kernel_is_hollow_on_the_ray(app, problem):
    """The ``L2`` kernel part A draws: zero on the ray itself, ``G(0) =
    0``, and largest in size off it, about six tenths of a Fresnel
    half-width out in either direction across the ray at mid-path, where
    it is negative: a faster mantle there makes the arrival earlier."""
    functionals = app["functionals"]
    path, band = problem.paths[0], problem.bands[0]
    middle = [int(np.argmin(np.abs(path.dynamic.arc_length - 0.5 * path.length)))]
    on = functionals.kernel(path, band, np.zeros(1), np.zeros(1), nodes=middle)
    assert on == pytest.approx(0.0, abs=1e-12)
    width = path.fresnel_half_width(band.period)
    offsets = np.linspace(0.0, 3.0 * width, 301)[None]
    for q1, q2 in ((offsets, 0.0 * offsets), (0.0 * offsets, offsets)):
        values = functionals.kernel(path, band, q1, q2, nodes=middle)[0]
        peak = int(np.argmax(np.abs(values)))
        assert 0.4 * width < offsets[0, peak] < 0.8 * width
        assert values[peak] < 0.0


# ------------------------------------------------------------------------- #
#                                The priors                                 #
# ------------------------------------------------------------------------- #


def test_every_prior_of_the_scale_is_on_the_one_space(problem):
    space, scale = problem.space, problem.scale
    assert space.length_scale == scale.length_scale
    assert set(problem.priors) == set(scale.prior_scales)
    assert all(prior.domain is space for prior in problem.priors.values())
    # A prior of other lengths is made on the same space, and kept.
    other = problem.prior(0.1, 0.3)
    assert other.domain is space
    assert problem.prior(0.1, 0.3) is other
    assert (
        problem.prior(scale.prior_lengths[0])
        is problem.priors[(scale.prior_lengths[0], scale.prior_lengths[0])]
    )


@pytest.mark.parametrize("index", [0, 1, 2])
def test_the_priors_pass_the_librarys_measure_checks(problem, index):
    """Each prior's draws against its covariance, and the covariance's
    traits, through the library's checks; and its pointwise standard
    deviation the scale's. Over three seeds the sample covariance's worst
    defect measured 0.045 of its scale at 4000 draws, whose standard error
    is about 0.022; 0.12 is allowed. The standard deviation at sixty points
    from the core to the surface measured within 4.0% of the scale's at
    degree 16, the truncation's doing; 8% is allowed."""
    from pygeoinf.testing import check_measure, check_traits

    space, scale = problem.space, problem.scale
    if index >= len(scale.prior_scales):
        pytest.skip("The quick scale has fewer priors.")
    key = scale.prior_scales[index]
    prior = problem.priors[key]
    check_measure(prior, rng=np.random.default_rng(11), samples=4000, rtol=0.12)
    report = check_traits(prior.covariance, rng=np.random.default_rng(5), measure=prior)
    assert report.worst < 1e-12
    rng = np.random.default_rng(3)
    count = 60
    points = np.column_stack(
        [
            np.linspace(0.551, 0.999, count),
            rng.uniform(-80.0, 80.0, count),
            rng.uniform(0.0, 360.0, count),
        ]
    )
    look = space.point_evaluation_operator(points)
    variance = np.diag(
        (look @ prior.covariance @ look.adjoint).matrix(form="components")
    )
    assert np.sqrt(variance) == pytest.approx(scale.prior_std, rel=0.08)


# ------------------------------------------------------------------------- #
#                            Part A at the quick scale                      #
# ------------------------------------------------------------------------- #


def test_part_a_writes_its_figures_and_reports_the_weight_dropped(
    app, scale, tmp_path, capsys
):
    built = app["setup"].run(scale, outdir=tmp_path)
    printed = capsys.readouterr().out
    assert "above the surface" in printed
    assert "below the core-mantle boundary" in printed
    assert "both positive" in printed
    assert (tmp_path / "rays.png").exists()
    for path in built.paths:
        assert (tmp_path / f"kernels_{path.distance:g}deg.png").exists()
