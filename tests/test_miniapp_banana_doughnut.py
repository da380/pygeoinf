"""
The banana-doughnut mini-app keeps working.

``miniapps/banana_doughnut`` sets the two linear functionals of a traveltime
side by side on one model space, ``dln c`` on the mantle annulus in ``H^2``:
ray theory's, point evaluation along the ray traced in the reference mantle,
and finite-frequency theory's, a quadrature of the paraxial kernel in
ray-centered coordinates about that ray, composed with point evaluation in
the same way. Checked here, at the quick scale: the quadrature in the detour
time against the Gaussian identity that reduces the one functional to the
other; the two functionals on models constant across the banana, with the
points above the free surface folded back to their mirror images and with
them dropped; the finite-frequency functional closing on the ray's as the
period falls; both through the library's operator check under a prior,
composed against assembled; the weight above the surface and the weight
dropped below the core-mantle boundary, reported apart; the full scale's
quadrature on the quick space; the priors, each with lengths of its own on
the one space, through the library's measure checks, and their
correlation from the ray's turning point read off the covariance; every
field of the scale as an option; the Sobolev gradients against the
functionals; the prior pushed forward to the two delays against draws;
every functional stacked, each theory's law of the delays as a marginal of
the prior pushed forward through the stack, and the divergence between
the two laws vanishing as the period falls, growing with the band and
falling as the prior lengthens; the cut-away's view of the ray; and the
figures and tables of parts A, B, C and D and of the priors, and every
part in order. The quick scale's priors are smoke values, long enough for
its degree of 16. The dynamic ray tracing behind the kernel is checked with
the tracer, in ``test_miniapp_raytracer.py``. Needs planetmodel, which the
ball is built on; without it the tests skip.
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
    "src.correlations",
    "src.problem",
    "src.sections",
)
SCRIPTS = ("setup", "priors", "gradients", "agreement", "pushforward", "run_all")

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
    assert "0.03,0.2 0.03,0.5" in shown


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

    Asked of the constant itself at every point where the paraxial formula
    puts it, the finite-frequency functional is the ray's times the
    Gaussian identity's sum, to rounding (measured 6e-16). Through the
    space, with the points above the surface folded back to their mirror
    images, nothing is lost at the surface and the two agree to what the
    projected constant's own departure from one makes of it at degree 16:
    1.5e-6 measured, 5e-5 allowed.
    """
    functionals = app["functionals"]
    path, band = problem.paths[0], problem.bands[0]
    quadrature = problem.quadratures[0][0]
    scale = problem.scale
    assert quadrature.surface == "fold" and quadrature.surface_fraction_absolute > 0.01
    defect = band.gaussian_defect(
        scale.detour_nodes, cutoff=scale.cutoff, frequencies=scale.frequency_nodes
    )
    ray = functionals.ray_weights(path).sum()
    exact = quadrature.integrate(lambda p: np.ones(len(p)), boundaries=False)
    assert exact / ray - 1.0 == pytest.approx(defect, abs=1e-13)
    folded = quadrature.integrate(lambda p: np.ones(len(p)))
    assert folded / ray - 1.0 == pytest.approx(defect, abs=1e-13)

    one = problem.space.project_function(lambda p: 1.0)
    through = problem.finite_frequency_operator(0)(one)[0]
    along = problem.ray_operator()(one)[0]
    assert through / along - 1.0 == pytest.approx(0.0, abs=5e-5)


def test_with_the_fold_the_identity_holds_at_every_distance_and_band(app, problem):
    """At every distance and band of the full scale, at the quick scale's
    counts: with the points above the surface folded back, a constant
    model gives the ray functional times the Gaussian identity's sum, less
    only the weight dropped below the core-mantle boundary, where the
    functional is zero by physics; the residual measured 5e-16. That
    share, up to 3.9% of the weight at 75 degrees in the widest band, is
    all that is dropped, and no folded point lands below the core."""
    functionals, planet, pulse = app["functionals"], app["planet"], app["pulse"]
    full = app["problem"].FULL
    rule = problem.scale.rule
    reached_core = False
    for distance in full.distances:
        path = planet.trace(distance, nodes=problem.scale.nodes)
        ray = functionals.ray_weights(path).sum()
        for period in full.periods:
            band = pulse.Band(period)
            defect = band.gaussian_defect(
                rule.detours, cutoff=rule.cutoff, frequencies=rule.frequencies
            )
            quadrature = functionals.quadrature(path, band, rule)
            value = quadrature.integrate(lambda p: np.ones(len(p))) / ray - 1.0
            core = quadrature.core_fraction
            assert value == pytest.approx(defect - core * (1.0 + defect), abs=1e-14)
            assert quadrature.dropped_fraction == core
            assert quadrature.folded_below_core == 0
            reached_core |= quadrature.core_fraction_absolute > 0.01
    assert reached_core


def test_clipping_drops_the_weight_above_the_surface(app, problem):
    """With ``surface="clip"`` the points above the surface are dropped
    with their weight, and the functional misses on a constant by that
    share: of the quick ray in its band -0.311% of the weight (1.05% of the
    absolute weight), through the space to 5.3e-6 at degree 16, 5e-5
    allowed."""
    functionals = app["functionals"]
    path, band = problem.paths[0], problem.bands[0]
    clipped = functionals.quadrature(path, band, problem.scale.rule, surface="clip")
    folded = problem.quadratures[0][0]
    assert clipped.surface_fraction == folded.surface_fraction
    assert clipped.surface_fraction == pytest.approx(-0.00311, abs=1e-5)
    assert clipped.surface_fraction_absolute == pytest.approx(0.0105, abs=1e-4)
    assert clipped.dropped_fraction == clipped.surface_fraction
    assert clipped.kept.sum() == folded.kept.sum() - clipped.above.sum()
    ray = functionals.ray_weights(path).sum()
    exact = clipped.integrate(lambda p: np.ones(len(p)))
    defect = folded.integrate(lambda p: np.ones(len(p))) / ray - 1.0
    assert exact / ray - 1.0 == pytest.approx(
        defect - clipped.dropped_fraction * (1.0 + defect), abs=1e-14
    )
    one = problem.space.project_function(lambda p: 1.0)
    operator = functionals.composed_operator(
        problem.space, [(clipped.lifted, clipped.kept_weights)]
    )
    through = operator(one)[0] / problem.ray_operator()(one)[0] - 1.0
    assert through == pytest.approx(-clipped.dropped_fraction, abs=5e-5)
    with pytest.raises(ValueError):
        functionals.quadrature(path, band, problem.scale.rule, surface="mirror")


def test_on_a_radial_model_of_long_wavelength_they_close_as_the_period_falls(
    app, problem, falling
):
    """``cos(2 pi (1 - r) / 0.9)``, asked at every point exactly where the
    paraxial formula puts it: what separates the two functionals is the model's curvature
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
            abs(quadrature.integrate(radial, boundaries=False) - ray) / path.travel_time
            for quadrature in falling
        ]
    )
    assert gaps[0] < 2e-5
    assert np.all(gaps[:-1] / gaps[1:] > 3.0)


def test_on_a_smooth_draw_they_close_as_the_period_falls(app, problem, falling):
    """Draws from the smoothest prior, through the space: the gap between
    the two functionals over the prior's standard deviation of the ray
    delay, in bands of falling period, with the points above the surface
    folded back and dropped. At degree 16 a draw holds nothing on the scale
    of the Fresnel zone, so the paraxial approach to the ray is second
    order, falling about fourfold as the period halves, and the weight
    dropped at the surface, first order and uneven between bands, stalls
    the gap when it is dropped. Over the twenty draws taken here the median
    gap measured 3.1e-4, 7.8e-5, 2.0e-5, 4.6e-6 folded and 1.2e-3, 3.2e-4,
    3.6e-4, 8.7e-6 clipped: folded, it is asked to fall at least threefold
    each time and to stay below 2e-3; at a quarter of the quick band's
    period the clipped one is asked to be five times the folded one
    (eighteen measured)."""
    functionals, pulse = app["functionals"], app["pulse"]
    space, scale = problem.space, problem.scale
    prior = problem.prior(max(scale.prior_lengths))
    ray = problem.ray_operator()
    spread = np.sqrt(
        (ray @ prior.covariance @ ray.adjoint).matrix(form="components")[0, 0]
    )
    path = problem.paths[0]
    clipped = [
        functionals.quadrature(path, pulse.Band(period), scale.rule, surface="clip")
        for period in FALLING
    ]

    def operators(quadratures):
        return [
            functionals.composed_operator(space, [(q.lifted, q.kept_weights)])
            for q in quadratures
        ]

    folded_operators, clipped_operators = operators(falling), operators(clipped)
    rng = np.random.default_rng(scale.seed)
    folded_gaps, clipped_gaps = [], []
    for _ in range(20):
        model = prior.sample(rng=rng)
        delay = ray(model)[0]
        folded_gaps.append([abs(op(model)[0] - delay) for op in folded_operators])
        clipped_gaps.append([abs(op(model)[0] - delay) for op in clipped_operators])
    folded = np.median(folded_gaps, axis=0) / spread
    clipped = np.median(clipped_gaps, axis=0) / spread
    assert folded[0] < 2e-3
    assert np.all(folded[:-1] / folded[1:] > 3.0)
    assert clipped[2] > 5.0 * folded[2]


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


def test_the_weight_above_the_surface_shrinks_with_the_band(problem, falling):
    """The paraxial kernel knows nothing of the free surface, and the points
    of the quadrature above it are folded back to their mirror images. The
    signed fraction of the weight there cancels between rings of either sign
    and moves with the nodes along the ray; the absolute one is steady,
    1.05% in the quick band, and falls with the period: 1.05%, 0.34%,
    0.061%, 0.0007%. Folded, none of it is lost. The quick ray turns 0.27
    above the core-mantle boundary, and in these bands no point reaches
    below it."""
    quadrature = problem.quadratures[0][0]
    assert quadrature.kept.all() and quadrature.kept.sum() == len(quadrature.lifted)
    assert abs(quadrature.surface_fraction) <= quadrature.surface_fraction_absolute
    assert quadrature.surface_fraction_absolute < 0.03
    for q in falling:
        assert q.core_fraction == 0.0 and q.core_fraction_absolute == 0.0
        assert q.dropped_fraction == 0.0
        assert np.all(np.linalg.norm(q.placed, axis=1) <= 1.0 + 1e-12)
    absolute = np.array([q.surface_fraction_absolute for q in falling])
    assert np.all(np.diff(absolute) < 0.0)
    assert absolute[-1] < 1e-4


def test_the_two_boundaries_are_reported_apart(app, problem):
    """In the full scale's widest band the quick ray's doughnut reaches
    below the core-mantle boundary, where the functional is zero by
    physics, as well as above the surface: 0.021% and 10.3% of the absolute
    weight. Folded, the first is all that is dropped; clipped, the two add
    up to what is."""
    functionals, pulse = app["functionals"], app["pulse"]
    full = app["problem"].FULL
    path, rule = problem.paths[0], problem.scale.rule
    band = pulse.Band(max(full.periods))
    folded = functionals.quadrature(path, band, rule)
    clipped = functionals.quadrature(path, band, rule, surface="clip")
    for quadrature in (folded, clipped):
        assert 0.0 < quadrature.core_fraction_absolute
        assert quadrature.core_fraction_absolute < quadrature.surface_fraction_absolute
    assert folded.core_fraction == clipped.core_fraction
    assert folded.dropped_fraction == folded.core_fraction
    assert clipped.core_fraction + clipped.surface_fraction == pytest.approx(
        clipped.dropped_fraction, abs=1e-15
    )
    assert (
        clipped.core_fraction_absolute + clipped.surface_fraction_absolute
    ) == pytest.approx(clipped.dropped_fraction_absolute, abs=1e-15)
    radii = clipped.radii
    assert np.array_equal(
        ~clipped.kept, functionals.above_surface(radii) | functionals.below_core(radii)
    )
    assert np.array_equal(~folded.kept, functionals.below_core(radii))


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
    deviation the scale's. Over two seeds the sample covariance's worst
    defect measured 0.027 of its scale at 4000 draws, whose standard error
    is about 0.022; 0.12 is allowed. The standard deviation at sixty points
    from the core to the surface measured within 4.5% of the scale's at
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
#                  The Sobolev gradients and the push-forwards              #
# ------------------------------------------------------------------------- #


@pytest.mark.parametrize("band", [None, 0])
def test_the_sobolev_gradient_reproduces_the_functional(problem, band):
    """The gradient's inner product with a draw is the functional applied:
    over five draws from each prior the worst gap measured 1.3e-14 of the
    delay's standard deviation; 1e-11 is allowed."""
    space = problem.space
    operator = (
        problem.ray_operator()
        if band is None
        else problem.finite_frequency_operator(band)
    )
    gradient = problem.representer(0, band)
    for key in problem.scale.prior_scales:
        prior = problem.prior(*key)
        spread = np.sqrt(space.inner_product(prior.covariance(gradient), gradient))
        rng = np.random.default_rng(5)
        for _ in range(5):
            model = prior.sample(rng=rng)
            assert (
                abs(space.inner_product(gradient, model) - operator(model)[0])
                <= 1e-11 * spread
            )


def test_the_pair_is_the_two_functionals_stacked(problem):
    """The operator onto ``R^2`` from its two rows and the one composed of
    point evaluation at both functionals' points push every prior forward
    to the same covariance, to 1.1e-14 of its size measured."""
    rows = problem.pair_operator(0, 0)
    composed = problem.pair_operator(0, 0, assembled=False)
    model = problem.prior(*problem.scale.prior_pairs[0]).sample(
        rng=np.random.default_rng(2)
    )
    pair = rows(model)
    assert pair[0] == pytest.approx(problem.ray_operator()(model)[0], rel=1e-12)
    assert pair[1] == pytest.approx(
        problem.finite_frequency_operator(0)(model)[0], rel=1e-12
    )
    for key in problem.scale.prior_scales:
        prior = problem.prior(*key)
        one = prior.push_forward(rows).covariance.matrix(form="components")
        other = prior.push_forward(composed).covariance.matrix(form="components")
        assert np.abs(one - other).max() <= 1e-12 * np.abs(one).max()


def test_the_push_forward_matches_draws_of_the_two_delays(app, problem):
    """The covariance of the two delays from the push-forward against their
    sample covariance over 400 draws of the shortest prior. Over ten seeds
    the worst entry missed by 0.20 of ``sqrt(C_ii C_jj)``, and the variance
    of the difference ``dT_FF - dT_ray`` by 0.18 of itself, against a
    standard error of about 0.07 for both; 0.3 is allowed."""
    agreement = app["agreement"]
    key = (min(problem.scale.prior_lengths),) * 2
    prior = problem.prior(*key)
    pair = problem.pair_operator(0, 0)
    covariance = agreement.pushed_covariance(problem, 0, 0, key)
    rng = np.random.default_rng(problem.scale.seed)
    delays = np.array([pair(prior.sample(rng=rng)) for _ in range(400)])
    sample = delays.T @ delays / len(delays)
    scale = np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
    assert np.all(np.abs(sample - covariance) <= 0.3 * scale)
    difference = covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1]
    spread = np.mean((delays[:, 1] - delays[:, 0]) ** 2)
    assert spread == pytest.approx(difference, rel=0.3)


def test_the_delays_agree_more_as_the_prior_lengthens(app, problem):
    """Under isotropic priors of nominal lengths 0.2, 0.3 and 0.4, smoke
    values that degree 16 resolves, the finite-frequency delay's standard
    deviation given the ray delay, over its own, measured 9.3e-4, 7.6e-4
    and 6.8e-4, falling with the length, and the longest's correlation 1 -
    2.3e-7; 1 - 1e-6 is asked. The field's correlation from the ray's
    turning point falls to 1/e at 0.29, 0.37 and 0.44 along the shell. The
    statistics read the same covariance two ways."""
    agreement = app["agreement"]
    lengths = (0.2, 0.3, 0.4)
    found = agreement.sweep(
        problem, [0], priors=[(length, length) for length in lengths]
    )
    conditional = np.array([row.conditional_std_ratio for row in found])
    assert np.all(np.diff(conditional) < 0.0)
    assert found[-1].correlation > 1.0 - 1e-6
    for row in found:
        assert row.conditional_std_ratio == pytest.approx(
            np.sqrt(1.0 - row.correlation**2), rel=1e-6
        )
        assert row.reach == row.shell_distance
        assert row.in_wavelengths == pytest.approx(row.reach / row.wavelength)
    distances = [row.shell_distance for row in found]
    assert np.all(np.diff(distances) > 0.0)


# ------------------------------------------------------------------------- #
#                                 The priors                                #
# ------------------------------------------------------------------------- #


def test_a_priors_correlation_is_read_off_its_covariance(problem):
    """The correlation from the ray's turning point is one where it starts,
    and falls to 1/e at the distance reported, along the shell and along
    the radius; the quick scale's smoke priors decorrelate along the shell
    at 0.29 and 0.44 for nominal 0.2 and 0.4, and the pair of 0.1 radial
    under 0.4 lateral at 0.18 along the radius and 0.61 along the shell."""
    from scipy.interpolate import interp1d

    lengths = sorted(problem.scale.prior_lengths)
    shell = []
    for key in problem.scale.prior_scales:
        found = problem.correlation(key)
        assert found.shell[0] == pytest.approx(1.0, abs=1e-12)
        assert found.down[0] == pytest.approx(1.0, abs=1e-12)
        reading = interp1d(found.shell_offsets, found.shell)(found.shell_distance)
        assert reading == pytest.approx(np.exp(-1.0), abs=1e-3)
        if key[0] == key[1]:
            shell.append(found.shell_distance)
        else:
            assert found.radial_distance < found.shell_distance
    assert len(shell) == len(lengths) and np.all(np.diff(shell) > 0.0)
    assert problem.correlation(problem.scale.prior_scales[0]) is problem.correlation(
        problem.scale.prior_scales[0]
    )


def test_the_shortest_prior_decorrelates_inside_the_widest_fresnel_zone(app):
    """The full scale's priors on a space of the full scale's length scale,
    at degree 96 rather than 160 to keep it cheap: the measured distance
    along the shell grows with the nominal length, 0.059, 0.085, 0.130,
    0.214 and 0.365 for 0.02 to 0.32 (0.058 to 0.37 at degree 160), and
    the shortest lies well inside the widest band's Fresnel half-width at
    45 degrees, 0.159, which is what lets the comparison reach where ray
    theory breaks down."""
    import dataclasses

    problem_module = app["problem"]
    full = problem_module.FULL
    scale = dataclasses.replace(full, lmax=96, distances=(45.0,), periods=(0.096,))
    problem = problem_module.build(scale=scale, quiet=True)
    width = problem.paths[0].fresnel_half_width(0.096)
    lengths = sorted(full.prior_lengths)
    shell = [problem.correlation((length, length)).shell_distance for length in lengths]
    assert np.all(np.diff(shell) > 0.0)
    assert shell[0] < 0.5 * width
    assert all(d > length for d, length in zip(shell, lengths))


# ------------------------------------------------------------------------- #
#                 The two theories' laws of the delays, and the cut-away    #
# ------------------------------------------------------------------------- #

#: Bands from the narrowest the Gaussian identity is checked in up to the
#: widest of the full scale, for the divergence between the two laws.
WIDENING = (0.0015, 0.006, 0.024, 0.096)


@pytest.fixture(scope="module")
def widening(app, scale):
    """The quick problem in bands of widening period, at its one distance."""
    return app["problem"].build(
        scale=dataclasses.replace(scale, periods=WIDENING), quiet=True
    )


def test_the_stack_is_every_functional_in_order(widening):
    """Ray theory's rows first, then finite-frequency theory's band by
    band, each block the operator of that theory and band."""
    count = len(widening.paths)
    stacked = widening.stacked_operator().matrix(form="galerkin")
    assert stacked.shape == (count * (1 + len(WIDENING)), widening.space.dim)
    blocks = [widening.ray_operator(assembled=True)] + [
        widening.finite_frequency_operator(band, assembled=True)
        for band in range(len(WIDENING))
    ]
    for k, block in enumerate(blocks):
        assert np.array_equal(
            stacked[count * k : count * (k + 1)], block.matrix(form="galerkin")
        )


def test_each_theorys_law_is_the_push_forward_through_its_operator(app, widening):
    """The joint law pushed forward onto one theory's block is the prior
    pushed forward through that theory's stacked operator: 1.2e-16 of the
    covariance's size measured."""
    pushforward = app["pushforward"]
    key = pushforward.chosen(widening)[0]
    prior = widening.prior(*key)
    joint = pushforward.joint_law(widening, key)
    for band in range(len(WIDENING)):
        ray, finite = pushforward.laws(widening, joint, band)
        for law, operator in (
            (ray, widening.ray_operator(assembled=True)),
            (finite, widening.finite_frequency_operator(band, assembled=True)),
        ):
            direct = prior.push_forward(operator).covariance.matrix(form="components")
            got = law.covariance.matrix(form="components")
            assert np.abs(got - direct).max() <= 1e-12 * np.abs(direct).max()


def test_the_divergence_grows_with_the_band_and_falls_as_the_prior_lengthens(
    app, widening
):
    """``KL(FF || ray)`` and ``KL(ray || FF)`` between the two laws of the
    quick ray's delay. Under the longest prior they measured 1.4e-12,
    3.7e-10, 9.0e-8 and 4.9e-6 as the period quadruples from 0.0015: the
    two theories agree there to what the narrowest band leaves, 1e-11
    allowed, and the divergence grows with the band, by 266 between the two
    narrowest, the fourth power of the ratio of their periods, 256, as the
    gap between the delays falls with the square of the period; at least a
    hundredfold is asked. Where it is that small the two directions agree,
    identical as measured, to 1e-3 of each other allowed. Under the shorter
    isotropic prior, whose correlation falls to 1/e at 0.29 along the shell
    against the longer's 0.44, the divergence is larger in every band,
    measured 1.9, 2.0, 2.0 and 1.3 times; and it grows with the band under
    every prior."""
    pushforward = app["pushforward"]
    keys = pushforward.chosen(widening)
    found, _ = pushforward.divergences(widening, keys, draws=0)
    by_prior = {key: [row for row in found if row.key == key] for key in keys}
    for rows in by_prior.values():
        forward = np.array([row.kl_ff_ray for row in rows])
        backward = np.array([row.kl_ray_ff for row in rows])
        assert np.all(np.diff(forward) > 0.0) and np.all(np.diff(backward) > 0.0)
        assert all(np.isnan(row.sample_correlation).all() for row in rows)
    smooth, rough = keys[0], keys[1]
    assert (
        widening.correlation(smooth).shell_distance
        > widening.correlation(rough).shell_distance
    )
    narrowest = by_prior[smooth][0]
    assert narrowest.kl_ff_ray < 1e-11
    assert narrowest.kl_ray_ff == pytest.approx(narrowest.kl_ff_ray, rel=1e-3)
    assert by_prior[smooth][1].kl_ff_ray > 100.0 * narrowest.kl_ff_ray
    for longer, shorter in zip(by_prior[smooth], by_prior[rough]):
        assert shorter.kl_ff_ray > longer.kl_ff_ray
        assert shorter.kl_ray_ff > longer.kl_ray_ff


def test_the_draws_scatter_as_the_push_forward_says(app, problem):
    """The sample correlation of the two delays over the quick scale's
    draws against the push-forward's, by Fisher's transform, whose standard
    error is ``1 / sqrt(n - 3)``: over ten seeds and the three priors the
    worst measured 2.3 standard errors; 4 is allowed."""
    pushforward = app["pushforward"]
    keys = pushforward.chosen(problem)
    draws = problem.scale.draws
    found, delays = pushforward.divergences(problem, keys, draws=draws)
    for row in found:
        assert row.draws == draws and delays[row.key].shape[0] == draws
        for sampled, exact in zip(row.sample_correlation, row.correlation):
            z = (np.arctanh(sampled) - np.arctanh(exact)) * np.sqrt(draws - 3)
            assert abs(z) < 4.0


def test_the_cutaway_shows_the_rays_first_half_on_the_floor_of_the_cut(app, problem):
    """For the scale's deepest ray, with the wedge north of the equator from
    a quarter of the way round west of the ray's turning point to it,
    under planetmodel's default view, the floor of the cut is the ray's
    plane from the source to mid-path: every node of that half is seen,
    and none of the other half, which lies inside the body. The projection
    is the picture's: the nodes lie within the planet's disc."""
    gradients = app["gradients"]
    index = gradients.deepest(problem)
    path = problem.paths[index]
    turn = gradients.turning_point(path)[2]
    nodes = path.dynamic.frame.point
    longitudes = np.degrees(np.arctan2(nodes[:, 1], nodes[:, 0]))
    geometry = gradients.cutaway_geometry(
        problem.space,
        problem.representer(index),
        longitudes=(turn - gradients.WEDGE_OPENING, turn),
        latitudes=(0.0, 90.0),
        inner=0.55,
    )
    seen = geometry.visible(nodes)
    assert seen[longitudes < turn - 1e-9].all()
    assert not seen[longitudes > turn + 1e-9].any()
    x, y = geometry.project(nodes)
    assert np.all(np.hypot(x, y) <= 1.0 + 1e-12)


# ------------------------------------------------------------------------- #
#                            Every part, quick                              #
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


def test_the_priors_part_writes_its_figure_and_table(app, problem, tmp_path, capsys):
    import csv

    found = app["priors"].run(problem.scale, outdir=tmp_path, problem=problem)
    capsys.readouterr()
    assert found["figure"].exists()
    lines = found["table"].read_text().splitlines()
    assert lines[0].startswith("# Banana-doughnut, the priors")
    read = list(csv.DictReader(line for line in lines if not line.startswith("#")))
    assert len(read) == len(problem.scale.prior_scales)
    for row, key in zip(read, problem.scale.prior_scales):
        assert float(row["shell_distance"]) == pytest.approx(
            problem.correlation(key).shell_distance, rel=1e-5
        )


def test_part_b_writes_a_figure_per_band_prior_and_section(
    app, problem, tmp_path, capsys
):
    """A figure for each band, each prior and the space's own gradient, on
    each section, the middle band under the middle isotropic prior first.
    At degree 16 nothing resolves the doughnut, and every field is filled
    in on the ray: ray theory's and finite-frequency theory's measured 0.95
    or more of their extremes across it; 0.9 is asked."""
    gradients = app["gradients"]
    numbers = gradients.run(problem.scale, outdir=tmp_path, problem=problem)
    printed = capsys.readouterr().out
    assert "Part B" in printed
    scale = problem.scale
    count = len(scale.periods) * (len(scale.prior_scales) + 1) * 2
    assert len(numbers["figures"]) == count
    assert all(path.exists() and path.parent == tmp_path for path in numbers["figures"])
    band, key = gradients.first(problem)
    distance = problem.paths[problem.reference].distance
    assert numbers["figures"][0].name == (
        f"gradients_across_{distance:g}deg_T{problem.bands[band].period:g}_"
        f"{gradients.prior_tag(key)}.png"
    )
    for fills in numbers["fills"].values():
        assert all(fill > 0.9 for fill in fills)
    cutaway = numbers["cutaway"]
    deepest = problem.paths[gradients.deepest(problem)].distance
    assert deepest == max(scale.distances)
    assert cutaway.exists() and cutaway.parent == tmp_path
    assert cutaway.name == (
        f"gradients_cutaway_{deepest:g}deg_T{problem.bands[band].period:g}_"
        f"{gradients.prior_tag(key)}.png"
    )


def test_part_c_writes_its_figures_and_table(app, problem, tmp_path, capsys):
    import csv

    agreement = app["agreement"]
    rows = agreement.run(problem.scale, outdir=tmp_path, problem=problem)
    capsys.readouterr()
    scale = problem.scale
    count = len(scale.distances) * len(scale.periods) * len(scale.prior_scales)
    assert len(rows) == count
    table = tmp_path / agreement.TABLE
    lines = table.read_text().splitlines()
    assert lines[0].startswith("# Banana-doughnut, part C")
    read = list(csv.DictReader(line for line in lines if not line.startswith("#")))
    assert len(read) == count
    first = read[0]
    assert float(first["correlation"]) == pytest.approx(rows[0].correlation)
    assert float(first["shell_distance_wavelengths"]) == pytest.approx(
        rows[0].shell_distance / rows[0].wavelength
    )
    assert float(first["length_scale_wavelengths"]) == pytest.approx(
        scale.length_scale / rows[0].wavelength
    )
    for distance in scale.distances:
        assert (tmp_path / f"agreement_{distance:g}deg.png").exists()
        assert (tmp_path / f"agreement_curves_{distance:g}deg.png").exists()
        assert (tmp_path / f"agreement_fresnel_{distance:g}deg.png").exists()


def test_part_d_writes_its_figures_and_table(app, problem, tmp_path, capsys):
    """A figure for each of part D's priors, the smoothest and the roughest
    isotropic one and the pair, and a row of the table for each prior and
    band."""
    import csv

    pushforward = app["pushforward"]
    found = pushforward.run(problem.scale, outdir=tmp_path, problem=problem)
    printed = capsys.readouterr().out
    assert "Part D" in printed
    keys = pushforward.chosen(problem)
    assert keys == [(0.4, 0.4), (0.2, 0.2), (0.1, 0.4)]
    assert [path.name for path in found["figures"]] == [
        "pushforward_prior0.4.png",
        "pushforward_prior0.2.png",
        "pushforward_prior0.1x0.4.png",
    ]
    assert all(path.exists() for path in found["figures"])
    lines = found["table"].read_text().splitlines()
    assert lines[0].startswith("# Banana-doughnut, part D")
    read = list(csv.DictReader(line for line in lines if not line.startswith("#")))
    assert len(read) == len(keys) * len(problem.scale.periods)
    for row, numbers in zip(read, found["divergences"]):
        assert float(row["kl_ff_ray"]) == pytest.approx(numbers.kl_ff_ray, rel=1e-9)
        assert int(row["draws"]) == problem.scale.draws


def test_run_all_runs_every_part_on_one_problem(app, scale, tmp_path, capsys):
    results = app["run_all"].run_all(scale, outdir=tmp_path)
    printed = capsys.readouterr().out
    assert set(results) == {"a", "priors", "b", "c", "d"}
    assert "Every part" in printed
    assert (tmp_path / "rays.png").exists()
    assert (tmp_path / "pushforward.csv").exists()
    assert results["b"]["cutaway"].exists()
