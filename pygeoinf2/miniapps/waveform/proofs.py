"""Part A: the solver, and the proofs that its derivatives are what they claim.

On the small proof grid first, then on the grid of the scale asked for: the
stencil's explicit transpose, the fused kernels against the unfused ones they
replace, and the library's check ladder on the operator -- the adjoint
identity of ``F'(m)`` and of ``F''(m)[dm, .]`` on a Sobolev box whose metric is
not the identity, and central differences of ``F`` against ``F'`` and of ``F'``
against ``F''``. The library's checks floor their scale at one; the defects
are also measured here relative to the size of the quantities compared, with
nothing floored, and the differences are taken at a ladder of steps to show
the second-order fall that marks a correct derivative. The linearization
point is a draw from the prior and so are the directions. A figure of
wavefield snapshots and a shot gather on the true model.
"""

from __future__ import annotations

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf2 import MaximumAPosteriori
from pygeoinf2.inference import ForwardProblem
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.testing import (
    check_derivative,
    check_gradient,
    check_operator,
    check_second_derivative,
    check_traits,
)

from common import (
    describe,
    output_directory,
    parse,
    part_rng,
    save,
    section,
    sponge_outline,
)
from survey import (
    DEPTH_CELLS,
    FULL,
    PROOF,
    QUICK,
    Experiment,
    Scale,
    Survey,
    box_average_operator,
    build,
    true_velocity,
)
from wave_operator import WaveformOperator

TRIALS = 5
STEPS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)


def _relative(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def _dot_test(space, data_space, operator, x, y) -> float:
    """``|(A x, y) - (x, A* y)| / (|A x| |y|)``: the defect against its scale.

    The pairing itself can be small by cancellation, so the defect is taken
    against the Cauchy-Schwarz bound on it, which is what rounding in the
    two sums is proportional to.
    """
    image = operator(x)
    left = data_space.inner_product(image, y)
    right = space.inner_product(x, operator.adjoint(y))
    return abs(left - right) / (data_space.norm(image) * data_space.norm(y))


def own_defects(
    experiment: Experiment, wave: WaveformOperator, rng: np.random.Generator
) -> dict:
    """Every identity, relative to the size of what is compared."""
    space, data_space, kernels = experiment.space, wave.data_space, wave.survey.model
    probes = experiment.probes
    point = experiment.prior.sample(rng=rng)
    shape = experiment.survey.shape
    out: dict[str, float] = {}

    u, v = rng.normal(size=shape), rng.normal(size=shape)
    lu, ltv = kernels.laplacian(u), kernels.laplacian(v, True)

    def against(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    out["stencil: L by columns against its literal sum"] = against(
        lu, kernels.laplacian(u, reference=True)
    )
    out["stencil: L^T by columns against the literal scatter"] = against(
        ltv, kernels.laplacian(v, True, reference=True)
    )
    out["stencil: (L u, v) == (u, L^T v)"] = _relative(np.sum(lu * v), np.sum(u * ltv))
    out["stencil: |L u - L^T u| / |L u|, L is not symmetric"] = float(
        np.linalg.norm(lu - kernels.laplacian(u, True)) / np.linalg.norm(lu)
    )

    direction = probes.sample(rng=rng)
    other = probes.sample(rng=rng)
    r = data_space.random(rng=rng)
    derivative = wave.derivative(point)
    second = wave.second_derivative(point, direction)
    traces = wave._shot_traces(r)
    # The unfused compositions release the cached fields, so the fused
    # kernels ask for them afresh each time.
    fused_gradient = sum(
        kernels.gradient(point, f, t)
        for f, t in zip(wave.fields(point)[0], traces, strict=True)
    )
    out["fused gradient against adjoint + correlate"] = float(
        np.abs(fused_gradient - wave.unfused_gradient(point, r)).max()
        / np.abs(fused_gradient).max()
    )
    fused_curvature = np.zeros(experiment.survey.shape)
    for f, t in zip(wave.fields(point)[0], traces, strict=True):
        linearized = kernels.born(point, f, direction)[0]
        fused_curvature += kernels.curvature(point, f, linearized, t, direction)
        kernels.release(linearized)
    unfused_curvature = wave.unfused_curvature(point, direction, r)
    out["fused curvature against adjoint + adjoint_born + correlate"] = float(
        np.abs(fused_curvature - unfused_curvature).max()
        / np.abs(fused_curvature).max()
    )
    fused_second = second(other)
    out["fused second_born against two born runs"] = float(
        np.abs(fused_second - wave.unfused_second_born(point, direction, other)).max()
        / np.abs(fused_second).max()
    )

    worst_first = worst_second = 0.0
    for _ in range(TRIALS):
        x, y = probes.sample(rng=rng), data_space.random(rng=rng)
        worst_first = max(worst_first, _dot_test(space, data_space, derivative, x, y))
        worst_second = max(worst_second, _dot_test(space, data_space, second, x, y))
    out["(F'(m) x, y) == (x, F'(m)* y), over |F'(m) x| |y|"] = worst_first
    out["(F''(m)[d, x], y) == (x, F''(m)[d, .]* y), likewise"] = worst_second

    # Central differences at a ladder of steps, the direction a prior draw.
    first = derivative(direction)
    curvature = second(other)
    taylor = []
    for step in STEPS:
        plus = space.axpy(step, direction, space.copy(point))
        minus = space.axpy(-step, direction, space.copy(point))
        numerical = (wave.value(plus) - wave.value(minus)) / (2 * step)
        numerical_second = (
            wave.derivative(plus)(other) - wave.derivative(minus)(other)
        ) / (2 * step)
        taylor.append(
            (
                step,
                float(np.linalg.norm(numerical - first) / np.linalg.norm(first)),
                float(
                    np.linalg.norm(numerical_second - curvature)
                    / np.linalg.norm(curvature)
                ),
            )
        )
    return {"defects": out, "taylor": taylor, "point": point}


def library_ladder(
    experiment: Experiment, wave: WaveformOperator, rng: np.random.Generator
) -> dict:
    """The library's checks, as a test suite would run them."""
    probes = experiment.probes
    point = experiment.prior.sample(rng=rng)
    operator = wave.operator
    reports = {
        "check_operator(F'(m))": check_operator(
            operator.derivative(point), rng=rng, measure=probes
        ),
        "check_derivative(F)": check_derivative(
            operator, point, rng=rng, measure=probes
        ),
        "check_second_derivative(F)": check_second_derivative(
            operator, point, rng=rng, measure=probes
        ),
        "check_operator(F''(m)[d, .])": check_operator(
            operator.second_derivative(point, probes.sample(rng=rng)),
            rng=rng,
            measure=probes,
        ),
    }
    if experiment.survey.boxes_inside():
        reports["check_operator(box means)"] = check_operator(
            box_average_operator(experiment.space, experiment.survey),
            rng=rng,
            measure=probes,
        )
    # The objective a Newton method minimizes, end to end: its gradient
    # against differences, and the exact Hessian's claimed symmetry, which
    # holds only if the curvature term's adjoint is right.
    clean = operator(experiment.truth)
    noise = GaussianMeasure.from_standard_deviation(
        wave.data_space, 0.05 * float(np.abs(clean).max())
    )
    problem = ForwardProblem(operator, error=noise)
    data = problem.synthetic_data(experiment.truth, rng=rng)
    objective = MaximumAPosteriori(problem, experiment.prior).objective(data)
    reports["check_gradient(MAP objective)"] = check_gradient(
        objective, point, rng=rng, measure=probes
    )
    reports["check_traits(exact Hessian)"] = check_traits(
        objective.hessian(point), rng=rng, measure=probes, rtol=1e-8
    )
    return reports


def prove(experiment: Experiment, rng: np.random.Generator, /) -> dict:
    """Both sets of checks on one grid, printed."""
    survey = experiment.survey
    wave = WaveformOperator(experiment.survey, experiment.space)
    started = time.perf_counter()
    print(
        f"grid {survey.shape[0]} x {survey.shape[1]}, {survey.scale.steps} steps, "
        f"{survey.n_shots} shots, {survey.n_receivers} receivers, "
        f"{wave.data_space.dim} data, one field {survey.field_megabytes:.0f} MB"
    )
    metric = experiment.space.metric_values
    print(
        f"model space: H^{experiment.space.order:g} box, metric from "
        f"{metric.min():.3g} to {metric.max():.3g}"
    )
    own = own_defects(experiment, wave, rng)
    print("  measured relative to the quantities compared:")
    for name, value in own["defects"].items():
        print(f"    {name:<62s} {value:.2e}")
    print("  central differences at a ladder of steps (relative error):")
    print(f"    {'step':>8s} {'F against F_prime':>18s} {'F_prime against F2':>18s}")
    for step, first, second in own["taylor"]:
        print(f"    {step:8.0e} {first:18.2e} {second:18.2e}")
    ladder = library_ladder(experiment, wave, rng)
    print("  the library's checks (scale floored at one):")
    for name, report in ladder.items():
        print(f"    {name:<34s} worst {report.worst:.2e}")
    print(
        f"  in {time.perf_counter() - started:.1f} s; kernels run: "
        f"{describe(wave.calls)}"
    )
    return {"own": own, "ladder": ladder}


def kernel_times(experiment: Experiment) -> dict:
    """One run of each kernel for one shot on the true model."""
    survey, kernels = experiment.survey, experiment.survey.model
    m = np.ascontiguousarray(experiment.truth)
    wavelet = np.ascontiguousarray(survey.wavelet[:, None])
    shot = [int(survey.source_x[survey.n_shots // 2])]
    rng = np.random.default_rng(0)
    dm = np.ascontiguousarray(experiment.probes.sample(rng=rng))
    times = {}
    t0 = time.perf_counter()
    field, data = kernels.forward(m, shot, [DEPTH_CELLS], wavelet)
    times["forward"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    linearized, _ = kernels.born(m, field, dm)
    times["born"] = time.perf_counter() - t0
    traces = np.ascontiguousarray(rng.normal(size=data.shape))
    t0 = time.perf_counter()
    kernels.gradient(m, field, traces)
    times["gradient"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    kernels.curvature(m, field, linearized, traces, dm)
    times["curvature"] = time.perf_counter() - t0
    kernels.release(field)
    kernels.release(linearized)
    return times


def reflection(scale: Scale, /, *, pad: int = 200) -> dict:
    """What the sponge sends back, against a run on a grid too big to reflect.

    Every shot over the true model, run on the scale's grid and on one padded
    by ``pad`` cells of the same medium on each absorbing side, with the same
    source and receivers: the difference is the reflected field. Its rms over
    the data of every shot is taken against the data's rms, and its peak over
    the section at the snapshot times against the peak of the field itself
    there; the worst shot is reported.
    """
    near, far = Survey(scale), Survey(scale, pad=pad)
    stride = scale.steps // 5
    wavelet = np.ascontiguousarray(near.wavelet[:, None])
    inside = near.interior()
    # The small grid's cells within the padded one, at the snapshot times.
    window = (slice(1, 5), slice(pad, pad + near.shape[0]), slice(0, near.shape[1]))
    data, reference, peaks = [], [], []
    for x_near, x_far in zip(near.source_x, far.source_x, strict=True):
        runs = []
        for survey, x in ((near, x_near), (far, x_far)):
            m = np.ascontiguousarray(1.0 / true_velocity(survey) ** 2)
            runs.append(
                survey.model.snapshots(m, [int(x)], [DEPTH_CELLS], wavelet, stride)
            )
        (kept, ours), (whole, theirs) = runs
        data.append(ours)
        reference.append(theirs)
        direct = whole[window][:, inside]
        peaks.append(np.abs(kept[1:5][:, inside] - direct).max() / np.abs(direct).max())
    data, reference = np.stack(data), np.stack(reference)
    return {
        "data": float(
            np.sqrt(np.mean((data - reference) ** 2) / np.mean(reference**2))
        ),
        "interior": float(max(peaks)),
        "times": (np.arange(1, 5) * stride + 1) * scale.dt,
        "reflected": data - reference,
        "reference": reference,
    }


def snapshots(experiment: Experiment, /, *, outdir: pathlib.Path) -> pathlib.Path:
    """Four snapshots of the middle shot over the true model, and its gather."""
    survey, kernels = experiment.survey, experiment.survey.model
    steps = survey.scale.steps
    stride = steps // 5
    shot = survey.n_shots // 2
    kept, data = kernels.snapshots(
        np.ascontiguousarray(experiment.truth),
        [int(survey.source_x[shot])],
        [DEPTH_CELLS],
        np.ascontiguousarray(survey.wavelet[:, None]),
        stride,
    )
    fig, axes = plt.subplots(
        3, 2, figsize=(10, 9.4), gridspec_kw={"height_ratios": [1, 1, 1.2]}
    )
    velocity = 1.0 / np.sqrt(experiment.truth)
    source = (survey.x[survey.source_x[shot]], survey.z[DEPTH_CELLS])
    for ax, index in zip(axes[:2].ravel(), range(1, 5)):
        snapshot = kept[index]
        limit = np.percentile(np.abs(snapshot), 99.5)
        section(ax, survey, velocity, whole=True, cmap="Greys", alpha=0.35)
        section(
            ax,
            survey,
            snapshot,
            whole=True,
            cmap="RdBu",
            vmin=-limit,
            vmax=limit,
            alpha=0.75,
        )
        sponge_outline(ax, survey)
        ax.plot(*source, "*", color="k", ms=10)
        ax.set_title(f"t = {(index * stride + 1) * survey.scale.dt:.2f} s", fontsize=10)
    gs = axes[2, 0].get_gridspec()
    for ax in axes[2]:
        ax.remove()
    gather = fig.add_subplot(gs[2, :])
    limit = np.percentile(np.abs(data), 99)
    h = survey.scale.h
    gather.imshow(
        data,
        aspect="auto",
        cmap="Greys",
        vmin=-limit,
        vmax=limit,
        extent=(
            survey.x[survey.receiver_x[0]] - h / 2,
            survey.x[survey.receiver_x[-1]] + h / 2,
            survey.times[-1],
            survey.times[0],
        ),
    )
    gather.set_xlabel("receiver x (km)")
    gather.set_ylabel("time (s)")
    gather.set_title(f"the shot gather at x = {source[0]:.2f} km")
    path = outdir / "snapshots.png"
    save(
        fig,
        path,
        title="Waveform, part A: the middle shot over the true model, on the whole "
        "grid; the section outlined, the sponge hatched",
    )
    return path


def run(scale: Scale, /, *, seed: int, outdir: pathlib.Path) -> dict:
    """The proofs on the proof grid and on the scale's grid, and the figure."""
    started = time.perf_counter()
    rng = part_rng(seed, "a")
    print(f"\n=== A. The solver and its proofs ({scale.name} scale) ===")
    print("\n-- the proof grid --")
    results = {"proof": prove(build(PROOF), rng)}
    experiment = build(scale)
    print(f"\n-- the {scale.name} grid --")
    results["scale"] = prove(experiment, rng)
    sponge = reflection(scale)
    print(
        f"the sponge, {scale.sponge} cells at rate {scale.sponge_rate:g}, against a "
        f"grid too big to reflect: what it sends back is {sponge['data']:.2%} of the "
        f"data's rms, and at most {sponge['interior']:.2%} of the field's peak in "
        "the section at the snapshot times"
    )
    results["reflection"] = sponge
    times = kernel_times(experiment)
    print(
        "one shot on the true model: "
        + ", ".join(f"{name} {value:.3f} s" for name, value in times.items())
    )
    results["times"] = times
    snapshots(experiment, outdir=outdir)
    print(f"A took {time.perf_counter() - started:.1f} s")
    return results


def main() -> None:
    args = parse(__doc__)
    run(
        QUICK if args.quick else FULL,
        seed=args.seed,
        outdir=output_directory(args.quick, args.outdir),
    )


if __name__ == "__main__":
    main()
