"""Part F: the operator with its grid split across MPI ranks.

Every rank runs this script, under the ``mpiexec`` of the MPI installation
``acoustic_mpi`` was built against::

    MPICXX=/path/to/mpicxx python cpp/build.py --mpi
    mpiexec -n 3 python mpi_run.py --quick

Each rank owns a block of the grid's columns and the fields on it; rank 0
also runs the library, and the others wait in ``serve()`` for its calls, each
of which scatters the model, broadcasts the traces, steps with two columns of
halo exchanged between neighbours, and gathers the data and the gradients
back to rank 0. The operator, the estimators and the parts above them are
the serial ones, unchanged.

Done in turn: on the proof grid, the distributed operator against the serial
one and the library's check ladder through it; on the grid of the scale
asked for, the same comparison, the time of one application of each kind
against the serial operator's, and part C's Gauss-Newton MAP, against the
mode part C saved serially if it is there. The numbers are also written to
``mpi_<scale>.json`` beside the figures. ``acoustic_mpi`` is imported from the
app's ``src/``, where ``cpp/build.py --mpi`` puts it.
"""

from __future__ import annotations

import contextlib
import json
import pathlib
import sys
import time
import traceback
from typing import Any

import numpy as np

from pygeoinf2 import MaximumAPosteriori

from proofs import library_ladder
from src.common import parse
from src.inversion import modes_path, newton
from src.survey import FULL, PROOF, QUICK, Experiment, Scale, build
from src.wave_operator import WaveformOperator
from synthetic import Setup, setup

try:
    import acoustic_mpi
except ImportError:  # the built module sits in the app's src/
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
    try:
        import acoustic_mpi
    finally:
        sys.path.pop(0)


def _relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


@contextlib.contextmanager
def serving(model: Any):
    """Rank 0's side of a model: the other ranks serve it until this ends.

    Anything raised here would leave them waiting, so it aborts the run.
    """
    try:
        yield
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        acoustic_mpi.abort(1)
    model.stop()


def agreement(distributed: Experiment, /, *, rng: np.random.Generator) -> dict:
    """The distributed operator against the serial one, at a prior draw."""
    serial = build(distributed.survey.scale)
    ours = WaveformOperator(distributed.survey, distributed.space)
    theirs = WaveformOperator(serial.survey, serial.space)
    point = distributed.prior.sample(rng=rng)
    d = distributed.probes.sample(rng=rng)
    e = distributed.probes.sample(rng=rng)
    r = ours.data_space.random(rng=rng)
    out = {
        "F(m)": _relative(ours.value(point), theirs.value(point)),
        "F'(m) d": _relative(ours.derivative(point)(d), theirs.derivative(point)(d)),
        "F'(m)* r": _relative(
            ours.derivative(point).adjoint(r), theirs.derivative(point).adjoint(r)
        ),
        "F''(m)[d, e]": _relative(
            ours.second_derivative(point, d)(e), theirs.second_derivative(point, d)(e)
        ),
        "F''(m)[d, .]* r": _relative(
            ours.second_derivative(point, d).adjoint(r),
            theirs.second_derivative(point, d).adjoint(r),
        ),
    }
    ours.clear()
    theirs.clear()
    print("  against the serial operator, largest relative difference:")
    for name, value in out.items():
        print(f"    {name:<18s} {value:.1e}")
    return out


def timings(distributed: Experiment, /) -> dict:
    """One application of each kind, distributed and serial, in seconds."""
    scale = distributed.survey.scale
    serial = build(scale)
    rows = {}
    for label, experiment, jobs in (
        ("ranks", distributed, None),
        ("serial", serial, scale.n_jobs),
    ):
        wave = WaveformOperator(
            experiment.survey,
            experiment.space,
            n_jobs=jobs,
            curvature_jobs=scale.curvature_jobs if jobs else None,
        )
        point = experiment.background
        d = experiment.probes.sample(rng=np.random.default_rng(0))
        r = wave.data_space.random(rng=np.random.default_rng(1))
        times = {}
        t0 = time.perf_counter()
        wave.fields(point)
        times["forward"] = time.perf_counter() - t0
        jacobian = wave.derivative(point)
        t0 = time.perf_counter()
        jacobian(d)
        times["F'"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        jacobian.adjoint(r)
        times["F'*"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        wave.second_derivative(point, d).adjoint(r)
        times["F''*"] = time.perf_counter() - t0
        wave.clear()
        rows[label] = times
    shots = distributed.survey.n_shots
    threads = f"{scale.n_jobs} threads" if scale.n_jobs else "one thread"
    print(f"  one application, seconds ({shots} shots; serially on {threads}):")
    print(f"    {'':<8s} " + " ".join(f"{name:>8s}" for name in rows["ranks"]))
    for label, times in rows.items():
        print(f"    {label:<8s} " + " ".join(f"{v:8.3f}" for v in times.values()))
    return rows


def on_the_proof_grid(experiment: Experiment, /, *, rng: np.random.Generator) -> dict:
    """The comparison and the check ladder on the proof grid."""
    model = experiment.survey.model
    print(f"\n-- the proof grid, its columns split at {list(model.column_starts)} --")
    match = agreement(experiment, rng=rng)
    wave = WaveformOperator(experiment.survey, experiment.space)
    ladder = library_ladder(experiment, wave, rng)
    wave.clear()
    print("  the library's checks through the distributed operator:")
    for name, report in ladder.items():
        print(f"    {name:<34s} worst {report.worst:.2e}")
    return {"agreement": match, "ladder": {k: r.worst for k, r in ladder.items()}}


def on_the_scale(
    state: Setup, /, *, outdir: pathlib.Path, rng: np.random.Generator
) -> dict:
    """The comparison, the timings and the Gauss-Newton MAP on the scale's grid."""
    experiment = state.experiment
    survey = experiment.survey
    scale = survey.scale
    model = survey.model
    columns = np.diff(model.column_starts)
    print(
        f"\n-- the {scale.name} grid, blocks of {columns.min()} to {columns.max()} "
        f"columns: a field is {survey.field_megabytes * columns.max() / survey.shape[0]:.0f}"
        f" MB on a rank, {survey.field_megabytes:.0f} MB whole --"
    )
    match = agreement(experiment, rng=rng)
    rows = timings(experiment)
    optimizer = newton(experiment, scale)
    t0 = time.perf_counter()
    result = MaximumAPosteriori(
        state.first_order, experiment.prior, optimizer=optimizer, gauss_newton=True
    ).solve(state.data, start=experiment.background)
    seconds = time.perf_counter() - t0
    error = experiment.relative_error(result.minimizer)
    print(
        f"  Gauss-Newton MAP across the ranks: {result.iterations} iterations, value "
        f"{result.value:.2f}, relative error {error:.3f}, {seconds:.1f} s"
    )
    saved = modes_path(outdir, scale)
    difference = None
    if saved.exists():
        difference = _relative(result.minimizer, np.load(saved)["gauss_newton"])
        print(f"  against the mode part C saved serially: {difference:.1e}")
    state.wave.clear()
    return {
        "agreement": match,
        "timings": rows,
        "iterations": result.iterations,
        "value": result.value,
        "error": error,
        "seconds": seconds,
        "difference": difference,
    }


def lead(scale: Scale, /, *, outdir: pathlib.Path) -> dict:
    """Rank 0: everything, in the order the other ranks serve it."""
    started = time.perf_counter()
    rng = np.random.default_rng([scale.seed, ord("f")])
    print(f"\n=== F. The grid split across {acoustic_mpi.size()} MPI ranks ===")
    results: dict[str, Any] = {"ranks": acoustic_mpi.size(), "scale": scale.name}
    proof = build(PROOF, backend=acoustic_mpi)
    with serving(proof.survey.model):
        results["proof"] = on_the_proof_grid(proof, rng=rng)
    state = setup(scale, quiet=True, backend=acoustic_mpi)
    with serving(state.experiment.survey.model):
        results["grid"] = on_the_scale(state, outdir=outdir, rng=rng)
    results["seconds"] = time.perf_counter() - started
    path = outdir / f"mpi_{scale.name}.json"
    path.write_text(json.dumps(results, indent=1, default=float))
    print(f"F took {results['seconds']:.1f} s; numbers written to {path}")
    return results


def follow(scale: Scale, /) -> None:
    """The other ranks: serve the proof grid's model, then the scale's."""
    for grid in (PROOF, scale):
        build(grid, backend=acoustic_mpi).survey.model.serve()


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    scale = args.scale
    if acoustic_mpi.rank() == 0:
        # Building a model makes no call on the other ranks, so a failure
        # before the first `serving` would leave them waiting: abort instead.
        try:
            lead(scale, outdir=args.outdir)
        except BaseException:
            traceback.print_exc()
            sys.stdout.flush()
            acoustic_mpi.abort(1)
    else:
        follow(scale)
    acoustic_mpi.finalize()


if __name__ == "__main__":
    main()
