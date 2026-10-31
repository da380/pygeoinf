"""Every part in order: A, the priors, B, C and D, on one problem.

``python run_all.py`` for the full scale, ``--quick`` for the small one. The
scale's fields are options, applied to the chosen preset. The problem is
built once, by part A, and handed to every part after it, so that the
functionals' rows and the priors' correlations one part forms serve the
rest; each part's time is printed at the end.
"""

from __future__ import annotations

import pathlib
import time

import agreement
import gradients
import priors
import pushforward
import setup
from src.common import parse
from src.problem import FULL, QUICK, Scale


def run_all(scale: Scale, /, *, outdir: pathlib.Path) -> dict:
    """Run the parts in order and return what each produced."""
    started = time.perf_counter()
    times: dict[str, float] = {}
    results: dict = {}

    t0 = time.perf_counter()
    problem = setup.run(scale, outdir=outdir)
    results["a"], times["A, the setup"] = problem, time.perf_counter() - t0
    for key, name, part in (
        ("priors", "the priors", priors),
        ("b", "B, the kernels", gradients),
        ("c", "C, the agreement", agreement),
        ("d", "D, the push-forwards", pushforward),
    ):
        t0 = time.perf_counter()
        results[key] = part.run(scale, outdir=outdir, problem=problem)
        times[name] = time.perf_counter() - t0
    print(f"\n=== Every part ({scale.name} scale) ===")
    for name, seconds in times.items():
        print(f"  {name}: {seconds:.1f} s")
    print(f"total {time.perf_counter() - started:.1f} s; figures in {outdir}")
    return results


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run_all(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
