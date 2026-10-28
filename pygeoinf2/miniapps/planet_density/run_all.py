"""Every part in order, on one problem: A to E.

``python run_all.py`` for the full scale, ``--quick`` for the small one.
"""

from __future__ import annotations

import pathlib
import time

import boundaries
import mixture
import setup
import tight
import uninformative
from common import output_directory, parse
from problem import FULL, QUICK, Scale


def run_all(scale: Scale, /, *, seed: int, outdir: pathlib.Path) -> dict:
    """Run the parts in order and return what each produced."""
    started = time.perf_counter()
    problem = setup.run(scale, seed=seed, outdir=outdir)
    results = {"a": problem}
    results["b"] = uninformative.run(scale, seed=seed, outdir=outdir, problem=problem)
    results["c"] = tight.run(
        scale, seed=seed, outdir=outdir, problem=problem, loose=results["b"]
    )
    results["d"] = mixture.run(scale, seed=seed, outdir=outdir, problem=problem)
    results["e"] = boundaries.run(scale, seed=seed, outdir=outdir, problem=problem)
    print(f"\ntotal {time.perf_counter() - started:.1f} s; figures in {outdir}")
    return results


def main() -> None:
    args = parse(__doc__)
    run_all(
        QUICK if args.quick else FULL,
        seed=args.seed,
        outdir=output_directory(args.quick, args.outdir),
    )


if __name__ == "__main__":
    main()
