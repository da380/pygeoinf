"""Every part in order, on one problem: A to D.

``python run_all.py`` for the full scale, ``--quick`` for the small one; the
scale's fields are options, applied to the preset and passed to every part.
"""

from __future__ import annotations

import pathlib
import time

import boundaries
import setup
import tight
import wide
from src.common import parse
from src.problem import FULL, QUICK, Scale


def run_all(scale: Scale, /, *, outdir: pathlib.Path) -> dict:
    """Run the parts in order and return what each produced."""
    started = time.perf_counter()
    problem = setup.run(scale, outdir=outdir)
    results = {"a": problem}
    results["b"] = tight.run(scale, outdir=outdir, problem=problem)
    results["c"] = wide.run(scale, outdir=outdir, problem=problem)
    results["d"] = boundaries.run(scale, outdir=outdir, problem=problem)
    print(f"\ntotal {time.perf_counter() - started:.1f} s; figures in {outdir}")
    return results


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run_all(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
