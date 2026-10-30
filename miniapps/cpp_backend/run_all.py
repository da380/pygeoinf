"""Every part in order, on one problem: A to D.

``python run_all.py`` for the full scale, ``--quick`` for the small one; the
scale's fields are options, applied to the preset and passed to every part.
"""

from __future__ import annotations

import pathlib
import time

import bayesian
import ladder
import point
import sets
from src.common import parse
from src.problem import FULL, QUICK, Scale, build


def run_all(scale: Scale, /, *, outdir: pathlib.Path) -> dict:
    """Run the parts in order and return what each produced."""
    started = time.perf_counter()
    problem = build(scale)
    results = {"problem": problem}
    for part, module in (("a", ladder), ("b", point), ("c", bayesian), ("d", sets)):
        results[part] = module.run(scale, outdir=outdir, problem=problem)
    print(f"\ntotal {time.perf_counter() - started:.1f} s; figures in {outdir}")
    return results


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run_all(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
