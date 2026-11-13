"""Every part in order, on one problem: A to E.

``python run_all.py`` for the full scale, ``--quick`` for the small one; the
scale's fields are options, applied to the preset and passed to every part.
"""

from __future__ import annotations

import pathlib
import time

import answers
import gibbs
import linear
import proofs
import synthetic
from src.common import parse
from src.shell import FULL, QUICK, Scale


def run_all(scale: Scale, /, *, outdir: pathlib.Path) -> dict:
    """Run the parts in order and return what each produced."""
    started = time.perf_counter()
    results = {"a": proofs.run(scale, outdir=outdir)}
    results["b"] = synthetic.run(scale, outdir=outdir)
    state = results["b"]["setup"]
    results["c"] = linear.run(scale, outdir=outdir, state=state)
    results["d"] = gibbs.run(scale, outdir=outdir, state=state)
    results["e"] = answers.run(scale, outdir=outdir, state=state, gibbs=results["d"])
    print(f"\ntotal {time.perf_counter() - started:.1f} s; figures in {outdir}")
    return results


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run_all(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
