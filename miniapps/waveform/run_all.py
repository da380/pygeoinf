"""Every serial part in order, on one problem: A to E.

``python run_all.py --quick`` for the small scale, which runs in well under
two minutes; the full scale runs each part for about a minute or two, so run
its scripts one at a time, C before D and E (both read the mode C saves).
Part F runs under ``mpiexec``. The scale's fields are options, applied to
the preset and passed to every part.
"""

from __future__ import annotations

import pathlib
import time

import laplace
import newton
import occam
import proofs
import synthetic
from src.common import parse
from src.survey import FULL, QUICK, Scale


def run_all(scale: Scale, /, *, outdir: pathlib.Path) -> dict:
    """Run the parts in order and return what each produced."""
    started = time.perf_counter()
    results = {"a": proofs.run(scale, outdir=outdir)}
    results["b"] = synthetic.run(scale, outdir=outdir)
    state = results["b"]["setup"]
    results["c"] = newton.run(scale, outdir=outdir, state=state)
    mode = results["c"]["runs"]["polish"]["model"]
    results["d"] = occam.run(scale, outdir=outdir, state=state, mode=mode)
    results["e"] = laplace.run(scale, outdir=outdir, state=state, mode=mode)
    print(f"\ntotal {time.perf_counter() - started:.1f} s; figures in {outdir}")
    return results


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run_all(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
