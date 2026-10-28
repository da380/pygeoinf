"""Every part in order, on one problem: A to D.

``python run_all.py --quick`` for the small scale, which runs in well under
two minutes; the full scale runs each part for about a minute or two, so run
its scripts one at a time, C before D (D reads the mode C saves).
"""

from __future__ import annotations

import pathlib
import time

import laplace
import newton
import occam
import proofs
import synthetic
from common import output_directory, parse
from survey import FULL, QUICK, Scale


def run_all(scale: Scale, /, *, seed: int, outdir: pathlib.Path) -> dict:
    """Run the parts in order and return what each produced."""
    started = time.perf_counter()
    results = {"a": proofs.run(scale, seed=seed, outdir=outdir)}
    results["b"] = synthetic.run(scale, seed=seed, outdir=outdir)
    state = results["b"]["setup"]
    results["c"] = newton.run(scale, seed=seed, outdir=outdir, state=state)
    mode = results["c"]["runs"]["polish"]["model"]
    results["d"] = laplace.run(scale, seed=seed, outdir=outdir, state=state, mode=mode)
    results["occam"] = occam.run(
        scale, seed=seed, outdir=outdir, state=state, mode=mode
    )
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
