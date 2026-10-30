"""Every part in order: A to E.

``python run_all.py`` for the full scale, ``--quick`` for the small one; the
scale's fields are options, applied to the preset and passed to every part.
"""

from __future__ import annotations

import pathlib
import time

import box_inverse
import box_ladder
import hilbert_form
import matrix_form
import matrix_inverse
from src.common import FULL, QUICK, Scale, parse


def run_all(scale: Scale, /, *, outdir: pathlib.Path) -> dict:
    """Run the parts in order and return what each produced."""
    started = time.perf_counter()
    results = {}
    for part, module in (
        ("a", hilbert_form),
        ("b", matrix_form),
        ("c", matrix_inverse),
        ("d", box_ladder),
        ("e", box_inverse),
    ):
        results[part] = module.run(scale, outdir=outdir)
    print(f"\ntotal {time.perf_counter() - started:.1f} s; figures in {outdir}")
    return results


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run_all(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
