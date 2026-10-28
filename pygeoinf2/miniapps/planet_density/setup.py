"""Part A: the setup, the reference against PREM, and the velocities it came from.

The Adams-Williamson reference is integrated from PREM's velocities with
PREM's jumps at the two core boundaries put in by hand. Above the
transition zone the two agree to a few percent; across it, whose phase
changes the integration cannot know, they part, and the core hangs from a
mantle that is too light. The data are then far from the
reference, which is what every later part has to deal with.
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np

from common import (
    draw_prem,
    draw_profile,
    finish_profile_axes,
    output_directory,
    parse,
    save,
)
from problem import FULL, PROPERTY_SHORT, QUICK, Problem, Scale, build
from reference import R_660, R_CMB, R_ICB, R_SURFACE, prem, prem_field


def compare(problem: Problem, /) -> dict:
    """The reference's density against PREM's at the radii that matter."""
    profile = problem.reference
    print(
        f"Adams-Williamson reference: self-consistent after {profile.iterations} "
        f"integrations; mass {profile.mass:.4g} kg ({profile.mass / 5.972e24 - 1:+.1%} "
        f"against the datum), moment {profile.moment:.4g} kg m^2 "
        f"({profile.moment / 8.034e37 - 1:+.1%})"
    )
    print(
        f"  {'where':<24s} {'radius':>7s} {'reference':>10s} {'PREM':>7s} {'diff':>7s}"
    )
    rows = (
        ("top of the mantle", R_SURFACE, "below", 6346.6),
        ("above 660 km", R_660 + 1e-6, "above", None),
        ("mantle at the CMB", R_CMB, "above", None),
        ("outer core at the CMB", R_CMB, "below", None),
        ("outer core at the ICB", R_ICB, "above", None),
        ("inner core at the ICB", R_ICB, "below", None),
        ("center", 0.0, "below", None),
    )
    differences = {}
    for name, radius, side, prem_radius in rows:
        ours = problem.reference(radius, side=side)
        theirs = float(
            prem_field("rho", np.array([prem_radius or radius]), side=side)[0]
        )
        differences[name] = ours / theirs - 1.0
        print(
            f"  {name:<24s} {radius:7.1f} {ours:10.3f} {theirs:7.3f} "
            f"{differences[name]:+7.1%}"
        )
    print("  the three properties (g/cm^3), reference and PREM:")
    values = problem.properties(problem.reference_model)
    for name, ours, theirs in zip(PROPERTY_SHORT, values, problem.prem_properties):
        print(f"    {name:<12s} {ours:7.3f} {theirs:7.3f}")
    return {"differences": differences, "reference_properties": values}


def setup_figure(problem: Problem, /, *, outdir: pathlib.Path) -> pathlib.Path:
    """The reference, PREM's density, and PREM's velocities."""
    space = problem.space
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.0), layout="constrained")
    draw_prem(left, color="black", lw=1.0, label="PREM")
    draw_profile(
        space,
        problem.reference_model,
        left,
        color="C0",
        lw=1.6,
        label="Adams-Williamson reference",
    )
    draw_profile(
        space,
        problem.reference_660_model,
        left,
        color="C1",
        lw=1.2,
        ls="--",
        label="the same with a 6% jump at 660 km",
    )
    finish_profile_axes(
        left, title="Densities: the reference is light below the transition zone"
    )

    for layer in prem().layers:
        lower, upper = layer.interval
        r = np.linspace(lower, upper, 60)
        first = layer.interval[0] == 0.0
        right.plot(
            r / 1e3,
            layer.fields["vpv"](r) / 1e3,
            color="C3",
            label="Vp" if first else None,
        )
        right.plot(
            r / 1e3,
            layer.fields["vsv"](r) / 1e3,
            color="C2",
            label="Vs" if first else None,
        )
    for radius in (R_ICB, R_CMB):
        right.axvline(radius, color="0.6", lw=0.6, ls=":")
    right.set_xlim(0.0, R_SURFACE)
    right.set_xlabel("radius (km)")
    right.set_ylabel("velocity (km/s)")
    right.set_title(
        "PREM's velocities, which the reference is integrated from", fontsize=10
    )
    right.legend(fontsize=8, frameon=False)
    path = outdir / "setup.png"
    save(fig, path)
    return path


def run(scale: Scale, /, *, seed: int, outdir: pathlib.Path) -> Problem:
    problem = build(scale=scale)
    compare(problem)
    setup_figure(problem, outdir=outdir)
    return problem


def main() -> None:
    args = parse(__doc__)
    run(
        QUICK if args.quick else FULL,
        seed=args.seed,
        outdir=output_directory(args.quick, args.outdir),
    )


if __name__ == "__main__":
    main()
