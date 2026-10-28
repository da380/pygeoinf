"""Part D: a mixture of two references.

Two Adams-Williamson references, one as before and one with a 6% density
jump at the 660 km discontinuity put into the integration, which makes
everything below it heavier. The model space stays three layers: the jump is
in the second prior's expectation only, projected onto the mantle's modes,
which round it over a hundred kilometers or so. Equal prior weights; the
posterior reweights the components by their evidence.

Asked at both prior strengths of parts B and C, so that what decides the
weights is visible: two numbers can say which reference carries the mass and
the moment of inertia the Earth has, and nothing about where in the mantle
that mass sits. A jump at 660 km is one way of making the mantle heavier;
the evidence rewards the weight it adds, not the jump.
"""

from __future__ import annotations

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf2 import GaussianMixture, LinearGaussianMixtureInversion

from common import (
    draw_prem,
    draw_profile,
    finish_profile_axes,
    output_directory,
    parse,
    part_rng,
    save,
)
from problem import (
    FULL,
    LOOSE_STD,
    PROPERTY_SHORT,
    QUICK,
    TIGHT_STD,
    Problem,
    Scale,
    build,
)
from reference import R_660, R_CMB, R_SURFACE

NAMES = ("no jump at 660 km", "a 6% jump at 660 km")


def mixture_at(problem: Problem, std: float, /, *, label: str) -> dict:
    """The mixture posterior with both references at one prior strength."""
    t0 = time.perf_counter()
    references = (problem.reference_model, problem.reference_660_model)
    components = [problem.prior(std, expectation=mean) for mean in references]
    inversion = LinearGaussianMixtureInversion(
        problem.problem, GaussianMixture(components, weights=[0.5, 0.5])
    )
    posterior = inversion(problem.data)
    terms = inversion.log_evidence_terms(problem.data)
    weights = np.asarray(posterior.weights)
    on_properties = inversion.push_forward(problem.properties)(problem.data)
    # How far the data lie from each component's prior predictive, ``N(A m_k,
    # A Q_k A* + R)``, in its own Mahalanobis distance: the part of the
    # evidence that the misfit decides.
    noise = problem.problem.error_measure.covariance.matrix(form="components")
    distances = []
    for component, mean in zip(components, references):
        spread = component.push_forward(problem.forward).covariance
        residual = problem.data - problem.forward(mean)
        matrix = spread.matrix(form="components") + noise
        distances.append(float(np.sqrt(residual @ np.linalg.solve(matrix, residual))))
    print(
        f"{label} (pointwise std {std:g} g/cm^3): posterior in "
        f"{time.perf_counter() - t0:.2f} s"
    )
    print(
        f"  {'component':<22s} {'log evidence':>12s} {'weight':>10s} "
        f"{'data from the predictive (Mahalanobis)':>40s}"
    )
    for name, term, weight, distance in zip(NAMES, terms, weights, distances):
        print(f"  {name:<22s} {term:12.2f} {weight:10.3g} {distance:40.1f}")
    print(f"  {'':<22s} " + " ".join(f"{name:>12s}" for name in PROPERTY_SHORT))
    for name, component in zip(NAMES, on_properties.components):
        deviation = np.sqrt(np.diag(component.covariance.matrix(form="components")))
        cells = " ".join(
            f"{m:6.3f}±{s:5.3f}" for m, s in zip(component.expectation, deviation)
        )
        print(f"  {name:<22s} {cells}")
    print(f"  {'PREM':<22s} " + " ".join(f"{p:12.3f}" for p in problem.prem_properties))
    return {
        "weights": weights,
        "log_evidence_terms": terms,
        "log_evidence": inversion.log_evidence(problem.data),
        "distances": np.array(distances),
        "posterior": posterior,
        "on_properties": on_properties,
        "means": [component.expectation for component in posterior.components],
    }


def mixture_answer(
    problem: Problem, /, *, rng: np.random.Generator, outdir: pathlib.Path
) -> dict:
    """The mixture at the loose and the tight strength, with a figure."""
    started = time.perf_counter()
    space = problem.space
    print(f"\n=== D. A mixture of two references ({problem.scale.name} scale) ===")
    difference = problem.forward(problem.reference_660_model) - problem.forward(
        problem.reference_model
    )
    print(
        f"the jump at {R_660:g} km adds {difference[0] / problem.data[0]:.1%} to the "
        f"mass and {difference[1] / problem.data[1]:.1%} to the moment of inertia"
    )
    loose = mixture_at(problem, LOOSE_STD, label="loose")
    tight = mixture_at(problem, TIGHT_STD, label="tight")

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12.0, 4.2), layout="constrained", width_ratios=[1.7, 1.0]
    )
    draw_prem(left, color="black", lw=1.0, label="PREM")
    for index, (name, color) in enumerate(zip(NAMES, ("C0", "C1"))):
        reference = (problem.reference_model, problem.reference_660_model)[index]
        draw_profile(
            space,
            reference,
            left,
            color=color,
            lw=1.0,
            ls="--",
            label=f"reference, {name}",
        )
        draw_profile(
            space,
            tight["means"][index],
            left,
            color=color,
            lw=1.6,
            label=f"posterior mean, weight {tight['weights'][index]:.3g}",
        )
    left.axvline(R_660, color="0.6", lw=0.6, ls=":")
    left.set_xlim(R_CMB, R_SURFACE)
    left.set_ylim(3.0, 6.5)
    finish_profile_axes(
        left, title=f"The mantle, under the tight mixture (std {TIGHT_STD:g} g/cm³)"
    )
    width = 0.35
    positions = np.arange(2)
    for offset, result, std, shade in (
        (-width / 2, loose, LOOSE_STD, "0.7"),
        (width / 2, tight, TIGHT_STD, "0.3"),
    ):
        right.bar(
            positions + offset,
            result["weights"],
            width,
            color=shade,
            label=f"prior std {std:g} g/cm³",
        )
    right.axhline(0.5, color="0.5", lw=0.8, ls=":", label="prior weight")
    right.set_xticks(positions, ["no jump", "jump at 660 km"])
    right.set_ylim(0.0, 1.05)
    right.set_ylabel("posterior weight")
    right.set_title("What the evidence decides", fontsize=10)
    right.legend(fontsize=7, frameon=False)
    save(fig, outdir / "mixture.png")
    print(f"D took {time.perf_counter() - started:.1f} s")
    return {"loose": loose, "tight": tight, "difference": difference}


def run(
    scale: Scale, /, *, seed: int, outdir: pathlib.Path, problem: Problem | None = None
) -> dict:
    if problem is None:
        problem = build(scale=scale)
    return mixture_answer(problem, rng=part_rng(seed, "d"), outdir=outdir)


def main() -> None:
    args = parse(__doc__)
    run(
        QUICK if args.quick else FULL,
        seed=args.seed,
        outdir=output_directory(args.quick, args.outdir),
    )


if __name__ == "__main__":
    main()
