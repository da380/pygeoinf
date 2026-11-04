"""Part A: the forward map and its proofs.

On the small proof scale and on the scale asked for: the library's check ladder
on the linear part of the forward map, ``T``, and on the property map that part
C records; the same functionals rebuilt from the quadratures that the indicator
is integrated by; the indicator itself, on a level set whose province is a cap,
against what can be worked out by hand along meridians and for the zonal
harmonics; the two identities the sharp map obeys in its contrasts and its level
set, and its derivative, which is zero in the level set; and the conditionals of the coupled prior, which ``Gibbs`` reads in closed
form, against ``condition`` by brute force. The figure is the network, the
modes' kernels, and what a province does to one wave's data.
"""

from __future__ import annotations

import dataclasses
import pathlib
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import eval_legendre

from pygeoinf.testing import check_measure, check_operator

from src.common import parse, part_rng, save
from src.forward import Forward, pooled, record_operator
from src.prior import (
    correlation_matrix,
    full_prior,
    sphere,
    thermal_prior,
    true_level_set,
)
from src.shell import FIELDS, FULL, PROOF, QUICK, Scale, network

# The province of the level set the indicator is proved on: a cap about the
# north pole, of this angular radius.
CAP_ANGLE = 0.8


def check_ladder(forward: Forward, thermal: Any, scale: Scale, /, *, rng: Any) -> dict:
    """The library's checks on the linear operators."""
    space = forward.space
    return {
        "check_operator(T)": check_operator(forward.thermal, rng=rng, measure=thermal),
        "check_operator(record)": check_operator(
            record_operator(scale, space), rng=rng, measure=thermal
        ),
        # Four hundred draws know a variance to 7%, so the bound allows for it.
        "check_measure(thermal prior)": check_measure(
            thermal, rng=rng, samples=400, rtol=0.3
        ),
    }


def quadrature_against_operators(
    forward: Forward, thermal: Any, scale: Scale, /, *, rng: Any
) -> dict:
    """What the indicator's quadratures make of a smooth field, against the
    library's path and cap averages on the same field: the same functionals, or
    the contrasts and the thermal fields would be seen differently."""
    space, grid = forward.space, forward.network
    field = thermal.sample(rng=rng)[0]
    arcs = [
        space.geodesic_quadrature(start, end, count=scale.arc_nodes)
        for start, end in grid.s_arcs
    ]
    nodes, weights = pooled(arcs)
    by_quadrature = weights @ space.evaluate(field, space.prepare_points(list(nodes)))
    by_operator = space.path_average_operator(grid.s_arcs, count=scale.arc_nodes)(field)
    caps = [
        space.geodesic_ball_quadrature(center, scale.cap_radius, count=scale.cap_nodes)
        for center in grid.s_caps
    ]
    nodes, weights = pooled(caps)
    cap_quadrature = weights @ space.evaluate(field, space.prepare_points(list(nodes)))
    cap_exact = space.geodesic_ball_average_operator(
        list(grid.s_caps), scale.cap_radius
    )(field)
    return {
        "arcs": float(np.max(np.abs(by_quadrature - by_operator))),
        "caps": float(np.max(np.abs(cap_quadrature - cap_exact))),
        "field_std": float(np.std(space.grid_values(field))),
    }


def cap_level_set(space: Any, angle: float, /) -> Any:
    """``cos(colatitude) - cos(angle)``: positive on a cap about the north pole."""
    cosines = np.cos(space.colatitudes)[:, None] - np.cos(angle)
    return space.from_grid_values(np.repeat(cosines, space.longitudes.size, axis=1))


def indicator_by_hand(forward: Forward, /) -> dict:
    """The indicator of a cap about the pole against what is known exactly: the
    share of a meridional arc inside the cap, and the zonal coefficients."""
    space = forward.space
    phi = cap_level_set(space, CAP_ANGLE)
    edge = 90.0 - np.degrees(CAP_ANGLE)
    arcs = [
        ((lat, lon), (lat - 60.0, lon)) for lat, lon in ((80.0, 10.0), (70.0, -90.0))
    ]
    errors = []
    for start, end in arcs:
        points, weights = space.geodesic_quadrature(start, end, count=2048)
        inside = space.evaluate(phi, space.prepare_points(points)) > 0.0
        share = float(np.asarray(weights) @ inside / np.sum(weights))
        exact = np.clip((start[0] - edge) / 60.0, 0.0, 1.0)
        errors.append(abs(share - exact))
    # The coefficient of l, m == 0 is the indicator's integral against the
    # basis function over the function's squared norm, whatever the basis's
    # normalization, which is read from its value at the pole.
    coefficients = {}
    found = forward.indicator_coefficients(phi)
    pole = space.basis_at((90.0, 0.0))
    upper = np.cos(CAP_ANGLE)
    for degree in forward.degrees:
        normal = pole[space.component_of(degree, 0)]
        integral = (
            eval_legendre(degree - 1, upper) - eval_legendre(degree + 1, upper)
        ) / (2 * degree + 1)
        exact = (2 * degree + 1) * integral / (2.0 * normal)
        position = forward.labels.index((degree, 0))
        coefficients[degree] = (float(found[position]), float(exact))
    return {"arcs": max(errors), "coefficients": coefficients}


def identities(forward: Forward, thermal: Any, scale: Scale, /, *, rng: Any) -> dict:
    """The sharp map's linearity in its contrasts, and its near-symmetry.

    ``c H(-phi) == c - c H(phi)``, so flipping the level set and the contrasts
    and absorbing the constant in the thermal fields changes no datum.
    """
    space = forward.space
    t, phi = thermal.sample(rng=rng), true_level_set(scale, space)
    c = rng.normal(size=len(FIELDS))
    other = rng.normal(size=len(FIELDS))
    base = forward((t, phi, c))
    linearity = np.max(
        np.abs(
            forward((t, phi, c + 2.0 * other))
            - 2.0 * forward((t, phi, other))
            - base
            + forward.thermal(t) * 2.0
        )
    )
    shifted = tuple(
        space.add(field, space.from_grid_values(np.full(space.grid_shape, value)))
        for field, value in zip(t, c)
    )
    flip = np.max(np.abs(forward((shifted, space.negative(phi), -c)) - base))
    # The derivative: along the thermal fields and the contrasts it is the
    # difference of the map, which is linear there, and it ignores the level set.
    domain = full_prior(scale, space).domain
    derivative = forward.derivative(domain, (t, phi, c))
    dt, dc = thermal.sample(rng=rng), rng.normal(size=len(FIELDS))
    moved = tuple(space.add(a, b) for a, b in zip(t, dt))
    difference = forward((moved, phi, c + dc)) - base
    along = np.max(np.abs(derivative((dt, space.zero(), dc)) - difference))
    across = np.max(np.abs(derivative((dt, space.random(rng=rng), dc)) - difference))
    adjoint = derivative.adjoint(difference)
    return {
        "linearity": float(linearity),
        "flip": float(flip),
        "along": float(along),
        "across": float(across),
        "level_set_gradient": float(space.norm(adjoint[1])),
        "derivative": check_operator(
            derivative, rng=rng, measure=full_prior(scale, space)
        ),
    }


def conditionals_by_brute_force(scale: Scale, /, *, rng: Any) -> dict:
    """The closed-form conditional of each thermal field given the others, against
    the Schur complement of the prior's dense covariance, at low degree.

    In components the covariance operator's blocks are the components'
    covariance times the one field's metric, which the Schur complement's
    inverse cancels, so the formula holds of the operator's matrix as it stands.
    """
    small = dataclasses.replace(scale, lmax=8)
    space = sphere(small)
    prior = thermal_prior(small, space)
    domain = prior.domain
    dim = space.dim
    covariance = prior.covariance.matrix(form="components")
    given = prior.sample(rng=rng)
    values = domain.to_components(given)
    results = {}
    for index, name in enumerate(FIELDS):
        own = np.arange(index * dim, (index + 1) * dim)
        rest = np.setdiff1d(np.arange(covariance.shape[0]), own)
        gain = np.linalg.solve(
            covariance[np.ix_(rest, rest)], covariance[np.ix_(rest, own)]
        ).T
        mean = gain @ values[rest]
        schur = covariance[np.ix_(own, own)] - gain @ covariance[np.ix_(rest, own)]
        conditional = prior.conditional(name, given)
        found_mean = space.to_components(conditional.expectation)
        found = conditional.covariance.matrix(form="components")
        results[name] = {
            "mean": float(np.linalg.norm(found_mean - mean) / np.linalg.norm(mean)),
            "covariance": float(np.linalg.norm(found - schur) / np.linalg.norm(schur)),
            "reduction": float(
                np.trace(schur) / np.trace(covariance[np.ix_(own, own)])
            ),
        }
    return results


def prove(scale: Scale, /, *, rng: Any) -> dict:
    """Every check at one scale, printed."""
    grid = network(scale)
    space = sphere(scale)
    forward = Forward(space, grid, scale)
    thermal = thermal_prior(scale, space)
    print(
        f"{scale.name} scale: degree {space.lmax}, {forward.data_space.dim} data, "
        f"thermal correlation {correlation_matrix(scale)[0, 1]:.2f} between vs and vp"
    )
    reports = check_ladder(forward, thermal, scale, rng=rng)
    for name, report in reports.items():
        print(f"  {name}: passes, worst defect {report.worst:.1e}")
    quadrature = quadrature_against_operators(forward, thermal, scale, rng=rng)
    print(
        f"  a field of std {quadrature['field_std']:.2f} by the indicator's "
        f"quadratures, against the operators': arcs {quadrature['arcs']:.1e}, caps "
        f"{quadrature['caps']:.1e}"
    )
    hand = indicator_by_hand(forward)
    print(
        f"  a cap's indicator by hand: the share of a meridional arc in {hand['arcs']:.1e}"
    )
    for degree, (found, exact) in hand["coefficients"].items():
        print(
            f"    the coefficient of degree {degree}, order 0: {found:+.4f} on the "
            f"fine grid, {exact:+.4f} exactly, off by {abs(found - exact):.1e}"
        )
    identity = identities(forward, thermal, scale, rng=rng)
    print(
        f"  linear in the contrasts to {identity['linearity']:.1e}; the data of "
        f"(t + c, -phi, -c) and (t, phi, c) differ by {identity['flip']:.1e}"
    )
    print(
        f"  the derivative at the truth's level set: passes check_operator, worst "
        f"defect {identity['derivative'].worst:.1e}; against the map's difference along "
        f"(t, c) {identity['along']:.1e}, with a level-set direction added "
        f"{identity['across']:.1e}; its adjoint's level-set part has norm "
        f"{identity['level_set_gradient']:.1e}"
    )
    return {
        "forward": forward,
        "reports": reports,
        "quadrature": quadrature,
        "hand": hand,
        "identities": identity,
    }


def draw(forward: Forward, scale: Scale, /, *, outdir: pathlib.Path) -> None:
    """The modes' kernels and what a province does to the S arcs' data."""
    space = forward.space
    grid = forward.network
    response = forward.response(true_level_set(scale, space))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    ax = axes[0]
    rows = np.arange(grid.kernels.shape[0])
    width = 0.27
    for k, (name, color) in enumerate(zip(FIELDS, ("#2a78d6", "#eb6834", "#1baf7a"))):
        ax.bar(
            rows + (k - 1) * width, grid.kernels[:, k], width, color=color, label=name
        )
    ax.set_xticks(rows, [f"mode {row + 1}" for row in rows])
    ax.set_ylabel("weight through the layer")
    ax.set_title("the modes' weights on the three fields", fontsize=10)
    ax.legend(frameon=False)
    ax = axes[1]
    for (kind, label), color in zip(
        (("s_arcs", "Sdiff arcs"), ("s_caps", "ScS caps")), ("#2a78d6", "#eb6834")
    ):
        ax.hist(
            response[forward.slices[kind], 0],
            bins=np.linspace(0.0, 1.0, 21),
            histtype="step",
            density=True,
            lw=1.6,
            color=color,
            label=label,
        )
    ax.set_xlabel("the share of a datum's path inside the true provinces")
    ax.set_yticks([])
    ax.legend(frameon=False)
    ax.set_title("how much of each S datum the provinces fill", fontsize=10)
    save(
        fig,
        outdir / "network.png",
        title="LLSVP, part A: the modes' weights, and what the provinces do to the "
        "S data",
    )


def run(scale: Scale, /, *, outdir: pathlib.Path) -> dict:
    started = time.perf_counter()
    print(f"\n=== A. The forward map and its proofs ({scale.name} scale) ===")
    rng = part_rng(scale.seed, "a")
    results = {"proof": prove(PROOF, rng=rng)}
    results["scale"] = prove(scale, rng=rng)
    print("  the conditionals of the coupled prior, at degree 8, against brute force:")
    results["conditionals"] = conditionals_by_brute_force(scale, rng=rng)
    for name, gaps in results["conditionals"].items():
        print(
            f"    {name} given the others: mean off by {gaps['mean']:.1e}, covariance "
            f"by {gaps['covariance']:.1e}, relatively; its variance is "
            f"{gaps['reduction']:.0%} of its marginal's"
        )
    draw(results["scale"]["forward"], scale, outdir=outdir)
    print(f"A took {time.perf_counter() - started:.1f} s")
    return results


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
