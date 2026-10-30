"""The problem: the Earth's radial density from its mass and moment of inertia.

The model space is PREM's own layering -- inner core, outer core, eight
layers of mantle and a crust -- of radial profiles under the volume measure,
each the Sobolev space ``H^2`` with a length scale of 200 km, the crust's
excepted. The layers are independent, so the density may jump wherever
PREM's layering has a boundary, and nowhere else.

The units: radius in Earth radii, density in g/cm^3, the data in 10^24 kg
and 10^37 kg m^2. The planet has radius one, so the volume integrals in the
space's inner product are of order one and so are its norms, as the
densities and the data are. Lengths given in km, the scale's among them, are
converted where they meet the space.

The reference is PREM in the space. Within every layer but the crust PREM's
density is smooth, and there it is fitted over the layer's domain by the
layer's basis functions, the leading eigenfunctions of its Sobolev
operator, which hold it to the truncation's accuracy. The crust holds two
of PREM's jumps, which a fit would chase into the padding, so there PREM is
sampled at the nodes, the padding given the value at the nearer end, and
settled into the span; the crust's length scale is a quarter of its
thickness, so that the padding the sample is continued into is no wider
than the crust, and the span is not mostly the padding's. The truth is the reference
plus a seeded draw from the tight prior about it, so that PREM is wrong by a
known amount and the tight prior is honest about how much. The data are two
numbers, the mass and the moment of inertia of the truth,

    M = int rho dV,        I = (2/3) int rho r^2 dV,

in units of 10^24 kg and 10^37 kg m^2 so that both are of order one, with
Gaussian errors of 0.1% of each drawn from the error measure. Both are
linear in the density, so the forward operator is a two-row matrix on the
space's components, assembled by Gauss-Legendre quadrature against the basis
between every pair of adjacent mesh nodes, where each basis function is a
polynomial.

The question asked of every answer is three numbers, also linear: the mean
density of the core (both core layers together), the mean density of the
mantle (everything above the core-mantle boundary, crust included), and the
jump in density at the core-mantle boundary, core less mantle. Two scales behind one ``build(scale=)``.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from pygeoinf import (
    EuclideanSpace,
    GaussianMeasure,
    LinearForwardProblem,
    LinearOperator,
)
from pygeoinf.interval.radial import Layered

from .reference import (
    R_CMB,
    R_ICB,
    R_SURFACE,
    layering,
    prem_layer_functions,
    prem_summary,
)

# The data's units. A volume integral of a density in g/cm^3 over radii in
# Earth radii is in units of R_SURFACE^3 km^3 g/cm^3, which is R_SURFACE^3
# 10^12 kg with R_SURFACE in km; a further r^2 is R_SURFACE^2 10^6 m^2.
MASS_UNIT = 1e24
MOMENT_UNIT = 1e37
TO_MASS = R_SURFACE**3 * 1e12 / MASS_UNIT
TO_MOMENT = R_SURFACE**5 * 1e18 / MOMENT_UNIT

PROPERTY_NAMES = ("core mean", "mantle mean", "CMB jump")

# The core's and the mantle's volumes, in Earth radii cubed.
CORE_VOLUME = 4.0 * np.pi / 3.0 * (R_CMB / R_SURFACE) ** 3
MANTLE_VOLUME = 4.0 * np.pi / 3.0 * (1.0 - (R_CMB / R_SURFACE) ** 3)


CORE_LAYERS = ("inner_core", "outer_core")


@dataclass(frozen=True)
class Scale:
    """One size of the problem, and its physics: every field but the name is
    a command-line option of every script.

    Attributes:
        name: the preset's name, for the printed headings.
        seed: the seed of the truth's departure from PREM, the data's errors
            and part D's draws.
        basis_functions: the basis functions kept in each of the eleven
            layers, inner core first: inner core, outer core, lowermost
            mantle, lower mantle, the layer above it, the transition zone's
            three, the low-velocity zone, the lid, the crust.
        order: the Sobolev order of every layer of the model space.
        length_scale: the length scale of every layer but the crust, in km,
            converted to the space's Earth radii.
        measure_order: the order of the priors' Sobolev measure on every
            layer, above one half: a radial profile's spectrum grows as on a
            line, and at or below that the covariance is not trace class.
        tight_std: the tight prior's pointwise standard deviation, the one
            the truth is drawn from, in g/cm^3.
        wide_std: the wide prior's pointwise standard deviation, in g/cm^3.
        relative_error: the standard deviation of each datum's error, as a
            fraction of the datum.
        level: the credibility of the discrepancy level and of the sets the
            priors and the noise are hardened to.
        radius_scale: part D's unit of boundary displacement, which is the
            displacements' prior standard deviation, in km, converted to the
            space's Earth radii.
        true_shifts: part D's true outward displacements of the inner-core
            and core-mantle boundaries, in km, converted likewise.
        check_points: the draws from the prior at which part D checks its
            derivatives.
    """

    name: str
    seed: int
    basis_functions: tuple[int, ...]
    order: float
    length_scale: float
    measure_order: float
    tight_std: float
    wide_std: float
    relative_error: float
    level: float
    radius_scale: float
    true_shifts: tuple[float, ...]
    check_points: int


QUICK = Scale(
    name="quick",
    seed=2027,
    basis_functions=(12, 12, 3, 12, 3, 3, 3, 3, 3, 3, 3),
    order=2.0,
    length_scale=200.0,
    measure_order=2.0,
    tight_std=0.1,
    wide_std=3.0,
    relative_error=1e-3,
    level=0.95,
    radius_scale=50.0,
    true_shifts=(-20.0, 30.0),
    check_points=3,
)
FULL = dataclasses.replace(
    QUICK, name="full", basis_functions=(16, 24, 4, 32, 4, 4, 5, 5, 4, 3, 3)
)


@dataclass
class Problem:
    """Everything ``build`` makes and the parts use."""

    scale: Scale
    space: Layered
    reference: tuple
    truth: tuple
    forward: LinearOperator
    problem: LinearForwardProblem
    data: np.ndarray
    properties: LinearOperator
    true_properties: np.ndarray
    jumps: np.ndarray

    def prior(self, std: float, /) -> GaussianMeasure:
        """The Sobolev measure about the reference; see :func:`sobolev_prior`.

        Args:
            std: the pointwise standard deviation, in g/cm^3.
        """
        return sobolev_prior(
            self.space, self.reference, std, order=self.scale.measure_order
        )


def sobolev_prior(
    space: Layered, reference: tuple, std: float, /, *, order: float
) -> GaussianMeasure:
    """The Sobolev measure about a reference, by its pointwise standard
    deviation, the same in every layer and independent between them.

    Assembled as a dense matrix once, which on a space of this size is cheap
    and gives it a precision.

    Args:
        space: the layered space.
        reference: the expectation.
        std: the pointwise standard deviation, in g/cm^3.
        order: the measure's Sobolev order.
    """
    return space.sobolev_measure(
        order, expectation=reference, pointwise_std=std
    ).with_dense_covariance()


def volume_integral(
    space: Layered, weight: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """The derivative components of ``rho -> int weight(r) rho dV``, with
    ``r`` in Earth radii and ``dV`` in Earth radii cubed.

    Five-point Gauss-Legendre between every pair of adjacent nodes of each
    layer, where every basis function is a polynomial of the mesh's degree,
    so that the rule is exact for any polynomial weight of low degree.
    """
    nodes, weights = np.polynomial.legendre.leggauss(5)
    radii, quadrature = [], []
    for layer in space.layers:
        ends = layer.interior_nodes
        middle, half = 0.5 * (ends[:-1] + ends[1:]), 0.5 * (ends[1:] - ends[:-1])
        radii.append((middle[:, None] + half[:, None] * nodes).ravel())
        quadrature.append((half[:, None] * weights).ravel())
    r, w = np.concatenate(radii), np.concatenate(quadrature)
    return (4.0 * np.pi * r**2 * weight(r) * w) @ space.basis_matrix(list(r))


def build(*, scale: Scale = FULL, quiet: bool = False) -> Problem:
    """The space, the reference, the truth, the forward problem, the data and
    the property operator.

    One generator seeded with the scale's seed draws the truth's departure from the
    reference and then the data's errors.
    """
    started = time.perf_counter()
    say = (lambda *args: None) if quiet else print
    say(f"\n=== A. The setup ({scale.name} scale) ===")
    names, boundaries = layering()
    # The space's radius is in Earth radii: PREM's boundaries and the length
    # scale, both in km, are converted.
    radii = [boundary / R_SURFACE for boundary in boundaries]
    crust = 0.25 * (radii[-1] - radii[-2])
    scales = [
        crust if name == "crust" else scale.length_scale / R_SURFACE for name in names
    ]
    space = Layered(
        radii,
        list(scale.basis_functions),
        order=scale.order,
        length_scale=scales,
        labels=names,
    )
    say(
        f"space: PREM's {len(names)} layers of radial profiles, radius in Earth radii, "
        f"H^{scale.order:g}, "
        f"{'/'.join(map(str, scale.basis_functions))} basis functions "
        f"({space.dim} in all), boundaries at "
        f"{', '.join(f'{b:g}' for b in boundaries[1:-1])} km"
    )
    # PREM's density is a function of the radius in km, read at R_SURFACE r.
    reference = space.truncate(
        tuple(
            layer.project_function(
                lambda r, prem=prem: prem(R_SURFACE * r),
                extension="constant" if name == "crust" else "fit",
            )
            for name, layer, prem in zip(names, space.layers, prem_layer_functions())
        )
    )
    rng = np.random.default_rng(scale.seed)
    truth = sobolev_prior(
        space, reference, scale.tight_std, order=scale.measure_order
    ).sample(rng=rng)

    mass_row = TO_MASS * volume_integral(space, np.ones_like)
    moment_row = TO_MOMENT * volume_integral(space, lambda r: 2.0 * r**2 / 3.0)
    data_space = EuclideanSpace(2)
    forward = LinearOperator.from_matrix(
        space, data_space, np.vstack([mass_row, moment_row]), form="components"
    )
    clean = forward(truth)
    error = GaussianMeasure.from_standard_deviations(
        data_space, scale.relative_error * clean
    )
    data = data_space.add(clean, error.sample(rng=rng))
    problem = LinearForwardProblem(forward, error=error)

    # The rows of the jumps at the inner-core and the core-mantle boundaries,
    # value above less value below.
    jumps = np.vstack(
        [
            space.jump_operator(
                int(np.argmin(np.abs(np.asarray(space.interfaces) - radius)))
            ).matrix(form="components")[0]
            for radius in (R_ICB / R_SURFACE, R_CMB / R_SURFACE)
        ]
    )
    volume = volume_integral(space, np.ones_like)
    core, mantle = np.zeros(space.dim), np.zeros(space.dim)
    for name in names:
        part = core if name in CORE_LAYERS else mantle
        part[space.columns(name)] = volume[space.columns(name)]
    properties = LinearOperator.from_matrix(
        space,
        EuclideanSpace(3),
        np.vstack([core / CORE_VOLUME, mantle / MANTLE_VOLUME, -jumps[1]]),
        form="components",
    )
    result = Problem(
        scale,
        space,
        reference,
        truth,
        forward,
        problem,
        data,
        properties,
        properties(truth),
        jumps,
    )

    own = prem_summary()
    predicted = forward(reference)
    say(
        f"PREM in the space, the reference: mass "
        f"{predicted[0] * MASS_UNIT:.5g} kg and moment of inertia "
        f"{predicted[1] * MOMENT_UNIT:.5g} kg m^2, against PREM's own "
        f"{own['mass']:.5g} and {own['moment']:.5g}"
    )
    say(
        f"norms on the space: the reference's {space.norm(reference):.3g}, the "
        f"truth's departure from it {space.norm(space.subtract(truth, reference)):.3g}"
    )
    say(
        f"the truth, the reference plus a draw from the prior of pointwise std "
        f"{scale.tight_std:g} g/cm^3: mass {clean[0] * MASS_UNIT:.5g} kg and moment "
        f"{clean[1] * MOMENT_UNIT:.5g} kg m^2"
    )
    say(
        f"data: those two with errors of {scale.relative_error:.1%} of each drawn from "
        f"the error measure, in units of {MASS_UNIT:.0e} kg and {MOMENT_UNIT:.0e} "
        f"kg m^2: {data[0]:.4f} and {data[1]:.4f}; chi-squared "
        f"{problem.chi_squared(truth, data):.2f} at the truth and "
        f"{problem.chi_squared(reference, data):.1f} at the reference"
    )
    say("the three properties (g/cm^3), of the truth, the reference and PREM itself:")
    for name, true, ours, theirs in zip(
        PROPERTY_NAMES,
        result.true_properties,
        properties(reference),
        (own["core"], own["mantle"], own["jump"]),
    ):
        say(f"  {name:<12s} {true:7.3f} {ours:7.3f} {theirs:7.3f}")
    say(f"setup took {time.perf_counter() - started:.1f} s")
    return result
