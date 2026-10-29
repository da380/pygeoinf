"""The problem: the Earth's radial density from its mass and moment of inertia.

The model space is PREM's own layering -- inner core, outer core, eight
layers of mantle and a crust -- of radial profiles under the volume measure,
each the Sobolev space ``H^2`` with a length scale of 200 km, the crust's
excepted: densities in g/cm^3, radii in km. The layers are independent, so the density may jump
wherever PREM's layering has a boundary, and nowhere else.

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

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from pygeoinf2 import (
    EuclideanSpace,
    GaussianMeasure,
    LinearForwardProblem,
    LinearOperator,
)
from pygeoinf2.interval.radial import Layered

from reference import (
    R_CMB,
    R_ICB,
    R_SURFACE,
    layering,
    prem_layer_functions,
    prem_summary,
)

ORDER = 2.0
LENGTH_SCALE = 200.0
MEASURE_ORDER = 2.0
LEVEL = 0.95

# The two priors' pointwise standard deviations, in g/cm^3.
TIGHT_STD = 0.1
WIDE_STD = 3.0

# The data's units: a volume integral in km^3 of a density in g/cm^3 is
# 10^12 kg, and a further km^2 is 10^6 m^2.
RELATIVE_ERROR = 1e-3
MASS_UNIT = 1e24
MOMENT_UNIT = 1e37
TO_MASS = 1e12 / MASS_UNIT
TO_MOMENT = 1e12 * 1e6 / MOMENT_UNIT

PROPERTY_NAMES = ("core mean", "mantle mean", "CMB jump")

CORE_VOLUME = 4.0 * np.pi / 3.0 * R_CMB**3
MANTLE_VOLUME = 4.0 * np.pi / 3.0 * (R_SURFACE**3 - R_CMB**3)


CORE_LAYERS = ("inner_core", "outer_core")


@dataclass(frozen=True)
class Scale:
    """One size of the problem: the basis functions kept in each layer, in
    the order of :func:`reference.layering`, inner core first."""

    name: str
    basis_functions: tuple[int, ...]


# Inner core, outer core, lowermost mantle, lower mantle, the layer above it,
# the transition zone's three, the low-velocity zone, the lid, the crust.
QUICK = Scale(name="quick", basis_functions=(12, 12, 3, 12, 3, 3, 3, 3, 3, 3, 3))
FULL = Scale(name="full", basis_functions=(16, 24, 4, 32, 4, 4, 5, 5, 4, 3, 3))


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
        return sobolev_prior(self.space, self.reference, std)


def sobolev_prior(space: Layered, reference: tuple, std: float, /) -> GaussianMeasure:
    """The Sobolev measure about a reference, by its pointwise standard
    deviation, the same in every layer and independent between them.

    Assembled as a dense matrix once, which on a space of this size is cheap
    and gives it a precision.

    Args:
        space: the layered space.
        reference: the expectation.
        std: the pointwise standard deviation, in g/cm^3.
    """
    return space.sobolev_measure(
        MEASURE_ORDER, expectation=reference, pointwise_std=std
    ).with_dense_covariance()


def volume_integral(
    space: Layered, weight: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """The derivative components of ``rho -> int weight(r) rho dV``.

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


def build(*, scale: Scale = FULL, seed: int = 2027, quiet: bool = False) -> Problem:
    """The space, the reference, the truth, the forward problem, the data and
    the property operator.

    One generator seeded with *seed* draws the truth's departure from the
    reference and then the data's errors.
    """
    started = time.perf_counter()
    say = (lambda *args: None) if quiet else print
    say(f"\n=== A. The setup ({scale.name} scale) ===")
    names, boundaries = layering()
    crust = 0.25 * (boundaries[-1] - boundaries[-2])
    space = Layered(
        list(boundaries),
        list(scale.basis_functions),
        order=ORDER,
        length_scale=[crust if name == "crust" else LENGTH_SCALE for name in names],
        labels=names,
    )
    say(
        f"space: PREM's {len(names)} layers of radial profiles, H^{ORDER:g}, "
        f"{'/'.join(map(str, scale.basis_functions))} basis functions "
        f"({space.dim} in all), boundaries at "
        f"{', '.join(f'{b:g}' for b in boundaries[1:-1])} km"
    )
    reference = space.truncate(
        tuple(
            layer.project_function(
                function, extension="constant" if name == "crust" else "fit"
            )
            for name, layer, function in zip(
                names, space.layers, prem_layer_functions()
            )
        )
    )
    rng = np.random.default_rng(seed)
    truth = sobolev_prior(space, reference, TIGHT_STD).sample(rng=rng)

    mass_row = TO_MASS * volume_integral(space, np.ones_like)
    moment_row = TO_MOMENT * volume_integral(space, lambda r: 2.0 * r**2 / 3.0)
    data_space = EuclideanSpace(2)
    forward = LinearOperator.from_matrix(
        space, data_space, np.vstack([mass_row, moment_row]), form="components"
    )
    clean = forward(truth)
    error = GaussianMeasure.from_standard_deviations(data_space, RELATIVE_ERROR * clean)
    data = data_space.add(clean, error.sample(rng=rng))
    problem = LinearForwardProblem(forward, error=error)

    # The rows of the jumps at the inner-core and the core-mantle boundaries,
    # value above less value below.
    jumps = np.vstack(
        [
            space.jump_operator(
                int(np.argmin(np.abs(np.asarray(space.interfaces) - radius)))
            ).matrix(form="components")[0]
            for radius in (R_ICB, R_CMB)
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
        f"the truth, the reference plus a draw from the prior of pointwise std "
        f"{TIGHT_STD:g} g/cm^3: mass {clean[0] * MASS_UNIT:.5g} kg and moment "
        f"{clean[1] * MOMENT_UNIT:.5g} kg m^2"
    )
    say(
        f"data: those two with errors of {RELATIVE_ERROR:.1%} of each drawn from "
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
