"""The problem: the Earth's radial density from its mass and moment of inertia.

The model space is three layers -- inner core, outer core, mantle -- of
radial profiles under the volume measure, each the Sobolev space ``H^2`` with
a length scale of 200 km: densities in g/cm^3, radii in km. The layers are
independent, so the density may jump at the inner-core boundary and the
core-mantle boundary, and nowhere else.

The data are two numbers, the mass and the moment of inertia,

    M = int rho dV,        I = (2/3) int rho r^2 dV,

in units of 10^24 kg and 10^37 kg m^2 so that both are of order one, with
Gaussian errors of 0.1% of each. Both are linear in the density, so the
forward operator is a two-row matrix on the space's components, assembled by
Gauss-Legendre quadrature against the basis between every pair of adjacent
mesh nodes, where each basis function is a polynomial.

The question asked of every answer is three numbers, also linear: the mean
density of the core (both core layers together), the mean density of the
mantle, and the jump in density at the core-mantle boundary, core less
mantle. Two scales behind one ``build(scale=)``; the full one is small, since
the data are two numbers.
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

import reference
from reference import BOUNDARIES, LAYERS, R_CMB, R_SURFACE, Profile

ORDER = 2.0
LENGTH_SCALE = 200.0
MEASURE_ORDER = 2.0
LEVEL = 0.95

# The two priors' pointwise standard deviations, in g/cm^3: effectively
# non-informative, and tight about the reference.
LOOSE_STD = 3.0
TIGHT_STD = 0.1

# The data, and their units: a volume integral in km^3 of a density in
# g/cm^3 is 10^12 kg, and a further km^2 is 10^6 m^2.
MASS = 5.972e24
MOMENT = 8.034e37
RELATIVE_ERROR = 1e-3
MASS_UNIT = 1e24
MOMENT_UNIT = 1e37
TO_MASS = 1e12 / MASS_UNIT
TO_MOMENT = 1e12 * 1e6 / MOMENT_UNIT

DATA_NAMES = ("mass", "moment of inertia")
PROPERTY_NAMES = ("core mean density", "mantle mean density", "CMB jump")
PROPERTY_SHORT = ("core mean", "mantle mean", "CMB jump")

CORE_VOLUME = 4.0 * np.pi / 3.0 * R_CMB**3
MANTLE_VOLUME = 4.0 * np.pi / 3.0 * (R_SURFACE**3 - R_CMB**3)


@dataclass(frozen=True)
class Scale:
    """One size of the problem: the modes kept in each layer."""

    name: str
    modes: tuple[int, int, int]


QUICK = Scale(name="quick", modes=(12, 12, 12))
FULL = Scale(name="full", modes=(16, 24, 48))


@dataclass
class Problem:
    """Everything ``build`` makes and the parts use."""

    scale: Scale
    space: Layered
    reference: Profile
    reference_660: Profile
    reference_model: tuple
    reference_660_model: tuple
    prem_model: tuple
    forward: LinearOperator
    problem: LinearForwardProblem
    data: np.ndarray
    properties: LinearOperator
    prem_properties: np.ndarray
    jumps: np.ndarray

    def prior(
        self, std: float, /, *, expectation: tuple | None = None
    ) -> GaussianMeasure:
        """The Sobolev prior about a reference, by its pointwise standard
        deviation, the same in every layer and independent between them.

        Assembled as a dense matrix once, which on a space of this size is
        cheap and gives it a precision.

        Args:
            std: the pointwise standard deviation, in g/cm^3.
            expectation: the mean; the Adams-Williamson reference if omitted.
        """
        mean = self.reference_model if expectation is None else expectation
        return self.space.sobolev_measure(
            MEASURE_ORDER, expectation=mean, pointwise_std=std
        ).with_dense_covariance()

    def predicted(self, model: tuple, /) -> np.ndarray:
        """The mass and the moment of inertia of a model, in data units."""
        return self.forward(model)

    def misfit(self, model: tuple, /) -> np.ndarray:
        """Each datum's residual in standard deviations, model less data."""
        return (self.forward(model) - self.data) / (RELATIVE_ERROR * self.data)


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


def build(*, scale: Scale = FULL, quiet: bool = False) -> Problem:
    """The space, the references, the data, the forward problem and the
    property operator."""
    started = time.perf_counter()
    say = (lambda *args: None) if quiet else print
    say(f"\n=== The setup ({scale.name} scale) ===")
    space = Layered(
        list(BOUNDARIES),
        list(scale.modes),
        order=ORDER,
        length_scale=LENGTH_SCALE,
        labels=LAYERS,
    )
    say(
        f"space: three layers of radial profiles, H^{ORDER:g} with a length scale "
        f"of {LENGTH_SCALE:g} km, {'/'.join(map(str, scale.modes))} modes "
        f"({space.dim} in all), boundaries at {BOUNDARIES[1]:g} and "
        f"{BOUNDARIES[2]:g} km"
    )

    profile = reference.adams_williamson()
    profile_660 = reference.adams_williamson(jump_660=reference.JUMP_660)
    reference_model = space.project_function(profile.layer_functions())
    reference_660_model = space.project_function(profile_660.layer_functions())
    prem_model = space.project_function(reference.prem_layer_functions())

    mass_row = TO_MASS * volume_integral(space, np.ones_like)
    moment_row = TO_MOMENT * volume_integral(space, lambda r: 2.0 * r**2 / 3.0)
    data_space = EuclideanSpace(2)
    forward = LinearOperator.from_matrix(
        space, data_space, np.vstack([mass_row, moment_row]), form="components"
    )
    data = np.array([MASS / MASS_UNIT, MOMENT / MOMENT_UNIT])
    noise = GaussianMeasure.from_standard_deviations(data_space, RELATIVE_ERROR * data)
    problem = LinearForwardProblem(forward, error=noise)

    # The rows of the jumps, value above less value below, one per interface.
    jumps = np.vstack(
        [space.jump_operator(k).matrix(form="components")[0] for k in range(2)]
    )
    volume = volume_integral(space, np.ones_like)
    core, mantle = np.zeros(space.dim), np.zeros(space.dim)
    for key in ("inner_core", "outer_core"):
        core[space.columns(key)] = volume[space.columns(key)]
    mantle[space.columns("mantle")] = volume[space.columns("mantle")]
    properties = LinearOperator.from_matrix(
        space,
        EuclideanSpace(3),
        np.vstack([core / CORE_VOLUME, mantle / MANTLE_VOLUME, -jumps[1]]),
        form="components",
    )
    summary = reference.prem_summary()
    prem_properties = np.array([summary["core"], summary["mantle"], summary["jump"]])

    result = Problem(
        scale,
        space,
        profile,
        profile_660,
        reference_model,
        reference_660_model,
        prem_model,
        forward,
        problem,
        data,
        properties,
        prem_properties,
        jumps,
    )
    say(
        f"data: mass {MASS:.4g} kg and moment of inertia {MOMENT:.4g} kg m^2, each "
        f"to {RELATIVE_ERROR:.1%}; in units of {MASS_UNIT:.0e} kg and "
        f"{MOMENT_UNIT:.0e} kg m^2, {data[0]:.3f} and {data[1]:.3f}"
    )
    for name, model, profile_ in (
        ("Adams-Williamson reference", reference_model, profile),
        ("the same with a 6% jump at 660 km", reference_660_model, profile_660),
        ("PREM, projected", prem_model, None),
    ):
        predicted = result.predicted(model)
        sigmas = result.misfit(model)
        extra = (
            ""
            if profile_ is None
            else f" (the profile itself {profile_.mass:.4g} kg, {profile_.moment:.4g} kg m^2)"
        )
        say(
            f"  {name}: M {predicted[0]:.4f} ({sigmas[0]:+.1f} sd), I "
            f"{predicted[1]:.4f} ({sigmas[1]:+.1f} sd){extra}"
        )
    say(f"setup took {time.perf_counter() - started:.1f} s")
    return result
