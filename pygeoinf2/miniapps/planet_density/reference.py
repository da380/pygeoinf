"""The reference Earth: PREM through planetmodel, and an Adams-Williamson
profile built from its velocities.

PREM supplies three things here: its seismic velocities, from which the
reference density is integrated; the two density jumps at the inner-core and
core-mantle boundaries, which the integration cannot know and are put in by
hand; and its own density, as the yardstick every answer is read against.

The reference is the Adams-Williamson profile, ``d rho / dr = -rho g / Phi``
with ``Phi = Vp^2 - 4 Vs^2 / 3`` the seismic parameter and ``g = G M(r) /
r^2``, ``M(r)`` the mass below ``r``. It is made self-consistent: the
gravity is that of the profile's own mass, and the integration is repeated
until the profile stops changing, so that ``M`` vanishes at the center and
nothing is singular there. The app's mantle runs to the surface: the crust
and the ocean are replaced by mantle material, with the velocities above the
Moho held at their values at the top of PREM's lid and the surface density
the lid's.

Units: radii in kilometers, densities in g/cm^3, velocities in km/s; masses
in kg and moments of inertia in kg m^2 where they are printed as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Callable

import numpy as np
from planetmodel import PREM
from scipy.integrate import cumulative_trapezoid, quad

R_SURFACE = 6371.0
R_CMB = 3480.0
R_ICB = 1221.5
# The 660 km discontinuity, where the second reference of the mixture has a
# density jump, and the jump as a ratio of the density below to that above.
R_660 = 5711.0
JUMP_660 = 1.06

LAYERS = ("inner_core", "outer_core", "mantle")
BOUNDARIES = (0.0, R_ICB, R_CMB, R_SURFACE)

# How finely the integration samples each segment between discontinuities.
_SAMPLES = 4001
_TOLERANCE = 1e-12
_MAX_ITERATIONS = 100


@cache
def prem() -> PREM:
    """PREM, as planetmodel gives it (SI units)."""
    return PREM()


def gravitational_constant() -> float:
    """``G`` in SI units, PREM's own."""
    return float(prem().G)


def _moho() -> float:
    """The top of PREM's lid, in km: where the app's mantle stops being PREM's."""
    lid = next(layer for layer in prem().layers if layer.name == "lid")
    return lid.interval[1] / 1e3


def prem_field(name: str, radii: np.ndarray, /, *, side: str = "below") -> np.ndarray:
    """A PREM field at radii in km: densities in g/cm^3, velocities in km/s.

    Args:
        name: the field, ``"rho"``, ``"vpv"`` or ``"vsv"``.
        radii: where, in km.
        side: at a discontinuity, which of the two values: the one from the
            layer ``"below"`` or ``"above"``.
    """
    radii = np.atleast_1d(np.asarray(radii, dtype=float))
    values = np.full(radii.shape, np.nan)
    meters = radii * 1e3
    layers = prem().layers
    for index, layer in enumerate(layers):
        lower, upper = layer.interval
        inside = (meters > lower) & (meters < upper)
        if side == "below":
            inside |= meters == upper
            if index == 0:
                inside |= meters == lower
        else:
            inside |= meters == lower
            if index == len(layers) - 1:
                inside |= meters == upper
        if np.any(inside):
            values[inside] = layer.fields[name](meters[inside])
    if np.any(np.isnan(values)):
        raise ValueError("A radius lies outside PREM.")
    return values / 1e3


def mantle_field(name: str, radii: np.ndarray, /) -> np.ndarray:
    """A PREM field as the app's mantle sees it: PREM's below the Moho, and
    held at its value at the top of the lid above it."""
    radii = np.atleast_1d(np.asarray(radii, dtype=float))
    return prem_field(name, np.minimum(radii, _moho()), side="below")


def _seismic_parameter(radii: np.ndarray, /) -> np.ndarray:
    """``Vp^2 - 4 Vs^2 / 3`` in m^2/s^2 at radii in km, as the app's layers see
    it: the two values at a discontinuity are each segment's own."""
    radii = np.asarray(radii, dtype=float)
    vp = 1e3 * mantle_field("vpv", radii)
    vs = 1e3 * mantle_field("vsv", radii)
    return vp**2 - 4.0 * vs**2 / 3.0


@dataclass(frozen=True)
class Profile:
    """A density profile sampled densely, segment by segment between its
    discontinuities, from the surface down.

    Attributes:
        tops: the upper radius of each segment, in km.
        radii: each segment's radii, descending, in km.
        densities: each segment's densities, in g/cm^3.
        mass: the total mass, in kg.
        moment: the moment of inertia, in kg m^2.
        iterations: how many integrations the gravity took to settle.
    """

    tops: tuple[float, ...]
    radii: tuple[np.ndarray, ...]
    densities: tuple[np.ndarray, ...]
    mass: float
    moment: float
    iterations: int

    def __call__(self, radius: float, /, *, side: str = "below") -> float:
        """The density at a radius; at a discontinuity, the value on the
        given side, ``"below"`` or ``"above"``."""
        for r, rho in zip(self.radii, self.densities):
            top, bottom = r[0], r[-1]
            if (
                bottom < radius < top
                or (radius == top and side == "below")
                or (radius == bottom and side == "above")
                or (radius == bottom == 0.0)
                or (radius == top == self.tops[0])
            ):
                return float(np.interp(radius, r[::-1], rho[::-1]))
        raise ValueError(f"The radius {radius} km lies outside the profile.")

    def layer_functions(self) -> list[Callable[[float], float]]:
        """One function of radius per layer of the app, inner core first, each
        asked only about its own layer and reading the profile just inside it,
        so that the side of every boundary is the layer's own."""
        return [_inside(self, lower, upper) for lower, upper in _layer_bounds()]


def _layer_bounds() -> list[tuple[float, float]]:
    """The radii bounding each layer of the app, inner core first."""
    return list(zip(BOUNDARIES[:-1], BOUNDARIES[1:]))


def _inside(
    function: Callable[[float], float], lower: float, upper: float, /
) -> Callable[[float], float]:
    """The function read just inside ``[lower, upper]``, never on its ends."""
    margin = 1e-9 * (upper - lower)
    return lambda r: function(float(np.clip(r, lower + margin, upper - margin)))


def adams_williamson(*, jump_660: float | None = None) -> Profile:
    """The self-consistent Adams-Williamson profile from PREM's velocities.

    Integrated from the surface down, with PREM's density jumps added at the
    core-mantle and inner-core boundaries and, if asked, a jump by the given
    ratio at the 660 km discontinuity. The gravity is that of the profile's
    own mass, recomputed after every integration from a uniform start until
    the profile changes by less than one part in ``10^12``.

    Args:
        jump_660: the ratio of the density below the 660 km discontinuity to
            that above it, or ``None`` for no jump there.

    Returns:
        The profile.
    """
    tops = [R_SURFACE, R_CMB, R_ICB]
    if jump_660 is not None:
        tops.insert(1, R_660)
    bottoms = tops[1:] + [0.0]
    radii = [np.linspace(top, bottom, _SAMPLES) for top, bottom in zip(tops, bottoms)]
    # Each segment's seismic parameter from its own side of every end.
    phi = []
    for r in radii:
        inner = r.copy()
        inner[0] -= 1e-9
        if inner[-1] > 0.0:
            inner[-1] += 1e-9
        phi.append(_seismic_parameter(inner))

    surface = float(mantle_field("rho", np.array([R_SURFACE]))[0])
    jump_cmb = float(
        prem_field("rho", np.array([R_CMB]), side="below")[0]
        - prem_field("rho", np.array([R_CMB]), side="above")[0]
    )
    jump_icb = float(
        prem_field("rho", np.array([R_ICB]), side="below")[0]
        - prem_field("rho", np.array([R_ICB]), side="above")[0]
    )
    G = gravitational_constant()

    densities = [np.full(r.shape, 5.5) for r in radii]
    for iteration in range(1, _MAX_ITERATIONS + 1):
        masses = _mass_below(radii, densities)
        updated = []
        start = surface
        for r, p, m, bottom in zip(radii, phi, masses, bottoms):
            meters = 1e3 * r
            with np.errstate(divide="ignore", invalid="ignore"):
                g = np.where(meters > 0.0, G * m / meters**2, 0.0)
            # ln rho(r) = ln rho(top) + the integral of g / Phi from r to the top.
            rho = start * np.exp(cumulative_trapezoid(g / p, -meters, initial=0.0))
            updated.append(rho)
            if bottom == R_CMB:
                start = rho[-1] + jump_cmb
            elif bottom == R_ICB:
                start = rho[-1] + jump_icb
            elif bottom == R_660:
                start = rho[-1] * jump_660
        change = max(
            float(np.max(np.abs(new - old) / new))
            for new, old in zip(updated, densities)
        )
        densities = updated
        if change < _TOLERANCE:
            break

    mass, moment = _moments(radii, densities)
    return Profile(tuple(tops), tuple(radii), tuple(densities), mass, moment, iteration)


def _mass_below(
    radii: list[np.ndarray], densities: list[np.ndarray], /
) -> list[np.ndarray]:
    """``M(r)`` in kg on every segment, accumulated from the center out."""
    masses: list[np.ndarray] = []
    below = 0.0
    for r, rho in zip(reversed(radii), reversed(densities)):
        meters = 1e3 * r[::-1]
        shell = cumulative_trapezoid(
            4.0 * np.pi * meters**2 * 1e3 * rho[::-1], meters, initial=0.0
        )
        masses.append((below + shell)[::-1])
        below += shell[-1]
    return masses[::-1]


def _moments(
    radii: list[np.ndarray], densities: list[np.ndarray], /
) -> tuple[float, float]:
    """The total mass in kg and the moment of inertia in kg m^2."""
    mass = moment = 0.0
    for r, rho in zip(radii, densities):
        meters, kg = 1e3 * r[::-1], 1e3 * rho[::-1]
        mass += float(np.trapezoid(4.0 * np.pi * meters**2 * kg, meters))
        moment += float(np.trapezoid(8.0 * np.pi / 3.0 * meters**4 * kg, meters))
    return mass, moment


@cache
def prem_summary() -> dict[str, float]:
    """PREM's mass and moment of inertia (SI), and the three properties the app
    asks of every answer, in g/cm^3, all by adaptive quadrature over PREM's own
    layers."""
    mass = moment = core = 0.0
    for layer in prem().layers:
        lower, upper = layer.interval
        rho = layer.fields["rho"]
        shell = quad(lambda r: 4.0 * np.pi * r**2 * rho(r), lower, upper, epsrel=1e-12)[
            0
        ]
        mass += shell
        moment += quad(
            lambda r: 8.0 * np.pi / 3.0 * r**4 * rho(r), lower, upper, epsrel=1e-12
        )[0]
        if upper <= R_CMB * 1e3:
            core += shell
    core_volume = 4.0 * np.pi / 3.0 * (R_CMB * 1e3) ** 3
    volume = 4.0 * np.pi / 3.0 * (R_SURFACE * 1e3) ** 3
    jump = float(
        prem_field("rho", np.array([R_CMB]), side="below")[0]
        - prem_field("rho", np.array([R_CMB]), side="above")[0]
    )
    return {
        "mass": mass,
        "moment": moment,
        "core": core / core_volume / 1e3,
        "mantle": (mass - core) / (volume - core_volume) / 1e3,
        "jump": jump,
    }


def prem_layer_functions() -> list[Callable[[float], float]]:
    """PREM's density, one function of radius per layer of the app, each read
    just inside its own layer."""

    def density(r: float) -> float:
        return float(prem_field("rho", np.array([r]))[0])

    return [_inside(density, lower, upper) for lower, upper in _layer_bounds()]
