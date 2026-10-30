"""PREM through planetmodel: its layering, its density layer by layer, and
its own mass, moment of inertia and the three properties the app asks of
every answer.

The app's layers are PREM's own: every boundary between two of PREM's
layers is a boundary of the space, so the density jumps where PREM's does.
PREM's three layers thinner than 20 km, the lower crust, the upper crust
and the ocean, are too thin to carry a few basis functions each and are
merged into one crust, which therefore holds PREM's two jumps at 6356 and
6368 km inside it.

Units: radii in kilometers and densities in g/cm^3 here; PREM's mass and
moment of inertia are returned in kg and kg m^2.
"""

from __future__ import annotations

from functools import cache
from typing import Callable

import numpy as np
from planetmodel import PREM
from scipy.integrate import quad

R_SURFACE = 6371.0
R_CMB = 3480.0
R_ICB = 1221.5

#: The thinnest a layer of the space may be, in km.
THINNEST = 20.0
#: PREM's layers merged into one, the crust, being thinner than that.
MERGED = ("lower_crust", "upper_crust", "ocean")


@cache
def prem() -> PREM:
    """PREM, as planetmodel gives it (SI units)."""
    return PREM()


@cache
def layering() -> tuple[tuple[str, ...], tuple[float, ...]]:
    """The space's layers: their names, and their boundaries in km from the
    center out.

    PREM's layers in order, with those in :data:`MERGED` made one layer,
    called the crust.

    Raises:
        ValueError: if a layer is thinner than :data:`THINNEST`.
    """
    names: list[str] = []
    boundaries = [0.0]
    for layer in prem().layers:
        top = layer.interval[1] / 1e3
        if layer.name in MERGED and names and names[-1] == "crust":
            boundaries[-1] = top
            continue
        names.append("crust" if layer.name in MERGED else layer.name)
        boundaries.append(top)
    thickness = np.diff(boundaries)
    if np.any(thickness < THINNEST):
        thin = [n for n, t in zip(names, thickness) if t < THINNEST]
        raise ValueError(f"Layers thinner than {THINNEST:g} km: {thin}.")
    return tuple(names), tuple(boundaries)


def prem_density(radius: float, /, *, side: str = "below") -> float:
    """PREM's density in g/cm^3 at a radius in km.

    Args:
        radius: where.
        side: at a discontinuity, the value of the layer ``"below"`` or
            ``"above"``.
    """
    meters = 1e3 * radius
    layers = prem().layers
    for index, layer in enumerate(layers):
        lower, upper = layer.interval
        inside = lower < meters < upper
        if side == "below":
            inside |= meters == upper or (index == 0 and meters == lower)
        else:
            inside |= meters == lower or (index == len(layers) - 1 and meters == upper)
        if inside:
            return float(layer.fields["rho"](meters)) / 1e3
    raise ValueError(f"The radius {radius} km lies outside PREM.")


def _inside(
    function: Callable[[float], float], lower: float, upper: float, /
) -> Callable[[float], float]:
    """The function read just inside ``[lower, upper]``, never on its ends, so
    that at a boundary each layer sees its own side."""
    margin = 1e-9 * (upper - lower)
    return lambda r: function(float(np.clip(r, lower + margin, upper - margin)))


def prem_layer_functions() -> list[Callable[[float], float]]:
    """PREM's density as one function of radius per layer of the app, inner
    core first, each read just inside its own layer."""
    _, boundaries = layering()
    return [
        _inside(prem_density, lower, upper)
        for lower, upper in zip(boundaries[:-1], boundaries[1:])
    ]


@cache
def prem_summary() -> dict[str, float]:
    """PREM's own mass (kg) and moment of inertia (kg m^2), and its core mean
    density, mantle mean density and jump at the core-mantle boundary, core
    less mantle (g/cm^3), all by adaptive quadrature over PREM's layers."""
    mass = moment = core = 0.0
    for layer in prem().layers:
        lower, upper = layer.interval
        rho = layer.fields["rho"]
        shell = quad(lambda r: 4.0 * np.pi * r**2 * rho(r), lower, upper, epsrel=1e-12)
        mass += shell[0]
        moment += quad(
            lambda r: 8.0 * np.pi / 3.0 * r**4 * rho(r), lower, upper, epsrel=1e-12
        )[0]
        if upper <= R_CMB * 1e3:
            core += shell[0]
    core_volume = 4.0 * np.pi / 3.0 * (R_CMB * 1e3) ** 3
    volume = 4.0 * np.pi / 3.0 * (R_SURFACE * 1e3) ** 3
    return {
        "mass": mass,
        "moment": moment,
        "core": core / core_volume / 1e3,
        "mantle": (mass - core) / (volume - core_volume) / 1e3,
        "jump": prem_density(R_CMB, side="below") - prem_density(R_CMB, side="above"),
    }
