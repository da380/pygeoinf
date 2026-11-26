"""The double pendulum: two point masses on two massless rods, in a plane.

The state is the two angles from the downward vertical and their rates,
``(a1, a2, w1, w2)``. The parameters are the masses and the lengths,
``(m1, m2, l1, l2)``, all positive; gravity is known. The forcing is an
angular acceleration of each arm.

Two things about this system matter to an inverse problem. The masses enter
the equations through their ratio alone, so a record of the motion fixes
``m2 / m1`` and says nothing about the masses' common size. And the motion
is regular at low energy and chaotic at high: the higher the release, the
faster two nearby parameter values part company.
"""

from __future__ import annotations

import numpy as np

from .system import DynamicalSystem

GRAVITY = 9.81
STATE_NAMES = ("angle 1", "angle 2", "rate 1", "rate 2")
PARAMETER_NAMES = ("m1", "m2", "l1", "l2")


def rule(x: np.ndarray, p: np.ndarray, /) -> np.ndarray:
    """The rate of change of ``(a1, a2, w1, w2)`` at masses and lengths ``(m1, m2, l1, l2)``."""
    a1, a2, w1, w2 = x
    m1, m2, l1, l2 = p
    gap = a1 - a2
    sine, cosine = np.sin(gap), np.cos(gap)
    denominator = 2.0 * m1 + m2 - m2 * np.cos(2.0 * gap)
    rate1 = (
        -GRAVITY * (2.0 * m1 + m2) * np.sin(a1)
        - m2 * GRAVITY * np.sin(a1 - 2.0 * a2)
        - 2.0 * sine * m2 * (w2 * w2 * l2 + w1 * w1 * l1 * cosine)
    ) / (l1 * denominator)
    rate2 = (
        2.0
        * sine
        * (
            w1 * w1 * l1 * (m1 + m2)
            + GRAVITY * (m1 + m2) * np.cos(a1)
            + w2 * w2 * l2 * m2 * cosine
        )
    ) / (l2 * denominator)
    return np.stack([w1 + 0.0 * a1, w2 + 0.0 * a1, rate1, rate2])


def double_pendulum() -> DynamicalSystem:
    """The double pendulum as a dynamical system, forced through its two angular rates."""
    forcing = np.zeros((4, 2))
    forcing[2, 0] = forcing[3, 1] = 1.0
    return DynamicalSystem(
        rule,
        4,
        4,
        forcing_matrix=forcing,
        state_names=STATE_NAMES,
        parameter_names=PARAMETER_NAMES,
    )


def energy(x: np.ndarray, p: np.ndarray, /) -> np.ndarray:
    """The total energy of a state, or of each column of states, zero at rest hanging down."""
    a1, a2, w1, w2 = x
    m1, m2, l1, l2 = p
    kinetic = 0.5 * (m1 + m2) * (l1 * w1) ** 2 + 0.5 * m2 * (l2 * w2) ** 2
    kinetic = kinetic + m2 * l1 * l2 * w1 * w2 * np.cos(a1 - a2)
    potential = (m1 + m2) * GRAVITY * l1 * (1.0 - np.cos(a1))
    potential = potential + m2 * GRAVITY * l2 * (1.0 - np.cos(a2))
    return kinetic + potential


def lower_bob(x: np.ndarray, p: np.ndarray, /) -> np.ndarray:
    """Where the lower mass is: its horizontal and vertical distance from the pivot.

    A function of both angles and both lengths, which is what a camera
    that follows the lower mass reports.
    """
    a1, a2 = x[0], x[1]
    l1, l2 = p[2], p[3]
    return np.stack(
        [l1 * np.sin(a1) + l2 * np.sin(a2), -l1 * np.cos(a1) - l2 * np.cos(a2)]
    )
