"""What a chain of the whole problem records at every step.

Nonlinear functions of the state, read off the grid in which the sphere's
vectors already are, so that a step pays no transform for them: the contrasts
themselves, the provinces' share of the sphere and their number, and the
difference between the mean anomaly inside the provinces and outside them, for
each field and as the ratio of shear to compressional that the thermal-or-chemical
question turns on. A thermal Vs and Vp, whatever their pattern, give the thermal
ratio; a contrast inside the provinces that breaks the scaling moves it.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy import ndimage

from pygeoinf import Operator
from pygeoinf.algebra.spaces import EuclideanSpace

from .shell import FIELDS

# The scalars of :func:`record_operator`, in order.
NAMES = (
    "c_s",
    "c_p",
    "c_rho",
    "share",
    "provinces",
    "jump vs",
    "jump vp",
    "jump rho",
    "ratio",
)
# A region holding less of the sphere than this is not counted as a province.
SMALLEST_PROVINCE = 0.01


def area_weights(space: Any, /) -> np.ndarray:
    """``cos(latitude)`` per grid cell, summing to one."""
    weights = np.repeat(
        np.sin(space.colatitudes)[:, None], space.longitudes.size, axis=1
    )
    return weights / weights.sum()


def provinces(inside: np.ndarray, weights: np.ndarray, /) -> int:
    """How many connected regions a mask holds, joined across the date line,
    those below :data:`SMALLEST_PROVINCE` of the sphere left out."""
    labels, count = ndimage.label(inside)
    if count == 0:
        return 0
    parent = np.arange(count + 1)

    def find(k: int) -> int:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for first, last in zip(labels[:, 0], labels[:, -1]):
        if first and last:
            parent[find(first)] = find(last)
    roots = np.array([find(k) for k in range(count + 1)])
    areas = np.bincount(roots[labels].ravel(), weights.ravel(), minlength=count + 1)
    return int(np.sum(areas[1:] >= SMALLEST_PROVINCE))


def summary(space: Any, model: Any, /) -> np.ndarray:
    """The recorded scalars of a model of the whole space.

    Args:
        space: the sphere of one field.
        model: the triple ``(thermal, level set, contrasts)``.

    Returns:
        One entry per name of :data:`NAMES`. The jumps are zero where the
        provinces fill the sphere or are absent, and the ratio then too.
    """
    thermal, phi, contrasts = model
    weights = area_weights(space)
    inside = space.grid_values(phi) > 0.0
    share = float(weights[inside].sum())
    jumps = np.zeros(len(FIELDS))
    if 0.0 < share < 1.0:
        for k, field in enumerate(thermal):
            values = space.grid_values(field)
            jumps[k] = (
                np.sum(weights[inside] * values[inside]) / share
                - np.sum(weights[~inside] * values[~inside]) / (1.0 - share)
                + contrasts[k]
            )
    ratio = jumps[0] / jumps[1] if jumps[1] != 0.0 else 0.0
    return np.array(
        [*contrasts, share, provinces(inside, weights), *jumps, ratio], dtype=float
    )


def record_operator(
    space: Any, domain: Any, /, *, read: Callable[[Any], tuple] = tuple
) -> Operator:
    """The scalars of :func:`summary` as a recording map.

    Args:
        space: the sphere of one field.
        domain: the whole model space.
        read: what makes ``(thermal, level set, contrasts)`` of its models.

    Returns:
        A nonlinear operator into a Euclidean space of one entry per name.
    """
    return Operator.from_callables(
        domain, EuclideanSpace(len(NAMES)), lambda model: summary(space, read(model))
    )
