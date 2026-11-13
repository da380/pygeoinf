"""The prior: three coupled thermal fields, a level set and three contrasts.

The model space is a nested direct sum, the thermal triple a summand of its
own::

    DirectSum(
        thermal=DirectSum(vs, vp, rho),  # three fields on one sphere
        level_set=sphere,                # the provinces are {phi > 0}
        chemistry=EuclideanSpace(3),     # the contrasts inside them
    )

Part D also groups the same measure differently, the two summands the forward
map is linear in as one, which ``Gibbs`` then updates together.

The thermal fields share their latent white noise mode by mode through
``from_product(correlation=)``: the scaling of mineral physics as a correlation
between the anomalies, each field keeping its own pointwise standard deviation.
They are correlated only inside their own sum, since a correlation ties factors
on one latent space, and a sphere and a Euclidean space have no modes in common.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pygeoinf import GaussianMeasure
from pygeoinf.algebra.spaces import EuclideanSpace
from pygeoinf.symmetric_space.sphere import Sobolev

from .shell import FIELDS, PROVINCES, Scale, separation, to_vector

# The summands of the model space, in order.
SUMMANDS = ("thermal", "level_set", "chemistry")


def sphere(scale: Scale, lmax: int | None = None, /) -> Sobolev:
    """The space of one field: the Sobolev space of the scale on the sphere.

    Args:
        scale: the order and the length scale.
        lmax: the truncation degree; the scale's if omitted.

    Returns:
        The space.
    """
    degree = scale.lmax if lmax is None else lmax
    return Sobolev(degree, scale.order, scale.length_scale)


def correlation_matrix(scale: Scale, /) -> np.ndarray:
    """The thermal fields' correlation matrix, from the scale's three numbers."""
    sp, sr, pr = scale.thermal_correlation
    return np.array([[1.0, sp, sr], [sp, 1.0, pr], [sr, pr, 1.0]])


def thermal_prior(scale: Scale, space: Sobolev, /) -> GaussianMeasure:
    """The three thermal fields, correlated through their latent noise."""
    factors = [
        space.sobolev_measure(scale.prior_order, scale.length_scale, pointwise_std=std)
        for std in scale.thermal_std
    ]
    return GaussianMeasure.from_product(
        factors, labels=FIELDS, correlation=correlation_matrix(scale)
    )


def level_set_prior(scale: Scale, space: Sobolev, /) -> GaussianMeasure:
    """The level set: a field of unit pointwise standard deviation."""
    return space.sobolev_measure(
        scale.prior_order, scale.level_set_length_scale, pointwise_std=1.0
    )


def chemistry_prior(scale: Scale, /) -> GaussianMeasure:
    """The contrasts inside the provinces, independent and centered on none."""
    return GaussianMeasure.from_standard_deviations(
        EuclideanSpace(len(FIELDS)), np.array(scale.chemistry_std)
    )


def full_prior(scale: Scale, space: Sobolev, /) -> GaussianMeasure:
    """The prior on the whole model space, the three summands independent."""
    return GaussianMeasure.from_product(
        [
            thermal_prior(scale, space),
            level_set_prior(scale, space),
            chemistry_prior(scale),
        ],
        labels=SUMMANDS,
    )


def linear_prior(scale: Scale, space: Sobolev, /) -> GaussianMeasure:
    """The thermal fields and the contrasts together, the part of the model the
    forward map is linear in, independent of one another."""
    return GaussianMeasure.from_product(
        [thermal_prior(scale, space), chemistry_prior(scale)],
        labels=("thermal", "chemistry"),
    )


def grouped_prior(scale: Scale, space: Sobolev, /) -> GaussianMeasure:
    """The prior with the linear part one summand and the level set the other:
    the same measure as :func:`full_prior`, on a sum that ``Gibbs`` splits in
    two rather than three."""
    return GaussianMeasure.from_product(
        [linear_prior(scale, space), level_set_prior(scale, space)],
        labels=("linear", "level_set"),
    )


def ungrouped(model: Any, /) -> tuple:
    """``(thermal, level set, contrasts)`` from a model of the grouped sum."""
    return model[0][0], model[1], model[0][1]


def grouped(model: Any, /) -> tuple:
    """A model of the grouped sum from ``(thermal, level set, contrasts)``."""
    return (model[0], model[2]), model[1]


def true_level_set(scale: Scale, space: Sobolev, /) -> Any:
    """The truth's level set: a bump under each province, less a constant."""
    latitudes = 90.0 - np.degrees(space.colatitudes)
    longitudes = np.degrees(space.longitudes)
    grid = np.array(np.meshgrid(latitudes, longitudes, indexing="ij"))
    shape = grid.shape[1:]
    points = to_vector(grid.reshape(2, -1).T)
    values = np.zeros(points.shape[0])
    for center in PROVINCES:
        distance = separation(points, to_vector([center]))
        values += np.exp(-0.5 * (distance / scale.province_width) ** 2)
    return space.from_grid_values((values - scale.province_level).reshape(shape))
