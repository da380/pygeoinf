"""Functions of time as vectors of the library.

A component of the trajectory, or of the forcing, is a function of time on
``[0, duration]`` in a Sobolev space of the library,
:class:`pygeoinf.interval.Sobolev`: a mesh of spectral elements, a function
held by its values at the mesh's nodes, and the inner product of ``H^s``
for a time scale. Every eigenfunction of the mesh is kept, so the space is
all the mesh can hold and a function is known by its nodal values alone:
the integrator's table of the state at the nodes of the time interval *is*
the trajectory, with nothing fitted and nothing truncated, and a value at
any other time is read off the element that holds it. The order is above
one half, so that value is a bounded functional and an observation of it a
well-posed datum.

A trajectory has one such function for each component of the state, and is
a vector of the tensor product of a Euclidean space, one axis for the
components, with the space of functions of time. The forcing is the same
with its own number of components.

The integrator works on plain tables of numbers, values at the nodes that
lie in the time interval. The two operators here pass between a table and
a function of the space, and are the identity on those values. The mesh
runs a little past each end of the interval, as the library's interval
spaces do; a table is continued there by its end values.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from pygeoinf import EuclideanSpace, LinearOperator, interval
from pygeoinf.algebra.direct_sum import ColumnLinearOperator
from pygeoinf.algebra.operators import LinearFunctional
from pygeoinf.algebra.tensor_product import TensorProduct

from .trajectory import TimeGrid


def time_space(
    duration: float, element_length: float, order: float, time_scale: float, /
) -> interval.Sobolev:
    """The Sobolev space of functions of time on ``[0, duration]``, every mode of its mesh kept.

    Args:
        duration: the length of the time interval.
        element_length: the longest a spectral element may be. A function
            is a polynomial of degree four on each, so this is what sets
            how sharp a trajectory the space holds.
        order: the Sobolev order, above one half.
        time_scale: the time scale of the inner product.
    """

    def build(modes: int) -> interval.Sobolev:
        return interval.Sobolev(
            modes,
            order,
            time_scale,
            lower=0.0,
            upper=float(duration),
            element_length=float(element_length),
        )

    return build(build(1).nodes.size)


def grid_of(space: interval.Sobolev, /) -> TimeGrid:
    """The nodes of a space's mesh that lie in the time interval, with their quadrature weights."""
    return TimeGrid(np.array(space.interior_nodes), np.array(space.basis.weights()))


def from_table(space: interval.Sobolev, /) -> LinearOperator:
    """The function of the space with given values at the nodes of the time interval."""
    inside = np.flatnonzero(space.interior_mask)
    extend = np.zeros((space.nodes.size, inside.size))
    extend[inside, np.arange(inside.size)] = 1.0
    extend[: inside[0], 0] = 1.0
    extend[inside[-1] + 1 :, -1] = 1.0
    return LinearOperator.from_matrix(
        EuclideanSpace(inside.size),
        space,
        space.to_components(extend),
        form="components",
    )


def to_table(space: interval.Sobolev, /) -> LinearOperator:
    """A function of the space, read at the nodes of the time interval."""
    values = space.from_components(np.eye(space.dim))[space.interior_mask]
    return LinearOperator.from_matrix(
        space, EuclideanSpace(values.shape[0]), values, form="components"
    )


def observation_operator(
    trajectory: TensorProduct,
    observed: Sequence[tuple[int, Sequence[float]]],
    /,
) -> LinearOperator:
    """The values of chosen components of a trajectory at chosen times.

    Args:
        trajectory: the space of trajectories, the tensor product of the
            state's components with the functions of time.
        observed: for each record, the index of the component it is of and
            the times it is read at. A component may be read by several
            records, or by none.

    Returns:
        The operator from a trajectory to its records, a vector of the
        direct sum of one Euclidean space for each.
    """
    components, space = trajectory.factors
    readers = []
    for index, times in observed:
        select = LinearFunctional.from_derivative_components(
            components, np.eye(components.dim)[index]
        )
        readers.append(
            space.point_evaluation_operator(list(times))
            @ trajectory.contraction(select, axis=0)
        )
    return ColumnLinearOperator(readers)
