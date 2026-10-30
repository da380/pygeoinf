"""Evaluation of spectral-element fields at scattered points, matrix-free.

Along the mesh a field of these spaces is, on each element, the polynomial
through its values at the element's ``ngll`` nodes. The kept eigenfunctions
are such piecewise polynomials, so the field with given components is one
too, and its value at a position is a combination of its values at the nodes
of the position's element, the weights being the Lagrange basis of those
nodes at the position. In a ball or a disk the values along the mesh are
those of coefficient functions, one per spherical harmonic or Fourier mode,
and the field at a point is the sum over the harmonics of the harmonic at the
point's angles times its coefficient function at the point's radius.

:class:`LocalEvaluation` does that for many points at once. From the
components it synthesizes the coefficient functions at every node of the
mesh; then, a chunk of points at a time and within a chunk one element at a
time, it multiplies the angular functions at the points by the coefficient
functions at the element's nodes, one matrix product, and contracts the
result with the points' Lagrange weights. Its transpose runs the same way
back: each datum, times the angular functions at its point, is spread onto
the nodes of its element, and the coefficient functions so accumulated are
projected onto the kept modes, which is the transpose of the synthesis.
Nothing the size of the points times the components is formed: what is held
is the points' elements and weights, ``ngll`` numbers each, and the angular
functions of one chunk, or of all the points when they fit
:data:`_TABLE_BUDGET`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import numpy as np
from planetmodel.mesh1d import Mesh1D

__all__ = ["LocalEvaluation", "harmonic_table"]

#: Relative allowance on the ends of the domain, as ``planetmodel`` grants
#: its own evaluation: a position this close outside is moved onto the end,
#: and one further out refused.
_POSITION_RTOL = 1e-12

#: Angular-function values per chunk, harmonics times points, so that a
#: chunk's work space is 32 MB whatever the truncation: 946 points on a ball
#: of degree 64. The chunk size hardly matters to the time: on a ball of
#: degree 64 at 20 000 points an application took the same to within
#: fifteen percent from a hundred points a chunk to four thousand.
_CHUNK_ENTRIES = 4_000_000

#: The bytes of angular functions an evaluation keeps between applications,
#: rather than recomputing them chunk by chunk: the same budget as a dense
#: evaluation's. Kept, they cost eight bytes per harmonic per point for the
#: life of the operator, and save most of an application: on a ball of degree
#: 64 at 20 000 points, 676 MB of harmonics, an application took 0.08 s with
#: them kept and 0.2 s without, on one thread; at degree 24 and 83 000
#: points, 415 MB, 0.045 s against 0.15 s, the dense form taking 0.07 s and
#: 2.8 GB.
_TABLE_BUDGET = 256 * 2**20


@lru_cache(maxsize=8)
def _legendre_recurrence(lmax: int, /) -> tuple[tuple[np.ndarray, ...], ...]:
    """The coefficients of the three-term recurrence in degree of the
    normalized associated Legendre functions, for each degree from two up to
    ``lmax``: ``a`` and ``b`` over the orders ``0 ... l - 2``, as columns."""
    first, second = [], []
    for degree in range(2, lmax + 1):
        orders = np.arange(degree - 1, dtype=float)
        squares = float(degree) ** 2 - orders**2
        first.append(np.sqrt((4.0 * degree**2 - 1.0) / squares)[:, None])
        below = float(degree - 1) ** 2
        second.append(np.sqrt((below - orders**2) / (4.0 * below - 1.0))[:, None])
    return tuple(first), tuple(second)


def harmonic_table(
    lmax: int, colatitudes: np.ndarray, longitudes: np.ndarray, /
) -> np.ndarray:
    """The real spherical harmonics up to ``lmax`` at many points, packed.

    Orthonormal on the unit sphere and without the Condon-Shortley phase, the
    convention of ``planetmodel.harmonics.real_harmonics``, whose values these
    are to rounding; packed as ``planetmodel.harmonics.packing`` orders them,
    by degree and within a degree the cosine harmonics of orders ``0 ... l``,
    then the sine ones of orders ``1 ... l``, which is how a ball orders the
    harmonics of its components. By the standard recurrence in degree, for
    every point at once, from the sectoral function of each order.

    Args:
        lmax: the largest degree.
        colatitudes: in radians, in ``[0, pi]``.
        longitudes: in radians, as many.

    Returns:
        Shape ``((lmax + 1)^2, points)``.
    """
    x = np.cos(colatitudes)
    y = np.sin(colatitudes)
    count = x.size
    orders = np.arange(lmax + 1)
    angles = orders[:, None] * np.asarray(longitudes, dtype=float)[None, :]
    cosines, sines = np.cos(angles), np.sin(angles)
    cosines[1:] *= np.sqrt(2.0)
    sines *= np.sqrt(2.0)
    first, second = _legendre_recurrence(lmax)

    out = np.empty(((lmax + 1) ** 2, count))
    previous = np.full((1, count), 1.0 / np.sqrt(4.0 * np.pi))
    before = previous
    out[0] = previous[0]
    for degree in range(1, lmax + 1):
        current = np.empty((degree + 1, count))
        if degree >= 2:
            np.multiply(x, previous[: degree - 1], out=current[: degree - 1])
            current[: degree - 1] -= second[degree - 2] * before
            current[: degree - 1] *= first[degree - 2]
        np.multiply(
            np.sqrt(2.0 * degree + 1.0) * x,
            previous[degree - 1],
            out=current[degree - 1],
        )
        np.multiply(
            np.sqrt((2.0 * degree + 1.0) / (2.0 * degree)) * y,
            previous[degree - 1],
            out=current[degree],
        )
        start = degree**2
        np.multiply(current, cosines[: degree + 1], out=out[start : start + degree + 1])
        np.multiply(
            current[1:],
            sines[1 : degree + 1],
            out=out[start + degree + 1 : start + 2 * degree + 1],
        )
        before, previous = previous[:degree], current
    return out


class LocalEvaluation:
    """Evaluation at scattered points of the field with given components,
    and its transpose, through the elements the points lie in.

    The geometry supplies the synthesis of its coefficient functions at the
    nodes and the transpose of that synthesis, and, in a ball or a disk, the
    angular functions at the points; what is here is the same for all of
    them.
    """

    def __init__(
        self,
        mesh: Mesh1D,
        positions: np.ndarray,
        interval: tuple[float, float],
        synthesis: Callable[[np.ndarray], np.ndarray],
        projection: Callable[[np.ndarray], np.ndarray],
        /,
        *,
        angular: Callable[[np.ndarray], np.ndarray] | None = None,
        count: int = 1,
    ) -> None:
        """
        Args:
            mesh: the padded mesh.
            positions: each point's position along the mesh: its coordinate
                on an axis, its radius in a ball or a disk.
            interval: the domain along the mesh.
            synthesis: components to the coefficient functions at every node
                of the mesh, shape ``(count, nodes)``.
            projection: the transpose of *synthesis*, from such functions to
                components.
            angular: given the indices of some of the points, the angular
                functions at them, shape ``(count, len(indices))``. Omitted on
                an axis, which has one coefficient function, the field, and
                nothing to sum over.
            count: the number of coefficient functions.

        Raises:
            ValueError: if a position lies outside the interval.
        """
        positions = np.asarray(positions, dtype=float).reshape(-1)
        lower, upper = (float(end) for end in interval)
        slack = _POSITION_RTOL * (upper - lower)
        outside = (positions < lower - slack) | (positions > upper + slack)
        if np.any(outside):
            raise ValueError(
                f"Every point must lie in the domain, at a position in "
                f"[{lower:g}, {upper:g}] along the mesh; one is at "
                f"{positions[outside][0]:g}."
            )
        elements, weights = mesh.interpolation_weights(np.clip(positions, lower, upper))
        order = np.argsort(elements, kind="stable")
        self._order = order
        self._elements = elements[order]
        self._weights = weights[order]
        self._gmap = mesh.gmap
        self._nglob = mesh.nglob
        self._synthesis = synthesis
        self._projection = projection
        self._angular = angular
        self._count = int(count)
        self._table: np.ndarray | None = None
        table_bytes = 8 * self._count * order.size
        self._keep = angular is not None and table_bytes <= _TABLE_BUDGET

        if angular is None:
            self._nodes = mesh.gmap[self._elements]
            self._chunks: list[tuple[int, int, list[tuple[int, int, int]]]] = []
            return
        # Runs of points in one element, within chunks of points: the points
        # are sorted by element, so each chunk holds few runs.
        size = max(1, _CHUNK_ENTRIES // self._count)
        self._chunks = []
        for start in range(0, order.size, size):
            stop = min(start + size, order.size)
            inside = self._elements[start:stop]
            cuts = [0, *(np.flatnonzero(np.diff(inside)) + 1), stop - start]
            runs = [
                (int(inside[a]), int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:])
            ]
            self._chunks.append((start, stop, runs))

    @property
    def size(self) -> int:
        """The number of points."""
        return int(self._order.size)

    def _chunk_table(self, start: int, stop: int, /) -> np.ndarray:
        """The angular functions at the sorted points ``start ... stop``."""
        if not self._keep:
            return self._angular(self._order[start:stop])
        if self._table is None:
            self._table = self._angular(self._order)
        return self._table[:, start:stop]

    def values(self, components: np.ndarray, /) -> np.ndarray:
        """The field with these components at every point.

        Args:
            components: of the space.

        Returns:
            One value per point, in the order the points were given.
        """
        functions = np.asarray(self._synthesis(components), dtype=float)
        ordered = np.empty(self.size)
        if self._angular is None:
            ordered[:] = np.einsum("ij,ij->i", functions[0][self._nodes], self._weights)
        for start, stop, runs in self._chunks:
            table = self._chunk_table(start, stop)
            for element, a, b in runs:
                local = table[:, a:b].T @ functions[:, self._gmap[element]]
                ordered[start + a : start + b] = np.einsum(
                    "ij,ij->i", local, self._weights[start + a : start + b]
                )
        out = np.empty(self.size)
        out[self._order] = ordered
        return out

    def derivative_components(self, data: np.ndarray, /) -> np.ndarray:
        """The components of the functional ``x -> sum_i data_i x(p_i)``:
        the transpose of :meth:`values`, with no metric applied.

        Args:
            data: one weight per point, in the order the points were given.

        Returns:
            The derivative components.
        """
        data = np.asarray(data, dtype=float).reshape(-1)
        weighted = self._weights * data[self._order][:, None]
        if self._angular is None:
            nodal = np.bincount(
                self._nodes.ravel(), weights=weighted.ravel(), minlength=self._nglob
            )
            return self._projection(nodal[None, :])
        functions = np.zeros((self._count, self._nglob))
        for start, stop, runs in self._chunks:
            table = self._chunk_table(start, stop)
            for element, a, b in runs:
                functions[:, self._gmap[element]] += (
                    table[:, a:b] @ weighted[start + a : start + b]
                )
        return self._projection(functions)
