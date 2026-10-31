"""The two functionals of a traveltime on the mantle: ray theory's and
finite-frequency theory's, in the paraxial approximation.

The model is ``m = dln c``, the relative perturbation of the speed the
phase travels at, a field on the mantle annulus. Both functionals are point
evaluation of the model composed with a row of weights, and both are built
on the model space as that composition, its adjoint the library's.

**Ray theory.** ``dT = -int m / c dl`` along the ray traced in the
reference planet: the model at the ray's nodes, weighted by minus the
tracer's weights over the travel time.

**Finite-frequency theory.** In ray-centered coordinates, a point ``xi(l) +
q1 e1 + q2 e2`` at arc length ``l`` from the source and offsets ``q1`` in
the ray's plane and ``q2`` across it, the paraxial kernel of a
cross-correlation traveltime is ``K = -sqrt(A1 A2) G(u) / (2 pi c)``, where
``u = (A1 q1^2 + A2 q2^2) / 2`` is the detour time, by which the path
through the point exceeds the ray, ``A_i`` the sums of the Hessians of the
travel time from the two ends, and ``G`` the band's frequency average
(:mod:`.pulse`). Put ``q1 = sqrt(2 u / A1) cos(t)`` and ``q2 = sqrt(2 u /
A2) sin(t)``: then ``sqrt(A1 A2) dq1 dq2 = du dt`` exactly, and

.. code-block:: text

    dT = -int_0^L dl / (2 pi c(l)) int_0^2pi dt int_0^inf du  G(u) m(xi + q) ,

whose integrand is bounded everywhere, the ends of the ray included, where
the sums are infinite and every point closes in on the ray. The quadrature
is Gauss-Legendre in ``u`` on ``[0, u_max]``, the trapezoid rule in ``t``
and the ray functional's own weights in ``l``, so that on a model constant
across every banana the two functionals agree to the quadrature in ``u``,
which is the Gaussian identity ``int G du = 1``. At the two ends every
point is the end itself, and the quadrature there is that one point.

**Boundaries.** The two boundaries of the mantle differ in kind. The outer
core has no three-dimensional structure, so the functional is zero below
the core-mantle boundary by physics: a point there is dropped, and the model
space is the annulus alone. The paraxial kernel knows nothing of the free
surface, and a point above it is dropped too, which is the approximation:
near a surface end, more than half of the banana lies above the surface.
The quadrature keeps the fraction of the weight lost at each boundary.

The rays lie in the equatorial plane, the source at longitude zero, and a
point of the tracer's ray plane ``(x, y, z)`` is the point of the ball of
those Cartesian coordinates.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy.sparse import csr_matrix

from pygeoinf import EuclideanSpace, LinearOperator

from .planet import CORE_RADIUS, RADIUS, Path
from .pulse import Band

#: How far outside the annulus a point may lie and still count as on its
#: boundary, where it is moved: the ends of a ray at the surface, which lie
#: on it to rounding.
_EDGE = 1e-12


def thread_map(
    function: Callable[[Any], Any], items: Iterable[Any], /, *, n_jobs: int | None
) -> list[Any]:
    """``[function(item) for item in items]``, in ``n_jobs`` threads at once,
    or one after another when it is ``None`` or one."""
    items = list(items)
    if not n_jobs or n_jobs <= 1 or len(items) < 2:
        return [function(item) for item in items]
    with ThreadPoolExecutor(max_workers=int(n_jobs)) as pool:
        return list(pool.map(function, items))


def direction(vectors: np.ndarray, /) -> np.ndarray:
    """``(latitude, longitude)`` in degrees of vectors, as a ``(count, 2)``
    array."""
    vectors = np.atleast_2d(vectors)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.column_stack(
        [
            np.degrees(np.arcsin(np.clip(vectors[:, 2], -1.0, 1.0))),
            np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])),
        ]
    )


def lift(points: np.ndarray, /) -> np.ndarray:
    """Points of the ray plane's coordinates as points ``(radius, latitude,
    longitude)`` of the ball, the radius moved onto the annulus where it
    lies on its boundary to rounding."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    radii = np.linalg.norm(points, axis=1)
    radii = np.where(np.abs(radii - RADIUS) <= _EDGE, RADIUS, radii)
    radii = np.where(np.abs(radii - CORE_RADIUS) <= _EDGE, CORE_RADIUS, radii)
    return np.column_stack([radii, direction(points)])


def inside(radii: np.ndarray, /) -> np.ndarray:
    """Whether each radius lies in the mantle annulus, its boundary
    included to rounding."""
    return ~(above_surface(radii) | below_core(radii))


def above_surface(radii: np.ndarray, /) -> np.ndarray:
    """Whether each radius lies above the free surface, beyond rounding."""
    return np.asarray(radii) > RADIUS + _EDGE


def below_core(radii: np.ndarray, /) -> np.ndarray:
    """Whether each radius lies below the core-mantle boundary, beyond
    rounding."""
    return np.asarray(radii) < CORE_RADIUS - _EDGE


@dataclass(frozen=True)
class Rule:
    """The counts of the quadrature in ``(u, t)`` at each node of a ray.

    Attributes:
        detours: Gauss-Legendre nodes in the detour time ``u``.
        angles: the trapezoid rule's nodes in the angle ``t`` about the ray.
        frequencies: Gauss-Legendre nodes in frequency for ``G``.
        cutoff: the exponent ``sigma^2 u^2 / 2`` of ``G``'s envelope at the
            largest detour time kept.
    """

    detours: int
    angles: int
    frequencies: int = 128
    cutoff: float = 30.0


@dataclass(frozen=True)
class Quadrature:
    """The quadrature of one finite-frequency functional: every point of
    every ``(l, u, t)`` and its weight, with the points outside the annulus
    marked.

    The weight dropped is reported at each boundary apart: below the
    core-mantle boundary, where the functional is zero by physics, and
    above the free surface, of which the paraxial kernel knows nothing,
    which is the approximation. Each fraction is signed, by how much the
    functional misses on a constant model, or absolute.

    Attributes:
        points: ``(count, 3)``: the points, in the ray plane's coordinates.
        weights: ``(count,)``: the weight of each, so that ``weights @
            m(points)`` is the traveltime's perturbation.
        kept: ``(count,)``: whether each lies in the mantle annulus.
    """

    points: np.ndarray
    weights: np.ndarray
    kept: np.ndarray

    @property
    def lifted(self) -> np.ndarray:
        """The points kept, as points ``(radius, latitude, longitude)`` of
        the ball."""
        return lift(self.points[self.kept])

    @property
    def kept_weights(self) -> np.ndarray:
        """The weights of the points kept."""
        return self.weights[self.kept]

    def _fraction(self, dropped: np.ndarray, /, *, absolute: bool) -> float:
        """The fraction of the weight, or of the absolute weight, at the
        points marked."""
        weights = np.abs(self.weights) if absolute else self.weights
        # Plus zero, so that nothing dropped is never a negative zero.
        return float(weights[dropped].sum() / weights.sum()) + 0.0

    @property
    def radii(self) -> np.ndarray:
        """The radius of every point."""
        return np.linalg.norm(self.points, axis=1)

    @property
    def clipped(self) -> float:
        """The fraction of the weight dropped with the points outside the
        annulus, at both boundaries: by how much the functional misses on a
        constant model."""
        return self._fraction(~self.kept, absolute=False)

    @property
    def clipped_absolute(self) -> float:
        """The fraction of the absolute weight dropped at both boundaries."""
        return self._fraction(~self.kept, absolute=True)

    @property
    def surface_clipped(self) -> float:
        """The fraction of the weight dropped above the free surface, the
        paraxial kernel's approximation."""
        return self._fraction(above_surface(self.radii), absolute=False)

    @property
    def surface_clipped_absolute(self) -> float:
        """The fraction of the absolute weight dropped above the surface."""
        return self._fraction(above_surface(self.radii), absolute=True)

    @property
    def core_clipped(self) -> float:
        """The fraction of the weight dropped below the core-mantle
        boundary, where the functional is zero by physics."""
        return self._fraction(below_core(self.radii), absolute=False)

    @property
    def core_clipped_absolute(self) -> float:
        """The fraction of the absolute weight dropped below the core-mantle
        boundary."""
        return self._fraction(below_core(self.radii), absolute=True)

    def integrate(
        self, function: Callable[[np.ndarray], np.ndarray], /, *, clip: bool = True
    ) -> float:
        """The functional applied to a function of position, evaluated at the
        points exactly rather than through a space.

        Args:
            function: called with a ``(count, 3)`` array of points ``(radius,
                latitude, longitude)``, returning the values there.
            clip: drop the points outside the annulus, as the functional on
                the model space does; without, the function is asked about
                them too.
        """
        chosen = self.kept if clip else np.ones_like(self.kept)
        points = lift(self.points[chosen])
        return float(self.weights[chosen] @ np.asarray(function(points), dtype=float))


def ray_points(path: Path, /) -> np.ndarray:
    """The ray's nodes, as points ``(radius, latitude, longitude)`` of the
    ball."""
    return lift(path.dynamic.frame.point)


def ray_weights(path: Path, /) -> np.ndarray:
    """The weights of ``dT = -int m / c dl`` at the ray's nodes: minus the
    weights over the travel time."""
    return -np.asarray(path.ray.time_weights, dtype=float)


def offsets(
    path: Path, band: Band, rule: Rule, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The offsets ``q1``, ``q2`` of the quadrature's points from each node
    of the ray, ``(nodes, detours, angles)``, and the weight of each point
    but the ray's own weight at its node. At the ends, where the sums are
    infinite, every offset is zero."""
    u, products = band.detours(
        rule.detours, cutoff=rule.cutoff, frequencies=rule.frequencies
    )
    t = 2.0 * np.pi * np.arange(rule.angles) / rule.angles
    sums = path.dynamic.sums
    with np.errstate(divide="ignore"):
        spread = np.sqrt(2.0 * u[None, :, None, None] / sums[:, None, None, :])
    q1 = spread[..., 0] * np.cos(t)
    q2 = spread[..., 1] * np.sin(t)
    weights = np.broadcast_to(products[:, None] / rule.angles, q1.shape[1:])
    return q1, q2, weights


def quadrature(path: Path, band: Band, rule: Rule, /) -> Quadrature:
    """The quadrature of the finite-frequency functional of one ray in one
    band.

    At each node of the ray inside its ends, ``detours * angles`` points
    ``xi + q1 e1 + q2 e2``, each of weight ``-G(u) w_u w_t / (2 pi c)``
    times the node's weight along the ray, which is the ray functional's,
    ``dl / c`` by the tracer; at each end the one point the offsets all
    collapse to, of weight ``-sum G(u) w_u`` times the ray functional's.
    """
    q1, q2, local = offsets(path, band, rule)
    along = ray_weights(path)
    ends = ~np.all(np.isfinite(path.dynamic.sums), axis=1)
    frame = path.dynamic.frame
    middle = ~ends
    points = frame.place(q1, q2)[middle].reshape(-1, 3)
    weights = (along[middle, None, None] * local[None]).reshape(-1)
    total = float(local.sum())
    points = np.concatenate([points, frame.point[ends]])
    weights = np.concatenate([weights, along[ends] * total])
    radii = np.linalg.norm(points, axis=1)
    return Quadrature(points, weights, inside(radii))


def kernel(
    path: Path,
    band: Band,
    q1: np.ndarray,
    q2: np.ndarray,
    /,
    *,
    nodes: Any = slice(None),
    frequencies: int = 128,
    cutoff: float = 30.0,
) -> np.ndarray:
    """The paraxial ``L2`` kernel ``K = -sqrt(A1 A2) G(u) / (2 pi c)`` at
    offsets ``q1``, ``q2`` from nodes of the ray, ``u = (A1 q1^2 + A2 q2^2) /
    2``, the time by which the path through the point exceeds the ray; zero
    beyond the largest detour time the quadrature keeps, as there.

    Args:
        path: the ray.
        band: the band.
        q1: the offsets in the ray's plane, an array whose leading axis runs
            over the nodes chosen, as in :meth:`~src.rays.Frame.place`.
        q2: the offsets across it, likewise; the two broadcast.
        nodes: the nodes, an index or slice of the ray's; all by default.
        frequencies: Gauss-Legendre nodes in frequency for ``G``.
        cutoff: the exponent of ``G``'s envelope at the largest detour time
            kept.

    Returns:
        The kernel, of the broadcast shape; infinite at the ends of the
        ray, where the sums are.
    """
    q1, q2 = np.broadcast_arrays(np.asarray(q1, float), np.asarray(q2, float))
    expand = (slice(None), *(None,) * (q1.ndim - 1))
    sums = np.atleast_2d(path.dynamic.sums[nodes])
    a1, a2 = sums[:, 0][expand], sums[:, 1][expand]
    c = np.atleast_1d(path.dynamic.velocity[nodes])[expand]
    with np.errstate(invalid="ignore"):
        detour = 0.5 * (a1 * q1**2 + a2 * q2**2)
        values = band.kernel_function(detour, frequencies=frequencies)
        values = np.where(detour <= band.cutoff(cutoff), values, 0.0)
        return -np.sqrt(a1 * a2) * values / (2.0 * np.pi * c)


def composed_operator(
    space: Any, pieces: Sequence[tuple[np.ndarray, np.ndarray]], /
) -> LinearOperator:
    """Functionals as point evaluation at every point of every piece composed
    with the sparse rows of their weights, one row a piece.

    Args:
        space: the model space.
        pieces: for each functional, its points ``(radius, latitude,
            longitude)`` and their weights.

    Returns:
        The operator into ``R^len(pieces)``, its adjoint the library's.
    """
    counts = [len(weights) for _, weights in pieces]
    points = np.concatenate([np.asarray(points) for points, _ in pieces])
    rows = np.repeat(np.arange(len(pieces)), counts)
    columns = np.arange(sum(counts))
    data = np.concatenate([np.asarray(weights) for _, weights in pieces])
    matrix = csr_matrix((data, (rows, columns)), shape=(len(pieces), sum(counts)))
    evaluation = space.point_evaluation_operator(points)
    weights = LinearOperator.from_matrix(
        evaluation.codomain, EuclideanSpace(len(pieces)), matrix, form="components"
    )
    return weights @ evaluation


def assembled_operator(
    space: Any,
    pieces: Sequence[tuple[np.ndarray, np.ndarray]],
    /,
    *,
    n_jobs: int | None = None,
) -> LinearOperator:
    """The same functionals as :func:`composed_operator`, from their rows.

    Each row is the derivative components of one functional, read off its
    own composition by one application of the adjoint, so that each pass
    over the points touches that functional's points alone; the rows are
    formed in ``n_jobs`` threads. The operator is the matrix of those rows:
    an application is a product with it.
    """

    def row(piece: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        return composed_operator(space, [piece]).matrix(form="galerkin")[0]

    rows = np.array(thread_map(row, pieces, n_jobs=n_jobs))
    return LinearOperator.from_matrix(
        space, EuclideanSpace(len(pieces)), rows, form="galerkin"
    )
