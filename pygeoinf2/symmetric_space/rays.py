"""
Rays through a velocity field, and the travel times along them.

The forward map of ray tomography is nonlinear: the travel time between a
source and a receiver is the integral of the slowness along the ray, and the
ray itself depends on the slowness. Here the velocity is ``c₀ exp(m)`` with
``m`` a field of a symmetric space, so that it stays positive and the model
``m = 0`` is the constant background whose rays are the geodesics; the
travel time is then ``∫ exp(−m) / c₀ dl`` along the ray that Fermat's
principle makes stationary.

Two ways of finding that ray are offered. **Bending** starts from the
geodesic sampled at a number of nodes and moves each node across the path
until the discrete travel time is stationary, by a damped Newton method
whose gradient and tridiagonal Hessian come from finite differences of the
travel times of the segments, taken three nodes at a time so that every
path's nodes are perturbed together and the field is evaluated at all of
them in one transform. **Shooting** integrates the ray equation from the
source with a take-off angle, finds the angle at which the ray arrives at
the receiver by a root find on the miss, and does so for every path in
lockstep. Both find the ray continuously connected to the geodesic, not
necessarily the first arrival where several rays connect a pair.

The derivative of the travel time with respect to the model is Fermat's
principle: the ray is stationary, so only the slowness along it varies, and
the derivative is the path integral of ``−exp(−m) / c₀`` along the traced
ray, which is the same ``W E`` construction the linear path operators use.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np

from ..algebra.operators import EvaluationError, LinearOperator, Operator
from ..algebra.spaces import EuclideanSpace
from .base import _gauss_legendre, _weight_matrix, _weight_operator

Method = Literal["bending", "shooting"]


# --------------------------------------------------------------------- #
#                                 Charts                                 #
# --------------------------------------------------------------------- #


class _Chart:
    """What the tracer needs of a geometry, in an embedding where rays are vectors."""

    dimension: int  # of the embedding

    def __init__(self, space: Any) -> None:
        self.space = space

    def embed(self, points: Sequence[Any]) -> np.ndarray:
        """The points as vectors of the embedding, one row each.

        Raises:
            NotImplementedError: on the base chart; a geometry supplies it.
        """
        raise NotImplementedError

    def unembed(self, vectors: np.ndarray) -> list[Any]:
        """Vectors of the embedding back as points of the space.

        Raises:
            NotImplementedError: on the base chart; a geometry supplies it.
        """
        raise NotImplementedError

    def evaluation_points(self, vectors: np.ndarray) -> Any:
        """Vectors of the embedding as points to evaluate a field at.

        The points of :meth:`unembed`, unless the geometry has something to
        add for the tracer's own evaluations.
        """
        return self.unembed(vectors)

    def separation_end(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """The receiver, placed so that the geodesic from the source is direct."""
        return end

    def between(
        self, a: np.ndarray, b: np.ndarray, fractions: np.ndarray
    ) -> np.ndarray:
        """Points along the geodesic from ``a`` to ``b`` at the given fractions.

        ``a`` and ``b`` have shape ``(n, k)``; the result ``(n, len(fractions), k)``.

                Raises:
                    NotImplementedError: on the base chart; a geometry supplies it.
        """
        raise NotImplementedError

    def length(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Geodesic distances between rows of ``a`` and ``b``.

        Raises:
            NotImplementedError: on the base chart; a geometry supplies it.
        """
        raise NotImplementedError

    def transverse(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """The direction a node of each path is displaced in, per path.

        Raises:
            NotImplementedError: on the base chart; a geometry supplies it.
        """
        raise NotImplementedError

    def displace(
        self, nodes: np.ndarray, offsets: np.ndarray, transverse: np.ndarray
    ) -> np.ndarray:
        """Nodes ``(P, N, k)`` moved by offsets ``(P, N)`` along the transverse directions ``(P, k)``.

        Raises:
            NotImplementedError: on the base chart; a geometry supplies it.
        """
        raise NotImplementedError

    def gradient_vectors(
        self, field: Any, vectors: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        """The field's gradient at embedded points, as embedded vectors.

        Raises:
            NotImplementedError: on the base chart; a geometry supplies it.
        """
        raise NotImplementedError

    def velocity(self, t: np.ndarray) -> np.ndarray:
        """``dx/dl`` for a unit tangent ``t``: the tangent itself in a flat embedding."""
        return t

    def acceleration(
        self, x: np.ndarray, t: np.ndarray, grad_log_slowness: np.ndarray
    ) -> np.ndarray:
        """``dt/dl`` for a ray at ``x`` with unit tangent ``t``: the transverse gradient, plus curvature."""
        along = np.sum(t * grad_log_slowness, axis=-1, keepdims=True)
        return grad_log_slowness - along * t

    def project(self, x: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Bring a ray state back onto the geometry after a step."""
        return x, t

    def frame(
        self, start: np.ndarray, end: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """The unit direction of the geodesic at the source, and the transverse unit.

        Raises:
            NotImplementedError: on the base chart; a geometry supplies it.
        """
        raise NotImplementedError

    def along_across(
        self, start: np.ndarray, end: np.ndarray, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates of ``x`` along the geodesic and across it, per path.

        Raises:
            NotImplementedError: on the base chart; a geometry supplies it.
        """
        raise NotImplementedError

    def takeoff(
        self, start: np.ndarray, end: np.ndarray, angle: np.ndarray
    ) -> np.ndarray:
        """The unit tangent at the source, turned by ``angle`` from the geodesic's."""
        direction, across = self.frame(start, end)
        return np.cos(angle)[:, None] * direction + np.sin(angle)[:, None] * across


class _BoxChart(_Chart):
    """A periodic or bounded box: straight lines, unwrapped coordinates."""

    def __init__(self, space: Any) -> None:
        super().__init__(space)
        self.dimension = space.spatial_dimension

    def embed(self, points: Sequence[Any]) -> np.ndarray:
        """The points as vectors of the embedding, one row each."""
        return np.stack([np.atleast_1d(np.asarray(p, dtype=float)) for p in points])

    def unembed(self, vectors: np.ndarray) -> Any:
        """Vectors of the embedding back as points of the space: an array of rows."""
        return np.asarray(self.space._wrap(vectors), dtype=float)

    def evaluation_points(self, vectors: np.ndarray) -> Any:
        """The points, prepared without the bounded box's domain check.

        A ray between two points of a bounded box can bend out of the domain
        on its way to convergence, or at it, and the tracer reads the field
        in the padding there rather than stop.
        """
        if hasattr(self.space, "bounds"):
            return self.space._prepared(np.asarray(vectors, dtype=float), check=False)
        return self.unembed(vectors)

    def separation_end(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """The receiver, placed so that the geodesic from the source is direct."""
        return np.stack([a + self.space._separation(a, b) for a, b in zip(start, end)])

    def between(
        self, a: np.ndarray, b: np.ndarray, fractions: np.ndarray
    ) -> np.ndarray:
        """Points along the geodesic from ``a`` to ``b`` at the given fractions."""
        return a[:, None, :] + fractions[None, :, None] * (b - a)[:, None, :]

    def length(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Geodesic distances between the rows of ``a`` and ``b``."""
        return np.linalg.norm(b - a, axis=-1)

    def transverse(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """The direction each path's nodes are displaced in."""
        if self.dimension != 2:
            return np.zeros_like(start)
        direction = end - start
        direction = direction / np.linalg.norm(direction, axis=-1, keepdims=True)
        return np.column_stack([-direction[:, 1], direction[:, 0]])

    def displace(
        self, nodes: np.ndarray, offsets: np.ndarray, transverse: np.ndarray
    ) -> np.ndarray:
        """The nodes moved by the offsets along the transverse directions."""
        return nodes + offsets[:, :, None] * transverse[:, None, :]

    def gradient_vectors(
        self, field: Any, vectors: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        """The field's gradient at embedded points, as vectors of the embedding."""
        return self.space.evaluate_gradient(
            field, self.evaluation_points(vectors), **kwargs
        )

    def frame(
        self, start: np.ndarray, end: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """The geodesic's unit direction at the source, and the transverse unit."""
        direction = end - start
        direction = direction / np.linalg.norm(direction, axis=-1, keepdims=True)
        return direction, self.transverse(start, end)

    def along_across(
        self, start: np.ndarray, end: np.ndarray, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates of ``x`` along each path's geodesic and across it."""
        direction, across = self.frame(start, end)
        offset = x - start
        return np.sum(offset * direction, axis=-1), np.sum(offset * across, axis=-1)


class _SphereChart(_Chart):
    """The sphere: unit vectors, great circles, rotations toward a pole."""

    dimension = 3

    def __init__(self, space: Any) -> None:
        super().__init__(space)
        self.radius = float(space.radius)

    def embed(self, points: Sequence[Any]) -> np.ndarray:
        """The points as vectors of the embedding, one row each."""
        return np.stack([self.space._to_vector(p) for p in points])

    def unembed(self, vectors: np.ndarray) -> list[Any]:
        """Vectors of the embedding back as points of the space."""
        return list(self.space._to_points(vectors))

    def between(
        self, a: np.ndarray, b: np.ndarray, fractions: np.ndarray
    ) -> np.ndarray:
        """Points along the geodesic from ``a`` to ``b`` at the given fractions."""
        cosine = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
        angle = np.arccos(cosine)[:, None]
        sine = np.sin(angle)
        safe = np.where(sine > 1.0e-12, sine, 1.0)
        first = np.sin((1.0 - fractions[None, :]) * angle) / safe
        second = np.sin(fractions[None, :] * angle) / safe
        tiny = (sine <= 1.0e-12)[:, :, None]
        out = first[:, :, None] * a[:, None, :] + second[:, :, None] * b[:, None, :]
        linear = a[:, None, :] + fractions[None, :, None] * (b - a)[:, None, :]
        out = np.where(tiny, linear, out)
        return out / np.linalg.norm(out, axis=-1, keepdims=True)

    def length(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Geodesic distances between the rows of ``a`` and ``b``."""
        cross = np.linalg.norm(np.cross(a, b), axis=-1)
        dot = np.sum(a * b, axis=-1)
        return self.radius * np.arctan2(cross, dot)

    def transverse(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """The pole of each path's great circle, which is the direction its nodes move in.

        Raises:
            ValueError: if a path's endpoints coincide or are antipodal, so
                that the great circle is not unique.
        """
        pole = np.cross(start, end)
        norm = np.linalg.norm(pole, axis=-1, keepdims=True)
        if np.any(norm < 1.0e-12):
            raise ValueError(
                "A path's endpoints coincide or are antipodal, so its great "
                "circle is not unique."
            )
        return pole / norm

    def displace(
        self, nodes: np.ndarray, offsets: np.ndarray, transverse: np.ndarray
    ) -> np.ndarray:
        """The nodes moved by the offsets along the transverse directions."""
        angle = (offsets / self.radius)[:, :, None]
        return np.cos(angle) * nodes + np.sin(angle) * transverse[:, None, :]

    def _tangent_frame(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Southward and eastward unit vectors at each unit vector."""
        up = np.array([0.0, 0.0, 1.0])
        east = np.cross(up, vectors)
        norm = np.linalg.norm(east, axis=-1, keepdims=True)
        east = east / np.where(norm > 1.0e-15, norm, 1.0)
        south = np.cross(east, vectors)
        return south, east

    def gradient_vectors(
        self, field: Any, vectors: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        """The field's gradient at embedded points, as vectors of the embedding."""
        components = self.space.evaluate_gradient(
            field, self.evaluation_points(vectors), **kwargs
        )
        south, east = self._tangent_frame(vectors)
        return components[:, :1] * south + components[:, 1:] * east

    def velocity(self, t: np.ndarray) -> np.ndarray:
        """A unit vector moves by ``t / R`` per unit of arc length."""
        return t / self.radius

    def acceleration(
        self, x: np.ndarray, t: np.ndarray, grad_log_slowness: np.ndarray
    ) -> np.ndarray:
        """``dt/dl`` for a ray at ``x`` with unit tangent ``t``."""
        transverse = super().acceleration(x, t, grad_log_slowness)
        # Great circles curve toward the centre: the geodesic term of the
        # ray equation on a sphere of radius R, for a unit tangent.
        return transverse - x / self.radius

    def project(self, x: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """A ray state brought back onto the geometry after a step."""
        x = x / np.linalg.norm(x, axis=-1, keepdims=True)
        t = t - np.sum(t * x, axis=-1, keepdims=True) * x
        return x, t / np.linalg.norm(t, axis=-1, keepdims=True)

    def frame(
        self, start: np.ndarray, end: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """The geodesic's unit direction at the source, and the transverse unit."""
        pole = self.transverse(start, end)
        direction = np.cross(pole, start)
        return direction / np.linalg.norm(direction, axis=-1, keepdims=True), pole

    def along_across(
        self, start: np.ndarray, end: np.ndarray, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates of ``x`` along each path's geodesic and across it."""
        direction, pole = self.frame(start, end)
        along = self.radius * np.arctan2(
            np.sum(x * direction, axis=-1), np.sum(x * start, axis=-1)
        )
        across = self.radius * np.arcsin(np.clip(np.sum(x * pole, axis=-1), -1.0, 1.0))
        return along, across


def _chart(space: Any) -> _Chart:
    from .fourier import PeriodicBox
    from .sphere import Sphere

    if isinstance(space, Sphere):
        return _SphereChart(space)
    if isinstance(space, PeriodicBox):
        if space.spatial_dimension > 2:
            raise NotImplementedError("Rays are traced in one and two dimensions.")
        return _BoxChart(space)
    raise NotImplementedError(f"{type(space).__name__} has no ray tracer.")


# --------------------------------------------------------------------- #
#                         The discrete travel time                       #
# --------------------------------------------------------------------- #


class _Slowness:
    """``exp(−m) / c₀`` and its gradient, at embedded points, in one transform."""

    def __init__(
        self, chart: _Chart, model: Any, reference_velocity: float, **kwargs: Any
    ) -> None:
        self.chart = chart
        self.model = model
        self.c0 = float(reference_velocity)
        self.kwargs = kwargs

    def values(self, vectors: np.ndarray) -> np.ndarray:
        """The slowness at embedded points, shaped like the points without their last axis."""
        flat = vectors.reshape(-1, vectors.shape[-1])
        if not np.all(np.isfinite(flat)):
            # A slowness model far outside the range the bending can follow
            # sends ray points to infinity; the transform behind the
            # evaluation does not survive them, and the optimizer that asked
            # treats a refusal as a step too far.
            raise EvaluationError(
                "The bent rays left the finite plane: the slowness model is "
                "outside the range the ray tracer can follow."
            )
        m = self.chart.space.evaluate(
            self.model, self.chart.evaluation_points(flat), **self.kwargs
        )
        return (np.exp(-m) / self.c0).reshape(vectors.shape[:-1])

    def gradient_log(self, vectors: np.ndarray) -> np.ndarray:
        """``∇ ln s == −∇m``, as embedded vectors."""
        flat = vectors.reshape(-1, vectors.shape[-1])
        g = self.chart.gradient_vectors(self.model, flat, **self.kwargs)
        return (-g).reshape(vectors.shape)


class _Polylines:
    """The rays of all paths as polylines, and the composite rule along them."""

    def __init__(
        self,
        chart: _Chart,
        start: np.ndarray,
        end: np.ndarray,
        interior: int,
        nodes_per_segment: int,
    ) -> None:
        self.chart = chart
        self.start, self.end = start, end
        self.interior = interior
        fractions = np.linspace(0.0, 1.0, interior + 2)[1:-1]
        self.base = chart.between(start, end, fractions)  # (P, N, k)
        self.transverse = chart.transverse(start, end)
        abscissae, weights = _gauss_legendre(nodes_per_segment)
        self.fractions = 0.5 * (abscissae + 1.0)
        self.weights = 0.5 * weights  # sum to one

    def nodes(self, offsets: np.ndarray) -> np.ndarray:
        """The polyline's nodes, endpoints included: ``(P, N + 2, k)``."""
        moved = self.chart.displace(self.base, offsets, self.transverse)
        return np.concatenate(
            [self.start[:, None, :], moved, self.end[:, None, :]], axis=1
        )

    def rule(self, polyline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Quadrature points ``(P, S, q, k)`` and weights ``(P, S, q)`` along the segments."""
        a, b = polyline[:, :-1, :], polyline[:, 1:, :]
        shape = a.shape
        points = self.chart.between(
            a.reshape(-1, shape[-1]), b.reshape(-1, shape[-1]), self.fractions
        )
        points = points.reshape(shape[0], shape[1], -1, shape[-1])
        lengths = self.chart.length(a.reshape(-1, shape[-1]), b.reshape(-1, shape[-1]))
        weights = (
            lengths.reshape(shape[0], shape[1])[:, :, None]
            * self.weights[None, None, :]
        )
        return points, weights

    def segment_times(self, slowness: _Slowness, offsets: np.ndarray) -> np.ndarray:
        """The travel time of each segment, ``(P, S)``."""
        points, weights = self.rule(self.nodes(offsets))
        return np.sum(weights * slowness.values(points), axis=-1)


class _Bands:
    """The gradient and tridiagonal Hessian of a sum over segments, by coloured differences.

    For a function of the node offsets that is a sum of segment terms, each
    depending on the two nodes it joins, the nodes are coloured three ways
    so that no segment sees two perturbed nodes of one colour. Six
    perturbed evaluations give the gradient and the Hessian's diagonal,
    six joint ones its off-diagonal, all second order in the step,
    whatever the number of paths.
    """

    def __init__(self, count: int) -> None:
        self.colors = [np.arange(count) % 3 == r for r in range(3)]

    def gradient(
        self,
        times: Any,
        offsets: np.ndarray,
        h: np.ndarray,
        current: np.ndarray,
        *,
        hessian: bool,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """``(gradient, diagonal, off_diagonal)`` of the segment sum at the offsets.

        Args:
            times: the segment terms as a function of the offsets, ``(P, S)``.
            offsets: where to differentiate, ``(P, N)``.
            h: the step per path.
            current: ``times(offsets)``, already evaluated.
            hessian: whether the bands are wanted as well.
        """
        P, N = offsets.shape
        gradient = np.zeros((P, N))
        diagonal = np.zeros((P, N)) if hessian else None
        off = np.zeros((P, max(N - 1, 0))) if hessian else None
        plus_times, minus_times = [], []
        for r in range(3):
            step = np.where(self.colors[r][None, :], h[:, None], 0.0)
            plus = times(offsets + step)
            minus = times(offsets - step)
            plus_times.append(plus)
            minus_times.append(minus)
            nodes = np.flatnonzero(self.colors[r])
            # Node i sits between segments i and i + 1.
            left = (
                plus[:, nodes]
                + plus[:, nodes + 1]
                - minus[:, nodes]
                - minus[:, nodes + 1]
            )
            gradient[:, nodes] = left / (2.0 * h[:, None])
            if hessian:
                curvature = (
                    plus[:, nodes]
                    + plus[:, nodes + 1]
                    - 2.0 * (current[:, nodes] + current[:, nodes + 1])
                    + minus[:, nodes]
                    + minus[:, nodes + 1]
                )
                diagonal[:, nodes] = curvature / h[:, None] ** 2
        if hessian:
            for r in range(3):
                # Nodes of colours r and r + 1 moved together: the mixed term
                # of the segment between consecutive nodes i and i + 1.
                step = np.where(
                    (self.colors[r] | self.colors[(r + 1) % 3])[None, :],
                    h[:, None],
                    0.0,
                )
                both_plus = times(offsets + step)
                both_minus = times(offsets - step)
                nodes = np.flatnonzero(self.colors[r])
                nodes = nodes[nodes + 1 < N]
                segment = nodes + 1
                # Second order in h, from the two joint moves and the four
                # single ones already taken.
                other = (r + 1) % 3
                mixed = (
                    both_plus[:, segment]
                    + both_minus[:, segment]
                    - plus_times[r][:, segment]
                    - minus_times[r][:, segment]
                    - plus_times[other][:, segment]
                    - minus_times[other][:, segment]
                    + 2.0 * current[:, segment]
                ) / (2.0 * h[:, None] ** 2)
                off[:, nodes] = mixed
        return gradient, diagonal, off


def _solve_banded(
    diagonal: np.ndarray, off: np.ndarray, right: np.ndarray, damping: np.ndarray | None
) -> np.ndarray:
    """Solve the tridiagonal systems of every path, with optional Levenberg damping."""
    from scipy.linalg import solve_banded

    P, N = right.shape
    solution = np.zeros((P, N))
    for p in range(P):
        band = np.zeros((3, N))
        band[1] = diagonal[p]
        if damping is not None:
            band[1] = band[1] + damping[p] * np.maximum(np.abs(diagonal[p]), 1.0e-300)
        if N > 1:
            band[0, 1:] = off[p]
            band[2, :-1] = off[p]
        try:
            solution[p] = solve_banded((1, 1), band, right[p])
        except (np.linalg.LinAlgError, ValueError):
            solution[p] = np.nan
    return solution


def _bend(
    polylines: _Polylines,
    slowness: _Slowness,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The offsets making every path's discrete travel time stationary, and the Hessian there.

    A damped Newton method on all paths at once, the gradient and the
    tridiagonal Hessian of the travel time in the offsets by the coloured
    differences of :class:`_Bands`: thirteen batched evaluations of the
    field per step, whatever the number of paths and nodes. Returned with the
    Hessian's bands and the step they were taken at, which the second
    derivative in the model reuses.
    """
    chart = polylines.chart
    P, N = polylines.base.shape[:2]
    lengths = chart.length(polylines.start, polylines.end)
    spacing = lengths / (N + 1)
    h = 1.0e-3 * spacing
    offsets = np.zeros((P, N))
    if N == 0:
        return offsets, (np.zeros((P, 0)), np.zeros((P, 0)), h)
    damping = np.full(P, 1.0e-3)
    bands = _Bands(N)

    def times(u: np.ndarray) -> np.ndarray:
        return polylines.segment_times(slowness, u)

    current = times(offsets)
    total = current.sum(axis=1)
    scale = np.full(P, np.inf)
    for _ in range(max_iterations):
        gradient, diagonal, off = bands.gradient(
            times, offsets, h, current, hessian=True
        )
        proposal = _solve_banded(diagonal, off, -gradient, damping)
        failed = ~np.all(np.isfinite(proposal), axis=1)
        proposal[failed] = -gradient[failed] * spacing[failed, None] ** 2
        scale = np.max(np.abs(proposal), axis=1)
        converged = scale <= tolerance * lengths
        if np.all(converged):
            return offsets, (diagonal, off, h)
        trial = offsets + np.where(converged[:, None], 0.0, proposal)
        trial_times = times(trial)
        trial_total = trial_times.sum(axis=1)
        accepted = (trial_total <= total + 1.0e-12 * np.abs(total)) | converged
        # A rejected step shrinks: halve it until the time no longer rises.
        factor = np.ones(P)
        for _ in range(12):
            if np.all(accepted):
                break
            factor = np.where(accepted, factor, 0.5 * factor)
            trial = offsets + factor[:, None] * np.where(
                converged[:, None], 0.0, proposal
            )
            trial_times = times(trial)
            trial_total = trial_times.sum(axis=1)
            accepted = accepted | (trial_total <= total + 1.0e-12 * np.abs(total))
        damping = np.where(accepted, np.maximum(damping * 0.3, 1.0e-6), damping * 10.0)
        offsets = np.where(accepted[:, None], trial, offsets)
        current = np.where(accepted[:, None], trial_times, current)
        total = current.sum(axis=1)
    raise EvaluationError(
        f"Bending did not converge in {max_iterations} iterations; the largest "
        f"step left is {float(np.max(scale / lengths)):.2e} of a path's length "
        f"against a tolerance of {tolerance:.1e}."
    )


# --------------------------------------------------------------------- #
#                                Shooting                                #
# --------------------------------------------------------------------- #


def _trace(
    chart: _Chart,
    slowness: _Slowness,
    start: np.ndarray,
    end: np.ndarray,
    angle: np.ndarray,
    steps: int,
    *,
    tangent: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    """Rays from the sources at the take-off angles: miss, travel time, nodes, tangents.

    Fourth-order Runge-Kutta in arc length on ``dx/dl = v(t)``,
    ``dt/dl = ∇_⊥ ln s`` (plus the curvature term of the geometry) and
    ``dT/dl = s``, stepped until the ray passes the receiver's position
    along the geodesic; the miss is the transverse coordinate there, by
    interpolation between the last two states. The nodes returned end at
    the receiver: every node from the crossing on is the receiver itself,
    so a polyline's last segment has its exact length and the padding
    segments none.

    With ``tangent`` a model direction ``δm₁``, two columns of the tangent
    linear system are integrated alongside, through the same stages: the
    ray's response to ``δm₁`` at a fixed take-off angle, and its response
    to a unit change of the angle. The field's second derivative along a
    response enters as a central difference of its gradient. Returned as
    the two responses' node histories and their misses, from which the
    response that keeps the receiver is assembled.
    """
    P = start.shape[0]
    lengths = chart.length(start, end)
    along_end, _ = chart.along_across(start, end, end)
    dl = 1.5 * lengths / steps
    x = start.copy()
    t = chart.takeoff(start, end, angle)
    time = np.zeros(P)
    arrived = np.zeros(P, dtype=bool)
    miss = np.full(P, np.nan)
    times_at = np.full(P, np.nan)
    cross_index = np.full(P, -1)
    history_x = [x.copy()]
    prev_along = chart.along_across(start, end, x)[0]
    prev_across, prev_time = np.zeros(P), time.copy()

    columns = 0 if tangent is None else 2
    if columns:
        along_unit, across_unit = chart.frame(start, end)
        # Column 0: δm₁ at fixed angle, starting from rest. Column 1: a
        # unit change of the take-off angle, which turns the tangent.
        dx = np.zeros((2, P, x.shape[1]))
        dt = np.stack(
            [np.zeros_like(t), chart.takeoff(start, end, angle + 0.5 * np.pi)]
        )
        dmiss = np.full((2, P), np.nan)
        prev_dacross = np.zeros((2, P))
        prev_dalong = np.zeros((2, P))
        history_dx = [dx.copy()]
        scale = 1.0e-6 * (np.ones(P) if chart.dimension == 3 else lengths)

    def rates(
        x: np.ndarray, t: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        g = slowness.gradient_log(x)
        return chart.velocity(t), chart.acceleration(x, t, g), slowness.values(x), g

    def tangent_rates(
        x: np.ndarray, t: np.ndarray, g: np.ndarray, dx: np.ndarray, dt: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """The linearized rates at ``(x, t)`` for the two columns ``(dx, dt)``."""
        # ∂g/∂x along each column by a central difference of the gradient,
        # both columns' points in one transform.
        size = np.maximum(np.linalg.norm(dx, axis=-1), 1.0e-300)
        eps = (scale / size)[:, :, None]
        shifted = np.concatenate([x[None] + eps * dx, x[None] - eps * dx], axis=0)
        g_shifted = slowness.gradient_log(shifted.reshape(-1, x.shape[1])).reshape(
            shifted.shape
        )
        dg = (g_shifted[:2] - g_shifted[2:]) / (2.0 * eps)
        # The forcing: column 0 carries δm₁, whose gradient perturbs ln s.
        forcing = np.zeros_like(dg)
        forcing[0] = -chart.gradient_vectors(tangent, x, **slowness.kwargs)
        dg = dg + forcing
        tg = np.sum(t * g, axis=-1)[None, :, None]
        da = (
            dg
            - np.sum(dt * g[None], axis=-1)[:, :, None] * t[None]
            - np.sum(t[None] * dg, axis=-1)[:, :, None] * t[None]
            - tg * dt
        )
        if chart.dimension == 3:
            da = da - dx / chart.radius
        dv = np.stack([chart.velocity(dt[c]) for c in range(2)])
        return dv, da

    def linearized_projection(
        x: np.ndarray, t: np.ndarray, dx: Any, dt: Any
    ) -> tuple[Any, Any]:
        """The responses after the geometry's projection of a stage point.

        Renormalizing a point onto the sphere and a tangent onto its unit
        length are what :meth:`_Chart.project` does; their linearizations
        drop the components along the point and along the tangent.
        """
        if dx is None:
            return dx, dt
        if chart.dimension == 3:
            dx = dx - np.sum(dx * x[None], axis=-1, keepdims=True) * x[None]
            dt = dt - np.sum(dt * x[None], axis=-1, keepdims=True) * x[None]
        dt = dt - np.sum(dt * t[None], axis=-1, keepdims=True) * t[None]
        return dx, dt

    def stage(x: np.ndarray, t: np.ndarray, dx: Any, dt: Any) -> tuple:
        v, a, s_at, g = rates(x, t)
        if not columns:
            return v, a, s_at, None, None
        dx, dt = linearized_projection(x, t, dx, dt)
        dv, da = tangent_rates(x, t, g, dx, dt)
        return v, a, s_at, dv, da

    for _ in range(int(1.5 * steps) + 5):
        active = ~arrived
        if not np.any(active):
            break
        h = dl[:, None]
        k1 = stage(x, t, dx if columns else None, dt if columns else None)
        x2, t2 = chart.project(x + 0.5 * h * k1[0], t + 0.5 * h * k1[1])
        k2 = stage(
            x2,
            t2,
            *(
                (dx + 0.5 * h * k1[3], dt + 0.5 * h * k1[4])
                if columns
                else (None, None)
            ),
        )
        x3, t3 = chart.project(x + 0.5 * h * k2[0], t + 0.5 * h * k2[1])
        k3 = stage(
            x3,
            t3,
            *(
                (dx + 0.5 * h * k2[3], dt + 0.5 * h * k2[4])
                if columns
                else (None, None)
            ),
        )
        x4, t4 = chart.project(x + h * k3[0], t + h * k3[1])
        k4 = stage(
            x4, t4, *((dx + h * k3[3], dt + h * k3[4]) if columns else (None, None))
        )
        x_new = x + h / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        t_new = t + h / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        x_new, t_new = chart.project(x_new, t_new)
        time_new = time + dl / 6.0 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
        along, across = chart.along_across(start, end, x_new)
        crossing = active & (along >= along_end)
        fraction = np.zeros(P)
        if np.any(crossing):
            fraction = np.where(
                crossing,
                (along_end - prev_along)
                / np.where(along - prev_along != 0.0, along - prev_along, 1.0),
                0.0,
            )
            miss = np.where(
                crossing, prev_across + fraction * (across - prev_across), miss
            )
            times_at = np.where(
                crossing, prev_time + fraction * (time_new - prev_time), times_at
            )
            # The node about to be recorded lies beyond the receiver; it and
            # every later one are replaced by the receiver itself below.
            cross_index = np.where(crossing, len(history_x), cross_index)
        if columns:
            dx_new = dx + h[None] / 6.0 * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])
            dt_new = dt + h[None] / 6.0 * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4])
            if chart.dimension == 3:
                # Keep the responses tangent to the sphere at the new point.
                dx_new = (
                    dx_new
                    - np.sum(dx_new * x_new[None], axis=-1, keepdims=True) * x_new[None]
                )
                dt_new = (
                    dt_new
                    - np.sum(dt_new * x_new[None], axis=-1, keepdims=True) * x_new[None]
                )
            # The tangent stays a unit vector, so its response stays across it.
            dt_new = (
                dt_new
                - np.sum(dt_new * t_new[None], axis=-1, keepdims=True) * t_new[None]
            )
            dacross = np.sum(dx_new * across_unit[None], axis=-1)
            if chart.dimension == 3:
                dacross = chart.radius * dacross
                # along == R atan2(x · d, x · s): its linearization at x_new.
                u = np.sum(x_new * along_unit, axis=-1)[None]
                w = np.sum(x_new * start, axis=-1)[None]
                dalong = (
                    chart.radius
                    * (
                        np.sum(dx_new * along_unit[None], axis=-1) * w
                        - u * np.sum(dx_new * start[None], axis=-1)
                    )
                    / (u**2 + w**2)
                )
            else:
                dalong = np.sum(dx_new * along_unit[None], axis=-1)
            if np.any(crossing):
                # The miss is the across coordinate where the along one
                # reaches the receiver's, so its change carries the across
                # slope times the shift of that crossing point.
                slope = np.where(
                    along - prev_along != 0.0,
                    (across - prev_across)
                    / np.where(along - prev_along != 0.0, along - prev_along, 1.0),
                    0.0,
                )
                dacross_at = prev_dacross + fraction[None] * (dacross - prev_dacross)
                dalong_at = prev_dalong + fraction[None] * (dalong - prev_dalong)
                dmiss = np.where(
                    crossing[None], dacross_at - slope[None] * dalong_at, dmiss
                )
            keep = active[None, :, None]
            dx = np.where(keep, dx_new, dx)
            dt = np.where(keep, dt_new, dt)
            history_dx.append(dx.copy())
            prev_dacross = dacross
            prev_dalong = dalong
        arrived = arrived | crossing
        keep = active[:, None]
        x = np.where(keep, x_new, x)
        t = np.where(keep, t_new, t)
        time = np.where(active, time_new, time)
        history_x.append(x.copy())
        prev_along, prev_across, prev_time = along, across, time_new
    if not np.all(arrived):
        raise EvaluationError(
            f"{int(np.sum(~arrived))} of {P} rays did not reach their receivers' "
            "distance; the field is too strong for shooting, or the step too long."
        )
    nodes = np.stack(history_x, axis=1)
    beyond = np.arange(nodes.shape[1])[None, :] >= cross_index[:, None]
    nodes = np.where(beyond[:, :, None], end[:, None, :], nodes)
    responses = None
    if columns:
        histories = np.stack(history_dx, axis=2)  # (2, P, n, k)
        histories = np.where(beyond[None, :, :, None], 0.0, histories)
        responses = (histories, dmiss)
    return miss, times_at, nodes, responses


def _shoot(
    chart: _Chart,
    slowness: _Slowness,
    start: np.ndarray,
    end: np.ndarray,
    *,
    steps: int,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The take-off angles at which the rays arrive: ``(angle, time, nodes)``.

    A safeguarded secant iteration on the miss as a function of the angle,
    every path in lockstep: a bracket is grown about the geodesic's angle
    until the miss changes sign, then shrunk by regula falsi with the
    Illinois modification, which converges superlinearly on a smooth miss.
    """
    P = start.shape[0]
    lengths = chart.length(start, end)
    lo = np.zeros(P)
    f_lo, _, _, _ = _trace(chart, slowness, start, end, lo, steps)
    hi = np.where(f_lo > 0.0, -0.05, 0.05)
    f_hi, _, _, _ = _trace(chart, slowness, start, end, hi, steps)
    for _ in range(12):
        bracketed = np.sign(f_lo) != np.sign(f_hi)
        if np.all(bracketed):
            break
        lo = np.where(bracketed, lo, hi)
        f_lo = np.where(bracketed, f_lo, f_hi)
        hi = np.where(bracketed, hi, hi * 1.8)
        f_hi, _, _, _ = _trace(chart, slowness, start, end, hi, steps)
    if not np.all(np.sign(f_lo) != np.sign(f_hi)):
        raise EvaluationError(
            "Shooting could not bracket the arrival angle of every ray; the "
            "field is too strong for the rays to be followed from the geodesics."
        )
    side = np.zeros(P)
    for _ in range(max_iterations):
        mid = hi - f_hi * (hi - lo) / (f_hi - f_lo)
        f_mid, time, nodes, _ = _trace(chart, slowness, start, end, mid, steps)
        if np.all(np.abs(f_mid) <= tolerance * lengths):
            return mid, time, nodes
        same_as_hi = np.sign(f_mid) == np.sign(f_hi)
        # Illinois: halve the retained end's value when it is kept twice.
        f_lo = np.where(same_as_hi, f_lo, np.where(side == -1, 0.5 * f_lo, f_lo))
        f_hi = np.where(same_as_hi, np.where(side == 1, 0.5 * f_hi, f_hi), f_hi)
        side = np.where(same_as_hi, 1.0, -1.0)
        lo = np.where(same_as_hi, lo, mid)
        f_lo = np.where(same_as_hi, f_lo, f_mid)
        hi = np.where(same_as_hi, mid, hi)
        f_hi = np.where(same_as_hi, f_mid, f_hi)
    raise EvaluationError(
        f"Shooting did not converge in {max_iterations} iterations; the largest "
        f"miss left is {float(np.max(np.abs(f_mid) / lengths)):.2e} of a path's length."
    )


# --------------------------------------------------------------------- #
#                              The operator                              #
# --------------------------------------------------------------------- #


class _Rays:
    """One traced state: nodes and quadrature along every ray, for the value and the derivative."""

    def __init__(
        self,
        points: np.ndarray,
        weights: np.ndarray,
        slowness_at: np.ndarray,
        *,
        times: np.ndarray | None = None,
        bending: tuple[Any, ...] | None = None,
        shooting: tuple[Any, ...] | None = None,
    ) -> None:
        self.points = points  # (P, n, k) embedded quadrature points
        self.weights = weights  # (P, n) arc-length weights
        self.slowness = slowness_at  # (P, n)
        self._times = times
        # (polylines, offsets, diagonal, off, h): the stationary point and
        # the Hessian's bands there, for the second derivative.
        self.bending = bending
        # (angle, steps): the arrival angles and the step count of the traced
        # rays, for the tangent linear system.
        self.shooting = shooting

    @property
    def times(self) -> np.ndarray:
        """The travel times: the quadrature's, unless the trace integrated them more accurately."""
        if self._times is not None:
            return self._times
        return np.sum(self.weights * self.slowness, axis=1)


def _trace_all(
    chart: _Chart,
    slowness: _Slowness,
    start: np.ndarray,
    end: np.ndarray,
    *,
    method: Method,
    interior: int,
    nodes_per_segment: int,
    tolerance: float,
    max_iterations: int,
) -> _Rays:
    if method == "bending" or chart.dimension == 1 or start.shape[1] == 1:
        polylines = _Polylines(chart, start, end, interior, nodes_per_segment)
        if chart.dimension == 1 or start.shape[1] == 1:
            # No direction to bend in: the ray is the interval.
            offsets = np.zeros(polylines.base.shape[:2])
            bending = None
        else:
            offsets, (diagonal, off, h) = _bend(
                polylines, slowness, tolerance=tolerance, max_iterations=max_iterations
            )
            bending = (polylines, offsets, diagonal, off, h)
        points, weights = polylines.rule(polylines.nodes(offsets))
        P = points.shape[0]
        points = points.reshape(P, -1, points.shape[-1])
        weights = weights.reshape(P, -1)
        return _Rays(points, weights, slowness.values(points), bending=bending)
    if method != "shooting":
        raise ValueError(f"The method is 'bending' or 'shooting', got {method!r}.")
    steps = (interior + 1) * nodes_per_segment
    angle, times, nodes = _shoot(
        chart,
        slowness,
        start,
        end,
        steps=steps,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    trapezoid = _trapezoid(chart, nodes)
    # The value is the integrated time, fourth order in the step; the
    # trapezoid weights along the ray serve the derivative, second order.
    return _Rays(
        nodes,
        trapezoid,
        slowness.values(nodes),
        times=times,
        shooting=(angle, steps),
    )


def _shooting_response(
    chart: _Chart,
    slowness: _Slowness,
    start: np.ndarray,
    end: np.ndarray,
    rays: "_Rays",
    direction: Any,
    integral_operator: Any,
) -> LinearOperator:
    """The change of a shot ray's derivative operator along its response to ``δm₁``.

    The tangent linear system is integrated along the ray for the response
    to ``δm₁`` at a fixed take-off angle and for the response to a unit
    change of the angle; the angle's change that keeps the receiver
    combines them. The derivative operator is then differenced along that
    response, the trapezoid weights following the moved nodes.
    """
    angle, steps = rays.shooting
    _, _, _, responses = _trace(
        chart, slowness, start, end, angle, steps, tangent=direction
    )
    histories, dmiss = responses
    sensitivity = np.where(np.abs(dmiss[1]) > 1.0e-300, dmiss[1], 1.0e-300)
    dangle = -dmiss[0] / sensitivity
    response = histories[0] + dangle[:, None, None] * histories[1]
    size = float(np.max(np.abs(response)))
    if size == 0.0:
        return (
            LinearOperator.zero_map(chart.space, EuclideanSpace(start.shape[0]))
            if hasattr(LinearOperator, "zero_map")
            else integral_operator(rays.points, np.zeros(rays.weights.shape))
        )
    lengths = chart.length(start, end)
    epsilon = 1.0e-5 * float(np.min(lengths)) / size

    def derivative_at(nodes: np.ndarray) -> LinearOperator:
        if chart.dimension == 3:
            nodes = nodes / np.linalg.norm(nodes, axis=-1, keepdims=True)
        weights = _trapezoid(chart, nodes)
        return integral_operator(nodes, -weights * slowness.values(nodes))

    return (
        derivative_at(rays.points + epsilon * response)
        - derivative_at(rays.points - epsilon * response)
    ) * (1.0 / (2.0 * epsilon))


def _trapezoid(chart: _Chart, nodes: np.ndarray) -> np.ndarray:
    """Trapezoid weights along a polyline: each segment's length, by the chart, split between its ends."""
    P, n, k = nodes.shape
    flat_a = nodes[:, :-1, :].reshape(-1, k)
    flat_b = nodes[:, 1:, :].reshape(-1, k)
    segments = chart.length(flat_a, flat_b).reshape(P, n - 1)
    weights = np.zeros((P, n))
    weights[:, :-1] += 0.5 * segments
    weights[:, 1:] += 0.5 * segments
    return weights


def ray_travel_time_operator(
    space: Any,
    paths: Sequence[tuple[Any, Any]],
    reference_velocity: float,
    /,
    *,
    count: int | None = None,
    nodes_per_segment: int = 4,
    method: Method = "bending",
    tolerance: float = 1.0e-8,
    max_iterations: int = 50,
    eps: float | None = None,
    nthreads: int | None = None,
    unsafe: bool = False,
) -> Operator:
    """Travel times through ``c₀ exp(m)`` along the rays traced in it.

    What :meth:`~pygeoinf2.symmetric_space.base.SymmetricSpace.ray_travel_time_operator`
    builds; see there for the account.

    Args:
        space: the symmetric space the model lives on.
        paths: ``(source, receiver)`` pairs.
        reference_velocity: ``c₀``.
        count: nodes along each ray; from the length scale if omitted.
        nodes_per_segment: Gauss-Legendre nodes between consecutive ray
            nodes.
        method: ``"bending"`` or ``"shooting"``.
        tolerance: on the ray's position, relative to the path's length.
        max_iterations: the budget of Newton steps or root-find iterations.
        eps: accuracy for the non-uniform transforms, where used.
        nthreads: threads for them.
        unsafe: build it even on a space too rough for its derivative, a
            path integral, to be bounded.

    Returns:
        The operator, from the space to ``EuclideanSpace(len(paths))``.

    Raises:
        ValueError: if no paths are given, the method is unknown, or the
            order is at or below ``(d - 1) / 2`` and *unsafe* is not set.
        NotImplementedError: on a geometry without a ray tracer.
    """
    space._require_path_evaluation("A ray travel-time operator", unsafe=unsafe)
    chart = _chart(space)
    paths = tuple(paths)
    if not paths:
        raise ValueError("At least one path is needed.")
    start = chart.embed([p[0] for p in paths])
    end = chart.separation_end(start, chart.embed([p[1] for p in paths]))
    lengths = chart.length(start, end)
    if count is None:
        count = int(max(2, np.ceil(2.0 * float(np.max(lengths)) / space.length_scale)))
    interior = max(count - 1, 1)
    kwargs = {k: v for k, v in (("eps", eps), ("nthreads", nthreads)) if v is not None}
    data_space = EuclideanSpace(len(paths))

    def traced(model: Any, components: np.ndarray | None = None) -> _Rays:
        if components is None:
            components = space.to_components(model)
        if not np.all(np.isfinite(components)):
            raise EvaluationError(
                "The model has non-finite values; the rays cannot be traced "
                "through it."
            )
        return _trace_all(
            chart,
            _Slowness(chart, model, reference_velocity, **kwargs),
            start,
            end,
            method=method,
            interior=interior,
            nodes_per_segment=nodes_per_segment,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    def integral_operator(points: np.ndarray, values: np.ndarray) -> LinearOperator:
        """``δm -> Σ values δm(points)`` per path: a weighted path integral."""
        P, n = values.shape
        nodes = chart.evaluation_points(points.reshape(-1, points.shape[-1]))
        rows = np.repeat(np.arange(P), n)
        columns = np.arange(P * n)
        sparse = _weight_matrix(P, P * n, rows, columns, values.reshape(-1))
        return _weight_operator(sparse) @ space.point_evaluation_operator(
            nodes, unsafe=True, **kwargs
        )

    def derivative_of(rays: _Rays) -> LinearOperator:
        return integral_operator(rays.points, -rays.weights * rays.slowness)

    memo: dict[str, Any] = {}

    def traced_memo(model: Any) -> _Rays:
        # The second derivative at a point follows a linearization there;
        # one trace serves both while the model's values are the same. The
        # key is the components, not the object: a vector updated in place
        # is the same object with other values.
        components = np.array(space.to_components(model), dtype=float)
        key = memo.get("components")
        if key is None or not np.array_equal(key, components):
            memo["rays"] = traced(model, components)
            memo["components"] = components
        return memo["rays"]

    def second_derivative(model: Any, direction: Any) -> LinearOperator:
        """``F''(m)[δm₁, ·]`` as an operator in ``δm₂``: the reduced Hessian of the discrete travel time."""
        rays = traced_memo(model)
        slowness = _Slowness(chart, model, reference_velocity, **kwargs)
        flat = rays.points.reshape(-1, rays.points.shape[-1])
        along = space.evaluate(
            direction, chart.evaluation_points(flat), **kwargs
        ).reshape(rays.weights.shape)
        # ∂²s/∂m² == s: the explicit term, a path integral of s δm₁ δm₂.
        explicit = integral_operator(rays.points, rays.weights * rays.slowness * along)
        if rays.shooting is not None:
            # The response is the ray's own change, so its term adds.
            return explicit + _shooting_response(
                chart, slowness, start, end, rays, direction, integral_operator
            )
        if rays.bending is None:
            return explicit
        polylines, offsets, diagonal, off, h = rays.bending
        if offsets.shape[1] == 0:
            return explicit

        def mixed_terms(u: np.ndarray) -> np.ndarray:
            # ∂T_seg/∂m [δm₁] at offsets u: the segment integrals of −s δm₁.
            points, weights = polylines.rule(polylines.nodes(u))
            flat = points.reshape(-1, points.shape[-1])
            s_values = slowness.values(points)
            d_values = space.evaluate(
                direction, chart.evaluation_points(flat), **kwargs
            )
            return np.sum(
                weights * (-s_values) * d_values.reshape(points.shape[:-1]), axis=-1
            )

        bands = _Bands(offsets.shape[1])
        current = mixed_terms(offsets)
        t_um, _, _ = bands.gradient(mixed_terms, offsets, h, current, hessian=False)
        response = _solve_banded(diagonal, off, t_um, None)
        response[~np.isfinite(response)] = 0.0
        # The ray moves by −T_uu⁻¹ T_um δm₁; the derivative operator's change
        # along that move, by a central difference in the offsets.
        size = float(np.max(np.abs(response)))
        if size == 0.0:
            return explicit
        epsilon = 1.0e-3 * float(np.min(h)) / size

        def derivative_at(u: np.ndarray) -> LinearOperator:
            points, weights = polylines.rule(polylines.nodes(u))
            P = points.shape[0]
            points = points.reshape(P, -1, points.shape[-1])
            weights = weights.reshape(P, -1)
            return integral_operator(points, -weights * slowness.values(points))

        moved = (
            derivative_at(offsets + epsilon * response)
            - derivative_at(offsets - epsilon * response)
        ) * (1.0 / (2.0 * epsilon))
        return explicit - moved

    def value(model: Any) -> np.ndarray:
        return traced(model).times

    def linearize(model: Any) -> Any:
        from ..algebra.linearization import Linearization

        rays = traced_memo(model)
        return Linearization(
            point=model, value=rays.times, derivative=derivative_of(rays)
        )

    operator = Operator.from_callables(
        space,
        data_space,
        value,
        linearize=linearize,
        second_derivative=second_derivative,
    )
    operator.trace_rays = lambda model: [  # type: ignore[attr-defined]
        chart.unembed(points) for points in traced(model).points
    ]
    return operator
