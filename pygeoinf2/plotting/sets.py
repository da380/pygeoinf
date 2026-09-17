"""Drawing a convex set, or a slice of one, in one or two dimensions.

The compact port of v1's ``SubspaceSlicePlotter`` (DESIGN §80): one
function, matplotlib only, one and two dimensions, and the route chosen by
what the set can do rather than by its class. A set with a support
function is drawn from it, as the polygon of its supporting lines; a set
with a level function is drawn as a filled contour of it; anything else is
rastered through membership. The exact polyhedral and quadratic paths v1
special-cased fall out of the first two routes.

A property space is where this earns its keep: the Backus–Gilbert–Parker
estimator returns a convex set there, and property spaces are one- or
two-dimensional in nearly every use. A higher-dimensional set is drawn
either sliced along an affine subspace, with ``subspace=``, or projected,
by ``push_forward`` onto a coordinate pair before calling this.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

__all__ = ["plot_set"]

Route = Literal["auto", "support", "level", "membership"]


def plot_set(
    subset: Any,
    /,
    *,
    subspace: Any = None,
    ax: Any = None,
    bounds: Any = None,
    resolution: int = 200,
    directions: int = 180,
    route: Route = "auto",
    padding: float = 0.05,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Draw a convex set in one or two dimensions.

    Args:
        subset: the set. Its domain must be a coordinate space of dimension
            one or two, unless *subspace* is given.
        subspace: an :class:`~pygeoinf2.geometry.subspaces.AffineSubspace`
            of the set's domain, of dimension one or two, to slice along.
            The picture is then in the subspace's own coordinates, an
            orthonormal basis of its tangent about its translation. A slice
            has no support function of its own, so the support route is
            not available for one.
        ax: the axes to draw on. A new figure by default.
        bounds: ``(low, high)`` in one dimension or ``((xlow, xhigh), (ylow,
            yhigh))`` in two, for the sampled routes. Derived from the set's
            support function when it has one, and required otherwise.
        resolution: samples per axis on the sampled routes.
        directions: supporting lines on the support route in two dimensions.
        route: ``"support"``, ``"level"``, ``"membership"``, or ``"auto"``,
            which takes the first of those the set offers.
        padding: the fraction by which derived bounds are widened.
        **kwargs: passed to the matplotlib call that draws the set.

    Returns:
        The ``(axes, artist)`` pair.

    Raises:
        ValueError: for a dimension other than one or two, a route the set
            does not offer, or a sampled route without bounds to sample in.
        TypeError: for a domain without components.
    """
    import matplotlib.pyplot as plt

    picture = _Picture(subset, subspace)
    chosen = _choose(subset, picture, route)
    if ax is None:
        _, ax = plt.subplots()
    kwargs.setdefault("alpha", 0.5)

    if chosen == "support":
        artist = (
            _support_1d(ax, subset, picture, **kwargs)
            if picture.dimension == 1
            else _support_2d(ax, subset, picture, directions, **kwargs)
        )
    else:
        box = _bounds(subset, picture, bounds, padding)
        if picture.dimension == 1:
            artist = _sampled_1d(ax, subset, picture, chosen, box, resolution, **kwargs)
        else:
            artist = _sampled_2d(ax, subset, picture, chosen, box, resolution, **kwargs)
    return ax, artist


class _Picture:
    """The coordinates the picture is drawn in, and how they embed."""

    def __init__(self, subset: Any, subspace: Any) -> None:
        from ..algebra.spaces import CoordinateSpace

        self.space = subset.domain
        self.sliced = subspace is not None
        if subspace is None:
            if not isinstance(self.space, CoordinateSpace):
                raise TypeError("Drawing a set needs a domain with components.")
            self.dimension = int(self.space.dim)
            self.translation = None
            self.basis: list[Any] = []
        else:
            if subspace.domain != self.space:
                raise ValueError("The subspace must lie in the set's domain.")
            self.dimension = int(subspace.dimension())
            self.translation = subspace.translation
            self.basis = list(subspace.tangent.projector.basis())
        if self.dimension not in (1, 2):
            raise ValueError(
                f"A set is drawn in one or two dimensions, not {self.dimension}."
            )

    def embed(self, coordinates: np.ndarray) -> Any:
        """The domain point at picture coordinates."""
        if not self.sliced:
            return self.space.from_components(np.asarray(coordinates, dtype=float))
        point = self.space.copy(self.translation)
        for value, vector in zip(coordinates, self.basis):
            point = self.space.axpy(float(value), vector, point)
        return point

    def direction(self, normal: np.ndarray) -> Any:
        """The domain vector ``q`` with ``(q, x) == normal . coordinates(x)``.

        Unsliced only: on a coordinate space with Gram ``G`` that is
        ``G^-1 normal`` in components, so a supporting line with this
        normal in the picture is the support in the direction ``q``.
        """
        return self.space.from_components(self.space.solve_gram(np.asarray(normal)))

    def support_along(self, subset: Any, axis: int, sign: float) -> float:
        """How far the set reaches along one picture axis, from the support function."""
        support = subset.support_function()
        if not self.sliced:
            normal = np.zeros(self.dimension)
            normal[axis] = sign
            return float(support(self.direction(normal)))
        vector = self.space.scale(sign, self.basis[axis])
        offset = float(self.space.inner_product(self.basis[axis], self.translation))
        return float(support(vector)) - sign * offset


def _choose(subset: Any, picture: _Picture, route: str) -> str:
    offers = {
        "support": bool(getattr(subset, "has_support_function", False))
        and not picture.sliced,
        "level": bool(getattr(subset, "has_level_function", False)),
        "membership": bool(getattr(subset, "has_membership", True)),
    }
    if route == "auto":
        for name in ("support", "level", "membership"):
            if offers[name]:
                return name
        raise ValueError("The set offers no way to be drawn.")
    if route not in offers:
        raise ValueError(
            f"The route is 'support', 'level', 'membership' or 'auto', got {route!r}."
        )
    if not offers[route]:
        why = " on a slice" if route == "support" and picture.sliced else ""
        raise ValueError(f"The set has no {route} route{why}.")
    return route


def _bounds(subset: Any, picture: _Picture, given: Any, padding: float) -> np.ndarray:
    """``(dimension, 2)`` bounds, given or derived from the support function."""
    if given is not None:
        box = np.asarray(given, dtype=float).reshape(picture.dimension, 2)
        if np.any(box[:, 0] >= box[:, 1]):
            raise ValueError("Each bound must have low < high.")
        return box
    if not getattr(subset, "has_support_function", False):
        raise ValueError(
            "Bounds are needed to sample a set without a support function: "
            "pass bounds=(low, high) or ((xlow, xhigh), (ylow, yhigh))."
        )
    box = np.empty((picture.dimension, 2))
    for axis in range(picture.dimension):
        high = picture.support_along(subset, axis, 1.0)
        low = -picture.support_along(subset, axis, -1.0)
        width = max(high - low, 1e-12)
        box[axis] = (low - padding * width, high + padding * width)
    return box


def _inside(subset: Any, picture: _Picture, route: str) -> Any:
    """A function of picture coordinates: the level value, or membership."""
    if route == "level":
        functional = subset.level_function()
        return lambda c: float(functional(picture.embed(c)))
    return lambda c: 1.0 if subset.contains(picture.embed(c)) else 0.0


def _support_1d(ax: Any, subset: Any, picture: _Picture, **kwargs: Any) -> Any:
    high = picture.support_along(subset, 0, 1.0)
    low = -picture.support_along(subset, 0, -1.0)
    return ax.axvspan(low, high, **kwargs)


def _support_2d(
    ax: Any, subset: Any, picture: _Picture, directions: int, **kwargs: Any
) -> Any:
    """The polygon of supporting lines: consecutive lines meet at its vertices.

    Exact for a polytope, whose vertices are where the normal cones change,
    and circumscribed for a smooth set, closer with more directions.
    """
    if directions < 3:
        raise ValueError(f"At least three directions are needed, got {directions}.")
    support = subset.support_function()
    angles = np.linspace(0.0, 2.0 * np.pi, directions, endpoint=False)
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    values = np.array([float(support(picture.direction(n))) for n in normals])
    vertices = []
    for k in range(directions):
        a, b = normals[k], normals[(k + 1) % directions]
        matrix = np.array([a, b])
        vertices.append(
            np.linalg.solve(matrix, [values[k], values[(k + 1) % directions]])
        )
    polygon = np.array(vertices)
    (artist,) = ax.fill(polygon[:, 0], polygon[:, 1], **kwargs)
    return artist


def _sampled_1d(
    ax: Any,
    subset: Any,
    picture: _Picture,
    route: str,
    box: np.ndarray,
    resolution: int,
    **kwargs: Any,
) -> Any:
    inside = _inside(subset, picture, route)
    xs = np.linspace(box[0, 0], box[0, 1], resolution)
    values = np.array([inside(np.array([x])) for x in xs])
    mask = values <= subset.level if route == "level" else values > 0.5
    artists = []
    start = None
    for index, flag in enumerate(np.append(mask, False)):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            artists.append(ax.axvspan(xs[start], xs[index - 1], **kwargs))
            start = None
    return artists


def _sampled_2d(
    ax: Any,
    subset: Any,
    picture: _Picture,
    route: str,
    box: np.ndarray,
    resolution: int,
    **kwargs: Any,
) -> Any:
    inside = _inside(subset, picture, route)
    xs = np.linspace(box[0, 0], box[0, 1], resolution)
    ys = np.linspace(box[1, 0], box[1, 1], resolution)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    values = np.empty(grid_x.shape)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            values[i, j] = inside(np.array([grid_x[i, j], grid_y[i, j]]))
    if route == "level":
        level = float(subset.level)
        floor = min(float(values.min()), level) - 1.0
        return ax.contourf(grid_x, grid_y, values, levels=[floor, level], **kwargs)
    return ax.contourf(grid_x, grid_y, values, levels=[0.5, 1.5], **kwargs)
