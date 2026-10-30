"""Drawing a convex set, or a slice of one, in one, two or three dimensions.

One function, :func:`plot_set`, with the route chosen by what the set can do
rather than by its class. A set with a support function is drawn from it; a
set with a level function is drawn as a filled contour of it; anything else
is rastered through membership. Polytopes and ellipsoids need no special
case: each offers a support function or a level function, and is drawn
through it.

A property space is where this earns its keep: the Backus–Gilbert–Parker
estimator returns a convex set there, and property spaces are one- or
two-dimensional in nearly every use. A higher-dimensional set is drawn
either sliced along an affine subspace, with ``subspace=``, or projected,
by ``push_forward`` onto a coordinate pair before calling this.

Three dimensions are drawn by the same routes, with the renderings they
force. The support route takes the convex hull of the points *attaining* the
support over a spread of directions, which is **inscribed** in the set,
where the two-dimensional polygon of supporting lines is circumscribed: a
three-dimensional support picture errs inside the set and a two-dimensional
one outside it. The sampled routes raster a ``resolution**3`` grid and draw
it as voxels; that cubes the cost of an oracle call, so the default
resolution drops and a warning names the count when it is pushed past what a
Python oracle answers quickly. A three-dimensional picture can also go to
plotly, with ``engine="plotly"``, to be turned by hand.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np

__all__ = ["plot_set"]

Route = Literal["auto", "support", "level", "membership"]
Engine = Literal["matplotlib", "plotly"]


def plot_set(
    subset: Any,
    /,
    *,
    subspace: Any = None,
    ax: Any = None,
    bounds: Any = None,
    resolution: int | None = None,
    directions: int = 180,
    route: Route = "auto",
    engine: Engine = "matplotlib",
    padding: float = 0.05,
    n_jobs: int | None = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Draw a convex set in one, two or three dimensions.

    Args:
        subset: the set. Its domain must be a coordinate space of dimension
            one, two or three, unless *subspace* is given.
        subspace: an :class:`~pygeoinf.geometry.subspaces.AffineSubspace`
            of the set's domain, of dimension one to three, to slice along.
            The picture is then in the subspace's own coordinates, an
            orthonormal basis of its tangent about its translation. A slice
            has no support function of its own, so the support route is
            not available for one.
        ax: the axes to draw on. A new figure by default; in three
            dimensions the axes must carry the ``3d`` projection, and the
            new figure's do.
        bounds: ``(low, high)`` in one dimension, or one pair per axis in
            two or three, for the sampled routes. Derived from the set's
            support function when it has one, and required otherwise.
        resolution: samples per axis on the sampled routes. Two hundred by
            default, but thirty in three dimensions, where the cost is its
            cube; pushing it past thirty there warns with the call count.
        directions: supporting lines on the support route in two
            dimensions, or support directions over the sphere in three.
        route: ``"support"``, ``"level"``, ``"membership"``, or ``"auto"``,
            which takes the first of those the set offers. In three
            dimensions the support route also needs the set's
            ``support_maximizer``, since it draws the points attaining the
            support rather than the planes bounding it.
        engine: ``"matplotlib"``, or ``"plotly"`` for a three-dimensional
            picture that can be turned by hand. plotly is the ``plot3d``
            extra, imported only when asked for.
        padding: the fraction by which derived bounds are widened.
        n_jobs: workers for the support route's directions, each a support
            value or a supporting point, which on a feasible property set
            is an optimization; a set with a ``support_sweep`` takes them
            through it, so that neighboring directions warm-start each
            other and the results are kept. Serial by default.
        **kwargs: passed to the matplotlib or plotly call that draws the
            set. ``color`` is honored on every route, given as ``colors``
            where the set is drawn by filled contours.

    Returns:
        The ``(axes, artist)`` pair. In three dimensions the artist is a
        ``Poly3DCollection`` on the support route and the dict that
        ``Axes3D.voxels`` returns on the sampled ones — cell index to face
        collection, for the filled cells with a visible face; cells buried
        inside the set draw nothing and are absent. With ``engine="plotly"`` the pair is
        ``(None, figure)``: no matplotlib axes exist, and the figure is a
        ``plotly.graph_objects.Figure`` to be shown or written out.

    Raises:
        ValueError: for a dimension other than one, two or three, a route
            the set does not offer, a sampled route without bounds to
            sample in, or the plotly engine outside three dimensions or
            with an ``ax``.
        TypeError: for a domain without components.
        ImportError: for ``engine="plotly"`` without plotly installed.
    """
    picture = _Picture(subset, subspace)
    chosen = _choose(subset, picture, route)
    if engine not in ("matplotlib", "plotly"):
        raise ValueError(f"The engine is 'matplotlib' or 'plotly', got {engine!r}.")
    if engine == "plotly":
        if picture.dimension != 3:
            raise ValueError(
                "The plotly engine draws three-dimensional sets; matplotlib "
                "draws one and two."
            )
        if ax is not None:
            raise ValueError("The plotly engine makes its own figure; pass no ax.")
    if resolution is None:
        resolution = 30 if picture.dimension == 3 else 200
    kwargs.setdefault("alpha", 0.5)

    if picture.dimension == 3:
        return _plot_3d(
            subset,
            picture,
            chosen,
            ax,
            bounds,
            resolution,
            directions,
            engine,
            padding,
            n_jobs,
            **kwargs,
        )

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    if chosen == "support":
        artist = (
            _support_1d(ax, subset, picture, **kwargs)
            if picture.dimension == 1
            else _support_2d(ax, subset, picture, directions, n_jobs, **kwargs)
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
        if self.dimension not in (1, 2, 3):
            raise ValueError(
                f"A set is drawn in one, two or three dimensions, not "
                f"{self.dimension}."
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
        "support": subset.has_support_function
        and not picture.sliced
        # In three dimensions the support route draws the points attaining
        # the support, so it needs the maximizer; a support value alone does
        # not determine the point.
        and (picture.dimension < 3 or subset.has_maximizer),
        "level": subset.has_level_function,
        "membership": subset.has_membership,
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
        why = ""
        if route == "support":
            if picture.sliced:
                why = " on a slice"
            elif picture.dimension == 3:
                why = " in three dimensions without a support maximizer"
        raise ValueError(f"The set has no {route} route{why}.")
    return route


def _bounds(subset: Any, picture: _Picture, given: Any, padding: float) -> np.ndarray:
    """``(dimension, 2)`` bounds, given or derived from the support function."""
    if given is not None:
        box = np.asarray(given, dtype=float).reshape(picture.dimension, 2)
        if np.any(box[:, 0] >= box[:, 1]):
            raise ValueError("Each bound must have low < high.")
        return box
    if not subset.has_support_function:
        raise ValueError(
            "Bounds are needed to sample a set without a support function: "
            "pass bounds=(low, high) in one dimension, or one (low, high) "
            "pair per axis."
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


def _support_values(
    subset: Any, picture: _Picture, normals: np.ndarray, n_jobs: int | None
) -> np.ndarray:
    """The support in each picture direction: through the set's own sweep
    when it has one, otherwise one support-function value per direction."""
    from ..parallel import parallel_map

    vectors = [picture.direction(normal) for normal in normals]
    if hasattr(subset, "support_values"):
        return np.asarray(subset.support_values(vectors, n_jobs=n_jobs), dtype=float)
    support = subset.support_function()
    return np.array(
        parallel_map(lambda q: float(support(q)), vectors, n_jobs=n_jobs), dtype=float
    )


def _support_2d(
    ax: Any,
    subset: Any,
    picture: _Picture,
    directions: int,
    n_jobs: int | None,
    **kwargs: Any,
) -> Any:
    """The polygon of supporting lines: consecutive lines meet at its vertices.

    Exact for a polytope, whose vertices are where the normal cones change,
    and circumscribed for a smooth set, closer with more directions.
    """
    if directions < 3:
        raise ValueError(f"At least three directions are needed, got {directions}.")
    angles = np.linspace(0.0, 2.0 * np.pi, directions, endpoint=False)
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    values = _support_values(subset, picture, normals, n_jobs)
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
    if "color" in kwargs:
        # contourf colors its levels by colors=, and ignores color= with a
        # warning, filling from the color map instead.
        kwargs["colors"] = [kwargs.pop("color")]
    if route == "level":
        level = float(subset.level)
        floor = min(float(values.min()), level) - 1.0
        return ax.contourf(grid_x, grid_y, values, levels=[floor, level], **kwargs)
    return ax.contourf(grid_x, grid_y, values, levels=[0.5, 1.5], **kwargs)


# --------------------------------------------------------------------------- #
#                              Three dimensions                               #
# --------------------------------------------------------------------------- #


def _plot_3d(
    subset: Any,
    picture: _Picture,
    route: str,
    ax: Any,
    bounds: Any,
    resolution: int,
    directions: int,
    engine: str,
    padding: float,
    n_jobs: int | None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    if route == "support":
        points, simplices = _support_hull_3d(subset, picture, directions, n_jobs=n_jobs)
        if engine == "plotly":
            return None, _plotly_mesh(points, simplices, **kwargs)
        axes = _axes_3d(ax)
        return axes, _hull_3d(axes, points, simplices, **kwargs)
    box = _bounds(subset, picture, bounds, padding)
    mask = _mask_3d(subset, picture, route, box, resolution)
    if engine == "plotly":
        return None, _plotly_volume(box, mask, resolution, **kwargs)
    axes = _axes_3d(ax)
    return axes, _voxels_3d(axes, box, mask, resolution, **kwargs)


def _axes_3d(ax: Any) -> Any:
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "3d"})
        return ax
    if getattr(ax, "name", None) != "3d":
        raise ValueError(
            "A three-dimensional set needs three-dimensional axes: make them "
            'with subplot_kw={"projection": "3d"}, or pass none and get them.'
        )
    return ax


def _fibonacci_sphere(count: int) -> np.ndarray:
    """``count`` nearly equal-area directions: the Fibonacci lattice."""
    indices = np.arange(count) + 0.5
    polar = np.arccos(1.0 - 2.0 * indices / count)
    azimuth = np.pi * (1.0 + np.sqrt(5.0)) * indices
    return np.column_stack(
        [
            np.cos(azimuth) * np.sin(polar),
            np.sin(azimuth) * np.sin(polar),
            np.cos(polar),
        ]
    )


def _support_hull_3d(
    subset: Any, picture: _Picture, directions: int, *, n_jobs: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """The convex hull of the points attaining the support, as vertices and triangles.

    **Inscribed**, where the two-dimensional polygon of supporting lines is
    circumscribed: each vertex is a point *of* the set, so their hull is
    contained in it, approaching a smooth set from inside as directions
    grow. Exact for a polytope once every vertex's normal cone contains a
    sampled direction. Intersecting the supporting planes instead would
    bound the set from outside, but in three dimensions it needs an
    interior point and a half-space intersection; the hull of points the
    set exhibits is the honest cheap one.
    """
    from scipy.spatial import ConvexHull, QhullError

    if directions < 4:
        raise ValueError(f"At least four directions are needed, got {directions}.")
    from ..parallel import parallel_map

    space = picture.space
    vectors = [picture.direction(normal) for normal in _fibonacci_sphere(directions)]
    if hasattr(subset, "support_sweep"):
        # Solved in a sweep, which keeps its results: the maximizers below
        # are then reads.
        subset.support_sweep(vectors, n_jobs=n_jobs)
        maximizers = [subset.support_maximizer(q) for q in vectors]
    else:
        maximizers = parallel_map(subset.support_maximizer, vectors, n_jobs=n_jobs)
    points = np.array([space.to_components(m) for m in maximizers], dtype=float)
    try:
        hull = ConvexHull(points)
    except QhullError as error:
        raise ValueError(
            "The supporting points span no volume, so there is no solid hull "
            "to draw; slice a flat set along its own plane instead."
        ) from error
    return points, hull.simplices


def _hull_3d(ax: Any, points: np.ndarray, simplices: np.ndarray, **kwargs: Any) -> Any:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    artist = Poly3DCollection(points[simplices], **kwargs)
    ax.add_collection3d(artist)
    # A collection does not feed the autoscaler; open the frame by hand.
    ax.auto_scale_xyz(points[:, 0], points[:, 1], points[:, 2])
    return artist


def _mask_3d(
    subset: Any, picture: _Picture, route: str, box: np.ndarray, resolution: int
) -> np.ndarray:
    if resolution > 30:
        warnings.warn(
            f"Rastering {resolution ** 3:,} points through a Python {route} "
            "oracle; a resolution of thirty or less keeps a three-dimensional "
            "picture quick.",
            stacklevel=2,
        )
    inside = _inside(subset, picture, route)
    axes = [np.linspace(low, high, resolution) for low, high in box]
    values = np.empty((resolution,) * 3)
    for i, x in enumerate(axes[0]):
        for j, y in enumerate(axes[1]):
            for k, z in enumerate(axes[2]):
                values[i, j, k] = inside(np.array([x, y, z]))
    return values <= float(subset.level) if route == "level" else values > 0.5


def _voxels_3d(
    ax: Any, box: np.ndarray, mask: np.ndarray, resolution: int, **kwargs: Any
) -> Any:
    # Corner grids one larger than the mask put the voxels at parameter
    # coordinates rather than index space.
    edges = [np.linspace(low, high, resolution + 1) for low, high in box]
    corners = np.meshgrid(*edges, indexing="ij")
    return ax.voxels(*corners, mask, **kwargs)


def _import_plotly() -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise ImportError(
            "engine='plotly' needs plotly, which the plot3d extra installs: "
            "pip install pygeoinf[plot3d]"
        ) from error
    return go


def _plotly_mesh(points: np.ndarray, simplices: np.ndarray, **kwargs: Any) -> Any:
    go = _import_plotly()
    opacity = kwargs.pop("alpha", 0.5)
    figure = go.Figure(
        data=[
            go.Mesh3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                i=simplices[:, 0],
                j=simplices[:, 1],
                k=simplices[:, 2],
                opacity=opacity,
                **kwargs,
            )
        ]
    )
    figure.update_layout(scene={"aspectmode": "data"})
    return figure


def _plotly_volume(
    box: np.ndarray, mask: np.ndarray, resolution: int, **kwargs: Any
) -> Any:
    go = _import_plotly()
    opacity = kwargs.pop("alpha", 0.5)
    axes = [np.linspace(low, high, resolution) for low, high in box]
    grid = np.meshgrid(*axes, indexing="ij")
    figure = go.Figure(
        data=[
            go.Volume(
                x=grid[0].ravel(),
                y=grid[1].ravel(),
                z=grid[2].ravel(),
                value=mask.astype(float).ravel(),
                # The inside is one and the outside zero, so the half level
                # is the boundary whatever the route computed.
                isomin=0.5,
                isomax=1.0,
                surface_count=1,
                opacity=opacity,
                showscale=False,
                **kwargs,
            )
        ]
    )
    figure.update_layout(scene={"aspectmode": "data"})
    return figure
