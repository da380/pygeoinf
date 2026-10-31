"""Fields in a ball or an annulus: shells, sections and profiles.

A ball is three-dimensional and has to be cut to be seen, and there are three
cuts: a **shell** at one radius, drawn as a map by the sphere's own renderer;
a **section** by a plane through the centre, which meets the ball in a great
circle and is named by that circle's pole, by two points it passes through,
or -- the common case -- by a longitude; and a **profile** along one ray.
``plot`` on a ball is the shell at its surface.

No rotation is involved in a section. The field is a sum of harmonics times
radial coefficient functions, so its values on a plane are the harmonics
evaluated round the great circle times those functions along the radius: one
small matrix product, whatever the plane.

**What is drawn is the field with the vector's components**, since off the
grid that is the only field there is: a product left on the grid is drawn as
its projection onto the kept modes.

A layered space of balls is drawn as a ball is, by the shell at ``radius=``,
and ``plot_section`` and ``plot_profile`` take one too, drawing each layer's
part with a break at every jump.

**Through planetmodel.** :func:`planet_model` wraps a field as a
``planetmodel`` ``Model`` on the domain's radii, so that everything
``planetmodel.slices`` and ``planetmodel.plotting`` draw for a planet --
sphere maps, sections along any arc, profiles, cutaways -- draws the field,
by the space's own point evaluation. :func:`plot_cutaway` is the one such
picture with a function here: the body seen from outside with a wedge cut
from it, the field on the faces of the cut and on what remains of the
surface.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from ..ball import Ball
from ..spectral_element import Layered
from .base import color_limits, plot, plot_points, subplots

__all__ = [
    "planet_model",
    "plot_cutaway",
    "plot_profile",
    "plot_section",
    "plot_section_points",
    "plot_shell",
    "section_values",
]


def _coefficient_functions(space: Ball, field: np.ndarray, /, *, padding: bool) -> Any:
    """The radial coefficient function of every harmonic, shape
    ``(2, lmax + 1, lmax + 1, radii)``, and the radii they are given at."""
    values = np.asarray(field, dtype=float)
    if values.shape != space.grid_shape:
        raise ValueError(
            f"A field on this ball has shape {space.grid_shape}, got {values.shape}."
        )
    functions = space.basis.synthesise(
        space.to_components(values), physical=not padding
    )
    return functions, (space.radii if padding else space.interior_radii)


def _on_a_shell(space: Ball, field: np.ndarray, radius: float, /) -> np.ndarray:
    """The field on the angular grid at one radius of the domain."""
    components = space.to_components(np.asarray(field, dtype=float))
    lmax = space.lmax
    coefficients = np.zeros((2, lmax + 1, lmax + 1))
    for degree, radial in enumerate(space.basis):
        modes = radial.evaluate(float(radius))
        for order in range(-degree, degree + 1):
            block = space.basis.block(int(order < 0), degree, abs(order))
            coefficients[int(order < 0), degree, abs(order)] = modes @ components[block]
    from planetmodel import harmonics

    return harmonics.synthesise_grid(coefficients, space.grid)


def _map_space(space: Ball, radius: float, /) -> Any:
    """The sphere whose renderer draws a shell: the same Gauss-Legendre grid."""
    from ..symmetric_space.sphere import Lebesgue

    return Lebesgue(space.lmax, radius=max(float(radius), 1e-12), grid="GLQ")


@subplots.register
def _(
    space: Ball,
    /,
    *,
    rows: int = 1,
    columns: int = 1,
    projection: Any = None,
    **kwargs: Any,
) -> Any:
    """Axes carrying a map projection, for shells: the sphere's.

    Sections and profiles are drawn on ordinary axes, which ``plt.subplots``
    gives.

    Args:
        space: the ball whose shells will be drawn.
        rows: number of panel rows.
        columns: number of panel columns.
        projection: a cartopy projection. ``PlateCarree`` by default.
        **kwargs: passed through to ``matplotlib.pyplot.subplots``.

    Returns:
        The ``(figure, axes)`` pair.
    """
    return subplots(
        _map_space(space, space.radius),
        rows=rows,
        columns=columns,
        projection=projection,
        **kwargs,
    )


def plot_shell(
    space: Ball | Layered,
    field: Any,
    /,
    *,
    radius: float | None = None,
    side: str | None = None,
    **kwargs: Any,
) -> Any:
    """Draw a field of a ball on the sphere of one radius, as a map.

    Args:
        space: the ball or annulus, or a layered space of them.
        field: a vector of the space.
        radius: the radius of the shell, within the domain. The outer radius
            if omitted.
        side: in a layered space, which of the two fields is meant at a
            radius that is an interface: ``"below"`` or ``"above"``.
        **kwargs: those of the sphere's ``plot``: ``ax``, ``cmap``,
            ``symmetric``, ``coasts``, ``contour``, ``colorbar_label`` and
            the rest.

    Returns:
        The ``(axes, mappable)`` pair.

    Raises:
        ValueError: if the radius lies outside the domain, or, in a layered
            space, in no layer or on an interface with no side named.
        TypeError: for a layered space whose layers are not balls.
    """
    if isinstance(space, Layered):
        shells, fields = _shells(space, field)
        radius = shells[-1].radius if radius is None else float(radius)
        layer = space.layer_index((radius, 0.0, 0.0), side=side)
        space, field = shells[layer], fields[layer]
    radius = space.radius if radius is None else float(radius)
    return plot(_map_space(space, radius), _on_a_shell(space, field, radius), **kwargs)


@plot.register
def _(space: Ball, field: np.ndarray, /, **kwargs: Any) -> Any:
    """Draw a field of a ball: the shell at its surface, or at ``radius=``.
    See :func:`plot_shell`; :func:`plot_section` and :func:`plot_profile` are
    the other two views."""
    return plot_shell(space, field, **kwargs)


@plot_points.register
def _(space: Ball, points: Any, /, **kwargs: Any) -> Any:
    """Scatter points of a ball on a map, by their latitude and longitude.

    The radius is set aside: a map has nowhere to put it. Stations on the
    surface are the usual case; :func:`plot_section_points` is where a depth
    can be seen.

    Args:
        space: the ball or annulus.
        points: ``(radius, latitude, longitude)`` triples, the angles in
            degrees.
        **kwargs: those of the sphere's ``plot_points``: ``data``, ``ax``,
            ``marker``, ``size``, ``color`` and the rest.

    Returns:
        What the sphere's ``plot_points`` returns.
    """
    located = np.asarray(points, dtype=float).reshape(-1, 3)
    return plot_points(_map_space(space, space.radius), located[:, 1:], **kwargs)


def _shells(space: Ball | Layered, field: Any, /) -> tuple[list[Ball], list[Any]]:
    """The balls a field lives in and its part in each: one, or a layered
    space's layers.

    Raises:
        TypeError: for a layered space whose layers are not balls.
    """
    if isinstance(space, Layered):
        if not issubclass(space.geometry, Ball):
            raise TypeError("Only layers that are balls have shells and sections.")
        return list(space.layers), list(field)
    return [space], [field]


@plot_points.register
def _(space: Layered, points: Any, /, **kwargs: Any) -> Any:
    """Scatter points of a layered ball on a map, by latitude and longitude."""
    shells, _ = _shells(space, [None] * len(space))
    return plot_points(shells[-1], points, **kwargs)


def _unit(latitude: float, longitude: float, /) -> np.ndarray:
    """The unit vector of a direction given in degrees."""
    lat, lon = np.radians(latitude), np.radians(longitude)
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def _section_frame(
    longitude: float | None,
    pole: Sequence[float] | None,
    through: Sequence[Sequence[float]] | None,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The plane's unit normal and the two in-plane axes of the drawing.

    The plane is looked down on from its pole, and up the page is the
    direction in the plane nearest north. So a section by a meridian, whose
    pole is taken ninety degrees to its west, has north at the top and its own
    longitude on the right; one through two points runs anticlockwise from the
    first to the second; and one by the equator, which has no direction nearest
    north, is seen from above the north pole with longitude zero on the right
    and ninety at the top, east running anticlockwise.

    Raises:
        ValueError: if more than one of the three names the plane, or the two
            points given are the same or opposite. With none, the plane is
            the meridian of longitude zero.
    """
    given = sum(x is not None for x in (longitude, pole, through))
    if given > 1:
        raise ValueError("Name the plane by one of longitude, pole and through.")
    if given == 0:
        longitude = 0.0
    if longitude is not None:
        normal = _unit(0.0, float(longitude) - 90.0)
    elif pole is not None:
        normal = _unit(float(pole[0]), float(pole[1]))
    else:
        first, second = through
        normal = np.cross(_unit(*map(float, first)), _unit(*map(float, second)))
        if np.linalg.norm(normal) < 1e-12:
            raise ValueError(
                "Two points that coincide or are opposite do not fix a great "
                "circle; name its pole."
            )
        normal = normal / np.linalg.norm(normal)
    north = np.array([0.0, 0.0, 1.0])
    up = north - (north @ normal) * normal
    if np.linalg.norm(up) < 1e-9:
        up = np.array([0.0, 1.0, 0.0]) * np.sign(normal[2])
        right = np.array([1.0, 0.0, 0.0])
        return normal, right, up
    up = up / np.linalg.norm(up)
    return normal, np.cross(up, normal), up


def section_values(
    space: Ball,
    field: np.ndarray,
    /,
    *,
    longitude: float | None = None,
    pole: Sequence[float] | None = None,
    through: Sequence[Sequence[float]] | None = None,
    angles: int | None = None,
    padding: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A field's values on the section of a ball by a plane through its centre.

    The numbers behind :func:`plot_section`, for a caller who wants them and
    not a picture.

    Args:
        space: the ball or annulus.
        field: a vector of the space.
        longitude: the plane of this meridian and its antimeridian. Zero if
            the plane is not named at all.
        pole: the plane's pole, ``(latitude, longitude)`` in degrees: the
            section is by the great circle ninety degrees from it.
            ``(90, 0)`` is the equatorial plane.
        through: two points ``(latitude, longitude)`` the great circle passes
            through.
        angles: how many directions round the circle to evaluate at. Four per
            degree of the truncation by default, and at least 180.
        padding: include the padded mesh, and not the domain alone.

    Returns:
        ``(values, radii, latitudes, longitudes)``: the values as a
        ``(radii, angles)`` array, the radii, and the latitude and longitude
        in degrees of each direction round the circle, which starts at the
        right of the drawing and runs anticlockwise.

    Raises:
        ValueError: if more than one of ``longitude``, ``pole`` and
            ``through`` is given, the two points of ``through`` coincide or
            are opposite, or the field has the wrong shape.
    """
    from planetmodel import harmonics

    _, right, up = _section_frame(longitude, pole, through)
    count = max(180, 4 * space.lmax) if angles is None else int(angles)
    psi = 2.0 * np.pi * np.arange(count) / count
    directions = np.cos(psi)[:, None] * right + np.sin(psi)[:, None] * up
    colatitude = np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
    east = np.arctan2(directions[:, 1], directions[:, 0])
    functions, radii = _coefficient_functions(space, field, padding=padding)
    angular = harmonics.real_harmonics(space.lmax, colatitude, east)
    values = np.einsum("slmr,slmp->rp", functions, angular)
    return values, radii, 90.0 - np.degrees(colatitude), np.degrees(east)


def plot_section(
    space: Ball | Layered,
    field: np.ndarray,
    /,
    *,
    longitude: float | None = None,
    pole: Sequence[float] | None = None,
    through: Sequence[Sequence[float]] | None = None,
    ax: Any = None,
    angles: int | None = None,
    padding: bool = False,
    cmap: str = "RdBu",
    symmetric: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    outline: bool = True,
    title: str | None = None,
    **kwargs: Any,
) -> Any:
    """Draw a field on the section of a ball by a plane through its centre.

    Any such plane: a meridian's by ``longitude``, the equator's by
    ``pole=(90, 0)``, the one holding a source and a receiver by
    ``through=(source, receiver)``. The plane is seen from its pole, with
    north, or the direction in the plane nearest it, up: a meridian has its
    own longitude on the right, and a section through two points runs
    anticlockwise from the first to the second.

    Args:
        space: the ball or annulus, or a layered space of them.
        field: a vector of the space.
        longitude: the plane of this meridian and its antimeridian, the
            meridian on the right. Zero if the plane is not named at all.
        pole: the plane's pole, ``(latitude, longitude)`` in degrees.
        through: two points ``(latitude, longitude)`` on the great circle.
        ax: ordinary axes to draw on. A new figure is made if omitted.
        angles: directions round the circle to evaluate at.
        padding: draw the padded mesh too, beyond the outline of the domain.
        cmap: color map.
        symmetric: put zero at the middle of the color scale.
        vmin: lower color limit.
        vmax: upper color limit.
        colorbar: attach a colorbar.
        colorbar_label: label for the colorbar.
        outline: draw the boundaries of the domain.
        title: a title for the axes.
        **kwargs: passed to ``pcolormesh``.

    Returns:
        The ``(axes, mappable)`` pair.

    Raises:
        ValueError: if more than one of ``longitude``, ``pole`` and
            ``through`` is given, the two points of ``through`` coincide or
            are opposite, the field has the wrong shape, or ``padding`` is
            asked of a layered space.
        TypeError: for a layered space whose layers are not balls.
    """
    import matplotlib.pyplot as pyplot

    shells, fields = _shells(space, field)
    if len(shells) > 1 and padding:
        raise ValueError(
            "The padding of one layer lies over its neighbours, and cannot be "
            "drawn beside them: draw that layer on its own."
        )
    sections = [
        section_values(
            shell,
            part,
            longitude=longitude,
            pole=pole,
            through=through,
            angles=angles,
            padding=padding,
        )[:2]
        for shell, part in zip(shells, fields)
    ]
    if ax is None:
        _, ax = pyplot.subplots(figsize=(5.0, 4.4), layout="constrained")
    count = sections[0][0].shape[1]
    psi = 2.0 * np.pi * np.arange(count + 1) / count
    # One color scale for every shell, or a jump would not look like one.
    low, high = color_limits(
        np.concatenate([values.ravel() for values, _ in sections]),
        vmin=vmin,
        vmax=vmax,
        symmetric=symmetric,
    )
    for values, radii in sections:
        closed = np.concatenate([values, values[:, :1]], axis=1)
        mappable = ax.pcolormesh(
            radii[:, None] * np.cos(psi)[None, :],
            radii[:, None] * np.sin(psi)[None, :],
            closed,
            cmap=cmap,
            vmin=low,
            vmax=high,
            shading="gouraud",
            **kwargs,
        )
    if outline:
        edges = {shell.inner_radius for shell in shells} | {
            shell.radius for shell in shells
        }
        for radius in sorted(edges):
            if radius > 0.0:
                ax.plot(
                    radius * np.cos(psi),
                    radius * np.sin(psi),
                    color="black",
                    linewidth=0.6,
                )
    ax.set_aspect("equal")
    ax.set_axis_off()
    if colorbar:
        bar = ax.figure.colorbar(mappable, ax=ax, shrink=0.8, pad=0.03)
        if colorbar_label is not None:
            bar.set_label(colorbar_label)
    if title is not None:
        ax.set_title(title)
    return ax, mappable


def plot_section_points(
    space: Ball | Layered,
    points: Any,
    /,
    *,
    ax: Any,
    longitude: float | None = None,
    pole: Sequence[float] | None = None,
    through: Sequence[Sequence[float]] | None = None,
    tolerance: float = 5.0,
    marker: str = "^",
    size: float = 30.0,
    color: str = "black",
    **kwargs: Any,
) -> Any:
    """Mark on a section the points that lie in its plane, or near it.

    The plane is named as :func:`plot_section` names it, and must be the one
    the axes show. A point within ``tolerance`` degrees of the plane, as seen
    from the centre, is drawn where its foot in the plane falls, at its own
    radius; the others are left out, being somewhere else.

    Args:
        space: the ball or annulus, or a layered space of them.
        points: ``(radius, latitude, longitude)`` triples, the angles in
            degrees.
        ax: the axes of the section.
        longitude: the plane of this meridian and its antimeridian.
        pole: the plane's pole, ``(latitude, longitude)`` in degrees.
        through: two points ``(latitude, longitude)`` on the great circle.
        tolerance: how far off the plane, in degrees, a point may be.
        marker: matplotlib marker.
        size: marker size.
        color: marker color.
        **kwargs: passed to ``Axes.scatter``.

    Returns:
        The scatter, and a boolean array saying which points were drawn.

    Raises:
        ValueError: if more than one of ``longitude``, ``pole`` and
            ``through`` is given, or the two points of ``through`` coincide
            or are opposite.
    """
    normal, right, up = _section_frame(longitude, pole, through)
    located = np.asarray(points, dtype=float).reshape(-1, 3)
    directions = np.stack([_unit(lat, lon) for _, lat, lon in located])
    off = np.degrees(np.arcsin(np.clip(np.abs(directions @ normal), 0.0, 1.0)))
    near = off <= float(tolerance)
    feet = directions[near] - np.outer(directions[near] @ normal, normal)
    feet = feet / np.linalg.norm(feet, axis=1, keepdims=True)
    radii = located[near, 0]
    scatter = ax.scatter(
        radii * (feet @ right),
        radii * (feet @ up),
        marker=marker,
        s=size,
        color=color,
        zorder=3,
        **kwargs,
    )
    return scatter, near


def plot_profile(
    space: Ball | Layered,
    field: np.ndarray,
    latitude: float,
    longitude: float,
    /,
    *,
    ax: Any = None,
    points: int = 200,
    **kwargs: Any,
) -> Any:
    """Draw a field of a ball along one ray, against radius.

    Args:
        space: the ball or annulus, or a layered space of them.
        field: a vector of the space.
        latitude: the ray's latitude in degrees.
        longitude: its longitude in degrees.
        ax: ordinary axes to draw on. A new figure is made if omitted.
        points: how many radii of the domain to evaluate at, shared among
            the layers of a layered space by their thickness, at least eight
            each.
        **kwargs: passed to ``Axes.plot``.

    Returns:
        The ``(axes, line)`` pair.

    Raises:
        TypeError: for a layered space whose layers are not balls.
    """
    import matplotlib.pyplot as pyplot

    if ax is None:
        _, ax = pyplot.subplots(figsize=(5.0, 3.2), layout="constrained")
    shells, fields = _shells(space, field)
    extent = shells[-1].radius - shells[0].inner_radius
    line = None
    for shell, part in zip(shells, fields):
        # As many points as the shell's share of the whole, and a few at least.
        share = (shell.radius - shell.inner_radius) / extent
        radii = np.linspace(
            shell.inner_radius, shell.radius, max(8, int(round(share * int(points))))
        )
        values = shell.evaluate(
            np.asarray(part, dtype=float),
            [(radius, float(latitude), float(longitude)) for radius in radii],
        )
        if line is not None:
            # One field, one color, one legend entry, and a break at the jump.
            kwargs = {**kwargs, "color": line.get_color(), "label": "_nolegend_"}
        (line,) = ax.plot(radii, values, **kwargs)
    ax.set_xlim(shells[0].inner_radius, shells[-1].radius)
    ax.set_xlabel("radius")
    return ax, line


def _point_values(
    space: Ball, field: Any, /
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """The field as a function of ``(r, theta, phi)`` in ``planetmodel``'s
    conventions -- colatitude and longitude in radians, broadcast against the
    radius -- by the space's point evaluation, the components found once.

    A radius within ``planetmodel``'s allowance outside the domain, which the
    field wrapping this function has already granted, is moved onto the end.

    Raises:
        ValueError: if the field has the wrong shape.
    """
    values = np.asarray(field, dtype=float)
    if values.shape != space.grid_shape:
        raise ValueError(
            f"A field on this ball has shape {space.grid_shape}, got {values.shape}."
        )
    components = space.to_components(values)
    lower, upper = space.inner_radius, space.radius

    def evaluate(r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        r, theta, phi = np.broadcast_arrays(
            np.asarray(r, dtype=float),
            np.asarray(theta, dtype=float),
            np.asarray(phi, dtype=float),
        )
        if r.size == 0:
            return np.zeros(r.shape)
        points = np.stack(
            [
                np.clip(r.ravel(), lower, upper),
                np.clip(90.0 - np.degrees(theta.ravel()), -90.0, 90.0),
                np.degrees(phi.ravel()),
            ],
            axis=-1,
        )
        return space._local_evaluation(points).values(components).reshape(r.shape)

    return evaluate


def planet_model(space: Ball | Layered, field: Any, /, *, name: str = "field") -> Any:
    """A field of a ball as a ``planetmodel`` ``Model``, for drawing it
    through ``planetmodel.slices`` and ``planetmodel.plotting``.

    The model's skeleton is the domain's radii: the inner and the outer
    radius of a ball or annulus, and every interface of a layered space
    besides. Its geometry is plain, the identity mapping of the reference
    sphere, so a radius of the model is a radius of the space, in the
    space's units of length. Each layer holds one field under ``name``, a
    ``planetmodel.fields.AnalyticField`` whose ``evaluate(r, theta, phi)``
    is the space's point evaluation of ``field`` within that layer: the
    coordinates broadcast against each other, colatitude and longitude in
    radians, as ``planetmodel`` takes them. A radius outside the layer is
    refused with a ``ValueError``, as ``planetmodel``'s own fields refuse
    one, so at an interface each layer gives its own side.

    ``name`` is declared a scalar with no units, a name of ``planetmodel``'s
    vocabulary such as ``"rho"`` included, so that a label states only the
    name: what the numbers mean is the caller's to say.

    Args:
        space: the ball or annulus, or a layered space of them.
        field: a vector of the space.
        name: the name the field is held under in every layer.

    Returns:
        The ``planetmodel.model.Model``.

    Raises:
        ValueError: if the field, or a layer's part of it, has the wrong
            shape.
        TypeError: for a layered space whose layers are not balls.
    """
    from planetmodel.character import SCALAR
    from planetmodel.fields import AnalyticField
    from planetmodel.geometry import Geometry
    from planetmodel.model import Model
    from planetmodel.skeleton import Skeleton
    from planetmodel.vocabulary import FieldSpec

    shells, parts = _shells(space, field)
    skeleton = Skeleton([shells[0].inner_radius] + [shell.radius for shell in shells])
    layers = [
        {
            name: AnalyticField(
                (shell.inner_radius, shell.radius),
                _point_values(shell, part),
                name=name,
            )
        }
        for shell, part in zip(shells, parts)
    ]
    return Model(
        Geometry(skeleton),
        layers,
        specs={name: FieldSpec(SCALAR, None, meaning="a field of a pygeoinf ball")},
    )


def plot_cutaway(
    space: Ball | Layered,
    field: Any,
    /,
    *,
    longitudes: Sequence[float] = (0.0, 90.0),
    latitudes: Sequence[float] | None = None,
    inner: float = 0.0,
    outer: float | None = None,
    view: Sequence[float] | None = None,
    n: int = 400,
    side: str | None = "below",
    ax: Any = None,
    name: str = "field",
    inner_color: Any = "field",
    **kwargs: Any,
) -> Any:
    """Draw a field of a ball as the body seen from outside with a wedge cut
    from it, through ``planetmodel``'s cutaway.

    The picture is ``planetmodel.slices.cutaway`` of :func:`planet_model`,
    drawn by ``planetmodel.plotting.cutaway``: an orthographic view, rendered
    pixel by pixel, the field on the faces of the cut, on the surface, and on
    the sphere left standing in the cut at its inner radius. The angles are
    this module's, in degrees and latitudes, and are converted to the
    radians and colatitudes ``planetmodel`` takes.

    Args:
        space: the ball or annulus, or a layered space of them.
        field: a vector of the space.
        longitudes: the wedge, from its first longitude east to its second,
            in degrees: ``planetmodel``'s ``longitudes``.
        latitudes: the span of latitudes cut, in degrees, its ends in either
            order: ``planetmodel``'s ``colatitudes``. The whole span if
            omitted; ``(-90, 0)`` cuts the part south of the equator only.
        inner: the radius the cut goes down to, zero for the centre, or the
            domain's inner radius where that is higher, which an annulus
            shows as its inner surface: ``planetmodel``'s ``inner``.
        outer: the radius of the sphere shown as the surface, the domain's
            outer radius if omitted; within the domain, the faces of the cut
            stand proud of it: ``planetmodel``'s ``outer``.
        view: the ``(latitude, longitude)`` in degrees of the direction
            toward the viewer: ``planetmodel``'s ``view``. If omitted, above
            the cut's first longitude and toward the pole of the hemisphere
            the cut lies in.
        n: the pixels across the picture: ``planetmodel``'s ``n``.
        side: in a layered space, the side of an interface a sphere of the
            picture is taken on, where it lies on one, ``"below"`` or
            ``"above"``: ``planetmodel``'s ``side``.
        ax: ordinary axes to draw on. A new figure is made if omitted.
        name: the name the field is drawn and labeled under.
        inner_color: ``"field"``, or a color for the sphere left standing in
            the cut: the drawing's ``inner``.
        **kwargs: passed to ``planetmodel.plotting.cutaway``: ``surface``
            (``"field"``, a color, or the Earth's ``"land"`` or relief
            ``"image"``, which need cartopy), ``coastlines``, ``interfaces``,
            ``scale``, ``cmap``, ``vmin``, ``vmax``, ``diverging``, ``clip``,
            ``normalised``, ``colorbar``, and the rest to ``imshow``.

    Returns:
        What ``planetmodel.plotting.cutaway`` returns: the image drawn.

    Raises:
        ValueError: for a wedge or radii ``planetmodel`` refuses, or a field
            of the wrong shape.
        TypeError: for a layered space whose layers are not balls.
    """
    import matplotlib.pyplot as pyplot
    from planetmodel import slices
    from planetmodel.plotting import cutaway

    picture: dict[str, Any] = {
        "longitudes": tuple(np.radians(float(value)) for value in longitudes),
        "inner": inner,
        "outer": outer,
        "n": n,
        "side": side,
    }
    if latitudes is not None:
        picture["colatitudes"] = tuple(
            sorted(np.radians(90.0 - float(value)) for value in latitudes)
        )
    if view is not None:
        latitude, longitude = (float(value) for value in view)
        picture["view"] = (np.radians(90.0 - latitude), np.radians(longitude))
    model = planet_model(space, field, name=name)
    cut = slices.cutaway(model, name, **picture)
    if ax is None:
        _, ax = pyplot.subplots(figsize=(5.0, 5.0), layout="constrained")
    return cutaway(ax, cut, inner=inner_color, **kwargs)
