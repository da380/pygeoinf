"""Piecewise-continuous fields: a direct sum of layers.

The fields of a layered medium are smooth within each layer and jump between
them -- a density across a core-mantle boundary, a wave speed across a
discontinuity in the mantle. No one space of this package holds such a field,
whose smoothness is what each is built on; a *direct sum* of them does, one
summand per layer, each on its own padded mesh with its own length scale,
order and truncation. That is all a :class:`Layered` space is, and the algebra
already knows what to do with it: a vector is a tuple of layer fields, a block
operator acts layer by layer, and a product of Gaussians is a prior under which
the layers are independent -- which is what a discontinuity means.

What is added here is what a direct sum cannot know, that its summands are
*adjacent in space*: which layer a point belongs to, evaluation and
observation at points anywhere in the medium, the jump across an interface,
integrals over the whole, one function sampled into every layer, and priors
given layer by layer.

**It is the direct sum, and compares equal to it.** A block operator or a
product measure lands on a plain ``DirectSum`` of the layers, which cannot
know what its user built. Equality and hashing depend on the summands alone,
as they do for any direct sum, so a ``Layered`` space and the plain direct
sum of its layers are the same space, and those results apply here.

**A point on an interface belongs to two layers**, and the field has two
values there. Evaluation at such a point is refused unless ``side="below"`` or
``"above"`` says which is meant; nowhere else is there anything to say. The
layers need not touch: a field that lives in a mantle and an inner core, and
not in the fluid between, is two layers with a gap, and a point in the gap is
in no layer.

The layers are all of one geometry, and each geometry has its own
``Layered``: :class:`pygeoinf.interval.Layered`,
:class:`pygeoinf.interval.radial.Layered` and :class:`pygeoinf.ball.Layered`
are subclasses of this one whose constructors build the layers from the
boundaries, with every other argument given once for all layers or, as a
*list*, layer by layer. This class takes layers already built, of any one
geometry, and is what the subclasses share; :meth:`Layered.from_layers` makes
a geometry's own from layers built by hand.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Sequence

import numpy as np

from ..algebra.direct_sum import BlockDiagonalLinearOperator, CoordinateDirectSum
from ..algebra.operators import LinearFunctional, LinearOperator
from ..algebra.spaces import EuclideanSpace, HilbertModule
from ..probability.gaussian import GaussianMeasure
from ._points import LocalEvaluation
from .axis import AxisSpace
from .base import (
    LengthScale,
    SpectralElementSpace,
    _evaluation_route,
    _local_operator,
)
from ..traits import Traits

__all__ = ["Layered"]

#: How close, relative to the extent of the medium, two ends must be to be one
#: interface, and a point to an interface to be on it.
_INTERFACE_RTOL = 1e-12


def _per_layer(value: Any, count: int, name: str, /) -> list:
    """An argument for each layer: a list is layer by layer, anything else is
    given to all. A tuple is one value -- a padding pair, say -- and not a
    sequence of them.

    Raises:
        ValueError: for a list of the wrong length.
    """
    if isinstance(value, list):
        if len(value) != count:
            raise ValueError(
                f"{name} has {len(value)} entries for {count} layers; a list is "
                f"read layer by layer."
            )
        return list(value)
    return [value] * count


def _geometry_of(layer: SpectralElementSpace, /) -> type:
    """The class that names a layer's geometry: the nearest ancestor that
    declares a ``_geometry`` tag of its own, which is the interval, the radial
    profile or the ball, and not the ``Lebesgue`` or ``Sobolev`` view of it."""
    return next(cls for cls in type(layer).__mro__ if "_geometry" in vars(cls))


def layers_between(
    kind: type,
    boundaries: Sequence[float],
    first: Any,
    names: tuple[str, str],
    options: dict[str, Any],
    /,
) -> list:
    """Layers of one geometry between consecutive boundaries.

    What the constructors of the geometries' ``Layered`` spaces share: the
    *first* positional argument of the layer class and every option are
    given once for all layers or, as a list, layer by layer, and the two
    boundaries of each layer go to the keyword arguments *names*.

    Args:
        kind: the layer class, called as ``kind(first, **options)``.
        boundaries: the ends of the layers, ascending.
        first: the layer class's leading positional argument, for all layers
            or as a list.
        names: the keyword arguments taking a layer's lower and upper
            boundary.
        options: the remaining keyword arguments, each for all layers or as a
            list.

    Returns:
        The layers, in order, each as the ``Lebesgue`` or ``Sobolev`` space its
        order names.

    Raises:
        ValueError: for fewer than two boundaries, or a list of the wrong
            length.
    """
    boundaries = [float(b) for b in boundaries]
    count = len(boundaries) - 1
    if count < 1:
        raise ValueError("At least two boundaries are needed, for one layer.")
    leading = _per_layer(first, count, "The first argument")
    spread = {name: _per_layer(value, count, name) for name, value in options.items()}
    layers = [
        kind(
            leading[i],
            **{names[0]: boundaries[i], names[1]: boundaries[i + 1]},
            **{name: values[i] for name, values in spread.items()},
        )
        for i in range(count)
    ]
    # As the subclass its order names, Lebesgue or Sobolev, which is what
    # `with_order` gives; the basis is shared, not recomputed.
    return [layer.with_order(layer.order) for layer in layers]


class _LayeredEvaluation:
    """The layers' matrix-free evaluations, each at the points in its layer,
    as one evaluation on the components of the whole."""

    def __init__(
        self, pieces: list[tuple[slice, np.ndarray, LocalEvaluation]], dim: int, /
    ) -> None:
        """
        Args:
            pieces: for each layer holding points, its columns among the
                components, which of the points it holds, and its evaluation
                at them.
            dim: the dimension of the whole.
        """
        self._pieces = pieces
        self._size = int(sum(where.size for _, where, _ in pieces))
        self._dim = dim

    def values(self, components: np.ndarray, /) -> np.ndarray:
        """The field with these components at every point, in the order the
        points were given."""
        components = np.asarray(components, dtype=float)
        out = np.empty(self._size)
        for columns, where, evaluation in self._pieces:
            out[where] = evaluation.values(components[columns])
        return out

    def derivative_components(self, data: np.ndarray, /) -> np.ndarray:
        """The components of ``x -> sum_i data_i x(p_i)``: each layer's from
        the data at its own points, zero in a layer with none."""
        data = np.asarray(data, dtype=float)
        out = np.zeros(self._dim)
        for columns, where, evaluation in self._pieces:
            out[columns] = evaluation.derivative_components(data[where])
        return out


class Layered(CoordinateDirectSum, HilbertModule):
    """Piecewise-continuous fields over layers that are adjacent in space.

    The layers are spectral-element spaces of one geometry, built already;
    the geometries' own ``Layered`` subclasses build them from boundaries.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "Layered":
        # A plain direct sum picks its class from its summands; here the class
        # is the one asked for, and the geometries' constructors take
        # boundaries, not layers, so the arguments are left to __init__.
        return object.__new__(cls)

    def __init__(
        self,
        layers: Sequence[SpectralElementSpace],
        /,
        *,
        labels: Sequence[str] | None = None,
    ) -> None:
        """
        Args:
            layers: the spaces of the layers, in order along the coordinate
                and all of one geometry. They may touch, which makes an
                interface, or leave a gap; they may not overlap.
            labels: names for the layers, so one can be reached by name.

        Raises:
            TypeError: if the layers are not all intervals, all radial
                profiles or all balls.
            ValueError: if there are none, or they are out of order or
                overlap.
        """
        layers = tuple(layers)
        if not layers:
            raise ValueError("At least one layer is needed.")
        kind = _geometry_of(layers[0])
        if not all(isinstance(layer, kind) for layer in layers):
            raise TypeError(
                "The layers must all be intervals, all radial profiles or all " "balls."
            )
        self._kind = kind
        # On an axis an interface is a point; in a ball it is a sphere.
        self._on_an_axis = issubclass(kind, AxisSpace)
        super().__init__(layers, labels=labels)

        ends = [layer.bounds for layer in layers]
        extent = ends[-1][1] - ends[0][0]
        self._tolerance = _INTERFACE_RTOL * abs(extent)
        for (_, upper), (lower, _) in zip(ends[:-1], ends[1:]):
            if lower < upper - self._tolerance:
                raise ValueError(
                    "The layers must be in order and must not overlap: one ends "
                    f"at {upper:g} and the next begins at {lower:g}."
                )
        self._ends = tuple(ends)
        self._offsets = np.concatenate(
            ([0], np.cumsum([layer.dim for layer in layers]))
        )

    # ----------------------------------------------------------------- #
    #                            Constructors                           #
    # ----------------------------------------------------------------- #

    @classmethod
    def from_layers(
        cls,
        layers: Sequence[SpectralElementSpace],
        /,
        *,
        labels: Sequence[str] | None = None,
    ) -> "Layered":
        """A layered space of this class over layers built by hand.

        The geometries' ``Layered`` subclasses build their layers from
        boundaries; this is their route in for layers built one by one, with
        a gap between two of them, say, or meshes that differ in more than a
        list of options can say.

        Args:
            layers: the spaces of the layers, in order and all of this
                class's geometry.
            labels: names for the layers.

        Returns:
            The layered space, an instance of this class.

        Raises:
            TypeError: if the layers are not all of one geometry.
            ValueError: if there are none, or they are out of order or
                overlap.
        """
        space = cls.__new__(cls)
        Layered.__init__(space, layers, labels=labels)
        return space

    # ----------------------------------------------------------------- #
    #                              Identity                             #
    # ----------------------------------------------------------------- #

    def __eq__(self, other: object) -> bool:
        """Equal to any direct sum of the same layers, which is what this is:
        a block operator or a product measure builds a plain one, and has to
        land here."""
        if self is other:
            return True
        if not isinstance(other, CoordinateDirectSum):
            return NotImplemented
        return self._key() == other._key()

    # Defining __eq__ discards the inherited hash; the sum's is the right one.
    __hash__ = CoordinateDirectSum.__hash__

    def __repr__(self) -> str:
        spans = ", ".join(f"[{lower:g}, {upper:g}]" for lower, upper in self._ends)
        return f"Layered({self._kind.__name__}: {spans}; dim={self.dim})"

    # ----------------------------------------------------------------- #
    #                             Structure                             #
    # ----------------------------------------------------------------- #

    @property
    def layers(self) -> tuple[SpectralElementSpace, ...]:
        """The spaces of the layers, in order."""
        return self.subspaces

    @property
    def geometry(self) -> type:
        """The class that names what the layers all are: the interval, the
        radial profile or the ball, and not the ``Lebesgue`` or ``Sobolev``
        view of it."""
        return self._kind

    @property
    def layer_bounds(self) -> tuple[tuple[float, float], ...]:
        """The two ends of each layer's domain."""
        return self._ends

    @property
    def interfaces(self) -> tuple[float, ...]:
        """Where two layers touch, and the field may jump."""
        return tuple(
            upper
            for (_, upper), (lower, _) in zip(self._ends[:-1], self._ends[1:])
            if abs(lower - upper) <= self._tolerance
        )

    @property
    def spatial_dimension(self) -> int:
        """The dimension of the domain the fields live on: the layers'."""
        return self.layers[0].spatial_dimension

    @property
    def domain_volume(self) -> float:
        """The measure of the layers' domains together, without their padding
        and without any gap between them."""
        return float(sum(layer.domain_volume for layer in self.layers))

    def with_order(self, order: float | list[float], /) -> "Layered":
        """The same fields, measured with different Sobolev orders.

        Args:
            order: one order for every layer, or a list of them.

        Returns:
            The layered space of the layers, each with its new order.
        """
        orders = _per_layer(order, len(self), "order")
        return type(self).from_layers(
            [layer.with_order(s) for layer, s in zip(self.layers, orders)],
            labels=self.labels,
        )

    def columns(self, key: int | str, /) -> slice:
        """Where a layer's components sit among the space's.

        Args:
            key: the layer, by position or by label.

        Returns:
            The slice of the component vector that is the layer's.
        """
        i = self.index(key)
        return slice(int(self._offsets[i]), int(self._offsets[i + 1]))

    # ----------------------------------------------------------------- #
    #                          Points and layers                        #
    # ----------------------------------------------------------------- #

    def _coordinate(self, point: Any, /) -> float:
        """A point's position along the layered coordinate: the point itself
        on an axis, the radius of a ``(radius, latitude, longitude)`` triple
        in a ball."""
        return float(np.asarray(point, dtype=float).reshape(-1)[0])

    def layer_index(self, point: Any, /, *, side: str | None = None) -> int:
        """Which layer a point belongs to.

        Args:
            point: a position, or a radius, or ``(radius, latitude,
                longitude)``, as the layers take them.
            side: for a point on an interface, which has two layers:
                ``"below"`` or ``"above"``. Ignored anywhere else.

        Returns:
            The layer's position among :attr:`layers`.

        Raises:
            ValueError: if the point is in no layer -- outside the medium or
                in a gap -- or on an interface with no side named, or *side*
                is not ``"below"``, ``"above"`` or ``None``.
        """
        if side not in (None, "below", "above"):
            raise ValueError("side must be 'below', 'above' or None.")
        position = self._coordinate(point)
        holding = [
            i
            for i, (lower, upper) in enumerate(self._ends)
            if lower - self._tolerance <= position <= upper + self._tolerance
        ]
        if not holding:
            raise ValueError(f"The point at {position:g} is in none of the layers.")
        if len(holding) == 1:
            return holding[0]
        if side is None:
            raise ValueError(
                f"The point at {position:g} is on an interface, where the field "
                f"has two values: say side='below' or side='above'."
            )
        return holding[0] if side == "below" else holding[-1]

    def _grouped(
        self, points: Sequence[Any], side: str | None, /
    ) -> list[tuple[int, np.ndarray, list]]:
        """The points layer by layer: the layer, which of the points are in it,
        and those points."""
        points = list(points)
        indices = np.array([self.layer_index(p, side=side) for p in points], dtype=int)
        groups = []
        for i in np.unique(indices):
            where = np.flatnonzero(indices == i)
            groups.append((int(i), where, [points[k] for k in where]))
        return groups

    def basis_matrix(
        self, points: Sequence[Any], /, *, side: str | None = None
    ) -> np.ndarray:
        """The basis at many points, as a ``(len(points), dim)`` array: a row
        is a layer's own, in that layer's columns, and zero elsewhere.

        Args:
            points: points of the medium.
            side: which layer is meant at a point on an interface.

        Returns:
            The rows of an observation operator's derivative matrix.

        Raises:
            ValueError: if a point is in no layer, or on an interface with no
                side named.
        """
        points = list(points)
        matrix = np.zeros((len(points), self.dim))
        for i, where, inside in self._grouped(points, side):
            matrix[where, self.columns(i)] = self.layers[i].basis_matrix(inside)
        return matrix

    def evaluate(
        self, x: tuple, points: Sequence[Any], /, *, side: str | None = None
    ) -> np.ndarray:
        """The field's values at points anywhere in the medium.

        Args:
            x: a vector of this space.
            points: points of the medium.
            side: which value is meant at a point on an interface.

        Returns:
            One value per point.

        Raises:
            ValueError: if a point is in no layer, or on an interface with no
                side named.
        """
        points = list(points)
        values = np.empty(len(points))
        for i, where, inside in self._grouped(points, side):
            values[where] = self.layers[i].evaluate(x[i], inside)
        return values

    def _require_point_evaluation(
        self,
        what: str,
        /,
        *,
        points: Sequence[Any] | None = None,
        side: str | None = None,
        unsafe: bool,
    ) -> None:
        """Each layer's own guard, on the points that are in it; every
        layer's, at any point, when no points are given."""
        if points is None:
            for layer in self.layers:
                layer._require_point_evaluation(what, unsafe=unsafe)
            return
        for i, _, inside in self._grouped(points, side):
            self.layers[i]._require_point_evaluation(what, points=inside, unsafe=unsafe)

    def dirac(
        self, point: Any, /, *, side: str | None = None, unsafe: bool = False
    ) -> LinearFunctional:
        """The evaluation functional at a point of the medium.

        Args:
            point: where to evaluate.
            side: which value is meant at a point on an interface.
            unsafe: build it even in a layer too rough to admit it.

        Returns:
            The functional, which sees the one layer the point is in.

        Raises:
            ValueError: if the point is in no layer, on an interface with no
                side named, or its layer's order is too low and *unsafe* is
                not set.
        """
        self._require_point_evaluation(
            "A Dirac functional", points=[point], side=side, unsafe=unsafe
        )
        return LinearFunctional.from_derivative_components(
            self, self.basis_matrix([point], side=side)[0]
        )

    def point_evaluation_operator(
        self,
        points: Sequence[Any],
        /,
        *,
        side: str | None = None,
        route: Literal["auto", "dense", "local"] = "auto",
        unsafe: bool = False,
    ) -> LinearOperator:
        """Evaluation at points anywhere in the medium, as an operator into a
        Euclidean space.

        By the layers' two routes, which agree to rounding and differ in
        memory. ``"dense"`` assembles :meth:`basis_matrix`, eight bytes per
        point per component of the whole, zero outside each point's layer.
        ``"local"`` is each layer's matrix-free evaluation at the points in
        it, holding ``ngll`` weights per point and, in balls and disks, the
        angular functions at the points; an application costs about what the
        dense one does. See
        :meth:`~pygeoinf.spectral_element.SpectralElementSpace.point_evaluation_operator`.
        ``"auto"`` is the dense route while its matrix fits 256 MB, and the
        local one beyond.

        Args:
            points: where to evaluate.
            side: which value is meant at the points on an interface.
            route: ``"auto"``, ``"dense"`` or ``"local"``.
            unsafe: build it even in layers too rough to admit it.

        Returns:
            The operator.

        Raises:
            ValueError: if no points are given, one is in no layer or on an
                interface with no side named, the route is none of the three,
                or a layer's order is too low and *unsafe* is not set.
        """
        points = list(points)
        if not points:
            raise ValueError("At least one point is needed.")
        route = _evaluation_route(route, len(points), self.dim)
        self._require_point_evaluation(
            "A point evaluation operator", points=points, side=side, unsafe=unsafe
        )
        codomain = EuclideanSpace(len(points))
        if route == "dense":
            return LinearOperator.from_matrix(
                self, codomain, self.basis_matrix(points, side=side), form="galerkin"
            )
        pieces = [
            (self.columns(i), where, self.layers[i]._local_evaluation(inside))
            for i, where, inside in self._grouped(points, side)
        ]
        return _local_operator(self, codomain, _LayeredEvaluation(pieces, self.dim))

    def jump_operator(
        self,
        interface: int,
        /,
        *,
        directions: Sequence[Sequence[float]] | None = None,
        unsafe: bool = False,
    ) -> LinearOperator:
        """The jump of the field across an interface: its value just above,
        less its value just below.

        What a reflection coefficient sees, and what a prior on a
        discontinuity is a statement about. Conditioning on its vanishing is
        how two layers are made continuous at the interface they share.

        Args:
            interface: which interface, by position among :attr:`interfaces`.
            directions: in balls, the ``(latitude, longitude)`` pairs at which
                the jump is taken. Not given for profiles or intervals, whose
                interface is a single point.
            unsafe: build it even in layers too rough to admit it.

        Returns:
            The operator into a Euclidean space, of one dimension for profiles
            and intervals, one per direction for balls and one per angle for
            disks.

        Raises:
            ValueError: if directions are given where there are none to give,
                or not given in balls, or a layer's order is too low and
                *unsafe* is not set.
            IndexError: if there is no such interface.
        """
        position = self.interfaces[interface]
        if not self._on_an_axis:
            if directions is None:
                raise ValueError("In balls the jump is taken at given directions.")
            points = [(position, float(lat), float(lon)) for lat, lon in directions]
        else:
            if directions is not None:
                raise ValueError("Only balls have directions to take a jump at.")
            points = [position]
        for side in ("below", "above"):
            self._require_point_evaluation(
                "A jump operator", points=points, side=side, unsafe=unsafe
            )
        matrix = self.basis_matrix(points, side="above") - self.basis_matrix(
            points, side="below"
        )
        # Onto: the jumps at distinct interfaces are independent functionals,
        # each layer's basis being free to take any value at its ends. So an
        # exact conditioning on the jumps is well posed and says so.
        return LinearOperator.from_matrix(
            self,
            EuclideanSpace(len(points)),
            matrix,
            form="galerkin",
            traits=Traits.SURJECTIVE,
        )

    # ----------------------------------------------------------------- #
    #                     Functions, products, integrals                #
    # ----------------------------------------------------------------- #

    def project_function(
        self,
        function: Callable[[Any], float] | list[Callable[[Any], float]],
        /,
        *,
        extension: str = "fit",
    ) -> tuple:
        """The field of this space that a function of position names.

        Each layer takes the function on its own domain, as its own
        ``project_function`` does. One function serves them all unless it
        jumps: at an interface it is asked by both layers about the same
        point and cannot tell them apart, so a function with discontinuities
        is given as a list, one smooth piece per layer.

        Args:
            function: called with a point of the medium, or a list of such,
                layer by layer.
            extension: how each layer treats its padding: ``"fit"``,
                ``"odd"`` or ``"constant"``.

        Returns:
            The field, a tuple of layer fields.

        Raises:
            ValueError: for a list of the wrong length, or an extension the
                layers do not know.
        """
        pieces = _per_layer(function, len(self), "function")
        return tuple(
            layer.project_function(piece, extension=extension)
            for layer, piece in zip(self.layers, pieces)
        )

    def multiply(self, x: tuple, y: tuple) -> tuple:
        """The pointwise product, layer by layer, left on each layer's grid."""
        return tuple(layer.multiply(a, b) for layer, a, b in zip(self.layers, x, y))

    def sqrt(self, x: tuple) -> tuple:
        """The pointwise square root, layer by layer."""
        return tuple(layer.sqrt(a) for layer, a in zip(self.layers, x))

    def multiplication_operator(self, f: tuple, /) -> LinearOperator:
        """The operator ``u -> f u``: each layer's own, side by side."""
        return BlockDiagonalLinearOperator(
            [layer.multiplication_operator(a) for layer, a in zip(self.layers, f)]
        )

    def truncate(self, x: tuple, /) -> tuple:
        """Each layer's field settled into the span of its kept modes."""
        return tuple(layer.truncate(a) for layer, a in zip(self.layers, x))

    def integral_functional(self) -> LinearFunctional:
        """The integral of a field over the whole medium: the sum of the
        layers' own, each over its domain alone."""
        return LinearFunctional.from_derivative_components(
            self,
            np.concatenate(
                [
                    layer.integral_functional().derivative_components
                    for layer in self.layers
                ]
            ),
        )

    # ----------------------------------------------------------------- #
    #                              Measures                             #
    # ----------------------------------------------------------------- #

    def sobolev_measure(
        self,
        order: float | list[float],
        /,
        *,
        amplitude: float | list[float] = 1.0,
        length_scale: LengthScale | list[LengthScale | None] | None = None,
        lateral_length_scale: LengthScale | list[LengthScale | None] | None = None,
        expectation: tuple | None = None,
        pointwise_std: Any = None,
        unsafe: bool = False,
    ) -> GaussianMeasure:
        """A Gaussian prior under which the layers are independent, each with
        covariance ``amplitude^2 A'^-order``, ``A'`` the layer's own ``A``
        or one with length scales of the prior's.

        Independence is what a discontinuity means: knowing the field on one
        side of an interface says nothing of the other. A medium continuous
        across an interface is this prior conditioned on the vanishing of
        :meth:`jump_operator` there.

        Args:
            order: the Sobolev order of each layer's measure, one for all or
                a list.
            amplitude: an overall scale on each layer's spectrum, likewise.
            length_scale: the prior's radial length scale in each layer, a
                number or a function for all or a list layer by layer, with
                ``None`` for a layer's own; see the layers'
                ``sobolev_measure``.
            lateral_length_scale: the prior's lateral length scale,
                likewise, in a ball or a disk.
            expectation: the mean, a vector of this space. Zero if omitted.
            pointwise_std: calibrate each layer by its pointwise standard
                deviation: one number for all, a list of numbers or fields
                layer by layer, or a vector of this space giving it
                everywhere.
            unsafe: build the layers' measures at an order at or below
                half the dimension anyway, for a test that means it.

        Returns:
            The measure, on this space.

        Raises:
            ValueError: for a list of the wrong length, or for an order at or
                below half the dimension without *unsafe*, a length scale
                that is not positive, or a lateral one in layers of one
                coordinate, which each layer's measure refuses.
        """
        count = len(self)
        orders = _per_layer(order, count, "order")
        amplitudes = _per_layer(amplitude, count, "amplitude")
        radial_scales = _per_layer(length_scale, count, "length_scale")
        lateral_scales = _per_layer(lateral_length_scale, count, "lateral_length_scale")
        if isinstance(pointwise_std, tuple):
            deviations = list(pointwise_std)
            if len(deviations) != count:
                raise ValueError(
                    f"pointwise_std has {len(deviations)} fields for {count} layers."
                )
        else:
            deviations = _per_layer(pointwise_std, count, "pointwise_std")
        means = [None] * count if expectation is None else list(expectation)
        return GaussianMeasure.from_product(
            [
                layer.sobolev_measure(
                    orders[i],
                    amplitude=amplitudes[i],
                    length_scale=radial_scales[i],
                    lateral_length_scale=lateral_scales[i],
                    expectation=means[i],
                    pointwise_std=deviations[i],
                    unsafe=unsafe,
                )
                for i, layer in enumerate(self.layers)
            ],
            domain=self,
        )
