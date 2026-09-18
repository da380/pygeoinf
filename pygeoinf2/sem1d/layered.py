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
know what its user built; identity is the summands alone (as a direct sum's
already is, for the same reason: DECISIONS.md D-23), so those land on this
space (D-126).

**A point on an interface belongs to two layers**, and the field has two
values there. Evaluation at such a point is refused unless ``side="below"`` or
``"above"`` says which is meant; nowhere else is there anything to say. The
layers need not touch: a field that lives in a mantle and an inner core, and
not in the fluid between, is two layers with a gap, and a point in the gap is
in no layer.

The layers are all of one geometry: :mod:`~pygeoinf2.sem1d.interval`,
:mod:`~pygeoinf2.sem1d.radial` or :mod:`~pygeoinf2.sem1d.ball`. Build them and
hand them over, or let :meth:`Layered.interval`, :meth:`Layered.radial` or
:meth:`Layered.ball` build them from the boundaries, with every other argument
given once for all layers or, as a *list*, layer by layer.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from ..algebra.direct_sum import BlockDiagonalLinearOperator, _CoordinateDirectSum
from ..algebra.operators import LinearFunctional, LinearOperator
from ..algebra.spaces import EuclideanSpace, HilbertModule
from ..probability.gaussian import GaussianMeasure
from .axis import AxisSpace
from .ball import Ball
from .interval import Interval
from .radial import Radial

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


def _bounds(layer: AxisSpace | Ball, /) -> tuple[float, float]:
    """The ends of a layer's domain along the layered coordinate."""
    if isinstance(layer, Ball):
        return (layer.inner_radius, layer.radius)
    return layer.bounds


class Layered(_CoordinateDirectSum, HilbertModule):
    """Piecewise-continuous fields over layers that are adjacent in space."""

    def __init__(
        self,
        layers: Sequence[AxisSpace | Ball],
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
            ValueError: if they are out of order or overlap.
        """
        layers = tuple(layers)
        for kind in (Interval, Radial, Ball):
            if all(isinstance(layer, kind) for layer in layers):
                self._kind = kind
                break
        else:
            raise TypeError(
                "The layers must all be sem1d intervals, all radial profiles or "
                "all balls."
            )
        super().__init__(layers, labels=labels)

        ends = [_bounds(layer) for layer in layers]
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
    def _built(
        cls,
        kind: type,
        boundaries: Sequence[float],
        first: Any,
        names: tuple[str, str],
        labels: Sequence[str] | None,
        options: dict[str, Any],
    ) -> "Layered":
        """Layers of one geometry between consecutive boundaries."""
        boundaries = [float(b) for b in boundaries]
        count = len(boundaries) - 1
        if count < 1:
            raise ValueError("At least two boundaries are needed, for one layer.")
        leading = _per_layer(first, count, "The first argument")
        spread = {
            name: _per_layer(value, count, name) for name, value in options.items()
        }
        layers = [
            kind(
                leading[i],
                **{names[0]: boundaries[i], names[1]: boundaries[i + 1]},
                **{name: values[i] for name, values in spread.items()},
            )
            for i in range(count)
        ]
        # As the subclass its order names, Lebesgue or Sobolev (DECISIONS.md
        # D-3), which is also what `with_order` gives: the basis is shared.
        return cls([layer.with_order(layer.order) for layer in layers], labels=labels)

    @classmethod
    def interval(
        cls,
        breakpoints: Sequence[float],
        modes: int | list[int],
        /,
        *,
        labels: Sequence[str] | None = None,
        **options: Any,
    ) -> "Layered":
        """Layers on the line, between consecutive breakpoints.

        Args:
            breakpoints: the ends of the layers, ascending.
            modes: eigenfunctions kept in each layer.
            labels: names for the layers.
            **options: the keyword arguments of
                :class:`~pygeoinf2.sem1d.interval.Interval` -- ``order``,
                ``length_scale``, ``padding``, ``boundary``, ``ngll``,
                ``element_length`` -- each given once for all layers or, as a
                list, layer by layer.

        Returns:
            The layered space.
        """
        return cls._built(
            Interval, breakpoints, modes, ("lower", "upper"), labels, options
        )

    @classmethod
    def radial(
        cls,
        boundaries: Sequence[float],
        modes: int | list[int],
        /,
        *,
        labels: Sequence[str] | None = None,
        **options: Any,
    ) -> "Layered":
        """Radial profiles in concentric shells, between consecutive radii.

        Args:
            boundaries: the radii bounding the shells, ascending. A first
                radius of zero makes the innermost layer a whole ball.
            modes: eigenfunctions kept in each layer.
            labels: names for the layers.
            **options: the keyword arguments of
                :class:`~pygeoinf2.sem1d.radial.Radial`, each given once for
                all layers or, as a list, layer by layer.

        Returns:
            The layered space.
        """
        return cls._built(
            Radial, boundaries, modes, ("inner_radius", "radius"), labels, options
        )

    @classmethod
    def ball(
        cls,
        boundaries: Sequence[float],
        lmax: int | list[int],
        /,
        *,
        labels: Sequence[str] | None = None,
        **options: Any,
    ) -> "Layered":
        """Fields of position in concentric shells, between consecutive radii.

        Args:
            boundaries: the radii bounding the shells, ascending. A first
                radius of zero makes the innermost layer a whole ball.
            lmax: the largest spherical-harmonic degree kept in each layer.
            labels: names for the layers.
            **options: the keyword arguments of
                :class:`~pygeoinf2.sem1d.ball.Ball`, each given once for all
                layers or, as a list, layer by layer.

        Returns:
            The layered space.
        """
        return cls._built(
            Ball, boundaries, lmax, ("inner_radius", "radius"), labels, options
        )

    # ----------------------------------------------------------------- #
    #                              Identity                             #
    # ----------------------------------------------------------------- #

    def __eq__(self, other: object) -> bool:
        """Equal to any direct sum of the same layers, which is what this is:
        a block operator or a product measure builds a plain one, and has to
        land here."""
        if self is other:
            return True
        if not isinstance(other, _CoordinateDirectSum):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash((_CoordinateDirectSum, self._key()))

    def __repr__(self) -> str:
        spans = ", ".join(f"[{lower:g}, {upper:g}]" for lower, upper in self._ends)
        return f"Layered({self._kind.__name__}: {spans}; dim={self.dim})"

    # ----------------------------------------------------------------- #
    #                             Structure                             #
    # ----------------------------------------------------------------- #

    @property
    def layers(self) -> tuple[AxisSpace | Ball, ...]:
        """The spaces of the layers, in order."""
        return self.subspaces

    @property
    def geometry(self) -> type:
        """What the layers all are: ``Interval``, ``Radial`` or ``Ball``."""
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
            The layered space of the layers ``with_order``.
        """
        orders = _per_layer(order, len(self), "order")
        return Layered(
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
        """A point's position along the layered coordinate."""
        if self._kind is Ball:
            return float(np.asarray(point, dtype=float).reshape(-1)[0])
        return float(point)

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
                in a gap -- or on an interface with no side named, or the
                side is neither.
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
        self, what: str, points: Sequence[Any], side: str | None, /, *, unsafe: bool
    ) -> None:
        """Each layer's own guard, on the points that are in it."""
        for i, _, inside in self._grouped(points, side):
            self.layers[i]._require_point_evaluation(what, inside, unsafe=unsafe)

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
            "A Dirac functional", [point], side, unsafe=unsafe
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
        unsafe: bool = False,
    ) -> LinearOperator:
        """Evaluation at points anywhere in the medium, as an operator into a
        Euclidean space.

        Args:
            points: where to evaluate.
            side: which value is meant at the points on an interface.
            unsafe: build it even in layers too rough to admit it.

        Returns:
            The operator, assembled from :meth:`basis_matrix`.

        Raises:
            ValueError: if no points are given, one is in no layer or on an
                interface with no side named, or a layer's order is too low
                and *unsafe* is not set.
        """
        points = list(points)
        if not points:
            raise ValueError("At least one point is needed.")
        self._require_point_evaluation(
            "A point evaluation operator", points, side, unsafe=unsafe
        )
        return LinearOperator.from_matrix(
            self,
            EuclideanSpace(len(points)),
            self.basis_matrix(points, side=side),
            form="galerkin",
        )

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
            and intervals and one per direction for balls.

        Raises:
            ValueError: if directions are given where there are none to give,
                or not given in balls, or a layer's order is too low and
                *unsafe* is not set.
            IndexError: if there is no such interface.
        """
        position = self.interfaces[interface]
        if self._kind is Ball:
            if directions is None:
                raise ValueError("In balls the jump is taken at given directions.")
            points = [(position, float(lat), float(lon)) for lat, lon in directions]
        else:
            if directions is not None:
                raise ValueError("Only balls have directions to take a jump at.")
            points = [position]
        for side in ("below", "above"):
            self._require_point_evaluation(
                "A jump operator", points, side, unsafe=unsafe
            )
        matrix = self.basis_matrix(points, side="above") - self.basis_matrix(
            points, side="below"
        )
        return LinearOperator.from_matrix(
            self, EuclideanSpace(len(points)), matrix, form="galerkin"
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
        expectation: tuple | None = None,
        pointwise_std: Any = None,
    ) -> GaussianMeasure:
        """A Gaussian prior under which the layers are independent, each with
        covariance ``amplitude^2 A^-order`` of its own ``A``.

        Independence is what a discontinuity means: knowing the field on one
        side of an interface says nothing of the other. A medium continuous
        across an interface is this prior conditioned on the vanishing of
        :meth:`jump_operator` there.

        Args:
            order: the Sobolev order of each layer's measure, one for all or
                a list.
            amplitude: an overall scale on each layer's spectrum, likewise.
            expectation: the mean, a vector of this space. Zero if omitted.
            pointwise_std: calibrate each layer by its pointwise standard
                deviation: one number for all, a list of numbers or fields
                layer by layer, or a vector of this space giving it
                everywhere.

        Returns:
            The measure, on this space.

        Raises:
            ValueError: for a list of the wrong length.
        """
        count = len(self)
        orders = _per_layer(order, count, "order")
        amplitudes = _per_layer(amplitude, count, "amplitude")
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
                    expectation=means[i],
                    pointwise_std=deviations[i],
                )
                for i, layer in enumerate(self.layers)
            ],
            labels=self.labels,
        )
