"""Fields of a flat layer: a Fourier series across it, spectral elements in
depth.

``A = 1 - div(kappa grad)`` on a layer that is periodic across, with ``kappa``
a function of depth alone, commutes with lateral translations, and in the
real Fourier basis of the lateral box block-diagonalizes into the family of
operators of depth

    A_k f = f - (kappa f')' + |k|^2 kappa f,

one per lateral wavevector ``k``, each self-adjoint on ``L^2(dz)``. That
family is ``planetmodel``'s ``SeparableOperatorFamily`` under the plain
measure, its lateral term of constant profile and its members indexed by the
value ``|k|^2``: the flat counterpart of the family of a disk, whose members
are indexed by order. Components of equal ``|k|^2`` share a member, hence one
eigenproblem; ``planetmodel.cartesian.LayerBasis`` holds one ``SpectralBasis``
per distinct ``|k|^2`` and fixes the order of the coefficients as one vector,
and ``planetmodel.fourier.Transform`` goes between the lateral grid and the
Fourier components.

:class:`FlatSpace` is the space built on them, for any number of lateral
axes; :mod:`pygeoinf.strip` names it with one, a field of ``(depth, x)``, and
:mod:`pygeoinf.slab` with two, a field of ``(depth, x, y)``.
:class:`FlatLayered` is the direct sum of such layers stacked in depth, over
one lateral box and grid, which :class:`pygeoinf.strip.Layered` and
:class:`pygeoinf.slab.Layered` name.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, Callable, Hashable, Literal, Sequence

import numpy as np
from numpy.random import Generator
from planetmodel.cartesian.randomfield import LayerBasis, distinct_eigenvalues
from planetmodel.fourier import PaddedDomain, Transform
from planetmodel.randomfield import (
    SeparableOperatorFamily,
    SpectralBasis,
    padded_mesh,
    restriction,
)

from ..algebra.operators import LinearFunctional, LinearOperator
from ..algebra.spaces import EuclideanSpace, resolve_rng
from ..traits import Traits
from . import base as _base
from ._points import _POSITION_RTOL, LocalEvaluation
from .layered import Layered, _LayeredEvaluation, layers_between
from .base import (
    _EXTENSIONS,
    _NODES_PER_MODE,
    Boundary,
    LengthScale,
    SpectralElementSpace,
    _RadialGroup,
    _checked_fit,
    _clamped_length_scale,
    _default_element_length,
    _fitted_over_the_domain,
    _length_scale_key,
    _local_operator,
    _resolved_boundary,
    _resolved_padding,
    _sampled_with_extension,
)

__all__ = ["FlatLayered", "FlatSpace", "LateralDomain"]

#: The lateral box: the periods from the origin, one number per axis (one
#: number alone, for one axis), when the domain is the whole period; or a
#: ``planetmodel.fourier.PaddedDomain``, or anything with its ``bounds`` and
#: ``padding``, when the domain is a bounded region inside a wider period.
type LateralDomain = float | Sequence[float] | PaddedDomain | Any

#: Relative margin on an eigenvalue bound, so that roundoff cannot drop the
#: mode that defines it.
_BOUND_RTOL = 1e-10

#: Lateral basis functions squared on the grid at a time, when a pointwise
#: variance is asked of weights that tell the components of one wavenumber
#: apart: 32 MB for them and as much for their unit components, whatever the
#: grid.
_SQUARES_ENTRIES = 4_000_000


class _NufftEvaluation:
    """Evaluation at scattered points of the field with given components,
    and its transpose, with the lateral sums by non-uniform FFT.

    The same contractions as :class:`LocalEvaluation`, to the same
    interface, with the product of the lateral basis at the points and the
    coefficient functions at an element's nodes taken by ``finufft``
    through ``planetmodel``'s ``Transform.synthesise``, and its transpose
    by ``Transform.accumulate``: one batched transform per element of the
    depth mesh that holds a point, of cost the lateral grid's and the
    points' and not their product. Nothing is held per point but its
    element and its ``ngll`` Lagrange weights. On a slab of 64 by 64 lateral
    points and 101 nodes in depth, on two threads, an application at 20 000
    points took 0.04 s against 0.9 s by the direct sums, and at 2000 points
    0.03 s against 0.06 s.
    """

    def __init__(
        self,
        space: FlatSpace,
        points: np.ndarray,
        /,
        *,
        epsilon: float,
    ) -> None:
        """
        Args:
            space: the space.
            points: shape ``(points, 1 + lateral axes)``, depth first, each
                in the domain.
            epsilon: the relative accuracy asked of ``finufft``.

        Raises:
            ValueError: if a depth lies outside the domain.
        """
        mesh = space.basis.family.mesh
        lower, upper = space.bounds
        depths = points[:, 0]
        slack = _POSITION_RTOL * (upper - lower)
        outside = (depths < lower - slack) | (depths > upper + slack)
        if np.any(outside):
            raise ValueError(
                f"Every point must lie in the domain, at a depth in "
                f"[{lower:g}, {upper:g}]; one is at {depths[outside][0]:g}."
            )
        elements, weights = mesh.interpolation_weights(np.clip(depths, lower, upper))
        order = np.argsort(elements, kind="stable")
        self._order = order
        self._weights = weights[order]
        self._lateral = points[order, 1:]
        inside = elements[order]
        cuts = [0, *(np.flatnonzero(np.diff(inside)) + 1), order.size]
        self._runs = [
            (mesh.gmap[int(inside[a])], int(a), int(b))
            for a, b in zip(cuts[:-1], cuts[1:])
        ]
        self._nglob = mesh.nglob
        self._space = space
        self._epsilon = float(epsilon)

    @property
    def size(self) -> int:
        """The number of points."""
        return int(self._order.size)

    def values(self, components: np.ndarray, /) -> np.ndarray:
        """The field with these components at every point.

        Args:
            components: of the space.

        Returns:
            One value per point, in the order the points were given.
        """
        space = self._space
        functions = space._nodal_functions(components)
        ordered = np.empty(self.size)
        for nodes, a, b in self._runs:
            local = space.transform.synthesise(
                functions[:, nodes].T, self._lateral[a:b], epsilon=self._epsilon
            )
            ordered[a:b] = np.einsum("ji,ij->i", local, self._weights[a:b])
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
        space = self._space
        data = np.asarray(data, dtype=float).reshape(-1)
        weighted = self._weights * data[self._order][:, None]
        functions = np.zeros((space.transform.size, self._nglob))
        for nodes, a, b in self._runs:
            functions[:, nodes] += space.transform.accumulate(
                weighted[a:b].T, self._lateral[a:b], epsilon=self._epsilon
            ).T
        return space._nodal_projection(functions)


class FlatSpace(SpectralElementSpace):
    """Fields on a flat layer, ``top <= depth <= bottom``, over a lateral box
    of one axis or two.

    Subclasses name the number of lateral axes, ``_lateral_axes``, and tag
    the geometry with ``_geometry``.
    """

    #: The number of lateral axes: one for a strip, two for a slab.
    _lateral_axes: int = 1
    #: The tag of the geometry, which names the layers of a layered space.
    _geometry: str = "flat"

    def __init__(
        self,
        shape: int | Sequence[int],
        /,
        *,
        lateral: LateralDomain | None = None,
        top: float = 0.0,
        bottom: float = 1.0,
        order: float = 0.0,
        length_scale: LengthScale = 1.0,
        depth_modes: int | None = None,
        max_eigenvalue: float | None = None,
        padding: float | tuple[float, float] | None = None,
        boundary: Boundary = "robin",
        ngll: int = 5,
        element_length: float | None = None,
    ) -> None:
        """
        Args:
            shape: the points of the lateral grid along each axis, which is
                also the lateral band: a grid holds exactly as many Fourier
                components as it has points. One number serves a single
                axis.
            lateral: the lateral box. The periods from the origin, one per
                axis, make the domain the whole period, a field periodic
                across it; a ``planetmodel.fourier.PaddedDomain`` gives the
                bounds of a domain proper and the padding that separates it
                from its periodic images, for a literal bounded region. A
                unit period on every axis if omitted.
            top: the depth of the top of the layer. Depth is positive
                downward.
            bottom: the depth of its base.
            order: the Sobolev order. Zero gives the Lebesgue space.
            length_scale: ``L`` in ``A``, a number or a function of depth. It
                is asked about the domain only; the padding takes its value
                at the nearer end.
            depth_modes: keep this many modes of depth at every lateral
                wavenumber.
            max_eigenvalue: keep, at every lateral wavenumber, the modes of
                depth whose eigenvalue of ``A`` is at most this, and never
                fewer than one: a grid holds every component of its band,
                so a wavenumber whose first eigenvalue lies above the bound
                keeps that one mode. With neither given the bound is the
                first eigenvalue of the largest wavenumber the grid holds
                along every axis, the Nyquist one of its coarsest axis: the
                truncation is then isotropic, the shortest wavelength kept
                in depth being the shortest the grid resolves in every
                lateral direction, and the corners of a grid of two axes,
                beyond that wavenumber, keep one mode each.
            padding: how far the mesh in depth extends past each end, one
                length or an ``(above, below)`` pair: above the top, into
                negative depth under a free surface, as below the base. Two
                length scales by default; four under the natural condition.
            boundary: the condition at the two ends of the mesh in depth,
                which lie in the padding; nothing is imposed at the top or
                the base of the layer itself. ``"robin"``, the default, ends
                the mesh as if it went on, ``u' = -+ 0.7 u / L``; ``None`` is
                the natural condition, and doubles the default padding; a
                number or a pair gives the coefficients themselves. One
                coefficient serves every wavenumber.
            ngll: Gauss-Lobatto-Legendre nodes per element in depth.
            element_length: the longest an element in depth may be. By
                default short enough for the shortest mode kept to be
                resolved, and no longer than half the smallest length scale.

        Raises:
            ValueError: if the shape or the lateral box has another number of
                axes than the geometry, ``top`` is not above ``bottom``, a
                length scale or the element length is not positive, both
                truncations are given, or more modes are asked for than the
                mesh holds.
        """
        axes = self._lateral_axes
        shape = (int(shape),) if np.ndim(shape) == 0 else tuple(int(n) for n in shape)
        if len(shape) != axes:
            raise ValueError(
                f"The lateral grid of this geometry has {axes} "
                f"{'axis' if axes == 1 else 'axes'}; got the shape {shape}."
            )
        domain = PaddedDomain.of((1.0,) * axes if lateral is None else lateral)
        if domain.dimension != axes:
            raise ValueError(
                f"The lateral box of this geometry has {axes} "
                f"{'axis' if axes == 1 else 'axes'}; got {domain!r}."
            )
        transform = Transform(shape, domain)
        shallow, deep = float(top), float(bottom)
        if not (np.isfinite(shallow) and np.isfinite(deep) and shallow < deep):
            raise ValueError("The depths must be finite and satisfy top < bottom.")
        if depth_modes is not None and max_eigenvalue is not None:
            raise ValueError("Give depth_modes or max_eigenvalue, not both.")

        clamped, probe = _clamped_length_scale(length_scale, shallow, deep)
        pads = _resolved_padding(padding, probe, boundary)
        robin = _resolved_boundary(boundary, probe, None)

        # The largest wavenumber held along every axis: the Nyquist one of
        # the coarsest, an axis of a single point holding none.
        nyquist = [
            (2.0 * np.pi * (n // 2) / length) ** 2
            for n, length in zip(shape, domain.lengths)
            if n > 1
        ]
        band = min(nyquist, default=0.0)

        ngll = int(ngll)

        def family_on(longest: float) -> SeparableOperatorFamily:
            mesh = padded_mesh(
                shallow, deep, pad=pads, weight="one", ngll=ngll, drmax=longest
            )
            return SeparableOperatorFamily(
                mesh,
                kappa=lambda z: clamped(z) ** 2,
                weight="one",
                lateral="one",
                index="value",
                robin=robin,
            )

        if element_length is None:
            # The shortest wavelength kept, estimated before there is a mesh
            # to ask: an eigenvalue bound is a wavenumber where the length
            # scale is smallest, and a number of modes is that many half
            # waves along the mesh. The default bound is the first
            # eigenvalue of the band's wavenumber, read here off a mesh that
            # resolves the length scale and no more: an eigenvalue from such
            # a mesh errs upwards, and so the element length downwards.
            if max_eigenvalue is not None:
                bound = float(max_eigenvalue)
            elif depth_modes is None:
                coarse = family_on(_default_element_length(0.5 * float(probe.min())))
                bound = float(coarse.eigvalsh(band)[0])
            if depth_modes is not None:
                wavenumber = int(depth_modes) * np.pi / (deep - shallow + sum(pads))
            else:
                wavenumber = np.sqrt(max(bound - 1.0, 0.0)) / probe.min()
            element_length = _default_element_length(
                min(
                    (ngll - 1) * np.pi / (_NODES_PER_MODE * max(wavenumber, 1e-300)),
                    0.5 * float(probe.min()),
                )
            )
        element_length = float(element_length)
        if element_length <= 0.0:
            raise ValueError("The element length must be positive.")

        family = family_on(element_length)
        mesh = family.mesh
        physical = restriction(mesh, shallow, deep)
        values, _ = distinct_eigenvalues(transform)
        if depth_modes is not None:
            count = int(depth_modes)
            if not 1 <= count <= mesh.nglob:
                raise ValueError(
                    f"A mesh of {mesh.nglob} nodes in depth holds between one "
                    f"and {mesh.nglob} modes; {count} were asked for. Shorten "
                    f"element_length for more."
                )
            bases = [
                SpectralBasis(family, float(value), restrict=physical, nmodes=count)
                for value in values
            ]
        else:
            if max_eigenvalue is None:
                held = float(values[np.argmin(np.abs(values - band))])
                max_eigenvalue = float(family.eigvalsh(held)[0]) * (1.0 + _BOUND_RTOL)
            bound = float(max_eigenvalue)
            bases = []
            for value in values:
                # The pairs computed here are the family's to keep, so the
                # basis built next asks for nothing again.
                kept = family.eig(float(value), theta_max=bound)[0].size
                rule = {"theta_max": bound} if kept else {"nmodes": 1}
                bases.append(
                    SpectralBasis(family, float(value), restrict=physical, **rule)
                )
        self._basis = LayerBasis(transform, bases)
        self._transform = transform

        self._depths = (shallow, deep)
        self._padding = pads
        self._robin = robin
        self._matched_boundary = isinstance(boundary, str)
        self._ngll = ngll
        self._element_length = element_length
        self._length_scale = length_scale
        self._length_scale_key = _length_scale_key(length_scale, clamped, mesh.rglob)
        super().__init__(self._basis.theta, order)

    # ----------------------------------------------------------------- #
    #                              Identity                             #
    # ----------------------------------------------------------------- #

    @property
    def top(self) -> float:
        """The depth of the top of the layer."""
        return self._depths[0]

    @property
    def bottom(self) -> float:
        """The depth of the base of the layer."""
        return self._depths[1]

    @property
    def bounds(self) -> tuple[float, float]:
        """The depths of the top and the base: the two ends of the domain
        along the mesh."""
        return self._depths

    @property
    def padding(self) -> tuple[float, float]:
        """The padding in depth, above the top and below the base."""
        return self._padding

    @property
    def robin(self) -> tuple[float, float]:
        """The Robin coefficients at the two ends of the mesh in depth, zero
        meaning the natural condition."""
        return self._robin

    @property
    def length_scale(self) -> LengthScale:
        """The length scale of ``A``, as it was given."""
        return self._length_scale

    @property
    def spatial_dimension(self) -> int:
        """Depth and the lateral axes: two for a strip, three for a slab."""
        return 1 + self._lateral_axes

    @property
    def lateral_shape(self) -> tuple[int, ...]:
        """The points of the lateral grid along each axis."""
        return self._transform.shape

    @property
    def lateral_domain(self) -> PaddedDomain:
        """The lateral box: the bounds of the domain proper, its padding, and
        the periods they make."""
        return self._transform.domain

    @property
    def domain_volume(self) -> float:
        """The volume of the domain proper (the area, with one lateral axis),
        excluding every padding."""
        return self._transform.domain.domain_volume * (self.bottom - self.top)

    @property
    def basis(self) -> LayerBasis:
        """``planetmodel``'s eigenbases of depth, one per distinct lateral
        ``|k|^2``, which do the numerics and fix the order of the
        components."""
        return self._basis

    @property
    def transform(self) -> Transform:
        """``planetmodel``'s Fourier transform of the lateral grid, which
        holds the grid, the packing of the components and the wavevectors."""
        return self._transform

    @property
    def eigenvalues(self) -> np.ndarray:
        """The eigenvalue of ``A`` attached to each component: ascending
        within a Fourier component, the same for every component of one
        ``|k|^2``, and at least one."""
        return self._basis.theta

    @cached_property
    def _starts(self) -> np.ndarray:
        """Where the coefficients of each Fourier component begin, and the
        dimension last: the components follow one another in the transform's
        order, each with the modes of its wavenumber's basis."""
        counts = self._basis.nmodes[self._basis.member]
        return np.concatenate(([0], np.cumsum(counts)))

    @cached_property
    def _entries(self) -> tuple[np.ndarray, ...]:
        """For each basis, the components that are its coefficients: shape
        ``(Fourier components sharing it, its modes)``."""
        basis = self._basis
        return tuple(
            self._starts[basis.components(i)][:, None]
            + np.arange(int(basis.nmodes[i]))[None, :]
            for i in range(len(basis))
        )

    @property
    def wavevectors(self) -> np.ndarray:
        """The signed integer lateral wavevector of each component, shape
        ``(lateral axes, dim)``; ``2 pi k_i / period_i`` is the physical
        one."""
        counts = np.diff(self._starts)
        return np.repeat(self._transform.wavevectors, counts, axis=1)

    @property
    def phases(self) -> np.ndarray:
        """Zero for each component of a cosine or of a mode that is its own
        conjugate, one for a sine's."""
        return np.repeat(self._transform.phases, np.diff(self._starts))

    @property
    def depth_indices(self) -> np.ndarray:
        """Which mode of depth of its wavenumber each component is, from zero
        in ascending eigenvalue."""
        return np.arange(self.dim) - np.repeat(self._starts[:-1], np.diff(self._starts))

    def _coordinate_key(self) -> Hashable:
        """The lateral grid and box, the mesh, the operator and the
        truncation, with the metric left out.

        Tagged by geometry and not by ``type(self)``: ``Lebesgue`` and
        ``Sobolev`` are views of one set of fields. Whether the Robin
        coefficients were matched to the length scale or given is part of
        it: a prior of another length scale matches its own on the first
        and keeps the coefficients on the second.
        """
        domain = self._transform.domain
        return (
            self._geometry,
            self._transform.shape,
            domain.bounds,
            domain.padding,
            tuple(int(n) for n in self._basis.nmodes),
            self._depths,
            self._padding,
            self._robin,
            self._matched_boundary,
            self._ngll,
            self._element_length,
            self._length_scale_key,
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(shape={self._transform.shape}, "
            f"dim={self.dim}, depths=[{self.top:g}, {self.bottom:g}], "
            f"order={self._order:g})"
        )

    # ----------------------------------------------------------------- #
    #                          The coordinate map                       #
    # ----------------------------------------------------------------- #

    def to_components(self, x: np.ndarray) -> np.ndarray:
        """The coefficients of the kept eigenfunctions: a Fourier analysis of
        every level of the grid, then each component's function of depth
        projected onto the modes of its wavenumber, both by the grid's own
        quadrature."""
        functions = self._transform.analyse_grid(np.asarray(x, dtype=float))
        return self._basis.analyse(functions.T)

    def from_components(self, c: np.ndarray) -> np.ndarray:
        """The values on the padded grid: the syntheses in depth, then a
        Fourier synthesis of every level."""
        functions = self._basis.synthesise(c, physical=False)
        return self._transform.synthesise_grid(functions.T)

    # ----------------------------------------------------------------- #
    #                               The grid                            #
    # ----------------------------------------------------------------- #

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """``(depths,) + lateral_shape``, the depths those of the padded
        mesh."""
        return (self._basis.family.mesh.nglob,) + self._transform.shape

    @property
    def depths(self) -> np.ndarray:
        """The nodes of the padded mesh in depth: the first axis of a
        vector."""
        return self._basis.family.mesh.rglob

    @property
    def interior_depths(self) -> np.ndarray:
        """The nodes in depth of the domain proper, both ends included."""
        return self._basis.r

    @property
    def lateral_axes(self) -> tuple[np.ndarray, ...]:
        """The coordinates of the lateral grid along each axis, from the lower
        edge of the period: the remaining axes of a vector."""
        return self._transform.grid_axes

    @property
    def x(self) -> np.ndarray:
        """The grid's coordinates along the first lateral axis."""
        return self._transform.grid_axes[0]

    @cached_property
    def _interior_slices(self) -> tuple[slice, ...]:
        """The domain proper as a slice of each axis of the grid: its nodes
        in depth, and along each lateral axis the run of grid points
        between the bounds."""
        mask = self._transform.mask
        slices = [self._basis.restriction.nodes]
        for axis in range(mask.ndim):
            others = tuple(a for a in range(mask.ndim) if a != axis)
            inside = np.flatnonzero(mask.any(axis=others))
            slices.append(
                slice(int(inside[0]), int(inside[-1]) + 1)
                if inside.size
                else slice(0, 0)
            )
        return tuple(slices)

    @cached_property
    def interior_mask(self) -> np.ndarray:
        """A boolean grid array, true on the domain and false on the padding,
        in depth and across."""
        mask = np.zeros(self.grid_shape, dtype=bool)
        mask[self._basis.restriction.nodes] = self._transform.mask
        return mask

    @property
    def _interior_index(self) -> tuple[slice, ...]:
        return self._interior_slices

    def _lateral_samples(
        self, function: Callable[[tuple[float, ...]], float], depth: float, /
    ) -> np.ndarray:
        """A function's values across the lateral grid at one depth, called
        at points of the domain proper alone: a grid point in the lateral
        padding takes the value at the nearest point of the domain."""
        axes, inverses = [], []
        for values, (low, high) in zip(
            self._transform.grid_axes, self._transform.domain.bounds
        ):
            unique, inverse = np.unique(np.clip(values, low, high), return_inverse=True)
            axes.append(unique)
            inverses.append(inverse)
        sampled = np.empty(tuple(axis.size for axis in axes))
        for index in np.ndindex(sampled.shape):
            point = (depth,) + tuple(float(axis[i]) for axis, i in zip(axes, index))
            sampled[index] = float(function(point))
        return sampled[np.ix_(*inverses)]

    def project_function(
        self,
        function: Callable[[tuple[float, ...]], float],
        /,
        *,
        extension: str = "fit",
    ) -> np.ndarray:
        """The field of this space that a function of position names.

        The function is never called outside the domain, where it need not
        be defined. In depth, ``"fit"``, the default, asks nothing of the
        padding: it returns the field in the span of the kept modes that
        best matches the function on the domain, Fourier component by
        component in the domain's own quadrature; it is what a mean, a
        reference model or a synthetic truth wants. ``"constant"`` and
        ``"odd"`` return the function's own values on the domain's part of
        the grid and continue them in depth across the padding, by the
        value at the nearer end or by ``2 f(end) - f(mirror)``; see
        :meth:`~pygeoinf.spectral_element.AxisSpace.project_function`.

        Across a lateral padding, where there is one, a grid point takes the
        function's value at the nearest point of the domain. Under ``"fit"``
        and ``"odd"`` that is rolled off to zero at the far edge of the
        padding by a raised cosine, so that the periodic field has no step
        where the two sides of the domain meet; under ``"constant"`` it is
        left as it is, which keeps a positive function positive on the
        padding, as a standard deviation handed to ``pointwise_std=`` must
        be.

        Args:
            function: called with a point ``(depth, x)`` of the domain, or
                ``(depth, x, y)`` with two lateral axes.
            extension: ``"fit"``, ``"odd"`` or ``"constant"``.

        Returns:
            The field. Under ``"fit"`` it is in the span of the kept modes;
            otherwise it holds the function's values on the domain's grid
            points.

        Raises:
            ValueError: for an extension that is none of these.
        """
        levels: dict[float, np.ndarray] = {}

        def sample(depths: np.ndarray) -> np.ndarray:
            for depth in np.unique(depths):
                if float(depth) not in levels:
                    levels[float(depth)] = self._lateral_samples(function, float(depth))
            return np.stack([levels[float(depth)] for depth in depths])

        if extension not in _EXTENSIONS:
            raise ValueError("extension must be 'odd', 'constant' or 'fit'.")
        if extension == "fit":
            return self._fitted(sample(self.interior_depths) * self._transform.taper)
        values = _sampled_with_extension(self.depths, *self._depths, sample, extension)
        return values if extension == "constant" else values * self._transform.taper

    def _fitted(self, values: np.ndarray, /) -> np.ndarray:
        """The field in the span of the kept modes that best matches, in
        depth, values given on the domain's levels of the grid."""
        functions = self._transform.analyse_grid(values)  # (levels, components)
        basis = self._basis
        weights = basis[0].weights()
        nodes = basis.restriction.nodes
        components = np.zeros(self.dim)
        for i, modes in enumerate(basis):
            fitted = _fitted_over_the_domain(
                modes.modes()[nodes], weights, functions[:, basis.components(i)]
            )
            components[self._entries[i]] = fitted.T
        return _checked_fit(self.from_components(components), self.interior_mask)

    def random_point(self, *, rng: Generator | None = None) -> np.ndarray:
        """A point ``(depth, x)`` or ``(depth, x, y)`` of the domain proper,
        drawn uniformly over it."""
        rng = resolve_rng(rng)
        ranges = (self._depths,) + self._transform.domain.bounds
        return np.array([float(rng.uniform(low, high)) for low, high in ranges])

    # ----------------------------------------------------------------- #
    #                        Integrals and points                       #
    # ----------------------------------------------------------------- #

    def _lateral_integrals(self) -> np.ndarray:
        """The integral of each lateral basis function over the lateral
        domain proper, in closed form: of ``exp(i theta_k)`` axis by axis,
        its real part for a cosine and minus its imaginary part for a sine,
        times the function's amplitude."""
        transform = self._transform
        domain = transform.domain
        integral = np.ones(transform.size, dtype=complex)
        for axis, ((low, high), origin, length) in enumerate(
            zip(domain.bounds, domain.origin, domain.lengths)
        ):
            rate = 2.0 * np.pi * transform.wavevectors[axis] / length
            safe = np.where(rate == 0.0, 1.0, rate)
            waves = (
                np.exp(1j * rate * (high - origin)) - np.exp(1j * rate * (low - origin))
            ) / (1j * safe)
            integral *= np.where(rate == 0.0, high - low, waves)
        amplitude = 1.0 / np.sqrt(transform.volume * transform.packing.measures)
        return amplitude * np.where(
            transform.phases == 0, integral.real, -integral.imag
        )

    def integral_functional(self) -> LinearFunctional:
        """The integral of a field over the domain proper.

        Each Fourier component's integral across the lateral domain, known in
        closed form, times the integral in depth of its coefficient function,
        taken by the quadrature of the domain's own elements so that the
        padding contributes nothing. Where the lateral domain is the whole
        period only the constant component has an integral. Applied to a
        grid array that is not in the span of the kept modes, such as the
        pointwise product of two fields, it reads the array through its
        components, that is the projection onto the modes, as every operator
        on the space does so that adjoints hold under composition; the exact
        integral of such an array is a quadrature the caller takes.
        """
        basis = self._basis
        weights = basis[0].weights()
        nodes = basis.restriction.nodes
        lateral = self._lateral_integrals()
        components = np.empty(self.dim)
        for i, modes in enumerate(basis):
            components[self._entries[i]] = (
                lateral[basis.components(i)][:, None]
                * (weights @ modes.modes()[nodes])[None, :]
            )
        return LinearFunctional.from_derivative_components(self, components)

    def _checked_points(self, points: Sequence[Sequence[float]], /) -> np.ndarray:
        """The points as an array ``(points, 1 + lateral axes)``, depth
        first.

        Along a lateral axis with padding the domain is bounded, and a point
        past its bounds is refused; along one without, the field is periodic
        and a point is read at its image in the period.

        Raises:
            ValueError: for a lateral coordinate outside a bounded domain.
        """
        points = np.asarray(points, dtype=float).reshape(-1, 1 + self._lateral_axes)
        domain = self._transform.domain
        for axis, ((low, high), pad) in enumerate(zip(domain.bounds, domain.padding)):
            if pad == 0.0:
                continue
            values = points[:, 1 + axis]
            slack = _POSITION_RTOL * (high - low)
            outside = (values < low - slack) | (values > high + slack)
            if np.any(outside):
                raise ValueError(
                    f"Every point must lie in the domain, in [{low:g}, {high:g}] "
                    f"along lateral axis {axis}; one is at {values[outside][0]:g}."
                )
        return points

    def basis_matrix(self, points: Sequence[Sequence[float]], /) -> np.ndarray:
        """The basis at many points ``(depth, x)`` or ``(depth, x, y)``, as a
        ``(len(points), dim)`` array: each lateral Fourier function at the
        point times each mode of depth of its wavenumber at the depth, the
        latter exact through the polynomial it is on each element.

        Raises:
            ValueError: if a depth lies outside the domain, or a lateral
                coordinate outside a bounded one.
        """
        points = self._checked_points(points)
        lateral = self._transform.basis(points[:, 1:])
        basis = self._basis
        matrix = np.empty((points.shape[0], self.dim))
        for i, modes in enumerate(basis):
            products = (
                lateral[:, basis.components(i), None]
                * modes.evaluate(points[:, 0])[:, None, :]
            )
            matrix[:, self._entries[i].reshape(-1)] = products.reshape(
                points.shape[0], -1
            )
        return matrix

    def _local_evaluation(
        self, points: Sequence[Sequence[float]], /
    ) -> LocalEvaluation:
        """Evaluation at the points through the nodes of their elements in
        depth and the lateral Fourier functions at their lateral positions.

        The field of components ``c`` has the coefficient function ``f_n =
        Phi_n c_n`` for each Fourier component ``n``, ``Phi_n`` the modes of
        depth of its wavenumber at every node of the mesh, and its value at
        a point is ``sum_n e_n(x) (L f_n)``, ``L`` the Lagrange weights of the
        point's depth on the nodes of its element. The transpose spreads
        ``y_i e_n(x_i)`` onto those nodes, giving ``H_n = L^T (e_n y)``, and
        the derivative components are ``Phi_n^T H_n``, component by
        component. The lateral functions are summed directly, every one at
        every point.

        Raises:
            ValueError: if a depth lies outside the domain, or a lateral
                coordinate outside a bounded one.
        """
        points = self._checked_points(points)
        lateral = points[:, 1:]
        transform = self._transform

        def angular(which: np.ndarray) -> np.ndarray:
            return transform.basis(lateral[which]).T

        return LocalEvaluation(
            self._basis.family.mesh,
            points[:, 0],
            self._basis.restriction.interval,
            self._nodal_functions,
            self._nodal_projection,
            angular=angular,
            count=transform.size,
        )

    def _nodal_functions(self, c: np.ndarray, /) -> np.ndarray:
        """The coefficient function of every Fourier component at every node
        of the mesh in depth, from components: shape ``(Fourier components,
        nodes)``, in the transform's order."""
        return self._basis.synthesise(np.asarray(c, dtype=float), physical=False)

    def _nodal_projection(self, functions: np.ndarray, /) -> np.ndarray:
        """The transpose of :meth:`_nodal_functions`: each component's
        function against the modes of depth of its wavenumber, summed over
        every node."""
        basis = self._basis
        out = np.empty(self.dim)
        for i, modes in enumerate(basis):
            out[self._entries[i]] = functions[basis.components(i)] @ modes.modes()
        return out

    def point_evaluation_operator(
        self,
        points: Sequence[Any],
        /,
        *,
        route: Literal["auto", "dense", "local", "nufft"] = "auto",
        unsafe: bool = False,
        epsilon: float = 1e-12,
    ) -> LinearOperator:
        """Evaluation at several points, as an operator into a Euclidean
        space.

        ``"dense"`` and ``"local"`` are those of every spectral-element
        space; see :meth:`SpectralElementSpace.point_evaluation_operator`.
        Both sum the lateral Fourier functions directly, every one at every
        point, which costs the lateral grid's size per point, in each
        application under ``"local"`` once the functions at the points no
        longer fit 256 MB.

        ``"nufft"`` is matrix-free with the lateral sums by non-uniform
        FFT: one batched transform per element of the mesh in depth that
        holds a point, of cost the lateral grid's and the points' rather
        than their product, and nothing kept per point but its element and
        Lagrange weights. It is the route for many points on a fine
        lateral grid. Its values agree with the others' to ``epsilon`` of
        the field's size and not to rounding, and its adjoint is the
        transpose of its action to the same accuracy.

        ``"auto"``, the default, is the dense route while its matrix,
        eight bytes per point per component, fits 256 MB, and ``"nufft"``
        beyond. While the matrix fits, a product with it is the quickest
        application and exact to rounding; beyond, the non-uniform
        transforms are quicker than the direct sums of ``"local"``, which
        ``"auto"`` never chooses here.

        Args:
            points: where to evaluate, each ``(depth, x)`` or ``(depth, x,
                y)``.
            route: ``"auto"``, ``"dense"``, ``"local"`` or ``"nufft"``.
            unsafe: build it even on a space too rough to admit it.
            epsilon: the relative accuracy asked of the non-uniform
                transforms, under ``"nufft"``.

        Returns:
            The operator.

        Raises:
            ValueError: if no points are given, one lies outside the domain,
                the route is none of the four, or the order is at or below
                half the spatial dimension and *unsafe* is not set.
        """
        points = list(points)
        if route == "auto" and points:
            fits = 8 * len(points) * self.dim <= _base._DENSE_EVALUATION_LIMIT
            route = "dense" if fits else "nufft"
        if route != "nufft":
            return super().point_evaluation_operator(points, route=route, unsafe=unsafe)
        if not points:
            raise ValueError("At least one point is needed.")
        self._require_point_evaluation(
            "A point evaluation operator", points=points, unsafe=unsafe
        )
        evaluation = _NufftEvaluation(
            self, self._checked_points(points), epsilon=epsilon
        )
        return _local_operator(self, EuclideanSpace(len(points)), evaluation)

    def _squared_modes(self, weights: np.ndarray, /) -> np.ndarray:
        """``sum_k weights_k phi_k(p)^2`` on the grid.

        When the weights do not tell the Fourier components of one ``|k|^2``
        apart, which is so for every function of ``A``, the lateral sum is
        done in closed form: on the grid every component has the mean square
        ``1 / V`` over the period ``V``, a cosine and its sine summing to
        that at each point, so the answer is a function of depth. Off the
        grid that holds of every component but the cosines at the Nyquist
        wavenumber of an axis of even size, whose square is not constant
        there. Otherwise the lateral functions are squared on the grid.
        """
        basis = self._basis
        transform = self._transform
        shape = self.grid_shape
        per_basis = []
        isotropic = True
        for i, modes in enumerate(basis):
            block = weights[self._entries[i]]
            isotropic = isotropic and bool(np.all(block == block[0]))
            per_basis.append((modes.modes() ** 2) @ block.T)  # (nodes, components)
        if isotropic:
            profile = sum(
                count * squares[:, 0]
                for count, squares in zip(basis.multiplicity, per_basis)
            )
            profile = profile / transform.volume
            return np.broadcast_to(
                profile.reshape((-1,) + (1,) * (len(shape) - 1)), shape
            ).copy()
        in_depth = np.empty((shape[0], transform.size))
        for i, squares in enumerate(per_basis):
            in_depth[:, basis.components(i)] = squares
        out = np.zeros(shape)
        step = max(1, _SQUARES_ENTRIES // transform.size)
        for start in range(0, transform.size, step):
            stop = min(start + step, transform.size)
            units = np.zeros((stop - start, transform.size))
            units[np.arange(stop - start), np.arange(start, stop)] = 1.0
            lateral = transform.synthesise_grid(units) ** 2
            out += np.tensordot(in_depth[:, start:stop], lateral, axes=(1, 0))
        return out

    def _radial_groups(self) -> list[_RadialGroup]:
        """One group per distinct ``|k|^2``: the Fourier components that
        share it share its modes of depth, scattered through the components
        as the transform orders them, and the squares of their lateral
        functions sum on the grid to their number over the period."""
        basis = self._basis
        return [
            _RadialGroup.scattered(
                modes,
                self._entries[i],
                float(basis.multiplicity[i]) / self._transform.volume,
            )
            for i, modes in enumerate(basis)
        ]

    def _scaled_family(
        self,
        radial: Callable[[np.ndarray], np.ndarray],
        lateral: Callable[[np.ndarray], np.ndarray] | None,
        probe: np.ndarray,
        /,
    ) -> SeparableOperatorFamily:
        """``A'`` on this space's mesh in depth, its coefficient in depth
        ``radial^2`` and its lateral one ``lateral^2``. Under
        ``boundary="robin"`` its Robin coefficients are matched to the scale
        in depth at the ends of the domain, ``probe[0]`` and ``probe[-1]``,
        as this space's are to its own."""
        robin = self._robin
        if self._matched_boundary:
            robin = _resolved_boundary("robin", probe, None)
        return SeparableOperatorFamily(
            self._basis.family.mesh,
            kappa=lambda z: radial(z) ** 2,
            kappa_h=None if lateral is None else (lambda z: lateral(z) ** 2),
            weight="one",
            lateral="one",
            index="value",
            robin=robin,
        )


# --------------------------------------------------------------------------- #
#                         Flat layers stacked in depth                        #
# --------------------------------------------------------------------------- #

#: The side of an interface as the shared :class:`Layered` names it, along
#: its coordinate, for each side named in depth. Depth increases downward, so
#: the side above an interface, the shallower, is the side of the smaller
#: coordinate, which the shared class calls ``"below"``.
_SIDE_ALONG_THE_COORDINATE = {"above": "below", "below": "above"}


def _require_one_lateral_grid(layers: Sequence[FlatSpace], /) -> None:
    """Refuse flat layers that do not all have the first one's lateral grid
    and lateral box.

    Raises:
        ValueError: naming the first layer whose grid or box differs.
    """
    first = layers[0]
    for i, layer in enumerate(layers[1:], start=1):
        if layer.lateral_shape != first.lateral_shape:
            raise ValueError(
                f"The layers must share one lateral grid: layer {i} has the "
                f"shape {layer.lateral_shape} and layer 0 {first.lateral_shape}."
            )
        if layer.lateral_domain != first.lateral_domain:
            raise ValueError(
                f"The layers must share one lateral box: layer {i} has "
                f"{layer.lateral_domain!r} and layer 0 {first.lateral_domain!r}."
            )


class FlatLayered(Layered):
    """Piecewise-continuous fields in flat layers stacked in depth: a
    :class:`FlatSpace` in each, all over one lateral box and one lateral grid.

    What :class:`pygeoinf.strip.Layered` and :class:`pygeoinf.slab.Layered`
    share, each naming the class of its layers as ``_layer_kind``. To the
    shared :class:`~pygeoinf.spectral_element.Layered` it adds three things.
    The sides of an interface are named in depth, which increases downward:
    ``"above"`` is the shallower side, the layer that comes first, and
    ``"below"`` the deeper. The jump across an interface, a horizontal line
    or plane, is taken at given lateral positions. And point evaluation has
    the flat layers' route by non-uniform FFT, which ``"auto"`` takes where
    the dense matrix would not fit.
    """

    #: The class of the layers, which each geometry's own class names; the
    #: shared class refuses flat layers to a class that does not name theirs.
    _layer_kind: type

    def __init__(
        self,
        boundaries: Sequence[float],
        shape: int | Sequence[int],
        /,
        *,
        lateral: LateralDomain | None = None,
        labels: Sequence[str] | None = None,
        **options: Any,
    ) -> None:
        """
        Args:
            boundaries: the depths bounding the layers, ascending: the top
                of the first layer, the interfaces in order of depth, and the
                base of the last.
            shape: the points of the lateral grid along each axis, one
                number for a single axis. The same for every layer, and never
                read layer by layer.
            lateral: the lateral box, the same for every layer: the periods
                from the origin, or a ``planetmodel.fourier.PaddedDomain``
                for a bounded domain inside a wider period. A unit period on
                every axis if omitted.
            labels: names for the layers, so one can be reached by name.
            **options: the other keyword arguments of the layers' class --
                ``order``, ``length_scale``, ``depth_modes``,
                ``max_eigenvalue``, ``padding``, ``boundary``, ``ngll``,
                ``element_length`` -- each given once for all layers or, as a
                list, layer by layer. A tuple, such as a padding pair, is one
                value.

        Raises:
            ValueError: for fewer than two boundaries, a list of the wrong
                length, or what the layers' class refuses.
        """
        # One lateral grid and box for every layer: a list of numbers is one
        # shape or one box here, and not a value for each layer.
        if isinstance(shape, list):
            shape = tuple(shape)
        if isinstance(lateral, list):
            lateral = tuple(lateral)
        super().__init__(
            layers_between(
                self._layer_kind,
                boundaries,
                shape,
                ("top", "bottom"),
                {**options, "lateral": lateral},
            ),
            labels=labels,
        )

    @classmethod
    def from_layers(
        cls,
        layers: Sequence[SpectralElementSpace],
        /,
        *,
        labels: Sequence[str] | None = None,
    ) -> "FlatLayered":
        """A layered space of this class over flat layers built by hand.

        Args:
            layers: the layers, in order of depth, all of this class's
                geometry and all on one lateral grid and box.
            labels: names for the layers.

        Returns:
            The layered space, an instance of this class.

        Raises:
            TypeError: if the layers are not all of this class's geometry.
            ValueError: if there are none, they are out of order or overlap,
                or they do not share one lateral grid and box: the same
                shape, and the same bounds and padding.
        """
        space = super().from_layers(layers, labels=labels)
        _require_one_lateral_grid(space.layers)
        return space

    def layer_index(self, point: Any, /, *, side: str | None = None) -> int:
        """Which layer a point belongs to.

        Args:
            point: ``(depth, x)`` in a strip or ``(depth, x, y)`` in a slab,
                of which only the depth is read.
            side: for a point on an interface, which has two layers:
                ``"above"``, the shallower, which comes first among
                :attr:`layers`, or ``"below"``, the deeper. Ignored anywhere
                else.

        Returns:
            The layer's position among :attr:`layers`.

        Raises:
            ValueError: if the point is in no layer -- above the first, below
                the last, or in a gap -- or on an interface with no side
                named, or *side* is not ``"above"``, ``"below"`` or ``None``.
        """
        return super().layer_index(
            point, side=_SIDE_ALONG_THE_COORDINATE.get(side, side)
        )

    def point_evaluation_operator(
        self,
        points: Sequence[Any],
        /,
        *,
        side: str | None = None,
        route: Literal["auto", "dense", "local", "nufft"] = "auto",
        unsafe: bool = False,
        epsilon: float = 1e-12,
    ) -> LinearOperator:
        """Evaluation at points anywhere in the layers, as an operator into a
        Euclidean space.

        By the layers' own routes, each at the points in its layer; see
        :meth:`FlatSpace.point_evaluation_operator`. ``"dense"`` assembles
        :meth:`basis_matrix`, eight bytes per point per component of the
        whole. ``"local"`` sums the lateral Fourier functions directly at
        each point. ``"nufft"`` takes the lateral sums by non-uniform FFT,
        and agrees with the others to ``epsilon`` of the field's size and
        not to rounding. ``"auto"``, the default, is the dense route while
        its matrix fits 256 MB, and ``"nufft"`` beyond: never ``"local"``,
        as in a single layer.

        Args:
            points: where to evaluate, each ``(depth, x)`` or ``(depth, x,
                y)``.
            side: which value is meant at the points on an interface:
                ``"above"``, the shallower layer's, or ``"below"``, the
                deeper's.
            route: ``"auto"``, ``"dense"``, ``"local"`` or ``"nufft"``.
            unsafe: build it even in layers too rough to admit it.
            epsilon: the relative accuracy asked of the non-uniform
                transforms, under ``"nufft"``.

        Returns:
            The operator.

        Raises:
            ValueError: if no points are given, one is in no layer, on an
                interface with no side named or outside a bounded lateral
                domain, the route is none of the four, or a layer's order is
                too low and *unsafe* is not set.
        """
        points = list(points)
        if route == "auto" and points:
            fits = 8 * len(points) * self.dim <= _base._DENSE_EVALUATION_LIMIT
            route = "dense" if fits else "nufft"
        if route != "nufft":
            return super().point_evaluation_operator(
                points, side=side, route=route, unsafe=unsafe
            )
        if not points:
            raise ValueError("At least one point is needed.")
        self._require_point_evaluation(
            "A point evaluation operator", points=points, side=side, unsafe=unsafe
        )
        pieces = [
            (
                self.columns(i),
                where,
                _NufftEvaluation(
                    self.layers[i],
                    self.layers[i]._checked_points(inside),
                    epsilon=epsilon,
                ),
            )
            for i, where, inside in self._grouped(points, side)
        ]
        return _local_operator(
            self, EuclideanSpace(len(points)), _LayeredEvaluation(pieces, self.dim)
        )

    def jump_operator(
        self,
        interface: int,
        /,
        *,
        positions: Sequence[Any] | None = None,
        unsafe: bool = False,
    ) -> LinearOperator:
        """The jump of the field across an interface: its value just above,
        less its value just below, at given lateral positions.

        Depth increases downward, so this is the value in the shallower
        layer less the value in the deeper, at the interface's depth: the
        sign a ball gives the same interface, outer less inner. The
        interface is a horizontal line of a strip or a horizontal plane of a
        slab, and the jump is taken at points of it named by their lateral
        coordinates.

        What a reflection coefficient sees, and what a prior on a
        discontinuity is a statement about. Conditioning on its vanishing is
        how two layers are made continuous at the positions named. The
        layers share their lateral grid, so the jump is a lateral Fourier
        series with as many terms as the grid has points; conditioned to
        vanish at every point of the grid, it vanishes at every position
        across.

        Args:
            interface: which interface, by position among :attr:`interfaces`,
                in order of depth.
            positions: the lateral positions at which the jump is taken:
                ``x`` coordinates along a strip, ``(x, y)`` pairs in a slab.
            unsafe: build it even in layers too rough to admit it.

        Returns:
            The operator into a Euclidean space of one dimension per
            position. It claims to be onto while there are no more positions
            than points of the lateral grid, the lateral Fourier functions,
            which holds for positions that tell those functions apart, such
            as distinct points of the grid. Positions repeated, or in a slab
            more on one line than the grid has points along it, make
            dependent rows, on which an exact conditioning is singular.

        Raises:
            ValueError: if no positions are given, a slab's are not pairs,
                one lies outside a bounded lateral domain, or a layer's
                order is too low and *unsafe* is not set.
            IndexError: if there is no such interface.
        """
        depth = self.interfaces[interface]
        if positions is None:
            raise ValueError(
                "Across a flat interface the jump is taken at given lateral "
                "positions: pass positions=."
            )
        axes = self.spatial_dimension - 1
        lateral = np.asarray(positions, dtype=float)
        if axes == 1:
            lateral = lateral.reshape(-1, 1)
        elif lateral.ndim != 2 or lateral.shape[1] != axes:
            raise ValueError(
                f"The positions in a slab are (x, y) pairs; got an array of "
                f"shape {lateral.shape}."
            )
        if lateral.shape[0] == 0:
            raise ValueError("At least one position is needed.")
        points = [(depth, *row) for row in lateral]
        for side in ("above", "below"):
            self._require_point_evaluation(
                "A jump operator", points=points, side=side, unsafe=unsafe
            )
        matrix = self.basis_matrix(points, side="above") - self.basis_matrix(
            points, side="below"
        )
        # The jump at a position is the lateral Fourier functions there
        # against each one's jump in depth, and those are independent, each
        # coefficient being free in both layers. So the rows are as
        # independent as the functions at the positions, which more
        # positions than functions cannot be.
        onto = len(points) <= self.layers[0].transform.size
        return LinearOperator.from_matrix(
            self,
            EuclideanSpace(len(points)),
            matrix,
            form="galerkin",
            traits=Traits.SURJECTIVE if onto else Traits.NONE,
        )
