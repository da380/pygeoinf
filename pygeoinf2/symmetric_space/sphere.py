"""
Fields on a two-sphere, through spherical harmonics.

The one space in the family that is genuinely its own implementation: the
harmonic transform is not an FFT and there is no way to obtain it from the
periodic box. Everything downstream of the transform is shared, though — the
Laplacian is diagonal, invariant operators are
:class:`~pygeoinf2.algebra.diagonal.DiagonalLinearOperator`, and invariant
measures come from :class:`~pygeoinf2.symmetric_space.base.SymmetricSpace` — so
what is written here is the transform, the spectrum and the geometry, and
nothing else.

Requires ``pyshtools``, which is an optional dependency. The import is deferred
to construction so that importing this module costs nothing when the sphere is
not used.

Conventions, all pinned by test rather than assumed: coefficients are
orthonormal (``norm=4`` in pyshtools' low-level interface) with the
Condon-Shortley phase *excluded* (pyshtools spells that ``csphase=1``), the
grid is Driscoll-Healy with the sampling given at construction, and **a point
is a ``(latitude, longitude)`` pair in degrees**.

That last is the convention of every station catalogue, every earthquake
catalogue, pyshtools itself, and every script written against v1. The
trigonometry inside is done in colatitude and radians, and each method converts
at its own boundary; :meth:`Sphere.to_colatitude_radians` and
:meth:`Sphere.to_latitude_degrees` are the conversion, public because anyone
working below the boundary needs it.

An *angle* on the sphere is likewise in degrees — the half-angle of a cap, for
instance. A *distance* is in the units of :attr:`Sphere.radius`, and every
argument that takes one says so.
"""

from __future__ import annotations

from functools import cached_property, lru_cache
from typing import Any, Callable, Hashable, Sequence

import numpy as np
from numpy.random import Generator

from ..algebra.operators import LinearFunctional, LinearOperator
from .base import SymmetricSpace, _distribute

__all__ = ["Sphere", "Lebesgue", "Sobolev"]

# pyshtools' low-level normalisation code for orthonormal harmonics.
_ORTHONORMAL = 4
# One basis matrix at a time is held in memory; this bounds its entry count,
# so a chunk is about 30 MB whatever the truncation.
_CHUNK_ENTRIES = 4_000_000

# When the transform route wins. Its cost is fixed in the number of points --
# one analysis, one FFT, one NUFFT -- while the direct sum costs one basis
# evaluation per point per component, so the crossover is a threshold on both.
#
# Retuned once the NUFFT stopped running on every core (see Sphere.evaluate).
# It was 512 and 512, measured when finufft's default threading was making the
# transform 4 to 20 times slower than it needed to be, which pushed the
# crossover out. Measured again at lmax 4, 8, 16, 32 and 64 across six point
# counts: the transform now wins from about 200 points once the dimension is a
# few hundred, and the old thresholds were leaving a factor of 13 on the table
# at lmax 64 and 1000 points (34 ms direct against 2.6 ms). Below dim 256 the
# two are within a millisecond of each other either way, so the dimension
# threshold is there to avoid paying the fixed cost for nothing.
# See DESIGN.md 21.15.
_TRANSFORM_MIN_POINTS = 200
_TRANSFORM_MIN_DIM = 256

# pyshtools spells "leave the Condon-Shortley phase out" as csphase=1.
_NO_CONDON_SHORTLEY = 1


# Grid weights, keyed on what they actually depend on: the truncation, the
# longitude sampling and the radius -- but not the Sobolev order, which is the
# whole point, since with_order is what makes new spaces in a hot loop. A plain
# dict rather than an lru_cache because the value is computed from an instance.
_QUADRATURES: dict[tuple[int, int, float], np.ndarray] = {}


@lru_cache(maxsize=8)
def _packing_for(lmax: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The component ordering for a truncation, built once per ``lmax``.

    Module-level rather than per instance because a space is cheap to make and
    these are not: ``with_order`` and ``with_degree`` both produce new spaces
    over the same harmonics, and a `multiplication_operator` makes one on every
    call.

    Returned read-only, since the whole point is that callers share it.
    """
    degrees = np.repeat(np.arange(lmax + 1), 2 * np.arange(lmax + 1) + 1)
    # Within a degree: cosines for m = 0..l, then sines for m = 1..l.
    position = np.arange(degrees.size) - degrees**2
    cosine = position <= degrees
    parts = np.where(cosine, 0, 1)
    orders = np.where(cosine, position, position - degrees)

    tables = (parts, degrees, orders)
    for table in tables:
        table.flags.writeable = False
    return tables


@lru_cache(maxsize=8)
def _legendre_indices_for(lmax: int) -> np.ndarray:
    """Where each component sits in pyshtools' packed Legendre array.

    ``PlmIndex(l, m)`` is ``l (l + 1) / 2 + m``, verified against pyshtools
    itself. Calling it per component instead made this a Python loop of length
    ``dim``: 212 ms at ``lmax`` 256, against 2 ms here.
    """
    _, degrees, orders = _packing_for(lmax)
    indices = degrees * (degrees + 1) // 2 + orders
    indices.flags.writeable = False
    return indices


def _require_pyshtools() -> object:
    """Import pyshtools, with a message that says what to install."""
    try:
        import pyshtools
    except ImportError as error:  # pragma: no cover - depends on the install
        raise ImportError(
            "Spherical harmonic spaces need pyshtools, which is an optional "
            "dependency. Install it with the 'sphere' extra."
        ) from error
    return pyshtools


class Sphere(SymmetricSpace[Any]):
    """A field on a sphere, expanded in spherical harmonics.

    Vectors are ``pyshtools.SHGrid`` objects on a Driscoll-Healy grid, not bare
    arrays. That a vector may be any object at all is a point of the library,
    and here it earns something concrete: a field arrives able to plot, rotate
    and expand itself through the library that produced it, and it is the type
    every caller with an existing pyshtools workflow already holds. The values
    are reachable as ``x.data``, or through
    :meth:`~pygeoinf2.symmetric_space.base.SymmetricSpace.grid_values`.

    Components are the real harmonic coefficients, scaled so that the Lebesgue
    basis is orthonormal on a sphere of the given radius.

    The grid has ``n == 2 (lmax + 1)`` rows and ``sampling * n`` columns.
    ``sampling`` defaults to 1, the square grid, which is pyshtools' default
    and halves the memory of every field and the cost of every pointwise
    operation against the rectangular one.
    """

    def __init__(
        self,
        lmax: int,
        /,
        *,
        radius: float = 1.0,
        order: float = 0.0,
        length_scale: float = 1.0,
        sampling: int = 1,
    ) -> None:
        """
        Args:
            lmax: the maximum spherical harmonic degree.
            radius: the sphere's radius, which sets both the area element and
                the Laplacian's eigenvalues.
            order: the Sobolev order. Zero gives the Lebesgue space.
            length_scale: the length at which the Sobolev weight turns over.
                Named in full because ``scale`` is the vector-scaling
                operation.
            sampling: grid columns per row, 1 or 2. One gives the square
                Driscoll-Healy grid, which is pyshtools' default and this
                one; two doubles the longitude resolution, and the memory.
                Both represent the same functions exactly -- the transforms
                cost the same and the components are identical -- so this is a
                statement about the grid, not about the space.

        Raises:
            ValueError: if lmax is negative, the radius or length scale is not
                positive, or the sampling is not 1 or 2.
        """
        if lmax < 0:
            raise ValueError("lmax must be non-negative.")
        if radius <= 0.0:
            raise ValueError("radius must be positive.")
        if length_scale <= 0.0:
            raise ValueError("length_scale must be positive.")
        if sampling not in (1, 2):
            raise ValueError(f"sampling is 1 or 2, got {sampling}.")
        _require_pyshtools()

        self._lmax = int(lmax)
        self._radius = float(radius)
        self._order = float(order)
        self._length_scale = float(length_scale)
        self._sampling = int(sampling)
        self._latitudes = 2 * (self._lmax + 1)

        degrees = self._degree_of_component
        eigenvalues = degrees * (degrees + 1.0) / self._radius**2
        self._laplacian_eigenvalues = eigenvalues
        metric = (
            np.ones(eigenvalues.size)
            if order == 0.0
            else (1.0 + self._length_scale**2 * eigenvalues) ** self._order
        )
        super().__init__(metric)

    # ----------------------------------------------------------------- #
    #                              Structure                            #
    # ----------------------------------------------------------------- #

    @property
    def lmax(self) -> int:
        """The maximum harmonic degree."""
        return self._lmax

    @property
    def radius(self) -> float:
        """The sphere's radius."""
        return self._radius

    @property
    def order(self) -> float:
        """The Sobolev order. Zero for a Lebesgue space."""
        return self._order

    @property
    def length_scale(self) -> float:
        """The Sobolev length scale."""
        return self._length_scale

    @property
    def grid_shape(self) -> tuple[int, int]:
        """The shape of a field's values: latitudes by longitudes."""
        return (self._latitudes, self._sampling * self._latitudes)

    @property
    def sampling(self) -> int:
        """Grid columns per row: 1 for the square grid, 2 for the wide one."""
        return self._sampling

    @property
    def area(self) -> float:
        """The sphere's surface area."""
        return 4.0 * np.pi * self._radius**2

    @property
    def degrees(self) -> np.ndarray:
        """The harmonic degree of each component."""
        return self._packing[1]

    @property
    def spatial_dimension(self) -> int:
        """Two: a sphere is a two-dimensional surface."""
        return 2

    @property
    def gaussian_curvature(self) -> float:
        """``1 / radius^2``, constant over the sphere."""
        return 1.0 / self._radius**2

    @property
    def laplacian_eigenvalues(self) -> np.ndarray:
        """``l (l + 1) / radius^2`` for each component."""
        return self._laplacian_eigenvalues

    def _key(self) -> Hashable:
        return (
            self._lmax,
            self._radius,
            self._order,
            self._length_scale,
            self._sampling,
        )

    def _coordinate_key(self) -> Hashable:
        """The grid, which the order and length scale do not touch.

        Tagged by geometry rather than by ``type(self)``: ``Lebesgue`` and
        ``Sobolev`` are thin subclasses over the same grid, and keying on the
        concrete class would say two views of one field were different fields.
        """
        return ("sphere", self._lmax, self._radius, self._sampling)

    def __repr__(self) -> str:
        kind = "Lebesgue" if self._order == 0.0 else f"Sobolev(order={self._order})"
        return f"Sphere(lmax={self._lmax}, radius={self._radius}, {kind})"

    # ----------------------------------------------------------------- #
    #                          Component packing                        #
    # ----------------------------------------------------------------- #

    @cached_property
    def _packing(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The flat ordering of the ``(2, lmax+1, lmax+1)`` coefficient array.

        Degree by degree: the cosine coefficients for ``m`` from zero to ``l``,
        then the sine coefficients for ``m`` from one to ``l``. That is
        ``2l + 1`` per degree and ``(lmax + 1)^2`` in all, which is the
        dimension of the space. Shared across instances, as it depends only on
        ``lmax``.
        """
        return _packing_for(self._lmax)

    @cached_property
    def _azimuthal(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Which components take a cosine, which a sine, and at which order."""
        parts, _, orders = self._packing
        cosine = np.flatnonzero(parts == 0)
        sine = np.flatnonzero(parts == 1)
        return cosine, sine, orders[cosine], orders[sine]

    @cached_property
    def _legendre_indices(self) -> np.ndarray:
        """Where each component sits in pyshtools' packed Legendre array.

        Depends only on ``lmax``, so it is shared between every space of that
        truncation rather than rebuilt per instance -- ``with_order`` makes a
        new space, and this was 212 ms of it at ``lmax`` 256, on every call.
        """
        return _legendre_indices_for(self._lmax)

    @property
    def _degree_of_component(self) -> np.ndarray:
        """The harmonic degree attached to each component."""
        return self._packing[1].astype(float)

    # ----------------------------------------------------------------- #
    #                        Vectors, which are grids                   #
    # ----------------------------------------------------------------- #

    def grid_values(self, x: Any, /) -> np.ndarray:
        """The field's values, reaching past the ``SHGrid`` wrapper.

        Args:
            x: a field of this space.

        Returns:
            The grid values, of shape :attr:`grid_shape`. Not a copy: this is
            the grid's own array.

        A bare array is accepted too, and returned as it is. That is a
        deliberate kindness at the boundary — grid values arrive from all
        sorts of places — and not an invitation to treat the two as
        interchangeable: :meth:`from_components` and everything downstream of
        it produce ``SHGrid``.
        """
        if isinstance(x, np.ndarray):
            return np.asarray(x, dtype=float)
        return np.asarray(x.data, dtype=float)

    def from_grid_values(self, values: np.ndarray, /) -> Any:
        """A field holding the given values.

        Args:
            values: an array of shape :attr:`grid_shape`.

        Returns:
            An ``SHGrid`` over those values.

        Raises:
            ValueError: if the shape is wrong.
        """
        from pyshtools import SHGrid

        array = np.asarray(values, dtype=float)
        if array.shape != self.grid_shape:
            raise ValueError(f"A field has shape {self.grid_shape}, got {array.shape}.")
        return SHGrid.from_array(array, grid="DH", copy=False)

    def copy(self, x: Any) -> Any:
        """An independent copy of the field."""
        return self.from_grid_values(self.grid_values(x).copy())

    def axpy(self, a: float, x: Any, y: Any) -> Any:
        """``y += a x``, in place on the grid's own array."""
        y.data += a * self.grid_values(x)
        return y

    def scale_inplace(self, a: float, x: Any) -> Any:
        """``x *= a``, in place."""
        x.data *= a
        return x

    def to_components(self, x: Any) -> np.ndarray:
        """Harmonic coefficients of a field, orthonormal in ``L2``.

        Args:
            x: a field, as an ``SHGrid`` or a bare array of grid values.

        Returns:
            One coefficient per component.

        Raises:
            ValueError: if the grid is not this sphere's shape.
        """
        from pyshtools.expand import SHExpandDH

        field = self.grid_values(x)
        if field.shape != self.grid_shape:
            raise ValueError(f"A field has shape {self.grid_shape}, got {field.shape}.")
        coefficients = SHExpandDH(
            field,
            norm=_ORTHONORMAL,
            sampling=self._sampling,
            csphase=_NO_CONDON_SHORTLEY,
            lmax_calc=self._lmax,
        )
        parts, degrees, orders = self._packing
        return self._radius * coefficients[parts, degrees, orders]

    def from_components(self, c: np.ndarray) -> np.ndarray:
        """The field with the given harmonic coefficients.

        Args:
            c: one coefficient per component.

        Returns:
            The field, as an ``SHGrid``.

        Raises:
            ValueError: if the component count is not the dimension.
        """
        from pyshtools.expand import MakeGridDH

        components = np.asarray(c, dtype=float)
        if components.shape != (self.dim,):
            raise ValueError(f"Expected {self.dim} components, got {components.shape}.")
        coefficients = np.zeros((2, self._lmax + 1, self._lmax + 1))
        parts, degrees, orders = self._packing
        coefficients[parts, degrees, orders] = components / self._radius
        return self.from_grid_values(
            MakeGridDH(
                coefficients,
                norm=_ORTHONORMAL,
                sampling=self._sampling,
                csphase=_NO_CONDON_SHORTLEY,
                lmax=self._lmax,
            )
        )

    @staticmethod
    def truncation_degree_for(
        order: float,
        length_scale: float,
        /,
        *,
        radius: float = 1.0,
        rtol: float = 1.0e-8,
        power_of_two: bool = False,
    ) -> int:
        """The degree at which a Sobolev spectrum has run out.

        For choosing a truncation from the prior rather than by habit, and
        *before* there is a space to ask -- which is the point of it being
        static: the answer is what you pass to the constructor.

        The rule is v1's, so it gives v1's numbers: sum the Sobolev weights
        mode by mode and stop when the newest term is a fraction ``rtol`` of
        the running total.

        Mode by mode, note, and not weighted by the ``2l + 1`` modes a degree
        holds. :meth:`~pygeoinf2.symmetric_space.base.SymmetricSpace.estimate_truncation_degree`
        answers the weighted question, which is the one about the field's
        energy, and the two are far apart where the spectrum decays slowly: at
        order 1.5 and length scale 0.5 this returns 721 and the weighted rule
        wants 13833. The weighted rule is the honest one for power and this one
        is the usable one for a grid.

        Args:
            order: the Sobolev order. Must exceed one -- below that the total
                power does not converge and no truncation is enough.
            length_scale: the length at which the weight turns over.
            radius: the sphere's radius, in the same units.
            rtol: the relative tolerance to stop at.
            power_of_two: round up to a power of two, which some transforms
                prefer.

        Returns:
            The degree.

        Raises:
            ValueError: if the order is not above one, or the tolerance is not
                in ``(0, 1)``.
            RuntimeError: if it has not converged by degree 10000.
        """
        if order <= 1.0:
            raise ValueError(
                f"The order must exceed one for the power to converge, got "
                f"{order}."
            )
        if not 0.0 < rtol < 1.0:
            raise ValueError(f"The tolerance lies in (0, 1), got {rtol}.")
        if length_scale <= 0.0 or radius <= 0.0:
            raise ValueError("The length scale and radius must be positive.")

        scale = (length_scale / radius) ** 2
        total, degree, relative = 1.0, 0, 1.0
        while relative > rtol:
            degree += 1
            term = (1.0 + scale * degree * (degree + 1)) ** -order
            total += term
            relative = term / total
            if degree > 10000:
                raise RuntimeError(
                    "No truncation below degree 10000 reaches this tolerance."
                )

        if power_of_two:
            degree = 2 ** (int(np.log2(degree)) + 1)
        return degree

    def to_coefficients(self, x: Any, /) -> Any:
        """The field's coefficients as a pyshtools ``SHCoeffs``.

        The other half of the seam that :meth:`grid_values` opens: components
        are this library's packed real vector, and this is the object
        pyshtools' own plotting, rotation and spectrum routines take. Anyone
        with an existing pyshtools workflow, or writing pyslfp-style interop,
        wants this and not the packed vector.

        These are pyshtools' orthonormal coefficients, with no Condon-Shortley
        phase, on their own scale. The relation to components is a single
        factor: a component is ``radius`` times the coefficient it comes from,
        read through the packing, and :meth:`from_coefficients` inverts this
        exactly.

        Args:
            x: a field on this sphere.

        Returns:
            An ``SHCoeffs`` of degree ``lmax``.

        Raises:
            ValueError: if the grid is not this sphere's shape.
        """
        from pyshtools import SHCoeffs
        from pyshtools.expand import SHExpandDH

        field = self.grid_values(x)
        if field.shape != self.grid_shape:
            raise ValueError(f"A field has shape {self.grid_shape}, got {field.shape}.")
        return SHCoeffs.from_array(
            SHExpandDH(
                field,
                norm=_ORTHONORMAL,
                sampling=self._sampling,
                csphase=_NO_CONDON_SHORTLEY,
                lmax_calc=self._lmax,
            ),
            normalization="ortho",
            csphase=_NO_CONDON_SHORTLEY,
        )

    def from_coefficients(self, coefficients: Any, /) -> Any:
        """The field with the given ``SHCoeffs``.

        The inverse of :meth:`to_coefficients`. Coefficients above this space's
        ``lmax`` are dropped and missing ones taken as zero, so a set expanded
        elsewhere can be read in without matching degrees up first.

        Args:
            coefficients: an ``SHCoeffs``, or a ``(2, l + 1, l + 1)`` array in
                pyshtools' orthonormal convention.

        Returns:
            A field on this sphere.

        Raises:
            ValueError: if the array is not shaped ``(2, l + 1, l + 1)``.
        """
        array = np.asarray(getattr(coefficients, "coeffs", coefficients), dtype=float)
        if array.ndim != 3 or array.shape[0] != 2 or array.shape[1] != array.shape[2]:
            raise ValueError(
                f"Coefficients have shape (2, l + 1, l + 1), got {array.shape}."
            )
        padded = np.zeros((2, self._lmax + 1, self._lmax + 1))
        shared = min(array.shape[1], self._lmax + 1)
        padded[:, :shared, :shared] = array[:, :shared, :shared]
        return self.from_components(self._radius * padded[self._packing])

    # ----------------------------------------------------------------- #
    #                          Points and grids                         #
    # ----------------------------------------------------------------- #

    @cached_property
    def colatitudes(self) -> np.ndarray:
        """The grid colatitudes, in radians, from the pole downwards."""
        return np.arange(self._latitudes) * np.pi / self._latitudes

    @cached_property
    def longitudes(self) -> np.ndarray:
        """The grid longitudes, in radians."""
        columns = self._sampling * self._latitudes
        return np.arange(columns) * 2.0 * np.pi / columns

    def basis_at(self, point: Any, /) -> np.ndarray:
        """The value of each orthonormal harmonic at a point.

        Args:
            point: a ``(latitude, longitude)`` pair in degrees.
        """
        from pyshtools.legendre import PlmON

        position = self._radians(point)
        colatitude, longitude = float(position[0]), float(position[1])

        legendre = PlmON(self._lmax, np.cos(colatitude), csphase=_NO_CONDON_SHORTLEY)
        parts, _, orders = self._packing
        indices = self._legendre_indices
        angle = orders * longitude
        harmonics = legendre[indices] * np.where(
            parts == 0, np.cos(angle), np.sin(angle)
        )
        # Components carry a factor of the radius, so the dual basis carries
        # its reciprocal: the basis functions are orthonormal on this sphere,
        # not on the unit one.
        return harmonics / self._radius

    def basis_matrix(self, points: Sequence[Any], /) -> np.ndarray:
        """The basis evaluated at many points, as a ``(len(points), dim)`` array.

        The batched form of :meth:`basis_at`, and the reason both
        :meth:`evaluate` and :meth:`accumulate` are usable at the scale a real
        acquisition geometry reaches. Calling ``basis_at`` in a loop spends
        almost all of its time in per-point Python: at ``lmax == 64`` the
        Legendre evaluation itself is 3% of it, and the other 97% is indexing
        and trigonometry that vectorises.

        Rows are ordered as ``points``, columns as the components.
        """
        from pyshtools.legendre import PlmON

        positions = self.to_colatitude_radians(np.asarray(list(points), dtype=float))

        indices = self._legendre_indices
        count = positions.shape[0]
        result = np.empty((count, indices.size))
        for row, colatitude in enumerate(positions[:, 0]):
            table = PlmON(self._lmax, np.cos(colatitude), csphase=_NO_CONDON_SHORTLEY)
            result[row] = table[indices]

        # The azimuthal factor depends only on the order m, of which there are
        # lmax + 1 values rather than dim. Computing cos(m phi) and sin(m phi)
        # once per order and gathering is the difference between a few hundred
        # thousand trigonometric evaluations and tens of millions.
        degrees = np.arange(self._lmax + 1)
        angles = positions[:, 1][:, None] * degrees[None, :]
        cosine, sine = np.cos(angles), np.sin(angles)
        cosine_columns, sine_columns, cosine_orders, sine_orders = self._azimuthal
        result[:, cosine_columns] *= cosine[:, cosine_orders]
        result[:, sine_columns] *= sine[:, sine_orders]
        return result / self._radius

    def _in_chunks(self, points: Sequence[Any], /) -> Any:
        """Split points so one basis matrix at a time stays a sensible size."""
        points = tuple(points)
        per_chunk = max(1, _CHUNK_ENTRIES // max(self.dim, 1))
        for start in range(0, len(points), per_chunk):
            yield points[start : start + per_chunk]

    # ----------------------------------------------------------------- #
    #                    The double Fourier sphere                      #
    # ----------------------------------------------------------------- #

    @cached_property
    def _quadrature(self) -> np.ndarray:
        """The transform's own quadrature weight for each grid row.

        The *shape* is Driscoll and Healy's own latitudinal weighting, which
        is what ``SHExpandDH`` uses, so it is the transform's weighting by
        construction rather than by coincidence. The *scale* is then fixed by
        probing the transform on a single row, as
        :meth:`_quadrature_from_transform` does on every row.

        One probe rather than none, because the scale is not free: the pole
        correction in :meth:`_synthesis_adjoint` carries no weight of its own,
        so a common factor on these does not cancel against it. And one rather
        than a closed-form constant, because that constant is pyshtools'
        normalisation convention, which is exactly the kind of thing this
        class should not be asserting from memory.

        One probe rather than ``rows`` of them is the whole saving: 2.9 ms
        against 583 ms at ``lmax`` 128, and 18 ms against 8.0 s at 256.

        **The pole row's weight is zero**, since the grid samples colatitude on
        ``[0, pi)`` and the quadrature gives the pole no area. Anything sitting
        there is invisible to the transform and has to be added back by hand.
        """
        from pyshtools.utils import DHaj

        # Shared between spaces that would compute the same weights. The
        # Sobolev order is not among them -- the weights are the transform's,
        # and the transform does not know the metric -- so with_order(0.0),
        # which every multiplication_operator call makes, gets them free.
        key = (self._lmax, self._sampling, self._radius)
        cached = _QUADRATURES.get(key)
        if cached is not None:
            return cached

        rows, columns = self.grid_shape
        shape = DHaj(rows)

        # Calibrate against the transform itself, on the first row that
        # carries any weight at all.
        row = int(np.flatnonzero(shape)[0])
        indicator = np.zeros(self.grid_shape)
        indicator[row] = 1.0
        reference = float(self.basis_at(self.reference_point)[0])
        measured = self.to_components(indicator)[0] / (columns * reference)
        weights = shape * (measured / shape[row])
        weights.flags.writeable = False
        _QUADRATURES[key] = weights
        return weights

    def _quadrature_from_transform(self) -> np.ndarray:
        """The same weights, read off the transform instead of a formula.

        :meth:`to_components` is a quadrature, so ``to_components(e_jk)`` is a
        row-dependent multiple of ``basis_at(p_jk)`` — verifiably so, the ratio
        being constant across every component. Reading that multiple off gives
        the weights from the transform that will be used with them, with no
        appeal to a convention.

        Kept as the check on :meth:`_quadrature`, not as the way to get them:
        it costs one full analysis transform per grid row.
        """
        rows, columns = self.grid_shape
        # Component zero is the constant mode, so the point is arbitrary.
        reference = float(self.basis_at(self.reference_point)[0])
        weights = np.empty(rows)
        for row in range(rows):
            indicator = np.zeros(self.grid_shape)
            indicator[row] = 1.0
            weights[row] = self.to_components(indicator)[0] / (columns * reference)
        return weights

    @cached_property
    def _south_pole_basis(self) -> np.ndarray:
        """The basis at colatitude ``pi``, which the grid does not sample."""
        return self.basis_at(np.array([-90.0, 0.0]))

    def _synthesis_adjoint(self, values: np.ndarray, /) -> np.ndarray:
        """``sum_jk v_jk phi(p_jk)``: the transpose of synthesis onto the grid.

        Not the analysis, which is the *inverse*. The two differ by the
        quadrature weights, and by the pole row that the quadrature drops
        entirely.
        """
        weights = self._quadrature
        live = weights > 0.0
        scaled = np.zeros(self.grid_shape)
        scaled[live] = values[live] / weights[live, None]
        total = self.to_components(self.from_grid_values(scaled))
        for row in np.flatnonzero(~live):
            total = total + values[row].sum() * self.basis_at(
                np.array([90.0 - np.degrees(float(self.colatitudes[row])), 0.0])
            )
        return total

    def _double(self, x: Any, /) -> np.ndarray:
        """The double-Fourier-sphere extension, sampled on a doubled grid.

        ``g(theta, phi) == f(theta, phi)`` on ``[0, pi]`` and
        ``g(2 pi - theta, phi) == f(theta, phi + pi)`` beyond it. For a
        band-limited ``f`` the result is a *trigonometric polynomial* on the
        torus, which is what makes one FFT of it exact.
        """
        rows, columns = self.grid_shape
        field = self.grid_values(x)
        doubled = np.empty((2 * rows, columns))
        doubled[:rows] = field
        # theta == pi is the south pole: not a grid row, so it is evaluated.
        doubled[rows] = float(self._south_pole_basis @ self.to_components(x))
        doubled[rows + 1 :] = np.roll(field[1:][::-1], columns // 2, axis=1)
        return doubled

    def _double_adjoint(self, doubled: np.ndarray, /) -> np.ndarray:
        """The transpose of :meth:`_double`, in components."""
        rows, columns = self.grid_shape
        folded = np.array(doubled[:rows], dtype=float)
        folded[1:] += np.roll(doubled[rows + 1 :][::-1], columns // 2, axis=1)
        return self._synthesis_adjoint(folded) + doubled[rows].sum() * (
            self._south_pole_basis
        )

    def _angles(self, points: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
        """Points as two contiguous arrays of radians, for the NUFFT."""
        positions = self.to_colatitude_radians(np.asarray(list(points), dtype=float))
        return (
            np.ascontiguousarray(positions[:, 0]),
            np.ascontiguousarray(positions[:, 1]),
        )

    def _use_transform(self, count: int) -> bool:
        """Whether the transform route is cheaper than summing the basis.

        The transform costs a fixed amount — one analysis, one FFT, one NUFFT —
        while the direct sum costs one basis evaluation per point per
        component. So both sides of the crossover matter: a large truncation
        makes the direct sum expensive per point, and a small one makes the
        transform's fixed cost not worth paying however many points there are.
        """
        try:
            import finufft  # noqa: F401
        except ImportError:  # pragma: no cover - depends on the install
            return False
        return count >= _TRANSFORM_MIN_POINTS and self.dim >= _TRANSFORM_MIN_DIM

    def evaluate(
        self,
        x: np.ndarray,
        points: Sequence[Any],
        /,
        *,
        eps: float = 1.0e-10,
        nthreads: int = 1,
    ) -> np.ndarray:
        """Field values at scattered points.

        Above a size threshold this goes through the double Fourier sphere and
        a type-2 non-uniform FFT, which costs ``O(dim log dim + n)`` against the
        direct sum's ``O(n dim)``. Below it, the fixed cost of the transforms is
        not worth paying and the basis is summed directly.

        Args:
            x: a field of this space.
            points: ``(latitude, longitude)`` pairs in degrees.
            eps: the NUFFT's requested accuracy, when it is used.
            nthreads: threads for the NUFFT. One by default, and measured:
                finufft's own default is every core, and on a 16-core machine
                that is 4 to 20 times *slower* here than a single thread --
                4.4 ms against 21 ms at ``lmax`` 128, at every point count from
                200 to 10000. These transforms are too small to pay for the
                threading, and a call inside a ``joblib`` loop would
                oversubscribe on top of that. Pass zero for finufft's default.
        """
        points = tuple(points)
        if not self._use_transform(len(points)):
            components = self.to_components(x)
            return np.concatenate(
                [
                    self.basis_matrix(chunk) @ components
                    for chunk in self._in_chunks(points)
                ]
            )

        import finufft

        rows, columns = self.grid_shape
        coefficients = np.fft.fft2(self._double(x)) / (2 * rows * columns)
        colatitudes, longitudes = self._angles(points)
        values = finufft.nufft2d2(
            colatitudes,
            longitudes,
            np.ascontiguousarray(np.fft.fftshift(coefficients)),
            isign=+1,
            eps=eps,
            nthreads=nthreads,
        )
        return np.ascontiguousarray(np.atleast_1d(values).real)

    def accumulate(
        self,
        weights: np.ndarray,
        points: Sequence[Any],
        /,
        *,
        eps: float = 1.0e-10,
        nthreads: int = 1,
    ) -> np.ndarray:
        """The derivative components of ``x -> sum_i y_i x(r_i)``.

        The transpose of :meth:`evaluate`, step for step: a type-1 NUFFT where
        that used a type-2, the same FFT — the discrete Fourier matrix is
        symmetric, so it is its own transpose — and then the transposes of the
        fold and of synthesis onto the grid.

        ``eps`` and ``nthreads`` mean what they do in :meth:`evaluate`, and
        Args:
            weights: one per point.
            points: ``(latitude, longitude)`` pairs in degrees.
            eps: the NUFFT's accuracy, as in :meth:`evaluate`.
            nthreads: threads for it, one by default and for the reason
                measured there.

        Returns:
            The derivative components.

        Raises:
            ValueError: if the weight count does not match the points.
        """
        points = tuple(points)
        values = np.asarray(weights, dtype=float)
        if values.size != len(points):
            raise ValueError(f"Got {values.size} weights for {len(points)} points.")

        if not self._use_transform(len(points)):
            total = np.zeros(self.dim)
            offset = 0
            for chunk in self._in_chunks(points):
                end = offset + len(chunk)
                total += self.basis_matrix(chunk).T @ values[offset:end]
                offset = end
            return total

        import finufft

        rows, columns = self.grid_shape
        colatitudes, longitudes = self._angles(points)
        spectrum = finufft.nufft2d1(
            colatitudes,
            longitudes,
            np.ascontiguousarray(values.astype(complex)),
            (2 * rows, columns),
            isign=+1,
            eps=eps,
            nthreads=nthreads,
        )
        doubled = np.fft.fft2(np.fft.ifftshift(spectrum)) / (2 * rows * columns)
        return self._double_adjoint(doubled.real)

    def project_function(self, function: Callable[[Any], float], /) -> np.ndarray:
        """Sample a function on the grid.

        The function receives a ``(latitude, longitude)`` pair in degrees.
        """
        colatitudes, longitudes = np.meshgrid(
            self.colatitudes, self.longitudes, indexing="ij"
        )
        latitudes = 90.0 - np.degrees(colatitudes.ravel())
        azimuths = (np.degrees(longitudes.ravel()) + 180.0) % 360.0 - 180.0
        values = np.array(
            [
                float(function(np.array([latitude, azimuth])))
                for latitude, azimuth in zip(latitudes, azimuths)
            ]
        )
        return self.from_grid_values(values.reshape(self.grid_shape))

    @property
    def reference_point(self) -> np.ndarray:
        """The north pole, as ``(latitude, longitude)`` in degrees.

        Any point would do; the sphere is homogeneous.
        """
        return np.array([90.0, 0.0])

    def random_point(self, *, rng: Generator | None = None) -> np.ndarray:
        """A point drawn uniformly over the sphere's area.

        Uniform in ``cos(colatitude)``, not in colatitude: sampling the angle
        uniformly would crowd the poles, which is the classic way to bias a
        set of station locations.
        """
        generator = np.random.default_rng() if rng is None else rng
        return np.array(
            [
                float(np.degrees(np.arcsin(generator.uniform(-1.0, 1.0)))),
                float(generator.uniform(-180.0, 180.0)),
            ]
        )

    def walk_from(self, point: Any, distances: np.ndarray, /) -> list[np.ndarray]:
        """Points at given distances from a point, along a meridian.

        Any direction would do, the sphere being homogeneous *and* isotropic;
        a meridian is the one that needs no tangent frame.

        **The walk continues past the pole**, which is where this used to go
        wrong: adding the angle to the colatitude and stopping there returned
        latitudes below -90 -- a distance of ``3 pi R / 2`` from the equator
        gave latitude -250 -- and the two evaluation routes then read that as
        two different points. The direct sum takes the colatitude at face value
        and lands at ``(2 pi - theta, phi)``; the non-uniform FFT sees the
        doubled grid, on which the same colatitude is ``(2 pi - theta,
        phi + pi)``. On a non-zonal field they disagreed by 1.02 on a field of
        maximum 0.47. Passing the pole reflects the meridian to the far side,
        which is what walking over a pole does, and both routes then agree.

        Args:
            point: where to start, ``(latitude, longitude)`` in degrees.
            distances: how far to walk, as *physical* distances. Negative
                distances walk the other way, over the north pole.

        Returns:
            One ``(latitude, longitude)`` point per distance.
        """
        position = self._radians(point)
        angles = np.asarray(distances, dtype=float) / self._radius
        colatitudes = (position[0] + angles) % (2.0 * np.pi)
        # Past a pole the meridian continues down the far side: reflect the
        # colatitude back into [0, pi] and turn the longitude through pi.
        beyond = colatitudes > np.pi
        colatitudes = np.where(beyond, 2.0 * np.pi - colatitudes, colatitudes)
        longitudes = np.where(beyond, position[1] + np.pi, position[1])
        return list(
            self.to_latitude_degrees(np.column_stack([colatitudes, longitudes]))
        )

    # ----------------------------------------------------------------- #
    #                              Geometry                             #
    # ----------------------------------------------------------------- #

    @staticmethod
    def to_colatitude_radians(points: Any, /) -> np.ndarray:
        """``(latitude, longitude)`` in degrees to ``(colatitude, longitude)``
        in radians.

        The public form of the conversion every method here does at its
        boundary. Points are given in degrees, because that is what a catalogue
        of stations or earthquakes holds, what pyshtools uses, and what every
        script written against v1 passes; the trigonometry inside is done in
        colatitude and radians, because that is what the spherical harmonics
        are written in.

        Args:
            points: one ``(latitude, longitude)`` pair or a sequence of them.

        Returns:
            An ``(n, 2)`` array of ``(colatitude, longitude)`` in radians.

        Raises:
            ValueError: if a point does not have two coordinates, or a
                latitude lies outside ``[-90, 90]`` -- which is the usual sign
                that colatitudes have been passed in by mistake.
        """
        positions = np.atleast_2d(np.asarray(points, dtype=float))
        if positions.shape[-1] != 2:
            raise ValueError(
                f"Points are (latitude, longitude) pairs in degrees, got an "
                f"array of shape {np.shape(points)}."
            )
        latitudes = positions[:, 0]
        # The check the docstring has always promised and never performed. It
        # is what catches colatitudes passed in by mistake, and it is what
        # would have caught walk_from returning -250 (§3.4). The tolerance is
        # for a latitude that came back from an arcsine as 90 + 1e-14; a
        # genuine error is degrees out, not nanodegrees.
        outside = np.abs(latitudes) > 90.0 + 1.0e-9
        if np.any(outside):
            worst = float(latitudes[np.argmax(np.abs(latitudes))])
            raise ValueError(
                f"A latitude lies in [-90, 90] degrees, got {worst}. Points "
                f"are (latitude, longitude) pairs in degrees; this is the "
                f"usual sign that colatitudes, or radians, have been passed "
                f"in instead."
            )
        return np.column_stack(
            [
                np.radians(90.0 - np.clip(latitudes, -90.0, 90.0)),
                np.radians(positions[:, 1]) % (2.0 * np.pi),
            ]
        )

    @staticmethod
    def to_latitude_degrees(points: Any, /) -> np.ndarray:
        """``(colatitude, longitude)`` in radians to ``(latitude, longitude)``
        in degrees. The inverse of :meth:`to_colatitude_radians`.

        Args:
            points: one ``(colatitude, longitude)`` pair or a sequence of them.

        Returns:
            An ``(n, 2)`` array of ``(latitude, longitude)`` in degrees, with
            longitude in ``[-180, 180)``.
        """
        positions = np.atleast_2d(np.asarray(points, dtype=float))
        return np.column_stack(
            [
                90.0 - np.degrees(positions[:, 0]),
                (np.degrees(positions[:, 1]) + 180.0) % 360.0 - 180.0,
            ]
        )

    def _radians(self, point: Any, /) -> np.ndarray:
        """One point, as ``(colatitude, longitude)`` in radians."""
        converted = self.to_colatitude_radians(point)
        if converted.shape != (1, 2):
            raise ValueError(
                f"Expected one (latitude, longitude) pair, got "
                f"{converted.shape[0]}."
            )
        return converted[0]

    @staticmethod
    def _to_vector(point: Any) -> np.ndarray:
        """A ``(latitude, longitude)`` pair in degrees as a unit vector."""
        colatitude, longitude = Sphere.to_colatitude_radians(point)[0]
        sine = np.sin(colatitude)
        return np.array(
            [
                sine * np.cos(longitude),
                sine * np.sin(longitude),
                np.cos(colatitude),
            ]
        )

    @staticmethod
    def _to_point(vector: np.ndarray) -> np.ndarray:
        """A unit vector as a ``(latitude, longitude)`` pair in degrees."""
        unit = np.asarray(vector, dtype=float)
        unit = unit / np.linalg.norm(unit)
        return Sphere.to_latitude_degrees(
            np.array(
                [
                    float(np.arccos(np.clip(unit[2], -1.0, 1.0))),
                    float(np.arctan2(unit[1], unit[0]) % (2.0 * np.pi)),
                ]
            )
        )[0]

    @staticmethod
    def _tangent_frame(centre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """An orthonormal basis for the tangent plane at a unit vector."""
        reference = (
            np.array([0.0, 0.0, 1.0])
            if abs(centre[2]) < 0.9
            else np.array([1.0, 0.0, 0.0])
        )
        first = np.cross(reference, centre)
        first /= np.linalg.norm(first)
        second = np.cross(centre, first)
        return first, second / np.linalg.norm(second)

    def geodesic_distance(self, start: Any, end: Any, /) -> float:
        """Great-circle distance, in the same units as :attr:`radius`.

        Computed as ``atan2(|u x v|, u . v)`` rather than ``acos(u . v)``.
        The two agree in exact arithmetic; in floating point the arccosine
        loses about half its digits for nearby points, because the cosine is
        flat there. At a separation of ``1e-6`` radians its relative error is
        around ``4e-5``, and it does not return exactly zero for a point
        against itself. Nearby points are the case a localised covariance is
        entirely about.
        """
        first, second = self._to_vector(start), self._to_vector(end)
        angle = np.arctan2(
            np.linalg.norm(np.cross(first, second)), np.dot(first, second)
        )
        return float(self._radius * angle)

    def geodesic_quadrature(
        self, start: Any, end: Any, /, *, count: int
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Gauss-Legendre nodes and weights along a great-circle arc.

        Weights carry the arc-length element and so sum to the distance between
        the endpoints. Antipodal endpoints are rejected rather than resolved:
        the great circle through them is not unique, and picking one silently
        would be a wrong answer rather than a missing one.

        Args:
            start: one endpoint, ``(latitude, longitude)`` in degrees.
            end: the other.
            count: how many Gauss-Legendre nodes to place along the arc.

        Returns:
            The nodes and their weights, which sum to the arc length.

        Raises:
            ValueError: for fewer than one node, or antipodal endpoints.
        """
        if count < 1:
            raise ValueError("A quadrature rule needs at least one node.")
        first, second = self._to_vector(start), self._to_vector(end)
        # atan2 of the cross and dot products, as in geodesic_distance and for
        # the same reason: arccos of a dot product loses half its digits for
        # nearby points, which is exactly where the short paths are.
        angle = float(
            np.arctan2(np.linalg.norm(np.cross(first, second)), np.dot(first, second))
        )

        if angle < 1.0e-12:
            return [np.asarray(start, dtype=float)] * count, np.zeros(count)
        if abs(angle - np.pi) < 1.0e-10:
            raise ValueError(
                "The endpoints are antipodal, so the great circle through "
                "them is not unique."
            )

        arc_length = self._radius * angle
        abscissae, weights = np.polynomial.legendre.leggauss(count)
        parameters = 0.5 * (abscissae + 1.0)
        sine = np.sin(angle)
        nodes = [
            self._to_point(
                (np.sin((1.0 - s) * angle) * first + np.sin(s * angle) * second) / sine
            )
            for s in parameters
        ]
        return nodes, weights * (0.5 * arc_length)

    def geodesic_ball_quadrature(
        self, centre: Any, radius: float, /, *, count: int
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Nodes and weights integrating over a spherical cap.

        Gauss-Legendre in ``cos(gamma)`` — which is the variable the area
        element is uniform in — and a trapezoidal rule in azimuth on each ring.
        The weights carry the area element, so they sum to the cap's area.

        Args:
            centre: the cap's centre, in degrees.
            radius: its *physical* radius along the sphere, not an angle.
            count: how many nodes to place.

        Returns:
            The nodes and their weights.

        Raises:
            ValueError: for a non-positive radius or count, or a radius
                larger than half the circumference.
        """
        if count < 1:
            raise ValueError("A quadrature rule needs at least one node.")
        if radius < 0.0 or radius > np.pi * self._radius:
            raise ValueError(
                "A geodesic ball on the sphere has radius in "
                f"[0, pi * {self._radius}], got {radius}."
            )
        if radius == 0.0:
            return [np.asarray(centre, dtype=float)] * count, np.zeros(count)

        angular_radius = radius / self._radius
        cosine = np.cos(angular_radius)
        ring_count = min(count, max(1, int(np.sqrt(count))))
        abscissae, weights = np.polynomial.legendre.leggauss(ring_count)
        half_width = 0.5 * (1.0 - cosine)
        heights = half_width * (abscissae + 1.0) + cosine
        ring_weights = half_width * weights
        ring_radii = np.sqrt(np.clip(1.0 - heights**2, 0.0, None))
        counts = _distribute(count, ring_radii)

        centre_vector = self._to_vector(centre)
        first, second = self._tangent_frame(centre_vector)

        nodes: list[np.ndarray] = []
        node_weights: list[float] = []
        for height, weight, ring_radius, points_here in zip(
            heights, ring_weights, ring_radii, counts
        ):
            azimuths = 2.0 * np.pi * np.arange(points_here, dtype=float) / points_here
            vectors = height * centre_vector[None, :] + ring_radius * (
                np.cos(azimuths)[:, None] * first[None, :]
                + np.sin(azimuths)[:, None] * second[None, :]
            )
            nodes.extend(self._to_point(vector) for vector in vectors)
            share = self._radius**2 * 2.0 * np.pi * weight / points_here
            node_weights.extend([share] * points_here)

        return nodes, np.asarray(node_weights)

    # ----------------------------------------------------------------- #
    #                          Cap averages                             #
    # ----------------------------------------------------------------- #

    def spherical_cap_integral(
        self, centre: Any, angular_radius: float, /, *, normalise: bool = False
    ) -> LinearFunctional:
        """The exact integral over a spherical cap, as a functional.

        Exact in the truncated basis rather than quadrature-approximate: the
        cap indicator's own harmonic coefficients are known in closed form, so
        the functional's derivative components are read off directly. That is
        both cheaper and more accurate than
        :meth:`geodesic_ball_quadrature`, which remains the fallback for
        anything that is not a cap.

        Args:
            centre: the cap's centre, as ``(latitude, longitude)`` in degrees.
            angular_radius: the cap's half-angle, **in degrees**. An angle, not
                a distance: :meth:`geodesic_ball_quadrature` takes a *physical*
                radius, in the same units as :attr:`radius`, and the two are
                related by the radius of the sphere.
            normalise: divide by the cap's area, giving the average rather than
                the integral.

        Returns:
            The functional.

        Raises:
            ValueError: if the angular radius is outside ``[0, 180]``, or if an
                average over a cap of zero area is asked for.
        """
        if angular_radius < 0.0 or angular_radius > 180.0:
            raise ValueError(
                f"A cap's angular radius lies in [0, 180] degrees, got "
                f"{angular_radius}."
            )
        area_fraction = 0.5 * (1.0 - np.cos(np.radians(angular_radius)))
        if area_fraction <= 0.0:
            if normalise:
                raise ValueError("A cap of zero area has no average.")
            return LinearFunctional.from_derivative_components(self, np.zeros(self.dim))

        from pyshtools import SHCoeffs

        position = np.atleast_2d(np.asarray(centre, dtype=float))[0]
        cap = SHCoeffs.from_cap(
            float(angular_radius),
            self._lmax,
            clat=float(position[0]),
            clon=float(position[1]),
            normalization="ortho",
            csphase=_NO_CONDON_SHORTLEY,
            kind="real",
            degrees=True,
        )
        parts, degrees, orders = self._packing
        coefficients = cap.to_array(lmax=self._lmax)[parts, degrees, orders]

        # from_cap normalises the indicator to global average one. Undo that,
        # then either keep the physical integral or divide by the cap area.
        components = coefficients / (self._radius * 4.0 * np.pi)
        if not normalise:
            components = components * self.area * area_fraction
        return LinearFunctional.from_derivative_components(self, components)

    def spherical_cap_average(
        self, centre: Any, angular_radius: float, /
    ) -> LinearFunctional:
        """The exact average over a spherical cap, as a functional.

        Args:
            centre: the cap's centre, as ``(latitude, longitude)`` in degrees.
            angular_radius: the cap's half-angle, in degrees.

        Returns:
            The functional.
        """
        return self.spherical_cap_integral(centre, angular_radius, normalise=True)

    def geodesic_ball_average_operator(
        self,
        centres: Sequence[Any],
        radius: float,
        /,
        *,
        count: int | None = None,
        normalise: bool = True,
        dense: bool = False,
    ) -> LinearOperator:
        """Cap averages, exactly, as an operator into a Euclidean space.

        The property operator of a spherical inference problem. Uses the exact
        cap functionals unless ``count`` is given, which forces the generic
        quadrature route and exists so the two can be checked against each
        other.

        Args:
            centres: the cap centres, in degrees.
            radius: the *physical* cap radius, not an angle.
            count: nodes per cap for the quadrature route. Ignored on the
                exact harmonic route.
            normalise: divide by the cap's area, giving an average rather
                than an integral.
            dense: assemble the derivative matrix rather than staying
                matrix-free.

        Returns:
            The operator.

        Raises:
            ValueError: for a non-positive radius or count, or no centres.
        """
        centres = tuple(centres)
        if not centres:
            raise ValueError("At least one centre is needed.")
        if count is not None:
            return super().geodesic_ball_average_operator(
                centres, radius, count=count, normalise=normalise, dense=dense
            )

        from ..algebra.spaces import EuclideanSpace

        # `radius` here is a *physical* distance, in the units of self.radius;
        # spherical_cap_integral takes the half-angle in degrees.
        angular_radius = np.degrees(radius / self._radius)
        rows = np.stack(
            [
                self.spherical_cap_integral(
                    centre, angular_radius, normalise=normalise
                ).derivative_components
                for centre in centres
            ]
        )
        return LinearOperator.from_matrix(
            self, EuclideanSpace(len(centres)), rows, form="galerkin"
        )

    # ----------------------------------------------------------------- #
    #                          Coefficients                             #
    # ----------------------------------------------------------------- #

    # ----------------------------------------------------------------- #
    #                       Acquisition geometry                        #
    # ----------------------------------------------------------------- #

    def stations(
        self, /, *, count: int | None = None, rng: Generator | None = None
    ) -> list[np.ndarray]:
        """Global Seismograph Network stations, as points on this sphere.

        A real network, so the coverage is the real coverage: dense in North
        America and Europe, thin over the southern oceans. Uniformly scattered
        receivers make an inverse problem easier than any real one, which is
        exactly the wrong direction for a test to err in.

        Args:
            count: how many to return, drawn without replacement. All of them
                if omitted.
            rng: the generator for that draw.

        Returns:
            ``(latitude, longitude)`` points in degrees.
        """
        table = _read_table("gsn_stations.csv")
        points = _as_points(table["Latitude"], table["Longitude"])
        return _subsample(points, count, rng)

    def earthquakes(
        self,
        /,
        *,
        count: int | None = None,
        minimum_magnitude: float = 0.0,
        rng: Generator | None = None,
    ) -> list[np.ndarray]:
        """Earthquake epicentres from a cached USGS catalogue.

        Sources cluster along plate boundaries, so the ray coverage they give
        is strongly anisotropic. That is the point of using them.

        Args:
            count: how many to return. All of them if omitted.
            minimum_magnitude: drop events below this, which is how the
                catalogue is thinned to the events a study would use.
            rng: the generator for the draw.

        Returns:
            ``(latitude, longitude)`` epicentres in degrees.
        """
        table = _read_table("usgs_event_cache.csv")
        keep = table["mag"] >= minimum_magnitude
        points = _as_points(table["latitude"][keep], table["longitude"][keep])
        return _subsample(points, count, rng)

    def domain_mask(
        self, /, *, ocean: bool = False, resolution: str = "110m"
    ) -> np.ndarray:
        """A field that is one on land and zero at sea, or the other way round.

        Needs ``cartopy`` and ``shapely``, which come with the ``sphere``
        extra. Sampled on the grid rather than expanded exactly, so the
        coastlines ring: smooth the result with a heat-kernel covariance before
        using it as a coefficient field.

        Args:
            ocean: mask the ocean rather than the land.
            resolution: the Natural Earth resolution, as cartopy names it.

        Returns:
            The mask as a field of ones and zeros.

        Raises:
            ImportError: without cartopy, which supplies the coastlines.
        """
        try:
            import shapely.geometry as geometry
            from cartopy.io import shapereader
            from shapely.prepared import prep
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError(
                "domain_mask needs cartopy and shapely, which come with the "
                "'sphere' extra."
            ) from error

        reader = shapereader.Reader(
            shapereader.natural_earth(
                resolution=resolution, category="physical", name="land"
            )
        )
        land = prep(geometry.MultiPolygon(list(reader.geometries())))

        def indicator(point: Any) -> float:
            latitude, longitude = np.asarray(point, dtype=float)
            on_land = land.contains(geometry.Point(float(longitude), float(latitude)))
            return float(on_land != ocean)

        return self.project_function(indicator)

    def source_receiver_paths(
        self,
        /,
        *,
        sources: int = 20,
        receivers: int = 10,
        minimum_magnitude: float = 5.5,
        minimum_separation: float = 0.0,
        rng: Generator | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Every source-receiver pair, as paths, from the shipped catalogues.

        The convenience every tomography script writes for itself.
        ``minimum_separation`` drops pairs too close together to carry
        information, which in a real network is a noticeable fraction of them.

        Args:
            sources: how many earthquakes to draw.
            receivers: how many stations.
            minimum_magnitude: the magnitude threshold for the sources.
            minimum_separation: drop pairs closer than this, as a *physical*
                distance. Zero keeps every pair.
            rng: the generator.

        Returns:
            The ``(source, receiver)`` pairs that survive the separation
            filter, so there may be fewer than ``sources * receivers``.
        """
        generator = np.random.default_rng() if rng is None else rng
        origins = self.earthquakes(
            count=sources, minimum_magnitude=minimum_magnitude, rng=generator
        )
        stations = self.stations(count=receivers, rng=generator)
        return [
            (origin, station)
            for origin in origins
            for station in stations
            if self.geodesic_distance(origin, station) > minimum_separation
        ]

    def _embedding(self, points: Sequence[Any], /) -> tuple[np.ndarray, Any]:
        """The points as vectors in R^3, scaled to the sphere's radius.

        Euclidean distance in the embedding is the *chord*, not the geodesic;
        the two are related by :meth:`_embedded_radius` and its inverse. Chord
        length is what makes a KD-tree usable here, and it is exactly zero on
        the diagonal, which the cosine route is not.
        """
        vectors = np.stack([self._to_vector(point) for point in points])
        return vectors * self._radius, None

    def _embedded_radius(self, distance: float, /) -> float:
        """A geodesic radius as a chord: ``2 R sin(d / 2R)``."""
        return float(2.0 * self._radius * np.sin(0.5 * distance / self._radius))

    def _geodesic_from_embedded(self, lengths: np.ndarray, /) -> np.ndarray:
        """Chords back to geodesics: ``2 R asin(c / 2R)``."""
        ratio = np.clip(np.asarray(lengths, dtype=float) / (2.0 * self._radius), -1.0, 1.0)
        return 2.0 * self._radius * np.arcsin(ratio)

    # ----------------------------------------------------------------- #
    #                            Resolution                             #
    # ----------------------------------------------------------------- #

    def with_degree(self, lmax: int, /) -> Sphere:
        """The same space, truncated at or extended to a different degree."""
        return Sphere(
            lmax,
            radius=self._radius,
            order=self._order,
            length_scale=self._length_scale,
            sampling=self._sampling,
        )

    def degree_transfer_operator(self, target: Sphere, /) -> LinearOperator:
        """Truncation to, or prolongation into, another degree.

        Restriction when the target is coarser, zero-padding when it is finer.
        The adjoint is derived rather than written down, which matters: it is
        the *other* one of the pair only when the two spaces carry the same
        metric on their shared components, and it is the ratio of the two
        metrics otherwise. Getting that by hand is the mistake of DESIGN.md
        section 5.6 wearing a different hat.

        Args:
            target: the sphere to map into. It must have the same radius --
                two spheres of different size are not two resolutions of one
                domain.

        Returns:
            The operator.

        Raises:
            ValueError: if the radii differ.
        """
        if target.radius != self._radius:
            raise ValueError("Degree transfer needs a common radius.")
        shared = min(self._lmax, target.lmax)
        keep = np.flatnonzero(self._packing[1] <= shared)
        place = np.flatnonzero(target._packing[1] <= shared)

        def value(x: np.ndarray) -> np.ndarray:
            components = np.zeros(target.dim)
            components[place] = self.to_components(x)[keep]
            return target.from_components(components)

        def derivative_components(y: np.ndarray) -> np.ndarray:
            pulled = target.apply_gram(target.to_components(y))
            total = np.zeros(self.dim)
            total[keep] = pulled[place]
            return total

        return LinearOperator.from_derivative_callables(
            self, target, value, derivative_components
        )

    def with_order(
        self, order: float, /, *, length_scale: float | None = None
    ) -> Sphere:
        """The same expansion, viewed with a different Sobolev order.

        Args:
            order: the new Sobolev order.
            length_scale: the new length scale. Kept as it is if omitted, so
                this is a change of order alone.

        Returns:
            The same expansion in the new metric. The components are
            unchanged; only the inner product moves.
        """
        return Sphere(
            self._lmax,
            radius=self._radius,
            order=order,
            length_scale=(self._length_scale if length_scale is None else length_scale),
            sampling=self._sampling,
        )


def _read_table(name: str) -> dict[str, np.ndarray]:
    """Read one of the shipped CSV tables into arrays, keyed by column."""
    import csv
    from importlib.resources import files

    text = (files("pygeoinf2.data") / name).read_text(encoding="utf-8")
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        raise ValueError(f"{name} is empty.")
    table: dict[str, np.ndarray] = {}
    for column in rows[0]:
        values = [row[column] for row in rows]
        try:
            table[column] = np.array([float(value) for value in values])
        except ValueError:
            table[column] = np.array(values, dtype=object)
    return table


def _as_points(latitudes: np.ndarray, longitudes: np.ndarray) -> list[np.ndarray]:
    """Two columns of a catalogue as points.

    No conversion: a catalogue holds degrees and so does a point, which is the
    whole reason for the convention. This used to turn them into colatitude and
    radians here, privately, so that a caller who read the same file themselves
    got different answers from the same numbers.
    """
    return [
        np.array([float(latitude), float(longitude)])
        for latitude, longitude in zip(
            np.asarray(latitudes, dtype=float),
            np.asarray(longitudes, dtype=float),
        )
    ]


def _subsample(
    points: list[np.ndarray], count: int | None, rng: Generator | None
) -> list[np.ndarray]:
    """Draw ``count`` points without replacement, or return all of them."""
    if count is None:
        return points
    if count > len(points):
        raise ValueError(f"Asked for {count} points from a table of {len(points)}.")
    generator = np.random.default_rng() if rng is None else rng
    chosen = generator.choice(len(points), size=count, replace=False)
    return [points[index] for index in chosen]


class Lebesgue(Sphere):
    """The ``L2`` space on a sphere, with an orthonormal harmonic basis.

    A class rather than a factory function, so that ``isinstance(x, Lebesgue)``
    answers what it looks like it answers and ``type(x).__name__`` names the
    geometry. Nothing is added: it is :class:`Sphere` at order zero.
    """

    def __init__(
        self,
        lmax: int,
        /,
        *,
        radius: float = 1.0,
        sampling: int = 1,
    ) -> None:
        """
        Args:
            lmax: the maximum spherical harmonic degree.
            radius: the sphere's radius.
            sampling: grid columns per row, 1 or 2.
        """
        super().__init__(lmax, radius=radius, order=0.0, sampling=sampling)


class Sobolev(Sphere):
    """The Sobolev space ``H^order`` on a sphere.

    The same expansion as :class:`Lebesgue`, with
    ``(1 + length_scale^2 l(l+1)/radius^2)^order`` as its metric — a
    diagonal-metric space, so every invariant operator on it stays diagonal.
    """

    def __init__(
        self,
        lmax: int,
        order: float,
        length_scale: float,
        /,
        *,
        radius: float = 1.0,
        sampling: int = 1,
    ) -> None:
        """
        Args:
            lmax: the maximum spherical harmonic degree.
            order: the Sobolev order.
            length_scale: the length at which the Sobolev weight turns over.
            radius: the sphere's radius.
            sampling: grid columns per row, 1 or 2.
        """
        super().__init__(
            lmax,
            radius=radius,
            order=order,
            length_scale=length_scale,
            sampling=sampling,
        )
