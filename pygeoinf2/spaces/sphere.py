"""
Fields on a two-sphere, through spherical harmonics.

The one space in the family that is genuinely its own implementation: the
harmonic transform is not an FFT and there is no way to obtain it from the
periodic box. Everything downstream of the transform is shared, though — the
Laplacian is diagonal, invariant operators are
:class:`~pygeoinf2.algebra.diagonal.DiagonalLinearOperator`, and invariant
measures come from :class:`~pygeoinf2.spaces.invariant.InvariantSpace` — so
what is written here is the transform, the spectrum and the geometry, and
nothing else.

Requires ``pyshtools``, which is an optional dependency. The import is deferred
to construction so that importing this module costs nothing when the sphere is
not used.

Conventions, all pinned by test rather than assumed: coefficients are
orthonormal (``norm=4`` in pyshtools' low-level interface) with the
Condon-Shortley phase included, the grid is Driscoll-Healy with
``sampling=2``, and a point is a ``(colatitude, longitude)`` pair in radians.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, Callable, Hashable

import numpy as np
from numpy.random import Generator

from .invariant import InvariantSpace

__all__ = ["Sphere", "Lebesgue", "Sobolev"]

# pyshtools' low-level normalisation code for orthonormal harmonics.
_ORTHONORMAL = 4
_CONDON_SHORTLEY = 1


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


class Sphere(InvariantSpace):
    """A field on a sphere, expanded in spherical harmonics.

    Vectors are grid arrays of shape ``(n, 2n)`` with ``n == 2 (lmax + 1)``,
    holding values on a Driscoll-Healy grid. Components are the real harmonic
    coefficients, scaled so that the Lebesgue basis is orthonormal on a sphere
    of the given radius.
    """

    def __init__(
        self,
        lmax: int,
        /,
        *,
        radius: float = 1.0,
        order: float = 0.0,
        length_scale: float = 1.0,
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
        """
        if lmax < 0:
            raise ValueError("lmax must be non-negative.")
        if radius <= 0.0:
            raise ValueError("radius must be positive.")
        if length_scale <= 0.0:
            raise ValueError("length_scale must be positive.")
        _require_pyshtools()

        self._lmax = int(lmax)
        self._radius = float(radius)
        self._order = float(order)
        self._length_scale = float(length_scale)
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
        """The shape of a field array: latitudes by longitudes."""
        return (self._latitudes, 2 * self._latitudes)

    @property
    def area(self) -> float:
        """The sphere's surface area."""
        return 4.0 * np.pi * self._radius**2

    @property
    def laplacian_eigenvalues(self) -> np.ndarray:
        """``l (l + 1) / radius^2`` for each component."""
        return self._laplacian_eigenvalues

    def _key(self) -> Hashable:
        return (self._lmax, self._radius, self._order, self._length_scale)

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
        dimension of the space.
        """
        parts, degrees, orders = [], [], []
        for degree in range(self._lmax + 1):
            for order in range(degree + 1):
                parts.append(0)
                degrees.append(degree)
                orders.append(order)
            for order in range(1, degree + 1):
                parts.append(1)
                degrees.append(degree)
                orders.append(order)
        return (
            np.asarray(parts, dtype=int),
            np.asarray(degrees, dtype=int),
            np.asarray(orders, dtype=int),
        )

    @property
    def _degree_of_component(self) -> np.ndarray:
        """The harmonic degree attached to each component."""
        return self._packing[1].astype(float)

    def to_components(self, x: np.ndarray) -> np.ndarray:
        """Harmonic coefficients of a field, orthonormal in ``L2``."""
        from pyshtools.expand import SHExpandDH

        field = np.asarray(x, dtype=float)
        if field.shape != self.grid_shape:
            raise ValueError(f"A field has shape {self.grid_shape}, got {field.shape}.")
        coefficients = SHExpandDH(
            field,
            norm=_ORTHONORMAL,
            sampling=2,
            csphase=_CONDON_SHORTLEY,
            lmax_calc=self._lmax,
        )
        parts, degrees, orders = self._packing
        return self._radius * coefficients[parts, degrees, orders]

    def from_components(self, c: np.ndarray) -> np.ndarray:
        """The field with the given harmonic coefficients."""
        from pyshtools.expand import MakeGridDH

        components = np.asarray(c, dtype=float)
        if components.shape != (self.dim,):
            raise ValueError(f"Expected {self.dim} components, got {components.shape}.")
        coefficients = np.zeros((2, self._lmax + 1, self._lmax + 1))
        parts, degrees, orders = self._packing
        coefficients[parts, degrees, orders] = components / self._radius
        return MakeGridDH(
            coefficients,
            norm=_ORTHONORMAL,
            sampling=2,
            csphase=_CONDON_SHORTLEY,
            lmax=self._lmax,
        )

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
        return np.arange(2 * self._latitudes) * np.pi / self._latitudes

    def basis_at(self, point: Any, /) -> np.ndarray:
        """The value of each orthonormal harmonic at a point.

        Args:
            point: a ``(colatitude, longitude)`` pair in radians.
        """
        from pyshtools.legendre import PlmON, PlmIndex

        position = np.atleast_1d(np.asarray(point, dtype=float))
        if position.shape != (2,):
            raise ValueError(
                f"A point is a (colatitude, longitude) pair, got {position.shape}."
            )
        colatitude, longitude = float(position[0]), float(position[1])

        legendre = PlmON(self._lmax, np.cos(colatitude), csphase=_CONDON_SHORTLEY)
        parts, degrees, orders = self._packing
        indices = np.array(
            [
                PlmIndex(int(degree), int(order))
                for degree, order in zip(degrees, orders)
            ]
        )
        angle = orders * longitude
        harmonics = legendre[indices] * np.where(
            parts == 0, np.cos(angle), np.sin(angle)
        )
        # Components carry a factor of the radius, so the dual basis carries
        # its reciprocal: the basis functions are orthonormal on this sphere,
        # not on the unit one.
        return harmonics / self._radius

    def project_function(self, function: Callable[[Any], float], /) -> np.ndarray:
        """Sample a function on the grid.

        The function receives a ``(colatitude, longitude)`` pair in radians.
        """
        colatitudes, longitudes = np.meshgrid(
            self.colatitudes, self.longitudes, indexing="ij"
        )
        values = np.array(
            [
                float(function(np.array([theta, phi])))
                for theta, phi in zip(colatitudes.ravel(), longitudes.ravel())
            ]
        )
        return values.reshape(self.grid_shape)

    def random_point(self, *, rng: Generator | None = None) -> np.ndarray:
        """A point drawn uniformly over the sphere's area.

        Uniform in ``cos(colatitude)``, not in colatitude: sampling the angle
        uniformly would crowd the poles, which is the classic way to bias a
        set of station locations.
        """
        generator = np.random.default_rng() if rng is None else rng
        return np.array(
            [
                float(np.arccos(generator.uniform(-1.0, 1.0))),
                float(generator.uniform(0.0, 2.0 * np.pi)),
            ]
        )

    def random_points(
        self, count: int, /, *, rng: Generator | None = None
    ) -> list[np.ndarray]:
        """Several points drawn uniformly over the sphere's area."""
        return [self.random_point(rng=rng) for _ in range(count)]

    def with_order(
        self, order: float, /, *, length_scale: float | None = None
    ) -> Sphere:
        """The same expansion, viewed with a different Sobolev order."""
        return Sphere(
            self._lmax,
            radius=self._radius,
            order=order,
            length_scale=(self._length_scale if length_scale is None else length_scale),
        )


def Lebesgue(lmax: int, /, *, radius: float = 1.0) -> Sphere:
    """The ``L2`` space on a sphere, with an orthonormal harmonic basis."""
    return Sphere(lmax, radius=radius, order=0.0)


def Sobolev(
    lmax: int, order: float, length_scale: float, /, *, radius: float = 1.0
) -> Sphere:
    """The Sobolev space ``H^order`` on a sphere.

    The same expansion as :func:`Lebesgue`, with
    ``(1 + length_scale^2 l(l+1)/radius^2)^order`` as its metric — a
    diagonal-metric space, so every invariant operator on it stays diagonal.
    """
    return Sphere(lmax, radius=radius, order=order, length_scale=length_scale)
