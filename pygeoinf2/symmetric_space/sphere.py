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
grid is Driscoll-Healy with ``sampling=2``, and a point is a
``(colatitude, longitude)`` pair in radians.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, Callable, Hashable, Sequence

import numpy as np
from numpy.random import Generator

from ..algebra.operators import LinearFunctional, LinearOperator
from .base import SymmetricSpace, _distribute

__all__ = ["Sphere", "Lebesgue", "Sobolev"]

# pyshtools' low-level normalisation code for orthonormal harmonics.
_ORTHONORMAL = 4
# pyshtools spells "leave the Condon-Shortley phase out" as csphase=1.
_NO_CONDON_SHORTLEY = 1


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


class Sphere(SymmetricSpace):
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
    def gaussian_curvature(self) -> float:
        """``1 / radius^2``, constant over the sphere."""
        return 1.0 / self._radius**2

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

    @cached_property
    def _legendre_indices(self) -> np.ndarray:
        """Where each component sits in pyshtools' packed Legendre array.

        Cached because it depends only on ``lmax``. Recomputing it per call
        made :meth:`basis_at` a Python loop of length ``dim``, and so made the
        adjoint of an observation operator quadratic in the truncation.
        """
        from pyshtools.legendre import PlmIndex

        _, degrees, orders = self._packing
        return np.array(
            [
                PlmIndex(int(degree), int(order))
                for degree, order in zip(degrees, orders)
            ]
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
            csphase=_NO_CONDON_SHORTLEY,
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
            csphase=_NO_CONDON_SHORTLEY,
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
        from pyshtools.legendre import PlmON

        position = np.atleast_1d(np.asarray(point, dtype=float))
        if position.shape != (2,):
            raise ValueError(
                f"A point is a (colatitude, longitude) pair, got {position.shape}."
            )
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

    def evaluate(self, x: np.ndarray, points: Sequence[Any], /) -> np.ndarray:
        """Field values at scattered points, through one harmonic expansion.

        Overrides the generic route, which builds the whole basis at every
        point. pyshtools evaluates the whole set in one call, which is the
        difference between a tomography problem being feasible and not.
        """
        from pyshtools import SHCoeffs

        positions = np.asarray([np.asarray(point, dtype=float) for point in points])
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("Points are (colatitude, longitude) pairs in radians.")

        coefficients = np.zeros((2, self._lmax + 1, self._lmax + 1))
        parts, degrees, orders = self._packing
        coefficients[parts, degrees, orders] = self.to_components(x) / self._radius
        expansion = SHCoeffs.from_array(
            coefficients,
            normalization="ortho",
            csphase=_NO_CONDON_SHORTLEY,
        )
        return np.atleast_1d(
            np.asarray(
                expansion.expand(
                    lat=90.0 - np.degrees(positions[:, 0]),
                    lon=np.degrees(positions[:, 1]),
                    degrees=True,
                ),
                dtype=float,
            )
        ).ravel()

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

    @property
    def reference_point(self) -> np.ndarray:
        """The north pole. Any point would do; the sphere is homogeneous."""
        return np.array([0.0, 0.0])

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

    # ----------------------------------------------------------------- #
    #                              Geometry                             #
    # ----------------------------------------------------------------- #

    @staticmethod
    def _to_vector(point: Any) -> np.ndarray:
        """A ``(colatitude, longitude)`` pair as a unit vector in R^3."""
        position = np.asarray(point, dtype=float)
        colatitude, longitude = float(position[0]), float(position[1])
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
        """A vector in R^3 as a ``(colatitude, longitude)`` pair."""
        unit = np.asarray(vector, dtype=float)
        unit = unit / np.linalg.norm(unit)
        return np.array(
            [
                float(np.arccos(np.clip(unit[2], -1.0, 1.0))),
                float(np.arctan2(unit[1], unit[0]) % (2.0 * np.pi)),
            ]
        )

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
        """
        if count < 1:
            raise ValueError("A quadrature rule needs at least one node.")
        first, second = self._to_vector(start), self._to_vector(end)
        angle = float(np.arccos(np.clip(np.dot(first, second), -1.0, 1.0)))

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
        """
        if angular_radius < 0.0 or angular_radius > np.pi:
            raise ValueError(
                f"A cap's angular radius lies in [0, pi], got {angular_radius}."
            )
        area_fraction = 0.5 * (1.0 - np.cos(angular_radius))
        if area_fraction <= 0.0:
            if normalise:
                raise ValueError("A cap of zero area has no average.")
            return LinearFunctional.from_derivative_components(self, np.zeros(self.dim))

        from pyshtools import SHCoeffs

        position = np.asarray(centre, dtype=float)
        cap = SHCoeffs.from_cap(
            np.degrees(angular_radius),
            self._lmax,
            clat=90.0 - np.degrees(float(position[0])),
            clon=np.degrees(float(position[1])),
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
        """The exact average over a spherical cap, as a functional."""
        return self.spherical_cap_integral(centre, angular_radius, normalise=True)

    def geodesic_ball_average_operator(
        self,
        centres: Sequence[Any],
        radius: float,
        /,
        *,
        count: int | None = None,
        normalise: bool = True,
    ) -> LinearOperator:
        """Cap averages, exactly, as an operator into a Euclidean space.

        The property operator of a spherical inference problem. Uses the exact
        cap functionals unless ``count`` is given, which forces the generic
        quadrature route and exists so the two can be checked against each
        other.
        """
        centres = tuple(centres)
        if not centres:
            raise ValueError("At least one centre is needed.")
        if count is not None:
            return super().geodesic_ball_average_operator(
                centres, radius, count=count, normalise=normalise
            )

        from ..algebra.spaces import EuclideanSpace

        angular_radius = radius / self._radius
        rows = np.stack(
            [
                self.spherical_cap_integral(
                    centre, angular_radius, normalise=normalise
                ).derivative_components
                for centre in centres
            ]
        )
        return LinearOperator.from_derivative_matrix(
            self, EuclideanSpace(len(centres)), rows
        )

    # ----------------------------------------------------------------- #
    #                          Coefficients                             #
    # ----------------------------------------------------------------- #

    def coefficient_operator(
        self, /, *, lmax: int | None = None, lmin: int = 0
    ) -> LinearOperator:
        """The harmonic coefficients of a field, as an operator.

        The property operator of Al-Attar (2021): estimate finitely many
        spherical harmonic coefficients from a finite set of point values. On a
        space whose components already *are* those coefficients this is a
        selection, so it is matrix-free and costs nothing.
        """
        from ..algebra.spaces import EuclideanSpace

        top = self._lmax if lmax is None else int(lmax)
        if not 0 <= lmin <= top <= self._lmax:
            raise ValueError(
                f"Degrees must satisfy 0 <= lmin <= lmax <= {self._lmax}, "
                f"got lmin={lmin}, lmax={top}."
            )
        degrees = self._packing[1]
        selected = np.flatnonzero((degrees >= lmin) & (degrees <= top))

        def value(x: np.ndarray) -> np.ndarray:
            return self.to_components(x)[selected]

        def derivative_components(y: np.ndarray) -> np.ndarray:
            total = np.zeros(self.dim)
            total[selected] = np.asarray(y, dtype=float)
            return total

        return LinearOperator.from_derivative_callables(
            self, EuclideanSpace(selected.size), value, derivative_components
        )

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
        """
        table = _read_table("gsn_stations.csv")
        points = _degrees_to_points(table["Latitude"], table["Longitude"])
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
        """
        table = _read_table("usgs_event_cache.csv")
        keep = table["mag"] >= minimum_magnitude
        points = _degrees_to_points(table["latitude"][keep], table["longitude"][keep])
        return _subsample(points, count, rng)

    def domain_mask(
        self, /, *, ocean: bool = False, resolution: str = "110m"
    ) -> np.ndarray:
        """A field that is one on land and zero at sea, or the other way round.

        Needs ``cartopy`` and ``shapely``, which come with the ``sphere``
        extra. Sampled on the grid rather than expanded exactly, so the
        coastlines ring: smooth the result with a heat-kernel covariance before
        using it as a coefficient field.
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
            position = np.asarray(point, dtype=float)
            latitude = 90.0 - np.degrees(float(position[0]))
            longitude = (np.degrees(float(position[1])) + 180.0) % 360.0 - 180.0
            on_land = land.contains(geometry.Point(longitude, latitude))
            return float(on_land != ocean)

        return self.project_function(indicator)

    def pairs_within_distance(
        self, points: Sequence[Any], distance: float, /
    ) -> tuple[np.ndarray, np.ndarray]:
        """Index pairs closer together than a given geodesic distance.

        What a localised covariance needs: the sparsity pattern of "these two
        data see overlapping parts of the model". Returned as two index arrays,
        ready for ``scipy.sparse``.
        """
        vectors = np.stack([self._to_vector(point) for point in points])
        # Chord length, then 2 asin(chord/2). Accurate for small separations,
        # and exactly zero on the diagonal, which the cosine route is not.
        chords = np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=-1)
        distances = 2.0 * self._radius * np.arcsin(np.clip(0.5 * chords, 0.0, 1.0))
        rows, columns = np.nonzero(distances <= distance)
        return rows, columns

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
        )

    def degree_transfer_operator(self, target: Sphere, /) -> LinearOperator:
        """Truncation to, or prolongation into, another degree.

        Restriction when the target is coarser, zero-padding when it is finer.
        The adjoint is derived rather than written down, which matters: it is
        the *other* one of the pair only when the two spaces carry the same
        metric on their shared components, and it is the ratio of the two
        metrics otherwise. Getting that by hand is the mistake of DESIGN.md
        section 5.6 wearing a different hat.
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
        """The same expansion, viewed with a different Sobolev order."""
        return Sphere(
            self._lmax,
            radius=self._radius,
            order=order,
            length_scale=(self._length_scale if length_scale is None else length_scale),
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


def _degrees_to_points(
    latitudes: np.ndarray, longitudes: np.ndarray
) -> list[np.ndarray]:
    """Latitude and longitude in degrees as (colatitude, longitude) radians."""
    colatitudes = np.radians(90.0 - np.asarray(latitudes, dtype=float))
    azimuths = np.radians(np.asarray(longitudes, dtype=float)) % (2.0 * np.pi)
    return [np.array([theta, phi]) for theta, phi in zip(colatitudes, azimuths)]


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
