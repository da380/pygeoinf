"""
Periodic boxes in any dimension, through the real FFT.

One implementation covers what v1 spreads over four files and roughly 4.1k
lines: the circle is a one-dimensional box, the torus a two-dimensional one,
and the three-dimensional case — the one that was not there — comes free.

Fields are grid arrays; components are coefficients in a real Fourier basis
that is **orthonormal in L2**, so the Lebesgue space has an identity Gram
matrix and v1's factor-of-two bookkeeping disappears. The Sobolev space is the
same coordinate map with the Sobolev symbol as its metric, so it is a
``DiagonalMetricSpace`` too and no mass-weighted construction is needed.

The one delicate part is packing ``rfftn`` output into real components in more
than one dimension: a mode and its conjugate must be counted once, and which
modes are self-conjugate depends on the parity of every axis length. Rather
than derive that, the conjugate orbits are enumerated once at construction and
the result is checked by test — round trip, Parseval, and the orthonormality of
the explicit basis functions.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, Callable, Hashable, Sequence

import numpy as np
from numpy.random import Generator
from scipy.fft import irfftn, rfftn

from ..algebra.operators import LinearOperator
from ..algebra.spaces import ArrayVectorMixin
from .base import PreparedPoints, SymmetricSpace, lift_formal_adjoint

__all__ = ["PeriodicBox", "Lebesgue", "Sobolev"]


class _FourierPacking:
    """The map between ``rfftn`` output and real orthonormal components.

    Built once per grid. Each conjugate orbit of Fourier modes contributes one
    component if it is a fixed point of conjugation and two — a cosine and a
    sine — if it is a genuine pair. The counts must come to exactly the number
    of grid points, since a real field has that many degrees of freedom; that
    identity is asserted here and is the first thing to fail if the enumeration
    is wrong.
    """

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.rfft_shape = shape[:-1] + (shape[-1] // 2 + 1,)
        self.size = int(np.prod(shape))

        grids = np.meshgrid(*[np.arange(n) for n in self.rfft_shape], indexing="ij")
        flat = [g.ravel() for g in grids]
        conjugate = [(-k) % n for k, n in zip(flat, shape)]

        # A conjugate partner is inside the retained half exactly when its last
        # index is. In one dimension that is only the zero and Nyquist modes;
        # in more, whole slices of the array pair up internally, which is what
        # makes the packing worth enumerating rather than deriving.
        inside = conjugate[-1] < self.rfft_shape[-1]
        linear = np.ravel_multi_index(flat, self.rfft_shape)

        partner = np.full(linear.size, -1, dtype=int)
        if np.any(inside):
            indices = [c[inside] for c in conjugate]
            partner[inside] = np.ravel_multi_index(indices, self.rfft_shape)

        fixed = partner == linear
        # Keep one representative per pair: the lower index when both members
        # are retained, and the entry itself when its partner was dropped.
        paired = (~fixed) & ((partner < 0) | (linear < partner))

        self.fixed_indices = np.flatnonzero(fixed)
        self.paired_indices = np.flatnonzero(paired)
        self.paired_partners = partner[self.paired_indices]
        self.dim = self.fixed_indices.size + 2 * self.paired_indices.size

        if self.dim != self.size:
            raise AssertionError(
                f"Fourier packing produced {self.dim} components for "
                f"{self.size} grid points; the conjugate orbits were "
                f"enumerated incorrectly for shape {shape}."
            )

        signed = [np.where(k > n // 2, k - n, k) for k, n in zip(flat, shape)]
        self.wavenumbers = np.stack(
            [
                np.concatenate(
                    [w[self.fixed_indices], np.repeat(w[self.paired_indices], 2)]
                )
                for w in signed
            ]
        )
        # Which half of a conjugate pair each component is: 0 for a cosine or a
        # fixed point, 1 for a sine. Together with the wavevector this names a
        # component uniquely, which is what lets two grids be matched up.
        self.phases = np.concatenate(
            [
                np.zeros(self.fixed_indices.size, dtype=int),
                np.tile(np.array([0, 1]), self.paired_indices.size),
            ]
        )


class PeriodicBox(ArrayVectorMixin, SymmetricSpace[np.ndarray]):
    """A field on a periodic box, in any number of dimensions.

    Vectors are real grid arrays of shape ``shape``; components are
    coefficients in an orthonormal real Fourier basis.
    """

    def __init__(
        self,
        shape: Sequence[int],
        /,
        *,
        lengths: Sequence[float] | None = None,
        order: float = 0.0,
        length_scale: float = 1.0,
    ) -> None:
        """
        Args:
            shape: grid points along each axis. One entry gives a circle, two a
                torus, three a periodic cube.
            lengths: the physical period along each axis. Defaults to ``2 pi``
                on every axis, so that a one-dimensional box is the unit
                circle.
            order: the Sobolev order. Zero gives the Lebesgue space, whose
                basis is orthonormal.
            length_scale: the length at which the Sobolev weight turns over.
                Ignored when the order is zero. Named in full because ``scale``
                is the space's vector-scaling operation: a space's own
                attributes share a namespace with the whole vector API.
        """
        shape = tuple(int(n) for n in shape)
        if not shape or any(n < 2 for n in shape):
            raise ValueError("Every axis needs at least two grid points.")
        lengths = (
            tuple(2.0 * np.pi for _ in shape)
            if lengths is None
            else tuple(float(x) for x in lengths)
        )
        if len(lengths) != len(shape):
            raise ValueError(f"Got {len(lengths)} lengths for {len(shape)} axes.")
        if any(x <= 0.0 for x in lengths):
            raise ValueError("Every length must be positive.")
        if length_scale <= 0.0:
            raise ValueError("length_scale must be positive.")

        self._shape = shape
        self._lengths = lengths
        self._order = float(order)
        self._length_scale = float(length_scale)
        self._packing = _FourierPacking(shape)
        self._volume = float(np.prod(lengths))

        eigenvalues = self._compute_eigenvalues()
        self._laplacian_eigenvalues = eigenvalues
        metric = (
            np.ones(self._packing.dim)
            if order == 0.0
            else (1.0 + self._length_scale**2 * eigenvalues) ** self._order
        )
        super().__init__(metric)

    # ----------------------------------------------------------------- #
    #                              Structure                            #
    # ----------------------------------------------------------------- #

    @property
    def shape(self) -> tuple[int, ...]:
        """Grid points along each axis."""
        return self._shape

    @property
    def lengths(self) -> tuple[float, ...]:
        """The period along each axis."""
        return self._lengths

    @cached_property
    def degrees(self) -> np.ndarray:
        """The wavenumber magnitude of each component, rounded down.

        The analogue of a harmonic degree: components sharing one are the
        modes of a common spatial scale.
        """
        wavenumbers = self._packing.wavenumbers
        return np.floor(np.sqrt(np.sum(wavenumbers.astype(float) ** 2, axis=0))).astype(
            int
        )

    @property
    def gaussian_curvature(self) -> float:
        """Zero: a periodic box is flat."""
        return 0.0

    def truncate(self, x: np.ndarray, /) -> np.ndarray:
        """The identity: ``rfftn`` packing gives one component per grid point.

        Overridden to skip two transforms per pointwise product. The base
        class's round trip would return the same array; on this grid there is
        nothing above the truncation to remove.
        """
        return np.asarray(x)

    @property
    def spatial_dimension(self) -> int:
        """The number of axes: one for a circle, two for a torus."""
        return len(self._shape)

    @property
    def volume(self) -> float:
        """The measure of the box."""
        return self._volume

    @property
    def order(self) -> float:
        """The Sobolev order. Zero for a Lebesgue space."""
        return self._order

    @property
    def length_scale(self) -> float:
        """The Sobolev length scale."""
        return self._length_scale

    @property
    def laplacian_eigenvalues(self) -> np.ndarray:
        """``|k|^2`` for each retained mode, with ``k`` the physical wavevector."""
        return self._laplacian_eigenvalues

    def _key(self) -> Hashable:
        return (self._shape, self._lengths, self._order, self._length_scale)

    def _coordinate_key(self) -> Hashable:
        """The grid, which the order and length scale do not touch.

        Tagged by geometry rather than by ``type(self)``, as the sphere's is
        and for the same reason: ``Circle``, ``Torus``, ``Lebesgue`` and
        ``Sobolev`` are thin subclasses over one grid and one point map, so
        keying on the concrete class said two views of one field were different
        fields -- and a formal-adjoint lift between them round-tripped through
        components instead of passing the vector straight through.
        """
        return ("periodic_box", self._shape, self._lengths)

    def __repr__(self) -> str:
        kind = "Lebesgue" if self._order == 0.0 else f"Sobolev(order={self._order})"
        return f"PeriodicBox({self._shape}, {kind})"

    def _compute_eigenvalues(self) -> np.ndarray:
        """``sum_i (2 pi k_i / L_i)^2`` for each component."""
        wavenumbers = self._packing.wavenumbers
        squared = np.zeros(self._packing.dim)
        for axis, length in enumerate(self._lengths):
            squared += (2.0 * np.pi * wavenumbers[axis] / length) ** 2
        return squared

    # ----------------------------------------------------------------- #
    #                             Coordinates                           #
    # ----------------------------------------------------------------- #

    @cached_property
    def _component_scale(self) -> float:
        """Turns ``rfftn`` output into orthonormal coefficients.

        With ``X = rfftn(x)``, Parseval on the grid gives
        ``integral |x|^2 == (V / N^2) sum_all_k |X_k|^2``, so a fixed-point
        mode contributes ``sqrt(V)/N times Re(X)`` and a conjugate pair
        contributes ``sqrt(2V)/N`` times each of its real and imaginary parts.
        """
        return float(np.sqrt(self._volume) / self._packing.size)

    def to_components(self, x: np.ndarray) -> np.ndarray:
        """Spectral coefficients of a field, orthonormal in L2."""
        transform = rfftn(np.asarray(x, dtype=float), s=self._shape).ravel()
        packing = self._packing
        fixed = transform[packing.fixed_indices].real * self._component_scale
        paired = transform[packing.paired_indices] * (
            self._component_scale * np.sqrt(2.0)
        )
        return np.concatenate(
            [fixed, np.stack([paired.real, paired.imag], axis=1).ravel()]
        )

    def from_components(self, c: np.ndarray) -> np.ndarray:
        """The field with the given spectral coefficients."""
        c = np.asarray(c, dtype=float)
        packing = self._packing
        transform = np.zeros(packing.rfft_shape, dtype=complex).ravel()

        count = packing.fixed_indices.size
        transform[packing.fixed_indices] = c[:count] / self._component_scale

        rest = c[count:].reshape(-1, 2) / (self._component_scale * np.sqrt(2.0))
        values = rest[:, 0] + 1j * rest[:, 1]
        transform[packing.paired_indices] = values

        # Partners retained inside the half must be filled in explicitly:
        # irfftn imposes Hermitian symmetry across the dropped half only, not
        # within the slices it keeps.
        retained = packing.paired_partners >= 0
        transform[packing.paired_partners[retained]] = np.conjugate(values[retained])

        return irfftn(transform.reshape(packing.rfft_shape), s=self._shape)

    # ----------------------------------------------------------------- #
    #                          Points and grids                         #
    # ----------------------------------------------------------------- #

    @cached_property
    def grid_axes(self) -> tuple[np.ndarray, ...]:
        """The sample coordinates along each axis."""
        return tuple(
            np.arange(n) * length / n for n, length in zip(self._shape, self._lengths)
        )

    @property
    def reference_point(self) -> np.ndarray:
        """The origin. Any point would do; the box is homogeneous."""
        return np.zeros(self.spatial_dimension)

    def basis_at(self, point: Any, /) -> np.ndarray:
        """The value of each orthonormal basis function at a point.

        A fixed-point mode contributes ``a cos(k.r)``; a conjugate pair
        contributes ``a cos(k.r)`` and ``-a sin(k.r)``, the sign following the
        convention that ``rfftn`` carries a negative exponent.

        Args:
            point: where to evaluate the basis.

        Returns:
            One value per component.

        Raises:
            ValueError: if the point does not have as many coordinates as the
                box has axes.
        """
        position = np.atleast_1d(np.asarray(point, dtype=float))
        if position.shape != (self.spatial_dimension,):
            raise ValueError(
                f"A point needs {self.spatial_dimension} coordinates, got "
                f"{position.shape}."
            )
        packing = self._packing
        phase = np.zeros(packing.dim)
        for axis, length in enumerate(self._lengths):
            phase += 2.0 * np.pi * packing.wavenumbers[axis] * position[axis] / length

        # A conjugate pair contributes a cosine and a sine of amplitude
        # sqrt(2/V); a fixed point of conjugation -- the constant mode, and the
        # Nyquist modes on even axes -- contributes a single cosine of
        # amplitude 1/sqrt(V), because it appears once in the spectrum rather
        # than twice.
        count = packing.fixed_indices.size
        values = np.empty(packing.dim)
        values[:count] = np.cos(phase[:count]) / np.sqrt(self._volume)

        pair_amplitude = np.sqrt(2.0 / self._volume)
        values[count::2] = pair_amplitude * np.cos(phase[count::2])
        values[count + 1 :: 2] = -pair_amplitude * np.sin(phase[count + 1 :: 2])
        return values

    # ----------------------------------------------------------------- #
    #                     Evaluation by non-uniform FFT                 #
    # ----------------------------------------------------------------- #

    @cached_property
    def _nufft_layout(
        self,
    ) -> tuple[tuple[int, ...], np.ndarray, np.ndarray, np.ndarray, int] | None:
        """Where each component sits in a full complex spectrum, and its size.

        ``None`` above three dimensions, which is as far as finufft goes, and
        also ``None`` when finufft is not installed. Both send
        :meth:`evaluate` back to the direct sum, which is slower but identical.

        The spectrum is **two larger than the grid along every axis**. That is
        not padding for accuracy: a Nyquist wavenumber is ``+n/2``, which lies
        outside the mode range ``[-N/2, N/2-1]`` that finufft indexes, and
        folding it back to ``-n/2`` changes the sign of its phase on every axis
        it appears. Widening the spectrum instead lets ``+k`` and ``-k`` be
        distinct slots for every mode, so fixed points of conjugation and
        genuine pairs need no separate handling at all.
        """
        dimension = self.spatial_dimension
        if dimension > 3:
            return None
        try:
            import finufft  # noqa: F401
        except ImportError:  # pragma: no cover - depends on the install
            return None

        packing = self._packing
        sizes = tuple(n + 2 for n in self._shape)
        offsets = [n // 2 for n in sizes]
        fixed = packing.fixed_indices.size
        unique = np.concatenate([np.arange(fixed), np.arange(fixed, packing.dim, 2)])
        wavenumbers = packing.wavenumbers[:, unique]

        plus = np.ravel_multi_index(
            [wavenumbers[axis] + offsets[axis] for axis in range(dimension)], sizes
        )
        minus = np.ravel_multi_index(
            [offsets[axis] - wavenumbers[axis] for axis in range(dimension)], sizes
        )
        amplitude = np.concatenate(
            [
                np.full(fixed, 1.0 / np.sqrt(self._volume)),
                np.full(unique.size - fixed, np.sqrt(2.0 / self._volume)),
            ]
        )
        return sizes, plus, minus, amplitude, fixed

    def _angles(self, points: Sequence[Any]) -> list[np.ndarray]:
        """Points as one contiguous array of angles per axis."""
        return self.prepare_points(points).data

    def prepare_points(self, points: Sequence[Any], /) -> PreparedPoints:
        """The points as one contiguous array of angles per axis.

        The conversion the non-uniform FFT starts with, done once for an
        operator rather than once per application: it was 32 of 61 ms on a
        512-square torus at 10^5 points (REVIEW2 4.2.7).

        Args:
            points: points of the box, or an already prepared set.

        Returns:
            The prepared points, carrying one angle array per axis.

        Raises:
            ValueError: if the points do not have this box's number of
                coordinates.
        """
        if isinstance(points, PreparedPoints):
            return points
        points = tuple(points)
        positions = np.asarray(
            [np.atleast_1d(np.asarray(point, dtype=float)) for point in points]
        )
        if positions.ndim != 2 or positions.shape[1] != self.spatial_dimension:
            raise ValueError(
                f"Points need {self.spatial_dimension} coordinates each, got "
                f"shape {positions.shape}."
            )
        angles = [
            np.ascontiguousarray(2.0 * np.pi * positions[:, axis] / self._lengths[axis])
            for axis in range(self.spatial_dimension)
        ]
        return PreparedPoints(points, data=angles)

    def evaluate(
        self,
        x: np.ndarray,
        points: Sequence[Any],
        /,
        *,
        eps: float = 1.0e-12,
        nthreads: int = 1,
    ) -> np.ndarray:
        """Field values at scattered points, by type-2 non-uniform FFT.

        The direct sum costs one basis evaluation per point, so ``len(points)``
        times the dimension. The NUFFT costs a padded FFT plus a local spread,
        which is what makes a tomography problem on a fine grid tractable.

        Falls back to the direct sum above three dimensions or without finufft.

        Args:
            x: a field on this box.
            points: where to evaluate it.
            eps: the NUFFT's requested accuracy.
            nthreads: threads for the NUFFT, one by default. finufft's own
                default is every core, which at these transform sizes costs
                more in threading than it saves in work. Pass zero for it.
        """
        layout = self._nufft_layout
        if layout is None:
            return super().evaluate(x, points)

        import finufft

        sizes, plus, minus, amplitude, fixed = layout

        components = self.to_components(x)
        real = np.concatenate([components[:fixed], components[fixed::2]])
        imaginary = np.concatenate([np.zeros(fixed), components[fixed + 1 :: 2]])
        weights = amplitude * (real + 1j * imaginary)

        spectrum = np.zeros(int(np.prod(sizes)), dtype=complex)
        np.add.at(spectrum, plus, 0.5 * weights)
        np.add.at(spectrum, minus, 0.5 * np.conj(weights))

        transform = (finufft.nufft1d2, finufft.nufft2d2, finufft.nufft3d2)[
            self.spatial_dimension - 1
        ]
        values = transform(
            *self._angles(points),
            spectrum.reshape(sizes),
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
        eps: float = 1.0e-12,
        nthreads: int = 1,
    ) -> np.ndarray:
        """The derivative components of ``x -> sum_i y_i x(r_i)``.

        The adjoint of :meth:`evaluate`, and a type-1 NUFFT is *literally* the
        adjoint of the type-2 one — so this is the same transform run the other
        way rather than a second implementation to keep in step.

        Args:
            weights: one per point.
            points: where the field was evaluated.
            eps: the NUFFT's accuracy, as in :meth:`evaluate`.
            nthreads: threads for it, one by default and for the reason given
                there.

        Returns:
            The derivative components.
        """
        layout = self._nufft_layout
        if layout is None:
            return super().accumulate(weights, points)

        import finufft

        sizes, plus, _, amplitude, fixed = layout

        transform = (finufft.nufft1d1, finufft.nufft2d1, finufft.nufft3d1)[
            self.spatial_dimension - 1
        ]
        spectrum = transform(
            *self._angles(points),
            np.ascontiguousarray(np.asarray(weights, dtype=complex)),
            sizes,
            isign=-1,
            eps=eps,
            nthreads=nthreads,
        )
        at_plus = spectrum.reshape(-1)[plus]

        components = np.empty(self.dim)
        components[:fixed] = amplitude[:fixed] * at_plus[:fixed].real
        components[fixed::2] = amplitude[fixed:] * at_plus[fixed:].real
        # phi carries -sin, and a type-1 transform returns -sum y sin as its
        # imaginary part, so the two negations cancel.
        components[fixed + 1 :: 2] = amplitude[fixed:] * at_plus[fixed:].imag
        return components

    def project_function(self, function: Callable[[Any], float], /) -> np.ndarray:
        """Sample a function on the grid."""
        mesh = np.meshgrid(*self.grid_axes, indexing="ij")
        points = np.stack([m.ravel() for m in mesh], axis=1)
        values = np.array([float(function(p if p.size > 1 else p[0])) for p in points])
        return values.reshape(self._shape)

    def random_point(self, *, rng: Generator | None = None) -> np.ndarray:
        """A point drawn uniformly from the box."""
        generator = np.random.default_rng() if rng is None else rng
        return np.array([generator.uniform(0.0, length) for length in self._lengths])

    # ----------------------------------------------------------------- #
    #                              Geometry                             #
    # ----------------------------------------------------------------- #

    def _separation(self, start: Any, end: Any, /) -> np.ndarray:
        """``end - start``, taken the short way round each periodic axis.

        The whole of what makes a torus a torus rather than a rectangle: the
        two ends of an axis are the same place, so the displacement between two
        points is whichever of the two ways round is shorter.
        """
        first = self._as_point(start)
        second = self._as_point(end)
        lengths = np.asarray(self._lengths, dtype=float)
        offset = second - first
        return offset - lengths * np.round(offset / lengths)

    def _as_point(self, point: Any, /) -> np.ndarray:
        """One point, checked."""
        position = np.atleast_1d(np.asarray(point, dtype=float))
        if position.shape != (self.spatial_dimension,):
            raise ValueError(
                f"A point needs {self.spatial_dimension} coordinates, got "
                f"{position.shape}."
            )
        return position

    def geodesic_distance(self, start: Any, end: Any, /) -> float:
        """The distance between two points, the short way round.

        Args:
            start, end: points of the box.

        Returns:
            The distance.
        """
        return float(np.linalg.norm(self._separation(start, end)))

    def geodesic_quadrature(
        self, start: Any, end: Any, /, *, count: int
    ) -> tuple[list[Any], np.ndarray]:
        """Nodes and weights integrating along the straight path between two
        points, taken the short way round each periodic axis.

        Gauss-Legendre on the segment, so the weights carry the arc-length
        element and sum to the distance -- the contract
        :meth:`~pygeoinf2.symmetric_space.base.SymmetricSpace.geodesic_quadrature`
        states, and what makes integrating the constant one give that distance.

        Args:
            start, end: the endpoints.
            count: how many nodes.

        Returns:
            ``(nodes, weights)``.

        Raises:
            ValueError: if fewer than one node is asked for.
        """
        if count < 1:
            raise ValueError(f"At least one node is needed, got {count}.")
        first = self._as_point(start)
        offset = self._separation(start, end)
        length = float(np.linalg.norm(offset))

        abscissae, weights = np.polynomial.legendre.leggauss(count)
        fractions = 0.5 * (abscissae + 1.0)
        nodes = [self._wrap(first + fraction * offset) for fraction in fractions]
        return nodes, 0.5 * length * weights

    def _wrap(self, point: np.ndarray, /) -> np.ndarray:
        """A point brought back into ``[0, L)`` on each axis."""
        return np.asarray(point, dtype=float) % np.asarray(self._lengths, dtype=float)

    def to_coefficients(self, x: np.ndarray, /) -> np.ndarray:
        """The field's complex Fourier coefficients, in numpy's convention.

        The other half of the seam that components open: components are this
        library's packed *real* vector, ordered by the packing and scaled to be
        orthonormal in ``L2``, and this is exactly what ``rfftn`` returns for
        the same field -- half the spectrum, on the unnormalised scale numpy
        and scipy share, shaped like the grid. The two differ by a factor: a
        component is its coefficient times ``_component_scale``, and times a
        further ``sqrt(2)`` where the coefficient has a conjugate partner that
        the half-spectrum drops.

        That convention rather than this library's, because the point of
        handing out coefficients is to hand them to something else.

        Args:
            x: a field on this box.

        Returns:
            A complex array of shape ``rfftn(x).shape``.
        """
        return rfftn(np.asarray(x, dtype=float), s=self._shape)

    def from_coefficients(self, coefficients: np.ndarray, /) -> np.ndarray:
        """The field with the given complex Fourier coefficients.

        The inverse of :meth:`to_coefficients`, and it inverts it exactly: the
        Hermitian symmetry ``irfftn`` imposes is the symmetry a real field
        already has.

        Args:
            coefficients: a complex array shaped as :meth:`to_coefficients`
                returns.

        Returns:
            A field on this box.

        Raises:
            ValueError: if the array is not shaped as ``rfftn`` returns for
                this grid.
        """
        expected = self._packing.rfft_shape
        array = np.asarray(coefficients, dtype=complex)
        if array.shape != expected:
            raise ValueError(
                f"Coefficients have shape {expected}, got {array.shape}."
            )
        return irfftn(array, s=self._shape)

    # ----------------------------------------------------------------- #
    #                            Resolution                             #
    # ----------------------------------------------------------------- #

    def with_shape(self, shape: Sequence[int], /) -> "PeriodicBox":
        """The same domain, on a different grid.

        The honest primitive for a box, where resolution is per-axis rather
        than a single number.

        Args:
            shape: grid points along each axis.

        Returns:
            The space on that grid.

        Raises:
            ValueError: if the number of axes changes.
        """
        shape = tuple(int(n) for n in shape)
        if len(shape) != self.spatial_dimension:
            raise ValueError(
                f"This box has {self.spatial_dimension} axes, got {len(shape)}."
            )
        return self._rebuilt(shape=shape)

    def with_degree(self, degree: int, /) -> "PeriodicBox":
        """The same domain, resolved to a given wavenumber on every axis.

        The counterpart of :meth:`~pygeoinf2.symmetric_space.sphere.Sphere.with_degree`.
        A grid of ``n`` points along an axis carries wavenumbers up to
        ``n // 2``, so this asks for ``2 * degree`` points on each -- an
        *isotropic* band limit, which is what a degree means. Where the axes
        should be resolved differently, say so with :meth:`with_shape`.

        Args:
            degree: the wavenumber to resolve on each axis.

        Returns:
            The space on that grid.

        Raises:
            ValueError: if the degree is not positive.
        """
        if degree < 1:
            raise ValueError(f"The degree must be positive, got {degree}.")
        return self.with_shape((2 * int(degree),) * self.spatial_dimension)

    def _component_index(self) -> dict:
        """Where each ``(wavevector, phase)`` label sits in the components."""
        packing = self._packing
        return {
            (tuple(int(k) for k in packing.wavenumbers[:, i]), int(packing.phases[i])): i
            for i in range(self.dim)
        }

    def degree_transfer_operator(self, target: "PeriodicBox", /) -> LinearOperator:
        """Truncation to, or prolongation into, another grid.

        Components are matched by their ``(wavevector, cosine-or-sine)`` label,
        not by the degree they fall in: on a box many wavevectors share a
        degree, so matching on that would pair up unrelated modes. Anything the
        target does not have is dropped, and anything it has and this does not
        is left zero.

        The adjoint is derived rather than written down, which matters: it is
        the other one of the pair only when the two spaces carry the same
        metric on their shared components, and it is the ratio of the two
        otherwise.

        Args:
            target: the space to map into.

        Returns:
            The operator.

        Raises:
            ValueError: if the two boxes are not the same domain.
        """
        if target.lengths != self._lengths:
            raise ValueError("Degree transfer needs a common domain.")

        here = self._component_index()
        there = target._component_index()
        shared = [(here[label], there[label]) for label in here if label in there]
        keep = np.array([a for a, _ in shared], dtype=int)
        place = np.array([b for _, b in shared], dtype=int)

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

    def _embedding(self, points: Sequence[Any], /) -> tuple[np.ndarray, Any]:
        """The coordinates themselves, with the box a tree should wrap in.

        Euclidean distance in the embedding *is* the geodesic here, once the
        wrap is accounted for -- which ``cKDTree`` does natively given the box.
        """
        vectors = np.stack([self._as_point(point) for point in points])
        return self._wrap(vectors), np.asarray(self._lengths, dtype=float)

    def geodesic_ball_quadrature(
        self, centre: Any, radius: float, /, *, count: int
    ) -> tuple[list[Any], np.ndarray]:
        """Nodes and weights integrating over a ball, so they sum to its volume.

        Gauss-Legendre in the radius, carrying the ``r^(d-1)`` element, crossed
        with a uniform rule over directions. In one dimension the ball is an
        interval and the directions are the two signs; in two it is a disc and
        they are equally spaced angles, where a uniform rule is exact for the
        trigonometric polynomials it meets.

        Args:
            centre: the ball's centre.
            radius: its radius, in the units of the domain.
            count: roughly how many nodes. Split between radius and direction.

        Returns:
            ``(nodes, weights)``.

        Raises:
            ValueError: for a negative radius or fewer than one node.
            NotImplementedError: in three dimensions or more, where a good set
                of directions on the sphere is its own problem and none of the
                geometries shipped here needs one.
        """
        if count < 1:
            raise ValueError("A quadrature rule needs at least one node.")
        if radius < 0.0:
            raise ValueError(f"The radius must be non-negative, got {radius}.")
        dimension = self.spatial_dimension
        if dimension > 2:
            raise NotImplementedError(
                f"A ball quadrature is implemented in one and two dimensions; "
                f"this box has {dimension}. Integrate with an explicit set of "
                f"nodes instead."
            )
        first = self._as_point(centre)

        if dimension == 1:
            abscissae, weights = np.polynomial.legendre.leggauss(max(count, 1))
            offsets = radius * abscissae
            nodes = [self._wrap(first + np.array([offset])) for offset in offsets]
            return nodes, radius * weights

        rings = max(1, int(np.sqrt(count)))
        spokes = max(1, count // rings)
        abscissae, weights = np.polynomial.legendre.leggauss(rings)
        radii = 0.5 * radius * (abscissae + 1.0)
        # r dr from the area element, and the half-width of the mapping.
        radial = 0.5 * radius * weights * radii
        angles = 2.0 * np.pi * np.arange(spokes) / spokes

        nodes, values = [], []
        for ring_radius, ring_weight in zip(radii, radial):
            for angle in angles:
                offset = ring_radius * np.array([np.cos(angle), np.sin(angle)])
                nodes.append(self._wrap(first + offset))
                values.append(ring_weight * 2.0 * np.pi / spokes)
        return nodes, np.array(values)

    def walk_from(self, point: Any, distances: np.ndarray, /) -> list[Any]:
        """Points at given distances from a point, along the first axis.

        Any direction would do -- a periodic box is homogeneous and isotropic
        -- and the first axis is the one that needs no frame.

        Args:
            point: where to start.
            distances: how far to go.

        Returns:
            The points.
        """
        position = self._as_point(point)
        result = []
        for distance in np.asarray(distances, dtype=float):
            moved = position.copy()
            moved[0] = moved[0] + float(distance)
            result.append(self._wrap(moved))
        return result

    # ----------------------------------------------------------------- #
    #                         The module structure                      #
    # ----------------------------------------------------------------- #

    def pointwise_multiply(self, x: np.ndarray, y: np.ndarray, /) -> np.ndarray:
        """Pointwise product of two fields.

        Exact only up to aliasing: the product of two band-limited fields is
        not band-limited, and the modes beyond the grid fold back. That is
        inherent to a spectral representation and not a defect of this method,
        but it means a product is less accurate than either factor.
        """
        return x * y

    def derivative_operator(self, /, *, axis: int = 0) -> LinearOperator:
        r"""``d/dx`` along one axis.

        Diagonal in the *complex* Fourier basis and block-diagonal in the real
        one this space uses: differentiating turns a cosine into a sine and
        back, so each conjugate pair rotates into itself with a factor of the
        wavenumber.

        **Modes fixed by conjugation are annihilated.** A Nyquist cosine's
        derivative is a sine that the truncation cannot hold, and the constant
        mode's derivative is zero. Dropping the first rather than aliasing it
        is the usual convention and the only one that stays in the space.

        Formally anti-self-adjoint in ``L2``, and lifted through the metric on
        a Sobolev space -- where it is not anti-self-adjoint, since
        differentiation does not commute with the Sobolev weight.

        Args:
            axis: which coordinate to differentiate along. Zero on a
                one-dimensional box, where it is the only choice.

        Returns:
            The operator.

        Raises:
            ValueError: if the axis is outside the box's dimensions.
        """
        if not 0 <= axis < self.spatial_dimension:
            raise ValueError(
                f"This box has {self.spatial_dimension} axes, got axis {axis}."
            )
        if self.order != 0.0:
            return lift_formal_adjoint(
                self.with_order(0.0).derivative_operator(axis=axis), self
            )

        packing = self._packing
        fixed = packing.fixed_indices.size
        rate = (
            2.0
            * np.pi
            * packing.wavenumbers[axis][fixed::2].astype(float)
            / self._lengths[axis]
        )

        def apply(components: np.ndarray, sign: float) -> np.ndarray:
            result = np.zeros(self.dim)
            cosines = components[fixed::2]
            sines = components[fixed + 1 :: 2]
            result[fixed::2] = -sign * rate * sines
            result[fixed + 1 :: 2] = sign * rate * cosines
            return result

        def value(x: np.ndarray) -> np.ndarray:
            return self.from_components(apply(self.to_components(x), 1.0))

        def adjoint(y: np.ndarray) -> np.ndarray:
            return self.from_components(apply(self.to_components(y), -1.0))

        return LinearOperator.from_callables(self, self, value, adjoint=adjoint)

    def with_order(
        self, order: float, /, *, length_scale: float | None = None
    ) -> PeriodicBox:
        """The same grid, viewed with a different Sobolev order.

        Args:
            order: the new order.
            length_scale: the new length scale, which sets where the Sobolev
                weight turns over. Kept as it is if omitted, so this is a
                change of order alone.

        Returns:
            The same domain and grid in the new metric, as :class:`Lebesgue` at
            order zero and :class:`Sobolev` otherwise.
        """
        return self._rebuilt(order=order, length_scale=length_scale)

    def _rebuilt(
        self,
        /,
        *,
        shape: Sequence[int] | None = None,
        order: float | None = None,
        length_scale: float | None = None,
    ) -> "PeriodicBox":
        """The same domain with some of its parameters changed.

        **Returns the D-3 subclass its order names** rather than the base
        class, which is what makes ``isinstance(X.with_order(0.0), Lebesgue)``
        true (REVIEW2 3.7). Each geometry over this one -- a circle, a torus, a
        bounded box -- overrides this and nothing else, so there is a single
        place per family that knows which class goes with which order.

        Args:
            shape: the new grid. Unchanged if omitted.
            order: the new Sobolev order. Unchanged if omitted.
            length_scale: the new Sobolev length scale. Unchanged if omitted.

        Returns:
            The space, as ``Lebesgue`` at order zero and ``Sobolev`` otherwise.
        """
        shape = self._shape if shape is None else tuple(int(n) for n in shape)
        order = self._order if order is None else float(order)
        scale = self._length_scale if length_scale is None else float(length_scale)
        if order == 0.0:
            return Lebesgue(shape, lengths=self._lengths)
        return Sobolev(shape, order, scale, lengths=self._lengths)


class Lebesgue(PeriodicBox):
    """The ``L2`` space on a periodic box, with an orthonormal spectral basis.

    A class rather than a factory function, so ``isinstance(x, Lebesgue)``
    answers what it looks like it answers. Nothing is added: it is
    :class:`PeriodicBox` at order zero.
    """

    def __init__(
        self, shape: Sequence[int], /, *, lengths: Sequence[float] | None = None
    ) -> None:
        """
        Args:
            shape: grid points along each axis.
            lengths: the period along each axis. Unit lengths by default.
        """
        super().__init__(shape, lengths=lengths, order=0.0)


class Sobolev(PeriodicBox):
    """The Sobolev space ``H^order`` on a periodic box.

    The same coordinate map as :class:`Lebesgue`, with
    ``(1 + length_scale^2 |k|^2)^order`` as its metric — so it is a
    diagonal-metric space rather than a mass-weighted one, and every invariant
    operator on it stays diagonal.
    """

    def __init__(
        self,
        shape: Sequence[int],
        order: float,
        length_scale: float,
        /,
        *,
        lengths: Sequence[float] | None = None,
    ) -> None:
        """
        Args:
            shape: grid points along each axis.
            order: the Sobolev order.
            length_scale: the length at which the Sobolev weight turns over.
            lengths: the period along each axis. Unit lengths by default.
        """
        super().__init__(shape, lengths=lengths, order=order, length_scale=length_scale)
