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

from .base import SymmetricSpace

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


class PeriodicBox(SymmetricSpace):
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
        positions = np.asarray(
            [np.atleast_1d(np.asarray(point, dtype=float)) for point in points]
        )
        if positions.ndim != 2 or positions.shape[1] != self.spatial_dimension:
            raise ValueError(
                f"Points need {self.spatial_dimension} coordinates each, got "
                f"shape {positions.shape}."
            )
        return [
            np.ascontiguousarray(2.0 * np.pi * positions[:, axis] / self._lengths[axis])
            for axis in range(self.spatial_dimension)
        ]

    def evaluate(
        self, x: np.ndarray, points: Sequence[Any], /, *, eps: float = 1.0e-12
    ) -> np.ndarray:
        """Field values at scattered points, by type-2 non-uniform FFT.

        The direct sum costs one basis evaluation per point, so ``len(points)``
        times the dimension. The NUFFT costs a padded FFT plus a local spread,
        which is what makes a tomography problem on a fine grid tractable.

        Falls back to the direct sum above three dimensions or without finufft.
        """
        import finufft

        layout = self._nufft_layout
        if layout is None:
            return super().evaluate(x, points)
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
        )
        return np.ascontiguousarray(np.atleast_1d(values).real)

    def accumulate(
        self,
        weights: np.ndarray,
        points: Sequence[Any],
        /,
        *,
        eps: float = 1.0e-12,
    ) -> np.ndarray:
        """The derivative components of ``x -> sum_i y_i x(r_i)``.

        The adjoint of :meth:`evaluate`, and a type-1 NUFFT is *literally* the
        adjoint of the type-2 one — so this is the same transform run the other
        way rather than a second implementation to keep in step.
        """
        import finufft

        layout = self._nufft_layout
        if layout is None:
            return super().accumulate(weights, points)
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

    def with_order(
        self, order: float, /, *, length_scale: float | None = None
    ) -> PeriodicBox:
        """The same grid, viewed with a different Sobolev order."""
        return PeriodicBox(
            self._shape,
            lengths=self._lengths,
            order=order,
            length_scale=(self._length_scale if length_scale is None else length_scale),
        )


def Lebesgue(
    shape: Sequence[int], /, *, lengths: Sequence[float] | None = None
) -> PeriodicBox:
    """The ``L2`` space on a periodic box, with an orthonormal spectral basis."""
    return PeriodicBox(shape, lengths=lengths, order=0.0)


def Sobolev(
    shape: Sequence[int],
    order: float,
    length_scale: float,
    /,
    *,
    lengths: Sequence[float] | None = None,
) -> PeriodicBox:
    """The Sobolev space ``H^order`` on a periodic box.

    The same coordinate map as :func:`Lebesgue`, with
    ``(1 + length_scale^2 |k|^2)^order`` as its metric — so it is a diagonal-metric space rather than a mass-weighted
    one, and every invariant operator on it stays diagonal.
    """
    return PeriodicBox(shape, lengths=lengths, order=order, length_scale=length_scale)
