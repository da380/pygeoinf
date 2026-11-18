"""
Periodic boxes in any dimension, through the real FFT.

One implementation for every dimension: the circle is a one-dimensional box,
the torus a two-dimensional one, and a periodic cube a three-dimensional one.

Fields are grid arrays; components are coefficients in a real Fourier basis
that is **orthonormal in L2**, so the Lebesgue space has an identity Gram
matrix. The Sobolev space is the
same coordinate map with the Sobolev symbol as its metric, so it is a
``DiagonalMetricSpace`` too and no mass-weighted construction is needed.

The basis, its packing and the transforms are those of
``planetmodel.fourier``, a box of the space's periods from the origin: each
conjugate orbit of the grid's modes is one component if it is a fixed point
of conjugation and two, a cosine and a sine, if it is a genuine pair; the
self-conjugate modes come first, in the flat order of ``rfftn``'s half
spectrum, then each pair's cosine and sine together. A pair contributes
``sqrt(2/V) cos(k.r)`` and ``-sqrt(2/V) sin(k.r)``, a self-conjugate mode
``cos(k.r) / sqrt(V)``, with ``k`` the physical wavevector and ``V`` the
volume.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, Callable, Hashable, Sequence

import numpy as np
from numpy.random import Generator
from planetmodel.fourier import PointTransform, Transform
from scipy.fft import irfftn, rfftn

from ..algebra.operators import LinearOperator
from ..algebra.spaces import ArrayVectorMixin, resolve_rng
from .base import PreparedPoints, SymmetricSpace, _gauss_legendre, lift_formal_adjoint

__all__ = ["PeriodicBox", "Lebesgue", "Sobolev"]


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
        # The periods from the origin: a torus, whose box is exactly these
        # lengths. A bounded box moves its points into it.
        self._transform = Transform(shape, lengths)
        self._packing = self._transform.packing
        self._volume = float(np.prod(lengths))

        # The space's own copy: what it hands out stays writable, as the
        # transform's shared tables are not.
        eigenvalues = np.array(self._transform.laplacian_eigenvalues)
        self._laplacian_eigenvalues = eigenvalues
        metric = (
            np.ones(self._packing.size)
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
        """The magnitude of each component's integer wavevector, rounded down.

        The analogue of a harmonic degree. It is counted in the box's own
        grid units, the wavevector ``k`` of the mode ``exp(2 pi i k . x /
        L)``, so components sharing a degree are the modes of a common
        spatial scale only when every axis has the same period; on a box
        whose periods differ, one degree groups wavelengths that differ by
        the ratio of the periods, and :meth:`power_measure`,
        :meth:`power_spectrum` and a band limit by degree inherit that. The
        physical wavenumbers are in :attr:`laplacian_eigenvalues`.
        """
        return np.array(self._packing.degrees)

    @cached_property
    def wavevectors(self) -> np.ndarray:
        """The integer wavevector of each component, one column per component.

        Shape ``(spatial_dimension, dim)``, signed, in the box's own grid
        units: the mode ``exp(2 pi i k . x / L)`` with ``k`` the column. A
        cosine and a sine component of one conjugate pair share a column and
        differ in :attr:`phases`. With that array this is the packing made
        public, as :attr:`degrees` is its magnitude rounded down.
        """
        return np.array(self._packing.wavevectors)

    @cached_property
    def phases(self) -> np.ndarray:
        """Zero for a cosine (or a self-conjugate mode), one for a sine."""
        return np.array(self._packing.phases)

    def component_of(self, wavevector: Any, /, *, phase: Any = 0) -> Any:
        """The position of the component with a given wavevector and phase.

        Vectorized: one wavevector of length ``spatial_dimension`` gives an
        integer, an array of shape
        ``(spatial_dimension, n)`` gives ``n`` positions. Wavevectors are
        taken modulo the grid, so a mode may be named by either sign of its
        aliased wavenumber.

        Args:
            wavevector: the integer wavevector, or several as columns.
            phase: zero for the cosine, one for the sine; a scalar or one
                per column.

        Returns:
            The index into the components.

        Raises:
            ValueError: for a label the space does not hold -- a sine of a
                self-conjugate mode, or the wrong number of wavenumbers.
        """
        wavevector = np.asarray(wavevector, dtype=int)
        if wavevector.shape[0] != len(self._packing.shape):
            raise ValueError(
                f"A wavevector on this box has {len(self._packing.shape)} "
                f"components, got shape {wavevector.shape}."
            )
        try:
            return self._packing.component_of(wavevector, phase=phase)
        except ValueError:
            raise ValueError(
                "No component has that wavevector and phase; a self-conjugate "
                "mode has no sine, and a wavevector outside the grid aliases."
            ) from None

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
        # The length scale enters the metric only above order zero.
        scale = self._length_scale if self._order != 0.0 else None
        return (self._shape, self._lengths, self._order, scale)

    def _coordinate_key(self) -> Hashable:
        """The grid, which the order and length scale do not touch.

        Tagged by geometry rather than by ``type(self)``, as the sphere's is
        and for the same reason: ``Circle``, ``Torus``, ``Lebesgue`` and
        ``Sobolev`` are thin subclasses over one grid and one point map, so
        keying on the concrete class would make two views of one field
        different fields, and a formal-adjoint lift between them would
        round-trip through components instead of passing the vector straight
        through.
        """
        return ("periodic_box", self._shape, self._lengths)

    def __repr__(self) -> str:
        kind = "Lebesgue" if self._order == 0.0 else f"Sobolev(order={self._order})"
        return f"PeriodicBox({self._shape}, {kind})"

    def _domain_key(self) -> Hashable:
        """The domain and where its points are, whatever the grid: what two
        boxes must share for a field to move between them by its modes."""
        return ("periodic_box", self._lengths)

    # ----------------------------------------------------------------- #
    #                             Coordinates                           #
    # ----------------------------------------------------------------- #

    def to_components(self, x: np.ndarray) -> np.ndarray:
        """Spectral coefficients of a field, orthonormal in L2.

        With ``X = rfftn(x)``, Parseval on the grid gives ``integral |x|^2 ==
        (V / N^2) sum_all_k |X_k|^2``, so a self-conjugate mode's component
        is ``sqrt(V)/N`` times ``Re(X)`` and a conjugate pair's are
        ``sqrt(2V)/N`` times its real and imaginary parts.
        """
        return self._transform.from_rfft(
            rfftn(np.asarray(x, dtype=float), s=self._shape)
        )

    def from_components(self, c: np.ndarray) -> np.ndarray:
        """The field with the given spectral coefficients."""
        return self._transform.synthesise_grid(c)

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
        # A conjugate pair contributes a cosine and a sine of amplitude
        # sqrt(2/V); a fixed point of conjugation -- the constant mode, and the
        # Nyquist modes on even axes -- contributes a single cosine of
        # amplitude 1/sqrt(V), because it appears once in the spectrum rather
        # than twice.
        return self._transform.basis(position)

    # ----------------------------------------------------------------- #
    #                     Evaluation by non-uniform FFT                 #
    # ----------------------------------------------------------------- #

    @cached_property
    def _point_transform(self) -> Transform | None:
        """The transform that evaluates at scattered points through finufft.

        ``None`` above three dimensions, which is as far as finufft goes, and
        also ``None`` when finufft is not installed. Both send
        :meth:`evaluate` and :meth:`accumulate` back to the direct sum, which
        is slower but identical, and :meth:`evaluate_gradient` to central
        differences.

        finufft's spectrum is **two larger than the grid along every axis**.
        That is not padding for accuracy: a Nyquist wavenumber is ``+n/2``,
        which lies outside the mode range ``[-N/2, N/2-1]`` that finufft
        indexes, and folding it back to ``-n/2`` changes the sign of its phase
        on every axis it appears. Widened, ``+k`` and ``-k`` are distinct
        entries for every mode, so fixed points of conjugation and genuine
        pairs need no separate handling at all.
        """
        if self.spatial_dimension > 3:
            return None
        try:
            import finufft  # noqa: F401
        except ImportError:  # pragma: no cover - depends on the install
            return None
        return self._transform

    def _at(self, points: Sequence[Any], eps: float, nthreads: int) -> PointTransform:
        """``planetmodel``'s transform at the points, to ``eps`` with
        ``nthreads`` threads: built on first use and kept with the prepared
        points, for this grid and box, accuracy and thread count."""
        prepared = self.prepare_points(points)
        transform = self._point_transform
        key = ("fourier", transform.shape, transform.domain, eps, nthreads)
        return prepared._cached(
            key,
            lambda: transform.at(prepared.data, epsilon=eps, nthreads=nthreads),
        )

    def prepare_points(self, points: Sequence[Any], /) -> PreparedPoints:
        """The points as one contiguous array of rows, one per point.

        The conversion the non-uniform FFT starts with, a Python loop over
        the points unless they come as an array of rows, done once for an
        operator rather than once per application. A coordinate outside
        ``[0, L)`` is the same point of the periodic box as its image
        inside.

        The prepared points also keep ``planetmodel``'s transform at them,
        ``Transform.at``, one for each accuracy and thread count they are
        evaluated with, built at the first evaluation: each coordinate taken
        to its angle ``2 pi x / L`` once, and finufft's plan of each type of
        transform, the points sorted, made at its first application and
        reused at every later one.

        Args:
            points: points of the box, or an already prepared set.

        Returns:
            The prepared points, carrying their coordinates as an array of
            shape ``(len(points), spatial_dimension)``.

        Raises:
            ValueError: if the points do not have this box's number of
                coordinates.
        """
        if isinstance(points, PreparedPoints):
            return points
        if isinstance(points, np.ndarray) and points.ndim == 2:
            # An array of rows is taken as it is: a tracer hands over
            # thousands of points at once, and a loop over them would cost
            # more than the transform.
            positions = np.asarray(points, dtype=float)
        else:
            points = tuple(points)
            positions = np.asarray(
                [np.atleast_1d(np.asarray(point, dtype=float)) for point in points]
            )
        if positions.ndim != 2 or positions.shape[1] != self.spatial_dimension:
            raise ValueError(
                f"Points need {self.spatial_dimension} coordinates each, got "
                f"shape {positions.shape}."
            )
        return PreparedPoints(points, data=np.array(positions, dtype=float, order="C"))

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
        transform = self._point_transform
        if transform is None:
            return super().evaluate(x, points)
        components = self.to_components(x)
        values = self._at(points, eps, nthreads).synthesise(components)
        return np.ascontiguousarray(values)

    def evaluate_gradient(
        self,
        x: np.ndarray,
        points: Sequence[Any],
        /,
        *,
        eps: float | None = None,
        nthreads: int | None = None,
    ) -> np.ndarray:
        """The gradient at scattered points: the spectrum times ``i k``, transformed.

        Each axis's derivative multiplies the mode ``exp(i k . θ)`` by
        ``i k 2π / L``, with ``L`` the period of that axis, and the partial
        derivatives of every axis go through one batched type-2 transform.
        Without finufft, central differences of :meth:`evaluate` at a step
        of a thousandth of the grid spacing.

        Args:
            x: a field on this box.
            points: where to evaluate the gradient.
            eps: the NUFFT's requested accuracy; ``1e-12`` by default.
            nthreads: threads for it; one by default, as in :meth:`evaluate`.

        Returns:
            An array of shape ``(len(points), spatial_dimension)``.
        """
        transform = self._point_transform
        points = self.prepare_points(points)
        if transform is None:
            return self._gradient_by_differences(x, points)
        components = self.to_components(x)
        values = self._at(
            points,
            1.0e-12 if eps is None else eps,
            1 if nthreads is None else nthreads,
        ).synthesise_gradient(components)
        return np.ascontiguousarray(values)

    def _gradient_by_differences(self, x: np.ndarray, points: Any) -> np.ndarray:
        """Central differences along each axis, for a box without the transform."""
        positions = np.stack(
            [np.atleast_1d(np.asarray(point, dtype=float)) for point in points.points]
        )
        gradient = np.empty((positions.shape[0], self.spatial_dimension))
        for axis in range(self.spatial_dimension):
            step = 1.0e-3 * self._lengths[axis] / self._shape[axis]
            offset = np.zeros(self.spatial_dimension)
            offset[axis] = step
            forward = self.evaluate(x, self._prepared_nearby(positions + offset))
            backward = self.evaluate(x, self._prepared_nearby(positions - offset))
            gradient[:, axis] = (forward - backward) / (2.0 * step)
        return gradient

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
        transform = self._point_transform
        if transform is None:
            return super().accumulate(weights, points)
        return self._at(points, eps, nthreads).accumulate(weights)

    def project_function(self, function: Callable[[Any], float], /) -> np.ndarray:
        """Sample a function on the grid.

        The function receives a point: a float on a one-dimensional box, an
        array of coordinates otherwise.
        """
        mesh = np.meshgrid(*self.grid_axes, indexing="ij")
        points = np.stack([m.ravel() for m in mesh], axis=1)
        values = np.array([float(function(p if p.size > 1 else p[0])) for p in points])
        return values.reshape(self._shape)

    def random_point(self, *, rng: Generator | None = None) -> np.ndarray:
        """A point drawn uniformly from the box."""
        generator = resolve_rng(rng)
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
        :meth:`~pygeoinf.symmetric_space.base.SymmetricSpace.geodesic_quadrature`
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

        abscissae, weights = _gauss_legendre(count)
        fractions = 0.5 * (abscissae + 1.0)
        nodes = list(self._wrap(first[None, :] + fractions[:, None] * offset[None, :]))
        return nodes, 0.5 * length * weights

    def _wrap(self, point: np.ndarray, /) -> np.ndarray:
        """A point brought back into ``[0, L)`` on each axis."""
        return np.asarray(point, dtype=float) % np.asarray(self._lengths, dtype=float)

    def _prepared_nearby(self, positions: np.ndarray, /) -> PreparedPoints:
        """Rows of positions a finite difference away from points of the box,
        prepared for evaluation; wrapped, since the box is periodic."""
        return self.prepare_points(self._wrap(positions))

    def to_coefficients(self, x: np.ndarray, /) -> np.ndarray:
        """The field's complex Fourier coefficients, in numpy's convention.

        For interchange with other FFT code. Components are this library's
        packed *real* vector, ordered by the packing and scaled to be
        orthonormal in ``L2``; this is exactly what ``rfftn`` returns for the
        same field -- half the spectrum, on the unnormalized scale numpy and
        scipy share, shaped like the grid. The two differ by a factor: a
        component is the real or imaginary part of its coefficient times
        ``sqrt(V) / N``, with ``V`` the box's volume and ``N`` its number of
        grid points, and times a further ``sqrt(2)`` where the coefficient has
        a conjugate partner.

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
            raise ValueError(f"Coefficients have shape {expected}, got {array.shape}.")
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

    def _extended_to(self, lmax: int, /) -> "PeriodicBox":
        # One wavenumber more than the band asks for: the modes of degree
        # ``lmax`` have every |k_i| <= lmax, and at exactly the Nyquist
        # wavenumber a mode is a single component rather than a pair.
        return self.with_degree(lmax + 1)

    def _degree_eigenvalues(self, degree: int, /) -> np.ndarray:
        """``|k|^2`` over the integer lattice shell ``degree <= |k| < degree + 1``.

        One entry per lattice point: a conjugate pair gives a cosine and a
        sine component, two components for its two points, and no mode of
        the shell is a fixed point of conjugation, so the count is the
        packing's own.
        """
        axis = np.arange(-degree, degree + 1)
        grids = np.meshgrid(*[axis] * self.spatial_dimension, indexing="ij")
        wavenumbers = np.stack([g.ravel() for g in grids]).astype(float)
        magnitude = np.floor(np.sqrt(np.sum(wavenumbers**2, axis=0))).astype(int)
        shell = wavenumbers[:, magnitude == degree]
        squared = np.zeros(shell.shape[1])
        for index, length in enumerate(self._lengths):
            squared += (2.0 * np.pi * shell[index] / length) ** 2
        return squared

    def with_degree(self, degree: int, /) -> "PeriodicBox":
        """The same domain, resolved to a given wavenumber on every axis.

        The counterpart of :meth:`~pygeoinf.symmetric_space.sphere.Sphere.with_degree`.
        A grid of ``n`` points along an axis carries wavenumbers up to
        ``n // 2``, so this asks for ``2 * degree`` points on each -- a band
        limit isotropic in the grid's wavenumbers, which is isotropic in
        space only when the periods are equal (see :attr:`degrees`). Where
        the axes should be resolved differently, say so with
        :meth:`with_shape`.

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

    def degree_transfer_operator(self, target: "PeriodicBox", /) -> LinearOperator:
        """Truncation to, or prolongation into, another grid.

        Components are matched by their ``(wavevector, cosine-or-sine)`` label,
        not by the degree they fall in: on a box many wavevectors share a
        degree, so matching on that would pair up unrelated modes. Anything the
        target does not have is dropped, and anything it has and this does not
        is left zero.

        The adjoint is derived through the metrics rather than written down,
        which matters: the adjoint of a restriction is the matching
        zero-padding only when the two spaces carry the same metric on their
        shared components, and carries the ratio of the two metrics
        otherwise.

        Args:
            target: the space to map into.

        Returns:
            The operator.

        Raises:
            ValueError: if the two boxes are not the same domain.
        """
        if not isinstance(target, PeriodicBox) or (
            target._domain_key() != self._domain_key()
        ):
            raise ValueError(
                f"Degree transfer needs a common domain; {target!r} is not on "
                f"the domain of {self!r}."
            )

        # Matched by signed wavevector and phase. A mode at the Nyquist
        # wavenumber of one grid is a conjugate pair on a finer one, and the
        # same cosine ``a cos(k.r)`` has coefficient ``a sqrt(V)`` as a fixed
        # point of conjugation and ``a sqrt(V/2)`` as the cosine member of a
        # pair. Copying the coefficient would scale the function by
        # ``sqrt(2)``; the scale carries the ratio of the two, so the
        # function is kept.
        #
        # That ratio is the grid's convention for a Nyquist cosine: the
        # inner product is the grid's quadrature, exact for every product of
        # two modes except at the Nyquist wavenumber of an even axis, where
        # ``cos^2`` aliases onto a constant, so the cosine stored once by the
        # real FFT has squared norm ``V`` on the grid where its continuous
        # norm is ``V/2``. The grid's convention is kept so that a pointwise
        # product of two grid functions is exactly self-adjoint and the
        # module structure every nonlinear forward leans on is exact.
        keep, place, scale = self._packing.matching(target._packing)

        def value(x: np.ndarray) -> np.ndarray:
            components = np.zeros(target.dim)
            components[place] = scale * self.to_components(x)[keep]
            return target.from_components(components)

        def derivative_components(y: np.ndarray) -> np.ndarray:
            pulled = target.apply_gram(target.to_components(y))
            total = np.zeros(self.dim)
            total[keep] = scale * pulled[place]
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
        self, center: Any, radius: float, /, *, count: int
    ) -> tuple[list[Any], np.ndarray]:
        """Nodes and weights integrating over a ball, so they sum to its volume.

        Gauss-Legendre in the radius, carrying the ``r^(d-1)`` element, crossed
        with a uniform rule over directions. In one dimension the ball is an
        interval and the directions are the two signs; in two it is a disc and
        they are equally spaced angles, where a uniform rule is exact for the
        trigonometric polynomials it meets.

        Args:
            center: the ball's center.
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
        first = self._as_point(center)

        if dimension == 1:
            abscissae, weights = _gauss_legendre(max(count, 1))
            offsets = radius * abscissae
            nodes = list(self._wrap(first[None, :] + offsets[:, None]))
            return nodes, radius * weights

        rings = max(1, int(np.sqrt(count)))
        spokes = max(1, count // rings)
        abscissae, weights = _gauss_legendre(rings)
        radii = 0.5 * radius * (abscissae + 1.0)
        # r dr from the area element, and the half-width of the mapping.
        radial = 0.5 * radius * weights * radii
        angles = 2.0 * np.pi * np.arange(spokes) / spokes

        # The outer product of rings and spokes, taken as one array rather
        # than as a double Python loop wrapping a point at a time.
        directions = np.column_stack([np.cos(angles), np.sin(angles)])
        offsets = (radii[:, None, None] * directions[None, :, :]).reshape(-1, 2)
        nodes = list(self._wrap(first[None, :] + offsets))
        values = np.repeat(radial * 2.0 * np.pi / spokes, spokes)
        return nodes, values

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

        transform = self._transform

        def value(x: np.ndarray) -> np.ndarray:
            return self.from_components(
                transform.derivative(self.to_components(x), axis)
            )

        def adjoint(y: np.ndarray) -> np.ndarray:
            return self.from_components(
                -transform.derivative(self.to_components(y), axis)
            )

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

        **Returns the subclass its order names**, :class:`Lebesgue` or
        :class:`Sobolev`, rather than the base class, which is what makes
        ``isinstance(X.with_order(0.0), Lebesgue)`` true. Each geometry over
        this one -- a circle, a torus, a bounded box -- overrides this and
        nothing else, so there is a single place per family that knows which
        class goes with which order.

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
            lengths: the period along each axis. ``2 pi`` on every axis by
                default.
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
            lengths: the period along each axis. ``2 pi`` on every axis by
                default.
        """
        super().__init__(shape, lengths=lengths, order=order, length_scale=length_scale)
