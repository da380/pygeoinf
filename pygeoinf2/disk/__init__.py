"""A disk or an annulus, in the eigenbasis of ``A = 1 - div(L^2 grad)``.

Fields of position in the plane, in ``inner_radius <= r <= radius``: the
two-dimensional counterpart of :mod:`pygeoinf2.ball`, and built the same way.
The length scale ``L`` may vary with radius, so ``A`` commutes with rotations
about the centre and its eigenfunctions separate: a Fourier mode of the angle
times a radial eigenfunction of the operator ``A_m`` of its order, under the
measure ``r dr``. A field's components are its coefficients in those
products, ordered by order ``m``, then the cosine mode and, for ``m >= 1``,
the sine mode, and within each mode the radial modes in ascending
eigenvalue; :attr:`Disk.orders` and :attr:`Disk.radial_indices` label them.

The radial mesh is padded, as the ball's is and for the same reason: ``A``
needs a boundary condition at each end, and the padding keeps its mark out
of the domain. Outwards the padding is a fictitious ring; inwards it stops at
the centre, which is a regular point of the operator and needs neither
padding nor a condition, so a disk is padded on the outside only.

**The conventions are the package's.** A vector is the array of the field's
values on the padded grid, shape ``(radii, angles)``, the angles being the
``2 mmax + 1`` equally spaced ones of the truncation order, from zero
anticlockwise; a product is left on the grid, untruncated; a point is
``(radius, angle)``, the angle in degrees anticlockwise from the ``x`` axis,
as a longitude is east of Greenwich on the sphere; the Fourier modes are
orthonormal on the unit circle -- ``1 / sqrt(2 pi)``, ``cos(m t) / sqrt(pi)``,
``sin(m t) / sqrt(pi)`` -- so the Lebesgue inner product is the integral over
the area; and point evaluation is refused at or below order one.

**A field regular at the centre grows there in a way its order fixes.** The
part of order ``m`` of a field analytic at the centre of a disk goes like
``r^m`` times a series in ``r^2``: a polynomial in ``x`` and ``y`` is the
safe example. The radial modes here do so by construction -- for a constant
``L`` they are the Bessel functions ``J_m(k r)``, to the accuracy of the mesh
-- so every draw of a prior is regular there automatically. A function
*handed in* has to be regular itself, or the expansion converges slowly at
the centre: a standard deviation, a multiplier or a mean given as a profile
in ``r`` is of order zero and must be even in ``r``. In an annulus there is no
centre and nothing to it.

The numerics are ``planetmodel``'s spectral elements under the ``r dr``
measure, which ``planetmodel`` does not offer: its radial operator family
with the assembly changed, one ``planetmodel`` ``SpectralBasis`` per order
held in a :class:`~pygeoinf2.spectral_element._cylindrical.PolarBasis`
(reachable as :attr:`Disk.basis`), and a real FFT round each ring.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, Callable, Hashable, Sequence

import numpy as np
from numpy.random import Generator
from planetmodel.randomfield import SpectralBasis, padded_mesh, restriction

from ..algebra.operators import LinearFunctional, LinearOperator
from ..algebra.spaces import EuclideanSpace, resolve_rng
from ..spectral_element import layered as _layered
from ..spectral_element._cylindrical import (
    CylindricalOperatorFamily,
    PolarBasis,
    RingTransform,
    circular_harmonics,
)
from ..spectral_element.base import (
    _EXTENSIONS,
    _NODES_PER_MODE,
    Boundary,
    LengthScale,
    SpectralElementSpace,
    _clamped_length_scale,
    _fitted_over_the_domain,
    _length_scale_key,
    _resolved_boundary,
    _resolved_padding,
    _sampled_with_extension,
)

__all__ = ["Disk", "Layered", "Lebesgue", "Sobolev"]

#: Relative margin on an eigenvalue bound, so that roundoff cannot drop the
#: mode that defines it.
_BOUND_RTOL = 1e-10


def _matched_robin(
    boundary: Boundary, probe: np.ndarray, ends: tuple[float, float], /
) -> tuple[float, float]:
    """The Robin coefficients at the two ends of the radial mesh.

    ``"robin"`` ends the mesh as if it went on: the condition the decaying
    solution of ``A u = 0`` satisfies there, ``u' = -+ gamma u``. In the plane
    that solution is ``K_0(r / L)``, which goes like ``exp(-r / L) /
    sqrt(r)``, so ``gamma = 1 / L + 1 / (2 r)`` at the outer end; at the inner
    the growing solution, ``I_0(r / L)``, gives ``gamma = 1 / L - 1 / (2 r)``,
    held at zero should it be negative. The curvature is half a ball's, the
    ``(d - 1) / 2r`` of a radius in ``d`` dimensions. As on the other
    geometries ``0.7 / L`` stands in for ``1 / L``, and an end at the centre
    takes no condition. Anything else is resolved as on a line.

    Raises:
        ValueError: for a name that is not ``"robin"``, or a negative
            coefficient.
    """
    lower, upper = _resolved_boundary(boundary, probe, None)
    if isinstance(boundary, str):
        lower = max(lower - 0.5 / ends[0], 0.0) if ends[0] > 0.0 else 0.0
        upper = upper + 0.5 / ends[1]
    return (lower, upper)


class Disk(SpectralElementSpace):
    """Fields on a disk, or on an annulus when ``inner_radius`` is positive."""

    #: The tag of the geometry, which names the layers of a layered space.
    _geometry: str = "disk"

    def __init__(
        self,
        mmax: int,
        /,
        *,
        radius: float = 1.0,
        inner_radius: float = 0.0,
        order: float = 0.0,
        length_scale: LengthScale = 1.0,
        radial_modes: int | None = None,
        max_eigenvalue: float | None = None,
        padding: float | tuple[float, float] | None = None,
        boundary: Boundary = "robin",
        ngll: int = 5,
        element_length: float | None = None,
    ) -> None:
        """
        Args:
            mmax: the largest angular order kept, which also fixes the grid of
                ``2 mmax + 1`` angles.
            radius: the outer radius.
            inner_radius: the inner radius. Zero, the default, is a disk.
            order: the Sobolev order. Zero gives the Lebesgue space.
            length_scale: ``L`` in ``A``, a number or a function of radius. It
                is asked about the domain only; the padding takes its value
                at the nearer end.
            radial_modes: keep this many radial modes at every order.
            max_eigenvalue: keep, at every order, the radial modes whose
                eigenvalue of ``A`` is at most this. With neither given it is
                the smallest eigenvalue of order ``mmax`` *on the domain
                proper*, without its padding: the truncation is then isotropic
                where it matters, the shortest wavelength kept being the
                angular one of order ``mmax`` at the outer radius. Should the
                mesh's own first mode of order ``mmax`` lie above that bound,
                as a Robin condition can make it, the bound is raised to it,
                so that every order keeps a mode.
            padding: how far the radial mesh extends past each end, one
                length or an ``(inwards, outwards)`` pair; inwards it stops at
                the centre. Two length scales by default; four under the
                natural condition.
            boundary: the condition at the ends of the radial mesh.
                ``"robin"``, the default, ends the mesh as if it went on,
                ``u' = -+ (0.7 / L +- 1 / 2 r) u``, and with it two length
                scales of padding do what four do under the natural
                condition. One coefficient serves every order. ``None`` is the
                natural condition, and doubles the default padding. A number
                or a pair gives the coefficients themselves; none is applied
                at the centre.
            ngll: Gauss-Lobatto-Legendre nodes per radial element.
            element_length: the longest a radial element may be. By default
                short enough for the shortest mode kept to be resolved, and
                no longer than half the smallest length scale.

        Raises:
            ValueError: if ``mmax`` is negative, the radii do not satisfy
                ``0 <= inner_radius < radius``, a length scale or the element
                length is not positive, both truncations are given, or some
                order has no mode within the truncation.
        """
        mmax = int(mmax)
        if mmax < 0:
            raise ValueError("mmax must be non-negative.")
        inner, outer = float(inner_radius), float(radius)
        if not 0.0 <= inner < outer:
            raise ValueError("The radii must satisfy 0 <= inner_radius < radius.")
        if radial_modes is not None and max_eigenvalue is not None:
            raise ValueError("Give radial_modes or max_eigenvalue, not both.")

        clamped, probe = _clamped_length_scale(length_scale, inner, outer)
        pads = _resolved_padding(padding, probe, boundary)
        pads = (min(pads[0], inner), pads[1])
        robin = _matched_robin(boundary, probe, (inner - pads[0], outer + pads[1]))

        ngll = int(ngll)
        if element_length is None:
            # The shortest wavelength kept, estimated before there is a mesh to
            # ask: a given eigenvalue bound is a wavenumber; a number of radial
            # modes is that many half waves along the mesh; and the first mode
            # of order m in a disk of radius R has wavenumber about
            # (m + 0.81 m^(1/3) + 1) / R, from the first zero of a Bessel
            # derivative, R being the outer radius of the domain.
            reach = outer + pads[1]
            if max_eigenvalue is not None:
                wavenumber = (
                    np.sqrt(max(float(max_eigenvalue) - 1.0, 0.0)) / probe.min()
                )
            elif radial_modes is not None:
                wavenumber = int(radial_modes) * np.pi / (reach - inner + pads[0])
            else:
                wavenumber = (mmax + 0.81 * mmax ** (1.0 / 3.0) + 1.0) / outer
                wavenumber *= probe.max() / probe.min()
            element_length = min(
                (ngll - 1) * np.pi / (_NODES_PER_MODE * max(wavenumber, 1e-300)),
                0.5 * float(probe.min()),
            )
        element_length = float(element_length)
        if element_length <= 0.0:
            raise ValueError("The element length must be positive.")

        # The r^2 mesh geometry is the one wanted: its inner pad stops at the
        # centre. The measure is the family's.
        mesh = padded_mesh(
            inner, outer, pad=pads, weight="r2", ngll=ngll, drmax=element_length
        )
        family = CylindricalOperatorFamily(
            mesh, kappa=lambda r: clamped(r) ** 2, robin=robin
        )
        physical = restriction(mesh, inner, outer)
        if radial_modes is not None:
            rule = {"nmodes": int(radial_modes)}
        else:
            if max_eigenvalue is None:
                # From the domain proper and not the padded mesh, whose first
                # mode of order mmax is longer by the ratio of the two radii;
                # and no lower than the padded mesh's own, so that every order
                # keeps one: first eigenvalues rise with the order.
                bare = CylindricalOperatorFamily(
                    padded_mesh(
                        inner,
                        outer,
                        pad=0.0,
                        weight="r2",
                        ngll=ngll,
                        drmax=element_length,
                    ),
                    kappa=lambda r: clamped(r) ** 2,
                )
                max_eigenvalue = max(
                    float(bare.eigvalsh(mmax)[0]), float(family.eigvalsh(mmax)[0])
                ) * (1.0 + _BOUND_RTOL)
            rule = {"theta_max": float(max_eigenvalue)}
        try:
            bases = [
                SpectralBasis(family, m, restrict=physical, **rule)
                for m in range(mmax + 1)
            ]
        except ValueError as error:
            raise ValueError(
                f"The truncation leaves some order up to {mmax} without a "
                f"radial mode, or asks for more than the mesh holds: {error}"
            ) from error
        self._basis = PolarBasis(bases)
        self._transform = RingTransform(mmax)

        self._mmax = mmax
        self._radii = (inner, outer)
        self._padding = pads
        self._robin = robin
        self._ngll = ngll
        self._element_length = element_length
        self._length_scale = length_scale
        self._length_scale_key = _length_scale_key(length_scale, clamped, mesh.rglob)
        super().__init__(self._basis.theta, order)

    # ----------------------------------------------------------------- #
    #                              Identity                             #
    # ----------------------------------------------------------------- #

    @property
    def mmax(self) -> int:
        """The largest angular order kept."""
        return self._mmax

    @property
    def radius(self) -> float:
        """The outer radius."""
        return self._radii[1]

    @property
    def inner_radius(self) -> float:
        """The inner radius: zero for a disk."""
        return self._radii[0]

    @property
    def bounds(self) -> tuple[float, float]:
        """The inner and outer radius: the two ends of the domain along the
        radial mesh."""
        return self._radii

    @property
    def padding(self) -> tuple[float, float]:
        """The radial padding inwards and outwards."""
        return self._padding

    @property
    def robin(self) -> tuple[float, float]:
        """The Robin coefficients at the two ends of the radial mesh, zero
        meaning the natural condition."""
        return self._robin

    @property
    def length_scale(self) -> LengthScale:
        """The length scale of ``A``, as it was given."""
        return self._length_scale

    @property
    def spatial_dimension(self) -> int:
        """Two."""
        return 2

    @property
    def domain_volume(self) -> float:
        """The area of the domain proper, excluding the padding."""
        inner, outer = self._radii
        return np.pi * (outer**2 - inner**2)

    @property
    def basis(self) -> PolarBasis:
        """The per-order eigenbases, ``planetmodel`` ``SpectralBasis``
        objects, which do the numerics and fix the order of the components."""
        return self._basis

    @property
    def grid(self) -> RingTransform:
        """The transform round every ring, which holds the angular grid: its
        angles in radians, and their common quadrature weight."""
        return self._transform

    @property
    def eigenvalues(self) -> np.ndarray:
        """The eigenvalue of ``A`` attached to each component: ascending
        within a mode, the same for the cosine and sine modes of an order,
        and at least one."""
        return self._basis.theta

    @cached_property
    def _labels(self) -> tuple[np.ndarray, np.ndarray]:
        """(signed order, radial index) of every component."""
        orders, indices = [], []
        for m, radial in enumerate(self._basis):
            for sign in (1,) if m == 0 else (1, -1):
                orders.append(np.full(radial.nmodes, sign * m))
                indices.append(np.arange(radial.nmodes))
        return np.concatenate(orders), np.concatenate(indices)

    @property
    def orders(self) -> np.ndarray:
        """The signed angular order of each component: non-negative for a
        cosine mode and negative for a sine one, as on the sphere."""
        return self._labels[0]

    @property
    def radial_indices(self) -> np.ndarray:
        """Which radial mode of its order each component is, from zero in
        ascending eigenvalue."""
        return self._labels[1]

    def _coordinate_key(self) -> Hashable:
        """The mesh, the operator and the truncation, with the metric left out.

        Tagged by geometry and not by ``type(self)``: ``Lebesgue`` and
        ``Sobolev`` are views of one set of fields.
        """
        return (
            "disk",
            self._mmax,
            tuple(int(n) for n in self._basis.nmodes),
            self._radii,
            self._padding,
            self._robin,
            self._ngll,
            self._element_length,
            self._length_scale_key,
        )

    def _class_for_order(self, order: float, /) -> type:
        return Lebesgue if order == 0.0 else Sobolev

    def __repr__(self) -> str:
        inner, outer = self._radii
        return (
            f"{type(self).__name__}(mmax={self._mmax}, dim={self.dim}, "
            f"radii=[{inner:g}, {outer:g}], order={self._order:g})"
        )

    # ----------------------------------------------------------------- #
    #                          The coordinate map                       #
    # ----------------------------------------------------------------- #

    def to_components(self, x: np.ndarray) -> np.ndarray:
        """The coefficients of the kept eigenfunctions: a Fourier analysis of
        every ring, then each mode's radial function projected onto the
        radial modes of its order, both by the grid's own quadrature."""
        return self._basis.analyse(self._transform.analyse(np.asarray(x, dtype=float)))

    def from_components(self, c: np.ndarray) -> np.ndarray:
        """The values on the padded grid: the radial syntheses, then a
        Fourier synthesis of every ring."""
        return self._transform.synthesise(self._basis.synthesise(c, physical=False))

    # ----------------------------------------------------------------- #
    #                               The grid                            #
    # ----------------------------------------------------------------- #

    @property
    def grid_shape(self) -> tuple[int, int]:
        """``(radii, angles)``, the radii those of the padded mesh."""
        return (self._basis.family.mesh.nglob, 2 * self._mmax + 1)

    @property
    def radii(self) -> np.ndarray:
        """The nodes of the padded radial mesh: the first axis of a vector."""
        return self._basis.family.mesh.rglob

    @property
    def interior_radii(self) -> np.ndarray:
        """The radial nodes of the domain proper, both ends included."""
        return self._basis.r

    @property
    def angles(self) -> np.ndarray:
        """The grid's angles in degrees, from zero anticlockwise: the second
        axis."""
        return np.degrees(self._transform.angles)

    @cached_property
    def interior_mask(self) -> np.ndarray:
        """A boolean grid array, true on the domain and false on the padding."""
        mask = np.zeros(self.grid_shape, dtype=bool)
        mask[self._basis.restriction.nodes] = True
        return mask

    @property
    def _interior_index(self) -> slice:
        return self._basis.restriction.nodes

    def project_function(
        self,
        function: Callable[[tuple[float, float]], float],
        /,
        *,
        extension: str = "fit",
    ) -> np.ndarray:
        """The field of this space that a function of position names.

        The function is never called outside the domain, where it need not be
        defined. ``"fit"``, the default, asks nothing of the padding: it
        returns the field in the span of the kept modes that best matches the
        function on the domain, mode by mode in the domain's own quadrature.
        For a function regular at the centre it is accurate everywhere, the
        centre included; it is what a mean, a reference model or a synthetic
        truth wants.

        ``"constant"`` and ``"odd"`` return the function's own values on the
        domain's part of the grid and continue them along each radius across
        the padding, by the value at the nearer end or by ``2 f(end) -
        f(mirror)``; see
        :meth:`~pygeoinf2.spectral_element.AxisSpace.project_function`. The
        constant keeps a positive function positive on the padding, which a
        standard deviation handed to ``pointwise_std=`` must be.

        Args:
            function: called with a point ``(radius, angle)`` of the domain,
                the angle in degrees anticlockwise from the ``x`` axis. A
                function of ``x`` and ``y`` is ``lambda p: f(p[0] *
                cos(radians(p[1])), p[0] * sin(radians(p[1])))``.
            extension: ``"fit"``, ``"odd"`` or ``"constant"``.

        Returns:
            The field. Under ``"fit"`` it is in the span of the kept modes;
            otherwise it holds the function's values on the domain's grid
            points.

        Raises:
            ValueError: for an extension that is none of these.
        """
        rings: dict[float, np.ndarray] = {}

        def sample(radii: np.ndarray) -> np.ndarray:
            for r in np.unique(radii):
                if float(r) not in rings:
                    rings[float(r)] = np.array(
                        [float(function((float(r), float(a)))) for a in self.angles]
                    )
            return np.stack([rings[float(r)] for r in radii])

        if extension not in _EXTENSIONS:
            raise ValueError("extension must be 'odd', 'constant' or 'fit'.")
        if extension == "fit":
            return self._fitted(sample(self.interior_radii))
        return _sampled_with_extension(self.radii, *self._radii, sample, extension)

    def _fitted(self, values: np.ndarray, /) -> np.ndarray:
        """The field in the span of the kept modes that best matches values
        given on the domain's own part of the grid."""
        functions = self._transform.analyse(values)
        components = np.zeros(self.dim)
        for m, radial in enumerate(self._basis):
            parts = 1 if m == 0 else 2
            interior = radial.modes()[radial.restriction.nodes]
            fitted = _fitted_over_the_domain(
                interior, radial.weights(), functions[:parts, m].T
            )
            start = self._basis.block(0, m).start
            components[start : start + fitted.size] = fitted.T.reshape(-1)
        return self.from_components(components)

    def random_point(self, *, rng: Generator | None = None) -> np.ndarray:
        """A point ``(radius, angle)`` of the domain, drawn uniformly over its
        area, the angle in degrees in ``[0, 360)``."""
        rng = resolve_rng(rng)
        inner, outer = self._radii
        return np.array(
            [
                float(np.sqrt(rng.uniform(inner**2, outer**2))),
                float(rng.uniform(0.0, 360.0)),
            ]
        )

    # ----------------------------------------------------------------- #
    #                        Integrals and points                       #
    # ----------------------------------------------------------------- #

    def integral_functional(self) -> LinearFunctional:
        """The integral of a field over the area of the domain proper.

        Only the mode of order zero has one, ``sqrt(2 pi)`` times the radial
        integral of its coefficient function against ``r``, taken by the
        quadrature of the domain's own elements so that the padding
        contributes nothing.
        """
        radial = self._basis[0]
        interior = radial.modes()[radial.restriction.nodes]
        components = np.zeros(self.dim)
        components[self._basis.block(0, 0)] = np.sqrt(2.0 * np.pi) * (
            radial.weights() @ interior
        )
        return LinearFunctional.from_derivative_components(self, components)

    def basis_matrix(self, points: Sequence[tuple[float, float]], /) -> np.ndarray:
        """The basis at many points ``(radius, angle)``, as a ``(len(points),
        dim)`` array: each Fourier mode at the angle times each radial mode of
        its order at the radius, the latter exact through the polynomial it is
        on each element.

        Raises:
            ValueError: if a radius lies outside the domain.
        """
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        angular = circular_harmonics(self._mmax, np.radians(points[:, 1]))
        matrix = np.empty((points.shape[0], self.dim))
        for m, radial in enumerate(self._basis):
            modes = radial.evaluate(points[:, 0])
            for s in (0,) if m == 0 else (0, 1):
                matrix[:, self._basis.block(s, m)] = angular[s, m][:, None] * modes
        return matrix

    def _squared_modes(self, weights: np.ndarray, /) -> np.ndarray:
        """``sum_k weights_k phi_k(p)^2`` on the grid.

        When the weights do not tell the cosine and sine modes of an order
        apart, which is so for every function of ``A``, the angular sum is
        done in closed form -- ``cos^2 + sin^2 = 1`` -- and the answer is a
        function of radius. Otherwise the modes are squared on the grid.
        """
        shape = self.grid_shape
        per_order = []
        isotropic = True
        for m, radial in enumerate(self._basis):
            parts = 1 if m == 0 else 2
            start = self._basis.block(0, m).start
            block = weights[start : start + parts * radial.nmodes]
            block = block.reshape(parts, radial.nmodes)
            isotropic = isotropic and bool(np.all(block == block[0]))
            per_order.append((radial.modes() ** 2) @ block.T)  # (radii, parts)
        if isotropic:
            profile = sum(
                (1.0 / (2.0 * np.pi) if m == 0 else 1.0 / np.pi) * squares[:, 0]
                for m, squares in enumerate(per_order)
            )
            return np.broadcast_to(profile[:, None], shape).copy()
        angular = circular_harmonics(self._mmax, self._transform.angles) ** 2
        out = np.zeros(shape)
        for m, squares in enumerate(per_order):
            out += squares[:, :1] * angular[0, m][None, :]
            if m:
                out += squares[:, 1:] * angular[1, m][None, :]
        return out


class Lebesgue(Disk):
    """The ``L2`` space on a disk or annulus."""

    def __init__(
        self,
        mmax: int,
        /,
        *,
        radius: float = 1.0,
        inner_radius: float = 0.0,
        length_scale: LengthScale = 1.0,
        radial_modes: int | None = None,
        max_eigenvalue: float | None = None,
        padding: float | tuple[float, float] | None = None,
        boundary: Boundary = "robin",
        ngll: int = 5,
        element_length: float | None = None,
    ) -> None:
        """
        Args:
            mmax: the largest angular order kept.
            radius, inner_radius: the radii; an inner radius of zero is a disk.
            length_scale: ``L`` in ``A``, which fixes the basis and the
                default padding though not this space's inner product.
            radial_modes, max_eigenvalue: the radial truncation at each
                order; isotropic when neither is given.
            padding: the radial mesh's reach past each end.
            boundary: the condition at the ends of the radial mesh; see
                :class:`Disk`.
            ngll: nodes per radial element.
            element_length: the longest a radial element may be.
        """
        super().__init__(
            mmax,
            radius=radius,
            inner_radius=inner_radius,
            order=0.0,
            length_scale=length_scale,
            radial_modes=radial_modes,
            max_eigenvalue=max_eigenvalue,
            padding=padding,
            boundary=boundary,
            ngll=ngll,
            element_length=element_length,
        )


class Sobolev(Disk):
    """The Sobolev space ``H^order`` on a disk or annulus, with the inner
    product ``(A^order u, v)``."""

    def __init__(
        self,
        mmax: int,
        order: float,
        length_scale: LengthScale,
        /,
        *,
        radius: float = 1.0,
        inner_radius: float = 0.0,
        radial_modes: int | None = None,
        max_eigenvalue: float | None = None,
        padding: float | tuple[float, float] | None = None,
        boundary: Boundary = "robin",
        ngll: int = 5,
        element_length: float | None = None,
    ) -> None:
        """
        Args:
            mmax: the largest angular order kept.
            order: the Sobolev order.
            length_scale: ``L`` in ``A``, a number or a function of radius:
                the length at which the Sobolev weight turns over.
            radius, inner_radius: the radii; an inner radius of zero is a disk.
            radial_modes, max_eigenvalue: the radial truncation at each
                order; isotropic when neither is given.
            padding: the radial mesh's reach past each end.
            boundary: the condition at the ends of the radial mesh; see
                :class:`Disk`.
            ngll: nodes per radial element.
            element_length: the longest a radial element may be.
        """
        super().__init__(
            mmax,
            radius=radius,
            inner_radius=inner_radius,
            order=order,
            length_scale=length_scale,
            radial_modes=radial_modes,
            max_eigenvalue=max_eigenvalue,
            padding=padding,
            boundary=boundary,
            ngll=ngll,
            element_length=element_length,
        )


class Layered(_layered.Layered):
    """Piecewise-continuous fields of position in the plane: a :class:`Disk`
    in each of the concentric rings between consecutive radii.

    A direct sum of the layers, one summand per layer on its own padded mesh
    with its own length scale and truncation, that knows its summands are
    adjacent in space: which layer a point is in, the field's two values at
    an interface and the jump between them, integrals over the whole, one
    function sampled into every layer, and priors given layer by layer, under
    which the layers are independent. It compares equal to the plain direct
    sum of its layers, which is what a block operator or a product measure
    lands on. Layers built by hand go in through :meth:`from_layers`.
    The jump across an interface, a circle, is taken at given angles.
    """

    def __init__(
        self,
        boundaries: Sequence[float],
        mmax: int | list[int],
        /,
        *,
        labels: Sequence[str] | None = None,
        **options: Any,
    ) -> None:
        """
        Args:
            boundaries: the radii bounding the rings, ascending. A first
                radius of zero makes the innermost layer a whole disk.
            mmax: the largest angular order kept in each layer, for all or as
                a list.
            labels: names for the layers, so one can be reached by name.
            **options: the keyword arguments of :class:`Disk` --
                ``order``, ``length_scale``, ``radial_modes``,
                ``max_eigenvalue``, ``padding``, ``boundary``, ``ngll``,
                ``element_length`` -- each given once for all layers or, as a
                list, layer by layer. A tuple, such as a padding pair, is
                one value.

        Raises:
            ValueError: for fewer than two boundaries, or a list of the
                wrong length.
        """
        super().__init__(
            _layered.layers_between(
                Disk, boundaries, mmax, ("inner_radius", "radius"), options
            ),
            labels=labels,
        )

    def jump_operator(
        self,
        interface: int,
        /,
        *,
        directions: Sequence[float] | None = None,
        unsafe: bool = False,
    ) -> LinearOperator:
        """The jump of the field across an interface: its value just outside,
        less its value just inside.

        What a reflection coefficient sees, and what a prior on a
        discontinuity is a statement about. Conditioning on its vanishing is
        how two layers are made continuous at the circle they share.

        Args:
            interface: which interface, by position among :attr:`interfaces`.
            directions: the angles, in degrees anticlockwise from the ``x``
                axis, at which the jump is taken.
            unsafe: build it even in layers too rough to admit it.

        Returns:
            The operator into a Euclidean space of one dimension per angle.

        Raises:
            ValueError: if no angles are given, or a layer's order is too low
                and *unsafe* is not set.
            IndexError: if there is no such interface.
        """
        position = self.interfaces[interface]
        if directions is None:
            raise ValueError("In disks the jump is taken at given directions.")
        points = [(position, float(angle)) for angle in np.ravel(directions)]
        if not points:
            raise ValueError("At least one direction is needed.")
        for side in ("below", "above"):
            self._require_point_evaluation(
                "A jump operator", points=points, side=side, unsafe=unsafe
            )
        matrix = self.basis_matrix(points, side="above") - self.basis_matrix(
            points, side="below"
        )
        return LinearOperator.from_matrix(
            self, EuclideanSpace(len(points)), matrix, form="galerkin"
        )
