"""A ball or an annulus, in the eigenbasis of ``A = 1 - div(L^2 grad)``.

Fields of position in ``inner_radius <= r <= radius``, which no periodic box
embeds with the right measure and no symmetric space here covers. The length
scale ``L`` may vary with radius, so ``A`` commutes with rotations and its
eigenfunctions separate: a real spherical harmonic in the angles times a
radial eigenfunction of the operator ``A_l`` of its degree, under the measure
``r^2 dr``. A field's components are its coefficients in those products,
ordered by degree ``l``, then the cosine harmonics of orders ``0 ... l``, then
the sine harmonics of orders ``1 ... l``, and within each harmonic the radial
modes in ascending eigenvalue; :attr:`Ball.degrees`, :attr:`Ball.orders` and
:attr:`Ball.radial_indices` label them.

The radial mesh is padded, as the interval's is and for the same reason:
``A`` needs a boundary condition at each end, and the padding keeps its mark
out of the domain. Outwards the padding is a fictitious shell; inwards it
stops at the centre, which is a regular point of the operator and needs
neither padding nor a condition, so a ball is padded on the outside only.

**The conventions are the package's.** A vector is the array of the field's
values on the padded grid, shape ``(radii, latitudes, longitudes)``, the
angular grid being the Gauss-Legendre one of the truncation degree; a product
is left on the grid, untruncated (DECISIONS.md D-87); a point is ``(radius,
latitude, longitude)`` with the angles in degrees (D-2); the harmonics are
orthonormal on the unit sphere without the Condon-Shortley phase (D-86); and
point evaluation is refused at or below order three halves (D-11).

**A field regular at the centre grows there in a way its degree fixes.** The
part of degree ``l`` of a field analytic at the centre of a ball goes like
``r^l`` times a series in ``r^2``: ``u ~ r^l Y_lm``. The radial modes here do
so by construction -- for a constant ``L`` they are the spherical Bessel
functions ``j_l(k r)``, to the accuracy of the mesh -- so every draw of a
prior is regular for nothing. A function *handed in* has to be regular
itself, or the expansion converges slowly at the centre: a standard
deviation, a multiplier or a mean given as a profile in ``r`` is of degree
zero and must be even in ``r``, and one linear in ``r`` is a cone. Measured,
the standard deviation a prior realizes at the centre under such a profile is
three per cent out at twelve radial modes and still 0.6 at forty-eight,
against a hundredth of that for a profile in ``r^2``. Targets in tests are
chosen accordingly. In an annulus there is no centre and nothing to it.

The numerics are ``planetmodel``'s: ``RadialOperatorFamily`` under the
``r^2`` measure, one ``SpectralBasis`` per degree held in a
``SphericalBasis`` (reachable as :attr:`Ball.basis`), and the grid transforms
of ``planetmodel.harmonics``, which go through ``pyshtools``.
"""

from __future__ import annotations

from functools import cached_property
from typing import Callable, Hashable, Sequence

import numpy as np
from numpy.random import Generator
from planetmodel import harmonics
from planetmodel.randomfield import (
    RadialOperatorFamily,
    SpectralBasis,
    SphericalBasis,
    padded_mesh,
    restriction,
)
from planetmodel.sampling import AngularGrid, gauss_legendre

from ..algebra.operators import LinearFunctional
from .base import (
    _NODES_PER_MODE,
    LengthScale,
    SpectralElementSpace,
    _clamped_length_scale,
    _length_scale_key,
    _resolved_padding,
    _sampled_with_extension,
)

__all__ = ["Ball", "Lebesgue", "Sobolev"]

#: Relative margin on an eigenvalue bound, so that roundoff cannot drop the
#: mode that defines it.
_BOUND_RTOL = 1e-10


class Ball(SpectralElementSpace):
    """Fields on a ball, or on an annulus when ``inner_radius`` is positive."""

    def __init__(
        self,
        lmax: int,
        /,
        *,
        radius: float = 1.0,
        inner_radius: float = 0.0,
        order: float = 0.0,
        length_scale: LengthScale = 1.0,
        radial_modes: int | None = None,
        max_eigenvalue: float | None = None,
        padding: float | tuple[float, float] | None = None,
        ngll: int = 5,
        element_length: float | None = None,
    ) -> None:
        """
        Args:
            lmax: the largest spherical-harmonic degree kept, which also fixes
                the angular grid.
            radius: the outer radius.
            inner_radius: the inner radius. Zero, the default, is a ball.
            order: the Sobolev order. Zero gives the Lebesgue space.
            length_scale: ``L`` in ``A``, a number or a function of radius. It
                is asked about the domain only; the padding takes its value
                at the nearer end.
            radial_modes: keep this many radial modes at every degree.
            max_eigenvalue: keep, at every degree, the radial modes whose
                eigenvalue of ``A`` is at most this. With neither given it is
                the smallest eigenvalue of degree ``lmax`` *on the domain
                proper*, without its padding: the truncation is then isotropic
                where it matters, the shortest wavelength kept being the
                angular one of degree ``lmax`` at the outer radius. The padded
                mesh holds more such modes than the domain, by about the ratio
                of the two volumes, and the dimension pays for it.
            padding: how far the radial mesh extends past each end, one
                length or an ``(inwards, outwards)`` pair; inwards it stops at
                the centre. Four length scales by default.
            ngll: Gauss-Lobatto-Legendre nodes per radial element.
            element_length: the longest a radial element may be. By default
                short enough for the shortest mode kept to be resolved, and
                no longer than half the smallest length scale.

        Raises:
            ValueError: if the radii are out of order, a length scale is not
                positive, both truncations are given, or some degree has no
                mode within the truncation.
        """
        lmax = int(lmax)
        if lmax < 0:
            raise ValueError("lmax must be non-negative.")
        inner, outer = float(inner_radius), float(radius)
        if not 0.0 <= inner < outer:
            raise ValueError("The radii must satisfy 0 <= inner_radius < radius.")
        if radial_modes is not None and max_eigenvalue is not None:
            raise ValueError("Give radial_modes or max_eigenvalue, not both.")

        clamped, probe = _clamped_length_scale(length_scale, inner, outer)
        pads = _resolved_padding(padding, probe)
        pads = (min(pads[0], inner), pads[1])

        ngll = int(ngll)
        if element_length is None:
            # The shortest wavelength kept, estimated before there is a mesh to
            # ask: a given eigenvalue bound is a wavenumber; a number of radial
            # modes is that many half waves along the mesh; and the first mode
            # of degree l in a ball of radius R has wavenumber about
            # (l + 2 l^(1/3) + 1) / R, the first zero of a Bessel derivative, R
            # being the outer radius of the domain, which sets the default.
            reach = outer + pads[1]
            if max_eigenvalue is not None:
                wavenumber = (
                    np.sqrt(max(float(max_eigenvalue) - 1.0, 0.0)) / probe.min()
                )
            elif radial_modes is not None:
                wavenumber = int(radial_modes) * np.pi / (reach - inner + pads[0])
            else:
                wavenumber = (lmax + 2.0 * lmax ** (1.0 / 3.0) + 1.0) / outer
                wavenumber *= probe.max() / probe.min()
            element_length = min(
                (ngll - 1) * np.pi / (_NODES_PER_MODE * max(wavenumber, 1e-300)),
                0.5 * float(probe.min()),
            )
        element_length = float(element_length)
        if element_length <= 0.0:
            raise ValueError("The element length must be positive.")

        mesh = padded_mesh(
            inner, outer, pad=pads, weight="r2", ngll=ngll, drmax=element_length
        )
        family = RadialOperatorFamily(
            mesh, kappa=lambda r: clamped(r) ** 2, weight="r2"
        )
        physical = restriction(mesh, inner, outer)
        if radial_modes is not None:
            rule = {"nmodes": int(radial_modes)}
        else:
            if max_eigenvalue is None:
                # From the domain proper and not the padded mesh, whose first
                # mode of degree lmax is longer by the ratio of the two radii.
                bare = RadialOperatorFamily(
                    padded_mesh(
                        inner,
                        outer,
                        pad=0.0,
                        weight="r2",
                        ngll=ngll,
                        drmax=element_length,
                    ),
                    kappa=lambda r: clamped(r) ** 2,
                    weight="r2",
                )
                max_eigenvalue = float(bare.eigvalsh(lmax)[0]) * (1.0 + _BOUND_RTOL)
            rule = {"theta_max": float(max_eigenvalue)}
        try:
            bases = [
                SpectralBasis(family, degree, restrict=physical, **rule)
                for degree in range(lmax + 1)
            ]
        except ValueError as error:
            raise ValueError(
                f"The truncation leaves some degree up to {lmax} without a "
                f"radial mode, or asks for more than the mesh holds: {error}"
            ) from error
        self._basis = SphericalBasis(bases)
        self._grid = gauss_legendre(lmax)

        self._lmax = lmax
        self._radii = (inner, outer)
        self._padding = pads
        self._ngll = ngll
        self._element_length = element_length
        self._length_scale = length_scale
        self._length_scale_key = _length_scale_key(length_scale, clamped, mesh.rglob)
        super().__init__(self._basis.theta, order)

    # ----------------------------------------------------------------- #
    #                              Identity                             #
    # ----------------------------------------------------------------- #

    @property
    def lmax(self) -> int:
        """The largest spherical-harmonic degree kept."""
        return self._lmax

    @property
    def radius(self) -> float:
        """The outer radius."""
        return self._radii[1]

    @property
    def inner_radius(self) -> float:
        """The inner radius: zero for a ball."""
        return self._radii[0]

    @property
    def padding(self) -> tuple[float, float]:
        """The radial padding inwards and outwards."""
        return self._padding

    @property
    def length_scale(self) -> LengthScale:
        """The length scale of ``A``, as it was given."""
        return self._length_scale

    @property
    def spatial_dimension(self) -> int:
        """Three."""
        return 3

    @property
    def domain_volume(self) -> float:
        """The volume of the domain proper, excluding the padding."""
        inner, outer = self._radii
        return 4.0 * np.pi * (outer**3 - inner**3) / 3.0

    @property
    def basis(self) -> SphericalBasis:
        """``planetmodel``'s per-degree eigenbases, which do the numerics and
        fix the order of the components."""
        return self._basis

    @property
    def grid(self) -> AngularGrid:
        """The Gauss-Legendre angular grid, in ``planetmodel``'s terms:
        colatitudes and longitudes in radians."""
        return self._grid

    @property
    def eigenvalues(self) -> np.ndarray:
        """The eigenvalue of ``A`` attached to each component: ascending
        within a harmonic, the same for every order of a degree, and at least
        one."""
        return self._basis.theta

    @cached_property
    def _labels(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(degree, signed order, radial index) of every component."""
        parts, degrees, orders = harmonics.packing(self._lmax)
        counts = self._basis.nmodes[degrees]
        signed = np.where(parts == 0, orders, -orders)
        indices = np.concatenate([np.arange(n) for n in counts])
        return np.repeat(degrees, counts), np.repeat(signed, counts), indices

    @property
    def degrees(self) -> np.ndarray:
        """The spherical-harmonic degree of each component."""
        return self._labels[0]

    @property
    def orders(self) -> np.ndarray:
        """The signed harmonic order of each component: non-negative for a
        cosine harmonic and negative for a sine one, as on the sphere."""
        return self._labels[1]

    @property
    def radial_indices(self) -> np.ndarray:
        """Which radial mode of its degree each component is, from zero in
        ascending eigenvalue."""
        return self._labels[2]

    def _coordinate_key(self) -> Hashable:
        """The mesh, the operator and the truncation, with the metric left out.

        Tagged by geometry and not by ``type(self)``: ``Lebesgue`` and
        ``Sobolev`` are views of one set of fields.
        """
        return (
            "radial_ball",
            self._lmax,
            tuple(int(n) for n in self._basis.nmodes),
            self._radii,
            self._padding,
            self._ngll,
            self._element_length,
            self._length_scale_key,
        )

    def _class_for_order(self, order: float, /) -> type:
        return Lebesgue if order == 0.0 else Sobolev

    def __repr__(self) -> str:
        inner, outer = self._radii
        return (
            f"{type(self).__name__}(lmax={self._lmax}, dim={self.dim}, "
            f"radii=[{inner:g}, {outer:g}], order={self._order:g})"
        )

    # ----------------------------------------------------------------- #
    #                          The coordinate map                       #
    # ----------------------------------------------------------------- #

    def to_components(self, x: np.ndarray) -> np.ndarray:
        """The coefficients of the kept eigenfunctions: a harmonic analysis of
        every radial shell, then each harmonic's radial function projected
        onto the modes of its degree, both by the grid's own quadrature."""
        coefficients = harmonics.analyse_grid(
            np.asarray(x, dtype=float), self._grid, lmax=self._lmax
        )
        return self._basis.analyse(coefficients)

    def from_components(self, c: np.ndarray) -> np.ndarray:
        """The values on the padded grid: the radial syntheses, then a
        harmonic synthesis of every shell."""
        return harmonics.synthesise_grid(
            self._basis.synthesise(c, physical=False), self._grid
        )

    # ----------------------------------------------------------------- #
    #                               The grid                            #
    # ----------------------------------------------------------------- #

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        """``(radii, latitudes, longitudes)``, the radii those of the padded
        mesh."""
        nodes = self._basis.family.mesh.nglob
        return (nodes, self._grid.ntheta, self._grid.nphi)

    @property
    def radii(self) -> np.ndarray:
        """The nodes of the padded radial mesh: the first axis of a vector."""
        return self._basis.family.mesh.rglob

    @property
    def interior_radii(self) -> np.ndarray:
        """The radial nodes of the domain proper, both ends included."""
        return self._basis.r

    @property
    def latitudes(self) -> np.ndarray:
        """The grid's latitudes in degrees, north to south: the second axis."""
        return 90.0 - np.degrees(self._grid.colatitudes)

    @property
    def longitudes(self) -> np.ndarray:
        """The grid's longitudes in degrees, from zero eastwards: the third
        axis."""
        return np.degrees(self._grid.longitudes)

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
        function: Callable[[tuple[float, float, float]], float],
        /,
        *,
        extension: str = "odd",
    ) -> np.ndarray:
        """Sample a function on the grid, continuing it along each radius
        across the padding.

        The function is never called outside the domain, where it need not be
        defined. A padding node is given ``2 f(end) - f(mirror)`` along its
        radius, the odd reflection through the value at the nearer end of the
        domain, which continues the function with its radial slope; see
        :meth:`~pygeoinf2.radial.interval.Interval.project_function` for what
        holding it constant costs.

        Args:
            function: called with a point ``(radius, latitude, longitude)`` of
                the domain, the angles in degrees.
            extension: ``"odd"``, the reflection, or ``"constant"``, the value
                at the end of the radius. The reflection of a positive
                function need not be positive; the constant is.

        Returns:
            The sampled field, which :meth:`truncate` settles into the span of
            the kept modes.

        Raises:
            ValueError: for an extension that is neither.
        """
        shells: dict[float, np.ndarray] = {}

        def sample(radii: np.ndarray) -> np.ndarray:
            for r in np.unique(radii):
                if float(r) not in shells:
                    shells[float(r)] = np.array(
                        [
                            [
                                float(function((float(r), float(lat), float(lon))))
                                for lon in self.longitudes
                            ]
                            for lat in self.latitudes
                        ]
                    )
            return np.stack([shells[float(r)] for r in radii])

        return _sampled_with_extension(self.radii, *self._radii, sample, extension)

    def random_point(self, *, rng: Generator | None = None) -> np.ndarray:
        """A point of the domain, drawn uniformly over its volume."""
        rng = np.random.default_rng() if rng is None else rng
        inner, outer = self._radii
        return np.array(
            [
                float(np.cbrt(rng.uniform(inner**3, outer**3))),
                float(np.degrees(np.arcsin(rng.uniform(-1.0, 1.0)))),
                float(rng.uniform(-180.0, 180.0)),
            ]
        )

    # ----------------------------------------------------------------- #
    #                        Integrals and points                       #
    # ----------------------------------------------------------------- #

    def integral_functional(self) -> LinearFunctional:
        """The integral of a field over the volume of the domain proper.

        Only the degree-zero harmonic has one, ``sqrt(4 pi)`` times the radial
        integral of its coefficient function against ``r^2``, taken by the
        quadrature of the domain's own elements so that the padding
        contributes nothing.
        """
        radial = self._basis[0]
        interior = radial.modes()[radial.restriction.nodes]
        components = np.zeros(self.dim)
        components[self._basis.block(0, 0, 0)] = np.sqrt(4.0 * np.pi) * (
            radial.weights() @ interior
        )
        return LinearFunctional.from_derivative_components(self, components)

    def basis_matrix(
        self, points: Sequence[tuple[float, float, float]], /
    ) -> np.ndarray:
        """The basis at many points ``(radius, latitude, longitude)``, as a
        ``(len(points), dim)`` array: each harmonic at the angles times each
        radial mode of its degree at the radius, the latter exact through the
        polynomial it is on each element.

        Raises:
            ValueError: if a radius lies outside the domain.
        """
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        colatitude = np.radians(90.0 - points[:, 1])
        longitude = np.radians(points[:, 2])
        angular = harmonics.real_harmonics(self._lmax, colatitude, longitude)
        matrix = np.empty((points.shape[0], self.dim))
        for degree, radial in enumerate(self._basis):
            columns = slice(
                self._basis.block(0, degree, 0).start,
                self._basis.block(1, degree, degree).stop
                if degree
                else self._basis.block(0, 0, 0).stop,
            )
            packed = np.concatenate(
                (angular[0, degree, : degree + 1], angular[1, degree, 1 : degree + 1])
            )
            modes = radial.evaluate(points[:, 0])
            matrix[:, columns] = (packed.T[:, :, None] * modes[:, None, :]).reshape(
                points.shape[0], -1
            )
        return matrix

    def _squared_modes(self, weights: np.ndarray, /) -> np.ndarray:
        """``sum_k weights_k phi_k(p)^2`` on the grid.

        When the weights do not depend on the order within a degree, which is
        so for every function of ``A``, the addition theorem does the angular
        sum -- ``sum_m Y_lm^2 = (2 l + 1) / 4 pi`` -- and the answer is a
        function of radius. Otherwise the harmonics are squared on the grid.
        """
        shape = self.grid_shape
        per_degree = []
        isotropic = True
        for degree, radial in enumerate(self._basis):
            start = self._basis.block(0, degree, 0).start
            block = weights[start : start + (2 * degree + 1) * radial.nmodes]
            block = block.reshape(2 * degree + 1, radial.nmodes)
            isotropic = isotropic and bool(np.all(block == block[0]))
            per_degree.append((radial.modes() ** 2) @ block.T)  # (radii, 2 l + 1)
        if isotropic:
            profile = sum(
                (2 * degree + 1) / (4.0 * np.pi) * squares[:, 0]
                for degree, squares in enumerate(per_degree)
            )
            return np.broadcast_to(profile[:, None, None], shape).copy()
        colatitude, longitude = np.meshgrid(
            self._grid.colatitudes, self._grid.longitudes, indexing="ij"
        )
        angular = harmonics.real_harmonics(self._lmax, colatitude, longitude) ** 2
        out = np.zeros(shape)
        for degree, squares in enumerate(per_degree):
            packed = np.concatenate(
                (angular[0, degree, : degree + 1], angular[1, degree, 1 : degree + 1])
            )
            out += np.einsum("rh,hij->rij", squares, packed)
        return out


class Lebesgue(Ball):
    """The ``L2`` space on a ball or annulus."""

    def __init__(
        self,
        lmax: int,
        /,
        *,
        radius: float = 1.0,
        inner_radius: float = 0.0,
        length_scale: LengthScale = 1.0,
        radial_modes: int | None = None,
        max_eigenvalue: float | None = None,
        padding: float | tuple[float, float] | None = None,
        ngll: int = 5,
        element_length: float | None = None,
    ) -> None:
        """
        Args:
            lmax: the largest spherical-harmonic degree kept.
            radius, inner_radius: the radii; an inner radius of zero is a ball.
            length_scale: ``L`` in ``A``, which fixes the basis and the
                default padding though not this space's inner product.
            radial_modes, max_eigenvalue: the radial truncation at each
                degree; isotropic when neither is given.
            padding: the radial mesh's reach past each end.
            ngll: nodes per radial element.
            element_length: the longest a radial element may be.
        """
        super().__init__(
            lmax,
            radius=radius,
            inner_radius=inner_radius,
            order=0.0,
            length_scale=length_scale,
            radial_modes=radial_modes,
            max_eigenvalue=max_eigenvalue,
            padding=padding,
            ngll=ngll,
            element_length=element_length,
        )


class Sobolev(Ball):
    """The Sobolev space ``H^order`` on a ball or annulus, with the inner
    product ``(A^order u, v)``."""

    def __init__(
        self,
        lmax: int,
        order: float,
        length_scale: LengthScale,
        /,
        *,
        radius: float = 1.0,
        inner_radius: float = 0.0,
        radial_modes: int | None = None,
        max_eigenvalue: float | None = None,
        padding: float | tuple[float, float] | None = None,
        ngll: int = 5,
        element_length: float | None = None,
    ) -> None:
        """
        Args:
            lmax: the largest spherical-harmonic degree kept.
            order: the Sobolev order.
            length_scale: ``L`` in ``A``, a number or a function of radius:
                the length at which the Sobolev weight turns over.
            radius, inner_radius: the radii; an inner radius of zero is a ball.
            radial_modes, max_eigenvalue: the radial truncation at each
                degree; isotropic when neither is given.
            padding: the radial mesh's reach past each end.
            ngll: nodes per radial element.
            element_length: the longest a radial element may be.
        """
        super().__init__(
            lmax,
            radius=radius,
            inner_radius=inner_radius,
            order=order,
            length_scale=length_scale,
            radial_modes=radial_modes,
            max_eigenvalue=max_eigenvalue,
            padding=padding,
            ngll=ngll,
            element_length=element_length,
        )
