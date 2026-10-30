"""The radial operators of a disk or an annulus, and the circle's transform.

``A = 1 - div(kappa grad)`` in the plane, with ``kappa = kappa(r)``, commutes
with rotations about the centre, and in polar coordinates block-diagonalizes
over the circle's Fourier modes into the family of radial operators

    A_m f = f - r^-1 (kappa r f')' + kappa m^2 r^-2 f,

one per order ``m >= 0``, each self-adjoint on ``L^2(r dr)``. That is the
cylindrical counterpart of ``planetmodel``'s ``RadialOperatorFamily``, whose
measures are ``r^2 dr`` (a ball) and ``dr`` (an interval of the line), and
:class:`CylindricalOperatorFamily` is that family with the measure changed:
the mass is ``int phi_i phi_j r dr``, the radial stiffness ``int kappa phi_i'
phi_j' r dr`` and the angular term ``m^2 int kappa phi_i phi_j / r dr``, all by
the Gauss-Lobatto-Legendre quadrature of the elements, so the mass and the
angular term are diagonal and the stiffness is banded.

The centre needs care but no condition, exactly as a ball's does. A field
regular there has a part of order ``m`` going like ``r^m``, so for ``m >= 1``
the centre's node is dropped and the field held to zero there, which also
keeps the ``1 / r`` of the angular term off the one node where it is
infinite; for ``m = 0`` the centre's value is genuine but carries no mass,
and is eliminated by static condensation. ``planetmodel`` does both for its
ball: its bookkeeping of the dropped node is inherited here, and its
condensation copied with the assembly.

:class:`RingTransform` goes between the coefficient functions of the circle's
modes and values on equally spaced angles, every ring at once, by a real FFT.
The modes are orthonormal on the unit circle -- ``1 / sqrt(2 pi)``,
``cos(m t) / sqrt(pi)``, ``sin(m t) / sqrt(pi)`` -- and :class:`PolarBasis`
holds one ``planetmodel`` ``SpectralBasis`` per order and fixes the order of
the coefficients as one vector.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

import numpy as np
from numpy.typing import ArrayLike
from planetmodel.mesh1d import Mesh1D
from planetmodel.randomfield import RadialOperatorFamily, SpectralBasis
from planetmodel.randomfield.mesh import Restriction

__all__ = [
    "CylindricalOperatorFamily",
    "PolarBasis",
    "RingTransform",
    "circular_harmonics",
]

#: The coefficient of a member of the family: a number, a function of radius,
#: or nodal values of the mesh's shape ``(nspec, ngll)``.
type Coefficient = ArrayLike | Callable[[np.ndarray], ArrayLike]

#: Robin coefficients at the ends of the mesh: ``None`` for the natural
#: condition at both, one number for both, or an ``(inner, outer)`` pair with
#: ``None`` for a natural end.
type Robin = float | tuple[float | None, float | None] | None


def _nodal_coefficient(mesh: Mesh1D, c: Coefficient, /) -> np.ndarray:
    """A coefficient as nodal values of shape ``(nspec, ngll)``.

    Raises:
        ValueError: for an array of the wrong shape, or a value that is not
            finite.
    """
    if callable(c):
        out = np.asarray(c(mesh.r), dtype=float)
        out = np.broadcast_to(out, mesh.r.shape).copy()
    elif np.ndim(c) == 0:
        out = np.full_like(mesh.r, float(c))
    else:
        out = np.asarray(c, dtype=float)
        if out.shape != mesh.r.shape:
            raise ValueError(
                f"A coefficient array must have the nodal shape {mesh.r.shape}, "
                f"got {out.shape}."
            )
        out = out.copy()
    if not np.all(np.isfinite(out)):
        raise ValueError("A coefficient must be finite everywhere on the mesh.")
    return out


def _robin_pair(robin: Robin, /) -> tuple[float, float]:
    """``(inner, outer)`` Robin coefficients from ``None``, one number, or a
    pair with ``None`` for a natural end.

    Raises:
        ValueError: for a negative coefficient.
    """
    if robin is None:
        return 0.0, 0.0
    if np.ndim(robin) == 0:
        inner = outer = float(robin)
    else:
        inner, outer = robin
        inner = 0.0 if inner is None else float(inner)
        outer = 0.0 if outer is None else float(outer)
    if inner < 0.0 or outer < 0.0:
        raise ValueError("Robin coefficients must be non-negative.")
    return inner, outer


class CylindricalOperatorFamily(RadialOperatorFamily):
    """The order-indexed radial operators ``A_m`` of a disk or annulus on a
    one-dimensional GLL mesh, under the measure ``r dr``.

    A ``planetmodel`` ``RadialOperatorFamily`` in every respect but the
    assembly, so that ``planetmodel``'s ``SpectralBasis``, which accepts that
    class and no other, takes it as it is. The assembly is the parent's with
    the measure ``r dr`` in place of ``r^2 dr`` and the angular term ``m^2 /
    r^2`` in place of ``l (l + 1) / r^2``; :meth:`pencil` is the parent's with
    the second change. Everything else -- the spectrum and its cache, the
    eigenpairs by value or by count, the application, solution, powers and
    white noise, the bookkeeping of the dropped centre -- is inherited, and
    reads the arrays assembled here.

    The order ``m`` is passed wherever the parent takes a degree: ``eig(m)``,
    ``eigvalsh(m)``, ``mass(m)``, ``embed(m, values)`` and the rest.
    """

    def __init__(
        self,
        mesh: Mesh1D,
        /,
        *,
        kappa: Coefficient = 1.0,
        kappa_h: Coefficient | None = None,
        robin: Robin = None,
    ) -> None:
        """
        Args:
            mesh: a ``planetmodel`` ``Mesh1D`` in ``r >= 0``. One that starts
                at the centre is a disk's, and is padded inwards no further.
            kappa: the radial coefficient, ``L^2`` for ``A = 1 - div(L^2
                grad)``: a number, a function of radius, or nodal values.
            kappa_h: the coefficient of the angular term, ``kappa`` if
                omitted.
            robin: Robin coefficients at the ends of the mesh, ``f' -+ gamma
                f = 0``: ``None`` for the natural condition at both, one
                number for both, or an ``(inner, outer)`` pair with ``None``
                for a natural end. An inner one is ignored at the centre.

        Raises:
            TypeError: if the mesh is not a ``Mesh1D``.
            ValueError: if the mesh reaches below ``r = 0``, a coefficient is
                not finite, ``kappa`` is not positive or ``kappa_h`` is
                negative, or a Robin coefficient is negative.
            RuntimeError: if a node other than the centre has no mass.
        """
        if not isinstance(mesh, Mesh1D):
            raise TypeError("The mesh must be a planetmodel Mesh1D.")
        if mesh.rglob[0] < 0.0:
            raise ValueError("The r dr measure needs a mesh in r >= 0.")

        self.mesh = mesh
        self.weight = "r"
        # The names the inherited methods read: a mesh from the centre drops
        # it above order zero and condenses it out at order zero, as a ball's
        # does; no order is refused.
        self._ball = bool(mesh.rglob[0] == 0.0)
        self._degree_zero_only = False

        kr = _nodal_coefficient(mesh, kappa)
        kh = kr if kappa_h is None else _nodal_coefficient(mesh, kappa_h)
        if np.any(kr <= 0.0) or np.any(kh < 0.0):
            raise ValueError("kappa must be positive and kappa_h non-negative.")

        r, jac, wq = mesh.r, mesh.jac, mesh.w
        derivative, gmap = mesh.deriv, mesh.gmap
        n, q = mesh.nglob, mesh.ngll - 1
        measure = r

        element_mass = wq[None, :] * jac[:, None] * measure
        element_mass.setflags(write=False)
        mass = np.zeros(n)
        np.add.at(mass, gmap, element_mass)
        # m^2 int kappa phi_i phi_j / r dr, lumped. The centre's entry is set
        # to zero: the node is dropped wherever it would be multiplied.
        safe = np.where(r != 0.0, r, 1.0)
        angular = np.zeros(n)
        np.add.at(
            angular, gmap, kh * (wq[None, :] * jac[:, None]) * (measure / safe**2)
        )

        # The radial stiffness, banded, upper symmetric storage:
        # band[q + i - j, j] = K[i, j] for i <= j.
        coefficient = wq[None, :] * kr * measure / jac[:, None]
        stiffness = np.einsum("qi,eq,qj->eij", derivative, coefficient, derivative)
        band = np.zeros((q + 1, n))
        rows_of, columns_of = gmap[:, :, None], gmap[:, None, :]
        upper = rows_of <= columns_of
        rows = np.broadcast_to(q + rows_of - columns_of, stiffness.shape)
        columns = np.broadcast_to(columns_of, stiffness.shape)
        np.add.at(band, (rows[upper], columns[upper]), stiffness[upper])

        inner, outer = _robin_pair(robin)
        if outer > 0.0:
            band[q, -1] += outer * kr[-1, -1] * measure[-1, -1]
        if inner > 0.0 and not self._ball:
            band[q, 0] += inner * kr[0, 0] * measure[0, 0]

        # Static condensation of the massless centre at order zero, whose row
        # reads d f_0 + a . f_(1..q) = 0.
        if self._ball:
            self._axis_a = np.array([band[q - j, j] for j in range(1, q + 1)])
            self._axis_d = float(band[q, 0])
            self._axis_c = -self._axis_a / self._axis_d
        if np.any(mass[1 if self._ball else 0 :] <= 0.0):
            raise RuntimeError("A node has no mass: the mesh is degenerate.")

        self._Me = element_mass
        self._M, self._Kh, self._Kr, self._q = mass, angular, band, q
        self._vals: dict[int, np.ndarray] = {}
        self._pairs: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
        self._pencils: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    @property
    def is_ball(self) -> bool:
        """Whether the mesh starts at the centre: the family of a disk, and
        not of an annulus. Named as ``planetmodel``'s family names it, whose
        bookkeeping of the centre this shares."""
        return self._ball

    def pencil(self, m: int, /) -> tuple[np.ndarray, np.ndarray]:
        """The banded ``A_m`` (upper storage) and the diagonal mass, on the
        active nodes of order ``m``.

        On a disk the centre is removed at every order: for ``m >= 1`` its
        couplings are discarded, holding the field to zero there; at ``m =
        0`` the exact Schur complement of its massless row is folded into the
        first element, so that the reduced pencil has the finite spectrum of
        the singular one.

        Args:
            m: the order, non-negative.

        Returns:
            The band, read-only, and the mass.

        Raises:
            ValueError: for a negative order.
        """
        if m in self._pencils:
            return self._pencils[m]
        n0, q = self._n0(m), self._q
        band = self._Kr.copy()
        if n0 and m == 0:
            a, d = self._axis_a, self._axis_d
            for i in range(1, q + 1):
                for j in range(i, q + 1):
                    band[q + i - j, j] -= a[i - 1] * a[j - 1] / d
        band = band[:, n0:]
        if n0:
            for row in range(q):
                column = q - 1 - row
                if column < band.shape[1]:
                    band[row, column] = 0.0
        mass = self._M[n0:]
        band[q] += mass + float(m) ** 2 * self._Kh[n0:]
        band.setflags(write=False)
        self._pencils[m] = (band, mass)
        return band, mass

    def __repr__(self) -> str:
        kind = "disk" if self._ball else "annulus"
        cached = sorted(set(self._vals) | set(self._pairs))
        return (
            f"CylindricalOperatorFamily({kind}, {self.mesh.nglob} nodes, "
            f"cached orders: {cached})"
        )


def circular_harmonics(mmax: int, angles: ArrayLike, /) -> np.ndarray:
    """The circle's real modes, orthonormal on the unit circle, at angles.

    Args:
        mmax: the largest order.
        angles: in radians, of any shape.

    Returns:
        Shape ``(2, mmax + 1) + angles.shape``: ``[0, m]`` the cosine mode of
        order ``m``, ``1 / sqrt(2 pi)`` at ``m = 0`` and ``cos(m t) /
        sqrt(pi)`` above, and ``[1, m]`` the sine mode ``sin(m t) /
        sqrt(pi)``, zero at ``m = 0``, where there is none.
    """
    t = np.asarray(angles, dtype=float)
    orders = np.arange(int(mmax) + 1).reshape((-1,) + (1,) * t.ndim)
    out = np.empty((2, int(mmax) + 1) + t.shape)
    out[0] = np.cos(orders * t) / np.sqrt(np.pi)
    out[1] = np.sin(orders * t) / np.sqrt(np.pi)
    out[0, 0] = 1.0 / np.sqrt(2.0 * np.pi)
    out[1, 0] = 0.0
    return out


class RingTransform:
    """Synthesis and analysis on ``2 mmax + 1`` equally spaced angles, for
    any number of rings.

    The angles are ``2 pi j / (2 mmax + 1)``, from zero anticlockwise. With
    that many the trapezoid rule integrates every product of two kept modes
    exactly, so analysis inverts synthesis on the kept modes, and it is the
    adjoint of synthesis under the rule: ``analyse = w S^T`` with ``S`` the
    synthesis and ``w = 2 pi / (2 mmax + 1)`` the weight of every angle.
    """

    def __init__(self, mmax: int, /) -> None:
        """
        Args:
            mmax: the largest order kept.

        Raises:
            ValueError: if it is negative.
        """
        mmax = int(mmax)
        if mmax < 0:
            raise ValueError("mmax must be non-negative.")
        self._mmax = mmax
        self._count = 2 * mmax + 1
        self._weight = 2.0 * np.pi / self._count

    @property
    def mmax(self) -> int:
        """The largest order kept."""
        return self._mmax

    @property
    def angles(self) -> np.ndarray:
        """The angles of the grid in radians, from zero anticlockwise."""
        return self._weight * np.arange(self._count)

    @property
    def weight(self) -> float:
        """The quadrature weight of every angle, ``2 pi`` over their number."""
        return self._weight

    # synthesise and analyse mirror planetmodel's SpectralBasis names.
    def synthesise(self, coefficients: np.ndarray, /) -> np.ndarray:
        """Values on every ring from coefficient functions.

        Args:
            coefficients: shape ``(2, mmax + 1, rings)``, cosine modes first;
                the sine slot of order zero is ignored.

        Returns:
            Shape ``(rings, angles)``.

        Raises:
            ValueError: for coefficients of the wrong shape.
        """
        c = np.asarray(coefficients, dtype=float)
        if c.ndim != 3 or c.shape[:2] != (2, self._mmax + 1):
            raise ValueError(
                f"Expected coefficients of shape (2, {self._mmax + 1}, rings), "
                f"got {c.shape}."
            )
        n = self._count
        spectrum = np.empty((c.shape[2], self._mmax + 1), dtype=complex)
        spectrum[:, 0] = n * c[0, 0] / np.sqrt(2.0 * np.pi)
        spectrum[:, 1:] = (0.5 * n / np.sqrt(np.pi)) * (c[0, 1:] - 1j * c[1, 1:]).T
        return np.fft.irfft(spectrum, n=n, axis=1)

    def analyse(self, values: np.ndarray, /) -> np.ndarray:
        """Coefficient functions from values on every ring, by the trapezoid
        rule.

        Args:
            values: shape ``(rings, angles)``.

        Returns:
            Shape ``(2, mmax + 1, rings)``, zero in the sine slot of order
            zero.

        Raises:
            ValueError: for values of the wrong shape.
        """
        v = np.asarray(values, dtype=float)
        if v.ndim != 2 or v.shape[1] != self._count:
            raise ValueError(
                f"Expected values of shape (rings, {self._count}), got {v.shape}."
            )
        spectrum = np.fft.rfft(v, axis=1)
        out = np.empty((2, self._mmax + 1, v.shape[0]))
        out[0, 0] = self._weight * spectrum[:, 0].real / np.sqrt(2.0 * np.pi)
        out[0, 1:] = (self._weight / np.sqrt(np.pi)) * spectrum[:, 1:].real.T
        out[1, 1:] = -(self._weight / np.sqrt(np.pi)) * spectrum[:, 1:].imag.T
        out[1, 0] = 0.0
        return out


class PolarBasis:
    """The bases of orders ``0 ... mmax`` of one family on one restriction,
    and the order of their coefficients as one vector.

    ``bases[m]`` is the ``planetmodel`` ``SpectralBasis`` of order ``m``. The
    vector runs over the orders; within an order over the cosine mode and
    then, for ``m >= 1``, the sine mode; within a mode over the radial modes
    in ascending eigenvalue. The number of radial modes may differ from order
    to order, so the vector is ragged in ``m``; :meth:`block` says where each
    mode sits.
    """

    def __init__(self, bases: Sequence[SpectralBasis], /) -> None:
        """
        Args:
            bases: the bases of orders ``0, 1, ... mmax``, in that order.

        Raises:
            ValueError: if there are none, one is of the wrong order, or they
                do not share one family and one restriction.
            TypeError: if one is not a ``SpectralBasis``.
        """
        bases = tuple(bases)
        if not bases:
            raise ValueError("The basis of order zero at least is needed.")
        first = bases[0]
        for m, basis in enumerate(bases):
            if not isinstance(basis, SpectralBasis):
                raise TypeError("The bases must be SpectralBasis objects.")
            if basis.degree != m:
                raise ValueError(f"bases[{m}] has order {basis.degree}.")
            if basis.family is not first.family:
                raise ValueError("The bases must share one family.")
            if (
                basis.restriction.interval != first.restriction.interval
                or basis.restriction.nodes != first.restriction.nodes
            ):
                raise ValueError("The bases must share one restriction.")
        self._bases = bases
        self._mmax = len(bases) - 1
        nmodes = np.array([basis.nmodes for basis in bases])
        nmodes.setflags(write=False)
        self._nmodes = nmodes
        counts = np.where(np.arange(self._mmax + 1) == 0, 1, 2) * nmodes
        self._offsets = np.concatenate(([0], np.cumsum(counts)))
        theta = np.concatenate(
            [np.tile(basis.theta, 1 if m == 0 else 2) for m, basis in enumerate(bases)]
        )
        theta.setflags(write=False)
        self._theta = theta

    @property
    def mmax(self) -> int:
        """The largest order."""
        return self._mmax

    @property
    def nmodes(self) -> np.ndarray:
        """The radial modes kept at each order, read-only."""
        return self._nmodes

    @property
    def size(self) -> int:
        """The length of the vector."""
        return int(self._offsets[-1])

    @property
    def theta(self) -> np.ndarray:
        """The eigenvalue belonging to each entry of the vector, read-only."""
        return self._theta

    @property
    def r(self) -> np.ndarray:
        """The radii of the domain's nodes."""
        return self._bases[0].r

    @property
    def family(self) -> CylindricalOperatorFamily:
        """The family the bases are of."""
        return self._bases[0].family

    @property
    def restriction(self) -> Restriction:
        """The domain the bases share, inside the padded mesh."""
        return self._bases[0].restriction

    def __len__(self) -> int:
        return self._mmax + 1

    def __getitem__(self, m: int) -> SpectralBasis:
        """The basis of order ``m``."""
        return self._bases[m]

    def __iter__(self) -> Iterator[SpectralBasis]:
        """The bases in order."""
        return iter(self._bases)

    def block(self, s: int, m: int, /) -> slice:
        """Where the radial coefficients of one mode sit in the vector.

        Args:
            s: ``0`` for the cosine mode, ``1`` for the sine one.
            m: the order; the sine mode needs ``m >= 1``.

        Returns:
            The slice.

        Raises:
            IndexError: if there is no such mode.
        """
        if not 0 <= m <= self._mmax:
            raise IndexError(f"Order {m} is outside 0..{self._mmax}.")
        if s not in (0, 1) or (s == 1 and m == 0):
            raise IndexError(f"There is no mode (s, m) = ({s}, {m}).")
        n = int(self._nmodes[m])
        start = int(self._offsets[m]) + s * n
        return slice(start, start + n)

    def synthesise(self, vector: ArrayLike, /, *, physical: bool = True) -> np.ndarray:
        """The radial coefficient function of every mode.

        Args:
            vector: the coefficients, of length :attr:`size`.
            physical: give them at the domain's nodes; with ``False`` at every
                node of the mesh, which is what :meth:`analyse` takes.

        Returns:
            Shape ``(2, mmax + 1, nodes)``, the layout of
            :class:`RingTransform`, zero in the sine slot of order zero.

        Raises:
            ValueError: for a vector of the wrong length.
        """
        v = np.asarray(vector, dtype=float)
        if v.shape != (self.size,):
            raise ValueError(
                f"Expected a vector of length {self.size}, got shape {v.shape}."
            )
        n = self.r.size if physical else self.family.mesh.nglob
        out = np.zeros((2, self._mmax + 1, n))
        for m, basis in enumerate(self._bases):
            parts = 1 if m == 0 else 2
            block = v[self._offsets[m] : self._offsets[m + 1]].reshape(
                parts, basis.nmodes
            )
            out[:parts, m] = basis.synthesise(block.T, physical=physical).T
        return out

    def analyse(self, coefficients: ArrayLike, /) -> np.ndarray:
        """The vector of coefficient functions given at every node of the
        mesh: each mode's function projected onto the basis of its order.

        Args:
            coefficients: shape ``(2, mmax + 1, nglob)``; the sine slot of
                order zero is ignored.

        Returns:
            The vector, of length :attr:`size`.

        Raises:
            ValueError: for coefficients of the wrong shape.
        """
        c = np.asarray(coefficients, dtype=float)
        want = (2, self._mmax + 1, self.family.mesh.nglob)
        if c.shape != want:
            raise ValueError(
                f"Expected coefficient functions of shape {want}, got {c.shape}."
            )
        out = np.empty(self.size)
        for m, basis in enumerate(self._bases):
            parts = 1 if m == 0 else 2
            out[self._offsets[m] : self._offsets[m + 1]] = basis.analyse(
                c[:parts, m].T
            ).T.reshape(-1)
        return out

    def __repr__(self) -> str:
        lower, upper = self.restriction.interval
        return (
            f"PolarBasis(mmax={self._mmax}, {self.size} coefficients, on "
            f"[{lower:g}, {upper:g}])"
        )
