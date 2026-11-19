"""The circle's transform and the polar basis of a disk or an annulus.

``A = 1 - div(kappa grad)`` in the plane, with ``kappa = kappa(r)``, commutes
with rotations about the centre, and in polar coordinates block-diagonalizes
over the circle's Fourier modes into the family of radial operators

    A_m f = f - r^-1 (kappa r f')' + kappa m^2 r^-2 f,

one per order ``m >= 0``, each self-adjoint on ``L^2(r dr)``. That family is
``planetmodel``'s ``SeparableOperatorFamily`` under the measure ``r dr`` and
indexed by order, ``weight="r", index="order"``: the cylindrical counterpart
of the family of a ball, whose measure is ``r^2 dr`` and whose members are
indexed by degree. At the centre of a disk it does what a ball's does: for
``m >= 1`` the centre's node is dropped and the field held to zero there, and
for ``m = 0`` the centre's value, which carries no mass, is eliminated by
static condensation.

:class:`RingTransform` goes between the coefficient functions of the circle's
modes and values on equally spaced angles, every ring at once, by a real FFT.
The modes are orthonormal on the unit circle -- ``1 / sqrt(2 pi)``,
``cos(m t) / sqrt(pi)``, ``sin(m t) / sqrt(pi)`` -- and :class:`PolarBasis`
holds one ``planetmodel`` ``SpectralBasis`` per order and fixes the order of
the coefficients as one vector.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from numpy.typing import ArrayLike
from planetmodel.randomfield import SeparableOperatorFamily, SpectralBasis
from planetmodel.randomfield.mesh import Restriction

__all__ = [
    "PolarBasis",
    "RingTransform",
    "circular_harmonics",
]


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
            ValueError: if there are none, one is of the wrong order, they do
                not share one family and one restriction, or the family is
                not indexed by order.
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
            if basis.family.index != "order":
                raise ValueError(
                    "A polar basis is of a family indexed by 'order', not by "
                    f"{basis.family.index!r}."
                )
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
    def family(self) -> SeparableOperatorFamily:
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
