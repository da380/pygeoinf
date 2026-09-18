"""Harmonic transforms of many radial shells at once.

``planetmodel.harmonics`` goes between coefficients and a Gauss-Legendre grid
through ``pyshtools``, one shell per call, and a ball has a hundred shells: the
loop over them, in Python, was the whole cost of a draw and of every
application of a covariance. The transform is separable, and written so it
handles every shell together: a table of the harmonics on the prime meridian
contracts the degrees by one batched matrix product, and the longitudes are a
second. Nothing loops over radii, or over anything else.

The conventions are ``planetmodel``'s, which are ``pyshtools``' -- real
orthonormal harmonics without the Condon-Shortley phase, coefficients laid out
as ``(2, lmax + 1, lmax + 1, radii)`` and values as ``(radii, colatitudes,
longitudes)`` -- and the tests hold this to that route to rounding. Exact for a
field band-limited to the grid's degree, the grid being Gauss-Legendre with
``2 lmax + 1`` longitudes; a projection otherwise.
"""

from __future__ import annotations

import numpy as np
from planetmodel import harmonics
from planetmodel.sampling import AngularGrid

__all__ = ["ShellTransform"]


class ShellTransform:
    """Synthesis and analysis on one Gauss-Legendre grid, for any number of
    shells."""

    def __init__(self, grid: AngularGrid, /) -> None:
        """
        Args:
            grid: ``planetmodel.sampling.gauss_legendre(lmax)``.

        Raises:
            ValueError: if the grid is not a Gauss-Legendre one with its
                weights and ``2 lmax + 1`` longitudes.
        """
        if grid.lmax is None or grid.weights is None:
            raise ValueError("The grid must be a Gauss-Legendre one, with weights.")
        lmax = int(grid.lmax)
        if grid.ntheta != lmax + 1 or grid.nphi != 2 * lmax + 1:
            raise ValueError(
                f"A Gauss-Legendre grid of degree {lmax} has {lmax + 1} "
                f"colatitudes and {2 * lmax + 1} longitudes."
            )
        self._lmax = lmax
        # The harmonics on the prime meridian: the cosine ones are the
        # normalized Legendre functions, and the sine ones share them.
        meridian = harmonics.real_harmonics(lmax, grid.colatitudes, 0.0)[0]
        # (order, colatitude, degree), so that a matmul contracts the degree.
        self._legendre = np.ascontiguousarray(meridian.transpose(1, 2, 0))
        self._weighted = np.ascontiguousarray(
            (meridian * grid.weights).transpose(1, 0, 2)
        )  # (order, degree, colatitude)
        orders = np.arange(lmax + 1)
        angles = orders[:, None] * grid.longitudes[None, :]
        self._cosines = np.cos(angles)  # (order, longitude)
        self._sines = np.sin(angles)
        self._cell = 2.0 * np.pi / grid.nphi

    @property
    def lmax(self) -> int:
        """The degree of the grid."""
        return self._lmax

    def synthesise(self, coefficients: np.ndarray, /) -> np.ndarray:
        """Values on every shell from coefficient functions.

        Args:
            coefficients: shape ``(2, lmax + 1, lmax + 1, radii)``.

        Returns:
            Shape ``(radii, colatitudes, longitudes)``.
        """
        c = np.asarray(coefficients, dtype=float)
        radii = c.shape[3]
        # (order, degree, radii) -> (order, colatitude, radii)
        even = np.matmul(self._legendre, c[0].transpose(1, 0, 2))
        odd = np.matmul(self._legendre, c[1].transpose(1, 0, 2))
        orders = self._lmax + 1
        # (longitude, colatitude * radii)
        flat = self._cosines.T @ even.reshape(orders, -1)
        flat += self._sines.T @ odd.reshape(orders, -1)
        return np.ascontiguousarray(
            flat.reshape(-1, self._lmax + 1, radii).transpose(2, 1, 0)
        )

    def analyse(self, values: np.ndarray, /) -> np.ndarray:
        """Coefficient functions from values on every shell, by the grid's own
        quadrature.

        Args:
            values: shape ``(radii, colatitudes, longitudes)``.

        Returns:
            Shape ``(2, lmax + 1, lmax + 1, radii)``, zero where there is no
            harmonic.
        """
        v = np.asarray(values, dtype=float)
        radii = v.shape[0]
        flat = v.transpose(2, 1, 0).reshape(v.shape[2], -1)  # (longitude, ...)
        out = np.empty((2, self._lmax + 1, self._lmax + 1, radii))
        for part, waves in enumerate((self._cosines, self._sines)):
            fourier = (self._cell * waves) @ flat  # (order, colatitude * radii)
            fourier = fourier.reshape(self._lmax + 1, self._lmax + 1, radii)
            # (order, degree, colatitude) @ (order, colatitude, radii)
            out[part] = np.matmul(self._weighted, fourier).transpose(1, 0, 2)
        return out
