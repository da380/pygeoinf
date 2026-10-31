"""The pulse and its bands: the frequency average behind the kernel.

The source-time power spectrum ``|m(w)|^2`` is a Gaussian in the angular
frequency ``w``, centered on ``w0 = 2 pi / T0`` for a dominant period
``T0``, with standard deviation ``w0 / 4``. A band is that spectrum for one
``T0``. What the kernel needs of it is the function of the detour time
``u``, the time by which the path through a point exceeds the ray's,

.. code-block:: text

    G(u) = int w^3 |m|^2 sin(w u) dw / int w^2 |m|^2 dw ,    int_0^inf G(u) du = 1 ,

the integrals over positive frequencies. The identity is the Gaussian
integral that reduces finite-frequency theory to ray theory on a model
constant across the ray: ``int_0^inf sin(w u) du = 1 / w`` for every ``w``.

``G`` depends on the band only through ``w0``: with ``w = w0 v`` and ``s =
w0 u``, ``G(u) = w0 g(w0 u)`` with one universal function

.. code-block:: text

    g(s) = int v^3 e^(-8 (v - 1)^2) sin(v s) dv / int v^2 e^(-8 (v - 1)^2) dv ,

taken here by Gauss-Legendre quadrature in ``v`` over ``[0, 3]``, the
spectrum being below ``e^-32`` of its peak beyond. ``g`` oscillates at about
``s = 2 pi`` under an envelope that falls like ``e^(-s^2 / 32)``, the
Gaussian's transform, so a detour time ``u`` beyond the cutoff ``s_max``,
where ``s^2 / 32`` reaches a given exponent, carries nothing; Gauss-Legendre
quadrature on ``[0, s_max]`` integrates ``g``, and the quadrature in ``u``
of every band is that one scaled by ``1 / w0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

#: The spectrum's standard deviation in units of its center frequency.
SPREAD = 0.25
#: How many standard deviations above the center the spectrum is kept: its
#: tail beyond is below ``e^-32`` of its peak.
REACH = 8.0


@lru_cache(maxsize=8)
def _spectrum(count: int, /) -> tuple[np.ndarray, np.ndarray, float]:
    """The Gauss-Legendre nodes ``v`` on ``[0, 1 + REACH SPREAD]``, the
    weights of ``int v^3 |m|^2 ... dv`` at them over the normalization
    ``int v^2 |m|^2 dv``, and that normalization."""
    x, w = np.polynomial.legendre.leggauss(int(count))
    top = 1.0 + REACH * SPREAD
    v = 0.5 * top * (x + 1.0)
    weight = 0.5 * top * w * np.exp(-0.5 * ((v - 1.0) / SPREAD) ** 2)
    norm = float(weight @ v**2)
    return v, weight * v**3 / norm, norm


def response(s: np.ndarray | float, /, *, frequencies: int = 128) -> np.ndarray:
    """The universal function ``g(s)``, ``G(u) = w0 g(w0 u)``.

    Args:
        s: the detour time in units of ``1 / w0``; any shape.
        frequencies: the Gauss-Legendre nodes in frequency.

    Returns:
        ``g`` at each ``s``, of its shape.
    """
    v, a, _ = _spectrum(frequencies)
    s = np.asarray(s, dtype=float)
    return (np.sin(s[..., None] * v) @ a).reshape(s.shape)


def cutoff_detour(cutoff: float, /) -> float:
    """The scaled detour time ``s_max = w0 u_max`` at which the envelope's
    exponent ``sigma^2 u^2 / 2 = (SPREAD s)^2 / 2`` reaches ``cutoff``."""
    return float(np.sqrt(2.0 * cutoff) / SPREAD)


@lru_cache(maxsize=16)
def detour_rule(
    count: int, /, *, cutoff: float = 30.0, frequencies: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """The quadrature in the scaled detour time: Gauss-Legendre nodes ``s``
    on ``[0, s_max]`` and the products ``g(s) w`` of ``g`` and the weights,
    so that ``sum g(s_j) w_j`` is ``int g ds = 1`` to the quadrature's
    accuracy."""
    x, w = np.polynomial.legendre.leggauss(int(count))
    top = cutoff_detour(cutoff)
    s = 0.5 * top * (x + 1.0)
    return s, response(s, frequencies=frequencies) * 0.5 * top * w


@dataclass(frozen=True)
class Band:
    """One band: the spectrum centered on the frequency of a dominant period.

    Attributes:
        period: the dominant period ``T0``.
    """

    period: float

    @property
    def frequency(self) -> float:
        """The center angular frequency ``w0 = 2 pi / T0``."""
        return 2.0 * np.pi / self.period

    @property
    def spread(self) -> float:
        """The spectrum's standard deviation in angular frequency, ``w0 / 4``."""
        return SPREAD * self.frequency

    def kernel_function(
        self, u: np.ndarray | float, /, *, frequencies: int = 128
    ) -> np.ndarray:
        """``G(u) = w0 g(w0 u)`` at detour times ``u``, of any shape."""
        return self.frequency * response(
            self.frequency * np.asarray(u, dtype=float), frequencies=frequencies
        )

    def cutoff(self, cutoff: float, /) -> float:
        """The largest detour time kept, ``u_max = s_max / w0``."""
        return cutoff_detour(cutoff) / self.frequency

    def detours(
        self, count: int, /, *, cutoff: float = 30.0, frequencies: int = 128
    ) -> tuple[np.ndarray, np.ndarray]:
        """The quadrature in the detour time ``u`` on ``[0, u_max]``: the
        nodes, and the products ``G(u_j) w_j``, whose sum is one to the
        quadrature's accuracy whatever the band."""
        s, products = detour_rule(count, cutoff=cutoff, frequencies=frequencies)
        return s / self.frequency, products

    def gaussian_defect(
        self, count: int, /, *, cutoff: float = 30.0, frequencies: int = 128
    ) -> float:
        """``sum G(u_j) w_j - 1``: how far the quadrature in ``u`` misses the
        Gaussian identity ``int G du = 1``."""
        return float(
            self.detours(count, cutoff=cutoff, frequencies=frequencies)[1].sum() - 1.0
        )
