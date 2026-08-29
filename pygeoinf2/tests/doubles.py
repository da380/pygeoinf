"""
Adversarial test doubles for the coordinate-free core.

Every other space in the suite is backed by NumPy arrays that *are* their own
components, which means code reaching for array arithmetic or for a coordinate
map works by accident. These doubles remove that accident:

- :class:`Opaque` vectors support no arithmetic at all — no ``+``, no ``*``, no
  ``.copy()`` — so anything not routed through the space raises ``TypeError``.
- :class:`OpaqueSpace` is a ``HilbertSpace`` and *not* a ``CoordinateSpace``,
  so there is no component map to fall back on.
- :class:`StrictSpace` wraps a coordinate space and raises if the coordinate
  map is touched, turning "this algorithm is coordinate-free" into an
  assertion rather than a hope.
- All of them count calls, which is how a claim like "``at()`` evaluates the
  operator once, not twice" gets tested.

A real backend (MFEM) tests one implementation; these test the contract.
"""

from __future__ import annotations

from collections import Counter
from typing import Hashable

import numpy as np
from numpy.random import Generator, default_rng

from pygeoinf2.algebra.spaces import CoordinateSpace, HilbertSpace


class Opaque:
    """A vector that refuses to do arithmetic.

    Holds a NumPy array but exposes none of its operators, so the only way to
    combine two of these is through the space that owns them.
    """

    __slots__ = ("_data",)

    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data, dtype=float)

    @property
    def data(self) -> np.ndarray:
        return self._data

    def __repr__(self) -> str:
        return f"Opaque({self._data!r})"

    # Deliberately absent: __add__, __sub__, __mul__, __neg__, copy, and the
    # in-place forms. Touching any of them is a bug in the caller.


class CallCounter:
    """Records how many times each primitive was used."""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()

    def record(self, name: str) -> None:
        self.counts[name] += 1

    def __getitem__(self, name: str) -> int:
        return self.counts[name]

    def reset(self) -> None:
        self.counts.clear()

    def __repr__(self) -> str:
        return f"CallCounter({dict(self.counts)!r})"


class OpaqueSpace(HilbertSpace[Opaque]):
    """A coordinate-free Hilbert space over :class:`Opaque` vectors.

    The inner product is deliberately not the component dot product — it
    carries a diagonal weight — so any code that quietly substitutes one for
    the other gives wrong answers rather than right ones by luck.
    """

    def __init__(self, weights: np.ndarray) -> None:
        self._weights = np.asarray(weights, dtype=float)
        if np.any(self._weights <= 0.0):
            raise ValueError("weights must be strictly positive.")
        self.calls = CallCounter()

    @property
    def dim(self) -> int:
        return self._weights.size

    def _key(self) -> Hashable:
        return tuple(self._weights.tolist())

    def zero(self) -> Opaque:
        self.calls.record("zero")
        return Opaque(np.zeros(self.dim))

    def copy(self, x: Opaque) -> Opaque:
        self.calls.record("copy")
        return Opaque(x.data.copy())

    def inner_product(self, x: Opaque, y: Opaque) -> float:
        self.calls.record("inner_product")
        return float(np.dot(x.data, self._weights * y.data))

    def axpy(self, a: float, x: Opaque, y: Opaque) -> Opaque:
        self.calls.record("axpy")
        y.data[:] = y.data + a * x.data
        return y

    def scale_inplace(self, a: float, x: Opaque) -> Opaque:
        self.calls.record("scale_inplace")
        x.data[:] = a * x.data
        return x

    def random(self, *, rng: Generator | None = None) -> Opaque:
        self.calls.record("random")
        rng = default_rng() if rng is None else rng
        return Opaque(rng.standard_normal(self.dim))

    def white_noise(self, *, rng: Generator | None = None) -> Opaque:
        self.calls.record("white_noise")
        rng = default_rng() if rng is None else rng
        return Opaque(rng.standard_normal(self.dim) / np.sqrt(self._weights))


class NoCoordinatesError(AssertionError):
    """Raised when code that should be coordinate-free asks for components."""


class StrictSpace(CoordinateSpace):
    """A coordinate space that forbids use of its coordinate map.

    Wraps another coordinate space and forwards everything except
    ``to_components`` and ``from_components``, which raise. Use it to assert
    that an algorithm advertised as coordinate-free really is: if it touches
    the component map anywhere, the test fails loudly instead of passing.
    """

    def __init__(self, base: CoordinateSpace) -> None:
        self._base = base

    @property
    def base(self) -> CoordinateSpace:
        return self._base

    @property
    def dim(self) -> int:
        return self._base.dim

    def _key(self) -> Hashable:
        return (type(self._base).__name__, self._base._key())

    def zero(self):
        return self._base.zero()

    def copy(self, x):
        return self._base.copy(x)

    def inner_product(self, x, y) -> float:
        return self._base.inner_product(x, y)

    def axpy(self, a, x, y):
        return self._base.axpy(a, x, y)

    def scale_inplace(self, a, x):
        return self._base.scale_inplace(a, x)

    def random(self, *, rng: Generator | None = None):
        return self._base.random(rng=rng)

    @property
    def uses_component_fast_paths(self) -> bool:
        # A coordinate map that raises is not one the library may use for its
        # internal arithmetic; the coordinate-free paths are what is under test.
        return False

    def to_components(self, x):
        raise NoCoordinatesError(
            "to_components was called on a StrictSpace: this code path is "
            "supposed to be coordinate-free."
        )

    def from_components(self, c):
        raise NoCoordinatesError(
            "from_components was called on a StrictSpace: this code path is "
            "supposed to be coordinate-free."
        )
