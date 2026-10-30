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

import math
from collections import Counter
from typing import Hashable, Sequence

import numpy as np
from numpy.random import Generator, default_rng

from pygeoinf.algebra.spaces import CoordinateSpace, HilbertSpace


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


# --------------------------------------------------------------------- #
#                  A space whose vectors are not arrays                 #
# --------------------------------------------------------------------- #


class ListVector:
    """A vector that is a plain list of floats and nothing more.

    :class:`Opaque` refuses arithmetic but still *holds* an ndarray, so code
    that unwraps it with ``np.asarray`` or ``.data`` works by accident. This
    one holds a Python list and exposes no ``__array__``, ``__len__``,
    ``__iter__``, ``__getitem__``, ``__float__`` or arithmetic operator at
    all: ``np.asarray(v)`` gives a useless zero-dimensional object array,
    ``np.asarray(v, dtype=float)`` raises, and ``a + b``, ``2.0 * v``,
    ``-v``, ``len(v)`` and ``v[0]`` all raise ``TypeError``. The only way to
    compute with one is through the :class:`ListSpace` that owns it.
    """

    __slots__ = ("_values",)

    def __init__(self, values: Sequence[float]) -> None:
        self._values = [float(v) for v in values]

    @property
    def values(self) -> list[float]:
        """The entries, as the list itself. For the space and for assertions."""
        return self._values

    def __repr__(self) -> str:
        return f"ListVector({self._values!r})"


def _cholesky(matrix: list[list[float]]) -> list[list[float]]:
    """The lower Cholesky factor of a symmetric positive definite nested list.

    Written out in plain Python so that nothing here depends on NumPy's
    linear algebra: this double exists to prove the library needs none of it.

    Raises:
        ValueError: if the matrix is not positive definite.
    """
    n = len(matrix)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = matrix[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if s <= 0.0:
                    raise ValueError("The Gram matrix is not positive definite.")
                lower[i][j] = math.sqrt(s)
            else:
                lower[i][j] = s / lower[j][j]
    return lower


class ListSpace(HilbertSpace[ListVector]):
    """A coordinate-free Hilbert space over :class:`ListVector` vectors.

    The Gram matrix is a fixed symmetric positive definite matrix held as
    nested lists and deliberately *not* diagonal, so the adjoint differs
    from the transpose and white noise differs from standard normal
    components. Every operation is an explicit Python loop over the lists;
    no NumPy array ever stands in for a vector.
    """

    def __init__(self, gram: Sequence[Sequence[float]]) -> None:
        self._gram = [[float(g) for g in row] for row in gram]
        n = len(self._gram)
        if any(len(row) != n for row in self._gram):
            raise ValueError("The Gram matrix must be square.")
        for i in range(n):
            for j in range(i):
                if abs(self._gram[i][j] - self._gram[j][i]) > 1e-12:
                    raise ValueError("The Gram matrix must be symmetric.")
        self._cholesky = _cholesky(self._gram)
        self.calls = CallCounter()

    @property
    def gram(self) -> list[list[float]]:
        """The Gram matrix, as nested lists."""
        return self._gram

    @property
    def dim(self) -> int:
        return len(self._gram)

    def _key(self) -> Hashable:
        return tuple(tuple(row) for row in self._gram)

    def from_list(self, values: Sequence[float]) -> ListVector:
        """A vector with these entries, for building test data."""
        if len(values) != self.dim:
            raise ValueError(f"Expected {self.dim} entries, got {len(values)}.")
        return ListVector(values)

    def zero(self) -> ListVector:
        self.calls.record("zero")
        return ListVector([0.0] * self.dim)

    def copy(self, x: ListVector) -> ListVector:
        self.calls.record("copy")
        return ListVector(x.values)

    def inner_product(self, x: ListVector, y: ListVector) -> float:
        self.calls.record("inner_product")
        total = 0.0
        for i, row in enumerate(self._gram):
            xi = x.values[i]
            for j, g in enumerate(row):
                total += xi * g * y.values[j]
        return total

    def axpy(self, a: float, x: ListVector, y: ListVector) -> ListVector:
        self.calls.record("axpy")
        ys = y.values
        for i, xi in enumerate(x.values):
            ys[i] = ys[i] + a * xi
        return y

    def scale_inplace(self, a: float, x: ListVector) -> ListVector:
        self.calls.record("scale_inplace")
        xs = x.values
        for i in range(len(xs)):
            xs[i] = a * xs[i]
        return x

    def random(self, *, rng: Generator | None = None) -> ListVector:
        self.calls.record("random")
        rng = default_rng() if rng is None else rng
        return ListVector([float(rng.standard_normal()) for _ in range(self.dim)])

    def white_noise(self, *, rng: Generator | None = None) -> ListVector:
        """``c = L^-T xi`` with ``G = L L^T``, by back substitution on lists."""
        self.calls.record("white_noise")
        rng = default_rng() if rng is None else rng
        n = self.dim
        xi = [float(rng.standard_normal()) for _ in range(n)]
        lower = self._cholesky
        c = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = xi[i] - sum(lower[k][i] * c[k] for k in range(i + 1, n))
            c[i] = s / lower[i][i]
        return ListVector(c)
