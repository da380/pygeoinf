"""
Hilbert spaces, coordinate-free at the core.

The base class knows only about vectors and an inner product. Everything to do
with components, basis vectors and matrices lives in ``CoordinateSpace``, which
is an optional capability: a space backed by PETSc or MFEM implements
``HilbertSpace`` alone and remains usable by every coordinate-free algorithm.

See DESIGN.md sections 3.1 and 3.2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Hashable, Sequence

import numpy as np
from numpy.random import Generator, default_rng
from scipy.linalg import solve_triangular

__all__ = [
    "HilbertSpace",
    "ArrayVectorMixin",
    "CoordinateSpace",
    "DiagonalMetricSpace",
    "OrthonormalSpace",
    "EuclideanSpace",
    "Reals",
    "default_rng",
]


_DEFAULT_RNG = default_rng()


def _resolve_rng(rng: Generator | None) -> Generator:
    return _DEFAULT_RNG if rng is None else rng


class HilbertSpace[V](ABC):
    """A real Hilbert space whose vectors are opaque objects of type ``V``.

    Vectors are whatever the backend hands you — a NumPy array, a PETSc ``Vec``,
    an MFEM ``GridFunction`` — and are never wrapped. All arithmetic is mediated
    by the space, which is what allows a backend to be plugged in without
    adapting its vector type.

    In-place contract
    -----------------
    ``axpy`` and ``scale_inplace`` update their target where the backend allows
    it and **return the result**. For an immutable vector type such as
    :class:`Reals` a new object is returned instead, so callers must always use
    the return value rather than relying on mutation.
    """

    # ----------------------------------------------------------------- #
    #                          Abstract interface                       #
    # ----------------------------------------------------------------- #

    @property
    @abstractmethod
    def dim(self) -> int:
        """The dimension of the space."""

    @abstractmethod
    def _key(self) -> Hashable:
        """Structural identity.

        Two spaces of the same type are equal exactly when their keys are.
        A space whose identity depends on a `LinearOperator` must build the key
        from the operator's *parameters*, not the operator itself: operator
        equality is object identity, so a key containing one makes two
        mathematically identical spaces compare unequal.
        """

    @abstractmethod
    def zero(self) -> V:
        """A new zero vector. This allocates; it is a method, not a property."""

    @abstractmethod
    def copy(self, x: V) -> V:
        """An independent copy of ``x``."""

    @abstractmethod
    def inner_product(self, x: V, y: V) -> float:
        """The inner product ``(x, y)``."""

    @abstractmethod
    def axpy(self, a: float, x: V, y: V) -> V:
        """``y <- y + a * x``. Returns the result; see the in-place contract."""

    @abstractmethod
    def scale_inplace(self, a: float, x: V) -> V:
        """``x <- a * x``. Returns the result; see the in-place contract."""

    # ----------------------------------------------------------------- #
    #                              Identity                             #
    # ----------------------------------------------------------------- #

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if type(self) is not type(other):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash((type(self), self._key()))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dim={self.dim})"

    # ----------------------------------------------------------------- #
    #                     Derived vector operations                     #
    # ----------------------------------------------------------------- #

    def add(self, x: V, y: V) -> V:
        """``x + y``, out of place."""
        return self.axpy(1.0, x, self.copy(y))

    def subtract(self, x: V, y: V) -> V:
        """``x - y``, out of place."""
        return self.axpy(-1.0, y, self.copy(x))

    def scale(self, a: float, x: V) -> V:
        """``a * x``, out of place."""
        return self.scale_inplace(a, self.copy(x))

    def negative(self, x: V) -> V:
        """``-x``, out of place."""
        return self.scale(-1.0, x)

    def squared_norm(self, x: V) -> float:
        return self.inner_product(x, x)

    def norm(self, x: V) -> float:
        return float(np.sqrt(self.squared_norm(x)))

    def mean(self, vectors: Sequence[V]) -> V:
        """The sample mean of a sequence of vectors."""
        n = len(vectors)
        if n == 0:
            raise ValueError("Cannot take the mean of an empty sequence.")
        result = self.zero()
        for x in vectors:
            result = self.axpy(1.0 / n, x, result)
        return result

    def gram_schmidt(self, vectors: Sequence[V], /, *, tol: float = 1e-12) -> list[V]:
        """Orthonormalise a sequence of linearly independent vectors."""
        result: list[V] = []
        for i, vector in enumerate(vectors):
            v = self.copy(vector)
            for w in result:
                v = self.axpy(-self.inner_product(v, w), w, v)
            norm = self.norm(v)
            if norm <= tol:
                raise ValueError(
                    f"Vector {i} is linearly dependent on its predecessors "
                    f"(residual norm {norm:g})."
                )
            result.append(self.scale_inplace(1.0 / norm, v))
        return result

    # ----------------------------------------------------------------- #
    #                             Randomness                            #
    # ----------------------------------------------------------------- #

    def random(self, rng: Generator | None = None) -> V:
        """An arbitrary random vector, for testing.

        This is **not** white noise: no claim is made about its covariance.
        Use :meth:`white_noise` when the distribution matters.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement random(). It is needed "
            f"by the checks in pygeoinf2.testing and by randomised algorithms."
        )

    def white_noise(self, rng: Generator | None = None) -> V:
        """A sample whose covariance is the identity *on this space*.

        That is, ``E[(x, u) (x, v)] == (u, v)`` for all ``u``, ``v``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement white_noise()."
        )


class ArrayVectorMixin:
    """Vector operations for spaces whose vectors are NumPy arrays.

    Supplies ``copy``, ``axpy`` and ``scale_inplace``; ``zero`` comes from
    :class:`CoordinateSpace`, which knows the component layout.
    """

    def copy(self, x: np.ndarray) -> np.ndarray:
        return x.copy()

    def axpy(self, a: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        y += a * x
        return y

    def scale_inplace(self, a: float, x: np.ndarray) -> np.ndarray:
        x *= a
        return x


class CoordinateSpace[V](HilbertSpace[V], ABC):
    """A Hilbert space with a distinguished finite basis.

    The Gram (mass) matrix ``G`` relates the inner product to the component
    basis via ``(x, y) == c_x . G c_y``. It lives here rather than on
    ``HilbertSpace`` because it is meaningless without a basis. Subclasses
    supply its action through :meth:`apply_gram` and :meth:`solve_gram`; the
    default is the identity, i.e. an orthonormal basis.
    """

    @abstractmethod
    def to_components(self, x: V) -> np.ndarray:
        """The components of ``x`` in the space's basis."""

    @abstractmethod
    def from_components(self, c: np.ndarray) -> V:
        """The vector with components ``c``."""

    # ----------------------------------------------------------------- #
    #                              The metric                           #
    # ----------------------------------------------------------------- #

    def apply_gram(self, c: np.ndarray) -> np.ndarray:
        """``G c``. The identity by default."""
        return c

    def solve_gram(self, c: np.ndarray) -> np.ndarray:
        """``G^-1 c``. The identity by default."""
        return c

    def white_noise_components(self, rng: Generator | None = None) -> np.ndarray:
        """Components drawn from ``N(0, G^-1)``.

        With ``G == L L^T`` the draw is ``c = L^-T xi``, which gives
        ``Cov(c) == L^-T L^-1 == G^-1``. Note the transpose: ``L^-1 xi`` would
        be wrong for a non-symmetric factor.

        The default forms the Gram matrix densely and factorises it once,
        caching the factor. Subclasses with a cheaper factor override this;
        both shipped ones do.
        """
        xi = _resolve_rng(rng).standard_normal(self.dim)
        return solve_triangular(self._gram_cholesky.T, xi, lower=False)

    @cached_property
    def _gram_cholesky(self) -> np.ndarray:
        """The lower Cholesky factor ``L`` of the Gram matrix, computed once."""
        return np.linalg.cholesky(self.gram_matrix())

    @property
    def is_orthonormal(self) -> bool:
        """True when the basis is orthonormal, so that ``G`` is the identity."""
        return False

    def gram_matrix(self) -> np.ndarray:
        """The Gram matrix, formed column by column. Costs ``dim`` applications."""
        matrix = np.zeros((self.dim, self.dim))
        c = np.zeros(self.dim)
        for i in range(self.dim):
            c[i] = 1.0
            matrix[:, i] = self.apply_gram(c)
            c[i] = 0.0
        return matrix

    # ----------------------------------------------------------------- #
    #                    Supplied from the coordinate map               #
    # ----------------------------------------------------------------- #

    def inner_product(self, x: V, y: V) -> float:
        return float(
            np.dot(self.to_components(x), self.apply_gram(self.to_components(y)))
        )

    def zero(self) -> V:
        return self.from_components(np.zeros(self.dim))

    def basis_vector(self, i: int) -> V:
        """The ``i``-th basis vector."""
        if not 0 <= i < self.dim:
            raise IndexError(f"Basis index {i} out of range for dim {self.dim}.")
        c = np.zeros(self.dim)
        c[i] = 1.0
        return self.from_components(c)

    def random(self, rng: Generator | None = None) -> V:
        return self.from_components(_resolve_rng(rng).standard_normal(self.dim))

    def white_noise(self, rng: Generator | None = None) -> V:
        """A sample with identity covariance on this space.

        The components are drawn from ``N(0, G^-1)``, which is what makes the
        covariance the identity *in the space's own inner product* rather than
        in the component basis. Drawing standard normal components instead
        gives covariance ``G``, which is the mistake this method exists to
        avoid; see DESIGN.md section 9.
        """
        return self.from_components(self.white_noise_components(rng))

    # ----------------------------------------------------------------- #
    #                     The functional pairing axiom                  #
    # ----------------------------------------------------------------- #

    def representer(self, derivative_components: np.ndarray) -> V:
        """The Riesz representer of the functional with the given derivative.

        Given ``g`` such that the functional acts as ``x -> g . c_x`` — which is
        what a numerical adjoint method returns — this applies ``G^-1`` to give
        the vector ``v`` with ``(v, x) == g . c_x`` for all ``x``. Skipping that
        step is the classic adjoint-method error; see DESIGN.md section 5.6.
        """
        return self.from_components(self.solve_gram(np.asarray(derivative_components)))


class DiagonalMetricSpace[V](CoordinateSpace[V], ABC):
    """A coordinate space whose basis is orthogonal but not normalised.

    The Gram matrix is ``diag(metric_values)``, so every metric operation is a
    pointwise multiply or divide. This is the shape of the harmonic bases used
    by the symmetric spaces.
    """

    def __init__(self, metric_values: np.ndarray) -> None:
        values = np.asarray(metric_values, dtype=float)
        if values.ndim != 1:
            raise ValueError("metric_values must be a one-dimensional array.")
        if np.any(values <= 0.0):
            raise ValueError("metric_values must be strictly positive.")
        self._metric_values = values
        self._sqrt_metric_values = np.sqrt(values)

    @property
    def metric_values(self) -> np.ndarray:
        """The diagonal of the Gram matrix."""
        return self._metric_values

    @property
    def dim(self) -> int:
        return self._metric_values.size

    def apply_gram(self, c: np.ndarray) -> np.ndarray:
        return self._metric_values * c

    def solve_gram(self, c: np.ndarray) -> np.ndarray:
        return c / self._metric_values

    def white_noise_components(self, rng: Generator | None = None) -> np.ndarray:
        xi = _resolve_rng(rng).standard_normal(self.dim)
        return xi / self._sqrt_metric_values

    def inner_product(self, x: V, y: V) -> float:
        cx = self.to_components(x)
        cy = self.to_components(y)
        return float(np.dot(cx, self._metric_values * cy))


class OrthonormalSpace[V](CoordinateSpace[V], ABC):
    """A coordinate space with an orthonormal basis, so ``G`` is the identity."""

    @property
    def is_orthonormal(self) -> bool:
        return True

    def white_noise_components(self, rng: Generator | None = None) -> np.ndarray:
        return _resolve_rng(rng).standard_normal(self.dim)

    def gram_matrix(self) -> np.ndarray:
        return np.identity(self.dim)

    def inner_product(self, x: V, y: V) -> float:
        return float(np.dot(self.to_components(x), self.to_components(y)))


class EuclideanSpace(ArrayVectorMixin, OrthonormalSpace[np.ndarray]):
    """``R^n`` with the standard inner product. Vectors are NumPy arrays.

    The coordinate map is the identity and does **not** copy: the array handed
    to ``from_components`` becomes the vector, and ``to_components`` returns the
    vector itself. That is deliberate — zero-copy interop is the point of
    leaving vectors as raw backend objects — but it means a caller who mutates
    an array afterwards mutates the vector too.
    """

    def __init__(self, dim: int) -> None:
        if dim < 0:
            raise ValueError("dim must be non-negative.")
        self._dim = int(dim)

    @property
    def dim(self) -> int:
        return self._dim

    def _key(self) -> Hashable:
        return self._dim

    def to_components(self, x: np.ndarray) -> np.ndarray:
        return x

    def from_components(self, c: np.ndarray) -> np.ndarray:
        return c

    def zero(self) -> np.ndarray:
        return np.zeros(self._dim)


class Reals(OrthonormalSpace[float]):
    """The real line as a one-dimensional Hilbert space.

    Vectors are plain floats, so a functional evaluates to a number rather than
    to an array of length one. Floats are immutable, so ``axpy`` and
    ``scale_inplace`` return new values; see the in-place contract on
    :class:`HilbertSpace`.
    """

    @property
    def dim(self) -> int:
        return 1

    def _key(self) -> Hashable:
        return ()

    def zero(self) -> float:
        return 0.0

    def copy(self, x: float) -> float:
        return float(x)

    def axpy(self, a: float, x: float, y: float) -> float:
        return float(y + a * x)

    def scale_inplace(self, a: float, x: float) -> float:
        return float(a * x)

    def inner_product(self, x: float, y: float) -> float:
        return float(x) * float(y)

    def to_components(self, x: float) -> np.ndarray:
        return np.array([float(x)])

    def from_components(self, c: np.ndarray) -> float:
        return float(np.asarray(c).reshape(-1)[0])

    def __repr__(self) -> str:
        return "Reals()"
