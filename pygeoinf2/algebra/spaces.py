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
from typing import TYPE_CHECKING, Hashable, Sequence

import numpy as np
from numpy.random import Generator, default_rng
from scipy.linalg import solve_triangular

if TYPE_CHECKING:  # pragma: no cover
    from .operators import LinearOperator

__all__ = [
    "HilbertSpace",
    "ArrayVectorMixin",
    "CoordinateSpace",
    "DiagonalMetricSpace",
    "HilbertModule",
    "MassWeightedSpace",
    "require_module",
    "OrthonormalSpace",
    "EuclideanSpace",
    "Reals",
    "default_rng",
]


_DEFAULT_RNG = default_rng()

# Reorthogonalise when a projection loses more than this fraction of the norm.
# The classical Daniel-Gragg-Kaufman-Stewart criterion, with the usual value.
_REORTHOGONALISATION_THRESHOLD = 1.0 / np.sqrt(2.0)


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

    def shares_vectors_with(self, other: "HilbertSpace", /) -> bool:
        """Whether *other*'s vectors are usable here without conversion.

        Two spaces can hold the same vectors and differ only in their inner
        product — a Sobolev space and the Lebesgue space over the same grid,
        or a mass-weighted space and its base. Where that is so, moving a
        vector between them is a no-op rather than a round trip through
        components, which on a spectral space is two transforms.

        Conservative by default: only a space itself. A family that knows
        better says so.
        """
        return self is other

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
        """The squared norm (x, x)."""
        return self.inner_product(x, x)

    def norm(self, x: V) -> float:
        """The norm sqrt((x, x))."""
        return float(np.sqrt(self.squared_norm(x)))

    def mean(self, vectors: Sequence[V]) -> V:
        """The sample mean of a sequence of vectors.

        Args:
            vectors: at least one vector.

        Returns:
            Their mean.

        Raises:
            ValueError: if the sequence is empty, there being no mean of
                nothing rather than a zero.
        """
        n = len(vectors)
        if n == 0:
            raise ValueError("Cannot take the mean of an empty sequence.")
        result = self.zero()
        for x in vectors:
            result = self.axpy(1.0 / n, x, result)
        return result

    def gram_schmidt(self, vectors: Sequence[V], /, *, rtol: float = 1e-12) -> list[V]:
        """Orthonormalise a sequence of linearly independent vectors.

        Args:
            vectors: the vectors, which must be independent.
            rtol: a vector is taken as dependent when what is left of it after
                projecting off its predecessors is this small a fraction of
                what went in. Relative, so it does not depend on scale.

        Returns:
            An orthonormal sequence spanning the same space.

        Raises:
            ValueError: on the first dependent vector, naming it. Use
                :meth:`orthonormal_basis` when a rank-deficient set should be
                reduced rather than rejected.
        """
        result: list[V] = []
        for i, vector in enumerate(vectors):
            v, norm, original = self._orthogonalise_against(vector, result)
            if norm <= rtol * original:
                raise ValueError(
                    f"Vector {i} is linearly dependent on its predecessors "
                    f"(residual norm {norm:g} against {original:g})."
                )
            result.append(self.scale_inplace(1.0 / norm, v))
        return result

    def orthonormal_basis(
        self, vectors: Sequence[V], /, *, rtol: float = 1e-10
    ) -> list[V]:
        """An orthonormal basis for the span, dropping dependent vectors.

        What a rank-revealing method wants: a randomised range finder feeds in
        blocks of probes that may well be numerically dependent, and needs the
        independent part rather than an exception.

        Args:
            vectors: the vectors, which need not be independent.
            rtol: the dependence threshold, as in :meth:`gram_schmidt`. Looser
                here by default, since dropping a marginal vector costs less
                than keeping a numerically dependent one.

        Returns:
            An orthonormal basis for the span, which may be shorter than the
            input.
        """
        result: list[V] = []
        for vector in vectors:
            v, norm, original = self._orthogonalise_against(vector, result)
            if norm > rtol * original:
                result.append(self.scale_inplace(1.0 / norm, v))
        return result

    def _orthogonalise_against(
        self, vector: V, basis: Sequence[V]
    ) -> tuple[V, float, float]:
        """Project a vector off an orthonormal basis, reorthogonalising once.

        A single modified Gram-Schmidt pass loses orthogonality when the new
        vector is nearly in the span, which is exactly the regime a
        rank-revealing method works in. The classical remedy is to repeat the
        projection when the norm drops sharply -- "twice is enough" -- which
        costs one extra pass only in the cases that need it.

        Returns the projected vector, its norm, and the norm it started with.
        """
        original = self.norm(vector)
        v = self.copy(vector)
        for w in basis:
            v = self.axpy(-self.inner_product(v, w), w, v)
        norm = self.norm(v)
        if norm < _REORTHOGONALISATION_THRESHOLD * original:
            for w in basis:
                v = self.axpy(-self.inner_product(v, w), w, v)
            norm = self.norm(v)
        return v, norm, max(original, 1e-300)

    # ----------------------------------------------------------------- #
    #                             Randomness                            #
    # ----------------------------------------------------------------- #

    def random(self, *, rng: Generator | None = None) -> V:
        """An arbitrary random vector, for testing.

        This is **not** white noise: no claim is made about its covariance.
        Use :meth:`white_noise` when the distribution matters.

        Returns:
            A vector of this space.

        Raises:
            NotImplementedError: unless the space provides one. The checks in
                :mod:`pygeoinf2.testing` and the randomised algorithms need it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement random(). It is needed "
            f"by the checks in pygeoinf2.testing and by randomised algorithms."
        )

    def white_noise(self, *, rng: Generator | None = None) -> V:
        """A sample whose covariance is the identity *on this space*.

        That is, ``E[(x, u) (x, v)] == (u, v)`` for all ``u``, ``v``. Note this
        is a statement about the space's own inner product, not about the
        components -- on a space with a non-trivial metric the two differ, and
        that difference is the reason this is a method rather than a call to
        ``standard_normal``.

        Returns:
            A draw.

        Raises:
            NotImplementedError: unless the space provides one.
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
        """An independent copy of the array."""
        return x.copy()

    def axpy(self, a: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """y += a * x, in place."""
        y += a * x
        return y

    def scale_inplace(self, a: float, x: np.ndarray) -> np.ndarray:
        """x *= a, in place."""
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

    def white_noise_components(self, *, rng: Generator | None = None) -> np.ndarray:
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

    @property
    def has_diagonal_metric(self) -> bool:
        """True when the Gram matrix is diagonal in this space's basis.

        This is what decides whether an operator that is diagonal in the same
        basis is self-adjoint: ``A`` with component matrix ``diag(d)`` has
        Galerkin matrix ``G diag(d)``, which is symmetric exactly when ``G``
        and ``diag(d)`` commute -- so, for a general ``d``, exactly when ``G``
        is itself diagonal.
        """
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
        """``c_x . G c_y``, from the coordinate map and the metric."""
        return float(
            np.dot(self.to_components(x), self.apply_gram(self.to_components(y)))
        )

    def squared_norm(self, x: V) -> float:
        """``c_x . G c_x``, converting ``x`` once rather than twice.

        The base class takes ``inner_product(x, x)``, which on a spectral
        space is two analyses of the same field; a Krylov loop takes a norm
        every iteration.
        """
        if not self.uses_component_fast_paths:
            return self.inner_product(x, x)
        c = self.to_components(x)
        return float(np.dot(c, self.apply_gram(c)))

    # ----------------------------------------------------------------- #
    #                Arithmetic on component arrays (fast paths)         #
    # ----------------------------------------------------------------- #
    #
    # Coordinate-free when it must be, and not otherwise. An inner product on
    # a spectral space transforms *both* of its arguments, so any routine that
    # takes O(k^2) inner products of k fixed vectors -- Gram-Schmidt, Lanczos
    # reorthogonalisation, a low-rank factor's adjoint -- pays O(k^2)
    # transforms for arithmetic that needs k. Measured on a sphere at lmax 64:
    # orthonormalising 50 fields cost 2650 analyses and 1.9 s through
    # ``inner_product``, and 100 transforms and 0.1 s on component arrays;
    # a 30-step Lanczos run 1047 analyses against 152. The methods below do
    # that arithmetic on ``(dim, k)`` arrays, converting once each way, with
    # the metric entering only through ``apply_gram`` -- which is what keeps
    # them right on a non-diagonal Gram matrix. The ``HilbertSpace`` versions
    # remain the fallback for spaces with no component map.

    @property
    def uses_component_fast_paths(self) -> bool:
        """Whether the library may do its internal arithmetic in components.

        True here. A subclass whose coordinate map exists only formally, or is
        too expensive to be worth a round trip, returns False and gets the
        coordinate-free code paths instead.
        """
        return True

    def components_of(self, vectors: Sequence[V], /) -> np.ndarray:
        """The components of several vectors, as the columns of one array.

        Args:
            vectors: the vectors.

        Returns:
            A ``(dim, k)`` array, one column per vector.
        """
        vectors = tuple(vectors)
        if not vectors:
            return np.zeros((self.dim, 0))
        return np.stack([self.to_components(v) for v in vectors], axis=1)

    def vectors_from(self, columns: np.ndarray, /) -> list[V]:
        """The vectors whose components are the columns of an array.

        Each column is copied, so the vectors do not alias the array or each
        other -- a space whose coordinate map does not copy would otherwise
        hand out views that in-place operations then corrupt.

        Args:
            columns: a ``(dim, k)`` array.

        Returns:
            ``k`` vectors.
        """
        return [
            self.from_components(np.array(columns[:, j], dtype=float))
            for j in range(columns.shape[1])
        ]

    def apply_gram_to_columns(self, columns: np.ndarray, /) -> np.ndarray:
        """``G`` applied to every column of an array.

        The default applies :meth:`apply_gram` column by column; a space with
        a structured metric overrides it with one vectorised operation.
        """
        if columns.shape[1] == 0:
            return columns.copy()
        return np.stack([self.apply_gram(columns[:, j]) for j in range(columns.shape[1])], axis=1)

    def gram_diagonal(self) -> np.ndarray:
        """The diagonal of the Gram matrix, ``(e_i, e_i)`` for each basis vector.

        What the Galerkin diagonal of a diagonal operator needs on a metric
        that is not itself diagonal. The default reads it one basis vector at
        a time through :meth:`apply_gram`, which costs ``dim`` metric
        applications and never forms the matrix; a space that holds its
        metric overrides it with a read.
        """
        diagonal = np.empty(self.dim)
        c = np.zeros(self.dim)
        for i in range(self.dim):
            c[i] = 1.0
            diagonal[i] = self.apply_gram(c)[i]
            c[i] = 0.0
        return diagonal

    def solve_gram_to_columns(self, columns: np.ndarray, /) -> np.ndarray:
        """``G^-1`` applied to every column of an array.

        The counterpart of :meth:`apply_gram_to_columns`; the same default and
        the same invitation to override. Converting a Galerkin matrix to the
        components form is this applied to it, so on a space with a dense
        Gram matrix the override is one multi-right-hand-side solve rather
        than ``dim`` separate ones.
        """
        if columns.shape[1] == 0:
            return columns.copy()
        return np.stack([self.solve_gram(columns[:, j]) for j in range(columns.shape[1])], axis=1)

    def _orthonormalise_columns(
        self,
        columns: np.ndarray,
        /,
        *,
        rtol: float,
        against: np.ndarray | None = None,
        strict: bool = False,
    ) -> tuple[np.ndarray, list[int]]:
        """Gram-Schmidt on component columns, in the space's metric.

        Classical Gram-Schmidt with a second pass whenever the first one
        removes more than ``1 - 1/sqrt(2)`` of a column's norm -- Kahan's
        "twice is enough" -- which costs one metric application per pass
        rather than one per basis vector, and is as stable as the modified
        form with that test in place.

        Args:
            columns: a ``(dim, k)`` array of candidate columns.
            rtol: a column is dependent when what is left of it after
                projection is below this fraction of what went in.
            against: an already ``G``-orthonormal ``(dim, m)`` block to
                project off first; its columns are not returned.
            strict: raise on a dependent column instead of dropping it.

        Returns:
            The accepted orthonormal columns as a ``(dim, r)`` array, and the
            indices of the input columns they came from.

        Raises:
            ValueError: on a dependent column, when *strict*.
        """
        dim = self.dim
        count = 0 if against is None else against.shape[1]
        basis = np.empty((dim, count + columns.shape[1]))
        if against is not None:
            basis[:, :count] = against
        kept: list[int] = []
        for index in range(columns.shape[1]):
            v = np.array(columns[:, index], dtype=float)
            weighted = self.apply_gram(v)
            original = float(np.sqrt(max(float(v @ weighted), 0.0)))
            norm = original
            if count > 0 and original > 0.0:
                for _ in range(2):
                    current = basis[:, :count]
                    v -= current @ (current.T @ weighted)
                    weighted = self.apply_gram(v)
                    before, norm = norm, float(np.sqrt(max(float(v @ weighted), 0.0)))
                    if norm >= _REORTHOGONALISATION_THRESHOLD * before:
                        break
            if norm <= rtol * max(original, 1e-300):
                if strict:
                    raise ValueError(
                        f"Vector {index} is linearly dependent on its predecessors "
                        f"(residual norm {norm:g} against {original:g})."
                    )
                continue
            basis[:, count] = v / norm
            count += 1
            kept.append(index)
        start = 0 if against is None else against.shape[1]
        return basis[:, start:count], kept

    def gram_schmidt(self, vectors: Sequence[V], /, *, rtol: float = 1e-12) -> list[V]:
        """Orthonormalise independent vectors; see :meth:`HilbertSpace.gram_schmidt`.

        Done on component arrays when :attr:`uses_component_fast_paths`
        allows, converting each vector once each way.

        Args:
            vectors: the vectors, which must be independent.
            rtol: the dependence threshold, relative to each vector's norm.

        Returns:
            An orthonormal sequence spanning the same space.

        Raises:
            ValueError: on the first dependent vector, naming it.
        """
        if not self.uses_component_fast_paths:
            return super().gram_schmidt(vectors, rtol=rtol)
        orthonormal, _ = self._orthonormalise_columns(
            self.components_of(vectors), rtol=rtol, strict=True
        )
        return self.vectors_from(orthonormal)

    def orthonormal_basis(
        self, vectors: Sequence[V], /, *, rtol: float = 1e-10
    ) -> list[V]:
        """An orthonormal basis for the span; see :meth:`HilbertSpace.orthonormal_basis`.

        Done on component arrays when :attr:`uses_component_fast_paths`
        allows, converting each vector once each way.

        Args:
            vectors: the vectors, which need not be independent.
            rtol: the dependence threshold, relative to each vector's norm.

        Returns:
            An orthonormal basis for the span, possibly shorter than the input.
        """
        if not self.uses_component_fast_paths:
            return super().orthonormal_basis(vectors, rtol=rtol)
        orthonormal, _ = self._orthonormalise_columns(
            self.components_of(vectors), rtol=rtol
        )
        return self.vectors_from(orthonormal)

    def zero(self) -> V:
        """The vector whose components are all zero."""
        return self.from_components(np.zeros(self.dim))

    def basis_vector(self, i: int) -> V:
        """The ``i``-th basis vector.

        Args:
            i: which one, from zero.

        Returns:
            The vector whose components are one there and zero elsewhere.
            Note this is a *basis* vector, orthonormal only where the metric
            is the identity.

        Raises:
            IndexError: if the index is outside the dimension.
        """
        if not 0 <= i < self.dim:
            raise IndexError(f"Basis index {i} out of range for dim {self.dim}.")
        c = np.zeros(self.dim)
        c[i] = 1.0
        return self.from_components(c)

    def random(self, *, rng: Generator | None = None) -> V:
        """A vector with independent standard normal *components*.

        Not white noise: its covariance is ``G``, not the identity. Use
        :meth:`white_noise` when the distribution matters.
        """
        return self.from_components(_resolve_rng(rng).standard_normal(self.dim))

    def white_noise(self, *, rng: Generator | None = None) -> V:
        """A sample with identity covariance on this space.

        The components are drawn from ``N(0, G^-1)``, which is what makes the
        covariance the identity *in the space's own inner product* rather than
        in the component basis. Drawing standard normal components instead
        gives covariance ``G``, which is the mistake this method exists to
        avoid; see DESIGN.md section 9.
        """
        return self.from_components(self.white_noise_components(rng=rng))

    # ----------------------------------------------------------------- #
    #                     The functional pairing axiom                  #
    # ----------------------------------------------------------------- #

    def coordinate_projection(self) -> "LinearOperator[V, np.ndarray]":
        """The map to this space's components, as an operator.

        ``x -> c_x`` into a Euclidean space. Its adjoint is
        :meth:`representer`, which is the whole reason it is worth having as an
        operator rather than a function: it carries the metric with it, and a
        composition involving it stays correct.
        """
        from .operators import LinearOperator

        codomain = EuclideanSpace(self.dim)
        return LinearOperator.from_callables(
            self,
            codomain,
            self.to_components,
            adjoint=self.representer,
        )

    def coordinate_inclusion(self) -> "LinearOperator[np.ndarray, V]":
        """The map from components into this space, as an operator.

        ``c -> from_components(c)``. Its adjoint is ``G c_x``, the *derivative*
        components — not the components themselves, which is the distinction of
        DESIGN.md section 5.6 in its smallest possible setting.
        """
        from .operators import LinearOperator

        domain = EuclideanSpace(self.dim)
        return LinearOperator.from_callables(
            domain,
            self,
            self.from_components,
            adjoint=lambda x: self.apply_gram(self.to_components(x)),
        )

    def representer(self, derivative_components: np.ndarray) -> V:
        """The Riesz representer of the functional with the given derivative.

        Given ``g`` such that the functional acts as ``x -> g . c_x`` — which is
        what a numerical adjoint method returns — this applies ``G^-1`` to give
        the vector ``v`` with ``(v, x) == g . c_x`` for all ``x``. Skipping that
        step is the classic adjoint-method error; see DESIGN.md section 5.6.
        """
        return self.from_components(self.solve_gram(np.asarray(derivative_components)))


class HilbertModule[V](HilbertSpace[V], ABC):
    """A space whose vectors can also be multiplied together, pointwise.

    A capability, in the same sense as :class:`CoordinateSpace`: spaces whose
    vectors are *functions* have it, and code that needs a pointwise product
    asks for it rather than the core assuming every vector is one. A finite
    element space can opt in; a space of abstract coefficients cannot, and
    should not have to pretend.

    What it buys is a coefficient that varies in space. Multiplication by a
    field is a linear operator, and the operators built from it — a variable
    wave speed, a variable rigidity, a mask — are where a physical model stops
    being invariant.

    **The product of two band-limited functions is not band-limited.** Both
    implementations here multiply on the grid, so the result is the exact
    product sampled and then projected back, which aliases whatever lies above
    the truncation. That is inherent rather than a defect, but it means the
    truncation has to be chosen with the products in mind and not just the
    fields.
    """

    @abstractmethod
    def multiply(self, x: V, y: V) -> V:
        """The pointwise product of two vectors."""

    @abstractmethod
    def sqrt(self, x: V) -> V:
        """The pointwise square root of a vector."""

    def multiplication_operator(self, f: V, /) -> "LinearOperator[V, V]":
        """The operator ``u -> f u``.

        **No self-adjointness is claimed here.** Multiplication is self-adjoint
        with respect to the ``L2`` inner product, and a space that weights its
        modes — a Sobolev space — has a different one. A subclass that knows
        its inner product is the ``L2`` one should override and say so; see
        :meth:`~pygeoinf2.symmetric_space.base.SymmetricSpace.multiplication_operator`,
        which lifts the ``L2`` operator through its metric instead.
        """
        from .operators import LinearOperator

        return LinearOperator.from_callables(self, self, lambda u: self.multiply(f, u))


def require_module(*spaces: HilbertSpace) -> None:
    """Raise unless every space supports pointwise multiplication.

    The counterpart of :func:`~pygeoinf2.algebra.operators.require_coordinates`,
    so an operation that needs fields to multiply fails by name rather than by
    ``AttributeError``.

    Args:
        *spaces: the spaces to check.

    Raises:
        TypeError: if any of them is not a ``HilbertModule``.
    """
    for space in spaces:
        if not isinstance(space, HilbertModule):
            raise TypeError(
                f"{type(space).__name__} does not support pointwise "
                f"multiplication, and this operation requires it. Its vectors "
                f"are not functions on a common domain."
            )


class MassWeightedSpace[V](HilbertSpace[V]):
    """``(x, y)_V == (M x, y)_base``, for a positive definite mass operator.

    The construction DESIGN.md section 3.5 sets against the Gram matrix, and
    the distinction is worth restating because they are easy to confuse:

    =========================  =========================================
    ``CoordinateSpace`` Gram   the inner product against the *components*
    a mass operator            one inner product against *another*, on the
                               same vectors
    =========================  =========================================

    Only the first is automatic. This is the second, and it needs no
    coordinates at all — just the base's inner product and ``M`` — so it works
    over a backend that has no component map. When the base *does* have
    coordinates so does this, and the two compose: the Gram matrix here is the
    base's times the mass operator's.

    Chaining these is how the concrete spaces are built. v1's are exactly such
    a chain: component-Euclidean, then L2 with a diagonal Gram, then Sobolev
    through an invariant mass operator on L2.

    The point of having it is :meth:`LinearOperator.from_formal_adjoint`: it is
    usually far easier to write down an operator's adjoint with respect to the
    *base* inner product than the weighted one, and the lift is exact.
    """

    def __init__(
        self,
        base: HilbertSpace[V],
        mass: "LinearOperator",
        /,
        *,
        mass_solver: object | None = None,
    ) -> None:
        """
        Args:
            base: the space whose inner product is being reweighted.
            mass: ``M``, an operator on *base* claiming SELF_ADJOINT and
                POSITIVE_DEFINITE. The claims are the caller's; verify them
                with :func:`~pygeoinf2.testing.check_traits`.
            mass_solver: how to apply ``M^-1``, which the lifted adjoint needs.
                A :class:`~pygeoinf2.numerics.solvers.LinearSolver`, or an
                operator that *is* the inverse. Defaults to conjugate
                gradients, and is free when the mass operator is diagonal.

                v1 makes the caller supply ``inverse_mass_operator`` outright;
                deriving it is one fewer thing to get wrong, and it makes the
                construction usable when the inverse has no closed form.

        Raises:
            ValueError: if the mass operator is not an endomorphism of *base*,
                or does not claim positive definiteness.
        """
        from ..traits import Traits as _Traits

        if mass.domain != base or mass.codomain != base:
            raise ValueError(
                f"The mass operator must map {base!r} to itself; it maps "
                f"{mass.domain!r} to {mass.codomain!r}."
            )
        required = _Traits.SELF_ADJOINT | _Traits.POSITIVE_DEFINITE
        missing = required & ~mass.traits
        if missing:
            raise ValueError(
                f"The mass operator must claim {required!s}; it claims "
                f"{mass.traits!s} (missing {missing!s}). Attach the traits "
                f"with with_traits() and verify them with "
                f"testing.check_traits()."
            )
        self._base = base
        self._mass = mass
        self._mass_solver = mass_solver

    @property
    def base(self) -> HilbertSpace[V]:
        """The space whose inner product is being reweighted."""
        return self._base

    @property
    def mass(self) -> "LinearOperator":
        """``M``, the operator defining the weighting."""
        return self._mass

    @cached_property
    def mass_inverse(self) -> "LinearOperator":
        """``M^-1``, from the solver, built once."""
        from ..numerics.solvers import CGSolver, LinearSolver

        solver = self._mass_solver
        if solver is None:
            solver = CGSolver()
        if isinstance(solver, LinearSolver):
            return solver(self._mass)
        return solver

    @property
    def dim(self) -> int:
        """The base's dimension: reweighting does not change the vectors."""
        return self._base.dim

    def _key(self) -> Hashable:
        return (self._base, id(self._mass))

    def shares_vectors_with(self, other: HilbertSpace, /) -> bool:
        """True for the base and for anything the base shares vectors with."""
        if self is other:
            return True
        if other is self._base or self._base.shares_vectors_with(other):
            return True
        return isinstance(other, MassWeightedSpace) and self._base.shares_vectors_with(
            other.base
        )

    def zero(self) -> V:
        """The base's zero vector."""
        return self._base.zero()

    def copy(self, x: V) -> V:
        """Delegated: the vectors are the base's."""
        return self._base.copy(x)

    def inner_product(self, x: V, y: V) -> float:
        """``(M x, y)_base``."""
        return self._base.inner_product(self._mass(x), y)

    def axpy(self, a: float, x: V, y: V) -> V:
        """Delegated: linear structure is unchanged by the weighting."""
        return self._base.axpy(a, x, y)

    def scale_inplace(self, a: float, x: V) -> V:
        """Delegated."""
        return self._base.scale_inplace(a, x)

    def random(self, *, rng: Generator | None = None) -> V:
        """Delegated. Not a white-noise draw; see :meth:`white_noise`."""
        return self._base.random(rng=rng)

    def __repr__(self) -> str:
        return f"MassWeightedSpace({self._base!r})"


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
    def has_diagonal_metric(self) -> bool:
        """True by construction."""
        return True

    @property
    def is_orthonormal(self) -> bool:
        """True when every metric value is one.

        Worth detecting rather than assuming false: a space built through this
        class may still have a trivial metric, and saying so lets the faster
        paths be taken.
        """
        return bool(np.all(self._metric_values == 1.0))

    @property
    def dim(self) -> int:
        """The dimension, taken from the metric."""
        return self._metric_values.size

    def apply_gram(self, c: np.ndarray) -> np.ndarray:
        """``G c``, a pointwise multiply."""
        return self._metric_values * c

    def solve_gram(self, c: np.ndarray) -> np.ndarray:
        """``G^-1 c``, a pointwise divide."""
        return c / self._metric_values

    def apply_gram_to_columns(self, columns: np.ndarray, /) -> np.ndarray:
        """``G`` on every column: one broadcast multiply."""
        return self._metric_values[:, None] * columns

    def solve_gram_to_columns(self, columns: np.ndarray, /) -> np.ndarray:
        """``G^-1`` on every column: one broadcast divide."""
        return columns / self._metric_values[:, None]

    def gram_diagonal(self) -> np.ndarray:
        """The metric values themselves."""
        return np.array(self._metric_values)

    def white_noise_components(self, *, rng: Generator | None = None) -> np.ndarray:
        """Components drawn from ``N(0, G^-1)``, using the diagonal factor."""
        xi = _resolve_rng(rng).standard_normal(self.dim)
        return xi / self._sqrt_metric_values

    def inner_product(self, x: V, y: V) -> float:
        """``c_x . (g * c_y)``, avoiding a full matrix apply."""
        cx = self.to_components(x)
        cy = self.to_components(y)
        return float(np.dot(cx, self._metric_values * cy))

    def squared_norm(self, x: V) -> float:
        """``c_x . (g * c_x)``, one conversion."""
        c = self.to_components(x)
        return float(np.dot(c, self._metric_values * c))


class ComponentView(ArrayVectorMixin, CoordinateSpace[np.ndarray]):
    """A coordinate space seen through its components.

    The vectors are the component arrays themselves and the metric is the
    viewed space's, so the inner product, norms and every axiom agree with
    the original exactly -- ``(c_x, c_y)`` here *is* ``(x, y)`` there -- while
    ``to_components`` and ``from_components`` are the identity. Arithmetic on
    this space costs no transforms.

    This is what the iterative solvers run on when the operator's space has
    coordinates: the right-hand side is converted in once, the solution out
    once, and the Krylov loop's inner products, norms and updates -- which on
    a sphere were seven transforms per iteration against the operator's two
    -- are array arithmetic. See :class:`~pygeoinf2.numerics.solvers.IterativeSolver`.
    """

    def __init__(self, space: CoordinateSpace) -> None:
        """
        Args:
            space: the coordinate space to view. Kept, and consulted for the
                metric and for white noise.
        """
        self._space = space

    @property
    def space(self) -> CoordinateSpace:
        """The space being viewed."""
        return self._space

    @property
    def dim(self) -> int:
        """The viewed space's dimension."""
        return self._space.dim

    def _key(self) -> Hashable:
        return ("components of", self._space)

    def to_components(self, x: np.ndarray) -> np.ndarray:
        """The array itself: a vector here is its components."""
        return x

    def from_components(self, c: np.ndarray) -> np.ndarray:
        """The array itself."""
        return c

    def apply_gram(self, c: np.ndarray) -> np.ndarray:
        """The viewed space's metric."""
        return self._space.apply_gram(c)

    def solve_gram(self, c: np.ndarray) -> np.ndarray:
        """The viewed space's inverse metric."""
        return self._space.solve_gram(c)

    def apply_gram_to_columns(self, columns: np.ndarray, /) -> np.ndarray:
        """The viewed space's metric on every column."""
        return self._space.apply_gram_to_columns(columns)

    def solve_gram_to_columns(self, columns: np.ndarray, /) -> np.ndarray:
        """The viewed space's inverse metric on every column."""
        return self._space.solve_gram_to_columns(columns)

    def gram_matrix(self) -> np.ndarray:
        """The viewed space's Gram matrix."""
        return self._space.gram_matrix()

    def gram_diagonal(self) -> np.ndarray:
        """The viewed space's Gram diagonal."""
        return self._space.gram_diagonal()

    def white_noise_components(self, *, rng: Generator | None = None) -> np.ndarray:
        """White noise as the viewed space draws it."""
        return self._space.white_noise_components(rng=rng)

    @property
    def is_orthonormal(self) -> bool:
        """Whether the viewed space's basis is orthonormal."""
        return self._space.is_orthonormal

    @property
    def has_diagonal_metric(self) -> bool:
        """Whether the viewed space's metric is diagonal."""
        return self._space.has_diagonal_metric

    def __repr__(self) -> str:
        return f"ComponentView({self._space!r})"


class OrthonormalSpace[V](CoordinateSpace[V], ABC):
    """A coordinate space with an orthonormal basis, so ``G`` is the identity."""

    @property
    def is_orthonormal(self) -> bool:
        """Always true for this class."""
        return True

    @property
    def has_diagonal_metric(self) -> bool:
        """True: the identity is diagonal."""
        return True

    def white_noise_components(self, *, rng: Generator | None = None) -> np.ndarray:
        """Standard normal components, since ``G`` is the identity here."""
        return _resolve_rng(rng).standard_normal(self.dim)

    def gram_matrix(self) -> np.ndarray:
        """The identity matrix."""
        return np.identity(self.dim)

    def apply_gram_to_columns(self, columns: np.ndarray, /) -> np.ndarray:
        """The columns themselves, copied: ``G`` is the identity."""
        return columns.copy()

    def solve_gram_to_columns(self, columns: np.ndarray, /) -> np.ndarray:
        """The columns themselves, copied: ``G`` is the identity."""
        return columns.copy()

    def gram_diagonal(self) -> np.ndarray:
        """All ones: ``G`` is the identity."""
        return np.ones(self.dim)

    def inner_product(self, x: V, y: V) -> float:
        """The plain component dot product."""
        return float(np.dot(self.to_components(x), self.to_components(y)))

    def squared_norm(self, x: V) -> float:
        """The squared component norm, one conversion."""
        c = self.to_components(x)
        return float(np.dot(c, c))


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
        """The dimension of the space."""
        return self._dim

    def _key(self) -> Hashable:
        return self._dim

    def to_components(self, x: np.ndarray) -> np.ndarray:
        """The vector itself: the coordinate map is the identity."""
        return x

    def from_components(self, c: np.ndarray) -> np.ndarray:
        """The array itself, without copying."""
        return c

    def zero(self) -> np.ndarray:
        """A new array of zeros."""
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
        """One."""
        return 1

    def _key(self) -> Hashable:
        return ()

    def zero(self) -> float:
        """Zero."""
        return 0.0

    def copy(self, x: float) -> float:
        """The value itself, floats being immutable."""
        return float(x)

    def axpy(self, a: float, x: float, y: float) -> float:
        """``y + a * x``. Floats are immutable, so this returns a new value."""
        return float(y + a * x)

    def scale_inplace(self, a: float, x: float) -> float:
        """``a * x``, returned rather than mutated."""
        return float(a * x)

    def inner_product(self, x: float, y: float) -> float:
        """The product of two reals."""
        return float(x) * float(y)

    def to_components(self, x: float) -> np.ndarray:
        """The value as a length-one array."""
        return np.array([float(x)])

    def from_components(self, c: np.ndarray) -> float:
        """The single component, as a plain float."""
        return float(np.asarray(c).reshape(-1)[0])

    def __repr__(self) -> str:
        return "Reals()"
