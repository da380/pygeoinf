"""
Randomised linear algebra: range finding, low-rank factorisation, estimators.

Coordinate-free wherever the mathematics allows. Range finding, the low-rank
factorisations and trace estimation need only the operator's action, its
adjoint, an inner product and ``axpy``. Only the diagonal estimator needs
components, because a diagonal is a statement about a basis.

v1 already had an abstract path here and it was the right idea, but it drew its
probes from ``white_noise_measure``, whose covariance on a mass-weighted space
is the Gram matrix rather than the identity (DESIGN.md section 9). So the
structure was coordinate-free and the *distribution* was not: the probes were
anisotropic in the space's own geometry, exactly where ``random_range``
documents a "geometric safety guard" for. Drawing from
``HilbertSpace.white_noise`` fixes it at the source.

The other change is orthogonalisation. A single Gram-Schmidt pass loses
orthogonality precisely when the new vector nearly lies in the span, which is
the regime a rank-revealing method spends its time in; the basis routines here
use the reorthogonalising ``orthonormal_basis``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import numpy as np
from numpy.random import Generator

from ..algebra.operators import LinearOperator, require_coordinates
from ..algebra.spaces import CoordinateSpace
from ..traits import Traits, close

__all__ = [
    "Estimate",
    "random_range",
    "LowRankEig",
    "LowRankSVD",
    "LowRankCholesky",
    "random_eig",
    "random_svd",
    "random_cholesky",
    "random_trace",
    "random_diagonal",
]


@dataclass(frozen=True)
class Estimate:
    """A stochastic estimate, with the uncertainty that comes with it.

    Returning the standard error alongside the value is not decoration: a
    Hutchinson trace estimate converges as ``1/sqrt(n)``, so a number reported
    without its error is uninterpretable.
    """

    value: float
    standard_error: float
    samples: int

    def __repr__(self) -> str:
        return (
            f"Estimate({self.value:.6g} +/- {self.standard_error:.3g}, "
            f"n={self.samples})"
        )


# --------------------------------------------------------------------- #
#                            Range finding                              #
# --------------------------------------------------------------------- #


def _probe_range(
    operator: LinearOperator, count: int, rng: Generator | None
) -> list[Any]:
    """Apply the operator to white-noise probes drawn on its domain."""
    domain = operator.domain
    return [operator(domain.white_noise(rng=rng)) for _ in range(count)]


def _power_iterate(
    operator: LinearOperator, basis: Sequence[Any], power: int
) -> list[Any]:
    """Sharpen a range basis by alternating with the adjoint.

    Each half step is reorthonormalised. Without that the vectors collapse onto
    the dominant direction in floating point, which is the well-known failure
    of naive subspace iteration.
    """
    domain, codomain = operator.domain, operator.codomain
    result = list(basis)
    for _ in range(power):
        pulled = domain.orthonormal_basis([operator.adjoint(q) for q in result])
        if not pulled:
            break
        result = codomain.orthonormal_basis([operator(z) for z in pulled])
        if not result:
            break
    return result


def random_range(
    operator: LinearOperator,
    /,
    *,
    rank: int | None = None,
    oversampling: int = 10,
    power: int = 1,
    block_size: int = 10,
    rtol: float = 1e-4,
    max_rank: int | None = None,
    rng: Generator | None = None,
) -> list[Any]:
    """An orthonormal basis for an approximate range of the operator.

    Args:
        operator: the operator whose range is wanted.
        rank: the target rank. When given, ``rank + oversampling`` probes are
            drawn in one block. When None, blocks are drawn until the residual
            falls below ``rtol``.
        oversampling: extra probes beyond the target rank, which is what makes
            the randomised bound hold with high probability.
        power: subspace iteration steps. One or two sharpen a slowly decaying
            spectrum considerably; zero is right only when the decay is fast.
        block_size: probes per block in the adaptive mode.
        rtol: relative residual at which the adaptive mode stops.
        max_rank: hard ceiling, defaulting to the smaller dimension.
        rng: generator for the probes.

    Returns:
        An orthonormal list of vectors in the codomain. It may be shorter than
        requested when the operator's range is genuinely smaller.
    """
    codomain = operator.codomain
    ceiling = (
        max_rank if max_rank is not None else min(operator.domain.dim, codomain.dim)
    )
    if ceiling <= 0:
        return []

    if rank is not None:
        if rank <= 0:
            raise ValueError("rank must be positive.")
        count = min(rank + oversampling, ceiling)
        basis = codomain.orthonormal_basis(_probe_range(operator, count, rng))
        return _power_iterate(operator, basis, power)

    # --- adaptive: grow until a fresh block is nearly in the span ------
    basis: list[Any] = []
    scale: float | None = None
    while len(basis) < ceiling:
        block = _probe_range(operator, min(block_size, ceiling - len(basis)), rng)
        if scale is None:
            scale = max((codomain.norm(y) for y in block), default=0.0)
            if scale == 0.0:
                return []
        residual = max(
            codomain.norm(codomain._orthogonalise_against(y, basis)[0]) for y in block
        )
        basis = codomain.orthonormal_basis(basis + block)
        if residual <= rtol * scale:
            break
    return _power_iterate(operator, basis, power)


# --------------------------------------------------------------------- #
#                        Low-rank representations                       #
# --------------------------------------------------------------------- #


class LowRankEig(LinearOperator):
    """``A ~ U D U*`` for a self-adjoint operator, with ``U`` an isometry.

    Self-adjoint by construction. Positive semidefinite when the eigenvalues
    are — which the palindrome rule would deduce anyway from ``U D U*``, but is
    cheaper to state directly here.
    """

    def __init__(self, factor: LinearOperator, eigenvalues: np.ndarray, /) -> None:
        """
        Args:
            factor: an isometry from ``R^k`` into the space, whose columns are
                the approximate eigenvectors.
            eigenvalues: the ``k`` approximate eigenvalues.
        """
        values = np.asarray(eigenvalues, dtype=float)
        traits = Traits.SELF_ADJOINT
        if np.all(values >= 0.0):
            traits |= Traits.POSITIVE_SEMIDEFINITE
        super().__init__(factor.codomain, factor.codomain, traits=close(traits))
        self._factor = factor
        self._eigenvalues = values

    @property
    def factor(self) -> LinearOperator:
        """The isometry ``U``."""
        return self._factor

    @property
    def eigenvalues(self) -> np.ndarray:
        """The approximate eigenvalues, largest first."""
        return self._eigenvalues

    @property
    def rank(self) -> int:
        """The number of retained directions."""
        return self._eigenvalues.size

    @property
    def trace(self) -> float:
        """The trace of the approximation, which is the sum of its eigenvalues."""
        return float(np.sum(self._eigenvalues))

    def _value(self, x: Any) -> Any:
        return self._factor(self._eigenvalues * self._factor.adjoint(x))

    def _adjoint_value(self, y: Any) -> Any:
        return self._value(y)

    def apply_function(
        self, function: Callable[[np.ndarray], np.ndarray], /
    ) -> LowRankEig:
        """``f(A)`` restricted to the retained subspace.

        Exact on the span of ``U`` and zero off it, so this is ``f`` applied to
        the truncated spectrum rather than to the operator. For ``f(0) != 0``
        the two differ substantially, and the caller must decide whether that
        matters.
        """
        return LowRankEig(self._factor, np.asarray(function(self._eigenvalues)))

    def __repr__(self) -> str:
        return f"LowRankEig(rank={self.rank})"


class LowRankSVD(LinearOperator):
    """``A ~ U S V*``, with ``U`` and ``V`` isometries."""

    def __init__(
        self,
        left: LinearOperator,
        singular_values: np.ndarray,
        right: LinearOperator,
        /,
    ) -> None:
        """
        Args:
            left: an isometry from ``R^k`` into the codomain.
            singular_values: the ``k`` approximate singular values.
            right: an isometry from ``R^k`` into the domain.
        """
        super().__init__(right.codomain, left.codomain)
        self._left = left
        self._right = right
        self._singular_values = np.asarray(singular_values, dtype=float)

    @property
    def left_factor(self) -> LinearOperator:
        """The isometry ``U`` into the codomain."""
        return self._left

    @property
    def right_factor(self) -> LinearOperator:
        """The isometry ``V`` into the domain."""
        return self._right

    @property
    def singular_values(self) -> np.ndarray:
        """The approximate singular values, largest first."""
        return self._singular_values

    @property
    def rank(self) -> int:
        """The number of retained directions."""
        return self._singular_values.size

    def _value(self, x: Any) -> Any:
        return self._left(self._singular_values * self._right.adjoint(x))

    def _adjoint_value(self, y: Any) -> Any:
        return self._right(self._singular_values * self._left.adjoint(y))

    def __repr__(self) -> str:
        return f"LowRankSVD(rank={self.rank})"


class LowRankCholesky(LinearOperator):
    """``A ~ L L*``, positive semidefinite by construction."""

    def __init__(self, factor: LinearOperator, /) -> None:
        """
        Args:
            factor: the operator ``L``, from a coefficient space into the space.
        """
        super().__init__(
            factor.codomain,
            factor.codomain,
            traits=close(Traits.SELF_ADJOINT | Traits.POSITIVE_SEMIDEFINITE),
        )
        self._factor = factor

    @property
    def factor(self) -> LinearOperator:
        """The factor ``L``, which is also a covariance factor for sampling."""
        return self._factor

    @property
    def rank(self) -> int:
        """The number of columns of the factor."""
        return self._factor.domain.dim

    def _value(self, x: Any) -> Any:
        return self._factor(self._factor.adjoint(x))

    def _adjoint_value(self, y: Any) -> Any:
        return self._value(y)

    def __repr__(self) -> str:
        return f"LowRankCholesky(rank={self.rank})"


# --------------------------------------------------------------------- #
#                          Factorisations                               #
# --------------------------------------------------------------------- #


def random_eig(
    operator: LinearOperator,
    /,
    *,
    rank: int | None = None,
    rng: Generator | None = None,
    **kwargs: Any,
) -> LowRankEig:
    """A randomised eigendecomposition of a self-adjoint operator.

    Builds a range basis ``Q``, forms the small matrix ``T`` with
    ``T_ij == (A q_i, q_j)`` — a ``k x k`` array assembled from inner products
    alone — and diagonalises it. The eigenvectors come back as ``Q S``.
    """
    if Traits.SELF_ADJOINT & operator.traits != Traits.SELF_ADJOINT:
        raise ValueError(
            f"A randomised eigendecomposition needs a self-adjoint operator; "
            f"this one claims {operator.traits!s}. Use random_svd otherwise."
        )
    space = operator.domain
    basis = random_range(operator, rank=rank, rng=rng, **kwargs)
    if not basis:
        raise ValueError("The operator's range appears to be trivial.")

    images = [operator(q) for q in basis]
    projected = np.array(
        [[space.inner_product(image, q) for image in images] for q in basis]
    )
    projected = 0.5 * (projected + projected.T)
    values, vectors = np.linalg.eigh(projected)

    order = np.argsort(np.abs(values))[::-1]
    if rank is not None:
        order = order[:rank]
    values, vectors = values[order], vectors[:, order]

    eigenvectors = [
        space.mean([])  # placeholder, replaced below
        for _ in range(0)
    ]
    eigenvectors = []
    for column in vectors.T:
        vector = space.zero()
        for weight, q in zip(column, basis):
            vector = space.axpy(float(weight), q, vector)
        eigenvectors.append(vector)

    factor = LinearOperator.from_vectors(space, eigenvectors, orthonormal=True)
    return LowRankEig(factor, values)


def random_svd(
    operator: LinearOperator,
    /,
    *,
    rank: int | None = None,
    rng: Generator | None = None,
    **kwargs: Any,
) -> LowRankSVD:
    """A randomised singular value decomposition.

    With ``Q`` a range basis, ``A ~ Q Q* A``, so the singular values of ``A``
    are those of ``B == Q* A``. Rather than forming ``B``, this assembles the
    ``k x k`` Gram matrix ``C_ij == (A* q_i, A* q_j)``, whose eigendecomposition
    ``C == S L S^T`` gives ``sigma == sqrt(L)``, left vectors ``Q S`` and right
    vectors ``(A* Q) S / sigma``. Every step is an inner product, so no
    component map is used.
    """
    domain, codomain = operator.domain, operator.codomain
    basis = random_range(operator, rank=rank, rng=rng, **kwargs)
    if not basis:
        raise ValueError("The operator's range appears to be trivial.")

    pulled = [operator.adjoint(q) for q in basis]
    gram = np.array([[domain.inner_product(u, v) for v in pulled] for u in pulled])
    gram = 0.5 * (gram + gram.T)
    values, vectors = np.linalg.eigh(gram)

    order = np.argsort(values)[::-1]
    if rank is not None:
        order = order[:rank]
    values, vectors = np.clip(values[order], 0.0, None), vectors[:, order]
    singular_values = np.sqrt(values)

    keep = singular_values > singular_values.max() * 1e-14 if values.size else []
    singular_values = singular_values[keep]
    vectors = vectors[:, keep]

    left, right = [], []
    for index, column in enumerate(vectors.T):
        u = codomain.zero()
        v = domain.zero()
        for weight, q, w in zip(column, basis, pulled):
            u = codomain.axpy(float(weight), q, u)
            v = domain.axpy(float(weight), w, v)
        left.append(u)
        right.append(domain.scale_inplace(1.0 / singular_values[index], v))

    return LowRankSVD(
        LinearOperator.from_vectors(codomain, left, orthonormal=True),
        singular_values,
        LinearOperator.from_vectors(domain, right, orthonormal=True),
    )


def random_cholesky(
    operator: LinearOperator,
    /,
    *,
    rank: int | None = None,
    rng: Generator | None = None,
    **kwargs: Any,
) -> LowRankCholesky:
    """A randomised factorisation ``A ~ L L*`` of a positive semidefinite operator.

    Obtained from :func:`random_eig` by folding the square root of the
    eigenvalues into the factor, so ``L == U D^(1/2)``. The result is directly
    usable as a covariance factor, which is how a low-rank Gaussian gets
    sampled.
    """
    if Traits.POSITIVE_SEMIDEFINITE & operator.traits != Traits.POSITIVE_SEMIDEFINITE:
        raise ValueError(
            f"A Cholesky-type factorisation needs a positive semidefinite "
            f"operator; this one claims {operator.traits!s}."
        )
    decomposition = random_eig(operator, rank=rank, rng=rng, **kwargs)
    values = np.clip(decomposition.eigenvalues, 0.0, None)
    space = operator.domain
    coefficients = decomposition.factor.domain
    scaled = [
        space.scale(
            float(np.sqrt(value)), decomposition.factor(coefficients.basis_vector(i))
        )
        for i, value in enumerate(values)
    ]
    return LowRankCholesky(LinearOperator.from_vectors(space, scaled))


# --------------------------------------------------------------------- #
#                              Estimators                               #
# --------------------------------------------------------------------- #


def random_trace(
    operator: LinearOperator,
    /,
    *,
    samples: int = 100,
    rng: Generator | None = None,
) -> Estimate:
    """The Hutchinson trace estimate, coordinate-free.

    ``E[(A x, x)] == tr A`` when ``x`` is white noise *on the space*, since the
    identity covariance is what makes the expectation the trace. With v1's
    probes the expectation is ``tr(G A)`` instead, which on a mass-weighted
    space is a different number.
    """
    if not operator.is_endomorphism:
        raise ValueError("A trace needs an operator from a space to itself.")
    if samples < 2:
        raise ValueError("At least two samples are needed for an error estimate.")

    space = operator.domain
    draws = np.empty(samples)
    for index in range(samples):
        probe = space.white_noise(rng=rng)
        draws[index] = space.inner_product(operator(probe), probe)
    return Estimate(
        float(draws.mean()),
        float(draws.std(ddof=1) / np.sqrt(samples)),
        samples,
    )


def random_diagonal(
    operator: LinearOperator,
    /,
    *,
    samples: int = 100,
    form: Literal["galerkin", "components"] = "galerkin",
    rng: Generator | None = None,
) -> np.ndarray:
    """The Bekas-Kokiopoulou-Saad diagonal estimate.

    Needs coordinates, because a diagonal is a statement about a basis, and
    needs to be told *which* matrix's diagonal is wanted, because no trait
    implies it (DESIGN.md 5.3). The Galerkin form is the default: it is the
    representation in which a self-adjoint operator is symmetric, so it is the
    diagonal a symmetric preconditioner wants.

    Rademacher probes are used rather than Gaussian ones: they have the same
    expectation and lower variance for this estimator.
    """
    require_coordinates(operator.domain, operator.codomain)
    domain: CoordinateSpace = operator.domain
    codomain: CoordinateSpace = operator.codomain
    if form not in ("galerkin", "components"):
        raise ValueError(f"Unknown form {form!r}.")

    generator = rng if rng is not None else np.random.default_rng()
    dimension = min(domain.dim, codomain.dim)
    numerator = np.zeros(dimension)
    denominator = np.zeros(dimension)

    for _ in range(samples):
        probe = generator.integers(0, 2, size=domain.dim) * 2.0 - 1.0
        image = codomain.to_components(operator(domain.from_components(probe)))
        if form == "galerkin":
            image = codomain.apply_gram(image)
        numerator += probe[:dimension] * image[:dimension]
        denominator += probe[:dimension] ** 2
    return numerator / denominator
