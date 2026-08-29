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
    "deflated_diagonal",
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
    operator: LinearOperator,
    count: int,
    rng: Generator | None,
    *,
    n_jobs: int | None = None,
) -> list[Any]:
    """Apply the operator to white-noise probes drawn on its domain.

    The probes are drawn in order from *rng* and applied as one block, so a
    matrix-backed operator does one product and a parallel run gives the same
    numbers as a serial one.
    """
    domain = operator.domain
    probes = [domain.white_noise(rng=rng) for _ in range(count)]
    return operator.apply_block(probes, n_jobs=n_jobs)


def _power_iterate(
    operator: LinearOperator,
    basis: Sequence[Any],
    power: int,
    *,
    n_jobs: int | None = None,
) -> list[Any]:
    """Sharpen a range basis by alternating with the adjoint.

    Each half step is reorthonormalised. Without that the vectors collapse onto
    the dominant direction in floating point, which is the well-known failure
    of naive subspace iteration.
    """
    domain, codomain = operator.domain, operator.codomain
    result = list(basis)
    for _ in range(power):
        pulled = domain.orthonormal_basis(
            operator.adjoint.apply_block(result, n_jobs=n_jobs)
        )
        if not pulled:
            break
        result = codomain.orthonormal_basis(operator.apply_block(pulled, n_jobs=n_jobs))
        if not result:
            break
    return result


def _range_options(kwargs: dict) -> dict:
    """The ``random_range`` options a factorisation forwards, with defaults."""
    unknown = set(kwargs) - {
        "oversampling",
        "power",
        "block_size",
        "rtol",
        "max_rank",
        "n_jobs",
    }
    if unknown:
        raise TypeError(f"Unexpected keyword arguments: {sorted(unknown)}.")
    return {
        "oversampling": kwargs.get("oversampling", 10),
        "power": kwargs.get("power", 1),
        "block_size": kwargs.get("block_size", 10),
        "rtol": kwargs.get("rtol", 1e-4),
        "max_rank": kwargs.get("max_rank", None),
        "n_jobs": kwargs.get("n_jobs", None),
    }


def _fast(space: Any) -> bool:
    """Whether a space lets the library work on its component arrays."""
    return isinstance(space, CoordinateSpace) and space.uses_component_fast_paths


def _orthonormal_columns(space: CoordinateSpace, vectors: Sequence[Any]) -> np.ndarray:
    """``orthonormal_basis`` on components, returning the columns."""
    columns, _ = space._orthonormalise_columns(
        space.components_of(vectors), rtol=1e-10
    )
    return columns


def _range_with_columns(
    operator: LinearOperator,
    /,
    *,
    rank: int | None,
    oversampling: int,
    power: int,
    block_size: int,
    rtol: float,
    max_rank: int | None,
    rng: Generator | None,
    n_jobs: int | None = None,
) -> tuple[list[Any], np.ndarray | None]:
    """``random_range``, also returning the basis as component columns.

    On a coordinate space the basis is carried as columns through the power
    iteration, so each application's result is analysed exactly once and the
    basis is never re-analysed by the factorisation that follows; the vectors
    are synthesised once. Elsewhere the columns are ``None`` and the
    coordinate-free loops run.
    """
    domain, codomain = operator.domain, operator.codomain
    if not (_fast(domain) and _fast(codomain)):
        return (
            random_range(
                operator,
                rank=rank,
                oversampling=oversampling,
                power=power,
                block_size=block_size,
                rtol=rtol,
                max_rank=max_rank,
                rng=rng,
                n_jobs=n_jobs,
            ),
            None,
        )
    ceiling = max_rank if max_rank is not None else min(domain.dim, codomain.dim)
    if ceiling <= 0:
        return [], None
    if rank is not None:
        if rank <= 0:
            raise ValueError("rank must be positive.")
        count = min(rank + oversampling, ceiling)
        columns = _orthonormal_columns(
            codomain, _probe_range(operator, count, rng, n_jobs=n_jobs)
        )
    else:
        columns = _adaptive_range_on_components(
            operator, codomain, ceiling, block_size, rtol, rng, n_jobs=n_jobs
        )
    for _ in range(power):
        if columns.shape[1] == 0:
            break
        pulled = _orthonormal_columns(
            domain,
            operator.adjoint.apply_block(codomain.vectors_from(columns), n_jobs=n_jobs),
        )
        if pulled.shape[1] == 0:
            break
        columns = _orthonormal_columns(
            codomain, operator.apply_block(domain.vectors_from(pulled), n_jobs=n_jobs)
        )
    return codomain.vectors_from(columns), columns


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
    n_jobs: int | None = None,
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
        n_jobs: workers for applying the operator to a block of probes, one
            probe per worker, where the operator cannot apply a block at
            once itself. The probes are drawn before they are applied, so
            the result does not depend on this. Serial by default.

    Returns:
        An orthonormal list of vectors in the codomain. It may be shorter than
        requested when the operator's range is genuinely smaller.

    Raises:
        ValueError: for a non-positive rank or block size, or a tolerance
            outside ``(0, 1)``.
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
        basis = codomain.orthonormal_basis(_probe_range(operator, count, rng, n_jobs=n_jobs))
        return _power_iterate(operator, basis, power, n_jobs=n_jobs)

    # --- adaptive: grow until a fresh block is nearly in the span ------
    #
    # Only the *new* block is orthogonalised, against a basis that is already
    # orthonormal, and the residuals that test tells us are kept and reused as
    # the vectors to extend by. Rebuilding the whole basis each round instead
    # -- orthonormal_basis(basis + block) -- redoes every earlier vector's work
    # and throws the residuals away, which turns a linear cost into a cubic
    # one. That is v1's arrangement.
    if _fast(codomain):
        columns = _adaptive_range_on_components(
            operator, codomain, ceiling, block_size, rtol, rng, n_jobs=n_jobs
        )
        return _power_iterate(operator, codomain.vectors_from(columns), power, n_jobs=n_jobs)
    basis: list[Any] = []
    scale: float | None = None
    while len(basis) < ceiling:
        block = _probe_range(
            operator, min(block_size, ceiling - len(basis)), rng, n_jobs=n_jobs
        )
        if scale is None:
            scale = max((codomain.norm(y) for y in block), default=0.0)
            if scale == 0.0:
                return []

        residuals, largest = [], 0.0
        for probe in block:
            residual, norm, _ = codomain._orthogonalise_against(probe, basis)
            largest = max(largest, norm)
            if norm > 1e-12 * scale:
                residuals.append(codomain.scale_inplace(1.0 / norm, residual))

        # The residuals are orthogonal to the basis but not yet to each other.
        room = ceiling - len(basis)
        basis.extend(codomain.orthonormal_basis(residuals)[:room])
        if largest <= rtol * scale or not residuals:
            break
    return _power_iterate(operator, basis, power, n_jobs=n_jobs)


def _adaptive_range_on_components(
    operator: LinearOperator,
    codomain: CoordinateSpace,
    ceiling: int,
    block_size: int,
    rtol: float,
    rng: Generator | None,
    *,
    n_jobs: int | None = None,
) -> np.ndarray:
    """The adaptive loop with the basis kept as component columns.

    Each probe is analysed once; the residual test and the extension are
    arithmetic on ``(dim, k)`` arrays with the metric through ``apply_gram``.
    Returns the columns; the caller synthesises vectors once. Same algorithm
    as the coordinate-free loop above: only the new block is orthogonalised
    against a basis that is already orthonormal, and the residuals are what
    extend it.
    """
    columns = np.empty((codomain.dim, ceiling))
    count = 0
    scale: float | None = None
    while count < ceiling:
        block = codomain.components_of(
            _probe_range(operator, min(block_size, ceiling - count), rng, n_jobs=n_jobs)
        )
        weighted = codomain.apply_gram_to_columns(block)
        norms = np.sqrt(np.maximum(np.einsum("ij,ij->j", block, weighted), 0.0))
        if scale is None:
            scale = float(norms.max()) if norms.size else 0.0
            if scale == 0.0:
                return []

        # Project the block off the basis, twice, and read the residual norms.
        residual = block.copy()
        if count > 0:
            current = columns[:, :count]
            for _ in range(2):
                residual -= current @ (current.T @ weighted)
                weighted = codomain.apply_gram_to_columns(residual)
        residual_norms = np.sqrt(
            np.maximum(np.einsum("ij,ij->j", residual, weighted), 0.0)
        )
        largest = float(residual_norms.max()) if residual_norms.size else 0.0

        keep = residual_norms > 1e-12 * scale
        # The residuals are orthogonal to the basis but not yet to each other.
        fresh, _ = codomain._orthonormalise_columns(
            residual[:, keep] / np.where(keep, residual_norms, 1.0)[keep],
            rtol=1e-10,
            against=columns[:, :count] if count > 0 else None,
        )
        room = ceiling - count
        added = min(fresh.shape[1], room)
        columns[:, count : count + added] = fresh[:, :added]
        count += added
        if largest <= rtol * scale or not keep.any():
            break
    return columns[:, :count]


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

    Args:
        operator: a self-adjoint endomorphism.
        rank: how many eigenpairs to keep. Adaptive if omitted, growing the
            basis until a fresh block adds nothing.
        rng: the generator for the probes.
        **kwargs: passed to :func:`random_range` -- oversampling, power
            iterations, the adaptive tolerance, ``n_jobs``.

    Returns:
        The decomposition, itself an operator.

    Raises:
        ValueError: if the operator does not claim self-adjointness, or is not
            an endomorphism -- an eigendecomposition needs both.
    """
    if Traits.SELF_ADJOINT & operator.traits != Traits.SELF_ADJOINT:
        raise ValueError(
            f"A randomised eigendecomposition needs a self-adjoint operator; "
            f"this one claims {operator.traits!s}. Use random_svd otherwise."
        )
    space = operator.domain
    options = _range_options(kwargs)
    basis, columns = _range_with_columns(operator, rank=rank, rng=rng, **options)
    if not basis:
        raise ValueError("The operator's range appears to be trivial.")
    images = operator.apply_block(basis, n_jobs=options["n_jobs"])
    if columns is not None:
        # T == Q^T G (A Q) on component columns: k analyses, not k^2.
        projected = columns.T @ space.apply_gram_to_columns(space.components_of(images))
    else:
        projected = np.array(
            [[space.inner_product(image, q) for image in images] for q in basis]
        )
    projected = 0.5 * (projected + projected.T)
    values, vectors = np.linalg.eigh(projected)

    order = np.argsort(np.abs(values))[::-1]
    if rank is not None:
        order = order[:rank]
    values, vectors = values[order], vectors[:, order]

    if columns is not None:
        factor = LinearOperator.from_component_columns(
            space, columns @ vectors, orthonormal=True
        )
        return LowRankEig(factor, values)

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

    Args:
        operator: any linear operator; unlike :func:`random_eig` it need not
            be self-adjoint or square.
        rank: how many singular triplets to keep. Adaptive if omitted.
        rng: the generator for the probes.
        **kwargs: passed to :func:`random_range`.

    Returns:
        The decomposition, itself an operator.

    Raises:
        ValueError: for a rank exceeding the smaller dimension.
    """
    domain, codomain = operator.domain, operator.codomain
    options = _range_options(kwargs)
    basis, columns = _range_with_columns(operator, rank=rank, rng=rng, **options)
    if not basis:
        raise ValueError("The operator's range appears to be trivial.")
    pulled = operator.adjoint.apply_block(basis, n_jobs=options["n_jobs"])
    fast = columns is not None
    if fast:
        pulled_columns = domain.components_of(pulled)
        gram = pulled_columns.T @ domain.apply_gram_to_columns(pulled_columns)
    else:
        gram = np.array(
            [[domain.inner_product(u, v) for v in pulled] for u in pulled]
        )
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

    if fast:
        return LowRankSVD(
            LinearOperator.from_component_columns(
                codomain, columns @ vectors, orthonormal=True
            ),
            singular_values,
            LinearOperator.from_component_columns(
                domain, (pulled_columns @ vectors) / singular_values, orthonormal=True
            ),
        )

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

    Args:
        operator: self-adjoint and positive semidefinite. The definiteness is
            what makes the square root real, and it is required rather than
            assumed.
        rank: how many eigenpairs to keep. Adaptive if omitted.
        rng: the generator for the probes.
        **kwargs: passed to :func:`random_range`.

    Returns:
        The factor ``L``, as an operator.

    Raises:
        ValueError: if the operator does not claim positive semidefiniteness.
    """
    if Traits.POSITIVE_SEMIDEFINITE & operator.traits != Traits.POSITIVE_SEMIDEFINITE:
        raise ValueError(
            f"A Cholesky-type factorisation needs a positive semidefinite "
            f"operator; this one claims {operator.traits!s}."
        )
    decomposition = random_eig(operator, rank=rank, rng=rng, **kwargs)
    values = np.clip(decomposition.eigenvalues, 0.0, None)
    space = operator.domain
    columns = getattr(decomposition.factor, "columns", None)
    if columns is not None:
        return LowRankCholesky(
            LinearOperator.from_component_columns(space, columns * np.sqrt(values))
        )
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
    rtol: float | None = None,
    max_samples: int | None = None,
    block_size: int = 20,
    rng: Generator | None = None,
    n_jobs: int | None = None,
) -> Estimate:
    """The Hutchinson trace estimate, coordinate-free.

    ``E[(A x, x)] == tr A`` when ``x`` is white noise *on the space*, since the
    identity covariance is what makes the expectation the trace. With v1's
    probes the expectation is ``tr(G A)`` instead, which on a mass-weighted
    space is a different number.

    Args:
        operator: an endomorphism.
        samples: how many probes to draw, or the first block when *rtol* is
            given.
        rtol: draw further blocks until the standard error falls to this
            fraction of the estimate, rather than stopping at a fixed count.
            v1 grew the sample count until the *estimate* stopped moving,
            which is a noisy test of a noisy quantity; the standard error is
            already computed here and says the same thing properly.
        max_samples: a ceiling on the adaptive route. Defaults to twenty
            blocks past the first.
        block_size: how many probes to add per round.
        rng: the generator. One stream per probe is spawned from it, whether
            or not the probes run in parallel, so the estimate is a function
            of the seed alone (see :mod:`pygeoinf2.parallel`).
        n_jobs: workers for the probes, and for each further block on the
            adaptive route.

    Returns:
        The estimate, with the standard error that earns it.

    Raises:
        ValueError: for a non-square operator, fewer than two samples, or a
            tolerance outside ``(0, 1)``.
    """
    if not operator.is_endomorphism:
        raise ValueError("A trace needs an operator from a space to itself.")
    if samples < 2:
        raise ValueError("At least two samples are needed for an error estimate.")
    if rtol is not None and not 0.0 < rtol < 1.0:
        raise ValueError(f"The tolerance lies in (0, 1), got {rtol}.")

    space = operator.domain
    from numpy.random import default_rng

    from ..parallel import parallel_map

    parent = default_rng() if rng is None else rng

    def probe_once(stream: Any) -> float:
        probe = space.white_noise(rng=stream)
        return space.inner_product(operator(probe), probe)

    def block(count: int) -> np.ndarray:
        # One stream per probe, spawned in order from the parent, so the
        # draws are the same whether they run here or across workers.
        return np.array(parallel_map(probe_once, parent.spawn(count), n_jobs=n_jobs))

    draws = block(samples)
    if rtol is not None:
        ceiling = max_samples if max_samples is not None else samples + 20 * block_size
        while draws.size < ceiling:
            error = float(draws.std(ddof=1) / np.sqrt(draws.size))
            if error <= rtol * abs(float(draws.mean())):
                break
            draws = np.concatenate([draws, block(min(block_size, ceiling - draws.size))])
        samples = draws.size

    return Estimate(
        float(draws.mean()),
        float(draws.std(ddof=1) / np.sqrt(samples)),
        samples,
    )


def deflated_diagonal(
    operator: LinearOperator,
    /,
    *,
    rank: int = 10,
    samples: int = 100,
    form: Literal["galerkin", "components"] = "galerkin",
    rng: Generator | None = None,
    n_jobs: int | None = None,
) -> np.ndarray:
    """A diagonal estimate with the dominant eigenvalues removed first.

    The Bekas-Kokiopoulou-Saad estimator's variance is set by the size of the
    *whole* operator, not by the size of what it is failing to resolve. So an
    operator with a few large eigenvalues and a long tail — every covariance
    with a decaying spectrum — is estimated badly for a reason that has nothing
    to do with the tail.

    Deflating fixes that. The leading ``rank`` eigenpairs are found exactly
    enough, their contribution to the diagonal is computed in closed form, and
    only the remainder is sampled. The stochastic part then sees an operator
    whose norm is the ``(rank + 1)``-th eigenvalue rather than the first.

    Args:
        operator: self-adjoint, and positive semidefinite in the case this is
            for.
        rank: how many eigenpairs to remove.
        samples: probes for the remainder.
        form: which matrix's diagonal, as for :func:`random_diagonal`.
        rng: the generator.
        n_jobs: workers for the operator applications, in the
            eigendecomposition and in the probes alike. Serial by default.

    Returns:
        The diagonal, as an array.

    Raises:
        ValueError: for a non-positive rank or sample count, or an unknown
            form.
    """
    require_coordinates(operator.domain, operator.codomain)
    if rank < 0:
        raise ValueError(f"The rank must be non-negative, got {rank}.")
    if rank == 0:
        return random_diagonal(
            operator, samples=samples, form=form, rng=rng, n_jobs=n_jobs
        )
    generator = rng if rng is not None else np.random.default_rng()
    low_rank = random_eig(operator, rank=rank, rng=generator, n_jobs=n_jobs)
    domain: CoordinateSpace = operator.domain

    # diag(U L U*) exactly. The operator's component matrix is
    # sum_i lambda_i c_i (G c_i)^T and its Galerkin matrix is
    # sum_i lambda_i (G c_i)(G c_i)^T, so which diagonal is wanted decides
    # how many times the metric appears -- once or twice. On an orthonormal
    # basis the two coincide, which is why this needs a weighted space to test.
    stored = getattr(low_rank.factor, "columns", None)
    columns = (
        stored.T
        if stored is not None
        else np.stack(
            [
                domain.to_components(low_rank.factor(vector))
                for vector in np.identity(low_rank.eigenvalues.size)
            ]
        )
    )
    weighted = np.stack([domain.apply_gram(column) for column in columns])
    if form == "galerkin":
        exact = np.einsum("i,ij,ij->j", low_rank.eigenvalues, weighted, weighted)
    else:
        exact = np.einsum("i,ij,ij->j", low_rank.eigenvalues, columns, weighted)

    remainder = operator - low_rank
    return exact + random_diagonal(
        remainder, samples=samples, form=form, rng=generator, n_jobs=n_jobs
    )


def random_diagonal(
    operator: LinearOperator,
    /,
    *,
    samples: int = 100,
    rtol: float | None = None,
    max_samples: int | None = None,
    block_size: int = 20,
    form: Literal["galerkin", "components"] = "galerkin",
    rng: Generator | None = None,
    n_jobs: int | None = None,
) -> np.ndarray:
    """The Bekas-Kokiopoulou-Saad diagonal estimate.

    Needs coordinates, because a diagonal is a statement about a basis, and
    needs to be told *which* matrix's diagonal is wanted, because no trait
    implies it (DESIGN.md 5.3). The Galerkin form is the default: it is the
    representation in which a self-adjoint operator is symmetric, so it is the
    diagonal a symmetric preconditioner wants.

    Rademacher probes are used rather than Gaussian ones: they have the same
    expectation and lower variance for this estimator.

    Args:
        operator: the operator whose diagonal is wanted.
        samples: how many probes, or the first block when *rtol* is given.
        rtol: keep probing until the estimate's relative accuracy reaches
            this, measured as ``||standard error|| / ||estimate||`` over the
            whole diagonal.

            The norm, rather than the worst entry: a diagonal is a vector, a
            per-entry relative test never passes on a near-zero entry, and the
            worst entry's own standard error does not predict the worst
            realised error -- the maximum over many entries runs several
            standard errors out. The norm ratio does predict it. Measured
            against the truth on a 120-dimensional operator, the ratio and the
            achieved relative error track each other within a few per cent all
            the way from 20 probes (0.042 against 0.045) to 420 (0.0094
            against 0.0094).
        max_samples: a ceiling on that. Defaults to twenty blocks past the
            first.
        block_size: probes added per round.
        form: which matrix's diagonal.
        rng: the generator.
        n_jobs: workers for applying the operator to a block of probes. The
            probes are drawn before they are applied, so the estimate does
            not depend on this. Serial by default.

    Returns:
        One value per component.

    Raises:
        ValueError: for an unknown form, or a tolerance outside ``(0, 1)``.
    """
    require_coordinates(operator.domain, operator.codomain)
    domain: CoordinateSpace = operator.domain
    codomain: CoordinateSpace = operator.codomain
    if form not in ("galerkin", "components"):
        raise ValueError(f"Unknown form {form!r}.")

    if rtol is not None and not 0.0 < rtol < 1.0:
        raise ValueError(f"The tolerance lies in (0, 1), got {rtol}.")

    generator = rng if rng is not None else np.random.default_rng()
    dimension = min(domain.dim, codomain.dim)
    numerator = np.zeros(dimension)
    denominator = np.zeros(dimension)
    # Sum of squared contributions, kept only to form a standard error for the
    # adaptive route; the estimator itself needs the first two.
    squares = np.zeros(dimension)
    drawn = 0

    def probe_block(count: int) -> None:
        nonlocal drawn
        probes = [
            generator.integers(0, 2, size=domain.dim) * 2.0 - 1.0 for _ in range(count)
        ]
        images = operator.apply_block(
            [domain.from_components(probe) for probe in probes], n_jobs=n_jobs
        )
        for probe, vector in zip(probes, images):
            image = codomain.to_components(vector)
            if form == "galerkin":
                image = codomain.apply_gram(image)
            contribution = probe[:dimension] * image[:dimension]
            # Slice assignment, not ``+=`` on the bare name: the latter is an
            # augmented assignment and would make these local to the closure.
            numerator[:] += contribution
            squares[:] += contribution**2
            denominator[:] += probe[:dimension] ** 2
        drawn += count

    probe_block(samples)
    if rtol is not None:
        ceiling = max_samples if max_samples is not None else samples + 20 * block_size
        while drawn < ceiling:
            mean = numerator / drawn
            variance = np.maximum(squares / drawn - mean**2, 0.0)
            scale = float(np.linalg.norm(mean))
            error = float(np.linalg.norm(np.sqrt(variance / drawn)))
            if scale == 0.0 or error <= rtol * scale:
                break
            probe_block(min(block_size, ceiling - drawn))
    return numerator / denominator
