"""
Functional calculus: ``f(A)`` for a self-adjoint operator.

Entirely coordinate-free. Lanczos tridiagonalisation needs only the operator's
action, an inner product and ``axpy``, so ``f(A)`` is available on a space with
no component map — which matters, because a covariance square root is how a
Gaussian gets sampled and a log-determinant is how evidence gets computed.

Two paths, chosen by structure rather than by hand:

- an operator **diagonal in a basis** evaluates ``f`` on its eigenvalues, which
  is exact and costs one array operation;
- anything else self-adjoint goes to **Lanczos**, which approximates ``f(A) x``
  from a small Krylov space without ever forming a matrix.

That split is the specialisation protocol of DESIGN.md 5.4 applied to a unary
operation, and it is why "diagonal in a basis" is a class (it carries
eigenvalues) while "self-adjoint" is a trait (it carries nothing).

Traits gate the operations that need them: a square root requires positive
semidefiniteness, a logarithm requires positive definiteness. Since the library
cannot inspect an arbitrary ``f``, the named operations declare what they need
and the general entry point takes the caller's word.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Sequence

import numpy as np
from scipy.linalg import eigh_tridiagonal

from ..algebra.diagonal import DiagonalLinearOperator
from ..algebra.operators import LinearOperator
from ..algebra.spaces import HilbertSpace
from ..traits import Traits, close

__all__ = [
    "lanczos_tridiagonalise",
    "iter_lanczos_tridiagonalise",
    "apply_operator_function",
    "operator_quadratic_form",
    "OperatorFunction",
    "operator_function",
    "operator_sqrt",
    "operator_inverse_sqrt",
    "operator_exp",
    "operator_log",
    "operator_power",
]


def _require_self_adjoint(operator: LinearOperator, what: str) -> None:
    """Raise unless the operator claims self-adjointness."""
    if Traits.SELF_ADJOINT & operator.traits != Traits.SELF_ADJOINT:
        raise ValueError(
            f"{what} needs a self-adjoint operator; this one claims "
            f"{operator.traits!s}. Traits are claims: attach them with "
            f"with_traits() and verify them with testing.check_traits()."
        )


# --------------------------------------------------------------------- #
#                              Lanczos                                  #
# --------------------------------------------------------------------- #


def iter_lanczos_tridiagonalise(
    operator: LinearOperator,
    start: Any,
    max_iterations: int,
    /,
    *,
    reorthogonalise: bool = True,
    breakdown_tol: float = 1e-12,
) -> Iterator[tuple[list[Any], np.ndarray]]:
    """Yield the Krylov basis and tridiagonal matrix after each iteration.

    Lazily, so a caller can stop on its own convergence test without deciding
    the iteration count in advance.

    Args:
        operator: a self-adjoint operator.
        start: the vector generating the Krylov space. Need not be normalised.
        max_iterations: the most Lanczos steps to take.
        reorthogonalise: re-project each new vector against the whole basis.
            Lanczos loses orthogonality in floating point after very few steps,
            and the cheap three-term recurrence alone is not usable for
            spectral work; full reorthogonalisation costs ``O(k)`` inner
            products per step and is worth it at the small ``k`` used here.
        breakdown_tol: relative norm below which the Krylov space is taken to
            have closed, which is exact termination rather than failure.

    Yields:
        ``(basis, T)`` after each step, with ``T`` symmetric tridiagonal of
        size equal to the basis.
    """
    space: HilbertSpace = operator.domain
    _require_self_adjoint(operator, "Lanczos tridiagonalisation")

    norm = space.norm(start)
    if norm == 0.0:
        raise ValueError("The starting vector for Lanczos must be nonzero.")

    basis = [space.scale(1.0 / norm, start)]
    diagonal: list[float] = []
    off_diagonal: list[float] = []

    previous_beta = 0.0
    for step in range(max_iterations):
        w = operator(basis[-1])
        alpha = space.inner_product(w, basis[-1])
        diagonal.append(alpha)

        w = space.axpy(-alpha, basis[-1], w)
        if step > 0:
            w = space.axpy(-previous_beta, basis[-2], w)
        if reorthogonalise:
            for q in basis:
                w = space.axpy(-space.inner_product(w, q), q, w)

        yield list(basis), _tridiagonal(diagonal, off_diagonal)

        beta = space.norm(w)
        if beta <= breakdown_tol * max(abs(alpha), 1.0) or step + 1 == max_iterations:
            return
        off_diagonal.append(beta)
        previous_beta = beta
        basis.append(space.scale_inplace(1.0 / beta, w))


def _tridiagonal(
    diagonal: Sequence[float], off_diagonal: Sequence[float]
) -> np.ndarray:
    """Assemble the symmetric tridiagonal matrix from its two bands."""
    k = len(diagonal)
    matrix = np.zeros((k, k))
    matrix[np.arange(k), np.arange(k)] = diagonal
    if k > 1:
        band = np.asarray(off_diagonal[: k - 1])
        matrix[np.arange(k - 1), np.arange(1, k)] = band
        matrix[np.arange(1, k), np.arange(k - 1)] = band
    return matrix


def lanczos_tridiagonalise(
    operator: LinearOperator,
    start: Any,
    max_iterations: int,
    /,
    *,
    reorthogonalise: bool = True,
) -> tuple[list[Any], np.ndarray]:
    """Run Lanczos to completion, returning the final basis and matrix."""
    basis, matrix = [], np.zeros((0, 0))
    for basis, matrix in iter_lanczos_tridiagonalise(
        operator, start, max_iterations, reorthogonalise=reorthogonalise
    ):
        pass
    return basis, matrix


# --------------------------------------------------------------------- #
#                         Applying a function                           #
# --------------------------------------------------------------------- #


def apply_operator_function(
    operator: LinearOperator,
    function: Callable[[np.ndarray], np.ndarray],
    x: Any,
    /,
    *,
    max_iterations: int = 50,
    rtol: float = 1e-10,
    reorthogonalise: bool = True,
) -> Any:
    """``f(A) x``, without forming ``f(A)``.

    Lanczos builds an orthonormal basis ``Q`` of the Krylov space with
    ``Q* A Q == T``, so ``f(A) x`` is approximated by
    ``||x|| Q f(T) e_1`` — a small dense eigendecomposition of ``T`` and a
    linear combination of the basis vectors.

    Convergence is judged on the change in the answer between successive
    Krylov dimensions, which is the honest test when nothing is known about
    ``f``.
    """
    space: HilbertSpace = operator.domain
    _require_self_adjoint(operator, "Applying an operator function")

    norm = space.norm(x)
    if norm == 0.0:
        return space.zero()

    previous: Any | None = None
    result: Any = space.zero()
    for basis, matrix in iter_lanczos_tridiagonalise(
        operator, x, max_iterations, reorthogonalise=reorthogonalise
    ):
        result = _combine(space, basis, matrix, function, norm)
        if previous is not None:
            change = space.norm(space.subtract(result, previous))
            if change <= rtol * max(space.norm(result), 1e-300):
                return result
        previous = result
    return result


def _combine(
    space: HilbertSpace,
    basis: Sequence[Any],
    matrix: np.ndarray,
    function: Callable[[np.ndarray], np.ndarray],
    norm: float,
) -> Any:
    """Form ``||x|| Q f(T) e_1`` from the Krylov basis and tridiagonal matrix."""
    values, vectors = _eigh_tridiagonal(matrix)
    weights = vectors @ (np.asarray(function(values)) * vectors[0, :])
    result = space.zero()
    for weight, q in zip(weights, basis):
        result = space.axpy(norm * float(weight), q, result)
    return result


def _eigh_tridiagonal(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose a small symmetric tridiagonal matrix."""
    k = matrix.shape[0]
    if k == 1:
        return matrix[0].copy(), np.ones((1, 1))
    return eigh_tridiagonal(np.diag(matrix), np.diag(matrix, 1))


def operator_quadratic_form(
    operator: LinearOperator,
    function: Callable[[np.ndarray], np.ndarray],
    x: Any,
    /,
    *,
    max_iterations: int = 30,
    reorthogonalise: bool = True,
) -> float:
    """``(x, f(A) x)``, by Gauss quadrature on the Lanczos spectrum.

    Cheaper than forming ``f(A) x`` and pairing it, because only the first row
    of the eigenvector matrix is needed. This is the kernel of stochastic
    Lanczos quadrature, which is how a log-determinant gets estimated without
    a factorisation.
    """
    space: HilbertSpace = operator.domain
    _require_self_adjoint(operator, "An operator quadratic form")

    squared_norm = space.squared_norm(x)
    if squared_norm == 0.0:
        return 0.0

    estimate = 0.0
    for _, matrix in iter_lanczos_tridiagonalise(
        operator, x, max_iterations, reorthogonalise=reorthogonalise
    ):
        values, vectors = _eigh_tridiagonal(matrix)
        # The Gauss quadrature weights are the squared first components.
        estimate = float(np.dot(vectors[0, :] ** 2, np.asarray(function(values))))
    return squared_norm * estimate


# --------------------------------------------------------------------- #
#                        f(A) as an operator                            #
# --------------------------------------------------------------------- #


class OperatorFunction(LinearOperator):
    """``f(A)`` as an operator, applied lazily by Lanczos.

    Nothing is precomputed: each application runs its own Krylov iteration.
    That is the right trade when ``f(A)`` is applied a few times to different
    vectors, and the wrong one when it is applied many times — in which case a
    low-rank factorisation of ``A`` is the better tool.
    """

    def __init__(
        self,
        operator: LinearOperator,
        function: Callable[[np.ndarray], np.ndarray],
        /,
        *,
        traits: Traits = Traits.NONE,
        max_iterations: int = 50,
        rtol: float = 1e-10,
        reorthogonalise: bool = True,
    ) -> None:
        _require_self_adjoint(operator, "An operator function")
        super().__init__(
            operator.domain,
            operator.domain,
            traits=close(traits | Traits.SELF_ADJOINT),
        )
        self._operator = operator
        self._function = function
        self._max_iterations = max_iterations
        self._rtol = rtol
        self._reorthogonalise = reorthogonalise

    @property
    def base_operator(self) -> LinearOperator:
        """The operator the function is applied to."""
        return self._operator

    def _value(self, x: Any) -> Any:
        return apply_operator_function(
            self._operator,
            self._function,
            x,
            max_iterations=self._max_iterations,
            rtol=self._rtol,
            reorthogonalise=self._reorthogonalise,
        )

    def _adjoint_value(self, y: Any) -> Any:
        return self._value(y)

    def __repr__(self) -> str:
        return f"OperatorFunction({self._operator!r})"


def operator_function(
    operator: LinearOperator,
    function: Callable[[np.ndarray], np.ndarray],
    /,
    *,
    traits: Traits = Traits.NONE,
    **kwargs: Any,
) -> LinearOperator:
    """``f(A)``, dispatching on how ``A`` is stored.

    A diagonal operator evaluates ``f`` on its eigenvalues, exactly. Anything
    else self-adjoint gets a lazily-applied :class:`OperatorFunction`.

    ``traits`` are the caller's claim about ``f(A)``: the library cannot
    inspect ``f``, so it cannot know whether the result is positive definite.
    The named helpers below supply the right claim for their own function.
    """
    if isinstance(operator, DiagonalLinearOperator):
        result = operator.apply_function(function)
        return result.with_traits(traits) if traits else result
    return OperatorFunction(operator, function, traits=traits, **kwargs)


def operator_sqrt(operator: LinearOperator, /, **kwargs: Any) -> LinearOperator:
    """``A^(1/2)``. Requires positive semidefiniteness; the result inherits it."""
    _require(operator, Traits.POSITIVE_SEMIDEFINITE, "A square root")
    return operator_function(
        operator, np.sqrt, traits=Traits.POSITIVE_SEMIDEFINITE, **kwargs
    )


def operator_inverse_sqrt(operator: LinearOperator, /, **kwargs: Any) -> LinearOperator:
    """``A^(-1/2)``. Requires positive definiteness."""
    _require(operator, Traits.POSITIVE_DEFINITE, "An inverse square root")
    return operator_function(
        operator,
        lambda values: 1.0 / np.sqrt(values),
        traits=Traits.POSITIVE_DEFINITE,
        **kwargs,
    )


def operator_exp(operator: LinearOperator, /, **kwargs: Any) -> LinearOperator:
    """``exp(A)``, which is positive definite for any self-adjoint ``A``."""
    return operator_function(
        operator, np.exp, traits=Traits.POSITIVE_DEFINITE, **kwargs
    )


def operator_log(operator: LinearOperator, /, **kwargs: Any) -> LinearOperator:
    """``log(A)``. Requires positive definiteness; the result is indefinite."""
    _require(operator, Traits.POSITIVE_DEFINITE, "A logarithm")
    return operator_function(operator, np.log, **kwargs)


def operator_power(
    operator: LinearOperator, power: float, /, **kwargs: Any
) -> LinearOperator:
    """``A^p``. A fractional power requires positive semidefiniteness."""
    if power != int(power):
        _require(operator, Traits.POSITIVE_SEMIDEFINITE, "A fractional power")
    traits = Traits.POSITIVE_SEMIDEFINITE if power != int(power) else Traits.NONE
    return operator_function(
        operator, lambda values: values**power, traits=traits, **kwargs
    )


def _require(operator: LinearOperator, needed: Traits, what: str) -> None:
    """Raise unless the operator claims the trait an operation needs."""
    if needed & operator.traits != needed:
        raise ValueError(
            f"{what} requires {needed!s}; this operator claims " f"{operator.traits!s}."
        )
