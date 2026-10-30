"""
Functional calculus: ``f(A)`` for a self-adjoint operator.

Entirely coordinate-free. Lanczos tridiagonalization needs only the operator's
action, an inner product and ``axpy``, so ``f(A)`` is available on a space with
no component map — which matters, because a covariance square root is how a
Gaussian gets sampled and a log-determinant is how evidence gets computed.

Two paths, chosen by structure rather than by hand:

- an operator **diagonal in a basis** evaluates ``f`` on its eigenvalues, which
  is exact and costs one array operation;
- anything else self-adjoint goes to **Lanczos**, which approximates ``f(A) x``
  from a small Krylov space without ever forming a matrix.

The split follows how the operator is stored: "diagonal in a basis" is a
class, :class:`~pygeoinf.algebra.diagonal.DiagonalLinearOperator`, because it
carries eigenvalues, while "self-adjoint" is a trait, because it carries
nothing.

Traits gate the operations that need them: a square root requires positive
semidefiniteness, a logarithm requires positive definiteness. Since the library
cannot inspect an arbitrary ``f``, the named operations declare what they need
and the general entry point takes the caller's word.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, Sequence

import numpy as np
from numpy.random import Generator
from scipy.linalg import eigh_tridiagonal

from ..algebra.diagonal import DiagonalLinearOperator
from ..algebra.operators import LinearOperator
from ..algebra.spaces import (
    REORTHOGONALIZATION_THRESHOLD,
    CoordinateSpace,
    HilbertSpace,
)

if TYPE_CHECKING:  # pragma: no cover
    from .randomized import Estimate
from ..traits import Traits, close

__all__ = [
    "log_determinant",
    "lanczos_tridiagonalize",
    "iter_lanczos_tridiagonalize",
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


# Ritz values of a claimed-definite operator below this fraction of the
# largest are treated as roundoff and floored; below minus this fraction they
# are treated as a false claim and refused. Lanczos roundoff on a formed
# covariance is 1e-15 of the largest eigenvalue; a real negative eigenvalue
# is never this small relative to the spectrum.
_SPECTRUM_TOLERANCE = 1e-8


def _guard_spectrum(
    operator: LinearOperator, function: Callable[[np.ndarray], np.ndarray]
) -> Callable[[np.ndarray], np.ndarray]:
    """Pull Ritz values back inside the spectrum the operator claims.

    A covariance formed as ``L L*`` with a null space has null eigenvalues
    that roundoff puts at ``-1e-13`` or so, and the Ritz values inherit the
    sign: ``log``, ``sqrt`` and a fractional power of one are NaN. The claim
    says what the spectrum is, so a Ritz value outside it by roundoff is
    moved to its edge -- zero for a semidefinite claim, ``eps`` times the
    largest Ritz value for a definite one, which is what keeps a logarithm
    finite. A value meaningfully outside is not roundoff but a wrong claim,
    and is refused, as the dense :func:`log_determinant` refuses an indefinite
    matrix. No claim, no guard.
    """
    traits = operator.traits
    if not (Traits.POSITIVE_SEMIDEFINITE & traits):
        return function
    definite = bool(Traits.POSITIVE_DEFINITE & traits)
    eps = float(np.finfo(float).eps)

    def guarded(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        scale = float(np.max(np.abs(values))) if values.size else 0.0
        if scale and float(np.min(values)) < -_SPECTRUM_TOLERANCE * scale:
            claim = "POSITIVE_DEFINITE" if definite else "POSITIVE_SEMIDEFINITE"
            raise ValueError(
                f"The operator has an eigenvalue near {float(np.min(values)):.3g} "
                f"against a largest of {scale:.3g}, so it is not {claim} as "
                f"claimed; verify the claim with testing.check_traits()."
            )
        floor = eps * scale if definite else 0.0
        return function(np.maximum(values, floor))

    return guarded


# --------------------------------------------------------------------- #
#                              Lanczos                                  #
# --------------------------------------------------------------------- #


def iter_lanczos_tridiagonalize(
    operator: LinearOperator,
    start: Any,
    max_iterations: int,
    /,
    *,
    reorthogonalize: bool = True,
    breakdown_tol: float = 1e-12,
) -> Iterator[tuple[list[Any], np.ndarray]]:
    """Yield the Krylov basis and tridiagonal matrix after each iteration.

    Lazily, so a caller can stop on its own convergence test without deciding
    the iteration count in advance.

    Args:
        operator: a self-adjoint operator.
        start: the vector generating the Krylov space. Need not be normalized.
        max_iterations: the most Lanczos steps to take.
        reorthogonalize: re-project each new vector against the whole basis.
            Lanczos loses orthogonality in floating point after very few steps,
            and the cheap three-term recurrence alone is not usable for
            spectral work; full reorthogonalization costs ``O(k)`` inner
            products per step and is worth it at the small ``k`` used here.
        breakdown_tol: the size of a new off-diagonal ``beta``, relative to
            the operator's scale on the Krylov space so far -- the largest
            ``|alpha|`` and ``beta`` met -- below which the space is taken to
            have closed, which is exact termination rather than failure. No
            absolute floor: an operator whose norm is ``1e-13`` in its
            units runs exactly as one of norm one does.

    Yields:
        ``(basis, T)`` after each step, with ``T`` symmetric tridiagonal of
        size equal to the basis.

    Raises:
        ValueError: if the operator does not claim self-adjointness, or the
            starting vector is zero -- which generates no Krylov space at all.
    """
    space: HilbertSpace = operator.domain
    _require_self_adjoint(operator, "Lanczos tridiagonalization")

    if isinstance(space, CoordinateSpace) and space.uses_component_fast_paths:
        yield from _iter_lanczos_on_components(
            operator,
            space,
            start,
            max_iterations,
            reorthogonalize=reorthogonalize,
            breakdown_tol=breakdown_tol,
        )
        return

    norm = space.norm(start)
    if norm == 0.0:
        raise ValueError("The starting vector for Lanczos must be nonzero.")

    basis = [space.scale(1.0 / norm, start)]
    diagonal: list[float] = []
    off_diagonal: list[float] = []

    previous_beta = 0.0
    scale = 0.0
    for step in range(max_iterations):
        w = operator(basis[-1])
        alpha = space.inner_product(w, basis[-1])
        diagonal.append(alpha)

        w = space.axpy(-alpha, basis[-1], w)
        if step > 0:
            w = space.axpy(-previous_beta, basis[-2], w)
        if reorthogonalize:
            # One pass, and *modified* Gram-Schmidt: each projection is taken
            # against the running w rather than the original, which is what
            # makes a single sweep enough here. A second ("twice is enough")
            # pass leaves the basis orthogonal to the same ~1e-15 even on
            # spectra chosen to break a single pass, and would double the
            # cost of the step that dominates a Lanczos run.
            for q in basis:
                w = space.axpy(-space.inner_product(w, q), q, w)

        yield list(basis), _tridiagonal(diagonal, off_diagonal)

        beta = space.norm(w)
        scale = max(scale, abs(alpha), previous_beta)
        if beta <= breakdown_tol * scale or step + 1 == max_iterations:
            return
        off_diagonal.append(beta)
        previous_beta = beta
        basis.append(space.scale_inplace(1.0 / beta, w))


def _iter_lanczos_on_components(
    operator: LinearOperator,
    space: CoordinateSpace,
    start: Any,
    max_iterations: int,
    /,
    *,
    reorthogonalize: bool,
    breakdown_tol: float,
) -> Iterator[tuple[list[Any], np.ndarray]]:
    """The same iteration with the Krylov basis kept as component columns.

    One analysis and one synthesis per step -- the synthesis is the basis
    vector the operator is applied to, so it is not extra -- against ``2k + 4``
    analyses per step through ``inner_product`` on a spectral space. The
    metric enters only through ``apply_gram``. Reorthogonalization is
    classical Gram-Schmidt against the whole basis with Kahan's second pass
    when the first removes more than ``1 - 1/sqrt(2)`` of the norm, which is
    one metric application per pass instead of one per basis vector.

    The columns are held in a buffer that doubles when it fills, starting
    from a few columns, so the memory held is proportional to the steps
    actually taken and never to *max_iterations*, whose default is the
    dimension: a run that terminates in five steps on a space of dimension
    ``10^5`` holds a few vectors, not ``10^5`` of them.
    """
    c = np.array(space.to_components(start), dtype=float)
    weighted = space.apply_gram(c)
    norm = float(np.sqrt(max(float(c @ weighted), 0.0)))
    if norm == 0.0:
        raise ValueError("The starting vector for Lanczos must be nonzero.")

    columns = np.empty((c.size, min(max_iterations, _INITIAL_COLUMNS)))
    columns[:, 0] = c / norm
    basis = [space.from_components(columns[:, 0].copy())]
    diagonal: list[float] = []
    off_diagonal: list[float] = []

    previous_beta = 0.0
    scale = 0.0
    for step in range(max_iterations):
        w = np.array(space.to_components(operator(basis[-1])), dtype=float)
        weighted = space.apply_gram(w)
        current = columns[:, : step + 1]
        if reorthogonalize:
            # The first pass's last coefficient is alpha itself, and its
            # second-to-last is beta; projecting off the whole basis at once
            # subsumes the three-term recurrence.
            before = float(np.sqrt(max(float(w @ weighted), 0.0)))
            coefficients = current.T @ weighted
            alpha = float(coefficients[-1])
            w -= current @ coefficients
            weighted = space.apply_gram(w)
            beta = float(np.sqrt(max(float(w @ weighted), 0.0)))
            if beta < REORTHOGONALIZATION_THRESHOLD * before:
                w -= current @ (current.T @ weighted)
                weighted = space.apply_gram(w)
                beta = float(np.sqrt(max(float(w @ weighted), 0.0)))
        else:
            alpha = float(columns[:, step] @ weighted)
            w -= alpha * columns[:, step]
            if step > 0:
                w -= previous_beta * columns[:, step - 1]
            weighted = space.apply_gram(w)
            beta = float(np.sqrt(max(float(w @ weighted), 0.0)))
        diagonal.append(alpha)

        yield list(basis), _tridiagonal(diagonal, off_diagonal)

        scale = max(scale, abs(alpha), previous_beta)
        if beta <= breakdown_tol * scale or step + 1 == max_iterations:
            return
        off_diagonal.append(beta)
        previous_beta = beta
        if step + 1 == columns.shape[1]:
            grown = np.empty(
                (columns.shape[0], min(2 * columns.shape[1], max_iterations))
            )
            grown[:, : step + 1] = columns
            columns = grown
        columns[:, step + 1] = w / beta
        basis.append(space.from_components(columns[:, step + 1].copy()))


# The component Lanczos buffer's first allocation, in columns; it doubles from
# there as the iteration needs.
_INITIAL_COLUMNS = 8


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


def lanczos_tridiagonalize(
    operator: LinearOperator,
    start: Any,
    max_iterations: int,
    /,
    *,
    reorthogonalize: bool = True,
    breakdown_tol: float = 1e-12,
) -> tuple[list[Any], np.ndarray]:
    """Run Lanczos to completion, returning the final basis and matrix.

    Args:
        operator: a self-adjoint operator.
        start: the vector whose Krylov space is built.
        max_iterations: the dimension to stop at.
        reorthogonalize: keep the basis orthogonal against rounding. Worth
            the cost: without it the Lanczos vectors lose orthogonality
            exactly as the method converges.
        breakdown_tol: as for :func:`iter_lanczos_tridiagonalize`.

    Returns:
        The basis and the tridiagonal matrix.
    """
    basis, matrix = [], np.zeros((0, 0))
    for basis, matrix in iter_lanczos_tridiagonalize(
        operator,
        start,
        max_iterations,
        reorthogonalize=reorthogonalize,
        breakdown_tol=breakdown_tol,
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
    max_iterations: int | None = None,
    rtol: float = 1e-10,
    reorthogonalize: bool = True,
) -> Any:
    """``f(A) x``, without forming ``f(A)``.

    Lanczos builds an orthonormal basis ``Q`` of the Krylov space with
    ``Q* A Q == T``, so ``f(A) x`` is approximated by
    ``||x|| Q f(T) e_1`` — a small dense eigendecomposition of ``T`` and a
    linear combination of the basis vectors.

    Convergence is judged on the change in the answer between successive
    Krylov dimensions, which is the honest test when nothing is known about
    ``f`` -- but judged on the *coefficients* ``f(T) e_1``, not on the vector
    they combine to. The two say the same thing, since ``Q`` is orthonormal,
    and only one of them is cheap: recombining at every step to compare costs
    ``O(k^2)`` vector operations over the run and makes the convergence test
    more expensive than the iteration it is testing. The basis is combined
    once, at the end.

    **The defaults are adaptive.** The cap is the dimension of the space,
    where Lanczos terminates exactly, so by default the tolerance decides
    when to stop and the cap never truncates. A cap the caller sets still
    truncates without complaint: it is a quadrature degree, and stochastic
    Lanczos quadrature relies on that.

    Args:
        operator: a self-adjoint ``A``.
        function: applied to the eigenvalues.
        x: the vector to apply ``f(A)`` to.
        max_iterations: the Krylov dimension to stop at. The space's
            dimension if omitted.
        rtol: relative change in the coefficients to stop at. Tighter than
            the solvers' ``1e-8`` because the change between two Krylov
            dimensions under-reports the distance to the answer, by about two
            orders of magnitude at ``1e-8``.
        reorthogonalize: keep the Krylov basis orthogonal.

    Returns:
        ``f(A) x``.
    """
    space: HilbertSpace = operator.domain
    _require_self_adjoint(operator, "Applying an operator function")
    function = _guard_spectrum(operator, function)
    if max_iterations is None:
        max_iterations = space.dim

    norm = space.norm(x)
    if norm == 0.0:
        return space.zero()

    previous: np.ndarray | None = None
    final_basis: Sequence[Any] = ()
    weights = np.ones(1)
    for basis, matrix in iter_lanczos_tridiagonalize(
        operator, x, max_iterations, reorthogonalize=reorthogonalize
    ):
        values, vectors = _eigh_tridiagonal(matrix)
        weights = vectors @ (np.asarray(function(values)) * vectors[0, :])
        final_basis = basis
        if previous is not None:
            padded = np.zeros_like(weights)
            padded[: previous.size] = previous
            change = float(np.linalg.norm(weights - padded))
            if change <= rtol * max(float(np.linalg.norm(weights)), 1e-300):
                break
        previous = weights

    result = space.zero()
    for weight, vector in zip(weights, final_basis):
        result = space.axpy(norm * float(weight), vector, result)
    return result


# Below this order the dense symmetric solver beats the tridiagonal one, whose
# cost at these sizes is Python argument validation rather than arithmetic; the
# crossover is between 20 and 40 on one BLAS thread, and a Lanczos run that
# stops on its own rarely reaches 20. The two agree to rounding, and both uses
# here -- ``V f(L) V^T e_1`` and ``V[0]^2`` -- are invariant under the
# eigenvector sign conventions the two drivers may differ on.
#
# The convergence test runs every step, and should: checking only every k
# steps never saves an operator application, since the application for a step
# has already happened by the time the step is yielded, so a coarser check
# only makes the iteration run further before it notices it has converged.
# Making the check cheaper is what pays.
_DENSE_EIGH_LIMIT = 32


def _eigh_tridiagonal(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose a small symmetric tridiagonal matrix.

    Called once per Lanczos step by the two convergence tests, so its
    per-call overhead is the convergence check's whole cost, which matters
    when the operator itself is cheap.

    Args:
        matrix: the tridiagonal matrix, dense.

    Returns:
        Its eigenvalues and eigenvectors, ascending.
    """
    k = matrix.shape[0]
    if k == 1:
        return matrix[0].copy(), np.ones((1, 1))
    if k <= _DENSE_EIGH_LIMIT:
        # The dense matrix is already in hand; the tridiagonal driver would
        # only re-extract its two bands from it.
        return np.linalg.eigh(matrix)
    return eigh_tridiagonal(np.diag(matrix), np.diag(matrix, 1))


def operator_quadratic_form(
    operator: LinearOperator,
    function: Callable[[np.ndarray], np.ndarray],
    x: Any,
    /,
    *,
    max_iterations: int | None = None,
    rtol: float = 1e-10,
    reorthogonalize: bool = True,
) -> float:
    """``(x, f(A) x)``, by Gauss quadrature on the Lanczos spectrum.

    Cheaper than forming ``f(A) x`` and pairing it, because only the first row
    of the eigenvector matrix is needed and the Krylov basis is never
    recombined. It is the quadrature that stochastic Lanczos quadrature
    averages over probes; :func:`log_determinant` takes the trace of
    ``log(A)`` through :func:`~pygeoinf.numerics.randomized.random_trace`
    instead, applying :class:`OperatorFunction` to each probe.

    Stops when the quadrature value settles, rather than running every one
    of ``max_iterations``: for a stochastic estimator built on it -- many
    probes, each its own Lanczos run -- that is the cost that matters.

    Args:
        operator: a self-adjoint ``A``.
        function: applied to the eigenvalues.
        x: the vector.
        max_iterations: the Krylov dimension to stop at. The space's
            dimension if omitted; see :func:`apply_operator_function` on why.
        rtol: relative change in the value to stop at.
        reorthogonalize: keep the Krylov basis orthogonal.

    Returns:
        ``(x, f(A) x)``.
    """
    space: HilbertSpace = operator.domain
    _require_self_adjoint(operator, "An operator quadratic form")
    function = _guard_spectrum(operator, function)
    if max_iterations is None:
        max_iterations = space.dim

    squared_norm = space.squared_norm(x)
    if squared_norm == 0.0:
        return 0.0

    estimate = 0.0
    previous: float | None = None
    for _, matrix in iter_lanczos_tridiagonalize(
        operator, x, max_iterations, reorthogonalize=reorthogonalize
    ):
        values, vectors = _eigh_tridiagonal(matrix)
        # The Gauss quadrature weights are the squared first components.
        estimate = float(np.dot(vectors[0, :] ** 2, np.asarray(function(values))))
        if previous is not None and abs(estimate - previous) <= rtol * max(
            abs(estimate), 1e-300
        ):
            break
        previous = estimate
    return squared_norm * estimate


# --------------------------------------------------------------------- #
#                        f(A) as an operator                            #
# --------------------------------------------------------------------- #


class OperatorFunction(LinearOperator):
    """``f(A)`` as an operator, applied lazily by Lanczos.

    Nothing is precomputed: each application runs its own Krylov iteration.
    That is the right trade when ``f(A)`` is applied a few times to different
    vectors, and the wrong one when it is applied many times — in which case a
    low-rank factorization of ``A`` is the better tool.
    """

    def __init__(
        self,
        operator: LinearOperator,
        function: Callable[[np.ndarray], np.ndarray],
        /,
        *,
        traits: Traits = Traits.NONE,
        max_iterations: int | None = None,
        rtol: float = 1e-10,
        reorthogonalize: bool = True,
    ) -> None:
        """
        Args:
            operator: a self-adjoint ``A``.
            function: applied to the eigenvalues.
            traits: the claim about ``f(A)``; self-adjointness is added.
            max_iterations: the Krylov dimension each application stops at.
                The space's dimension if omitted, so that *rtol* decides.
            rtol: relative change in the coefficients to stop at.
            reorthogonalize: keep the Krylov basis orthogonal.
        """
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
        self._reorthogonalize = reorthogonalize

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
            reorthogonalize=self._reorthogonalize,
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
    else gets a lazily-applied :class:`OperatorFunction`.

    **Self-adjointness is required on both routes**, so that one request gets
    one answer whichever route it takes. The Lanczos route needs it outright
    -- a three-term recurrence in the space's own inner product computes
    ``f(A)`` only for self-adjoint ``A`` -- and the named helpers below attach
    traits to the result, ``operator_sqrt`` claiming ``POSITIVE_SEMIDEFINITE``
    of what it returns, which is true only of the self-adjoint case. An
    operator diagonal *in components* is self-adjoint only where its values
    commute with the metric, which on a non-diagonal Gram matrix they do not.

    Where the metric is not diagonal and the exact answer is still wanted,
    :class:`~pygeoinf.algebra.diagonal.DiagonalLinearOperator` carries its
    own calculus -- :attr:`~pygeoinf.algebra.diagonal.DiagonalLinearOperator.sqrt`,
    :attr:`~pygeoinf.algebra.diagonal.DiagonalLinearOperator.log`,
    :meth:`~pygeoinf.algebra.diagonal.DiagonalLinearOperator.apply_function`
    -- which gates on the spectrum instead. That is exact on any metric,
    because an eigendecomposition is a property of the map; what it does not
    give there is a self-adjoint result, and it says so through the traits it
    returns rather than by refusing.

    On a Euclidean space, and on every symmetric space here, a diagonal
    operator's metric is diagonal in the same basis, so it commutes and the
    trait is deduced: the requirement bites only where it should.

    **The claimed spectrum is enforced on the Lanczos route.** Ritz values
    that roundoff puts outside a positive semidefinite or definite claim are
    moved to its edge before ``f`` sees them, so a formed covariance whose
    null eigenvalues are ``-1e-13`` still has a finite ``log``, ``sqrt`` and
    fractional power; a value meaningfully outside the claim is refused.
    The diagonal route is exact and needs no such guard.

    Args:
        operator: a self-adjoint ``A``.
        function: applied to the eigenvalues.
        traits: the caller's claim about ``f(A)``. The library cannot inspect
            ``f``, so it cannot know whether the result is positive definite.
            The named helpers below supply the right claim for their own
            function.
        **kwargs: passed to :class:`OperatorFunction` on the Lanczos route.

    Returns:
        ``f(A)``.

    Raises:
        ValueError: if the operator does not claim self-adjointness.
    """
    _require_self_adjoint(operator, "An operator function")
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


def log_determinant(
    operator: LinearOperator,
    /,
    *,
    method: Literal["auto", "dense", "stochastic"] = "auto",
    samples: int = 100,
    rng: Generator | None = None,
    dense_limit: int = 4000,
    max_iterations: int = 40,
    rtol: float = 1e-3,
    sample_rtol: float | None = None,
    max_samples: int | None = None,
    n_jobs: int | None = None,
) -> "Estimate":
    """``log det A``, densely or by stochastic Lanczos quadrature.

    The identity is ``log det A == tr(log A)``, and the second form needs
    neither the matrix nor its eigenvalues: ``log(A) z`` comes from a Lanczos
    iteration on the Krylov space of ``z``, and Hutchinson's estimator turns
    a handful of those into the trace. Both cost only applications of ``A``,
    which is what makes an evidence calculation possible at all on a data space
    too large to assemble — the case a log-determinant otherwise rules out.

    **Which determinant.** The *component* matrix's, not the Galerkin matrix's:
    ``det(G A_c) == det G det A_c``, so only the first is a property of the
    operator rather than of the metric. The dense route subtracts the metric's
    own determinant to get there; the stochastic route needs no correction,
    because ``random_trace`` probes with white noise on the space and so
    estimates ``tr A_c`` already. That the two agree is the parity test.

    Args:
        operator: a positive definite self-adjoint operator.
        method: ``"dense"`` forms the matrix; ``"stochastic"`` never does;
            ``"auto"`` takes the dense route when the space has a component
            map, is small enough to hold and factorize the matrix
            (*dense_limit*), and either the operator can hand its matrix over
            or probing it costs no more than the stochastic route's own
            budget of applications. The dense route is exact, so where the
            two cost the same it is the one to take.
        samples: Hutchinson probes, for the stochastic route -- or the first
            block of them when *sample_rtol* is given.
        rng: the generator for those probes.
        dense_limit: the largest dimension at which ``"auto"`` will form and
            factorize a ``dim x dim`` matrix. It is a **memory and
            factorization** budget, not a statement about applications:
            ``dim**2`` doubles is 128 MB at 4000 and the ``slogdet`` is
            ``O(dim**3)``. Below it the dense route is exact, and on a
            well-conditioned operator also faster than the stochastic one.

            Whether the *probe* is affordable is decided separately, since a
            matrix that must be probed costs ``dim`` applications: above
            ``samples * max_iterations`` -- the stochastic route's own
            application budget -- the dense route is taken only when the
            operator can write its matrix down without being applied.
        max_iterations: the Krylov dimension allowed for each ``log(A) z``.
        rtol: the Lanczos budget for each ``log(A) z``. Note
            this ``rtol`` is the *inner* one: it says how well each
            ``log(A) z`` is computed, not how well the trace over them is
            estimated. The two are separate budgets and tightening the wrong
            one buys nothing.
        sample_rtol: draw further probes until the estimate's standard error
            falls to this fraction of it, rather than stopping at a fixed
            count. This is the tolerance on the answer.
        max_samples: a ceiling on that.

    Returns:
        An :class:`~pygeoinf.numerics.randomized.Estimate`. The dense route
        reports a standard error of zero, so a caller can treat the two
        uniformly and still see which it got.

    Raises:
        ValueError: if the operator is not a positive definite endomorphism,
            if the method is unknown, or if the dense route finds the matrix
            singular or indefinite. A determinant needs an endomorphism and a
            *logarithm* of one needs the definiteness.
    """
    from ..algebra.diagonal import DiagonalLinearOperator
    from .randomized import Estimate, random_trace

    _require(operator, Traits.POSITIVE_DEFINITE, "A log determinant")
    if not operator.is_endomorphism:
        raise ValueError("A determinant needs an operator from a space to itself.")

    space = operator.domain
    if method not in ("auto", "dense", "stochastic"):
        raise ValueError(
            f"The method is 'auto', 'dense' or 'stochastic', got {method!r}."
        )

    if method == "auto" and isinstance(operator, DiagonalLinearOperator):
        # ``sum(log lambda)``: exact, O(dim), and no applications at all. Worth
        # a special case because it is the common one — every invariant measure
        # on a symmetric space has a diagonal covariance — and because the
        # alternatives are both bad here: the dense route spends ``dim``
        # applications assembling a matrix that is already known, and above
        # dense_limit the stochastic one estimates a number that can be summed.
        return Estimate(operator.log_determinant, 0.0, 0)

    matrix: np.ndarray | None = None
    if method == "auto":
        from ..algebra.spaces import CoordinateSpace

        if not isinstance(space, CoordinateSpace) or space.dim > dense_limit:
            method = "stochastic"
        else:
            # Above the probe budget the dense route is taken only if the
            # matrix can be *read*: ``known_matrix`` applies the operator
            # zero times by contract, so this costs nothing when it answers
            # None, and hands over the matrix when it does not.
            matrix = operator.known_matrix(form="galerkin")
            method = (
                "dense"
                if matrix is not None or space.dim <= samples * max_iterations
                else "stochastic"
            )

    if method == "dense":
        if matrix is None:
            matrix = operator.matrix(form="galerkin", n_jobs=n_jobs)
        sign, logarithm = np.linalg.slogdet(0.5 * (matrix + matrix.T))
        if sign <= 0:
            raise ValueError(
                "The operator's matrix is singular or indefinite, so it has no "
                "log determinant. It was claimed POSITIVE_DEFINITE; verify that "
                "with testing.check_traits()."
            )
        # ``log det I == 0``: the branch saves a factorization as large as the
        # operator's own.
        metric = 0.0
        if space.has_diagonal_metric:
            # A diagonal metric's determinant is the product of its diagonal,
            # with no O(dim^3) slogdet of a diagonal matrix.
            metric = float(np.sum(np.log(space.gram_diagonal())))
        elif not space.is_orthonormal:
            _, metric = np.linalg.slogdet(space.gram_matrix())
        return Estimate(float(logarithm - metric), 0.0, 0)

    return random_trace(
        operator_log(operator, max_iterations=max_iterations, rtol=rtol),
        samples=samples,
        rtol=sample_rtol,
        max_samples=max_samples,
        rng=rng,
        n_jobs=n_jobs,
    )
