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

from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, Sequence

import numpy as np
from numpy.random import Generator
from scipy.linalg import eigh_tridiagonal

from ..algebra.diagonal import DiagonalLinearOperator
from ..algebra.operators import LinearOperator
from ..algebra.spaces import HilbertSpace

if TYPE_CHECKING:  # pragma: no cover
    from .randomised import Estimate
from ..traits import Traits, close

__all__ = [
    "log_determinant",
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
            # One pass, and *modified* Gram-Schmidt: each projection is taken
            # against the running w rather than the original, which is what
            # makes a single sweep enough here.
            #
            # The review asked for a second ("twice is enough") pass. Measured
            # on three cases chosen to break it -- a spectrum spanning 1e-6 to
            # 1e6 over 200 steps, a run taken to 250 of 300 dimensions, and
            # tenfold repeated eigenvalues -- the second pass moves
            # ``max |Q Q^T - I|`` from 1.3e-15 to between 1.1e-15 and 1.8e-15.
            # That is no change, and it doubles the cost of the step that
            # dominates a Lanczos run. Not done, on the measurement.
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
    ``f`` -- but judged on the *coefficients* ``f(T) e_1``, not on the vector
    they combine to. The two say the same thing, since ``Q`` is orthonormal,
    and only one of them is cheap: recombining at every step to compare costs
    ``O(k^2)`` vector operations over the run and makes the convergence test
    more expensive than the iteration it is testing. The basis is combined
    once, at the end. That is v1's arrangement.

    Args:
        operator: a self-adjoint ``A``.
        function: applied to the eigenvalues.
        x: the vector to apply ``f(A)`` to.
        max_iterations: the Krylov dimension to stop at.
        rtol: relative change in the coefficients to stop at.
        reorthogonalise: keep the Krylov basis orthogonal.

    Returns:
        ``f(A) x``.
    """
    space: HilbertSpace = operator.domain
    _require_self_adjoint(operator, "Applying an operator function")

    norm = space.norm(x)
    if norm == 0.0:
        return space.zero()

    previous: np.ndarray | None = None
    final_basis: Sequence[Any] = ()
    weights = np.ones(1)
    for basis, matrix in iter_lanczos_tridiagonalise(
        operator, x, max_iterations, reorthogonalise=reorthogonalise
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
    rtol: float = 1e-10,
    reorthogonalise: bool = True,
) -> float:
    """``(x, f(A) x)``, by Gauss quadrature on the Lanczos spectrum.

    Cheaper than forming ``f(A) x`` and pairing it, because only the first row
    of the eigenvector matrix is needed. This is the kernel of stochastic
    Lanczos quadrature, which is how a log-determinant gets estimated without
    a factorisation.

    Stops when the quadrature settles, as v1 did. It used to run every one of
    ``max_iterations`` however early the value converged, which for the outer
    stochastic estimator -- many probes, each its own Lanczos run -- is the
    cost that matters.

    Args:
        operator: a self-adjoint ``A``.
        function: applied to the eigenvalues.
        x: the vector.
        max_iterations: the Krylov dimension to stop at.
        rtol: relative change in the value to stop at.
        reorthogonalise: keep the Krylov basis orthogonal.

    Returns:
        ``(x, f(A) x)``.
    """
    space: HilbertSpace = operator.domain
    _require_self_adjoint(operator, "An operator quadratic form")

    squared_norm = space.squared_norm(x)
    if squared_norm == 0.0:
        return 0.0

    estimate = 0.0
    previous: float | None = None
    for _, matrix in iter_lanczos_tridiagonalise(
        operator, x, max_iterations, reorthogonalise=reorthogonalise
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
    else gets a lazily-applied :class:`OperatorFunction`.

    **Self-adjointness is required on both routes.** It always was on the
    Lanczos one, and the diagonal one used to skip the check -- so the same
    request was refused for a general operator and quietly accepted for a
    diagonal one. It is not a formality: an operator diagonal *in components*
    is self-adjoint only if its values commute with the metric, which on a
    non-diagonal Gram matrix they do not. What the exact route computes there
    is ``f`` applied component by component, which is a functional calculus in
    the basis but not the spectral one the name promises, and the two differ.
    :meth:`~pygeoinf2.algebra.diagonal.DiagonalLinearOperator.apply_function`
    is the component-wise operation, for when that is what is wanted.

    On a Euclidean space, and on every symmetric space here, a diagonal
    operator's metric is diagonal in the same basis, so it commutes and the
    trait is deduced: the requirement bites only where it should.

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
    dense_limit: int = 512,
    max_iterations: int = 40,
    rtol: float = 1e-3,
    sample_rtol: float | None = None,
    max_samples: int | None = None,
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
            ``"auto"`` takes the dense route when the space is small enough to
            afford it and has a component map, and the stochastic one
            otherwise.
        samples: Hutchinson probes, for the stochastic route -- or the first
            block of them when *sample_rtol* is given.
        rng: the generator for those probes.
        dense_limit: the dimension above which ``"auto"`` goes stochastic.
        max_iterations, rtol: the Lanczos budget for each ``log(A) z``. Note
            this ``rtol`` is the *inner* one: it says how well each
            ``log(A) z`` is computed, not how well the trace over them is
            estimated. The two are separate budgets and tightening the wrong
            one buys nothing.
        sample_rtol: draw further probes until the estimate's standard error
            falls to this fraction of it, rather than stopping at a fixed
            count. This is the tolerance on the answer.
        max_samples: a ceiling on that.

    Returns:
        An :class:`~pygeoinf2.numerics.randomised.Estimate`. The dense route
        reports a standard error of zero, so a caller can treat the two
        uniformly and still see which it got.
    """
    from ..algebra.diagonal import DiagonalLinearOperator
    from .randomised import Estimate, random_trace

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
        # dense_limit the stochastic one estimates a number that can be summed
        # (measured: 455.4 +/- 2.8 against an exact 456.7 at dimension 1000).
        return Estimate(operator.log_determinant, 0.0, 0)

    if method == "auto":
        from ..algebra.spaces import CoordinateSpace

        affordable = isinstance(space, CoordinateSpace) and space.dim <= dense_limit
        method = "dense" if affordable else "stochastic"

    if method == "dense":
        matrix = operator.matrix(form="galerkin")
        sign, logarithm = np.linalg.slogdet(0.5 * (matrix + matrix.T))
        if sign <= 0:
            raise ValueError(
                "The operator's matrix is singular or indefinite, so it has no "
                "log determinant. It was claimed POSITIVE_DEFINITE; verify that "
                "with testing.check_traits()."
            )
        _, metric = np.linalg.slogdet(space.gram_matrix())
        return Estimate(float(logarithm - metric), 0.0, 0)

    return random_trace(
        operator_log(operator, max_iterations=max_iterations, rtol=rtol),
        samples=samples,
        rtol=sample_rtol,
        max_samples=max_samples,
        rng=rng,
    )
