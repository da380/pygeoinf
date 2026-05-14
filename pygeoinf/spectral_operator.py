"""
Spectral functional calculus for self-adjoint operators.

Given an approximate eigendecomposition $A \\approx U \\Lambda U^*$ of a
self-adjoint operator on a Hilbert space, this module builds a matrix-free
``LinearOperator`` representing $f(A) \\approx U f(\\Lambda) U^*$ for any
real-valued function $f$ on the spectrum. The intended use case is
covariance-derived ellipsoid metrics in `pygeoinf.gaussian_measure`,
specifically the family of fractional powers $C^\\theta$ for
$\\theta \\in \\mathbb{R}$.

The construction mirrors `pygeoinf.low_rank.LowRankEig` but lifts an
arbitrary function $f$ rather than a fixed identity on the eigenvalues.
For the standard square-root, inverse, and inverse-square-root operators
this is more flexible than the diagonal functional calculus on
`DiagonalSparseMatrixLinearOperator`, which only handles the natural
coefficient basis.

Mathematical background: see
``docs/agent-docs/theory/function-space-hardening.md`` section 3 and the
sequel.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np

from .hilbert_space import EuclideanSpace
from .linear_operators import (
    DiagonalSparseMatrixLinearOperator,
    LinearOperator,
)
from .low_rank import LowRankEig


_RealFunction = Callable[[np.ndarray], np.ndarray]


class SpectralFractionalOperator(LinearOperator):
    """Matrix-free $f(A)$ from an eigendecomposition $A = U \\Lambda U^*$.

    The operator acts as

    $$
        f(A)\\, v = U\\,\\mathrm{diag}(f(\\lambda_1), \\ldots, f(\\lambda_k))\\, U^* v.
    $$

    When $U$ is an isometry onto $\\mathrm{ran}(A)$ this is the standard
    functional calculus restricted to that range; outside the range the
    operator returns zero, consistent with $f(0) \\cdot 0 = 0$ for typical
    fractional powers.

    Args:
        u_op: A ``LinearOperator`` mapping the coefficient space ($\\mathbb{R}^k$
            for some rank $k$) to the ambient Hilbert space $H$.
        eigenvalues: A 1-D array of $k$ eigenvalues.
        func: A vectorised function that maps the eigenvalue array to the
            transformed-eigenvalue array. For fractional powers,
            ``func = lambda x: x ** theta``.

    Raises:
        ValueError: if ``eigenvalues`` length does not match
            ``u_op.domain.dim``.
    """

    def __init__(
        self,
        u_op: LinearOperator,
        eigenvalues: Union[np.ndarray, "np.typing.ArrayLike"],
        func: _RealFunction,
    ) -> None:
        eig = np.asarray(eigenvalues, dtype=float).ravel()
        if eig.size != u_op.domain.dim:
            raise ValueError(
                f"eigenvalues has length {eig.size} but u_op.domain has "
                f"dim {u_op.domain.dim}."
            )
        self._u_op = u_op
        self._raw_eigenvalues = eig
        self._func = func

        transformed = np.asarray(func(eig), dtype=float).ravel()
        if transformed.shape != eig.shape:
            raise ValueError(
                "func must be vectorised: applied to the eigenvalue array "
                f"it must return an array of the same shape {eig.shape}, "
                f"got shape {transformed.shape}."
            )
        self._transformed_eigenvalues = transformed

        coefficient_space = u_op.domain
        if not isinstance(coefficient_space, EuclideanSpace):
            # Spectral calculus requires a canonical basis in the coefficient
            # space; the LowRankEig contract guarantees a EuclideanSpace.
            raise TypeError(
                "u_op.domain must be a EuclideanSpace (the coefficient "
                f"space carrying the eigenvalues); got {type(coefficient_space)!r}."
            )

        d_op = DiagonalSparseMatrixLinearOperator.from_diagonal_values(
            coefficient_space,
            coefficient_space,
            transformed,
        )
        self._d_op = d_op

        composed = u_op @ d_op @ u_op.adjoint
        super().__init__(
            composed.domain,
            composed.codomain,
            composed,
            adjoint_mapping=composed.adjoint,
        )

    # ------------------------------------------------------------------ #
    # Factories
    # ------------------------------------------------------------------ #

    @classmethod
    def from_low_rank_eig(
        cls,
        eig: LowRankEig,
        power: float,
        *,
        regularization: float = 0.0,
    ) -> "SpectralFractionalOperator":
        """Build $A^{\\text{power}}$ from a ``LowRankEig`` decomposition.

        For ``power < 0``, the eigenvalues must be strictly positive — in
        practice $A$ is the covariance and the trailing eigenvalues from
        a randomised decomposition can be numerically zero or tiny. The
        ``regularization`` argument adds a constant offset to the
        eigenvalues before applying the power, to avoid blow-up.

        Args:
            eig: A ``LowRankEig`` instance, typically obtained via
                ``LowRankEig.from_randomized(covariance, ...)``.
            power: The real exponent.
            regularization: Constant added to every eigenvalue before the
                power; ignored if ``power >= 0``.

        Returns:
            A ``SpectralFractionalOperator`` acting as $A^{\\text{power}}$
            on the range of ``eig.u_factor``.
        """
        eigvals = np.asarray(eig.eigenvalues, dtype=float)
        reg = float(regularization) if power < 0 else 0.0
        if power < 0 and not np.all(eigvals + reg > 0):
            raise ValueError(
                "Negative power requires strictly positive eigenvalues "
                "(after regularization); supply a larger regularization or "
                "truncate the decomposition."
            )

        def func(x: np.ndarray) -> np.ndarray:
            return np.power(x + reg, power)

        return cls(eig.u_factor, eigvals, func)

    @classmethod
    def from_callable(
        cls,
        u_op: LinearOperator,
        eigenvalues: np.ndarray,
        func: _RealFunction,
    ) -> "SpectralFractionalOperator":
        """Build $f(A)$ from explicit factors and a callable.

        This is the lowest-level constructor; the typical user wants
        :meth:`from_low_rank_eig`.
        """
        return cls(u_op, eigenvalues, func)

    # ------------------------------------------------------------------ #
    # Attribute accessors
    # ------------------------------------------------------------------ #

    @property
    def u_factor(self) -> LinearOperator:
        """Eigenvectors $U$ as a ``LinearOperator`` $\\mathbb{R}^k \\to H$."""
        return self._u_op

    @property
    def diagonal_operator(self) -> DiagonalSparseMatrixLinearOperator:
        """The diagonal operator $\\mathrm{diag}(f(\\lambda_j))$ on $\\mathbb{R}^k$."""
        return self._d_op

    @property
    def raw_eigenvalues(self) -> np.ndarray:
        """The input eigenvalues $\\lambda_j$ (before $f$)."""
        return self._raw_eigenvalues

    @property
    def transformed_eigenvalues(self) -> np.ndarray:
        """The output eigenvalues $f(\\lambda_j)$."""
        return self._transformed_eigenvalues

    @property
    def rank(self) -> int:
        """Number of eigenvalue/eigenvector pairs retained."""
        return self._raw_eigenvalues.size

    # ------------------------------------------------------------------ #
    # Coefficient-space helpers (used by the gauge evaluator)
    # ------------------------------------------------------------------ #

    def quadratic_form_squared(self, v) -> float:
        """Return $\\langle v, f(A)\\, v\\rangle_H$ via the spectral expansion.

        Equals $\\sum_j f(\\lambda_j)\\, c_j^2$ where $c = U^* v$ is the
        coefficient vector. Uses the underlying ``u_op.adjoint`` to compute
        coefficients, avoiding the extra round-trip through ``u_op``.
        """
        coefficients = np.asarray(self._u_op.adjoint(v), dtype=float).ravel()
        return float(
            np.dot(self._transformed_eigenvalues, coefficients * coefficients)
        )

    def coefficients(self, v) -> np.ndarray:
        """Return $U^* v$ as a NumPy array."""
        return np.asarray(self._u_op.adjoint(v), dtype=float).ravel()


def fractional_operators_from_eig(
    eig: LowRankEig,
    theta: float,
    *,
    regularization: Optional[float] = None,
) -> tuple[
    SpectralFractionalOperator,
    SpectralFractionalOperator,
    SpectralFractionalOperator,
]:
    """Convenience triple for the weakened-ellipsoid construction.

    Returns ``(A, A_inv, A_inv_sqrt)`` corresponding to $C^{-\\theta}$,
    $C^{\\theta}$, $C^{\\theta/2}$ respectively. These are exactly the
    operator/inverse/inverse-sqrt arguments that :class:`pygeoinf.subsets.Ellipsoid`
    needs to support its quadratic form and support-function evaluations.

    Args:
        eig: Eigendecomposition of the covariance $C$.
        theta: Fractional power; expected in $(0, 1)$.
        regularization: Constant offset added to eigenvalues before applying
            negative powers. Defaults to a relative floor of ``1e-12`` times
            the largest eigenvalue.
    """
    eigvals = np.asarray(eig.eigenvalues, dtype=float)
    if regularization is None:
        regularization = 1e-12 * float(np.max(np.abs(eigvals)))

    operator = SpectralFractionalOperator.from_low_rank_eig(
        eig, -theta, regularization=regularization
    )
    inverse = SpectralFractionalOperator.from_low_rank_eig(eig, theta)
    inverse_sqrt = SpectralFractionalOperator.from_low_rank_eig(
        eig, 0.5 * theta
    )
    return operator, inverse, inverse_sqrt
