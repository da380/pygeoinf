"""
Operators that are diagonal in a space's own basis.

This is the v2 home for what v1 calls ``InvariantLinearAutomorphism``: an
operator stored as its eigenvalues, closed under the algebra, and carrying a
complete functional calculus evaluated pointwise on the spectrum.

It is a **class**, not a trait, because it carries data. That is the rule from
DESIGN.md section 4: mathematical properties are traits, representations are
classes. Self-adjointness is a trait this class happens to claim; being
diagonal is a fact about how it is stored.

Self-adjointness holds only when the space's metric is also diagonal. With
component matrix ``diag(d)`` the Galerkin matrix is ``G diag(d)``, symmetric
exactly when ``G`` and ``diag(d)`` commute — so, for general ``d``, exactly
when ``G`` is diagonal. That is why ``CoordinateSpace.has_diagonal_metric``
exists, and why this class checks it rather than assuming it.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..traits import Traits, close
from .operators import LinearOperator, Operator
from .spaces import CoordinateSpace

__all__ = ["DiagonalLinearOperator"]


class DiagonalLinearOperator[V](LinearOperator[V, V]):
    """An operator stored as its eigenvalues in a space's basis."""

    def __init__(
        self,
        domain: CoordinateSpace[V],
        eigenvalues: np.ndarray,
        /,
        *,
        traits: Traits = Traits.NONE,
    ) -> None:
        """
        Args:
            domain: the space, whose basis the operator is diagonal in.
            eigenvalues: one value per basis direction.
            traits: extra claims. Symmetry and definiteness are deduced from
                the values and the metric, so they rarely need supplying.
        """
        if not isinstance(domain, CoordinateSpace):
            raise TypeError(
                f"A diagonal operator needs a basis to be diagonal in, and "
                f"{type(domain).__name__} provides no coordinate map."
            )
        values = np.asarray(eigenvalues, dtype=float)
        if values.shape != (domain.dim,):
            raise ValueError(
                f"Got {values.shape} eigenvalues for a space of dimension "
                f"{domain.dim}."
            )
        super().__init__(
            domain, domain, traits=close(traits | self._deduce_traits(domain, values))
        )
        self._eigenvalues = values

    @staticmethod
    def _deduce_traits(domain: CoordinateSpace, values: np.ndarray) -> Traits:
        """Read off the traits the eigenvalues and metric already imply."""
        if not domain.has_diagonal_metric:
            # Diagonal in the components, but not symmetric in the metric.
            return Traits.NONE
        result = Traits.SELF_ADJOINT
        if np.all(values >= 0.0):
            result |= Traits.POSITIVE_SEMIDEFINITE
        if np.all(values > 0.0):
            result |= Traits.POSITIVE_DEFINITE
        if np.all(values != 0.0):
            result |= Traits.INVERTIBLE
        if np.all(np.abs(values) == 1.0):
            result |= Traits.UNITARY
        if np.all((values == 0.0) | (values == 1.0)):
            result |= Traits.IDEMPOTENT
        return result

    # ----------------------------------------------------------------- #
    #                              Structure                            #
    # ----------------------------------------------------------------- #

    @property
    def eigenvalues(self) -> np.ndarray:
        """The diagonal, one value per basis direction."""
        return self._eigenvalues

    @property
    def trace(self) -> float:
        """The sum of the eigenvalues, exactly."""
        return float(np.sum(self._eigenvalues))

    def _value(self, x: V) -> V:
        space: CoordinateSpace = self.domain
        return space.from_components(self._eigenvalues * space.to_components(x))

    def _adjoint_value(self, y: V) -> V:
        if Traits.SELF_ADJOINT & self.traits:
            return self._value(y)
        # Without a diagonal metric the adjoint is G^-1 diag(d) G, which is not
        # diagonal, so it is not this class's business to represent it.
        space: CoordinateSpace = self.domain
        components = space.to_components(y)
        return space.from_components(
            space.solve_gram(self._eigenvalues * space.apply_gram(components))
        )

    def __repr__(self) -> str:
        return f"DiagonalLinearOperator(dim={self.domain.dim})"

    # ----------------------------------------------------------------- #
    #                      Structure-preserving algebra                 #
    # ----------------------------------------------------------------- #
    #
    # The specialisation protocol of DESIGN.md 5.4. Without these, the sum of
    # two diagonal operators would be a generic node and would lose both the
    # eigenvalues and the functional calculus that makes them worth storing.

    def _rebuild(self, eigenvalues: np.ndarray) -> DiagonalLinearOperator[V]:
        """Build an operator of this class. Subclasses override to stay in theirs."""
        return DiagonalLinearOperator(self.domain, eigenvalues)

    def _combine_add(self, other: Operator) -> Operator | None:
        if isinstance(other, DiagonalLinearOperator) and other.domain == self.domain:
            return self._rebuild(self._eigenvalues + other.eigenvalues)
        return None

    def _combine_radd(self, other: Operator) -> Operator | None:
        return self._combine_add(other)

    def _combine_compose(self, other: Operator) -> Operator | None:
        if isinstance(other, DiagonalLinearOperator) and other.domain == self.domain:
            return self._rebuild(self._eigenvalues * other.eigenvalues)
        return None

    def _combine_rcompose(self, other: Operator) -> Operator | None:
        return self._combine_compose(other)

    def _combine_scale(self, alpha: float) -> Operator | None:
        return self._rebuild(alpha * self._eigenvalues)

    # ----------------------------------------------------------------- #
    #                         Functional calculus                       #
    # ----------------------------------------------------------------- #

    def apply_function(
        self, function: Callable[[np.ndarray], np.ndarray], /
    ) -> DiagonalLinearOperator[V]:
        """``f(A)``, evaluated pointwise on the spectrum.

        Exact, and costs one array operation. This is the fast path that
        ``numerics.functional_calculus.operator_function`` dispatches to.
        """
        values = np.asarray(function(self._eigenvalues), dtype=float)
        if values.shape != self._eigenvalues.shape:
            raise ValueError(
                f"The function returned shape {values.shape}, expected "
                f"{self._eigenvalues.shape}."
            )
        return self._rebuild(values)

    @property
    def inverse(self) -> DiagonalLinearOperator[V]:
        """``A^-1``. Requires every eigenvalue to be nonzero."""
        if np.any(self._eigenvalues == 0.0):
            raise ZeroDivisionError(
                "The operator is singular: some eigenvalues are zero."
            )
        return self.apply_function(np.reciprocal)

    @property
    def sqrt(self) -> DiagonalLinearOperator[V]:
        """``A^(1/2)``. Requires positive semidefiniteness."""
        self._require(Traits.POSITIVE_SEMIDEFINITE, "a square root")
        return self.apply_function(np.sqrt)

    @property
    def inverse_sqrt(self) -> DiagonalLinearOperator[V]:
        """``A^(-1/2)``. Requires positive definiteness."""
        self._require(Traits.POSITIVE_DEFINITE, "an inverse square root")
        return self.apply_function(lambda d: 1.0 / np.sqrt(d))

    @property
    def exp(self) -> DiagonalLinearOperator[V]:
        """``exp(A)``."""
        return self.apply_function(np.exp)

    @property
    def log(self) -> DiagonalLinearOperator[V]:
        """``log(A)``. Requires positive definiteness."""
        self._require(Traits.POSITIVE_DEFINITE, "a logarithm")
        return self.apply_function(np.log)

    @property
    def log_determinant(self) -> float:
        """``log det A``, exactly. Requires positive definiteness."""
        self._require(Traits.POSITIVE_DEFINITE, "a log determinant")
        return float(np.sum(np.log(self._eigenvalues)))

    def __abs__(self) -> DiagonalLinearOperator[V]:
        """``|A|``, the pointwise absolute value of the spectrum."""
        return self.apply_function(np.abs)

    def __pow__(self, power: float) -> DiagonalLinearOperator[V]:
        """``A^p``. Requires positive semidefiniteness for fractional powers."""
        if power != int(power):
            self._require(Traits.POSITIVE_SEMIDEFINITE, "a fractional power")
        return self.apply_function(lambda d: d**power)

    def _require(self, needed: Traits, what: str) -> None:
        """Raise unless the operator has the trait a calculus step needs."""
        if needed & self.traits != needed:
            raise ValueError(
                f"{what.capitalize()} requires {needed!s}; this operator has "
                f"{self.traits!s}."
            )
