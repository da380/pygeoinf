"""
Provides a self-checking mechanism for LinearOperator implementations.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

# Import the base checks from the sibling module
from .nonlinear_operators import NonLinearOperatorAxiomChecks


if TYPE_CHECKING:
    from ..hilbert_space import Vector


class LinearOperatorAxiomChecks(NonLinearOperatorAxiomChecks):
    """
    A mixin for checking the properties of a LinearOperator.

    Inherits the derivative check from NonLinearOperatorAxiomChecks and adds
    checks for linearity and the adjoint identity.
    """

    def _check_linearity(self, x: Vector, y: Vector, a: float, b: float):
        """Verifies the linearity property: L(ax + by) = a*L(x) + b*L(y)"""
        ax_plus_by = self.domain.add(
            self.domain.multiply(a, x), self.domain.multiply(b, y)
        )
        lhs = self(ax_plus_by)

        aLx = self.codomain.multiply(a, self(x))
        bLy = self.codomain.multiply(b, self(y))
        rhs = self.codomain.add(aLx, bLy)

        # Compare the results in the codomain
        diff_norm = self.codomain.norm(self.codomain.subtract(lhs, rhs))
        rhs_norm = self.codomain.norm(rhs)
        relative_error = diff_norm / (rhs_norm + 1e-12)

        if relative_error > 1e-9:
            raise AssertionError(
                f"Linearity check failed: L(ax+by) != aL(x)+bL(y). Relative error: {relative_error:.2e}"
            )

    def _check_adjoint_definition(self, x: Vector, y: Vector):
        """Verifies the adjoint identity: <L(x), y> = <x, L*(y)>"""
        lhs = self.codomain.inner_product(self(x), y)
        rhs = self.domain.inner_product(x, self.adjoint(y))

        if not np.isclose(lhs, rhs):
            raise AssertionError(
                f"Adjoint definition failed: <L(x),y> = {lhs:.4e}, but <x,L*(y)> = {rhs:.4e}"
            )

    def check(self, n_checks: int = 5) -> None:
        """
        Runs all checks for the LinearOperator, including non-linear checks.
        """
        # First, run the parent (non-linear) checks from the base class
        super().check(n_checks)

        # Now, run the linear-specific checks
        print(
            f"Running {n_checks} additional randomized checks for linearity and adjoints..."
        )
        for _ in range(n_checks):
            x1 = self.domain.random()
            x2 = self.domain.random()
            y = self.codomain.random()
            a, b = np.random.randn(), np.random.randn()

            self._check_linearity(x1, x2, a, b)
            self._check_adjoint_definition(x1, y)

        print(f"✅ All {n_checks} linear operator checks passed successfully.")
