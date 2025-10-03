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

    def _check_algebraic_identities(self, op1, op2, x, y, a):
        """
        Verifies the algebraic properties of the adjoint and dual operators.
        Requires a second compatible operator (op2).
        """
        # --- Adjoint Identities ---
        # (A+B)* = A* + B*
        op_sum_adj = (op1 + op2).adjoint
        adj_sum = op1.adjoint + op2.adjoint
        diff = op1.domain.subtract(op_sum_adj(y), adj_sum(y))
        if op1.domain.norm(diff) > 1e-9:
            raise AssertionError("Axiom failed: (A+B)* != A* + B*")

        # (a*A)* = a*A*
        op_scaled_adj = (a * op1).adjoint
        adj_scaled = a * op1.adjoint
        diff = op1.domain.subtract(op_scaled_adj(y), adj_scaled(y))
        if op1.domain.norm(diff) > 1e-9:
            raise AssertionError("Axiom failed: (a*A)* != a*A*")

        # (A*)* = A
        op_adj_adj = op1.adjoint.adjoint
        diff = op1.codomain.subtract(op_adj_adj(x), op1(x))
        if op1.codomain.norm(diff) > 1e-9:
            raise AssertionError("Axiom failed: (A*)* != A")

        # (A@B)* = B*@A*
        if op1.domain == op2.codomain:
            op_comp_adj = (op1 @ op2).adjoint
            adj_comp = op2.adjoint @ op1.adjoint
            diff = op2.domain.subtract(op_comp_adj(y), adj_comp(y))
            if op2.domain.norm(diff) > 1e-9:
                raise AssertionError("Axiom failed: (A@B)* != B*@A*")

        # --- Dual Identities ---
        # (A+B)' = A' + B'
        op_sum_dual = (op1 + op2).dual
        dual_sum = op1.dual + op2.dual
        y_dual = op1.codomain.to_dual(y)
        # The result of applying a dual operator is a LinearForm, which supports subtraction
        diff_dual = op_sum_dual(y_dual) - dual_sum(y_dual)
        if op1.domain.dual.norm(diff_dual) > 1e-9:
            raise AssertionError("Axiom failed: (A+B)' != A' + B'")

    def check(self, n_checks: int = 5, op2=None) -> None:
        """
        Runs all checks for the LinearOperator, including non-linear checks
        and algebraic identities.
        """
        # First, run the parent (non-linear) checks from the base class
        super().check(n_checks, op2=op2)

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

            if op2:
                self._check_algebraic_identities(self, op2, x1, y, a)

        print(f"✅ All {n_checks} linear operator checks passed successfully.")
