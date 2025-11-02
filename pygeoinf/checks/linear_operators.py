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
    from ..linear_forms import LinearForm


class LinearOperatorAxiomChecks(NonLinearOperatorAxiomChecks):
    """
    A mixin for checking the properties of a LinearOperator.

    Inherits the derivative check from NonLinearOperatorAxiomChecks and adds
    checks for linearity and the adjoint identity.
    """

    def _check_linearity(
        self,
        x: Vector,
        y: Vector,
        a: float,
        b: float,
        check_rtol: float = 1e-5,
        check_atol: float = 1e-8,
    ):
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

        if relative_error > check_rtol and diff_norm > check_atol:
            raise AssertionError(
                f"Linearity check failed: L(ax+by) != aL(x)+bL(y). "
                f"Relative error: {relative_error:.2e} (Tol: {check_rtol:.2e}), "
                f"Absolute error: {diff_norm:.2e} (Tol: {check_atol:.2e})"
            )

    def _check_adjoint_definition(
        self,
        x: Vector,
        y: Vector,
        check_rtol: float = 1e-5,
        check_atol: float = 1e-8,
    ):
        """Verifies the adjoint identity: <L(x), y> = <x, L*(y)>"""
        lhs = self.codomain.inner_product(self(x), y)
        rhs = self.domain.inner_product(x, self.adjoint(y))

        if not np.isclose(lhs, rhs, rtol=check_rtol, atol=check_atol):
            raise AssertionError(
                f"Adjoint definition failed: <L(x),y> = {lhs:.4e}, "
                f"but <x,L*(y)> = {rhs:.4e} (RelTol: {check_rtol:.2e}, AbsTol: {check_atol:.2e})"
            )

    def _check_algebraic_identities(
        self,
        op1,
        op2,
        x,
        y,
        a,
        check_rtol: float = 1e-5,
        check_atol: float = 1e-8,
    ):
        """
        Verifies the algebraic properties of the adjoint and dual operators.
        Requires a second compatible operator (op2).
        """

        def _check_norm_based(res1, res2, space, axiom_name):
            """Helper to perform norm-based comparison."""
            diff_norm = space.norm(space.subtract(res1, res2))
            norm_res2 = space.norm(res2)
            if diff_norm > check_atol and diff_norm > check_rtol * (norm_res2 + 1e-12):
                raise AssertionError(
                    f"Axiom failed: {axiom_name}. "
                    f"Absolute error: {diff_norm:.2e}, Relative error: {diff_norm / (norm_res2 + 1e-12):.2e}"
                )

        # --- Adjoint Identities ---
        # (A+B)* = A* + B*
        res1 = (op1 + op2).adjoint(y)
        res2 = (op1.adjoint + op2.adjoint)(y)
        _check_norm_based(res1, res2, op1.domain, "(A+B)* != A* + B*")

        # (a*A)* = a*A*
        res1 = (a * op1).adjoint(y)
        res2 = (a * op1.adjoint)(y)
        _check_norm_based(res1, res2, op1.domain, "(a*A)* != a*A*")

        # (A*)* = A
        res1 = op1.adjoint.adjoint(x)
        res2 = op1(x)
        _check_norm_based(res1, res2, op1.codomain, "(A*)* != A")

        # (A@B)* = B*@A*
        if op1.domain == op2.codomain:
            res1 = (op1 @ op2).adjoint(y)
            res2 = (op2.adjoint @ op1.adjoint)(y)
            _check_norm_based(res1, res2, op2.domain, "(A@B)* != B*@A*")

        # --- Dual Identities ---
        # (A+B)' = A' + B'
        op_sum_dual = (op1 + op2).dual
        dual_sum = op1.dual + op2.dual
        y_dual = op1.codomain.to_dual(y)

        # The result of applying a dual operator is a LinearForm
        res1_form: LinearForm = op_sum_dual(y_dual)
        res2_form: LinearForm = dual_sum(y_dual)

        # CORRECTED: Use LinearForm subtraction and dual space norm
        # (This assumes LinearForm overloads __sub__)
        try:
            diff_form = res1_form - res2_form
            diff_norm = op1.domain.dual.norm(diff_form)
            norm_res2 = op1.domain.dual.norm(res2_form)

            if diff_norm > check_atol and diff_norm > check_rtol * (norm_res2 + 1e-12):
                raise AssertionError(
                    f"Axiom failed: (A+B)' != A' + B'. "
                    f"Absolute error: {diff_norm:.2e}, Relative error: {diff_norm / (norm_res2 + 1e-12):.2e}"
                )
        except (AttributeError, TypeError):
            # Fallback if LinearForm doesn't support subtraction or norm
            if not np.allclose(
                res1_form.components,
                res2_form.components,
                rtol=check_rtol,
                atol=check_atol,
            ):
                raise AssertionError(
                    "Axiom failed: (A+B)' != A' + B' (component check)."
                )

    def check(
        self,
        n_checks: int = 5,
        op2=None,
        check_rtol: float = 1e-5,
        check_atol: float = 1e-8,
    ) -> None:
        """
        Runs all checks for the LinearOperator, including non-linear checks
        and algebraic identities.

        Args:
            n_checks: The number of randomized trials to perform.
            op2: An optional second operator for testing algebraic rules.
            check_rtol: The relative tolerance for numerical checks.
            check_atol: The absolute tolerance for numerical checks.
        """
        # First, run the parent (non-linear) checks from the base class
        super().check(n_checks, op2=op2, check_rtol=check_rtol, check_atol=check_atol)

        # Now, run the linear-specific checks
        print(
            f"Running {n_checks} additional randomized checks for linearity and adjoints..."
        )
        for _ in range(n_checks):
            x1 = self.domain.random()
            x2 = self.domain.random()
            y = self.codomain.random()
            a, b = np.random.randn(), np.random.randn()

            self._check_linearity(
                x1, x2, a, b, check_rtol=check_rtol, check_atol=check_atol
            )
            self._check_adjoint_definition(
                x1, y, check_rtol=check_rtol, check_atol=check_atol
            )

            if op2:
                self._check_algebraic_identities(
                    self, op2, x1, y, a, check_rtol=check_rtol, check_atol=check_atol
                )

        print(f"✅ All {n_checks} linear operator checks passed successfully.")
