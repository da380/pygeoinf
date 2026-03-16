"""
Provides a self-checking mechanism for AffineOperator implementations.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

# Assuming this is in the same directory as nonlinear_operators.py
from .nonlinear_operators import NonLinearOperatorAxiomChecks

if TYPE_CHECKING:
    from ..hilbert_space import Vector
    from ..gaussian_measure import GaussianMeasure


class AffineOperatorAxiomChecks(NonLinearOperatorAxiomChecks):
    """
    A mixin for checking the mathematical properties of an AffineOperator.

    Inherits the finite-difference derivative checks from
    NonLinearOperatorAxiomChecks and adds specific checks for affine mappings.
    """

    def _check_translation_recovery(
        self,
        check_rtol: float = 1e-5,
        check_atol: float = 1e-8,
    ) -> None:
        """Verifies that F(0) = b."""
        zero_vec = self.domain.zero
        f_zero = self(zero_vec)
        b = self.translation_part

        diff_norm = self.codomain.norm(self.codomain.subtract(f_zero, b))
        b_norm = self.codomain.norm(b)

        if diff_norm > check_atol and diff_norm > check_rtol * (b_norm + 1e-12):
            raise AssertionError(
                f"Translation recovery failed: F(0) != b. "
                f"Relative error: {diff_norm / (b_norm + 1e-12):.2e}, "
                f"Absolute error: {diff_norm:.2e}"
            )

    def _check_affine_combination(
        self,
        x: Vector,
        y: Vector,
        alpha: float,
        check_rtol: float = 1e-5,
        check_atol: float = 1e-8,
    ) -> None:
        """Verifies F(alpha*x + (1-alpha)*y) = alpha*F(x) + (1-alpha)*F(y)."""
        one_minus_alpha = 1.0 - alpha

        # LHS: F(alpha*x + (1-alpha)*y)
        ax = self.domain.multiply(alpha, x)
        oma_y = self.domain.multiply(one_minus_alpha, y)
        comb_in = self.domain.add(ax, oma_y)
        lhs = self(comb_in)

        # RHS: alpha*F(x) + (1-alpha)*F(y)
        aFx = self.codomain.multiply(alpha, self(x))
        oma_Fy = self.codomain.multiply(one_minus_alpha, self(y))
        rhs = self.codomain.add(aFx, oma_Fy)

        diff_norm = self.codomain.norm(self.codomain.subtract(lhs, rhs))
        rhs_norm = self.codomain.norm(rhs)

        if diff_norm > check_atol and diff_norm > check_rtol * (rhs_norm + 1e-12):
            raise AssertionError(
                f"Affine combination check failed. "
                f"Relative error: {diff_norm / (rhs_norm + 1e-12):.2e}, "
                f"Absolute error: {diff_norm:.2e}"
            )

    def _check_derivative_consistency(
        self,
        x: Vector,
        v: Vector,
        check_rtol: float = 1e-5,
        check_atol: float = 1e-8,
    ) -> None:
        """Verifies that F'(x)v = A(v) where A is the linear part."""
        df_v = self.derivative(x)(v)
        A_v = self.linear_part(v)

        diff_norm = self.codomain.norm(self.codomain.subtract(df_v, A_v))
        A_v_norm = self.codomain.norm(A_v)

        if diff_norm > check_atol and diff_norm > check_rtol * (A_v_norm + 1e-12):
            raise AssertionError(
                f"Derivative consistency failed: F'(x) != A. "
                f"Relative error: {diff_norm / (A_v_norm + 1e-12):.2e}, "
                f"Absolute error: {diff_norm:.2e}"
            )

    def check(
        self,
        /,
        *,
        n_checks: int = 5,
        op2=None,
        check_rtol: float = 1e-5,
        check_atol: float = 1e-8,
        measure: GaussianMeasure = None,
    ) -> None:
        """
        Runs all checks for the AffineOperator, including base non-linear
        checks and affine-specific identities.
        """
        # Run base non-linear checks (finite differences, chain rules, etc.)
        super().check(
            n_checks=n_checks,
            op2=op2,
            check_rtol=check_rtol,
            check_atol=check_atol,
            measure=measure,
        )

        if measure is None:
            sampler = self.domain.random
        else:
            if measure.domain != self.domain:
                raise ValueError(
                    "Provided measure must be defined on the operator's domain"
                )
            sampler = measure.sample

        print(
            f"Running {n_checks} additional randomized checks for affine properties..."
        )

        # F(0) = b only needs to be checked once
        self._check_translation_recovery(check_rtol=check_rtol, check_atol=check_atol)

        for _ in range(n_checks):
            x = sampler()
            y = sampler()
            v = sampler()
            alpha = np.random.randn()

            self._check_affine_combination(
                x, y, alpha, check_rtol=check_rtol, check_atol=check_atol
            )
            self._check_derivative_consistency(
                x, v, check_rtol=check_rtol, check_atol=check_atol
            )

        print(f"[✓] All {n_checks} affine operator checks passed successfully.")
