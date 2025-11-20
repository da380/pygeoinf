"""
Utilities for Robin boundary condition eigenvalue computation.

This module provides a unified implementation of root-finding methods
for computing eigenvalues with Robin boundary conditions, eliminating
code duplication across providers.py and function_providers.py.
"""

import math
import numpy as np
from typing import Callable, Tuple


class RobinRootFinder:
    """
    Unified Robin boundary condition eigenvalue computation.

    This class provides methods for finding eigenvalues μ_n that satisfy
    the characteristic equation for Robin boundary conditions:

        D(μ) = (α₀α_L + β₀β_L μ²) sin(μL) + μ(α₀β_L - β₀α_L) cos(μL) = 0

    where:
        - α₀, β₀: left boundary condition coefficients (α₀u(a) + β₀u'(a) = 0)
        - α_L, β_L: right boundary condition coefficients (α_Lu(b) + β_Lu'(b) = 0)
        - L: domain length (b - a)

    The class handles special cases including:
        - Pure Neumann (α₀ = α_L = 0): zero eigenvalue exists
        - Pure Dirichlet (β₀ = β_L = 0): no zero eigenvalue
        - Mixed and general Robin conditions

    Methods use bisection for robustness, with automatic bracket expansion
    and fallback strategies to ensure convergence.
    """

    @staticmethod
    def bisect(
        F: Callable[[float], float],
        a: float,
        b: float,
        tol: float = 1e-12,
        maxit: int = 100
    ) -> float:
        """
        Standard bisection method for root-finding.

        Args:
            F: Function whose root we seek
            a: Left bracket endpoint
            b: Right bracket endpoint
            tol: Tolerance for convergence (both function value and interval width)
            maxit: Maximum number of iterations

        Returns:
            Approximate root of F in [a, b]

        Raises:
            ValueError: If F(a) and F(b) have the same sign (no bracket)

        Example:
            >>> finder = RobinRootFinder()
            >>> root = finder.bisect(lambda x: x**2 - 2, 0, 2)
            >>> abs(root - math.sqrt(2)) < 1e-10
            True
        """
        fa, fb = F(a), F(b)

        # Check for exact roots at endpoints
        if fa == 0.0:
            return a
        if fb == 0.0:
            return b

        # Ensure sign change
        if fa * fb > 0:
            raise ValueError(
                f"Bisection requires F(a)·F(b) ≤ 0, but "
                f"F({a}) = {fa} and F({b}) = {fb} have the same sign."
            )

        # Bisection iteration
        for _ in range(maxit):
            c = 0.5 * (a + b)
            fc = F(c)

            # Check convergence
            if abs(fc) < tol or 0.5 * (b - a) < tol:
                return c

            # Update bracket
            if fa * fc <= 0.0:
                b, fb = c, fc
            else:
                a, fa = c, fc

        # Return midpoint if max iterations reached
        return 0.5 * (a + b)

    @staticmethod
    def find_bracket_with_expansion(
        F: Callable[[float], float],
        left: float,
        right: float,
        max_attempts: int = 6
    ) -> Tuple[float, float]:
        """
        Find a bracketing interval by gentle expansion.

        If F(left) and F(right) have the same sign, gradually expand
        the interval by 10% on each side until a sign change is found.

        Args:
            F: Function to bracket
            left: Initial left bound
            right: Initial right bound
            max_attempts: Maximum number of expansion attempts

        Returns:
            Tuple (left, right) where F(left) * F(right) <= 0

        Raises:
            RuntimeError: If bracket cannot be found after max_attempts

        Example:
            >>> F = lambda x: math.tan(x)
            >>> left, right = RobinRootFinder.find_bracket_with_expansion(
            ...     F, 3.0, 3.2, max_attempts=10
            ... )
            >>> F(left) * F(right) <= 0
            True
        """
        Fl, Fr = F(left), F(right)
        attempts = 0

        while Fl * Fr > 0 and attempts < max_attempts:
            # Expand bracket by 10% on each side
            left *= 0.9
            right *= 1.1
            Fl, Fr = F(left), F(right)
            attempts += 1

        if Fl * Fr > 0:
            # Could not find bracket via expansion
            raise RuntimeError(
                f"Failed to bracket root after {max_attempts} expansion attempts. "
                f"F({left}) = {Fl}, F({right}) = {Fr}"
            )

        return left, right

    @staticmethod
    def find_bracket_by_scanning(
        F: Callable[[float], float],
        left: float,
        right: float,
        n_samples: int = 129
    ) -> Tuple[float, float]:
        """
        Find a bracketing interval by scanning for sign changes.

        This is a fallback method when expansion fails. It evaluates F
        at n_samples points in [left, right] and finds the first interval
        where F changes sign.

        Args:
            F: Function to bracket
            left: Interval left bound
            right: Interval right bound
            n_samples: Number of sample points to check

        Returns:
            Tuple (left, right) containing a single root

        Raises:
            RuntimeError: If no sign change found in any subinterval

        Example:
            >>> F = lambda x: math.sin(x)
            >>> left, right = RobinRootFinder.find_bracket_by_scanning(
            ...     F, 3.0, 3.5
            ... )
            >>> F(left) * F(right) <= 0
            True
        """
        xs = np.linspace(left, right, n_samples)
        vals = np.array([F(xi) for xi in xs])
        sgn = np.sign(vals)

        # Find indices where sign changes
        idx = np.where(sgn[:-1] * sgn[1:] <= 0)[0]

        if len(idx) == 0:
            raise RuntimeError(
                f"No sign change found in [{left}, {right}] "
                f"with {n_samples} samples."
            )

        # Return first bracketing interval
        i = idx[0]
        return xs[i], xs[i + 1]

    @staticmethod
    def compute_robin_eigenvalue(
        index: int,
        alpha_0: float,
        beta_0: float,
        alpha_L: float,
        beta_L: float,
        length: float,
        tol: float = 1e-12,
        maxit: int = 100
    ) -> float:
        """
        Compute the n-th eigenvalue for Robin boundary conditions.

        Finds the index-th root of the characteristic equation:
            D(μ) = (α₀α_L + β₀β_L μ²) sin(μL) + μ(α₀β_L - β₀α_L) cos(μL) = 0

        The method handles special cases:
            - Pure Neumann (α₀ = α_L = 0): μ₀ = 0 for index=0
            - Pure Dirichlet (β₀ = β_L = 0): no zero eigenvalue
            - General Robin: roots found by bisection

        Roots are located in intervals approximately [nπ/L, (n+1)π/L]
        where n depends on the boundary conditions and index.

        Args:
            index: Eigenvalue index (0-based)
            alpha_0: Left BC coefficient for u(a)
            beta_0: Left BC coefficient for u'(a)
            alpha_L: Right BC coefficient for u(b)
            beta_L: Right BC coefficient for u'(b)
            length: Domain length L = b - a
            tol: Root-finding tolerance
            maxit: Maximum bisection iterations

        Returns:
            The index-th eigenvalue μ_n > 0 (or μ₀ = 0 for pure Neumann)

        Raises:
            RuntimeError: If root cannot be bracketed or found

        Example:
            >>> # Pure Neumann case: α₀ = α_L = 0, β₀ = β_L = 1
            >>> mu_0 = RobinRootFinder.compute_robin_eigenvalue(
            ...     0, 0.0, 1.0, 0.0, 1.0, 1.0
            ... )
            >>> abs(mu_0) < 1e-10  # Should be zero
            True
            >>>
            >>> # Pure Dirichlet case: α₀ = α_L = 1, β₀ = β_L = 0
            >>> mu_1 = RobinRootFinder.compute_robin_eigenvalue(
            ...     0, 1.0, 0.0, 1.0, 0.0, 1.0
            ... )
            >>> abs(mu_1 - math.pi) < 0.01  # Should be ~ π
            True
        """
        # Define characteristic equation
        def D(mu: float) -> float:
            return (
                (alpha_0 * alpha_L + beta_0 * beta_L * mu * mu) * math.sin(mu * length)
                + mu * (alpha_0 * beta_L - beta_0 * alpha_L) * math.cos(mu * length)
            )

        # Special case: pure Neumann has μ₀ = 0
        if index == 0 and alpha_0 == 0.0 and alpha_L == 0.0:
            return 0.0

        # Determine which interval to search based on index and boundary conditions
        # For pure Neumann, index=0 is handled above, so start from index=1 → n=1
        # For other BCs (including pure Dirichlet), index=0 → n=1
        start_n = 1  # Most cases: first non-zero root is near π/L

        if alpha_0 == 0.0 and alpha_L == 0.0 and index > 0:
            # Pure Neumann: μ₀=0 exists, so index=1 → n=1, index=2 → n=2, etc.
            n = index
        else:
            # Pure Dirichlet or mixed Robin: index=0 → n=1, index=1 → n=2, etc.
            n = index + 1

        # Initial bracket: approximately [nπ/L, (n+1)π/L] with small buffer
        left = (n - 0.5) * math.pi / length
        right = (n + 0.5) * math.pi / length

        # Try to ensure sign change by gentle expansion
        try:
            left, right = RobinRootFinder.find_bracket_with_expansion(
                D, left, right, max_attempts=6
            )
        except RuntimeError:
            # Expansion failed, try scanning
            try:
                left, right = RobinRootFinder.find_bracket_by_scanning(
                    D, left, right, n_samples=129
                )
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to bracket Robin eigenvalue for index={index}, "
                    f"α₀={alpha_0}, β₀={beta_0}, α_L={alpha_L}, β_L={beta_L}, L={length}. "
                    f"Original error: {e}"
                )

        # Find root by bisection
        mu = RobinRootFinder.bisect(D, left, right, tol, maxit)
        return mu

    @staticmethod
    def compute_coefficients_from_left_bc(
        mu: float,
        alpha_0: float,
        beta_0: float,
        alpha_L: float,
        beta_L: float,
        length: float
    ) -> Tuple[float, float]:
        """
        Compute eigenfunction coefficients (A, B) from left boundary condition.

        For an eigenfunction of the form:
            φ(x) = A cos(μ(x-a)) + B sin(μ(x-a))

        The left BC α₀u(a) + β₀u'(a) = 0 gives:
            α₀A + β₀μB = 0

        We choose (A, B) = (β₀μ, -α₀) to satisfy this (up to scaling).

        If both α₀ and β₀ are zero (degenerate), we use the right BC instead.

        Args:
            mu: Eigenvalue
            alpha_0: Left BC coefficient for u(a)
            beta_0: Left BC coefficient for u'(a)
            alpha_L: Right BC coefficient for u(b)
            beta_L: Right BC coefficient for u'(b)
            length: Domain length (needed for right BC evaluation)

        Returns:
            Tuple (A, B) of coefficients satisfying the boundary conditions

        Example:
            >>> # Dirichlet left: α₀=1, β₀=0 → A=0, B=-1
            >>> A, B = RobinRootFinder.compute_coefficients_from_left_bc(
            ...     math.pi, 1.0, 0.0, 0.0, 1.0, 1.0
            ... )
            >>> abs(A) < 1e-10 and B != 0
            True
        """
        # Try left BC first
        if abs(alpha_0) + abs(beta_0) > 0.0:
            # From α₀A + β₀μB = 0, choose (A,B) = (β₀μ, -α₀)
            A, B = beta_0 * mu, -alpha_0
            if abs(A) + abs(B) > 0.0:
                return A, B

        # Degenerate left BC: use right BC
        # At x=b: α_L(A cos(μL) + B sin(μL)) + β_L μ(-A sin(μL) + B cos(μL)) = 0
        cosL = math.cos(mu * length)
        sinL = math.sin(mu * length)

        # Coefficients of A and B in the right BC equation
        coeff_A = alpha_L * cosL - beta_L * mu * sinL
        coeff_B = alpha_L * sinL + beta_L * mu * cosL

        # Choose (A, B) perpendicular to (coeff_A, coeff_B)
        # i.e., (A, B) = (-coeff_B, coeff_A)
        A, B = -coeff_B, coeff_A

        if abs(A) + abs(B) == 0.0:
            # Ultimate fallback: arbitrary non-zero
            A, B = 1.0, 0.0

        return A, B
