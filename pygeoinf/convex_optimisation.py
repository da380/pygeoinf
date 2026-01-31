"""
Convex optimisation utilities for non-smooth problems.

This module provides a minimal subgradient descent implementation suitable
for learning and experimentation. It assumes the objective is a NonLinearForm
that can provide a subgradient oracle via form.subgradient(x).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, TYPE_CHECKING

import numpy as np

from .nonlinear_forms import NonLinearForm

if TYPE_CHECKING:
    from .hilbert_space import HilbertSpace, Vector


@dataclass
class SubgradientResult:
    """Result from subgradient descent optimisation.

    Attributes:
        x_best: Best point found (lowest function value).
        f_best: Best function value found.
        x_final: Final iterate (may differ from x_best).
        f_final: Final function value.
        num_iterations: Number of iterations performed.
        converged: Whether convergence criterion was met.
        function_values: History of function values at each iteration.
        iterates: Optional history of all iterates (memory intensive).
    """

    x_best: "Vector"
    f_best: float
    x_final: "Vector"
    f_final: float
    num_iterations: int
    converged: bool
    function_values: List[float]
    iterates: Optional[List["Vector"]] = None


class SubgradientDescent:
    """
    Basic subgradient descent for minimising non-smooth convex functions.

    Algorithm:
        x_{k+1} = x_k - α * g_k

    where g_k ∈ ∂f(x_k) is a subgradient (obtained via oracle.subgradient(x_k)).

    This implementation uses CONSTANT step size α for all k. Convergence is
    not guaranteed with constant step size; use for learning/testing only.

    Parameters:
        oracle: A NonLinearForm with subgradient() method returning subgradients.
        step_size: Constant step size α > 0.
        max_iterations: Maximum number of iterations.
        store_iterates: Whether to store full history (memory intensive).
        stagnation_window: Optional number of iterations without improvement
            to declare convergence.
    """

    def __init__(
        self,
        oracle: NonLinearForm,
        /,
        *,
        step_size: float,
        max_iterations: int = 500,
        store_iterates: bool = False,
        stagnation_window: Optional[int] = None,
    ) -> None:
        if not isinstance(oracle, NonLinearForm):
            raise ValueError("oracle must be a NonLinearForm")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if stagnation_window is not None and stagnation_window <= 0:
            raise ValueError("stagnation_window must be positive if provided")

        self._oracle = oracle
        self._step_size = float(step_size)
        self._max_iterations = int(max_iterations)
        self._store_iterates = bool(store_iterates)
        self._stagnation_window = stagnation_window

    @property
    def oracle(self) -> NonLinearForm:
        return self._oracle

    @property
    def domain(self) -> "HilbertSpace":
        return self._oracle.domain

    def solve(self, x0: "Vector") -> SubgradientResult:
        """Run subgradient descent from initial point x0."""
        if not self.domain.is_element(x0):
            raise ValueError("x0 must be an element of the oracle domain")
        if not self._oracle.has_subgradient:
            raise ValueError("oracle must provide a subgradient")

        x = x0
        f_best = float("inf")
        x_best = x0
        function_values: List[float] = []
        iterates: Optional[List["Vector"]] = [] if self._store_iterates else None

        no_improve = 0
        converged = False

        for _ in range(self._max_iterations):
            f_x = self._oracle(x)
            function_values.append(float(f_x))

            if f_x < f_best:
                f_best = float(f_x)
                x_best = x
                no_improve = 0
            else:
                no_improve += 1

            if self._stagnation_window is not None:
                if no_improve >= self._stagnation_window:
                    converged = True
                    break

            if iterates is not None:
                iterates.append(x)

            g = self._oracle.subgradient(x)
            step = self.domain.multiply(self._step_size, g)
            x = self.domain.subtract(x, step)

        f_final = float(self._oracle(x))

        return SubgradientResult(
            x_best=x_best,
            f_best=f_best,
            x_final=x,
            f_final=f_final,
            num_iterations=len(function_values),
            converged=converged,
            function_values=function_values,
            iterates=iterates,
        )
