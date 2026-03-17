"""
Module for Backus-Gilbert like methods for solving inference problems. To be done...
"""

from __future__ import annotations
from typing import Optional, Union


from .hilbert_space import Vector
from .linear_operators import LinearOperator
from .linear_solvers import LinearSolver, CholeskySolver
from .forward_problem import LinearForwardProblem
from .inversion import LinearInference
from .linear_optimisation import (
    LinearMinimumNormInversion,
)


class BackusInference(LinearInference):
    """
    Solves a linear inference problem using Backus' method.
    """

    def __init__(
        self,
        forward_problem: LinearForwardProblem,
        property_operator: LinearOperator,
        prior_norm_bound: float,
        significance_level: float,
        /,
        *,
        constraint_solver=None,
        constraint_preconditioner=None,
    ):
        """
        Args:
            forward_problem: An instance of a linear forward problem that defines the
                relationship between model parameters and data.
            property_operator: A linear mapping takes elements of the model space to
                property vector of interest.
            prior_norm_bound: Prior bound on the norm of the model
            significance_level: The desired significance level (e.g., 0.95).
            constraint_solver: LinearSolver to use when imposing property constraints.
                Defaults to Choleksy solver.
            constraint_preconditioner: Preconditioner to use when imposing property
                constraints. Defaults to None

        Raises:
            ValueError: If the domain of the property operator is
                not equal to the model space.
            ValueError: If the prior norm bound is not positive.
            ValueError: If the significance level is not in the range (0,1)
        """

        super().__init__(forward_problem, property_operator)

        self.prior_norm_bound = prior_norm_bound
        self.signficance_level = significance_level

        self._constraint_solver = (
            CholeskySolver if constraint_solver is None else constraint_solver
        )
        self._constraint_preconditioner = constraint_preconditioner

    @property
    def prior_norm_bound(self) -> float:
        """
        Returns the prior norm bound.
        """
        return self._prior_norm_bound

    @prior_norm_bound.setter
    def prior_norm_bound(self, value: float):
        """
        Sets the prior norm bound.
        """

        if value <= 0:
            raise ValueError("Prior norm bound must be positive")
        self._prior_norm_bound = value

    @property
    def significance_level(self) -> float:
        """
        Returns the significance level.
        """
        return self._significance_level

    @significance_level.setter
    def significance_level(self, value: float):
        """
        Sets the prior norm bound.
        """

        if not (0 < value < 1):
            raise ValueError("Significance level must be in the range (0,1)")

        self._critical_chi_squared = self.forward_problem.critical_chi_squared(
            self.significance_level
        )
        self._significance_level = value

    @property
    def critical_chi_squared(self) -> float:
        """
        Returns the critical Chi squared.
        """
        return self._critical_chi_squared

    def test_data_compatibility(
        self,
        data: Vector,
        solver: LinearSolver,
        /,
        *,
        preconditioner: Optional[Union[LinearOperator, LinearSolver]] = None,
        minimum_damping: float = 0.0,
        maxiter: int = 100,
        rtol: float = 1.0e-6,
        atol: float = 0.0,
    ) -> bool:
        """
        Returns true if there exists a model that is compatible with both the
        data and the norm bound.
        """

        minimum_norm_inversion = LinearMinimumNormInversion(self.forward_problem)

        minimum_norm_solver = minimum_norm_inversion.minimum_norm_operator(
            solver,
            preconditioner=preconditioner,
            significance_level=self.significance_level,
            minimum_damping=minimum_damping,
            maxiter=maxiter,
            rtol=rtol,
            atol=atol,
        )

        minimum_norm_solution = minimum_norm_solver(data)

        minimum_norm_value = self.model_space.norm(minimum_norm_solution)

        return minimum_norm_value <= self.prior_norm_bound
