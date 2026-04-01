"""
Backus-Gilbert style dual master cost function.

Provides :class:`DualMasterCostFunction` — the oracle $\\varphi(\\lambda; q)$
minimised over $\\lambda$ in convex Backus-Gilbert / dual-level-set inversion.
"""

from __future__ import annotations

from typing import Optional, Union
import numpy as np

from .hilbert_space import HilbertSpace, Vector
from .linear_operators import LinearOperator
from .linear_solvers import LinearSolver, CholeskySolver
from .forward_problem import LinearForwardProblem
from .inversion import LinearInference
from .linear_optimisation import LinearMinimumNormInversion
from .nonlinear_forms import NonLinearForm
from .convex_analysis import SupportFunction


class DualMasterCostFunction(NonLinearForm):
    """
    Cost function for the master dual equation (Hilbert form):

        h_U(q)
        = inf_{λ ∈ D}
          { (λ, d̃)_D + σ_B(T* q - G* λ) + σ_V(-λ) }

    i.e.

        φ(λ; q) = (λ, d̃)_D + σ_B(T* q - G* λ) + σ_V(-λ)

    where:
      - σ_B is the support function of the model prior convex set B ⊆ M
      - σ_V is the support function of the data error convex set V ⊆ D

    Minimizing φ(λ; q) over λ ∈ D yields h_U(q).
    """

    def __init__(
        self,
        data_space: HilbertSpace,
        property_space: HilbertSpace,
        model_space: HilbertSpace,
        G: LinearOperator,
        T: LinearOperator,
        model_prior_support: SupportFunction,
        data_error_support: SupportFunction,
        observed_data: Vector,
        q_direction: Vector,
    ) -> None:

        self._validation(data_space, property_space, model_space,
                         G, T, model_prior_support, data_error_support,
                         observed_data, q_direction)
        self._data_space = data_space
        self._property_space = property_space
        self._model_space = model_space
        self._G = G
        self._T = T
        self._G_adj = G.adjoint
        self._model_prior_support = model_prior_support
        self._data_error_support = data_error_support
        self._observed_data = observed_data
        self._q = q_direction

        self._Tstar_q = self._T.adjoint(q_direction)

        super().__init__(
            data_space, self._mapping, subgradient=self._subgradient
        )

    @property
    def observed_data(self) -> Vector:
        """Observed data vector d̃ ∈ D."""
        return self._observed_data

    @property
    def direction(self) -> Vector:
        """Current property direction q ∈ P."""
        return self._q

    def set_direction(self, q: Vector) -> None:
        """Update the property direction q and recompute T* q."""
        if not self._property_space.is_element(q):
            raise ValueError("q must be an element of property_space")
        self._q = q
        self._Tstar_q = self._T.adjoint(q)

    def _mapping(self, lam: Vector) -> float:
        # Term 1: ⟨λ, d̃⟩_D
        term1 = self.domain.inner_product(lam, self._observed_data)

        # Term 2: σ_B(T*q - G*λ)
        Gstar_lam = self._G_adj(lam)
        hilbert_residual = self._model_space.subtract(self._Tstar_q, Gstar_lam)
        term2 = self._model_prior_support(hilbert_residual)

        # Term 3: σ_V(-λ)
        neg_lam = self.domain.negative(lam)
        term3 = self._data_error_support(neg_lam)

        return term1 + term2 + term3

    def _subgradient(self, lam: Vector) -> Vector:
        """
        Compute a subgradient of φ(λ; q).

        For the dual master cost function:
            φ(λ) = ⟨λ, d̃⟩ + σ_B(T*q - G*λ) + σ_V(-λ)

        A subgradient is:
            g ∈ d̃ - G*∂σ_B(T*q - G*λ) - ∂σ_V(-λ)

        where ∂σ_B and ∂σ_V are subdifferentials of the support functions.
        The support_point() method returns an element of the subdifferential.
        """
        term1_subgrad = self._observed_data

        Gstar_lam = self._G_adj(lam)
        hilbert_residual = self._model_space.subtract(self._Tstar_q, Gstar_lam)

        v = self._model_prior_support.support_point(hilbert_residual)
        w = self._data_error_support.support_point(self.domain.negative(lam))

        if v is None or w is None:
            return self._finite_difference_gradient(lam)

        term2_subgrad = self.domain.negative(self._G(v))
        term3_subgrad = self.domain.negative(w)

        return self.domain.add(
            term1_subgrad,
            self.domain.add(term2_subgrad, term3_subgrad),
        )

    def value_and_subgradient(self, lam: "Vector") -> "tuple[float, Vector]":
        """Compute the value and a subgradient of $\\varphi(\\lambda; q)$ in one pass.

        Shares the computation of $G^* \\lambda$ and the support points $v$, $w$
        between the value and subgradient evaluations, avoiding redundant work.

        The dual master cost function is

        .. math::

            \\varphi(\\lambda; q)
            = \\langle \\lambda, \\tilde{d} \\rangle_D
              + \\sigma_B(T^* q - G^* \\lambda)
              + \\sigma_V(-\\lambda)

        and a subgradient at $\\lambda$ is

        .. math::

            g = \\tilde{d} - G v - w,

        where $v \\in \\partial \\sigma_B(T^* q - G^* \\lambda)$ and
        $w \\in \\partial \\sigma_V(-\\lambda)$.

        Args:
            lam: Dual variable $\\lambda \\in D$.

        Returns:
            A tuple ``(f, g)`` where ``f = φ(λ; q)`` is the scalar value and
            ``g ∈ ∂φ(λ; q)`` is a subgradient vector in $D$.
        """
        # Shared computation: G* λ (using cached adjoint)
        Gstar_lam = self._G_adj(lam)

        hilbert_residual = self._model_space.subtract(self._Tstar_q, Gstar_lam)
        neg_lam = self.domain.negative(lam)

        # Fused support value + support point (one call each)
        term2, v = self._model_prior_support.value_and_support_point(hilbert_residual)
        term3, w = self._data_error_support.value_and_support_point(neg_lam)

        if v is None or w is None:
            return self._mapping(lam), self._finite_difference_gradient(lam)

        # Value: term2 and term3 from fused call above
        term1 = self.domain.inner_product(lam, self._observed_data)
        f = term1 + term2 + term3

        # Subgradient
        term1_subgrad = self._observed_data
        term2_subgrad = self.domain.negative(self._G(v))
        term3_subgrad = self.domain.negative(w)
        g = self.domain.add(
            term1_subgrad,
            self.domain.add(term2_subgrad, term3_subgrad),
        )

        return f, g

    def _gradient(self, lam: "Vector") -> "Vector":
        """
        Alias for _subgradient for backward compatibility.

        Note: This function is technically a subgradient, not a gradient,
        since the dual master cost function is non-smooth due to the
        support functions. The gradient() method will call this, but
        users should prefer using subgradient() for clarity.
        """
        return self._subgradient(lam)

    def _finite_difference_gradient(
        self, lam: Vector, eps: float = 1e-6
    ) -> Vector:
        if eps <= 0:
            raise ValueError("eps must be positive")

        comps = self.domain.to_components(lam)
        grad = np.zeros_like(comps, dtype=float)

        for i in range(self.domain.dim):
            step = np.zeros_like(comps, dtype=float)
            step[i] = eps

            lam_plus = self.domain.from_components(comps + step)
            lam_minus = self.domain.from_components(comps - step)

            f_plus = self._mapping(lam_plus)
            f_minus = self._mapping(lam_minus)
            grad[i] = (f_plus - f_minus) / (2.0 * eps)

        return self.domain.from_components(grad)

    def _validation(
        self, data_space, property_space, model_space,
        G, T, model_prior_support, data_error_support,
        observed_data, q_direction,
    ) -> None:
        if not isinstance(data_space, HilbertSpace):
            raise ValueError("data_space must be a HilbertSpace")
        if not isinstance(property_space, HilbertSpace):
            raise ValueError("property_space must be a HilbertSpace")
        if not isinstance(model_space, HilbertSpace):
            raise ValueError("model_space must be a HilbertSpace")

        if not isinstance(G, LinearOperator):
            raise ValueError("G must be a LinearOperator")
        if not isinstance(T, LinearOperator):
            raise ValueError("T must be a LinearOperator")

        if G.domain != model_space or G.codomain != data_space:
            raise ValueError("G must map from model_space to data_space")
        if T.domain != model_space or T.codomain != property_space:
            raise ValueError("T must map from model_space to property_space")

        if not isinstance(model_prior_support, SupportFunction):
            raise ValueError("model_prior_support must be a SupportFunction")
        if not isinstance(data_error_support, SupportFunction):
            raise ValueError("data_error_support must be a SupportFunction")

        if model_prior_support.primal_domain != model_space:
            raise ValueError(
                "model_prior_support must be defined on model_space"
            )
        if data_error_support.primal_domain != data_space:
            raise ValueError(
                "data_error_support must be defined on data_space"
            )

        if not data_space.is_element(observed_data):
            raise ValueError("observed_data must be an element of data_space")
        if not property_space.is_element(q_direction):
            raise ValueError(
                "q_direction must be an element of property_space"
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
