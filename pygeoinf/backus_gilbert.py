"""
Module for Backus-Gilbert like methods for solving inference problems. To be done...
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .hilbert_space import HilbertSpace, Vector
from .linear_operators import LinearOperator
from .nonlinear_forms import NonLinearForm
from .convex_analysis import SupportFunction


@dataclass
class DualMasterStats:
    """Lightweight instrumentation counters for :class:`DualMasterCostFunction`.

    All timing values are in **seconds** (wall-clock, via
    :func:`time.perf_counter`).  Counters start at zero and are accumulated
    across all calls since the last
    :meth:`DualMasterCostFunction.reset_instrumentation`.
    """

    num_value_and_subgradient_calls: int = 0
    """Total calls to :meth:`~DualMasterCostFunction.value_and_subgradient`."""

    time_value_and_subgradient_s: float = 0.0
    """Cumulative wall-clock time inside ``value_and_subgradient``."""

    time_gstar_apply_s: float = 0.0
    """Cumulative time spent computing $G^* \\lambda$."""

    time_support_point_model_s: float = 0.0
    """Cumulative time in ``model_prior_support.support_point``."""

    time_support_point_data_s: float = 0.0
    """Cumulative time in ``data_error_support.support_point``."""

    time_support_value_model_s: float = 0.0
    """Cumulative time evaluating ``model_prior_support`` (scalar value)."""

    time_support_value_data_s: float = 0.0
    """Cumulative time evaluating ``data_error_support`` (scalar value)."""

    num_support_point_failures: int = 0
    """Number of calls where a support-point returned ``None``."""

    num_finite_difference_fallbacks: int = 0
    """Number of times the finite-difference gradient fallback was used."""

    time_finite_difference_s: float = 0.0
    """Cumulative time spent inside
    :meth:`~DualMasterCostFunction._finite_difference_gradient`.
    """


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
        self._model_prior_support = model_prior_support
        self._data_error_support = data_error_support
        self._observed_data = observed_data
        self._q = q_direction

        self._Tstar_q = self._T.adjoint(q_direction)

        self._stats = DualMasterStats()

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

    @property
    def instrumentation_stats(self) -> DualMasterStats:
        """Current accumulated instrumentation statistics.

        Returns a reference to the live :class:`DualMasterStats` object.
        Call :meth:`reset_instrumentation` to clear before a new experiment.
        """
        return self._stats

    def reset_instrumentation(self) -> None:
        """Reset all instrumentation counters and timers to zero."""
        self._stats = DualMasterStats()

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
        Gstar_lam = self._G.adjoint(lam)
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

        Gstar_lam = self._G.adjoint(lam)
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
        _t0_total = time.perf_counter()
        s = self._stats
        s.num_value_and_subgradient_calls += 1

        # Shared computation: G* λ
        _t0 = time.perf_counter()
        Gstar_lam = self._G.adjoint(lam)
        s.time_gstar_apply_s += time.perf_counter() - _t0

        hilbert_residual = self._model_space.subtract(self._Tstar_q, Gstar_lam)
        neg_lam = self.domain.negative(lam)

        # Support points
        _t0 = time.perf_counter()
        v = self._model_prior_support.support_point(hilbert_residual)
        s.time_support_point_model_s += time.perf_counter() - _t0

        _t0 = time.perf_counter()
        w = self._data_error_support.support_point(neg_lam)
        s.time_support_point_data_s += time.perf_counter() - _t0

        if v is None or w is None:
            s.num_support_point_failures += 1
            s.num_finite_difference_fallbacks += 1
            result = self._mapping(lam), self._finite_difference_gradient(lam)
            s.time_value_and_subgradient_s += time.perf_counter() - _t0_total
            return result

        # Value: σ_B and σ_V evaluations
        term1 = self.domain.inner_product(lam, self._observed_data)

        _t0 = time.perf_counter()
        term2 = self._model_prior_support(hilbert_residual)
        s.time_support_value_model_s += time.perf_counter() - _t0

        _t0 = time.perf_counter()
        term3 = self._data_error_support(neg_lam)
        s.time_support_value_data_s += time.perf_counter() - _t0

        f = term1 + term2 + term3

        # Subgradient
        term1_subgrad = self._observed_data
        term2_subgrad = self.domain.negative(self._G(v))
        term3_subgrad = self.domain.negative(w)
        g = self.domain.add(
            term1_subgrad,
            self.domain.add(term2_subgrad, term3_subgrad),
        )

        s.time_value_and_subgradient_s += time.perf_counter() - _t0_total
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

        _t0 = time.perf_counter()
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

        self._stats.time_finite_difference_s += time.perf_counter() - _t0
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
