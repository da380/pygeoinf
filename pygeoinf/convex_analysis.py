from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Any

from pygeoinf.linear_operators import LinearOperator
from pygeoinf.nonlinear_forms import NonLinearForm

import numpy as np

if TYPE_CHECKING:
    from .hilbert_space import HilbertSpace, Vector


class SupportFunction(NonLinearForm, ABC):
    """
    Support function of a closed convex set S ⊆ H:

        h_S(q) = sup_{x ∈ S} ⟨q, x⟩,    q ∈ H

    In a Hilbert space, we identify H ≅ H* via the Riesz map, so the
    functional is defined directly on the primal space H.

    The set S is uniquely recovered as:
        S = {x : ⟨q, x⟩ ≤ h_S(q) for all q ∈ H}
    """

    def __init__(self, primal_domain: "HilbertSpace") -> None:
        """
        Args:
            primal_domain: The Hilbert space H where the convex set lives.
        """
        self._primal = primal_domain
        super().__init__(
            primal_domain, self._mapping, subgradient=self._subgradient_impl
        )

    @property
    def primal_domain(self) -> "HilbertSpace":
        """The Hilbert space H in which the underlying convex set lives."""
        return self._primal

    @abstractmethod
    def _mapping(self, q: object) -> float:
        """
        Evaluate h(q) for q ∈ H.
        Subclasses must implement this method.
        """
        raise NotImplementedError

    def support_point(self, q: "Vector") -> Optional["Vector"]:
        """
        Optional: return x*(q) ∈ argmax_{x∈S} ⟨q, x⟩ if available/computable.
        Default: not provided (returns None). This is the subgradient of h_S at
        q.
        """
        return None

    def _subgradient_impl(self, q: "Vector") -> "Vector":
        """
        Return a subgradient of the support function at q.

        For support functions, any maximizer x*(q) ∈ argmax_{x∈S} ⟨q, x⟩
        is a subgradient. This method delegates to support_point(q).

        Raises:
            NotImplementedError: If a support point is not available.
        """
        point = self.support_point(q)
        if point is None:
            raise NotImplementedError(
                "Support point not available; subgradient cannot be computed."
            )
        return point

    def subgradient(self, q: "Vector") -> "Vector":
        """Return a subgradient of the support function at q."""
        return self._subgradient_impl(q)


class BallSupportFunction(SupportFunction):
    """
    Support function of a closed ball B(c, r) = {x : ||x - c|| ≤ r}:

        h(q) = ⟨q, c⟩ + r ||q||
    """

    def __init__(
        self, primal_domain: "HilbertSpace", center: "Vector", radius: float
    ) -> None:
        super().__init__(primal_domain)
        self._center = center
        self._radius = float(radius)

    def _mapping(self, q: "Vector") -> float:
        H = self.primal_domain
        center_term = H.inner_product(q, self._center)
        q_norm = H.norm(q)
        return center_term + self._radius * q_norm

    def support_point(self, q: "Vector") -> Optional["Vector"]:
        """Return x* = c + r * (q / ||q||) achieving the supremum."""
        H = self.primal_domain
        n = H.norm(q)
        if n < 1e-14:
            # q ≈ 0: any point in the ball is a maximizer; return center
            return self._center
        # x* = c + r * (q / ||q||)
        return H.add(self._center, H.multiply(self._radius / n, q))


class EllipsoidSupportFunction(SupportFunction):
    """
    Support function of an ellipsoid E(c, r, A) defined by:

        E = {x : ⟨A(x-c), (x-c)⟩ ≤ r²}   with A SPD

    Then:
        h(q) = ⟨q, c⟩ + r ||A^{-1/2} q||

    Args:
        primal_domain: The Hilbert space H.
        center: The center c of the ellipsoid.
        radius: The radius r.
        shape_operator: The SPD operator A.
        inverse_operator: A^{-1}. Required for computing h(q) and support_point.
        inverse_sqrt_operator: A^{-1/2}. Required for computing h(q).

    Note:
        If inverse operators are not provided, the support function cannot be
        evaluated and support_point will return None.
    """

    def __init__(
        self,
        primal_domain: "HilbertSpace",
        center: "Vector",
        radius: float,
        shape_operator: LinearOperator,
        inverse_operator: Optional[LinearOperator] = None,
        inverse_sqrt_operator: Optional[LinearOperator] = None,
    ) -> None:
        super().__init__(primal_domain)
        self._center = center
        self._radius = float(radius)
        self._A = shape_operator
        self._A_inv = inverse_operator
        self._A_inv_sqrt = inverse_sqrt_operator

    def _mapping(self, q: "Vector") -> float:
        if self._A_inv_sqrt is None:
            raise ValueError(
                "inverse_sqrt_operator must be provided to evaluate the support function. "
                "Pass A^{-1/2} when constructing EllipsoidSupportFunction."
            )

        H = self.primal_domain
        center_term = H.inner_product(q, self._center)
        # Compute ||A^{-1/2} q||
        w = self._A_inv_sqrt(q)
        shape_norm = H.norm(w)
        return center_term + self._radius * shape_norm

    def support_point(self, q: "Vector") -> Optional["Vector"]:
        """
        Return x* = c + r * (A^{-1} q) / ||A^{-1/2} q|| achieving the supremum.

        For an ellipsoid E(c, r, A), the extreme point in direction q is found
        by transforming q through the inverse metric A^{-1} and normalizing.

        Returns None if inverse_operator was not provided.
        """
        if self._A_inv is None:
            return None

        H = self.primal_domain
        A_inv_q = self._A_inv(q)

        # Compute ||A^{-1/2} q|| = sqrt(⟨q, A^{-1} q⟩)
        q_term_squared = H.inner_product(q, A_inv_q)
        if q_term_squared < 0:
            q_term_squared = 0.0  # Numerical noise
        norm_term = q_term_squared ** 0.5

        if norm_term < 1e-14:
            # q ≈ 0: center is a maximizer
            return self._center

        # x* = c + r * (A^{-1} q) / ||A^{-1/2} q||
        scaled = H.multiply(self._radius / norm_term, A_inv_q)
        return H.add(self._center, scaled)


class HalfSpaceSupportFunction(NonLinearForm):
    r"""
    Support function of a (closed) half-space in a Hilbert space H.

    We support two conventions:

      (<=)  H = { x ∈ H : ⟨a, x⟩ ≤ b }
      (>=)  H = { x ∈ H : ⟨a, x⟩ ≥ b }  which is equivalent to { x : ⟨-a, x⟩ ≤ -b }.

    Mathematical facts (extended-real-valued):
      For H = {x : ⟨a, x⟩ ≤ b} with a ≠ 0,

        σ_H(q) = sup_{x∈H} ⟨q, x⟩
              = { α b,  if q = α a with α ≥ 0,
                  +∞,   otherwise. }

      For H = {x : ⟨a, x⟩ ≥ b},

        σ_H(q) = { α b,  if q = α a with α ≤ 0,
                  +∞,   otherwise. }

    Notes:
      - Half-spaces are unbounded, so σ_H is typically +∞.
      - This implementation returns float('inf') for unbounded directions.
      - A support point is not unique when σ_H(q) is finite; the maximizers form
        the boundary hyperplane ⟨a, x⟩ = b. We optionally return the minimum-norm
        boundary point as a canonical representative.
    """

    def __init__(
        self,
        primal_domain: "HilbertSpace",
        normal_vector: "Vector",
        offset: float,
        inequality_type: str = "<=",
        *,
        parallel_rtol: float = 1e-12,
        parallel_atol: float = 1e-14,
        return_min_norm_support_point: bool = True,
    ) -> None:
        self._H = primal_domain
        self._a = normal_vector
        self._b = float(offset)

        if inequality_type not in ("<=", ">="):
            raise ValueError("inequality_type must be '<=' or '>='.")
        self._ineq = inequality_type

        self._rtol = float(parallel_rtol)
        self._atol = float(parallel_atol)
        self._return_min_norm = bool(return_min_norm_support_point)

        # Basic validation
        a_norm_sq = self._H.inner_product(self._a, self._a)
        if a_norm_sq <= 0:
            raise ValueError("normal_vector must be nonzero (a ≠ 0).")

        super().__init__(primal_domain, self._mapping, subgradient=self._subgradient_impl)

    @property
    def normal_vector(self) -> "Vector":
        return self._a

    @property
    def offset(self) -> float:
        return self._b

    @property
    def inequality_type(self) -> str:
        return self._ineq

    def _decompose_parallel(self, q: "Vector") -> tuple[float, float]:
        """
        Return (alpha, resid_norm), where alpha is the best scalar such that
        q_parallel = alpha * a (least-squares in Hilbert sense) and
        resid_norm = || q - alpha a ||.

        In a Hilbert space:
          alpha = <q,a>/<a,a>.
        """
        H = self._H
        aa = H.inner_product(self._a, self._a)
        qa = H.inner_product(q, self._a)
        alpha = qa / aa
        q_parallel = H.multiply(alpha, self._a)
        resid = H.subtract(q, q_parallel)
        resid_norm = H.norm(resid)
        return float(alpha), float(resid_norm)

    def _is_parallel(self, q: "Vector", alpha: float, resid_norm: float) -> bool:
        """
        Decide if q is (numerically) parallel to a by checking ||q - alpha a|| small.
        """
        H = self._H
        q_norm = H.norm(q)
        # relative-to-scale tolerance
        return resid_norm <= max(self._atol, self._rtol * max(1.0, q_norm))

    def _alpha_sign_ok(self, alpha: float) -> bool:
        """
        Check the half-space-specific sign restriction on alpha.
        For ⟨a,x⟩ ≤ b: require alpha ≥ 0.
        For ⟨a,x⟩ ≥ b: require alpha ≤ 0.
        """
        if self._ineq == "<=":
            return alpha >= -self._atol
        else:
            return alpha <= self._atol

    def _mapping(self, q: "Vector") -> float:
        """
        Extended-real support function value: returns +∞ when unbounded.
        """
        alpha, resid_norm = self._decompose_parallel(q)

        # Finite iff q is parallel to a AND alpha has the correct sign
        if self._is_parallel(q, alpha, resid_norm) and self._alpha_sign_ok(alpha):
            return alpha * self._b

        return float("inf")

    def support_point(self, q: "Vector") -> Optional["Vector"]:
        """
        Return a canonical maximizer when σ_H(q) is finite.

        When finite, the maximizers are all x with ⟨a,x⟩ = b (boundary hyperplane).
        If return_min_norm_support_point=True, we return the minimum-norm boundary point:
            x_min = (b / ||a||^2) a.
        Otherwise return None (non-unique support set).
        """
        if not self._return_min_norm:
            return None

        val = self._mapping(q)
        if not np.isfinite(val):
            return None

        H = self._H
        aa = H.inner_product(self._a, self._a)
        coeff = self._b / aa
        return H.multiply(coeff, self._a)

    def _subgradient_impl(self, q: "Vector") -> "Vector":
        """
        For support functions, any maximizer is a subgradient, when the value is finite.
        We return a canonical maximizer if enabled; otherwise raise.
        """
        x_star = self.support_point(q)
        if x_star is None:
            raise NotImplementedError(
                "Subgradient not available: either σ_H(q)=+∞ or support point is non-unique "
                "and return_min_norm_support_point=False."
            )
        return x_star