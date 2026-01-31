from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Any

from pygeoinf.linear_operators import LinearOperator
from pygeoinf.nonlinear_forms import NonLinearForm

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
        super().__init__(primal_domain, self._mapping)

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