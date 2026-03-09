from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Callable

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

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def callable(
        cls,
        primal_domain: "HilbertSpace",
        mapping: "Callable[[Vector], float]",
        support_point: "Optional[Callable[[Vector], Vector]]" = None,
    ) -> "CallableSupportFunction":
        """Construct a support function from a user-supplied callable.

        Args:
            primal_domain: The Hilbert space H.
            mapping: A callable ``q -> float`` that evaluates $h(q)$.
            support_point: An optional callable ``q -> Vector`` returning
                $x^*(q) \\in \\arg\\max_{x \\in C} \\langle q, x \\rangle$.
                When provided, ``subgradient(q)`` delegates to it.

        Returns:
            A :class:`CallableSupportFunction` instance.
        """
        return CallableSupportFunction(primal_domain, mapping, support_point_fn=support_point)

    @classmethod
    def point(
        cls,
        primal_domain: "HilbertSpace",
        point: "Vector",
    ) -> "PointSupportFunction":
        """Construct the support function of the singleton set $\\{p\\}$.

        For a fixed point $p \\in H$, the support function is
        $h(q) = \\langle q, p \\rangle$.

        Args:
            primal_domain: The Hilbert space H containing $p$.
            point: The fixed point $p$.

        Returns:
            A :class:`PointSupportFunction` instance.
        """
        return PointSupportFunction(primal_domain, point)

    # ------------------------------------------------------------------
    # Algebraic composition methods (Phase 2)
    # ------------------------------------------------------------------

    def image(self, operator: "LinearOperator") -> "LinearImageSupportFunction":
        r"""Return the support function of the linear image $A(C)$.

        For a bounded linear operator $A$ with ``A.domain == self.primal_domain``,
        returns the support function of the image set $A(C)$, which lives in
        ``A.codomain``.  Its value is $h_{A(C)}(q) = h_C(A^* q)$.

        Args:
            operator: A bounded linear operator $A: H \to K$ with
                ``operator.domain`` equal to ``self.primal_domain``.

        Returns:
            A :class:`LinearImageSupportFunction` on ``operator.codomain``.

        Raises:
            ValueError: If ``operator.domain != self.primal_domain``.
        """
        return LinearImageSupportFunction(self, operator)

    def translate(self, point: "Vector") -> "MinkowskiSumSupportFunction":
        r"""Return the support function of the translated set $C + p$.

        Translation by $p \in H$ satisfies $h_{C+p}(q) = h_C(q) + \langle q, p \rangle$.

        Args:
            point: The translation vector $p \in H$ (same space as ``primal_domain``).

        Returns:
            A :class:`MinkowskiSumSupportFunction` on the same space.
        """
        return MinkowskiSumSupportFunction(self, PointSupportFunction(self.primal_domain, point))

    def scale(self, alpha: float) -> "ScaledSupportFunction":
        r"""Return the support function of the scaled set $\alpha C$.

        Scaling satisfies $h_{\alpha C}(q) = \alpha\, h_C(q)$ for $\alpha \geq 0$.

        Args:
            alpha: A nonnegative scalar.

        Returns:
            A :class:`ScaledSupportFunction` on the same space.

        Raises:
            ValueError: If ``alpha < 0``.
        """
        return ScaledSupportFunction(self, alpha)

    # ------------------------------------------------------------------
    # Arithmetic operator overrides
    # ------------------------------------------------------------------

    def __add__(self, other: object) -> "MinkowskiSumSupportFunction":
        """Return the Minkowski-sum support function $h_C + h_D$.

        Both operands must be :class:`SupportFunction` instances with the
        same ``primal_domain``.

        Raises:
            TypeError: If ``other`` is not a :class:`SupportFunction`.
            ValueError: If ``primal_domain`` values differ.
        """
        if not isinstance(other, SupportFunction):
            raise TypeError(
                f"unsupported operand type(s) for +: 'SupportFunction' and {type(other).__name__!r}. "
                "Both operands must be SupportFunction instances to preserve support-function algebra."
            )
        return MinkowskiSumSupportFunction(self, other)

    def __mul__(self, alpha: object) -> "ScaledSupportFunction":
        """Return the scaled support function $\\alpha h_C$.

        Args:
            alpha: A nonnegative scalar.

        Raises:
            TypeError: If ``alpha`` is not a real number.
            ValueError: If ``alpha < 0``.
        """
        if not isinstance(alpha, (int, float, np.floating, np.integer)):
            raise TypeError(
                f"unsupported operand type(s) for *: 'SupportFunction' and {type(alpha).__name__!r}. "
                "Scalar must be a real number."
            )
        return ScaledSupportFunction(self, float(alpha))

    def __rmul__(self, alpha: object) -> "ScaledSupportFunction":
        """Return the scaled support function $\\alpha h_C$ (reversed)."""
        return self.__mul__(alpha)


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


# ---------------------------------------------------------------------------
# Phase-1 constructors: CallableSupportFunction and PointSupportFunction
# ---------------------------------------------------------------------------


class CallableSupportFunction(SupportFunction):
    r"""Support function defined by a user-provided callable.

    Wraps an arbitrary callable $q \mapsto h(q)$ as a :class:`SupportFunction`.
    Optionally accepts a second callable $q \mapsto x^*(q)$ that returns the
    support point (subgradient of $h$ at $q$).

    Args:
        primal_domain: The Hilbert space $H$ on which $h$ is defined.
        fn: A callable ``fn(q) -> float`` computing the support value $h(q)$.
        support_point_fn: An optional callable ``support_point_fn(q) -> Vector``
            returning $x^*(q) \in \arg\max_{x \in C} \langle q, x \rangle$.
            When provided, :meth:`support_point` delegates to it and
            :meth:`subgradient` returns the result.
            When ``None``, :meth:`support_point` returns ``None`` and
            :meth:`subgradient` raises :exc:`NotImplementedError`.

    Example::

        fn = lambda q: float(np.linalg.norm(q))          # L2-ball support
        sp = lambda q: q / np.linalg.norm(q)             # support point
        h = CallableSupportFunction(space, fn, support_point_fn=sp)
    """

    def __init__(
        self,
        primal_domain: "HilbertSpace",
        fn: "Callable[[Vector], float]",
        support_point_fn: "Optional[Callable[[Vector], Vector]]" = None,
    ) -> None:
        super().__init__(primal_domain)
        self._fn = fn
        self._support_point_fn = support_point_fn

    def _mapping(self, q: "Vector") -> float:
        return float(self._fn(q))

    def support_point(self, q: "Vector") -> "Optional[Vector]":
        """Return $x^*(q)$ via the user-supplied callback, or ``None``."""
        if self._support_point_fn is None:
            return None
        return self._support_point_fn(q)


class PointSupportFunction(SupportFunction):
    r"""Support function of the singleton set $\{p\}$.

    For a fixed point $p \in H$, the support function of the singleton
    set $\{p\}$ is

    .. math::

        h_{\{p\}}(q) = \langle q, p \rangle, \quad q \in H.

    The support point is always $p$ (the unique element of the set),
    so :meth:`subgradient` is available for all query directions.

    Args:
        primal_domain: The Hilbert space $H$ containing the point $p$.
        point: The fixed point $p$.

    Example::

        p = np.array([1.0, 2.0])
        h = PointSupportFunction(space, p)
        h(np.array([3.0, -1.0]))   # returns 3*1 + (-1)*2 = 1.0
    """

    def __init__(
        self,
        primal_domain: "HilbertSpace",
        point: "Vector",
    ) -> None:
        super().__init__(primal_domain)
        self._point = point

    def _mapping(self, q: "Vector") -> float:
        return float(self.primal_domain.inner_product(q, self._point))

    def support_point(self, q: "Vector") -> "Vector":
        """Return $p$ — the unique maximiser for any query direction."""
        return self._point


# ---------------------------------------------------------------------------
# Phase-2 algebraic combinators
# ---------------------------------------------------------------------------


class LinearImageSupportFunction(SupportFunction):
    r"""Support function of the linear image $A(C)$ of a convex set $C$.

    For a convex set $C \subseteq H$ with support function $h_C$ and a
    bounded linear operator $A: H \to K$, the support function of the
    image $A(C) \subseteq K$ is

    .. math::

        h_{A(C)}(q) = h_C(A^* q), \quad q \in K,

    where $A^*: K \to H$ is the Hilbert-space adjoint of $A$.

    Args:
        base: The support function $h_C$ of the base set $C \subseteq H$.
            Its ``primal_domain`` must equal ``operator.domain``.
        operator: A bounded linear operator $A: H \to K$.
            ``operator.domain`` must equal ``base.primal_domain``.

    Raises:
        ValueError: If ``operator.domain`` does not equal ``base.primal_domain``.

    Note:
        The ``primal_domain`` of the returned object is ``operator.codomain``
        (the space $K$ where the image $A(C)$ lives).

    Note:
        Phase 2: :meth:`support_point` returns ``None``.
        Support-point propagation is deferred to Phase 3.
    """

    def __init__(
        self,
        base: "SupportFunction",
        operator: "LinearOperator",
    ) -> None:
        if operator.domain is not base.primal_domain:
            raise ValueError(
                "operator.domain must equal base.primal_domain. "
                f"Got operator.domain={operator.domain!r}, "
                f"base.primal_domain={base.primal_domain!r}."
            )
        super().__init__(operator.codomain)
        self._base = base
        self._operator = operator
        self._adjoint = operator.adjoint

    def _mapping(self, q: "Vector") -> float:
        return float(self._base(self._adjoint(q)))


class MinkowskiSumSupportFunction(SupportFunction):
    r"""Support function of the Minkowski sum $C \oplus D$.

    For two convex sets $C, D \subseteq H$ with support functions
    $h_C$ and $h_D$ on the same Hilbert space $H$, the support function
    of their Minkowski sum $C \oplus D = \{c + d : c \in C,\, d \in D\}$
    is

    .. math::

        h_{C \oplus D}(q) = h_C(q) + h_D(q), \quad q \in H.

    Args:
        left: Support function $h_C$.
        right: Support function $h_D$.
            ``right.primal_domain`` must equal ``left.primal_domain``.

    Raises:
        ValueError: If ``left.primal_domain`` and ``right.primal_domain`` differ.

    Note:
        Phase 2: :meth:`support_point` returns ``None``.
        Support-point propagation is deferred to Phase 3.
    """

    def __init__(
        self,
        left: "SupportFunction",
        right: "SupportFunction",
    ) -> None:
        if left.primal_domain is not right.primal_domain:
            raise ValueError(
                "Both summands must share the same primal_domain. "
                f"Got left.primal_domain={left.primal_domain!r}, "
                f"right.primal_domain={right.primal_domain!r}."
            )
        super().__init__(left.primal_domain)
        self._left = left
        self._right = right

    def _mapping(self, q: "Vector") -> float:
        return float(self._left(q)) + float(self._right(q))


class ScaledSupportFunction(SupportFunction):
    r"""Support function of a nonnegatively scaled convex set $\alpha C$.

    For a convex set $C \subseteq H$ with support function $h_C$ and a
    scalar $\alpha \geq 0$, the support function of the scaled set is

    .. math::

        h_{\alpha C}(q) = \alpha\, h_C(q), \quad q \in H.

    When $\alpha = 0$ the set $0 \cdot C = \{0\}$ and $h(q) = 0$ for all $q$.

    Args:
        base: The support function $h_C$.
        alpha: A nonnegative scalar.

    Raises:
        ValueError: If ``alpha < 0``.

    Note:
        Phase 2: :meth:`support_point` returns ``None``.
        Support-point propagation is deferred to Phase 3.
    """

    def __init__(
        self,
        base: "SupportFunction",
        alpha: float,
    ) -> None:
        alpha = float(alpha)
        if alpha < 0.0:
            raise ValueError(
                f"alpha must be nonnegative for support-function scaling; got alpha={alpha}."
            )
        super().__init__(base.primal_domain)
        self._base = base
        self._alpha = alpha

    def _mapping(self, q: "Vector") -> float:
        if self._alpha == 0.0:
            return 0.0
        return self._alpha * float(self._base(q))
