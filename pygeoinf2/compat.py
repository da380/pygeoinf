"""
Presenting a v1 object to the v2 core.

The point of this module is to get the existing concrete spaces — the sphere,
circle, torus, plane and line, roughly 9.5k lines of numerically delicate
transform code — usable against the new core immediately, so that the core is
exercised on real problems long before those spaces are rewritten natively.

The mapping is short, because the two designs agree on more than they differ:

===========================  ==========================================
v1                           v2
===========================  ==========================================
``to_components``            ``to_components``
``to_dual`` then components  ``apply_gram``   (see below)
``from_dual``                ``solve_gram``
``inner_product``            ``inner_product`` (delegated, not rederived)
``operator.adjoint``         ``adjoint``
``LinearForm.components``    ``from_derivative_components``
``space.random``             ``random`` -- but NOT ``white_noise``
===========================  ==========================================

The Gram matrix is the interesting one. In v1 the duality pairing is
``<xp, x> == dot(dual.to_components(xp), to_components(x))`` and the inner
product is ``(x, y) == <to_dual(x), y>``, so the components of ``to_dual(x)``
are exactly ``G c_x``. The mass matrix does not have to be dug out of the
space; it is already there, wearing a different name.

``white_noise`` is deliberately *not* delegated. v1 draws standard normal
components, giving covariance ``G`` rather than the identity; the adapted
space uses the corrected draw. See DESIGN.md section 9.

Nothing here is imported by ``pygeoinf2/__init__.py``: importing this module
is what pulls in v1.
"""

from __future__ import annotations

import warnings
from typing import Any, Hashable

import numpy as np
from numpy.random import Generator

from .algebra.operators import LinearFunctional, LinearOperator
from .algebra.spaces import CoordinateSpace, _resolve_rng
from .traits import Traits

__all__ = [
    "AdaptedSpace",
    "AdaptedOperator",
    "adapt_space",
    "adapt_operator",
    "adapt_form",
]


class _V1SpaceKey:
    """A hashable handle on a v1 space, which is itself unhashable.

    v1 declares ``__eq__`` on ``HilbertSpace`` and never defines ``__hash__``,
    so Python sets ``__hash__ = None`` and no v1 space can be a dict key. This
    wrapper compares exactly, by delegating to v1's own equality, and hashes
    coarsely on the dimension.

    That is legal and sufficient: the hash contract requires only that equal
    objects hash equally, and equal spaces necessarily share a dimension.
    Distinct spaces of the same dimension collide, which costs nothing when a
    program holds a handful of spaces.
    """

    __slots__ = ("space",)

    def __init__(self, space: Any) -> None:
        self.space = space

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _V1SpaceKey):
            return NotImplemented
        return bool(self.space == other.space)

    def __hash__(self) -> int:
        return hash(self.space.dim)

    def __repr__(self) -> str:
        return f"<v1 {type(self.space).__name__} dim={self.space.dim}>"


class AdaptedSpace(CoordinateSpace):
    """A v1 ``HilbertSpace`` presented as a v2 ``CoordinateSpace``."""

    def __init__(self, space: Any, /, *, gram: str = "auto") -> None:
        """
        Args:
            space: the v1 space to wrap.
            gram: how to factorise the Gram matrix for ``white_noise``.
                ``"diagonal"`` assumes a diagonal Gram and costs one
                application; ``"dense"`` forms and factorises it, which is
                ``O(dim^2)`` in memory and unusable on a large space;
                ``"auto"`` probes for diagonality and picks. See
                :meth:`_detect_diagonal_gram`.
        """
        if gram not in ("auto", "diagonal", "dense"):
            raise ValueError(f"Unknown gram strategy {gram!r}.")
        self._space = space
        self._gram_strategy = gram

    @property
    def v1_space(self) -> Any:
        """The wrapped v1 space."""
        return self._space

    # ----------------------------------------------------------------- #
    #                             Identity                              #
    # ----------------------------------------------------------------- #

    @property
    def dim(self) -> int:
        """The dimension of the wrapped space."""
        return self._space.dim

    def _key(self) -> Hashable:
        return _V1SpaceKey(self._space)

    def __repr__(self) -> str:
        return f"AdaptedSpace({type(self._space).__name__}, dim={self.dim})"

    # ----------------------------------------------------------------- #
    #                       Delegated vector algebra                    #
    # ----------------------------------------------------------------- #

    def zero(self) -> Any:
        """The wrapped space's zero vector. v1 exposes it as a property."""
        return self._space.zero

    def copy(self, x: Any) -> Any:
        """An independent copy, delegated to v1."""
        return self._space.copy(x)

    def inner_product(self, x: Any, y: Any) -> float:
        """Delegated to v1, not rederived from the Gram matrix."""
        # Delegated rather than rederived from the Gram: v1's implementations
        # are often specialised, and going through the Gram would be slower and
        # would lose whatever accuracy the specialisation buys.
        return float(self._space.inner_product(x, y))

    def axpy(self, a: float, x: Any, y: Any) -> Any:
        """``y += a * x``. v1 mutates and returns None; v2 returns the result."""
        # v1 mutates and returns None; v2 requires the result.
        self._space.axpy(a, x, y)
        return y

    def scale_inplace(self, a: float, x: Any) -> Any:
        """``x *= a``. v1's ``ax``, adapted to return the result."""
        self._space.ax(a, x)
        return x

    def random(self, *, rng: Generator | None = None) -> Any:
        """An arbitrary random vector, drawn from an explicit generator.

        Not delegated: v1's ``random`` ignores any generator and uses NumPy's
        legacy global state.
        """
        # v1's random() ignores any generator and uses the legacy global state.
        return self._space.from_components(_resolve_rng(rng).standard_normal(self.dim))

    # ----------------------------------------------------------------- #
    #                            Coordinates                            #
    # ----------------------------------------------------------------- #

    def to_components(self, x: Any) -> np.ndarray:
        """The wrapped space's components."""
        return self._space.to_components(x)

    def from_components(self, c: np.ndarray) -> Any:
        """The vector with the given components."""
        return self._space.from_components(c)

    def apply_gram(self, c: np.ndarray) -> np.ndarray:
        """``G c``, read off v1's Riesz map.

        ``to_dual`` maps a vector to the functional pairing with it, whose
        components are by definition ``G c``.
        """
        x = self._space.from_components(c)
        return self._space.dual.to_components(self._space.to_dual(x))

    def solve_gram(self, c: np.ndarray) -> np.ndarray:
        """``G^-1 c``, read off v1's inverse Riesz map."""
        xp = self._space.dual.from_components(c)
        return self._space.to_components(self._space.from_dual(xp))

    # ----------------------------------------------------------------- #
    #                            White noise                            #
    # ----------------------------------------------------------------- #

    @property
    def has_diagonal_metric(self) -> bool:
        """Whether the wrapped space's Gram matrix is diagonal.

        Detected by the same probe as :meth:`_diagonal_gram`, so it costs a
        couple of Gram applications rather than forming the matrix.
        """
        return self._diagonal_gram() is not None

    def white_noise_components(self, *, rng: Generator | None = None) -> np.ndarray:
        """Components drawn from ``N(0, G^-1)``.

        Not delegated to v1, which draws ``N(0, I)`` components and so produces
        covariance ``G`` on any space whose basis is not orthonormal — that is,
        on every Sobolev space in the library.
        """
        rng = _resolve_rng(rng)
        diagonal = self._diagonal_gram()
        if diagonal is not None:
            return rng.standard_normal(self.dim) / np.sqrt(diagonal)
        if self._gram_strategy == "auto":
            warnings.warn(
                f"{self!r} has a non-diagonal Gram matrix, so white noise "
                f"requires forming and factorising it densely "
                f"({self.dim} x {self.dim}). Pass gram='dense' to silence this, "
                f"or implement white_noise_components natively.",
                RuntimeWarning,
                stacklevel=2,
            )
        return super().white_noise_components(rng=rng)

    def _diagonal_gram(self) -> np.ndarray | None:
        """The Gram diagonal, or None when the Gram is not diagonal.

        Cached. Under ``gram="auto"`` this probes rather than forming the
        matrix, because the spaces this adapter exists for are large enough
        that an ``O(dim^2)`` build is not an option: a Sobolev space on the
        sphere at degree 128 has ``dim`` above 16000.
        """
        if self._gram_strategy == "dense":
            return None
        cached = self.__dict__.get("_diagonal_gram_cache", _UNSET)
        if cached is not _UNSET:
            return cached

        diagonal = self.apply_gram(np.ones(self.dim))
        if self._gram_strategy == "auto":
            diagonal = diagonal if self._probe_is_diagonal(diagonal) else None
        self.__dict__["_diagonal_gram_cache"] = diagonal
        return diagonal

    def _probe_is_diagonal(self, diagonal: np.ndarray, /, *, probes: int = 2) -> bool:
        """Test ``G s == diagonal * s`` for random ``s``.

        If ``G`` is diagonal this holds for every ``s``. If it is not, the
        probe vectors would have to lie in the kernel of ``G - diag(G 1)``,
        which for random vectors is a probability-zero coincidence. Two probes
        cost two Gram applications; forming the matrix costs ``dim`` of them.
        """
        rng = np.random.default_rng(0)  # fixed, so the answer is reproducible
        for _ in range(probes):
            s = rng.standard_normal(self.dim)
            if not np.allclose(
                self.apply_gram(s), diagonal * s, rtol=1e-10, atol=1e-12
            ):
                return False
        return True


_UNSET: Any = object()


class AdaptedOperator(LinearOperator):
    """A v1 ``LinearOperator`` presented as a v2 ``LinearOperator``.

    Traits are not inferred — v1 records none — so anything a solver needs to
    know must be claimed here and verified with
    ``pygeoinf2.testing.check_traits``.
    """

    def __init__(
        self,
        operator: Any,
        /,
        *,
        domain: CoordinateSpace | None = None,
        codomain: CoordinateSpace | None = None,
        traits: Traits = Traits.NONE,
    ) -> None:
        super().__init__(
            adapt_space(operator.domain) if domain is None else domain,
            adapt_space(operator.codomain) if codomain is None else codomain,
            traits=traits,
        )
        self._operator = operator
        # v1 rebuilds the adjoint on every attribute access.
        self._v1_adjoint = operator.adjoint

    @property
    def v1_operator(self) -> Any:
        """The wrapped v1 operator."""
        return self._operator

    def _value(self, x: Any) -> Any:
        return self._operator(x)

    def _adjoint_value(self, y: Any) -> Any:
        return self._v1_adjoint(y)

    def __repr__(self) -> str:
        return (
            f"AdaptedOperator({type(self._operator).__name__}: "
            f"{self.domain!r} -> {self.codomain!r})"
        )


def adapt_space(space: Any, /, *, gram: str = "auto") -> CoordinateSpace:
    """Present a v1 space as a v2 ``CoordinateSpace``."""
    if isinstance(space, CoordinateSpace):
        return space
    return AdaptedSpace(space, gram=gram)


def adapt_operator(
    operator: Any,
    /,
    *,
    domain: CoordinateSpace | None = None,
    codomain: CoordinateSpace | None = None,
    traits: Traits = Traits.NONE,
) -> LinearOperator:
    """Present a v1 linear operator as a v2 ``LinearOperator``."""
    if isinstance(operator, LinearOperator):
        return operator
    return AdaptedOperator(operator, domain=domain, codomain=codomain, traits=traits)


def adapt_form(
    form: Any, /, *, domain: CoordinateSpace | None = None
) -> LinearFunctional:
    """Present a v1 ``LinearForm`` as a v2 ``LinearFunctional``.

    A v1 form stores its action as a component array in the **derivative**
    convention — ``<f, x> == dot(components, c_x)`` — so this is exactly
    ``from_derivative_components``. The representer, which v1 obtains with
    ``from_dual``, is then ``f.adjoint(1.0)``. See DESIGN.md section 5.6.
    """
    space = adapt_space(form.domain) if domain is None else domain
    return LinearFunctional.from_derivative_components(space, form.components)
