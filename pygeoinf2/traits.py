"""
Structural properties of linear operators, and how they survive the algebra.

Traits record *mathematical* properties that a numerical method dispatches on.
Representational structure (dense, sparse, diagonal, low rank) is carried by
classes instead, because it comes with data and extra API; a trait carries no
data. See DESIGN.md section 4.

Traits are claims made by whoever constructs an operator. They are not
verified here. ``pygeoinf2.testing.check_traits`` verifies them numerically.
"""

from __future__ import annotations

from enum import Flag, auto

__all__ = [
    "Traits",
    "close",
    "sum_traits",
    "scale_traits",
    "adjoint_traits",
    "compose_traits",
    "inverse_traits",
    "congruence_traits",
    "gramian_traits",
]


class Traits(Flag):
    """Mathematical properties of a linear operator.

    Positive-(semi)definiteness implies self-adjointness, following the real
    convention used throughout the library: there are no complex spaces, so a
    positive operator is symmetric by definition rather than by assumption.
    """

    NONE = 0
    SELF_ADJOINT = auto()
    POSITIVE_SEMIDEFINITE = auto()
    POSITIVE_DEFINITE = auto()
    INVERTIBLE = auto()
    ISOMETRY = auto()
    UNITARY = auto()
    IDEMPOTENT = auto()


_T = Traits


# Implication rules, applied to a fixed point so that no inconsistent trait set
# is representable. Each entry is (required, implied).
_IMPLICATIONS: list[tuple[Traits, Traits]] = [
    (_T.POSITIVE_DEFINITE, _T.POSITIVE_SEMIDEFINITE | _T.INVERTIBLE | _T.SELF_ADJOINT),
    (_T.POSITIVE_SEMIDEFINITE, _T.SELF_ADJOINT),
    (_T.UNITARY, _T.ISOMETRY | _T.INVERTIBLE),
    (_T.IDEMPOTENT | _T.SELF_ADJOINT, _T.POSITIVE_SEMIDEFINITE),
    (_T.POSITIVE_SEMIDEFINITE | _T.INVERTIBLE, _T.POSITIVE_DEFINITE),
    (_T.ISOMETRY | _T.INVERTIBLE, _T.UNITARY),
]


def close(traits: Traits) -> Traits:
    """Add every trait implied by those present, to a fixed point."""
    result = traits
    changed = True
    while changed:
        changed = False
        for required, implied in _IMPLICATIONS:
            if required & result == required and implied & result != implied:
                result |= implied
                changed = True
    return result


def _has(traits: Traits, member: Traits) -> bool:
    return traits & member == member


def sum_traits(a: Traits, b: Traits) -> Traits:
    """Traits of ``A + B``."""
    a, b = close(a), close(b)
    result = _T.NONE
    if _has(a, _T.SELF_ADJOINT) and _has(b, _T.SELF_ADJOINT):
        result |= _T.SELF_ADJOINT
    if _has(a, _T.POSITIVE_SEMIDEFINITE) and _has(b, _T.POSITIVE_SEMIDEFINITE):
        result |= _T.POSITIVE_SEMIDEFINITE
        # definiteness needs only one of the two summands to be definite
        if _has(a, _T.POSITIVE_DEFINITE) or _has(b, _T.POSITIVE_DEFINITE):
            result |= _T.POSITIVE_DEFINITE
    return close(result)


def scale_traits(traits: Traits, alpha: float, *, square: bool) -> Traits:
    """Traits of ``alpha * A``.

    Args:
        traits: what ``A`` claims.
        alpha: the scalar. Its *sign* is what matters: a negative one turns
            positive semidefinite into negative semidefinite, and zero leaves
            nothing definite at all.
        square: whether the domain equals the codomain. Self-adjointness is
            only meaningful when it does.

    Returns:
        What ``alpha * A`` may claim.
    """
    traits = close(traits)
    if alpha == 0.0:
        # The zero operator is self-adjoint and positive semidefinite, but only
        # on a square space; it is singular either way.
        return close(_T.SELF_ADJOINT | _T.POSITIVE_SEMIDEFINITE) if square else _T.NONE

    result = traits & (_T.SELF_ADJOINT | _T.INVERTIBLE)
    if alpha > 0.0:
        result |= traits & (_T.POSITIVE_SEMIDEFINITE | _T.POSITIVE_DEFINITE)
    if abs(alpha) == 1.0:
        result |= traits & (_T.ISOMETRY | _T.UNITARY)
    if alpha == 1.0:
        result |= traits & _T.IDEMPOTENT
    return close(result)


def adjoint_traits(traits: Traits) -> Traits:
    """Traits of ``A.adjoint``.

    ISOMETRY is deliberately not preserved: the adjoint of an isometry is a
    co-isometry, which is an isometry only when the operator is unitary. That
    case is recovered by ``close``.
    """
    traits = close(traits)
    keep = (
        _T.SELF_ADJOINT
        | _T.POSITIVE_SEMIDEFINITE
        | _T.POSITIVE_DEFINITE
        | _T.INVERTIBLE
        | _T.UNITARY
        | _T.IDEMPOTENT
    )
    return close(traits & keep)


def compose_traits(a: Traits, b: Traits, *, square: bool) -> Traits:
    """Traits of ``A @ B``, knowing nothing about how the factors relate.

    Structural recognition of adjoint-palindromic compositions such as
    ``L @ L.adjoint`` and ``S @ C @ S.adjoint`` needs the operators themselves
    and lives with the expression nodes; see ``congruence_traits`` and
    ``gramian_traits`` for the trait half of those rules.

    Args:
        a: the outer operator's traits.
        b: the inner operator's.
        square: whether the composition maps a space to itself.

    Returns:
        What the composition may claim, which is little: two self-adjoint
        operators compose to a self-adjoint one only if they commute, and
        that is not a fact about traits.
    """
    a, b = close(a), close(b)
    result = _T.NONE
    if _has(a, _T.ISOMETRY) and _has(b, _T.ISOMETRY):
        result |= _T.ISOMETRY
    if _has(a, _T.UNITARY) and _has(b, _T.UNITARY):
        result |= _T.UNITARY
    if square and _has(a, _T.INVERTIBLE) and _has(b, _T.INVERTIBLE):
        result |= _T.INVERTIBLE
    return close(result)


def inverse_traits(traits: Traits) -> Traits:
    """Traits of ``A^-1``.

    Bare positive-semidefiniteness is not carried across, since a singular
    operator has no inverse; positive-*definiteness* is.
    """
    traits = close(traits)
    keep = _T.SELF_ADJOINT | _T.POSITIVE_DEFINITE | _T.UNITARY | _T.IDEMPOTENT
    return close((traits & keep) | _T.INVERTIBLE)


def congruence_traits(inner: Traits, *, outer_invertible: bool) -> Traits:
    """Traits of ``S @ A @ S.adjoint`` given the traits of ``A``.

    A congruence preserves self-adjointness and semidefiniteness whatever
    ``S`` is, which is what makes this rule worth having separately from
    :func:`compose_traits`.

    Args:
        inner: what ``A`` claims.
        outer_invertible: whether ``S`` is invertible. Definiteness survives
            only then -- a singular ``S`` can send a positive definite ``A``
            to something merely semidefinite, by collapsing a direction.

    Returns:
        What the congruence may claim.
    """
    inner = close(inner)
    result = _T.NONE
    if _has(inner, _T.SELF_ADJOINT):
        result |= _T.SELF_ADJOINT
    if _has(inner, _T.POSITIVE_SEMIDEFINITE):
        result |= _T.POSITIVE_SEMIDEFINITE
        if outer_invertible and _has(inner, _T.POSITIVE_DEFINITE):
            result |= _T.POSITIVE_DEFINITE
    return close(result)


def gramian_traits(*, invertible: bool) -> Traits:
    """Traits of ``L @ L.adjoint``, which is positive semidefinite always.

    The palindrome rule, and the reason a covariance built as a factor times
    its own adjoint needs no assertion from anyone.

    Args:
        invertible: whether ``L`` has full rank. Positive *definiteness*
            needs it; semidefiniteness does not.

    Returns:
        What the product may claim.
    """
    result = _T.SELF_ADJOINT | _T.POSITIVE_SEMIDEFINITE
    if invertible:
        result |= _T.POSITIVE_DEFINITE
    return close(result)
