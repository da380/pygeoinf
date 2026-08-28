"""
Subsets of a Hilbert space, and the algebra of them.

A subset knows one thing for certain: whether a point is in it. Everything else
— a projection, a support function, a boundary — is structure that some subsets
have and others do not, and is declared rather than assumed.

Nothing here needs coordinates. Membership is a predicate on vectors, and the
set operations are predicates on predicates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

from ..algebra.spaces import HilbertSpace

__all__ = [
    "Subset",
    "EmptySet",
    "UniversalSet",
    "Complement",
    "Intersection",
    "Union",
]


class Subset(ABC):
    """A subset of a Hilbert space, defined by membership."""

    def __init__(self, domain: HilbertSpace, /) -> None:
        """
        Args:
            domain: the space the subset lives in.
        """
        self._domain = domain

    @property
    def domain(self) -> HilbertSpace:
        """The space the subset lives in."""
        return self._domain

    @abstractmethod
    def contains(self, x: Any, /, *, rtol: float = 1e-9) -> bool:
        """Whether a point is in the subset.

        Args:
            x: the point to test.
            rtol: relative tolerance for the boundary. A set defined by an
                equality is empty in floating point without one, so this is not
                a convenience but a necessity.
        """

    def __contains__(self, x: Any) -> bool:
        """``x in subset``, at the default tolerance."""
        return self.contains(x)

    def complement(self) -> Subset:
        """The set of points not in this one."""
        return Complement(self)

    def intersect(self, other: Subset, /) -> Subset:
        """The intersection with another subset."""
        return Intersection([self, other])

    def union(self, other: Subset, /) -> Subset:
        """The union with another subset."""
        return Union([self, other])

    def __and__(self, other: Subset) -> Subset:
        return self.intersect(other)

    def __or__(self, other: Subset) -> Subset:
        return self.union(other)

    def __invert__(self) -> Subset:
        return self.complement()

    def _check_domain(self, other: Subset) -> None:
        """Raise unless two subsets live in the same space."""
        if self._domain != other.domain:
            raise ValueError(
                f"Subsets must share a domain: {self._domain!r} against "
                f"{other.domain!r}."
            )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._domain!r})"


class EmptySet(Subset):
    """The subset containing nothing."""

    def contains(self, x: Any, /, *, rtol: float = 1e-9) -> bool:
        """Always false.

        Args:
            x: a vector of the space.
            rtol: unused; the empty set contains nothing to any tolerance.

        Returns:
            ``False``.
        """
        return False

    def complement(self) -> Subset:
        """The whole space."""
        return UniversalSet(self._domain)


class UniversalSet(Subset):
    """The whole space."""

    def contains(self, x: Any, /, *, rtol: float = 1e-9) -> bool:
        """Always true.

        Args:
            x: a vector of the space.
            rtol: unused; the universal set contains everything to any
                tolerance.

        Returns:
            ``True``.
        """
        return True

    def complement(self) -> Subset:
        """The empty set."""
        return EmptySet(self._domain)


class Complement(Subset):
    """The points not in another subset."""

    def __init__(self, subset: Subset, /) -> None:
        """
        Args:
            subset: the subset to complement.
        """
        super().__init__(subset.domain)
        self._subset = subset

    @property
    def subset(self) -> Subset:
        """The subset being complemented."""
        return self._subset

    def contains(self, x: Any, /, *, rtol: float = 1e-9) -> bool:
        """True when the point is not in the underlying subset.

        Args:
            x: a vector of the space.
            rtol: passed to the underlying subset. Note the tolerance then
                works the other way round: a point just outside the original
                set counts as *inside* its complement only if it is outside
                to within this.

        Returns:
            Whether the complement contains it.
        """
        return not self._subset.contains(x, rtol=rtol)

    def complement(self) -> Subset:
        """The original subset, rather than a doubly-wrapped one."""
        return self._subset

    def __repr__(self) -> str:
        return f"Complement({self._subset!r})"


class Intersection(Subset):
    """The points in every one of several subsets."""

    def __init__(self, subsets: Iterable[Subset], /) -> None:
        """
        Args:
            subsets: the subsets to intersect. Nested intersections are
                flattened, so the structure stays readable.
        """
        parts: list[Subset] = []
        for subset in subsets:
            if isinstance(subset, Intersection):
                parts.extend(subset.subsets)
            else:
                parts.append(subset)
        if not parts:
            raise ValueError("An intersection needs at least one subset.")
        super().__init__(parts[0].domain)
        for subset in parts[1:]:
            self._check_domain(subset)
        self._subsets = tuple(parts)

    @property
    def subsets(self) -> tuple[Subset, ...]:
        """The intersected subsets."""
        return self._subsets

    def contains(self, x: Any, /, *, rtol: float = 1e-9) -> bool:
        """True when every subset contains the point.

        Args:
            x: a vector of the space.
            rtol: passed to each subset in turn.

        Returns:
            Whether all of them contain it.
        """
        return all(subset.contains(x, rtol=rtol) for subset in self._subsets)

    def __repr__(self) -> str:
        return f"Intersection({', '.join(repr(s) for s in self._subsets)})"


class Union(Subset):
    """The points in at least one of several subsets."""

    def __init__(self, subsets: Iterable[Subset], /) -> None:
        """
        Args:
            subsets: the subsets to unite. Nested unions are flattened.
        """
        parts: list[Subset] = []
        for subset in subsets:
            if isinstance(subset, Union):
                parts.extend(subset.subsets)
            else:
                parts.append(subset)
        if not parts:
            raise ValueError("A union needs at least one subset.")
        super().__init__(parts[0].domain)
        for subset in parts[1:]:
            self._check_domain(subset)
        self._subsets = tuple(parts)

    @property
    def subsets(self) -> tuple[Subset, ...]:
        """The united subsets."""
        return self._subsets

    def contains(self, x: Any, /, *, rtol: float = 1e-9) -> bool:
        """True when some subset contains the point.

        Args:
            x: a vector of the space.
            rtol: passed to each subset in turn.

        Returns:
            Whether any of them contains it.
        """
        return any(subset.contains(x, rtol=rtol) for subset in self._subsets)

    def __repr__(self) -> str:
        return f"Union({', '.join(repr(s) for s in self._subsets)})"
