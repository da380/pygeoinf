"""
Expression nodes for the operator algebra.

An algebraic operation builds a small node holding its operands, rather than a
closure that has forgotten them. That is what lets traits propagate by rule,
lets ``A.adjoint.adjoint is A``, and lets an adjoint-palindromic composition
such as ``L @ L.adjoint`` or ``A @ C @ A.adjoint`` be recognised as positive
semidefinite *after the fact* rather than only at the moment it is built.

Simplification is limited to the obviously safe and locally decidable:
flattening nested sums and compositions, collapsing double adjoints, dropping
identity factors. There is no simplification engine here, and there should not
be one.

See DESIGN.md sections 5.4 and 4.1.
"""

from __future__ import annotations

from typing import Sequence

from ..traits import (
    Traits,
    adjoint_traits,
    close,
    compose_traits,
    congruence_traits,
    gramian_traits,
    scale_traits,
    sum_traits,
)
from .linearisation import Linearisation
from .operators import LinearOperator, Operator
from .spaces import HilbertSpace

__all__ = ["make_sum", "make_scaled", "make_composition"]


# --------------------------------------------------------------------- #
#                          Linear expression nodes                      #
# --------------------------------------------------------------------- #


class _Identity[X](LinearOperator[X, X]):
    """The identity on a space."""

    def __init__(self, domain: HilbertSpace[X]) -> None:
        super().__init__(
            domain,
            domain,
            traits=Traits.SELF_ADJOINT
            | Traits.POSITIVE_DEFINITE
            | Traits.UNITARY
            | Traits.IDEMPOTENT,
        )

    def _value(self, x: X) -> X:
        return x

    def _adjoint_value(self, y: X) -> X:
        return y

    def _combine_compose(self, other: Operator) -> Operator | None:
        return other

    def _combine_rcompose(self, other: Operator) -> Operator | None:
        return other

    def __repr__(self) -> str:
        return f"Identity({self.domain!r})"


class _Zero[X, Y](LinearOperator[X, Y]):
    """The zero operator."""

    def __init__(self, domain: HilbertSpace[X], codomain: HilbertSpace[Y]) -> None:
        traits = (
            Traits.SELF_ADJOINT | Traits.POSITIVE_SEMIDEFINITE
            if domain == codomain
            else Traits.NONE
        )
        super().__init__(domain, codomain, traits=traits)

    def _value(self, x: X) -> Y:
        return self.codomain.zero()

    def _adjoint_value(self, y: Y) -> X:
        return self.domain.zero()

    def _combine_add(self, other: Operator) -> Operator | None:
        return other

    def _combine_radd(self, other: Operator) -> Operator | None:
        return other

    def _combine_scale(self, alpha: float) -> Operator | None:
        return self

    def __repr__(self) -> str:
        return f"Zero({self.domain!r} -> {self.codomain!r})"


class _Adjoint[X, Y](LinearOperator[Y, X]):
    """The adjoint of another operator, kept structurally."""

    def __init__(self, base: LinearOperator[X, Y]) -> None:
        super().__init__(base.codomain, base.domain, traits=adjoint_traits(base.traits))
        self._base = base
        # A.adjoint.adjoint is A, without building a second wrapper.
        self.__dict__["_adjoint_cache"] = base

    @property
    def base(self) -> LinearOperator[X, Y]:
        return self._base

    def _value(self, y: Y) -> X:
        return self._base._adjoint_value(y)

    def _adjoint_value(self, x: X) -> Y:
        return self._base(x)

    def __repr__(self) -> str:
        return f"Adjoint({self._base!r})"


class _Scaled[X, Y](LinearOperator[X, Y]):
    """A scalar multiple of an operator."""

    def __init__(self, alpha: float, base: LinearOperator[X, Y]) -> None:
        super().__init__(
            base.domain,
            base.codomain,
            traits=scale_traits(
                base.traits, alpha, square=base.domain == base.codomain
            ),
        )
        self._alpha = float(alpha)
        self._base = base

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def base(self) -> LinearOperator[X, Y]:
        return self._base

    def _value(self, x: X) -> Y:
        return self.codomain.scale(self._alpha, self._base(x))

    def _adjoint_value(self, y: Y) -> X:
        return self.domain.scale(self._alpha, self._base.adjoint(y))

    def _combine_scale(self, alpha: float) -> Operator | None:
        """Fold nested scalings rather than nesting nodes."""
        product = self._alpha * alpha
        if product == 1.0:
            return self._base
        return _Scaled(product, self._base)

    def __repr__(self) -> str:
        return f"Scaled({self._alpha!r}, {self._base!r})"


class _Sum[X, Y](LinearOperator[X, Y]):
    """A sum of operators, flattened."""

    def __init__(self, terms: Sequence[LinearOperator[X, Y]]) -> None:
        flat: list[LinearOperator[X, Y]] = []
        for term in terms:
            if isinstance(term, _Sum):
                flat.extend(term.terms)
            else:
                flat.append(term)
        traits = flat[0].traits
        for term in flat[1:]:
            traits = sum_traits(traits, term.traits)
        super().__init__(flat[0].domain, flat[0].codomain, traits=traits)
        self._terms = tuple(flat)

    @property
    def terms(self) -> tuple[LinearOperator[X, Y], ...]:
        return self._terms

    def _value(self, x: X) -> Y:
        result = self._terms[0](x)
        for term in self._terms[1:]:
            result = self.codomain.add(result, term(x))
        return result

    def _adjoint_value(self, y: Y) -> X:
        result = self._terms[0].adjoint(y)
        for term in self._terms[1:]:
            result = self.domain.add(result, term.adjoint(y))
        return result

    def _make_adjoint(self) -> LinearOperator[Y, X]:
        result = _Sum([term.adjoint for term in self._terms])
        # Close the loop, so that A.adjoint.adjoint is A. This is not tidiness:
        # the palindrome rule below compares factors by identity and would not
        # fire on an operator whose adjoint had been rebuilt.
        result.__dict__["_adjoint_cache"] = self
        return result

    def __repr__(self) -> str:
        return f"Sum({', '.join(repr(t) for t in self._terms)})"


class _Composition[X, Y](LinearOperator[X, Y]):
    """A composition of operators, flattened, with structural trait recovery.

    Traits come from two sources. The generic rules of ``compose_traits`` apply
    always; on top of them, a factor list that is **adjoint-palindromic** —
    ``factors[i].adjoint is factors[n-1-i]`` for every ``i`` — makes the whole
    composition self-adjoint, and positive semidefinite when the middle is.

    That single rule covers both patterns that matter:
    ``L @ L.adjoint`` (a Gramian) and ``A @ C @ A.adjoint`` (a congruence, i.e.
    the covariance pushforward). It works only because ``adjoint`` is memoised,
    so identity comparison is meaningful.
    """

    def __init__(self, factors: Sequence[LinearOperator]) -> None:
        flat: list[LinearOperator] = []
        for factor in factors:
            if isinstance(factor, _Composition):
                flat.extend(factor.factors)
            else:
                flat.append(factor)
        domain = flat[-1].domain
        codomain = flat[0].codomain

        traits = flat[0].traits
        for factor in flat[1:]:
            traits = compose_traits(traits, factor.traits, square=domain == codomain)
        traits |= self._palindrome_traits(flat, domain == codomain)

        super().__init__(domain, codomain, traits=close(traits))
        self._factors = tuple(flat)

    @staticmethod
    def _palindrome_traits(factors: Sequence[LinearOperator], square: bool) -> Traits:
        n = len(factors)
        if n < 2 or not square:
            return Traits.NONE
        for i in range(n // 2):
            if factors[i].adjoint is not factors[n - 1 - i]:
                return Traits.NONE

        outer_invertible = all(
            Traits.INVERTIBLE & factors[i].traits == Traits.INVERTIBLE
            for i in range(n // 2)
        )
        if n % 2 == 0:
            return gramian_traits(invertible=outer_invertible)
        middle = factors[n // 2]
        if middle.adjoint is not middle:
            return Traits.NONE
        return congruence_traits(middle.traits, outer_invertible=outer_invertible)

    @property
    def factors(self) -> tuple[LinearOperator, ...]:
        return self._factors

    def _value(self, x: X) -> Y:
        result = x
        for factor in reversed(self._factors):
            result = factor(result)
        return result

    def _adjoint_value(self, y: Y) -> X:
        result = y
        for factor in self._factors:
            result = factor.adjoint(result)
        return result

    def _make_adjoint(self) -> LinearOperator[Y, X]:
        """``(A B ... Z)* == Z* ... B* A*``."""
        result = _Composition([factor.adjoint for factor in reversed(self._factors)])
        result.__dict__["_adjoint_cache"] = self
        return result

    def __repr__(self) -> str:
        return f"Composition({', '.join(repr(f) for f in self._factors)})"


# --------------------------------------------------------------------- #
#                        Nonlinear expression nodes                     #
# --------------------------------------------------------------------- #


class _OperatorSum[X, Y](Operator[X, Y]):
    def __init__(self, terms: Sequence[Operator[X, Y]]) -> None:
        super().__init__(terms[0].domain, terms[0].codomain)
        self._terms = tuple(terms)

    @property
    def has_derivative(self) -> bool:
        return all(term.has_derivative for term in self._terms)

    @property
    def has_second_derivative(self) -> bool:
        return all(term.has_second_derivative for term in self._terms)

    def _value(self, x: X) -> Y:
        result = self._terms[0](x)
        for term in self._terms[1:]:
            result = self.codomain.add(result, term(x))
        return result

    def _linearise(self, x: X) -> Linearisation[X, Y]:
        parts = [term.at(x) for term in self._terms]
        value = parts[0].value
        derivative = parts[0].derivative
        for part in parts[1:]:
            value = self.codomain.add(value, part.value)
            derivative = derivative + part.derivative
        return Linearisation(x, value, derivative)

    def _derivative(self, x: X) -> LinearOperator[X, Y]:
        return self._linearise(x).derivative

    def _second_derivative(self, x: X, dx: X) -> LinearOperator[X, Y]:
        result = self._terms[0].second_derivative(x, dx)
        for term in self._terms[1:]:
            result = result + term.second_derivative(x, dx)
        return result


class _OperatorScaled[X, Y](Operator[X, Y]):
    def __init__(self, alpha: float, base: Operator[X, Y]) -> None:
        super().__init__(base.domain, base.codomain)
        self._alpha = float(alpha)
        self._base = base

    @property
    def has_derivative(self) -> bool:
        return self._base.has_derivative

    @property
    def has_second_derivative(self) -> bool:
        return self._base.has_second_derivative

    def _value(self, x: X) -> Y:
        return self.codomain.scale(self._alpha, self._base(x))

    def _linearise(self, x: X) -> Linearisation[X, Y]:
        part = self._base.at(x)
        return Linearisation(
            x,
            self.codomain.scale(self._alpha, part.value),
            part.derivative * self._alpha,
        )

    def _derivative(self, x: X) -> LinearOperator[X, Y]:
        return self._base.derivative(x) * self._alpha

    def _second_derivative(self, x: X, dx: X) -> LinearOperator[X, Y]:
        return self._base.second_derivative(x, dx) * self._alpha


class _OperatorComposition[X, Y](Operator[X, Y]):
    """``self(x) == outer(inner(x))``, with the chain rule for derivatives.

    The second derivative follows from differentiating the chain rule:
    ``(F o G)''(x)[d, .] == F''(Gx)[G'd, .] @ G' + F'(Gx) @ G''(x)[d, .]``.
    """

    def __init__(self, outer: Operator, inner: Operator) -> None:
        super().__init__(inner.domain, outer.codomain)
        self._outer = outer
        self._inner = inner

    @property
    def has_derivative(self) -> bool:
        return self._outer.has_derivative and self._inner.has_derivative

    @property
    def has_second_derivative(self) -> bool:
        return (
            self._outer.has_second_derivative
            and self._inner.has_second_derivative
            and self._outer.has_derivative
            and self._inner.has_derivative
        )

    def _value(self, x: X) -> Y:
        return self._outer(self._inner(x))

    def _linearise(self, x: X) -> Linearisation[X, Y]:
        inner = self._inner.at(x)
        outer = self._outer.at(inner.value)
        return Linearisation(x, outer.value, outer.derivative @ inner.derivative)

    def _derivative(self, x: X) -> LinearOperator[X, Y]:
        return self._linearise(x).derivative

    def _second_derivative(self, x: X, dx: X) -> LinearOperator[X, Y]:
        inner = self._inner.at(x)
        inner_prime = inner.derivative
        outer_prime = self._outer.derivative(inner.value)
        return self._outer.second_derivative(
            inner.value, inner_prime(dx)
        ) @ inner_prime + outer_prime @ self._inner.second_derivative(x, dx)


# --------------------------------------------------------------------- #
#                               Constructors                            #
# --------------------------------------------------------------------- #


def make_sum(a: Operator, b: Operator) -> Operator:
    """A sum node of the right kind for the operands."""
    if isinstance(a, LinearOperator) and isinstance(b, LinearOperator):
        return _Sum([a, b])
    return _OperatorSum([a, b])


def make_scaled(alpha: float, base: Operator) -> Operator:
    if isinstance(base, LinearOperator):
        return base * alpha
    return _OperatorScaled(alpha, base)


def make_composition(outer: Operator, inner: Operator) -> Operator:
    if isinstance(outer, LinearOperator) and isinstance(inner, LinearOperator):
        return _Composition([outer, inner])
    return _OperatorComposition(outer, inner)
