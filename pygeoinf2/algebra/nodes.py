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

import numpy as np

from typing import Any, Sequence

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
from .linearisation import Linearisation, QuadraticModel
from .operators import (
    REALS,
    Functional,
    LinearFunctional,
    LinearOperator,
    Operator,
)
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

    def _known_matrix(self, form: str) -> np.ndarray | None:
        from .spaces import CoordinateSpace

        if not isinstance(self.domain, CoordinateSpace):
            return None
        if form == "components":
            return np.identity(self.domain.dim)
        return self.domain.gram_matrix()

    def _known_diagonals(
        self, offsets: tuple[int, ...], form: str
    ) -> np.ndarray | None:
        from .diagonal import DiagonalLinearOperator
        from .spaces import CoordinateSpace

        if not isinstance(self.domain, CoordinateSpace):
            return None
        ones = DiagonalLinearOperator(self.domain, np.ones(self.domain.dim))
        return ones._known_diagonals(offsets, form)

    def apply_block(
        self, vectors: Sequence[Any], /, *, n_jobs: int | None = None
    ) -> list[Any]:
        """Block application through the node; see ``LinearOperator.apply_block``.

        Args:
            vectors: the inputs.
            n_jobs: workers, passed down to the parts. Serial by default.

        Returns:
            The images, in order.
        """
        return list(vectors)

    def _adjoint_apply_block(
        self, vectors: Sequence[Any], /, *, n_jobs: int | None = None
    ) -> list[Any]:
        return list(vectors)

    def _combine_compose(self, other: Operator) -> Operator | None:
        return other

    def _combine_rcompose(self, other: Operator) -> Operator | None:
        return other

    def _combine_scale(self, alpha: float) -> Operator | None:
        """A multiple of the identity is a diagonal operator, and saying so
        is what lets it keep folding.

        ``sigma * I`` was a scaled node, which nothing downstream recognises,
        so ``(sigma I) (sigma I)`` -- the covariance every
        ``from_standard_deviation`` measure has -- stayed a composition and
        had to be probed a column at a time to give up its own diagonal. As a
        diagonal operator the two fold to ``sigma^2 I`` and the diagonal is
        already there.

        Only where the space has a basis to be diagonal in; elsewhere the
        scaled node is still the best available.
        """
        import numpy as np

        from ..algebra.spaces import CoordinateSpace
        from .diagonal import DiagonalLinearOperator

        if not isinstance(self.domain, CoordinateSpace):
            return None
        # The traits a scaled identity has in *any* metric: self-adjoint, and
        # definite with the sign of alpha. A diagonal operator on a
        # non-diagonal metric cannot deduce them from its values alone, so
        # they are stated here, where they are known.
        return DiagonalLinearOperator(
            self.domain,
            np.full(self.domain.dim, float(alpha)),
            traits=scale_traits(self.traits, alpha, square=True),
        )

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

    def _known_matrix(self, form: str) -> np.ndarray | None:
        from .spaces import CoordinateSpace

        if not isinstance(self.domain, CoordinateSpace) or not isinstance(
            self.codomain, CoordinateSpace
        ):
            return None
        return np.zeros((self.codomain.dim, self.domain.dim))

    def _known_diagonals(
        self, offsets: tuple[int, ...], form: str
    ) -> np.ndarray | None:
        from .spaces import CoordinateSpace

        if not isinstance(self.domain, CoordinateSpace) or not isinstance(
            self.codomain, CoordinateSpace
        ):
            return None
        return np.zeros((len(offsets), min(self.domain.dim, self.codomain.dim)))

    def apply_block(
        self, vectors: Sequence[Any], /, *, n_jobs: int | None = None
    ) -> list[Any]:
        """Block application through the node; see ``LinearOperator.apply_block``.

        Args:
            vectors: the inputs.
            n_jobs: workers, passed down to the parts. Serial by default.

        Returns:
            The images, in order.
        """
        return [self.codomain.zero() for _ in vectors]

    def _adjoint_apply_block(
        self, vectors: Sequence[Any], /, *, n_jobs: int | None = None
    ) -> list[Any]:
        return [self.domain.zero() for _ in vectors]

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
        self._link_adjoint(base)

    @property
    def base(self) -> LinearOperator[X, Y]:
        """The operator whose adjoint this is."""
        return self._base

    def _value(self, y: Y) -> X:
        return self._base._adjoint_value(y)

    def _adjoint_value(self, x: X) -> Y:
        return self._base(x)

    def _known_matrix(self, form: str) -> np.ndarray | None:
        # Galerkin(A*) == Galerkin(A)^T exactly: (A* e_j, e_i)_X == (e_j, A e_i)_Y.
        # The components form then needs the domain's inverse metric down each
        # column, since A*_c == G_X^-1 Galerkin(A)^T.
        galerkin = self._base._known_matrix("galerkin")
        if galerkin is None:
            return None
        transposed = np.array(galerkin.T)
        if form == "galerkin":
            return transposed
        return self.codomain.solve_gram_to_columns(transposed)

    def _known_diagonals(
        self, offsets: tuple[int, ...], form: str
    ) -> np.ndarray | None:
        if form != "galerkin":
            # Only the Galerkin form transposes cleanly; the components form of
            # an adjoint carries G_X^-1 across the rows.
            return None
        size = min(self.domain.dim, self.codomain.dim)
        # Transposing swaps offset k for -k, and in the spdiags alignment the
        # entries slide by k columns: (A^T)[c - k, c] == A[c, c - k].
        mirrored = self._base._known_diagonals(tuple(-k for k in offsets), form)
        if mirrored is None:
            return None
        result = np.zeros((len(offsets), size))
        for index, offset in enumerate(offsets):
            row = mirrored[index]
            if offset >= 0:
                result[index, offset:] = row[: size - offset]
            else:
                result[index, : size + offset] = row[-offset:]
        return result

    def apply_block(
        self, vectors: Sequence[Any], /, *, n_jobs: int | None = None
    ) -> list[Any]:
        """Block application through the node; see ``LinearOperator.apply_block``.

        Args:
            vectors: the inputs.
            n_jobs: workers, passed down to the parts. Serial by default.

        Returns:
            The images, in order.
        """
        return self._base._adjoint_apply_block(vectors, n_jobs=n_jobs)

    def _adjoint_apply_block(
        self, vectors: Sequence[Any], /, *, n_jobs: int | None = None
    ) -> list[Any]:
        return self._base.apply_block(vectors, n_jobs=n_jobs)

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
        """The scalar factor."""
        return self._alpha

    @property
    def base(self) -> LinearOperator[X, Y]:
        """The operator being scaled."""
        return self._base

    def _value(self, x: X) -> Y:
        return self.codomain.scale(self._alpha, self._base(x))

    def _adjoint_value(self, y: Y) -> X:
        return self.domain.scale(self._alpha, self._base.adjoint(y))

    def _known_matrix(self, form: str) -> np.ndarray | None:
        known = self._base._known_matrix(form)
        return None if known is None else self._alpha * known

    def _known_diagonals(
        self, offsets: tuple[int, ...], form: str
    ) -> np.ndarray | None:
        known = self._base._known_diagonals(offsets, form)
        return None if known is None else self._alpha * known

    def apply_block(
        self, vectors: Sequence[Any], /, *, n_jobs: int | None = None
    ) -> list[Any]:
        """Block application through the node; see ``LinearOperator.apply_block``.

        Args:
            vectors: the inputs.
            n_jobs: workers, passed down to the parts. Serial by default.

        Returns:
            The images, in order.
        """
        return [
            self.codomain.scale(self._alpha, y)
            for y in self._base.apply_block(vectors, n_jobs=n_jobs)
        ]

    def _adjoint_apply_block(
        self, vectors: Sequence[Any], /, *, n_jobs: int | None = None
    ) -> list[Any]:
        return [
            self.domain.scale(self._alpha, x)
            for x in self._base._adjoint_apply_block(vectors, n_jobs=n_jobs)
        ]

    def _combine_scale(self, alpha: float) -> Operator | None:
        """Fold nested scalings rather than nesting nodes."""
        product = self._alpha * alpha
        if product == 1.0:
            return self._base
        return linear_scaled(product, self._base)

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
        """The summands, flattened."""
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

    def _known_matrix(self, form: str) -> np.ndarray | None:
        total = None
        for term in self._terms:
            known = term._known_matrix(form)
            if known is None:
                return None
            total = np.array(known) if total is None else total + known
        return total

    def _known_diagonals(
        self, offsets: tuple[int, ...], form: str
    ) -> np.ndarray | None:
        total = None
        for term in self._terms:
            known = term._known_diagonals(offsets, form)
            if known is None:
                return None
            total = np.array(known) if total is None else total + known
        return total

    def apply_block(
        self, vectors: Sequence[Any], /, *, n_jobs: int | None = None
    ) -> list[Any]:
        """Block application through the node; see ``LinearOperator.apply_block``.

        Args:
            vectors: the inputs.
            n_jobs: workers, passed down to the parts. Serial by default.

        Returns:
            The images, in order.
        """
        vectors = list(vectors)
        images = self._terms[0].apply_block(vectors, n_jobs=n_jobs)
        for term in self._terms[1:]:
            images = [
                self.codomain.add(a, b)
                for a, b in zip(images, term.apply_block(vectors, n_jobs=n_jobs))
            ]
        return images

    def _adjoint_apply_block(
        self, vectors: Sequence[Any], /, *, n_jobs: int | None = None
    ) -> list[Any]:
        vectors = list(vectors)
        images = self._terms[0]._adjoint_apply_block(vectors, n_jobs=n_jobs)
        for term in self._terms[1:]:
            images = [
                self.domain.add(a, b)
                for a, b in zip(
                    images, term._adjoint_apply_block(vectors, n_jobs=n_jobs)
                )
            ]
        return images

    def _make_adjoint(self) -> LinearOperator[Y, X]:
        result = _Sum([term.adjoint for term in self._terms])
        # Close the loop, so that A.adjoint.adjoint is A. This is not tidiness:
        # the palindrome rule below compares factors by identity and would not
        # fire on an operator whose adjoint had been rebuilt.
        result._link_adjoint(self)
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
            if not LinearOperator.adjoints_are_linked(factors[i], factors[n - 1 - i]):
                return Traits.NONE

        outer_invertible = all(
            Traits.INVERTIBLE & factors[i].traits == Traits.INVERTIBLE
            for i in range(n // 2)
        )
        if n % 2 == 0:
            return gramian_traits(invertible=outer_invertible)
        middle = factors[n // 2]
        if not LinearOperator.adjoints_are_linked(middle, middle):
            return Traits.NONE
        return congruence_traits(middle.traits, outer_invertible=outer_invertible)

    @property
    def factors(self) -> tuple[LinearOperator, ...]:
        """The factors, flattened, outermost first."""
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

    def _known_matrix(self, form: str) -> np.ndarray | None:
        # Components matrices compose by multiplication; the Galerkin form of
        # the product is then the codomain's metric on the result.
        product = None
        for factor in self._factors:
            known = factor._known_matrix("components")
            if known is None:
                return None
            product = np.array(known) if product is None else product @ known
        if form == "components":
            return product
        return self.codomain.apply_gram_to_columns(product)

    def apply_block(
        self, vectors: Sequence[Any], /, *, n_jobs: int | None = None
    ) -> list[Any]:
        """Block application through the node; see ``LinearOperator.apply_block``.

        Args:
            vectors: the inputs.
            n_jobs: workers, passed down to the parts. Serial by default.

        Returns:
            The images, in order.
        """
        result = list(vectors)
        for factor in reversed(self._factors):
            result = factor.apply_block(result, n_jobs=n_jobs)
        return result

    def _adjoint_apply_block(
        self, vectors: Sequence[Any], /, *, n_jobs: int | None = None
    ) -> list[Any]:
        result = list(vectors)
        for factor in self._factors:
            result = factor._adjoint_apply_block(result, n_jobs=n_jobs)
        return result

    def _make_adjoint(self) -> LinearOperator[Y, X]:
        """``(A B ... Z)* == Z* ... B* A*``."""
        result = _Composition([factor.adjoint for factor in reversed(self._factors)])
        result._link_adjoint(self)
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
        """True only when every term carries a derivative."""
        return all(term.has_derivative for term in self._terms)

    @property
    def has_second_derivative(self) -> bool:
        """True only when every term carries a second derivative."""
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
        """True when the scaled operator carries a derivative."""
        return self._base.has_derivative

    @property
    def has_second_derivative(self) -> bool:
        """True when the scaled operator carries a second derivative."""
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
        """True when the chain rule can be applied, so both factors have one."""
        return self._outer.has_derivative and self._inner.has_derivative

    @property
    def has_second_derivative(self) -> bool:
        """True when both factors carry first *and* second derivatives.

        The second derivative of a composition needs the first derivatives
        too, since it differentiates the chain rule.
        """
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


class _LinearFunctionalSum[X](_Sum[X, float], LinearFunctional[X]):
    """``f + g`` for linear functionals, still one."""


class _LinearFunctionalScaled[X](_Scaled[X, float], LinearFunctional[X]):
    """``alpha f`` for a linear functional, still one."""


class _LinearFunctionalComposition[X](_Composition[X, float], LinearFunctional[X]):
    """``f @ A`` for a linear functional, still one."""


def linear_sum(terms: Sequence[LinearOperator]) -> LinearOperator:
    """A linear sum node, staying a functional when the terms are.

    The node classes carry the *value* of a functional correctly whatever they
    are called; what they lose is the type, and with it ``representer``,
    ``derivative_components``, and an ``at()`` that returns a quadratic model
    rather than a linearisation. Nothing here computes anything new -- all
    three of those are already derivable from the adjoint -- but they live on
    :class:`LinearFunctional`, so the type is what makes them reachable.
    """
    if terms[0].codomain == REALS:
        return _LinearFunctionalSum(terms)
    return _Sum(terms)


def linear_scaled(alpha: float, base: LinearOperator) -> LinearOperator:
    """A linear scaling node, staying a functional when the base is."""
    if base.codomain == REALS:
        return _LinearFunctionalScaled(alpha, base)
    return _Scaled(alpha, base)


def linear_composition(factors: Sequence[LinearOperator]) -> LinearOperator:
    """A linear composition node, staying a functional when the result is."""
    if factors[0].codomain == REALS:
        return _LinearFunctionalComposition(factors)
    return _Composition(factors)


# --------------------------------------------------------------------- #
#            The same nodes, closed over scalar-valued operators        #
# --------------------------------------------------------------------- #
#
# A functional is an operator into Reals, so the nodes above already carry its
# *value* correctly. What they lose is its type, and with it everything a
# functional adds: at() returns a Linearisation rather than a QuadraticModel,
# so there is no `.gradient`; there is no `.hessian`; and `isinstance(_,
# Functional)` is False, which is what an optimiser checks.
#
# That mattered concretely. A misfit plus a regulariser is a sum of
# functionals, and `numerics.optimisation` asks for `functional.at(x).gradient`
# — so every composed objective, which is to say every real one, raised
# AttributeError.
#
# The Hessian rules are DESIGN.md section 5.5's, and each is available exactly
# when its ingredients are.


class _FunctionalSum[X](_OperatorSum[X, float], Functional[X]):
    """``phi + psi``, still a functional."""

    def _linearise(self, x: X) -> QuadraticModel[X]:
        parts = [term.at(x) for term in self._terms]
        value = parts[0].value
        derivative = parts[0].derivative
        for part in parts[1:]:
            value = value + part.value
            derivative = derivative + part.derivative
        hessian = self._hessian(x) if self.has_hessian else None
        return QuadraticModel(x, value, derivative, hessian)

    @property
    def has_hessian(self) -> bool:
        """True when every term has one."""
        return all(
            isinstance(term, Functional) and term.has_hessian for term in self._terms
        )

    def _hessian(self, x: X) -> LinearOperator[X, X]:
        """The sum of the terms' Hessians."""
        total = self._terms[0].hessian(x)
        for term in self._terms[1:]:
            total = total + term.hessian(x)
        return total


class _FunctionalScaled[X](_OperatorScaled[X, float], Functional[X]):
    """``alpha * phi``, still a functional."""

    def _linearise(self, x: X) -> QuadraticModel[X]:
        part = self._base.at(x)
        hessian = self._hessian(x) if self.has_hessian else None
        return QuadraticModel(
            x,
            self._alpha * part.value,
            part.derivative * self._alpha,
            hessian,
        )

    @property
    def has_hessian(self) -> bool:
        """True when the scaled functional has one."""
        return isinstance(self._base, Functional) and self._base.has_hessian

    def _hessian(self, x: X) -> LinearOperator[X, X]:
        """``alpha H``."""
        return self._base.hessian(x) * self._alpha


class _FunctionalComposition[X](_OperatorComposition[X, float], Functional[X]):
    """``phi @ F``, still a functional."""

    def _linearise(self, x: X) -> QuadraticModel[X]:
        inner = self._inner.at(x)
        outer = self._outer.at(inner.value)
        hessian = None
        if self.has_hessian:
            hessian = self._compose_hessian(inner, outer)
        return QuadraticModel(
            x, outer.value, outer.derivative @ inner.derivative, hessian
        )

    @property
    def has_hessian(self) -> bool:
        """True when the chain rule for second derivatives can be applied.

        The outer functional needs a Hessian and the inner map a derivative.
        The inner map's *second* derivative is needed only when it has one:
        a linear inner map has none and needs none, which is the common case
        (``phi @ A``) and the one where the rule collapses to ``A* H A``.
        """
        if not isinstance(self._outer, Functional):
            return False
        return (
            self._outer.has_hessian
            and self._inner.has_derivative
            and (
                isinstance(self._inner, LinearOperator)
                or self._inner.has_second_derivative
            )
        )

    def _hessian(self, x: X) -> LinearOperator[X, X]:
        inner = self._inner.at(x)
        return self._compose_hessian(inner, self._outer.at(inner.value))

    def _compose_hessian(
        self, inner: Linearisation, outer: QuadraticModel
    ) -> LinearOperator[X, X]:
        r"""``F'* H F' + F''[.]* grad``, the chain rule differentiated once more.

        The first term is the Gauss-Newton one: it is what a least-squares
        problem keeps and the whole Hessian when the inner map is linear. The
        second is the curvature of the inner map, weighted by the outer
        gradient, and it is the term that makes the Hessian indefinite far from
        a minimum.
        """
        derivative = inner.derivative
        curved = isinstance(self._inner, Operator) and not isinstance(
            self._inner, LinearOperator
        )

        def action(dx: X) -> X:
            step = derivative(dx)
            result = derivative.adjoint(outer.hessian(step))
            if curved and self._inner.has_second_derivative:
                second = self._inner.second_derivative(inner.point, dx)
                result = self.domain.add(result, second.adjoint(outer.gradient))
            return result

        return LinearOperator.self_adjoint(self.domain, action)


# --------------------------------------------------------------------- #
#                               Constructors                            #
# --------------------------------------------------------------------- #


def make_sum(a: Operator, b: Operator) -> Operator:
    """A sum node of the right kind for the operands.

    "The right kind" includes staying a functional when both operands are: an
    algebra that is not closed loses, at the first ``+``, everything the type
    was carrying.
    """
    if isinstance(a, LinearOperator) and isinstance(b, LinearOperator):
        return linear_sum([a, b])
    if isinstance(a, Functional) and isinstance(b, Functional):
        return _FunctionalSum([a, b])
    return _OperatorSum([a, b])


def make_scaled(alpha: float, base: Operator) -> Operator:
    """A scaling node of the right kind for the operand."""
    if isinstance(base, LinearOperator):
        return base * alpha
    if isinstance(base, Functional):
        return _FunctionalScaled(alpha, base)
    return _OperatorScaled(alpha, base)


def make_composition(outer: Operator, inner: Operator) -> Operator:
    """A composition node of the right kind for the operands."""
    if isinstance(outer, LinearOperator) and isinstance(inner, LinearOperator):
        return linear_composition([outer, inner])
    if isinstance(outer, Functional):
        return _FunctionalComposition(outer, inner)
    return _OperatorComposition(outer, inner)
