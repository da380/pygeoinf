"""
Operators between Hilbert spaces.

``Operator`` is the general, possibly nonlinear mapping; ``LinearOperator``
specialises it and gains an adjoint, structural traits and matrix
representations. Scalar-valued operators are ``Functional`` and
``LinearFunctional``, which subsume what v1 calls forms.

Two things are deliberate here. Operators are defined by **subclassing**, with
``from_callables`` for the quick path, rather than only by injecting callables.
And the algebra consults a **specialisation protocol** before falling back to a
generic expression node, so that a family closed under the algebra stays in its
class.

See DESIGN.md section 5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal, Self

import numpy as np

from ..traits import Traits
from .linearisation import Linearisation, QuadraticModel
from .spaces import CoordinateSpace, HilbertSpace, Reals

if TYPE_CHECKING:
    pass

__all__ = [
    "Operator",
    "LinearOperator",
    "Functional",
    "LinearFunctional",
    "AffineOperator",
    "require_coordinates",
]


REALS = Reals()


def require_coordinates(*spaces: HilbertSpace) -> None:
    """Raise unless every space provides a coordinate map.

    Numerical methods that need components call this at construction time, so
    the failure names the missing capability instead of surfacing three calls
    deeper as an ``AttributeError``.
    """
    for space in spaces:
        if not isinstance(space, CoordinateSpace):
            raise TypeError(
                f"{type(space).__name__} provides no coordinate map, and this "
                f"operation requires one. Coordinate-free alternatives exist "
                f"for most of them."
            )


class Operator[X, Y]:
    """A mapping between Hilbert spaces, linear or not."""

    def __init__(self, domain: HilbertSpace[X], codomain: HilbertSpace[Y]) -> None:
        self._domain = domain
        self._codomain = codomain

    # ----------------------------------------------------------------- #
    #                              Structure                            #
    # ----------------------------------------------------------------- #

    @property
    def domain(self) -> HilbertSpace[X]:
        return self._domain

    @property
    def codomain(self) -> HilbertSpace[Y]:
        return self._codomain

    @property
    def is_square(self) -> bool:
        """True when the domain and codomain have the same dimension."""
        return self.domain.dim == self.codomain.dim

    @property
    def is_endomorphism(self) -> bool:
        """True when the operator maps a space to itself."""
        return self.domain == self.codomain

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.domain!r} -> {self.codomain!r})"

    # ----------------------------------------------------------------- #
    #                          Evaluation paths                         #
    # ----------------------------------------------------------------- #

    def __call__(self, x: X) -> Y:
        """The value alone. The cheap path."""
        return self._value(x)

    def at(self, x: X) -> Linearisation[X, Y]:
        """The value and the derivative, sharing work where possible."""
        return self._linearise(x)

    def derivative(self, x: X) -> LinearOperator[X, Y]:
        """The Fréchet derivative at ``x``."""
        return self.at(x).derivative

    def second_derivative(self, x: X, dx: X) -> LinearOperator[X, Y]:
        """``F''(x)[dx, .]``, the second derivative curried on its first slot.

        Optional, exactly as the first derivative is. Supplying it is what
        makes an exact Newton Hessian available for a composed functional
        rather than only the Gauss-Newton approximation.
        """
        return self._second_derivative(x, dx)

    @property
    def has_derivative(self) -> bool:
        cls = type(self)
        return (
            cls._derivative is not Operator._derivative
            or cls._linearise is not Operator._linearise
        )

    @property
    def has_second_derivative(self) -> bool:
        return type(self)._second_derivative is not Operator._second_derivative

    # ----------------------------------------------------------------- #
    #                         Subclass interface                        #
    # ----------------------------------------------------------------- #

    def _value(self, x: X) -> Y:
        raise NotImplementedError(f"{type(self).__name__} does not implement _value.")

    def _derivative(self, x: X) -> LinearOperator[X, Y]:
        raise NotImplementedError(f"{type(self).__name__} carries no derivative.")

    def _second_derivative(self, x: X, dx: X) -> LinearOperator[X, Y]:
        raise NotImplementedError(
            f"{type(self).__name__} carries no second derivative."
        )

    def _linearise(self, x: X) -> Linearisation[X, Y]:
        """Override when one backend call yields both value and derivative."""
        return Linearisation(x, self._value(x), self._derivative(x))

    # ----------------------------------------------------------------- #
    #                      Specialisation protocol                      #
    # ----------------------------------------------------------------- #
    #
    # Returning None means "no specialisation, use a generic node". Both
    # operands are asked, so the result does not depend on the order of the
    # arguments -- which for a commutative operation it must not.

    def _combine_add(self, other: Operator) -> Operator | None:
        return None

    def _combine_radd(self, other: Operator) -> Operator | None:
        return None

    def _combine_compose(self, other: Operator) -> Operator | None:
        """``self @ other``."""
        return None

    def _combine_rcompose(self, other: Operator) -> Operator | None:
        """``other @ self``."""
        return None

    def _combine_scale(self, alpha: float) -> Operator | None:
        return None

    # ----------------------------------------------------------------- #
    #                               Algebra                             #
    # ----------------------------------------------------------------- #

    def _check_same_spaces(self, other: Operator) -> None:
        if self.domain != other.domain:
            raise ValueError(
                f"Domain mismatch: {self.domain!r} against {other.domain!r}."
            )
        if self.codomain != other.codomain:
            raise ValueError(
                f"Codomain mismatch: {self.codomain!r} against {other.codomain!r}."
            )

    def __add__(self, other: Operator) -> Operator:
        if not isinstance(other, Operator):
            return NotImplemented
        self._check_same_spaces(other)
        for candidate in (self._combine_add(other), other._combine_radd(self)):
            if candidate is not None:
                return candidate
        from .nodes import make_sum

        return make_sum(self, other)

    def __sub__(self, other: Operator) -> Operator:
        if not isinstance(other, Operator):
            return NotImplemented
        return self + (-other)

    def __neg__(self) -> Operator:
        return self * -1.0

    def __mul__(self, alpha: float) -> Operator:
        if not isinstance(alpha, (int, float, np.floating, np.integer)):
            return NotImplemented
        alpha = float(alpha)
        candidate = self._combine_scale(alpha)
        if candidate is not None:
            return candidate
        from .nodes import make_scaled

        return make_scaled(alpha, self)

    def __rmul__(self, alpha: float) -> Operator:
        return self.__mul__(alpha)

    def __truediv__(self, alpha: float) -> Operator:
        if not isinstance(alpha, (int, float, np.floating, np.integer)):
            return NotImplemented
        if alpha == 0.0:
            raise ZeroDivisionError("Cannot divide an operator by zero.")
        return self.__mul__(1.0 / float(alpha))

    def __matmul__(self, other: Operator) -> Operator:
        """``(self @ other)(x) == self(other(x))``."""
        if not isinstance(other, Operator):
            return NotImplemented
        if self.domain != other.codomain:
            raise ValueError(
                f"Cannot compose: {self.domain!r} does not match "
                f"{other.codomain!r}."
            )
        for candidate in (self._combine_compose(other), other._combine_rcompose(self)):
            if candidate is not None:
                return candidate
        from .nodes import make_composition

        return make_composition(self, other)

    # ----------------------------------------------------------------- #
    #                              Factories                            #
    # ----------------------------------------------------------------- #

    @classmethod
    def from_callables(
        cls,
        domain: HilbertSpace[X],
        codomain: HilbertSpace[Y],
        value: Callable[[X], Y],
        /,
        *,
        derivative: Callable[[X], LinearOperator[X, Y]] | None = None,
        second_derivative: Callable[[X, X], LinearOperator[X, Y]] | None = None,
        linearise: Callable[[X], Linearisation[X, Y]] | None = None,
    ) -> Operator[X, Y]:
        """Build an operator from functions, for the quick path.

        Supply ``linearise`` when a single call yields value and derivative
        together; that is the whole reason ``at()`` exists.
        """
        return _CallableOperator(
            domain,
            codomain,
            value,
            derivative=derivative,
            second_derivative=second_derivative,
            linearise=linearise,
        )


class _CallableOperator[X, Y](Operator[X, Y]):
    """An operator defined by injected functions."""

    def __init__(
        self,
        domain: HilbertSpace[X],
        codomain: HilbertSpace[Y],
        value: Callable[[X], Y],
        /,
        *,
        derivative: Callable[[X], LinearOperator[X, Y]] | None = None,
        second_derivative: Callable[[X, X], LinearOperator[X, Y]] | None = None,
        linearise: Callable[[X], Linearisation[X, Y]] | None = None,
    ) -> None:
        super().__init__(domain, codomain)
        self._value_fn = value
        self._derivative_fn = derivative
        self._second_derivative_fn = second_derivative
        self._linearise_fn = linearise

    @property
    def has_derivative(self) -> bool:
        return self._derivative_fn is not None or self._linearise_fn is not None

    @property
    def has_second_derivative(self) -> bool:
        return self._second_derivative_fn is not None

    def _value(self, x: X) -> Y:
        return self._value_fn(x)

    def _derivative(self, x: X) -> LinearOperator[X, Y]:
        if self._derivative_fn is None:
            if self._linearise_fn is not None:
                return self._linearise_fn(x).derivative
            raise NotImplementedError("No derivative was supplied.")
        return self._derivative_fn(x)

    def _second_derivative(self, x: X, dx: X) -> LinearOperator[X, Y]:
        if self._second_derivative_fn is None:
            raise NotImplementedError("No second derivative was supplied.")
        return self._second_derivative_fn(x, dx)

    def _linearise(self, x: X) -> Linearisation[X, Y]:
        if self._linearise_fn is not None:
            return self._linearise_fn(x)
        return super()._linearise(x)


class LinearOperator[X, Y](Operator[X, Y]):
    """A linear operator, carrying an adjoint and structural traits.

    Traits are *claims* made by whoever constructs the operator. They are not
    verified here; ``pygeoinf2.testing.check_traits`` verifies them
    numerically.
    """

    def __init__(
        self,
        domain: HilbertSpace[X],
        codomain: HilbertSpace[Y],
        /,
        *,
        traits: Traits = Traits.NONE,
    ) -> None:
        super().__init__(domain, codomain)
        from ..traits import close

        traits = close(traits)
        if Traits.SELF_ADJOINT & traits and domain != codomain:
            raise ValueError(
                "SELF_ADJOINT was claimed for an operator whose domain and "
                f"codomain differ: {domain!r} against {codomain!r}."
            )
        self._traits = traits

    # ----------------------------------------------------------------- #
    #                          Structure                                #
    # ----------------------------------------------------------------- #

    @property
    def traits(self) -> Traits:
        return self._traits

    def with_traits(self, traits: Traits) -> Self:
        """A view of this operator carrying additional claimed traits."""
        from ..traits import close

        return _RetraitedOperator(self, close(self._traits | traits))

    @property
    def adjoint(self) -> LinearOperator[Y, X]:
        """The adjoint, memoised so that ``A.adjoint is A.adjoint``.

        The memoisation is not an optimisation: the structural recognition of
        adjoint-palindromic compositions in ``nodes.py`` compares factors by
        identity, and would not fire without it.
        """
        cached = self.__dict__.get("_adjoint_cache")
        if cached is None:
            if Traits.SELF_ADJOINT & self._traits:
                cached = self
            else:
                cached = self._make_adjoint()
            self.__dict__["_adjoint_cache"] = cached
        return cached

    def _make_adjoint(self) -> LinearOperator[Y, X]:
        from .nodes import _Adjoint

        return _Adjoint(self)

    def _adjoint_value(self, y: Y) -> X:
        """The action of the adjoint. Subclasses that know it override this."""
        raise NotImplementedError(
            f"{type(self).__name__} does not supply an adjoint action."
        )

    # ----------------------------------------------------------------- #
    #                       Operator specialisation                     #
    # ----------------------------------------------------------------- #

    def _linearise(self, x: X) -> Linearisation[X, Y]:
        return Linearisation(x, self._value(x), self)

    def _derivative(self, x: X) -> LinearOperator[X, Y]:
        return self

    def _second_derivative(self, x: X, dx: X) -> LinearOperator[X, Y]:
        from .nodes import _Zero

        return _Zero(self.domain, self.codomain)

    @property
    def has_derivative(self) -> bool:
        return True

    @property
    def has_second_derivative(self) -> bool:
        return True

    # ----------------------------------------------------------------- #
    #                         Matrix representations                    #
    # ----------------------------------------------------------------- #

    def matrix(
        self,
        /,
        *,
        form: Literal["auto", "components", "galerkin"] = "auto",
    ) -> np.ndarray:
        """A dense matrix representation. Requires coordinates on both sides.

        ``"components"`` gives ``A_c`` with ``c_{Ax} == A_c c_x``.
        ``"galerkin"`` gives ``G_Y A_c``, the matrix of the bilinear form
        ``(A y, x)_Y``, which is symmetric exactly when the operator is
        self-adjoint.

        ``"auto"`` picks the Galerkin form when self-adjointness is claimed, so
        that a matrix handed to a symmetric solver is symmetric. This works for
        *extraction* only: on construction the caller must say which
        representation their array is in, since no trait implies it. See
        DESIGN.md section 5.3.
        """
        require_coordinates(self.domain, self.codomain)
        if form == "auto":
            form = "galerkin" if Traits.SELF_ADJOINT & self._traits else "components"
        if form not in ("components", "galerkin"):
            raise ValueError(f"Unknown matrix form {form!r}.")

        columns = np.zeros((self.codomain.dim, self.domain.dim))
        for j in range(self.domain.dim):
            image = self(self.domain.basis_vector(j))
            columns[:, j] = self.codomain.to_components(image)
        if form == "galerkin":
            columns = np.column_stack(
                [
                    self.codomain.apply_gram(columns[:, j])
                    for j in range(self.domain.dim)
                ]
            )
        return columns

    # ----------------------------------------------------------------- #
    #                               Algebra                             #
    # ----------------------------------------------------------------- #

    def __add__(self, other: Operator) -> Operator:
        if isinstance(other, LinearOperator):
            self._check_same_spaces(other)
            for candidate in (self._combine_add(other), other._combine_radd(self)):
                if candidate is not None:
                    return candidate
            from .nodes import _Sum

            return _Sum([self, other])
        return super().__add__(other)

    def __mul__(self, alpha: float) -> Operator:
        if not isinstance(alpha, (int, float, np.floating, np.integer)):
            return NotImplemented
        alpha = float(alpha)
        candidate = self._combine_scale(alpha)
        if candidate is not None:
            return candidate
        from .nodes import _Scaled

        if alpha == 1.0:
            return self
        return _Scaled(alpha, self)

    def __matmul__(self, other: Operator) -> Operator:
        if isinstance(other, LinearOperator):
            if self.domain != other.codomain:
                raise ValueError(
                    f"Cannot compose: {self.domain!r} does not match "
                    f"{other.codomain!r}."
                )
            for candidate in (
                self._combine_compose(other),
                other._combine_rcompose(self),
            ):
                if candidate is not None:
                    return candidate
            from .nodes import _Composition

            return _Composition([self, other])
        return super().__matmul__(other)

    # ----------------------------------------------------------------- #
    #                              Factories                            #
    # ----------------------------------------------------------------- #

    @classmethod
    def from_callables(
        cls,
        domain: HilbertSpace[X],
        codomain: HilbertSpace[Y],
        value: Callable[[X], Y],
        /,
        *,
        adjoint: Callable[[Y], X] | None = None,
        traits: Traits = Traits.NONE,
    ) -> LinearOperator[X, Y]:
        """Build a linear operator from its action and, ideally, its adjoint."""
        return _CallableLinearOperator(
            domain, codomain, value, adjoint=adjoint, traits=traits
        )

    @classmethod
    def self_adjoint(
        cls,
        domain: HilbertSpace[X],
        value: Callable[[X], X],
        /,
        *,
        traits: Traits = Traits.NONE,
    ) -> LinearOperator[X, X]:
        """A self-adjoint operator, whose adjoint action is its own."""
        return _CallableLinearOperator(
            domain,
            domain,
            value,
            adjoint=value,
            traits=traits | Traits.SELF_ADJOINT,
        )

    @classmethod
    def identity(cls, domain: HilbertSpace[X]) -> LinearOperator[X, X]:
        from .nodes import _Identity

        return _Identity(domain)

    @classmethod
    def zero(
        cls, domain: HilbertSpace[X], codomain: HilbertSpace[Y] | None = None
    ) -> LinearOperator[X, Y]:
        from .nodes import _Zero

        return _Zero(domain, domain if codomain is None else codomain)

    @classmethod
    def from_component_matrix(
        cls,
        domain: CoordinateSpace[X],
        codomain: CoordinateSpace[Y],
        matrix: np.ndarray,
        /,
        *,
        traits: Traits = Traits.NONE,
    ) -> LinearOperator[X, Y]:
        """From ``M`` with ``c_{Ax} == M c_x``."""
        require_coordinates(domain, codomain)
        matrix = np.asarray(matrix, dtype=float)
        expected = (codomain.dim, domain.dim)
        if matrix.shape != expected:
            raise ValueError(f"Matrix has shape {matrix.shape}, expected {expected}.")

        def value(x: X) -> Y:
            return codomain.from_components(matrix @ domain.to_components(x))

        def adjoint(y: Y) -> X:
            cy = codomain.apply_gram(codomain.to_components(y))
            return domain.from_components(domain.solve_gram(matrix.T @ cy))

        return _CallableLinearOperator(
            domain, codomain, value, adjoint=adjoint, traits=traits
        )

    @classmethod
    def from_derivative_matrix(
        cls,
        domain: CoordinateSpace[X],
        codomain: CoordinateSpace[Y],
        matrix: np.ndarray,
        /,
        *,
        traits: Traits = Traits.NONE,
    ) -> LinearOperator[X, Y]:
        """From ``M == G_Y A_c``, whose rows are derivative components.

        Row ``i`` holds the derivative components of the ``i``-th output
        functional, which is the form a numerical adjoint method produces. The
        adjoint then applies ``G_X^-1`` on its own, which is what makes ``A*``
        return representers rather than raw component arrays. See DESIGN.md
        section 5.6.
        """
        require_coordinates(domain, codomain)
        matrix = np.asarray(matrix, dtype=float)
        expected = (codomain.dim, domain.dim)
        if matrix.shape != expected:
            raise ValueError(f"Matrix has shape {matrix.shape}, expected {expected}.")

        def value(x: X) -> Y:
            return codomain.from_components(
                codomain.solve_gram(matrix @ domain.to_components(x))
            )

        def adjoint(y: Y) -> X:
            cy = codomain.to_components(y)
            return domain.from_components(domain.solve_gram(matrix.T @ cy))

        return _CallableLinearOperator(
            domain, codomain, value, adjoint=adjoint, traits=traits
        )


class _CallableLinearOperator[X, Y](LinearOperator[X, Y]):
    """A linear operator defined by injected functions."""

    def __init__(
        self,
        domain: HilbertSpace[X],
        codomain: HilbertSpace[Y],
        value: Callable[[X], Y],
        /,
        *,
        adjoint: Callable[[Y], X] | None = None,
        traits: Traits = Traits.NONE,
    ) -> None:
        super().__init__(domain, codomain, traits=traits)
        self._value_fn = value
        self._adjoint_fn = adjoint

    def _value(self, x: X) -> Y:
        return self._value_fn(x)

    def _adjoint_value(self, y: Y) -> X:
        if self._adjoint_fn is None:
            raise NotImplementedError(
                "No adjoint action was supplied for this operator. Deriving "
                "one numerically is possible only with coordinates and is "
                "prohibitively expensive, so it is not done implicitly."
            )
        return self._adjoint_fn(y)


class _RetraitedOperator[X, Y](LinearOperator[X, Y]):
    """A view of an operator carrying extra claimed traits."""

    def __init__(self, base: LinearOperator[X, Y], traits: Traits) -> None:
        super().__init__(base.domain, base.codomain, traits=traits)
        self._base = base

    def _value(self, x: X) -> Y:
        return self._base(x)

    def _adjoint_value(self, y: Y) -> X:
        return self._base.adjoint(y)


class Functional[X](Operator[X, float]):
    """A scalar-valued operator: what v1 calls a non-linear form.

    Subsumed rather than removed — a functional is an operator into
    :class:`Reals`, so it inherits the whole algebra instead of duplicating it.
    """

    def __init__(
        self, domain: HilbertSpace[X], codomain: HilbertSpace[float] | None = None
    ) -> None:
        # The codomain argument exists so that this cooperates with
        # LinearOperator.__init__ in LinearFunctional's MRO. It is always Reals.
        if codomain is not None and codomain != REALS:
            raise ValueError(f"A functional maps into Reals, not {codomain!r}.")
        super().__init__(domain, REALS)

    def at(self, x: X) -> QuadraticModel[X]:
        return self._linearise(x)

    def _linearise(self, x: X) -> QuadraticModel[X]:
        hessian = self._hessian(x) if self.has_hessian else None
        return QuadraticModel(x, self._value(x), self._derivative(x), hessian)

    def gradient(self, x: X) -> X:
        """The Riesz representer of the derivative — a vector in the domain."""
        return self.at(x).gradient

    def hessian(self, x: X) -> LinearOperator[X, X]:
        return self._hessian(x)

    def _hessian(self, x: X) -> LinearOperator[X, X]:
        raise NotImplementedError(f"{type(self).__name__} carries no Hessian.")

    @property
    def has_hessian(self) -> bool:
        return type(self)._hessian is not Functional._hessian

    @classmethod
    def from_callables(
        cls,
        domain: HilbertSpace[X],
        value: Callable[[X], float],
        /,
        *,
        derivative: Callable[[X], LinearFunctional[X]] | None = None,
        gradient: Callable[[X], X] | None = None,
        hessian: Callable[[X], LinearOperator[X, X]] | None = None,
    ) -> Functional[X]:
        """Build a functional from functions.

        ``derivative`` is the documented route, because a numerical adjoint
        method produces a derivative. ``gradient`` is accepted for callers who
        genuinely hold one, but supplying a derivative array there is the
        classic error of DESIGN.md section 5.6; ``testing.check_gradient``
        catches it.
        """
        if derivative is not None and gradient is not None:
            raise ValueError("Supply either derivative or gradient, not both.")
        return _CallableFunctional(
            domain, value, derivative=derivative, gradient=gradient, hessian=hessian
        )


class _CallableFunctional[X](Functional[X]):
    def __init__(
        self,
        domain: HilbertSpace[X],
        value: Callable[[X], float],
        /,
        *,
        derivative: Callable[[X], LinearFunctional[X]] | None = None,
        gradient: Callable[[X], X] | None = None,
        hessian: Callable[[X], LinearOperator[X, X]] | None = None,
    ) -> None:
        super().__init__(domain)
        self._value_fn = value
        self._derivative_fn = derivative
        self._gradient_fn = gradient
        self._hessian_fn = hessian

    @property
    def has_derivative(self) -> bool:
        return self._derivative_fn is not None or self._gradient_fn is not None

    @property
    def has_hessian(self) -> bool:
        return self._hessian_fn is not None

    def _value(self, x: X) -> float:
        return float(self._value_fn(x))

    def _derivative(self, x: X) -> LinearFunctional[X]:
        if self._derivative_fn is not None:
            return self._derivative_fn(x)
        if self._gradient_fn is not None:
            return LinearFunctional.from_representer(self.domain, self._gradient_fn(x))
        raise NotImplementedError("No derivative or gradient was supplied.")

    def _hessian(self, x: X) -> LinearOperator[X, X]:
        if self._hessian_fn is None:
            raise NotImplementedError("No Hessian was supplied.")
        return self._hessian_fn(x)


class LinearFunctional[X](LinearOperator[X, float], Functional[X]):
    """A continuous linear functional: what v1 calls a ``LinearForm``.

    The two ways of reading it are the derivative and the gradient, and the
    adjoint is what separates them:

    - ``self.matrix()`` is the derivative, the row vector ``g``.
    - ``self.representer`` is the gradient, with components ``G^-1 g``.

    See DESIGN.md section 5.6.
    """

    def __init__(
        self, domain: HilbertSpace[X], /, *, traits: Traits = Traits.NONE
    ) -> None:
        LinearOperator.__init__(self, domain, REALS, traits=traits)

    @property
    def representer(self) -> X:
        """The Riesz representer. Equal to ``self.adjoint(1.0)``."""
        cached = self.__dict__.get("_representer_cache")
        if cached is None:
            cached = self.adjoint(1.0)
            self.__dict__["_representer_cache"] = cached
        return cached

    def _linearise(self, x: X) -> QuadraticModel[X]:
        from .nodes import _Zero

        return QuadraticModel(x, self._value(x), self, _Zero(self.domain, self.domain))

    @property
    def has_hessian(self) -> bool:
        return True

    def _hessian(self, x: X) -> LinearOperator[X, X]:
        from .nodes import _Zero

        return _Zero(self.domain, self.domain)

    @classmethod
    def from_representer(cls, domain: HilbertSpace[X], v: X) -> LinearFunctional[X]:
        """The functional ``x -> (v, x)``, from a gradient you already hold."""
        return _RepresenterFunctional(domain, v)

    @classmethod
    def from_derivative_components(
        cls, domain: CoordinateSpace[X], g: np.ndarray
    ) -> LinearFunctional[X]:
        """The functional ``x -> g . c_x``, from what an adjoint solve returns.

        This is the derivative convention. The representer is obtained by
        applying the inverse metric, which happens inside ``adjoint`` — so the
        correction is structural rather than something a caller must remember.
        """
        require_coordinates(domain)
        g = np.asarray(g, dtype=float)
        if g.shape != (domain.dim,):
            raise ValueError(f"Expected shape {(domain.dim,)}, got {g.shape}.")
        return _RepresenterFunctional(domain, domain.representer(g))


class _RepresenterFunctional[X](LinearFunctional[X]):
    """A linear functional stored by its Riesz representer."""

    def __init__(self, domain: HilbertSpace[X], representer: X) -> None:
        super().__init__(domain)
        self.__dict__["_representer_cache"] = representer

    def _value(self, x: X) -> float:
        return self.domain.inner_product(self.representer, x)

    def _adjoint_value(self, t: float) -> X:
        return self.domain.scale(float(t), self.representer)


class AffineOperator[X, Y](Operator[X, Y]):
    """``x -> linear_part(x) + translation``.

    Its derivative is constant, which is the whole of its structure. The
    specialisation protocol keeps sums and compositions with linear operators
    inside this class, so affineness survives the algebra without the
    string-based type check v1 uses in ``LinearOperator.__add__``.
    """

    def __init__(self, linear_part: LinearOperator[X, Y], translation: Y) -> None:
        super().__init__(linear_part.domain, linear_part.codomain)
        self._linear_part = linear_part
        self._translation = translation

    @property
    def linear_part(self) -> LinearOperator[X, Y]:
        return self._linear_part

    @property
    def translation(self) -> Y:
        return self._translation

    @property
    def has_derivative(self) -> bool:
        return True

    def _value(self, x: X) -> Y:
        return self.codomain.add(self._linear_part(x), self._translation)

    def _derivative(self, x: X) -> LinearOperator[X, Y]:
        return self._linear_part

    def _linearise(self, x: X) -> Linearisation[X, Y]:
        return Linearisation(x, self._value(x), self._linear_part)

    # --- specialisation: stay affine -----------------------------------

    def _combine_add(self, other: Operator) -> Operator | None:
        if isinstance(other, AffineOperator):
            return AffineOperator(
                self._linear_part + other.linear_part,
                self.codomain.add(self._translation, other.translation),
            )
        if isinstance(other, LinearOperator):
            return AffineOperator(self._linear_part + other, self._translation)
        return None

    def _combine_radd(self, other: Operator) -> Operator | None:
        return self._combine_add(other)

    def _combine_scale(self, alpha: float) -> Operator | None:
        return AffineOperator(
            self._linear_part * alpha, self.codomain.scale(alpha, self._translation)
        )

    def _combine_compose(self, other: Operator) -> Operator | None:
        """``self @ other`` with ``other`` linear stays affine."""
        if isinstance(other, LinearOperator):
            return AffineOperator(self._linear_part @ other, self._translation)
        return None

    def _combine_rcompose(self, other: Operator) -> Operator | None:
        """``other @ self`` with ``other`` linear stays affine."""
        if isinstance(other, LinearOperator):
            return AffineOperator(other @ self._linear_part, other(self._translation))
        return None

    def __repr__(self) -> str:
        return f"AffineOperator({self.domain!r} -> {self.codomain!r})"
