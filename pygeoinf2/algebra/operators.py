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

from typing import TYPE_CHECKING, Any, Callable, Literal, Self, Sequence

import numpy as np

from ..traits import Traits
from .linearisation import Linearisation, QuadraticModel
from .spaces import CoordinateSpace, EuclideanSpace, HilbertSpace, Reals

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


def _read_diagonals(
    dense: np.ndarray, offsets: tuple[int, ...], size: int
) -> np.ndarray:
    """Diagonals of a dense array in the ``spdiags`` layout ``diagonals`` uses:
    ``result[index, column] == dense[column - offset, column]``."""
    result = np.zeros((len(offsets), size))
    for index, offset in enumerate(offsets):
        values = np.diagonal(dense, offset=offset)
        start = max(offset, 0)
        stop = min(size, start + values.size)
        result[index, start:stop] = values[: stop - start]
    return result


def _as_matrix(matrix: Any) -> Any:
    """A dense array, or a sparse matrix left alone.

    Both support ``@`` and ``.T``, which is all the matrix constructors need,
    so a sparse forward operator needs no separate constructor — and a sparse
    one is what an observation operator with local support naturally is.
    """
    from scipy.sparse import issparse

    if issparse(matrix):
        return matrix
    return np.asarray(matrix, dtype=float)


class _Mass:
    """A mass operator relating two inner products, and its inverse.

    Held as a pair of callables rather than as operators, because for the
    common cases — the identity, and a metric ratio applied through
    ``apply_gram``/``solve_gram`` — there is no operator to build.
    """

    __slots__ = ("forward", "inverse")

    def __init__(self, forward: Any, inverse: Any) -> None:
        self.forward = forward
        self.inverse = inverse


def _identity_mass() -> _Mass:
    """No reweighting."""
    return _Mass(lambda x: x, lambda x: x)


def _relating_mass(target: HilbertSpace, base: HilbertSpace) -> _Mass:
    """``M`` on *base* with ``(x, y)_target == (M x, y)_base``, and ``M^-1``.

    Recursive, because a direct sum's mass is the block diagonal of its
    summands' and a response space of several fields is exactly that.

    Raises:
        ValueError: if the two spaces hold different vectors, so that no mass
            operator relates them.
    """
    from .direct_sum import DirectSum
    from .spaces import CoordinateSpace, MassWeightedSpace

    if target is base or target == base:
        return _identity_mass()

    if target.dim != base.dim:
        raise ValueError(
            f"Cannot relate {target!r} to {base!r}: dimensions {target.dim} "
            f"and {base.dim} differ, so they do not hold the same vectors."
        )

    if isinstance(target, MassWeightedSpace) and target.base == base:
        mass, inverse = target.mass, target.mass_inverse
        return _Mass(mass, inverse)

    if isinstance(target, DirectSum) and isinstance(base, DirectSum):
        if len(target) != len(base):
            raise ValueError(
                f"Cannot relate {target!r} to {base!r}: {len(target)} summands "
                f"against {len(base)}."
            )
        parts = [
            _relating_mass(one, other)
            for one, other in zip(target.subspaces, base.subspaces)
        ]

        def blockwise(which: str) -> Any:
            def apply(x: Any) -> Any:
                return tuple(
                    getattr(part, which)(component) for part, component in zip(parts, x)
                )

            return apply

        return _Mass(blockwise("forward"), blockwise("inverse"))

    if isinstance(target, CoordinateSpace) and isinstance(base, CoordinateSpace):
        # M has component matrix G_base^-1 G_target, applied through the two
        # Gram actions so that nothing is assembled. Both are pointwise on a
        # diagonal metric, which is the usual case.
        def forward(x: Any) -> Any:
            components = target.to_components(x)
            return base.from_components(base.solve_gram(target.apply_gram(components)))

        def inverse(x: Any) -> Any:
            components = base.to_components(x)
            return target.from_components(
                target.solve_gram(base.apply_gram(components))
            )

        return _Mass(forward, inverse)

    raise ValueError(
        f"Cannot relate {target!r} to {base!r}: neither is mass-weighted over "
        f"the other, they are not direct sums of spaces that are, and at least "
        f"one has no component map. Wrap one in a MassWeightedSpace, or supply "
        f"the adjoint directly with from_callables."
    )


def _transfer(source: HilbertSpace, target: HilbertSpace) -> Any:
    """Move a vector from *source* to *target*, as cheaply as they allow.

    A no-op when the two hold the same vectors, which is the case that matters:
    on a spectral space the round trip through components is two transforms per
    application, and the lift would pay four for nothing.
    """
    from .spaces import CoordinateSpace

    # Asked both ways round: holding the same vectors is symmetric, but only
    # one of the two spaces may be in a position to know it -- a mass-weighted
    # space knows its base, and the base knows nothing of the weighting.
    if (
        source is target
        or target.shares_vectors_with(source)
        or source.shares_vectors_with(target)
    ):
        return lambda x: x
    if isinstance(source, CoordinateSpace) and isinstance(target, CoordinateSpace):
        return lambda x: target.from_components(source.to_components(x))
    raise ValueError(
        f"Cannot move a vector from {source!r} to {target!r}: they do not "
        f"share vectors and at least one has no component map."
    )


def require_coordinates(*spaces: HilbertSpace) -> None:
    """Raise unless every space provides a coordinate map.

    Numerical methods that need components call this at construction time, so
    the failure names the missing capability instead of surfacing three calls
    deeper as an ``AttributeError``.

    Args:
        *spaces: the spaces to check.

    Raises:
        TypeError: naming the first space that provides no component map.
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
        """The space the operator maps from."""
        return self._domain

    @property
    def codomain(self) -> HilbertSpace[Y]:
        """The space the operator maps into."""
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
        """True when a derivative is available.

        Detected by whether the subclass overrode ``_derivative`` or
        ``_linearise``; a subclass that decides at runtime overrides this.
        """
        cls = type(self)
        return (
            cls._derivative is not Operator._derivative
            or cls._linearise is not Operator._linearise
        )

    @property
    def has_second_derivative(self) -> bool:
        """True when a second derivative is available."""
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

        Args:
            domain: the operator's domain.
            codomain: its codomain.
            value: the action.
            derivative: ``x -> A'(x)``, a linear operator at each point.
            second_derivative: ``x -> A''(x)[.,.]``, for a Newton method.
            linearise: ``x -> Linearisation``, when one backend call yields
                the value and the derivative together. Given this, the two
                above are not needed.

        Returns:
            The operator.
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
        """True when a derivative or a linearisation callable was supplied."""
        return self._derivative_fn is not None or self._linearise_fn is not None

    @property
    def has_second_derivative(self) -> bool:
        """True when a second-derivative callable was supplied."""
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
        """The structural properties claimed for this operator.

        Claims, not proofs: verify them with ``testing.check_traits``.
        """
        return self._traits

    def with_traits(self, traits: Traits) -> Self:
        """The same operator, carrying additional claimed traits.

        A shallow copy of *this class*, not a wrapper. That matters more than
        it looks: the specialisation protocol of the algebra dispatches on
        type, so an operator that forgets what it is on the way through
        ``with_traits`` loses every fast path it had. A wrapper cost a
        ``DiagonalLinearOperator`` its exact log-determinant — sending a
        diagonal covariance through a hundred Hutchinson probes to estimate a
        number it could have summed — and cost a ``NormalOperator`` its
        factors, so that every structure-aware preconditioner refused it.

        The copy shares the original's data, which is safe because operators
        are immutable, but drops the memoised adjoint: the original may have
        cached an ``_Adjoint`` of itself, and the copy claiming SELF_ADJOINT
        must return *itself* instead.

        Args:
            traits: claims to add. The result carries these and the existing
                ones, closed under implication.

        Returns:
            An operator of the same class with the combined claims.

        Raises:
            ValueError: if SELF_ADJOINT is claimed for an operator whose domain
                and codomain differ.
        """
        from ..traits import close

        combined = close(self._traits | traits)
        if Traits.SELF_ADJOINT & combined and self.domain != self.codomain:
            raise ValueError(
                "SELF_ADJOINT was claimed for an operator whose domain and "
                f"codomain differ: {self.domain!r} against {self.codomain!r}."
            )
        # Not copy.copy: the block operators define a __new__ that takes their
        # blocks, and the copy protocol would call it with none. Building the
        # instance directly and taking the dictionary sidesteps __new__ and
        # __init__ together, which is what a copy of an immutable object wants.
        clone = object.__new__(type(self))
        clone.__dict__.update(self.__dict__)
        clone._traits = combined
        clone.__dict__.pop("_adjoint_cache", None)
        return clone

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

    def _link_adjoint(self, other: LinearOperator[Y, X], /) -> None:
        """Record that *other* is this operator's adjoint, without building it.

        The memo is what makes the structural recognition in ``nodes.py`` work
        — it compares factors by identity — so an operator that can produce its
        adjoint cheaply says so here rather than writing into ``__dict__`` by
        name, which seven call sites in three modules were doing.
        """
        self.__dict__["_adjoint_cache"] = other

    @staticmethod
    def adjoints_are_linked(first: LinearOperator, second: LinearOperator, /) -> bool:
        """Whether two operators are *already known* to be adjoint to each other.

        Reads the memo; never fills it. That distinction is the whole point.
        Asking ``first.adjoint is second`` *constructs* the adjoint when it has
        not been built, and for the inverse of a direct solver that means
        extracting a second matrix and factorising it — so testing whether a
        composition happened to be a palindrome cost an ``O(n^3)`` detour, at
        composition time, on an expression that may never be applied. Measured:
        ``solver(A) @ B`` cost 60 applications of ``A`` at dimension 60, while
        ``B @ solver(A)`` cost none.

        Nothing is lost by reading only. Two operators that are adjoint but
        unlinked would fail the identity test anyway; the links that matter are
        made by writing ``A.adjoint`` in the expression itself, which happens
        before the composition is built.

        Args:
            first, second: the operators to test.

        Returns:
            True when the adjoint relation is already recorded.
        """
        if first is second:
            return bool(Traits.SELF_ADJOINT & first.traits)
        return (
            first.__dict__.get("_adjoint_cache") is second
            or second.__dict__.get("_adjoint_cache") is first
        )

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
        """Always true: a linear operator is its own derivative."""
        return True

    @property
    def has_second_derivative(self) -> bool:
        """Always true: the second derivative of a linear operator is zero."""
        return True

    # ----------------------------------------------------------------- #
    #                         Matrix representations                    #
    # ----------------------------------------------------------------- #

    def matrix(
        self,
        /,
        *,
        form: Literal["auto", "components", "galerkin"] = "auto",
        by: Literal["auto", "columns", "rows"] = "auto",
        n_jobs: int | None = None,
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

        ``by`` chooses which way the matrix is filled in. Columns costs
        ``dim(X)`` applications of the operator; rows costs ``dim(Y)``
        applications of its adjoint. For an observation operator — a few
        hundred data from a model space of many thousands — the difference is
        two orders of magnitude, so ``"auto"`` takes whichever side is smaller.

        Those applications are independent of each other, which is what
        ``n_jobs`` is for: one column per worker. It is worth it when a single
        application is expensive — an operator with a PDE solve inside it, or
        an inverse that runs a Krylov iteration per column — and not otherwise.
        Serial by default; see :mod:`pygeoinf2.parallel`.

        Args:
            form: which representation to return.
            by: which way to fill it in.
            n_jobs: workers for the column or row loop. Serial by default.

        Returns:
            The matrix.

        Raises:
            ValueError: for an unknown *form* or *by*.
        """
        require_coordinates(self.domain, self.codomain)
        if form == "auto":
            form = "galerkin" if Traits.SELF_ADJOINT & self._traits else "components"
        if form not in ("components", "galerkin"):
            raise ValueError(f"Unknown matrix form {form!r}.")
        if by == "auto":
            by = "rows" if self.codomain.dim < self.domain.dim else "columns"
        if by not in ("columns", "rows"):
            raise ValueError(f"Unknown fill direction {by!r}.")
        known = self._known_matrix(form)
        if known is not None:
            return known

        from ..parallel import parallel_map

        if by == "columns":

            def column(index: int) -> np.ndarray:
                return self.codomain.to_components(
                    self(self.domain.basis_vector(index))
                )

            columns = parallel_map(column, range(self.domain.dim), n_jobs=n_jobs)
            matrix = (
                np.column_stack(columns)
                if columns
                else np.zeros((self.codomain.dim, 0))
            )
            if form == "galerkin":
                matrix = self.codomain.apply_gram_to_columns(matrix)
            return matrix

        # Row i of the Galerkin matrix holds the derivative components of
        # x -> (A x, e_i)_Y == (x, A* e_i)_X, which is G_X applied to the
        # components of the adjoint's image. See DESIGN.md section 5.6.
        adjoint = self.adjoint

        def row(index: int) -> np.ndarray:
            representer = adjoint(self.codomain.basis_vector(index))
            return self.domain.apply_gram(self.domain.to_components(representer))

        rows = parallel_map(row, range(self.codomain.dim), n_jobs=n_jobs)
        matrix = np.stack(rows) if rows else np.zeros((0, self.domain.dim))
        if form == "components":
            # A_c == G_Y^-1 M, so the inverse metric acts down each column.
            matrix = self.codomain.solve_gram_to_columns(matrix)
        return matrix

    def _known_matrix(self, form: str) -> np.ndarray | None:
        """The dense matrix in *form*, if the operator can produce it without
        applying itself; ``None`` otherwise.

        The hook behind :meth:`matrix`. An operator built from a matrix reads
        it; a diagonal one writes its spectrum down; the expression nodes ask
        their parts and combine what they get -- a sum adds, a composition
        multiplies, an adjoint transposes the Galerkin form -- and answer
        ``None`` as soon as any part does, at which point :meth:`matrix`
        probes. So ``(A + t I).matrix()`` on a matrix-backed ``A`` costs a
        read, and a direct solver handed it factorises without ``dim``
        applications first.

        *form* is ``"components"`` or ``"galerkin"``, already resolved.
        """
        return None

    def _known_diagonals(
        self, offsets: tuple[int, ...], form: str
    ) -> np.ndarray | None:
        """Selected diagonals, if the operator can produce them without being
        applied; ``None`` otherwise.

        The hook behind :meth:`diagonals`, kept separate from
        :meth:`_known_matrix` because a diagonal is wanted precisely where the
        matrix would not fit: a Jacobi preconditioner on a space of ``10^4``
        components must not form ``10^8`` entries to read ``10^4``. Same
        layout as :meth:`diagonals`: one row per offset, aligned as
        ``scipy.sparse.spdiags`` expects.
        """
        return None

    def apply_block(
        self, vectors: Sequence[X], /, *, n_jobs: int | None = None
    ) -> list[Y]:
        """The operator applied to several vectors.

        One at a time by default, in parallel when asked -- this is where the
        ``n_jobs`` of the randomised routines lands, as one probe per worker.
        An operator that can do better does: a matrix-backed one applies its
        array to all the components at once, a diagonal one broadcasts, and
        the expression nodes pass a block through their parts so a product
        of matrix-backed operators stays a product of matrix products.

        Args:
            vectors: the inputs.
            n_jobs: workers for the one-at-a-time route. Serial by default.

        Returns:
            The images, in order.
        """
        from ..parallel import parallel_map

        return parallel_map(self, list(vectors), n_jobs=n_jobs)

    def _adjoint_apply_block(
        self, vectors: Sequence[Y], /, *, n_jobs: int | None = None
    ) -> list[X]:
        """:meth:`apply_block` for the adjoint, overridden alongside it."""
        from ..parallel import parallel_map

        return parallel_map(self.adjoint, list(vectors), n_jobs=n_jobs)

    def _components_action(self) -> Callable[[np.ndarray], np.ndarray] | None:
        """The action on components, ``c_x -> c_{Ax}``, when the operator can
        give it without leaving coordinates; ``None`` otherwise.

        The hook behind the fused application of products and sums. On a
        spectral space every operator that goes through the vectors costs a
        transform in and a transform out; a matrix-backed forward operator,
        a diagonal covariance and a matrix-backed adjoint applied in turn
        cost four transforms where the arithmetic needs none. With this,
        :class:`~pygeoinf2.algebra.nodes._Composition` converts once at each
        end of a run of operators that have it, and ``A Q A* + R`` on a
        sphere is a matrix product, a broadcast and a matrix product.

        The returned callable maps the domain's components to the codomain's,
        with the metric handled inside -- a Galerkin-stored matrix applies
        ``solve_gram`` itself -- so the results agree with :meth:`__call__` to
        rounding.
        """
        return None

    def _components_adjoint_action(
        self,
    ) -> Callable[[np.ndarray], np.ndarray] | None:
        """:meth:`_components_action` for the adjoint: ``c_y -> c_{A* y}``."""
        return None

    def diagonals(
        self,
        /,
        *,
        offsets: Sequence[int] = (0,),
        form: Literal["auto", "components", "galerkin"] = "auto",
        probe: Literal["exact", "banded"] = "exact",
        n_jobs: int | None = None,
    ) -> np.ndarray:
        """Selected diagonals of the operator's matrix, without forming it.

        Returned as one row per offset, padded to the operator's dimension and
        aligned as ``scipy.sparse.spdiags`` expects, so the result can be handed
        straight to a banded preconditioner.

        ``probe="exact"`` costs one application per column, which is what an
        exact answer costs for a general operator. ``probe="banded"`` costs one
        application per *offset span* instead — independent of dimension — by
        probing with vectors that are one on a whole residue class of columns.
        That is exact when the operator really is banded within ``offsets``,
        and sums in the out-of-band entries when it is not: an approximation,
        and named as one.

        An operator that knows its diagonals -- one built from a matrix, a
        diagonal one, a sum or scaling or adjoint of those, a block-diagonal
        arrangement of them -- gives them up without being applied, whatever
        *probe* says; see :meth:`_known_diagonals`. A composition does not,
        and pays the probe.

        Args:
            offsets: which diagonals, ``0`` being the main one.
            form: which matrix representation to read.
            probe: how to obtain the entries, when they have to be probed.
            n_jobs: workers for the exact probe, whose columns are
                independent. Serial by default.

        Returns:
            One row per offset, aligned as ``scipy.sparse.spdiags`` expects.

        Raises:
            ValueError: for an unknown form or probe, or an empty offset
                list.
        """
        require_coordinates(self.domain, self.codomain)
        offsets = tuple(int(offset) for offset in offsets)
        if not offsets:
            raise ValueError("At least one offset is needed.")
        if form == "auto":
            form = "galerkin" if Traits.SELF_ADJOINT & self._traits else "components"
        if probe not in ("exact", "banded"):
            raise ValueError(f"Unknown probe {probe!r}.")
        known = self._known_diagonals(offsets, form)
        if known is not None:
            return known
        size = min(self.domain.dim, self.codomain.dim)
        result = np.zeros((len(offsets), size))

        def read(image: Y) -> np.ndarray:
            components = self.codomain.to_components(image)
            if form == "galerkin":
                return self.codomain.apply_gram(components)
            return components

        if probe == "exact":
            from ..parallel import parallel_map

            def probe_column(column: int) -> np.ndarray:
                return read(self(self.domain.basis_vector(column)))

            for column, entries in enumerate(
                parallel_map(probe_column, range(size), n_jobs=n_jobs)
            ):
                for index, offset in enumerate(offsets):
                    row = column - offset
                    if 0 <= row < size:
                        result[index, column] = entries[row]
            return result

        width = max(offsets) - min(offsets) + 1
        for residue in range(width):
            probe_components = np.zeros(self.domain.dim)
            probe_components[residue::width] = 1.0
            entries = read(self(self.domain.from_components(probe_components)))
            for index, offset in enumerate(offsets):
                columns = np.arange(residue, size, width)
                rows = columns - offset
                keep = (rows >= 0) & (rows < size)
                result[index, columns[keep]] = entries[rows[keep]]
        return result

    def assembled(self, /, *, n_jobs: int | None = None) -> LinearOperator[X, Y]:
        """The same operator, with its matrix formed once and stored.

        Trades memory for repeated application. Nothing else changes: the
        traits are carried across, and the result agrees with this operator to
        rounding.

        **The components form is stored, not the Galerkin form.** The point of
        assembling is that applications get cheaper, and the components form is
        the one in which a forward application is a bare matrix product:
        ``c_{Ax} == A_c c_x``. The Galerkin form is ``G_Y A_c``, so applying it
        forwards means undoing the metric again, one ``solve_gram`` per
        application — free on an orthonormal metric, a broadcast on a diagonal
        one, and a triangular solve on any other. Measured, conjugate
        gradients on a 2000-dimensional operator over a dense Gram matrix:
        320 ms stored as Galerkin against 95 ms stored as components.

        Extraction is cheaper this way too. Probing a matrix by columns
        *gives* the components form — each column is ``to_components(A e_j)``
        — and the Galerkin form was then a further metric application down
        every column.

        What the Galerkin form is good for is a symmetric factorisation, since
        that is the representation in which a self-adjoint operator is
        symmetric. Nothing is lost there: a direct solver asks for the form it
        wants through :meth:`_known_matrix`, and the stored matrix is
        converted once, at factorisation time, against the ``dim``
        factorisation itself.

        This is why no observation operator needs a ``matrix_free`` flag. Build
        it matrix-free, and assemble it here if it is small enough to be worth
        assembling.

        Args:
            n_jobs: workers for the extraction, one column per worker. Serial
                by default.

        Returns:
            An equivalent :class:`MatrixLinearOperator`.
        """
        return LinearOperator.from_matrix(
            self.domain,
            self.codomain,
            self.matrix(form="components", n_jobs=n_jobs),
            form="components",
            traits=self._traits,
        )

    # ----------------------------------------------------------------- #
    #                               Algebra                             #
    # ----------------------------------------------------------------- #

    def __add__(self, other: Operator) -> Operator:
        if isinstance(other, LinearOperator):
            self._check_same_spaces(other)
            for candidate in (self._combine_add(other), other._combine_radd(self)):
                if candidate is not None:
                    return candidate
            from .nodes import linear_sum

            return linear_sum([self, other])
        return super().__add__(other)

    def __mul__(self, alpha: float) -> Operator:
        if not isinstance(alpha, (int, float, np.floating, np.integer)):
            return NotImplemented
        alpha = float(alpha)
        candidate = self._combine_scale(alpha)
        if candidate is not None:
            return candidate
        from .nodes import linear_scaled

        if alpha == 1.0:
            return self
        return linear_scaled(alpha, self)

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
            from .nodes import linear_composition

            return linear_composition([self, other])
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
        """Build a linear operator from its action and, ideally, its adjoint.

        Args:
            domain: the operator's domain.
            codomain: its codomain.
            value: the action.
            adjoint: the adjoint's action. Without it the adjoint is derived
                by solving, which is correct and far more expensive -- so
                supply it whenever it is known.
            traits: claims about the operator. Not verified here;
                ``testing.check_traits`` does that.

        Returns:
            The operator.
        """
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
        """A self-adjoint operator, whose adjoint action is its own.

        Args:
            domain: the space it acts on.
            value: the action, which is also the adjoint's.
            traits: further claims. Self-adjointness is added to whatever is
                given, that being the point of this constructor.

        Returns:
            The operator.
        """
        return _CallableLinearOperator(
            domain,
            domain,
            value,
            adjoint=value,
            traits=traits | Traits.SELF_ADJOINT,
        )

    @classmethod
    def identity(cls, domain: HilbertSpace[X]) -> LinearOperator[X, X]:
        """The identity on a space."""
        from .nodes import _Identity

        return _Identity(domain)

    @classmethod
    def zero(
        cls,
        domain: HilbertSpace[X],
        /,
        *,
        codomain: HilbertSpace[Y] | None = None,
    ) -> LinearOperator[X, Y]:
        """The zero operator, into ``codomain`` or back into ``domain``.

        Args:
            domain: the domain.
            codomain: the codomain. The domain itself if omitted, which makes
                the result an endomorphism and lets it claim
                self-adjointness.

        Returns:
            The zero operator.
        """
        from .nodes import _Zero

        return _Zero(domain, domain if codomain is None else codomain)

    @classmethod
    def from_vectors(
        cls,
        codomain: HilbertSpace[Y],
        vectors: Sequence[Y],
        /,
        *,
        orthonormal: bool = False,
    ) -> LinearOperator[np.ndarray, Y]:
        """The map ``c -> sum_i c_i v_i`` from coefficients into the space.

        Its adjoint is ``y -> [(v_i, y)]``, so an orthonormal family gives an
        isometry — a trait worth claiming, because it is what makes
        ``U D U*`` recognisable as positive semidefinite when ``D`` is.

        Entirely coordinate-free: only the codomain's inner product and
        ``axpy`` are used, so this is how a low-rank factor is represented on a
        space with no component map.

        Args:
            codomain: the space the vectors live in.
            vectors: the family, as a sequence.
            orthonormal: claim that the family is orthonormal. Verified by
                ``testing.check_traits``, not here.

        Returns:
            The operator from coefficients into the space.

        Raises:
            ValueError: if no vectors are given.
        """
        vectors = tuple(vectors)
        if not vectors:
            raise ValueError("At least one vector is needed.")
        if isinstance(codomain, CoordinateSpace) and codomain.uses_component_fast_paths:
            # The adjoint through inner products analyses ``y`` once per
            # vector; through the stored columns it analyses it once. The
            # columns are computed on first use, so construction stays free.
            return _ColumnOperator(codomain, vectors=vectors, orthonormal=orthonormal)
        domain = EuclideanSpace(len(vectors))

        def value(c: np.ndarray) -> Y:
            result = codomain.zero()
            for weight, vector in zip(c, vectors):
                result = codomain.axpy(float(weight), vector, result)
            return result

        def adjoint(y: Y) -> np.ndarray:
            return np.array([codomain.inner_product(v, y) for v in vectors])

        return _CallableLinearOperator(
            domain,
            codomain,
            value,
            adjoint=adjoint,
            traits=Traits.ISOMETRY if orthonormal else Traits.NONE,
        )

    @classmethod
    def from_component_columns(
        cls,
        codomain: CoordinateSpace[Y],
        columns: np.ndarray,
        /,
        *,
        orthonormal: bool = False,
    ) -> LinearOperator[np.ndarray, Y]:
        """The map ``c -> sum_i c_i v_i`` with the ``v_i`` given by components.

        :meth:`from_vectors` with the vectors already in coordinates, which is
        how a randomised factorisation has them; it saves synthesising ``k``
        vectors only to analyse them again.

        Args:
            codomain: the coordinate space the vectors live in.
            columns: a ``(dim, k)`` array, one vector's components per column.
            orthonormal: claim that the family is orthonormal in the space's
                inner product.

        Returns:
            The operator from ``R^k`` into the space.

        Raises:
            ValueError: if the array has no columns or the wrong number of rows.
        """
        columns = np.asarray(columns, dtype=float)
        if columns.ndim != 2 or columns.shape[1] == 0:
            raise ValueError("At least one column is needed.")
        if columns.shape[0] != codomain.dim:
            raise ValueError(
                f"The columns have {columns.shape[0]} rows; the space has "
                f"dimension {codomain.dim}."
            )
        return _ColumnOperator(codomain, columns=columns, orthonormal=orthonormal)

    @classmethod
    def from_tensor_product(
        cls, u: Y, v: X, /, *, domain: HilbertSpace[X], codomain: HilbertSpace[Y]
    ) -> LinearOperator[X, Y]:
        """The rank-one outer product ``x -> (v, x) u``.

        Not a tensor product of *spaces* — see DESIGN.md 3.3 — but the operator
        construction that shares the name, and the building block of every
        low-rank representation.

        Args:
            left: the vector on the left of the product.
            right: the vector on the right.
            domain: the space *right* belongs to. Taken from the vectors when
                they carry one.
            codomain: the space *left* belongs to.

        Returns:
            The rank-one operator ``x -> (right, x) left``.
        """

        def value(x: X) -> Y:
            return codomain.scale(domain.inner_product(v, x), u)

        def adjoint(y: Y) -> X:
            return domain.scale(codomain.inner_product(u, y), v)

        traits = Traits.NONE
        if domain == codomain and u is v:
            traits = Traits.POSITIVE_SEMIDEFINITE
        return _CallableLinearOperator(
            domain, codomain, value, adjoint=adjoint, traits=traits
        )

    @classmethod
    def from_formal_adjoint(
        cls,
        domain: HilbertSpace[X],
        codomain: HilbertSpace[Y],
        operator: LinearOperator,
        /,
        *,
        traits: Traits = Traits.NONE,
    ) -> LinearOperator[X, Y]:
        """Reuse an operator, and its adjoint, under a different inner product.

        The workflow this exists for: derive the action and the adjoint on the
        space where both are easy — an L2 space, or whatever the discretisation
        naturally gives — and then use the operator on the weighted space the
        problem is actually posed in. Writing the weighted adjoint directly is
        usually much harder and is the step most often got wrong.

        With ``(x, y)_V == (M x, y)_U`` on each side, requiring
        ``(A x, y)_{V_Y} == (x, A^{*V} y)_{V_X}`` gives

        .. code-block:: text

            A^{*V} == M_X^-1 . A^{*U} . M_Y

        so only the two mass operators are needed, and each is read off the
        pair of spaces rather than supplied.

        **The recursion is the point.** Each side may be

        * a :class:`~pygeoinf2.algebra.spaces.MassWeightedSpace`, giving its
          own mass operator;
        * a :class:`~pygeoinf2.algebra.direct_sum.DirectSum`, whose mass is
          block diagonal over the summands' — so a response space of several
          fields *and* a couple of scalars lifts in one call;
        * any other coordinate space carrying a different metric over the same
          vectors, where the mass is ``G_U^-1 G_V``, applied through
          ``apply_gram`` and ``solve_gram`` and so never assembled;
        * a space with the same inner product as the operator's, where the mass
          is the identity and nothing happens.

        The last two are what make a Euclidean domain work: a parameter space
        needs no weighting, and saying so should not require a special case.

        No self-adjointness is claimed. An operator that is formally
        self-adjoint is self-adjoint under the new inner product only if it
        commutes with the ratio of the two, which in general it does not.
        Claim it through *traits* where it holds, and check it.

        Args:
            domain: the space to present the operator's domain as.
            codomain: likewise for its codomain.
            operator: the operator on the base spaces, with a working adjoint.
            traits: claims about the lifted operator.

        Returns:
            The same action, with an adjoint taken in the new inner products.

        Raises:
            ValueError: if a side's dimension does not match the operator's, or
                if the two spaces on a side hold different vectors so that no
                mass operator relates them.
        """
        from .spaces import CoordinateSpace

        base_domain, base_codomain = operator.domain, operator.codomain
        for target, base, side in (
            (domain, base_domain, "domain"),
            (codomain, base_codomain, "codomain"),
        ):
            if target.dim != base.dim:
                raise ValueError(
                    f"The operator's {side} has dimension {base.dim}, but the "
                    f"space to present it as has {target.dim}: they do not "
                    f"hold the same vectors."
                )

        spaces = (domain, codomain, base_domain, base_codomain)
        if all(isinstance(space, CoordinateSpace) for space in spaces):
            # Everything in components. The two spaces on a side share their
            # component map, so moving a vector between them is nothing at all
            # and each mass operator is a pair of Gram actions on an array
            # already in hand. Going through vectors instead would cost two
            # transforms per mass application on a spectral space -- four per
            # call doing no work.
            # The forward action needs no reweighting at all -- only the
            # adjoint does -- so where the two spaces hold the same vectors it
            # is the operator's own action and nothing else.
            into = _transfer(domain, base_domain)
            out_of = _transfer(base_codomain, codomain)

            def value(x: X) -> Y:
                return out_of(operator(into(x)))

            def adjoint(y: Y) -> X:
                # M_Y in components is G_{U_Y}^-1 G_{V_Y}.
                weighted = base_codomain.solve_gram(
                    codomain.apply_gram(codomain.to_components(y))
                )
                pulled = base_domain.to_components(
                    operator.adjoint(base_codomain.from_components(weighted))
                )
                # M_X^-1 in components is G_{V_X}^-1 G_{U_X}.
                return domain.from_components(
                    domain.solve_gram(base_domain.apply_gram(pulled))
                )

            lifted = _CallableLinearOperator(
                domain, codomain, value, adjoint=adjoint, traits=traits
            )
            # Where the two spaces on each side share their component map,
            # the operator's action on components is the lifted operator's,
            # and the adjoint's is the same formula as above on arrays. So a
            # lift of an operator that acts on components acts on them too,
            # and stays inside a fused product or a Krylov loop.
            same_components = domain.shares_vectors_with(
                base_domain
            ) and codomain.shares_vectors_with(base_codomain)
            action = operator._components_action() if same_components else None
            adjoint_action = (
                operator._components_adjoint_action() if same_components else None
            )
            if action is not None:
                lifted._components_action_fn = action
            if adjoint_action is not None:

                def lifted_adjoint_action(c: np.ndarray) -> np.ndarray:
                    weighted = base_codomain.solve_gram(codomain.apply_gram(c))
                    return domain.solve_gram(
                        base_domain.apply_gram(adjoint_action(weighted))
                    )

                lifted._components_adjoint_action_fn = lifted_adjoint_action
            return lifted

        # The coordinate-free route, for a mass-weighted space over a backend
        # with no component map.
        forward_in = _transfer(domain, base_domain)
        forward_out = _transfer(base_codomain, codomain)
        back_in = _transfer(codomain, base_codomain)
        back_out = _transfer(base_domain, domain)

        domain_mass_inverse = _relating_mass(domain, base_domain).inverse
        codomain_mass = _relating_mass(codomain, base_codomain).forward

        def value(x: X) -> Y:
            return forward_out(operator(forward_in(x)))

        def adjoint(y: Y) -> X:
            weighted = codomain_mass(back_in(y))
            return back_out(domain_mass_inverse(operator.adjoint(weighted)))

        return _CallableLinearOperator(
            domain, codomain, value, adjoint=adjoint, traits=traits
        )

    @classmethod
    def from_matrix(
        cls,
        domain: CoordinateSpace[X],
        codomain: CoordinateSpace[Y],
        matrix: Any,
        /,
        *,
        form: Literal["components", "galerkin"],
        traits: Traits = Traits.NONE,
    ) -> LinearOperator[X, Y]:
        """From a matrix, saying which representation it is in.

        ``form`` is **required**, because no trait implies it and the two
        differ by a factor of the metric:

        .. code-block:: text

            "components"   M == A_c        c_{Ax} == M c_x
            "galerkin"     M == G_Y A_c    M_ij   == (A e_j, e_i)_Y

        They coincide exactly when the codomain's basis is orthonormal, which
        is why a wrong choice is invisible on a Euclidean space and wrong
        everywhere else. Round-trips exactly with :meth:`matrix`: what
        ``matrix(form=f)`` returns is what ``from_matrix(..., form=f)`` takes.

        Which one you have depends on where it came from. A matrix of numbers
        read off a discretisation is usually in components. A matrix assembled
        from a bilinear form, or handed back by a numerical adjoint method as
        rows of derivative components, is in Galerkin form — that is what makes
        a symmetric operator's matrix symmetric.

        Args:
            domain: the domain, which must have coordinates.
            codomain: the codomain, which must have coordinates.
            matrix: the array, dense or sparse. A sparse one stays sparse.
            form: which representation *matrix* is in.
            traits: claims about the operator.

        Returns:
            A :class:`MatrixLinearOperator`, which keeps the array rather than
            capturing it in a closure, so :meth:`matrix`, :meth:`diagonals` and
            :meth:`assembled` are reads and a direct solver factorises what it
            was given instead of re-deriving it.

        Raises:
            ValueError: if the shape is wrong or the form is not one of the two.
        """
        return MatrixLinearOperator(domain, codomain, matrix, form=form, traits=traits)

    @classmethod
    def from_derivative_callables(
        cls,
        domain: CoordinateSpace[X],
        codomain: HilbertSpace[Y],
        value: Callable[[X], Y],
        derivative_components: Callable[[Y], np.ndarray],
        /,
        *,
        traits: Traits = Traits.NONE,
    ) -> LinearOperator[X, Y]:
        """Matrix-free, from the action and the *derivative* of its pullback.

        The counterpart of :meth:`from_matrix` in Galerkin form, for an operator
        large to assemble. ``derivative_components(y)`` returns

        .. code-block:: text

            d (A x, y)_Y / d c_x

        the components of the functional ``x -> (A x, y)_Y`` in the domain's
        basis — which for a Euclidean codomain is just the weighted sum of rows
        that a numerical adjoint method accumulates. The inverse metric is
        applied here, once, by :meth:`~pygeoinf2.algebra.spaces.CoordinateSpace.representer`.

        Only the **domain** needs coordinates. That is the right way round: the
        model space is the one worth keeping matrix-free, and the one whose
        metric is not the identity.

        Passing an adjoint to :meth:`from_callables` instead means passing a
        *gradient*-valued map, and getting that wrong is the error of DESIGN.md
        section 5.6 in the setting where it is hardest to see. Prefer this.

        Args:
            domain: the operator's domain.
            codomain: its codomain.
            value: the action.
            derivative_components: the derivative's action in *components*,
                which is where the metric would otherwise creep in.
            traits: claims about the operator, unverified here.

        Returns:
            The operator, with an adjoint derived from the components.

        Raises:
            TypeError: if the domain provides no component map. Only the
                domain needs one -- which is the right way round, the model
                space being the one worth keeping matrix-free.
        """
        require_coordinates(domain)

        def adjoint(y: Y) -> X:
            g = np.asarray(derivative_components(y), dtype=float)
            if g.shape != (domain.dim,):
                raise ValueError(
                    f"derivative_components returned shape {g.shape}, "
                    f"expected {(domain.dim,)}."
                )
            return domain.representer(g)

        return _CallableLinearOperator(
            domain, codomain, value, adjoint=adjoint, traits=traits
        )


class _ColumnOperator[Y](LinearOperator[np.ndarray, Y]):
    """``c -> sum_i c_i v_i`` on a coordinate space, through the components.

    Holds the vectors' components as the columns of one array, so the action
    is one matrix-vector product and a synthesis, and the adjoint one analysis,
    one metric application and one product: ``[(v_i, y)] == C^T G c_y``.
    Built from vectors (the columns are computed on first use) or from the
    columns themselves.
    """

    def __init__(
        self,
        codomain: CoordinateSpace[Y],
        /,
        *,
        vectors: Sequence[Y] | None = None,
        columns: np.ndarray | None = None,
        orthonormal: bool = False,
    ) -> None:
        if (vectors is None) == (columns is None):
            raise ValueError("Give either the vectors or their columns.")
        count = len(vectors) if vectors is not None else columns.shape[1]
        super().__init__(
            EuclideanSpace(count),
            codomain,
            traits=Traits.ISOMETRY if orthonormal else Traits.NONE,
        )
        self._vectors = None if vectors is None else tuple(vectors)
        self._columns = columns

    @property
    def columns(self) -> np.ndarray:
        """The ``(dim, k)`` array of the vectors' components."""
        if self._columns is None:
            self._columns = self.codomain.components_of(self._vectors)
        return self._columns

    @property
    def vectors(self) -> tuple[Y, ...]:
        """The vectors themselves, synthesised once if built from columns."""
        if self._vectors is None:
            self._vectors = tuple(self.codomain.vectors_from(self._columns))
        return self._vectors

    def _value(self, c: np.ndarray) -> Y:
        return self.codomain.from_components(self.columns @ np.asarray(c, dtype=float))

    def _adjoint_value(self, y: Y) -> np.ndarray:
        weighted = self.codomain.apply_gram(self.codomain.to_components(y))
        return self.columns.T @ weighted

    def _known_matrix(self, form: str) -> np.ndarray | None:
        # The domain is R^k with the standard basis, so the components matrix
        # is the columns themselves.
        if form == "components":
            return np.array(self.columns, dtype=float)
        return self.codomain.apply_gram_to_columns(self.columns)

    def _components_action(self) -> Callable[[np.ndarray], np.ndarray] | None:
        columns = self.columns
        return lambda c: columns @ c

    def _components_adjoint_action(
        self,
    ) -> Callable[[np.ndarray], np.ndarray] | None:
        columns, codomain = self.columns, self.codomain
        return lambda c: columns.T @ codomain.apply_gram(c)

    def apply_block(
        self, vectors: Sequence[np.ndarray], /, *, n_jobs: int | None = None
    ) -> list[Y]:
        """One product of the columns with all the coefficient vectors.

        Args:
            vectors: coefficient arrays in ``R^k``.
            n_jobs: accepted for the protocol and unused.

        Returns:
            The images, in order.
        """
        coefficients = np.stack([np.asarray(c, dtype=float) for c in vectors], axis=1)
        return self.codomain.vectors_from(self.columns @ coefficients)

    def _adjoint_apply_block(
        self, vectors: Sequence[Y], /, *, n_jobs: int | None = None
    ) -> list[np.ndarray]:
        weighted = self.codomain.apply_gram_to_columns(
            self.codomain.components_of(vectors)
        )
        return list((self.columns.T @ weighted).T)

    def __repr__(self) -> str:
        return f"ColumnOperator(k={self.domain.dim}, into={self.codomain!r})"


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

    _components_action_fn: Callable[[np.ndarray], np.ndarray] | None = None
    _components_adjoint_action_fn: Callable[[np.ndarray], np.ndarray] | None = None

    def _components_action(self) -> Callable[[np.ndarray], np.ndarray] | None:
        return self._components_action_fn

    def _components_adjoint_action(
        self,
    ) -> Callable[[np.ndarray], np.ndarray] | None:
        return self._components_adjoint_action_fn

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


class MatrixLinearOperator[X, Y](LinearOperator[X, Y]):
    """A linear operator that remembers the matrix it was built from.

    The matrix constructors used to hand back a closure with the array captured
    inside it, so an operator built *from* a matrix could not produce one:
    :meth:`matrix` re-derived it by ``dim`` applications, :meth:`diagonals`
    likewise, :meth:`assembled` extracted an already-assembled operator, and
    every direct solver paid a full extraction before factorising something it
    had been given outright. At dimension 1200 that is 50-70 ms per call and
    grows as ``n^3``; here each is a read.

    The stored *form* is part of the object, because no trait implies it (see
    DESIGN.md section 5.3): ``"components"`` means ``A_c`` with
    ``c_{Ax} == A_c c_x``, and ``"galerkin"`` means ``G_Y A_c``, the matrix of
    the bilinear form. The two differ on any space whose basis is not
    orthonormal, and the difference is a metric factor that has to enter
    exactly once.

    The array may be sparse, and stays sparse: nothing here densifies it except
    :meth:`matrix`, which is asked for a dense array by name.
    """

    def __init__(
        self,
        domain: CoordinateSpace[X],
        codomain: CoordinateSpace[Y],
        matrix: Any,
        /,
        *,
        form: Literal["components", "galerkin"],
        traits: Traits = Traits.NONE,
    ) -> None:
        """
        Args:
            domain: the domain, which must have coordinates.
            codomain: the codomain, which must have coordinates.
            matrix: the array, dense or sparse.
            form: which representation *matrix* is in. Required, because no
                trait implies it.
            traits: claims about the operator.

        Raises:
            ValueError: if the shape is wrong or the form is not one of the two.
        """
        require_coordinates(domain, codomain)
        if form not in ("components", "galerkin"):
            raise ValueError(f"The form is 'components' or 'galerkin', got {form!r}.")
        stored = _as_matrix(matrix)
        expected = (codomain.dim, domain.dim)
        if stored.shape != expected:
            raise ValueError(f"Matrix has shape {stored.shape}, expected {expected}.")
        super().__init__(domain, codomain, traits=traits)
        self._stored = stored
        self._form = form

    @property
    def stored_form(self) -> str:
        """Which representation :attr:`stored_matrix` is in."""
        return self._form

    @property
    def stored_matrix(self) -> Any:
        """The array as given, dense or sparse.

        Returned rather than copied, so treat it as read-only: the operator is
        immutable and this is its state.
        """
        return self._stored

    def _value(self, x: X) -> Y:
        components = self._stored @ self.domain.to_components(x)
        if self._form == "galerkin":
            # M c_x == G_Y c_{Ax}, so the metric comes off here.
            components = self.codomain.solve_gram(components)
        return self.codomain.from_components(components)

    def _adjoint_value(self, y: Y) -> X:
        components = self.codomain.to_components(y)
        if self._form == "components":
            components = self.codomain.apply_gram(components)
        return self.domain.from_components(
            self.domain.solve_gram(self._stored.T @ components)
        )

    def _dense(self) -> np.ndarray:
        from scipy.sparse import issparse

        if issparse(self._stored):
            return np.asarray(self._stored.todense())
        return self._stored

    def _in_form(self, form: str) -> np.ndarray:
        """The stored matrix in the requested form, converting if need be."""
        dense = self._dense()
        if form == self._form:
            return dense
        if form == "galerkin":
            return self.codomain.apply_gram_to_columns(dense)
        return self.codomain.solve_gram_to_columns(dense)

    def _known_matrix(self, form: str) -> np.ndarray | None:
        return self._in_form(form)

    def _components_action(self) -> Callable[[np.ndarray], np.ndarray] | None:
        stored, codomain = self._stored, self.codomain
        if self._form == "galerkin":
            return lambda c: codomain.solve_gram(np.asarray(stored @ c))
        return lambda c: np.asarray(stored @ c)

    def _components_adjoint_action(
        self,
    ) -> Callable[[np.ndarray], np.ndarray] | None:
        stored, domain, codomain = self._stored, self.domain, self.codomain
        if self._form == "components":
            return lambda c: domain.solve_gram(
                np.asarray(stored.T @ codomain.apply_gram(c))
            )
        return lambda c: domain.solve_gram(np.asarray(stored.T @ c))

    def _known_diagonals(
        self, offsets: tuple[int, ...], form: str
    ) -> np.ndarray | None:
        from scipy.sparse import issparse

        size = min(self.domain.dim, self.codomain.dim)
        if form == self._form and issparse(self._stored):
            # Read off the sparse array without densifying it.
            result = np.zeros((len(offsets), size))
            for index, offset in enumerate(offsets):
                # spdiags alignment: result[index, column] == M[column - offset, column]
                values = np.asarray(self._stored.diagonal(offset)).ravel()
                start = max(offset, 0)
                stop = min(size, start + values.size)
                result[index, start:stop] = values[: stop - start]
            return result
        return _read_diagonals(self._in_form(form), offsets, size)

    def apply_block(
        self, vectors: Sequence[X], /, *, n_jobs: int | None = None
    ) -> list[Y]:
        """One matrix product for all the vectors' components.

        Args:
            vectors: the inputs.
            n_jobs: accepted for the protocol and unused: the product is
                one BLAS call, which threads on its own.

        Returns:
            The images, in order.
        """
        images = np.asarray(self._stored @ self.domain.components_of(vectors))
        if self._form == "galerkin":
            images = self.codomain.solve_gram_to_columns(images)
        return self.codomain.vectors_from(images)

    def _adjoint_apply_block(
        self, vectors: Sequence[Y], /, *, n_jobs: int | None = None
    ) -> list[X]:
        components = self.codomain.components_of(vectors)
        if self._form == "components":
            components = self.codomain.apply_gram_to_columns(components)
        pulled = np.asarray(self._stored.T @ components)
        return self.domain.vectors_from(self.domain.solve_gram_to_columns(pulled))

    def assembled(self, /, *, n_jobs: int | None = None) -> LinearOperator[X, Y]:
        """Itself: it is already assembled."""
        return self

    def __repr__(self) -> str:
        return (
            f"MatrixLinearOperator({self.domain!r} -> {self.codomain!r}, "
            f"form={self._form!r})"
        )


class Functional[X](Operator[X, float]):
    """A scalar-valued operator: what v1 calls a non-linear form.

    Subsumed rather than removed — a functional is an operator into
    :class:`Reals`, so it inherits the whole algebra instead of duplicating it.
    """

    def __init__(  # noqa: positional - cooperative __init__, see below
        self,
        domain: HilbertSpace[X],
        codomain: HilbertSpace[float] | None = None,
        /,
    ) -> None:
        # The codomain argument is positional *and* optional, which the
        # keyword-only rule otherwise forbids, because LinearFunctional's MRO
        # has LinearOperator.__init__ calling it as __init__(domain, codomain).
        # Making it keyword-only breaks that chain, and making it *required*
        # breaks the twenty-odd subclasses that call super().__init__(domain):
        # a functional's codomain is always Reals and none of them names it.
        # Positional-only is the part of the rule that can be kept -- it stops
        # the parameter's *name* becoming API, as LinearFunctional already
        # does below -- and the escape covers the rest.
        if codomain is not None and codomain != REALS:
            raise ValueError(f"A functional maps into Reals, not {codomain!r}.")
        super().__init__(domain, REALS)

    def at(self, x: X) -> QuadraticModel[X]:
        """The value, derivative and Hessian at ``x``, from one evaluation."""
        return self._linearise(x)

    def _linearise(self, x: X) -> QuadraticModel[X]:
        hessian = self._hessian(x) if self.has_hessian else None
        return QuadraticModel(x, self._value(x), self._derivative(x), hessian)

    def gradient(self, x: X) -> X:
        """The Riesz representer of the derivative — a vector in the domain."""
        return self.at(x).gradient

    def hessian(self, x: X) -> LinearOperator[X, X]:
        """The Hessian at ``x``, a self-adjoint operator on the domain."""
        return self._hessian(x)

    def _hessian(self, x: X) -> LinearOperator[X, X]:
        raise NotImplementedError(f"{type(self).__name__} carries no Hessian.")

    @property
    def has_hessian(self) -> bool:
        """True when a Hessian is available."""
        return type(self)._hessian is not Functional._hessian

    # ----------------------------------------------------------------- #
    #                         The convex interface                      #
    # ----------------------------------------------------------------- #

    def subgradient(self, x: X) -> X:
        """An element of the subdifferential at ``x``, as a **vector**.

        For a convex differentiable functional the subdifferential is the
        single point ``{grad f(x)}``, so the default defers to the gradient.
        A non-smooth functional overrides this.

        Like the gradient, this is a vector rather than a functional, so the
        metric is applied exactly once and in one place.

        Args:
            x: where to take it.

        Returns:
            A subgradient, as a vector of the domain.

        Raises:
            NotImplementedError: unless the functional supplies one. A
                differentiable functional's gradient is a subgradient, and is
                used when no explicit one is given.
        """
        if self.has_derivative:
            return self.gradient(x)
        raise NotImplementedError(
            f"{type(self).__name__} provides neither a gradient nor a " f"subgradient."
        )

    @property
    def has_subgradient(self) -> bool:
        """True when a subgradient is available, by gradient or otherwise."""
        return (
            self.has_derivative or type(self).subgradient is not Functional.subgradient
        )

    def prox(self, x: X, step: float, /) -> X:
        """The proximal operator ``argmin_y f(y) + ||y - x||^2 / (2 step)``.

        The norm is the **space's**, so the proximal step is taken in the
        geometry the modeller chose rather than in a coordinate basis. That is
        what makes a proximal method mesh-independent, and it is why the
        closed forms in the convex module are written with norms and
        directions rather than with components.

        Args:
            x: the point to take the proximal step from.
            step: the step size.

        Returns:
            The proximal point.

        Raises:
            NotImplementedError: unless the functional supplies one. Not every
                convex functional has a proximal operator in closed form, and
                a numerical one is a different object.
        """
        raise NotImplementedError(
            f"{type(self).__name__} provides no proximal operator."
        )

    @property
    def has_prox(self) -> bool:
        """True when a proximal operator is available."""
        return type(self).prox is not Functional.prox

    def conjugate(self) -> Functional[X]:
        """The Fenchel conjugate ``f*(y) == sup_x (y, x) - f(x)``.

        A functional on the *same* space, which is a small dividend of Riesz
        identification: without it the conjugate would live on the dual and
        every duality argument would carry a transport map.

        Returns:
            The convex conjugate, on the same space.

        Raises:
            NotImplementedError: unless the functional supplies one. A
                conjugate has no general closed form.
        """
        raise NotImplementedError(f"{type(self).__name__} provides no conjugate.")

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
        subgradient: Callable[[X], X] | None = None,
        prox: Callable[[X, float], X] | None = None,
    ) -> Functional[X]:
        """Build a functional from functions.

        ``derivative`` is the documented route, because a numerical adjoint
        method produces a derivative. ``gradient`` is accepted for callers who
        genuinely hold one, but supplying a derivative array there is the
        classic error of DESIGN.md section 5.6; ``testing.check_gradient``
        catches it.

        Args:
            domain: the functional's domain.
            value: the action.
            derivative: ``x -> LinearFunctional``, the derivative proper.
            gradient: ``x -> vector``, its Riesz representer. Give one or the
                other, not both -- they are the same information and
                supplying both invites them to disagree.
            hessian: ``x -> LinearOperator``, self-adjoint.
            subgradient: ``x -> vector``, for a non-differentiable functional.
            prox: ``(x, step) -> vector``, for a proximal method.

        Returns:
            The functional.

        Raises:
            ValueError: if both a derivative and a gradient are given.
        """
        if derivative is not None and gradient is not None:
            raise ValueError("Supply either derivative or gradient, not both.")
        return _CallableFunctional(
            domain,
            value,
            derivative=derivative,
            gradient=gradient,
            hessian=hessian,
            subgradient=subgradient,
            prox=prox,
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
        subgradient: Callable[[X], X] | None = None,
        prox: Callable[[X, float], X] | None = None,
    ) -> None:
        super().__init__(domain)
        self._value_fn = value
        self._derivative_fn = derivative
        self._gradient_fn = gradient
        self._hessian_fn = hessian
        self._subgradient_fn = subgradient
        self._prox_fn = prox

    @property
    def has_derivative(self) -> bool:
        """True when a derivative or a gradient callable was supplied."""
        return self._derivative_fn is not None or self._gradient_fn is not None

    @property
    def has_hessian(self) -> bool:
        """True when a Hessian callable was supplied."""
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

    @property
    def has_subgradient(self) -> bool:
        """True when a subgradient or a gradient was supplied."""
        return self._subgradient_fn is not None or self.has_derivative

    def subgradient(self, x: X) -> X:
        """The supplied subgradient, or the gradient when there is none."""
        if self._subgradient_fn is not None:
            return self._subgradient_fn(x)
        return super().subgradient(x)

    @property
    def has_prox(self) -> bool:
        """True when a proximal operator was supplied."""
        return self._prox_fn is not None

    def prox(self, x: X, step: float, /) -> X:
        """The supplied proximal operator.

        Args:
            x: the point to step from.
            step: the step size.

        Returns:
            The proximal point.

        Raises:
            NotImplementedError: if this functional was built without one.
        """
        if self._prox_fn is None:
            raise NotImplementedError("No proximal operator was supplied.")
        return self._prox_fn(x, step)


class LinearFunctional[X](LinearOperator[X, float], Functional[X]):
    """A continuous linear functional: what v1 calls a ``LinearForm``.

    The two ways of reading it are the derivative and the gradient, and the
    adjoint is what separates them:

    - ``self.matrix()`` is the derivative, the row vector ``g``.
    - ``self.representer`` is the gradient, with components ``G^-1 g``.

    See DESIGN.md section 5.6.
    """

    def __init__(  # noqa: positional - cooperative __init__, see below
        self,
        domain: HilbertSpace[X],
        codomain: HilbertSpace[float] | None = None,
        /,
        *,
        traits: Traits = Traits.NONE,
    ) -> None:
        # The codomain argument exists for the same reason Functional's does:
        # so that the expression nodes, which call
        # ``super().__init__(domain, codomain, traits=...)``, can have this in
        # their MRO and stay linear functionals. It is always Reals.
        if codomain is not None and codomain != REALS:
            raise ValueError(f"A functional maps into Reals, not {codomain!r}.")
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
        """Always true: the Hessian of a linear functional is zero."""
        return True

    def _hessian(self, x: X) -> LinearOperator[X, X]:
        from .nodes import _Zero

        return _Zero(self.domain, self.domain)

    @property
    def derivative_components(self) -> np.ndarray:
        """The components of the derivative, ``g`` with ``f(x) == g . c_x``.

        The other half of :meth:`from_derivative_components`, and the form a
        functional has to be in before it can become a row of a Galerkin
        matrix. ``G c_representer``, so the metric is applied here and not
        anywhere the caller has to remember.
        """
        require_coordinates(self.domain)
        return self.domain.apply_gram(self.domain.to_components(self.representer))

    @classmethod
    def from_callables(  # type: ignore[override]
        cls,
        domain: HilbertSpace[X],
        value: Callable[[X], float],
        /,
        *,
        representer: Callable[[], X] | None = None,
        derivative_components: Callable[[], np.ndarray] | None = None,
        traits: Traits = Traits.NONE,
    ) -> LinearFunctional[X]:
        """A linear functional from its action, and one of its two readings.

        There was no way to build one of these from a mapping at all: this name
        resolved by MRO to :meth:`LinearOperator.from_callables`, which wants a
        codomain and an adjoint and hands back a plain operator — so v1's
        ``LinearForm(domain, mapping=...)`` had no counterpart.

        Exactly one of *representer* and *derivative_components* is needed, and
        which one you have says which convention you are in. They differ by the
        metric, and confusing them is the error of DESIGN.md section 5.6:

        * the **representer** is the gradient, the vector ``v`` with
          ``f(x) == (v, x)``;
        * the **derivative components** are the row ``g`` with
          ``f(x) == g . c_x``.

        They are given as callables rather than values so that a functional
        whose representer costs a solve does not pay for it until asked.

        Args:
            domain: the space the functional acts on.
            value: the action, ``x -> f(x)``.
            representer: returns the Riesz representer.
            derivative_components: returns the derivative components. Needs a
                domain with coordinates.
            traits: claims about the functional.

        Returns:
            The functional.

        Raises:
            ValueError: if neither reading is supplied, or if both are.
        """
        supplied = (representer is not None) + (derivative_components is not None)
        if supplied != 1:
            raise ValueError(
                "Supply exactly one of representer and derivative_components. "
                "They differ by the metric, so which one you hold is not a "
                "detail the library can guess."
            )
        return _CallableLinearFunctional(
            domain,
            value,
            representer=representer,
            derivative_components=derivative_components,
            traits=traits,
        )

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

        Args:
            domain: the functional's domain.
            components: the derivative's components, *not* the gradient's.

        Returns:
            The linear functional.

        Raises:
            ValueError: if the component count is not the dimension.
        """
        require_coordinates(domain)
        g = np.asarray(g, dtype=float)
        if g.shape != (domain.dim,):
            raise ValueError(f"Expected shape {(domain.dim,)}, got {g.shape}.")
        return _RepresenterFunctional(domain, domain.representer(g))


class _CallableLinearFunctional[X](LinearFunctional[X]):
    """A linear functional given by its action and one of its two readings."""

    def __init__(
        self,
        domain: HilbertSpace[X],
        value: Callable[[X], float],
        /,
        *,
        representer: Callable[[], X] | None,
        derivative_components: Callable[[], np.ndarray] | None,
        traits: Traits = Traits.NONE,
    ) -> None:
        super().__init__(domain, traits=traits)
        self._value_fn = value
        self._representer_fn = representer
        self._components_fn = derivative_components

    def _value(self, x: X) -> float:
        return float(self._value_fn(x))

    def _adjoint_value(self, t: float) -> X:
        return self.domain.scale(float(t), self.representer)

    @property
    def representer(self) -> X:
        """The Riesz representer, from whichever reading was supplied."""
        cached = self.__dict__.get("_representer_cache")
        if cached is None:
            if self._representer_fn is not None:
                cached = self._representer_fn()
            else:
                require_coordinates(self.domain)
                cached = self.domain.representer(np.asarray(self._components_fn()))
            self.__dict__["_representer_cache"] = cached
        return cached


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
        """The linear part ``A``."""
        return self._linear_part

    @property
    def translation(self) -> Y:
        """The translation ``b``."""
        return self._translation

    @property
    def has_derivative(self) -> bool:
        """Always true: the derivative is the constant linear part."""
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
