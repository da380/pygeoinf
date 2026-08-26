"""
Linear and affine subspaces, and the projections onto them.

A subspace is a convex set whose projection happens to be linear, so it belongs
with the convex sets and inherits their whole interface: an indicator whose
proximal operator is the projection, a support function where one exists, and
membership by predicate.

The projector is the object that does the work, and it carries its structure:
``SELF_ADJOINT | IDEMPOTENT``, which closes to positive semidefinite. Its
complement is another projector rather than a generic difference, which matters
because a generic difference would have forgotten that it is idempotent.

Three constructions, in increasing order of what they need:

- **from a basis** — Gram-Schmidt and a sum of outer products. Coordinate-free.
- **from the kernel of an operator** — project off the range of its adjoint,
  which needs a solve in the *codomain*. Coordinate-free, since the operator
  involved is ``A A*`` and so positive semidefinite.
- **from an explicit matrix** — only where coordinates exist, and only worth it
  when the dimension is small.
"""

from __future__ import annotations

import numpy as np

from typing import Any, Sequence

from ..algebra.operators import require_coordinates, LinearOperator
from ..algebra.spaces import HilbertSpace
from ..numerics.solvers import CGSolver, LinearSolver
from ..traits import Traits
from .convex import ConvexSet

__all__ = ["OrthogonalProjector", "LinearSubspace", "AffineSubspace"]


class OrthogonalProjector(LinearOperator):
    """An orthogonal projection: ``P == P* == P^2``.

    Self-adjoint and idempotent, which by trait closure makes it positive
    semidefinite. Stating that is not decoration: it is what lets a projector
    be handed to a method that requires semidefiniteness without anyone
    asserting anything.
    """

    def __init__(
        self,
        domain: HilbertSpace,
        mapping: Any,
        /,
        *,
        complement: OrthogonalProjector | None = None,
    ) -> None:
        """
        Args:
            domain: the space projected within.
            mapping: the action ``P(x)``.
            complement: the projector onto the orthogonal complement, when it
                is already known. Supplying it avoids rebuilding one, and keeps
                ``P.complement().complement()`` the original object.
        """
        super().__init__(
            domain,
            domain,
            traits=Traits.SELF_ADJOINT | Traits.IDEMPOTENT,
        )
        self._mapping = mapping
        self._complement = complement

    def _value(self, x: Any) -> Any:
        return self._mapping(x)

    def _adjoint_value(self, y: Any) -> Any:
        return self._mapping(y)

    def basis(self) -> list[Any]:
        """An orthonormal basis for the range, in the ambient space.

        Recovered by projecting the ambient basis and orthonormalising, which
        needs coordinates. It is *an* orthonormal basis, not *the* one: a
        subspace has many, and every question a projector answers is
        independent of the choice.
        """
        require_coordinates(self.domain)
        projected = [
            self(self.domain.basis_vector(index)) for index in range(self.domain.dim)
        ]
        return self.domain.orthonormal_basis(projected)

    def complement(self) -> OrthogonalProjector:
        """The projector onto the orthogonal complement, ``I - P``.

        Another projector, not a generic difference: the difference would be
        correct and would have forgotten that it is idempotent.
        """
        if self._complement is None:
            space = self.domain
            self._complement = OrthogonalProjector(
                space,
                lambda x: space.subtract(x, self._mapping(x)),
                complement=self,
            )
        return self._complement

    @classmethod
    def from_basis(
        cls,
        domain: HilbertSpace,
        vectors: Sequence[Any],
        /,
        *,
        orthonormal: bool = False,
    ) -> OrthogonalProjector:
        """Projection onto the span of a family of vectors.

        Coordinate-free: Gram-Schmidt and a sum of outer products, both of
        which need only the inner product.

        Args:
            domain: the space.
            vectors: the family spanning the subspace.
            orthonormal: skip the orthonormalisation, if the family is already
                orthonormal. Unverified — ``testing.check_traits`` on the
                resulting projector catches a false claim.
        """
        vectors = tuple(vectors)
        if not vectors:
            return cls(domain, lambda x: domain.zero())
        basis = vectors if orthonormal else tuple(domain.orthonormal_basis(vectors))
        if not basis:
            return cls(domain, lambda x: domain.zero())

        def project(x: Any) -> Any:
            result = domain.zero()
            for vector in basis:
                result = domain.axpy(domain.inner_product(vector, x), vector, result)
            return result

        return cls(domain, project)

    @classmethod
    def onto_kernel(
        cls,
        operator: LinearOperator,
        /,
        *,
        solver: LinearSolver | None = None,
    ) -> OrthogonalProjector:
        """Projection onto ``{ x : A x == 0 }``.

        The kernel is the orthogonal complement of the range of ``A*``, so

            ``P x == x - A* (A A*)^-1 A x``

        and the only solve is in the codomain, on ``A A*`` — which the
        palindrome rule already recognises as positive semidefinite, so a
        conjugate-gradient solve is admissible without anyone claiming
        anything. Coordinate-free throughout.

        Args:
            operator: the operator whose kernel is wanted.
            solver: the solver for ``A A*``. Defaults to conjugate gradients.
        """
        domain, codomain = operator.domain, operator.codomain
        normal = (operator @ operator.adjoint).with_traits(Traits.POSITIVE_DEFINITE)
        inverse = (solver or CGSolver(rtol=1e-12))(normal)

        def project(x: Any) -> Any:
            correction = operator.adjoint(inverse(operator(x)))
            return domain.subtract(x, correction)

        _ = codomain  # named for the docstring's sake
        return cls(domain, project)


class AffineSubspace(ConvexSet):
    """``translation + subspace``: the solution set of a linear equation.

    A convex set, so it carries an indicator whose proximal operator is the
    projection — which is how a linear constraint enters a proximal method.
    """

    def __init__(
        self,
        projector: OrthogonalProjector,
        /,
        *,
        translation: Any = None,
    ) -> None:
        """
        Args:
            projector: the orthogonal projection onto the tangent subspace.
            translation: a point of the affine subspace. Defaults to zero,
                which makes it linear.
        """
        super().__init__(projector.domain)
        self._projector = projector
        self._translation = (
            projector.domain.zero() if translation is None else translation
        )
        self._equation: tuple[LinearOperator, Any] | None = None
        self._solver: LinearSolver | None = None

    @property
    def projector(self) -> OrthogonalProjector:
        """The projection onto the tangent subspace."""
        return self._projector

    @property
    def translation(self) -> Any:
        """A point of the subspace."""
        return self._translation

    @property
    def tangent(self) -> LinearSubspace:
        """The linear subspace this one is a translate of."""
        return LinearSubspace(self._projector)

    def project(self, x: Any, /) -> Any:
        """Translate to the origin, project, and translate back."""
        space = self._domain
        offset = space.subtract(x, self._translation)
        return space.add(self._projector(offset), self._translation)

    def dimension(self) -> int:
        """The dimension of the tangent subspace, from its projector's trace.

        The trace is the sum of the **component** matrix's diagonal, not
        ``sum (P e_i, e_i)``. Those agree only on an orthonormal basis: on a
        weighted space the second is the Galerkin diagonal, and gives a number
        with no meaning. It is the same mistake as confusing a derivative with
        a gradient, in a different costume.

        Costs one application per basis direction, so it is a diagnostic rather
        than something to call in a loop, and it needs coordinates.
        """
        from ..algebra.operators import require_coordinates

        require_coordinates(self._domain)
        space = self._domain
        total = 0.0
        for index in range(space.dim):
            image = self._projector(space.basis_vector(index))
            total += float(space.to_components(image)[index])
        return int(round(total))

    @classmethod
    def from_linear_equation(
        cls,
        operator: LinearOperator,
        value: Any,
        /,
        *,
        solver: LinearSolver | None = None,
    ) -> AffineSubspace:
        """The solution set of ``A x == b``.

        The tangent space is the kernel of ``A``, and the translation is the
        minimum-norm solution ``A* (A A*)^-1 b`` — which is the right choice of
        representative, since any other would make the reported translation
        depend on how the equation was written down.
        """
        normal = (operator @ operator.adjoint).with_traits(Traits.POSITIVE_DEFINITE)
        inverse = (solver or CGSolver(rtol=1e-12))(normal)
        translation = operator.adjoint(inverse(value))
        subspace = cls(
            OrthogonalProjector.onto_kernel(operator, solver=solver),
            translation=translation,
        )
        subspace._equation = (operator, value)
        subspace._solver = solver
        return subspace

    @classmethod
    def from_tangent_basis(
        cls,
        domain: HilbertSpace,
        vectors: Sequence[Any],
        /,
        *,
        translation: Any = None,
    ) -> AffineSubspace:
        """The subspace spanned by given vectors, offset by a translation.

        The vectors need not be orthonormal;
        :meth:`OrthogonalProjector.from_basis` orthonormalises them and drops
        any that were dependent, so the dimension is the rank rather than the
        count.
        """
        return cls(
            OrthogonalProjector.from_basis(domain, vectors), translation=translation
        )

    @classmethod
    def from_complement_basis(
        cls,
        domain: HilbertSpace,
        vectors: Sequence[Any],
        /,
        *,
        translation: Any = None,
    ) -> AffineSubspace:
        """The subspace *orthogonal* to given vectors.

        The other way of describing the same object: by what it excludes rather
        than by what it contains. Which is natural depends on whether there are
        few constraints or few degrees of freedom.
        """
        return cls(
            OrthogonalProjector.from_basis(domain, vectors).complement(),
            translation=translation,
        )

    @classmethod
    def from_hyperplanes(cls, hyperplanes: Sequence[Any], /) -> AffineSubspace:
        """The intersection of a set of hyperplanes.

        Each hyperplane is one linear equation, so this is
        :meth:`from_linear_equation` with the equations stacked — and it is
        written that way rather than by intersecting sets, because the
        intersection of hyperplanes has a translation and a tangent space that
        the general intersection of sets does not.
        """
        from ..algebra.operators import LinearOperator

        hyperplanes = tuple(hyperplanes)
        if not hyperplanes:
            raise ValueError("At least one hyperplane is needed.")
        domain = hyperplanes[0].domain
        if any(plane.domain != domain for plane in hyperplanes):
            raise ValueError("The hyperplanes must live on a common space.")

        # from_vectors builds c -> sum c_i n_i, out of a Euclidean space; its
        # adjoint is x -> [(n_i, x)], which is the constraint operator.
        operator = LinearOperator.from_vectors(
            domain, [plane.normal for plane in hyperplanes]
        ).adjoint
        values = np.array([float(plane.offset) for plane in hyperplanes])
        return cls.from_linear_equation(operator, values)

    def to_hyperplanes(self) -> list[Any]:
        """The subspace as a list of hyperplanes, one per constraint.

        The inverse of :meth:`from_hyperplanes`, up to the choice of basis for
        the complement: any orthonormal basis of it gives the same set.
        """
        from .convex import Hyperplane

        complement = self._projector.complement()
        normals = complement.basis()
        return [
            Hyperplane(
                self._domain,
                normal,
                offset=self._domain.inner_product(normal, self._translation),
            )
            for normal in normals
        ]

    @property
    def has_explicit_equation(self) -> bool:
        """Whether the subspace remembers the equation that defined it.

        It does only when it was built from one. A subspace built from a basis
        knows its tangent space but not which particular ``A x == b`` a caller
        had in mind, and inventing one would be a different equation with the
        same solution set.
        """
        return self._equation is not None

    @property
    def constraint_operator(self) -> LinearOperator:
        """The ``A`` of the equation this subspace was built from."""
        if self._equation is None:
            raise AttributeError(
                "This subspace was not built from an explicit equation, so it "
                "has no constraint operator. Use to_hyperplanes() for an "
                "equation with the same solution set."
            )
        return self._equation[0]

    @property
    def constraint_value(self) -> Any:
        """The ``b`` of the equation this subspace was built from."""
        if self._equation is None:
            raise AttributeError("This subspace was not built from an equation.")
        return self._equation[1]

    def with_translation(self, translation: Any, /) -> AffineSubspace:
        """The same tangent space, through a different point."""
        return AffineSubspace(self._projector, translation=translation)

    def with_constraint_value(self, value: Any, /) -> AffineSubspace:
        """The same constraint operator, set equal to a different value.

        Needs the equation, since it is the equation that is being changed.
        The translation moves to the new minimum-norm solution and the tangent
        space, being the kernel, does not move at all.
        """
        operator = self.constraint_operator
        return AffineSubspace.from_linear_equation(operator, value, solver=self._solver)

    def pseudo_inverse(self) -> LinearOperator:
        """``A* (A A*)^-1``: the map from a constraint value to its subspace.

        The minimum-norm right inverse of the constraint operator, so
        ``pseudo_inverse()(b)`` is the translation of the subspace ``A x == b``
        and composing it with the constraint operator gives the projection onto
        the orthogonal complement of the kernel.
        """
        operator = self.constraint_operator
        normal = (operator @ operator.adjoint).with_traits(Traits.POSITIVE_DEFINITE)
        return operator.adjoint @ (self._solver or CGSolver(rtol=1e-12))(normal)

    def projection_operator(self) -> Any:
        """The projection onto the subspace, as an affine operator.

        ``x -> P x + (I - P) t``. Distinct from :attr:`projector`, which is the
        *linear* projection onto the tangent space: this one lands on the
        subspace itself, and is affine exactly when the translation is nonzero.
        """
        from ..algebra.operators import AffineOperator

        offset = self._domain.subtract(
            self._translation, self._projector(self._translation)
        )
        return AffineOperator(self._projector, offset)

    def boundary(self) -> "AffineSubspace":
        """The subspace itself: an affine subspace is its own boundary.

        It has empty interior in the ambient space whenever it is proper, so
        every one of its points is a boundary point. Provided because the
        question is reasonable and the answer is easy to get wrong.
        """
        return self

    def __repr__(self) -> str:
        return f"AffineSubspace({self._domain!r})"


class LinearSubspace(AffineSubspace):
    """A linear subspace: an affine one through the origin."""

    def __init__(self, projector: OrthogonalProjector, /) -> None:
        """
        Args:
            projector: the orthogonal projection onto the subspace.
        """
        super().__init__(projector)

    def complement(self) -> LinearSubspace:
        """The orthogonal complement, which is again a subspace.

        Note this shadows ``Subset.complement``, and deliberately: the
        set-theoretic complement of a subspace is not a subspace and is of no
        use, while the orthogonal complement is the thing anyone asking for it
        wants.
        """
        return LinearSubspace(self._projector.complement())

    def project(self, x: Any, /) -> Any:
        """Apply the projector."""
        return self._projector(x)

    @classmethod
    def from_basis(
        cls,
        domain: HilbertSpace,
        vectors: Sequence[Any],
        /,
        *,
        orthonormal: bool = False,
    ) -> LinearSubspace:
        """The span of a family of vectors."""
        return cls(
            OrthogonalProjector.from_basis(domain, vectors, orthonormal=orthonormal)
        )

    @classmethod
    def from_kernel(
        cls, operator: LinearOperator, /, *, solver: LinearSolver | None = None
    ) -> LinearSubspace:
        """The kernel of an operator."""
        return cls(OrthogonalProjector.onto_kernel(operator, solver=solver))

    def __repr__(self) -> str:
        return f"LinearSubspace({self._domain!r})"
