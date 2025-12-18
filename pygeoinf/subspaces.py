"""
Defines classes for representing affine and linear subspaces.

The primary abstraction is the `AffineSubspace`, which represents a subset of
a Hilbert space defined by a translation and a closed linear tangent space.
`LinearSubspace` is a specialization where the translation is zero.
"""

from __future__ import annotations
from typing import List, Optional, Any, Callable, TYPE_CHECKING
import numpy as np

from .linear_operators import LinearOperator
from .hilbert_space import HilbertSpace, Vector, EuclideanSpace
from .linear_solvers import LinearSolver, CholeskySolver, IterativeLinearSolver

if TYPE_CHECKING:
    # Avoid circular imports for type checking
    pass


class OrthogonalProjector(LinearOperator):
    """
    Internal engine for subspace projections.

    Represents an orthogonal projection operator P = P* = P^2.
    While this class can be used directly, it is generally recommended to use
    `AffineSubspace` or `LinearSubspace` for high-level problem definitions.
    """

    def __init__(
        self,
        domain: HilbertSpace,
        mapping: Callable[[Any], Any],
        complement_projector: Optional[LinearOperator] = None,
    ) -> None:
        super().__init__(domain, domain, mapping, adjoint_mapping=mapping)
        self._complement_projector = complement_projector

    @property
    def complement(self) -> LinearOperator:
        """Returns the projector onto the orthogonal complement (I - P)."""
        if self._complement_projector is None:
            identity = self.domain.identity_operator()
            self._complement_projector = identity - self
        return self._complement_projector

    @classmethod
    def from_basis(
        cls,
        domain: HilbertSpace,
        basis_vectors: List[Vector],
        orthonormalize: bool = True,
    ) -> OrthogonalProjector:
        """Constructs P from a basis spanning the range."""
        if not basis_vectors:
            # Return zero operator if basis is empty
            return domain.zero_operator(domain)

        if orthonormalize:
            e_vectors = domain.gram_schmidt(basis_vectors)
        else:
            e_vectors = basis_vectors

        # P = sum (v_i x v_i)
        tensor_op = LinearOperator.self_adjoint_from_tensor_product(domain, e_vectors)
        return cls(domain, tensor_op)


class AffineSubspace:
    """
    Represents an affine subspace A = x0 + V.
    """

    def __init__(
        self,
        projector: OrthogonalProjector,
        translation: Optional[Vector] = None,
        constraint_operator: Optional[LinearOperator] = None,
        constraint_value: Optional[Vector] = None,
    ) -> None:
        self._projector = projector

        if translation is None:
            self._translation = projector.domain.zero
        else:
            if not projector.domain.is_element(translation):
                raise ValueError("Translation vector not in domain.")
            self._translation = translation

        self._constraint_operator = constraint_operator
        self._constraint_value = constraint_value

    @property
    def domain(self) -> HilbertSpace:
        return self._projector.domain

    @property
    def translation(self) -> Vector:
        return self._translation

    @property
    def projector(self) -> OrthogonalProjector:
        return self._projector

    @property
    def tangent_space(self) -> LinearSubspace:
        return LinearSubspace(self._projector)

    @property
    def has_constraint_equation(self) -> bool:
        return self._constraint_operator is not None

    @property
    def constraint_operator(self) -> LinearOperator:
        if self._constraint_operator is None:
            raise AttributeError("This subspace is not defined by a linear equation.")
        return self._constraint_operator

    @property
    def constraint_value(self) -> Vector:
        if self._constraint_value is None:
            raise AttributeError("This subspace is not defined by a linear equation.")
        return self._constraint_value

    def project(self, x: Vector) -> Vector:
        diff = self.domain.subtract(x, self.translation)
        proj_diff = self.projector(diff)
        return self.domain.add(self.translation, proj_diff)

    def is_element(self, x: Vector, rtol: float = 1e-6) -> bool:
        proj = self.project(x)
        diff = self.domain.subtract(x, proj)
        norm_diff = self.domain.norm(diff)
        norm_x = self.domain.norm(x)
        scale = norm_x if norm_x > 1e-12 else 1.0
        return norm_diff <= rtol * scale

    @classmethod
    def from_linear_equation(
        cls,
        operator: LinearOperator,
        value: Vector,
        solver: Optional[LinearSolver] = None,
        preconditioner: Optional[LinearOperator] = None,
    ) -> AffineSubspace:
        """Constructs the subspace {u | B(u) = w}."""
        domain = operator.domain
        G = operator @ operator.adjoint

        if solver is None:
            solver = CholeskySolver(galerkin=True)

        if isinstance(solver, IterativeLinearSolver):
            G_inv = solver(G, preconditioner=preconditioner)
        else:
            G_inv = solver(G)

        intermediate = G_inv(value)
        translation = operator.adjoint(intermediate)
        P_perp_op = operator.adjoint @ G_inv @ operator

        def mapping(x: Any) -> Any:
            return domain.subtract(x, P_perp_op(x))

        projector = OrthogonalProjector(domain, mapping, complement_projector=P_perp_op)

        return cls(
            projector, translation, constraint_operator=operator, constraint_value=value
        )

    @classmethod
    def from_tangent_basis(
        cls,
        domain: HilbertSpace,
        basis_vectors: List[Vector],
        translation: Optional[Vector] = None,
        orthonormalize: bool = True,
    ) -> AffineSubspace:
        """
        Constructs the subspace passing through 'translation' with the given
        tangent basis.

        Note: This does not define a constraint equation B(u)=w, so it cannot
        be used directly with ConstrainedLinearBayesianInversion.
        """
        projector = OrthogonalProjector.from_basis(
            domain, basis_vectors, orthonormalize=orthonormalize
        )
        return cls(projector, translation)

    @classmethod
    def from_complement_basis(
        cls,
        domain: HilbertSpace,
        basis_vectors: List[Vector],
        translation: Optional[Vector] = None,
        orthonormalize: bool = True,
    ) -> AffineSubspace:
        """
        Constructs the subspace orthogonal to the given basis, passing through
        'translation'.

        This automatically constructs the constraint operator B such that
        the subspace is {u | B(u) = B(translation)}.
        """
        # 1. Orthonormalize basis for stability
        if orthonormalize:
            e_vectors = domain.gram_schmidt(basis_vectors)
        else:
            e_vectors = basis_vectors

        # 2. Construct Projector P_perp
        complement_projector = OrthogonalProjector.from_basis(
            domain, e_vectors, orthonormalize=False  # Already done
        )

        # 3. Construct Projector P = I - P_perp
        def mapping(x: Any) -> Any:
            return domain.subtract(x, complement_projector(x))

        projector = OrthogonalProjector(
            domain, mapping, complement_projector=complement_projector
        )

        # 4. Construct Constraint Operator B implicitly defined by the basis
        # B: E -> R^k, u -> [<e_1, u>, ..., <e_k, u>]
        # Since e_i are orthonormal, BB* = I, which is perfect for solvers.
        codomain = EuclideanSpace(len(e_vectors))

        def constraint_mapping(u: Vector) -> np.ndarray:
            return np.array([domain.inner_product(e, u) for e in e_vectors])

        def constraint_adjoint(c: np.ndarray) -> Vector:
            # sum c_i e_i
            res = domain.zero
            for i, e in enumerate(e_vectors):
                domain.axpy(c[i], e, res)
            return res

        B = LinearOperator(
            domain, codomain, constraint_mapping, adjoint_mapping=constraint_adjoint
        )

        # 5. Determine Constraint Value w = B(translation)
        # If translation is None (zero), w is zero.
        if translation is None:
            _translation = domain.zero
            w = codomain.zero
        else:
            _translation = translation
            w = B(_translation)

        return cls(projector, _translation, constraint_operator=B, constraint_value=w)


class LinearSubspace(AffineSubspace):
    """
    Represents a linear subspace (an affine subspace passing through zero).
    """

    def __init__(self, projector: OrthogonalProjector) -> None:
        super().__init__(projector, translation=None)

    @property
    def complement(self) -> LinearSubspace:
        op_perp = self.projector.complement
        if isinstance(op_perp, OrthogonalProjector):
            return LinearSubspace(op_perp)
        p_perp = OrthogonalProjector(self.domain, op_perp._mapping)
        return LinearSubspace(p_perp)

    @classmethod
    def from_kernel(
        cls, operator: LinearOperator, solver: Optional[LinearSolver] = None
    ) -> LinearSubspace:
        affine = AffineSubspace.from_linear_equation(
            operator, operator.codomain.zero, solver
        )
        instance = cls(affine.projector)
        instance._constraint_operator = operator
        instance._constraint_value = operator.codomain.zero
        return instance

    @classmethod
    def from_basis(
        cls,
        domain: HilbertSpace,
        basis_vectors: List[Vector],
        orthonormalize: bool = True,
    ) -> LinearSubspace:
        projector = OrthogonalProjector.from_basis(
            domain, basis_vectors, orthonormalize=orthonormalize
        )
        return cls(projector)

    @classmethod
    def from_complement_basis(
        cls,
        domain: HilbertSpace,
        basis_vectors: List[Vector],
        orthonormalize: bool = True,
    ) -> LinearSubspace:
        affine = AffineSubspace.from_complement_basis(
            domain, basis_vectors, translation=None, orthonormalize=orthonormalize
        )
        # Copy constraint info from the affine instance
        instance = cls(affine.projector)
        instance._constraint_operator = affine.constraint_operator
        instance._constraint_value = affine.constraint_value
        return instance
