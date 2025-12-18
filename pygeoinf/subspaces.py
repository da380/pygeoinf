"""
Defines classes for representing affine and linear subspaces.

The primary abstraction is the `AffineSubspace`, which represents a subset of
a Hilbert space defined by a translation and a closed linear tangent space.
"""

from __future__ import annotations
from typing import List, Optional, Any, Callable, TYPE_CHECKING
import numpy as np
import warnings

from .linear_operators import LinearOperator
from .hilbert_space import HilbertSpace, Vector, EuclideanSpace
from .linear_solvers import LinearSolver, CholeskySolver, IterativeLinearSolver

if TYPE_CHECKING:
    from .gaussian_measure import GaussianMeasure


class OrthogonalProjector(LinearOperator):
    """
    Internal engine for subspace projections.
    Represents an orthogonal projection operator P = P* = P^2.
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
        """Constructs a projector P onto the span of the provided basis vectors."""
        if not basis_vectors:
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
        solver: Optional[LinearSolver] = None,
        preconditioner: Optional[LinearOperator] = None,
    ) -> None:
        """
        Initializes the AffineSubspace.
        """
        self._projector = projector

        if translation is None:
            self._translation = projector.domain.zero
        else:
            if not projector.domain.is_element(translation):
                raise ValueError("Translation vector not in domain.")
            self._translation = translation

        self._constraint_operator = constraint_operator
        self._constraint_value = constraint_value

        # Logic: If explicit equation exists, default to Cholesky.
        # If implicit, leave None (requires robust solver from user).
        if self._constraint_operator is not None and solver is None:
            self._solver = CholeskySolver(galerkin=True)
        else:
            self._solver = solver

        self._preconditioner = preconditioner

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
    def has_explicit_equation(self) -> bool:
        """True if defined by B(u)=w, False if defined only by geometry."""
        return self._constraint_operator is not None

    @property
    def constraint_operator(self) -> LinearOperator:
        """
        Returns B for {u | B(u)=w}.
        Falls back to (I - P) if no explicit operator exists.
        """
        if self._constraint_operator is None:
            return self._projector.complement
        return self._constraint_operator

    @property
    def constraint_value(self) -> Vector:
        """
        Returns w for {u | B(u)=w}.
        Falls back to (I - P)x0 if no explicit operator exists.
        """
        if self._constraint_value is None:
            complement = self._projector.complement
            return complement(self._translation)
        return self._constraint_value

    def project(self, x: Vector) -> Vector:
        """Orthogonally projects x onto the affine subspace."""
        diff = self.domain.subtract(x, self.translation)
        proj_diff = self.projector(diff)
        return self.domain.add(self.translation, proj_diff)

    def is_element(self, x: Vector, rtol: float = 1e-6) -> bool:
        """Returns True if x lies in the subspace."""
        proj = self.project(x)
        diff = self.domain.subtract(x, proj)
        norm_diff = self.domain.norm(diff)
        norm_x = self.domain.norm(x)
        scale = norm_x if norm_x > 1e-12 else 1.0
        return norm_diff <= rtol * scale

    def condition_gaussian_measure(
        self, prior: GaussianMeasure, geometric: bool = False
    ) -> GaussianMeasure:
        """
        Conditions a Gaussian measure on this subspace.
        """
        if geometric:
            # Geometric Projection: u -> P(u - x0) + x0
            # Affine Map: u -> P(u) + (I-P)x0
            shift = self.domain.subtract(
                self.translation, self.projector(self.translation)
            )
            return prior.affine_mapping(operator=self.projector, translation=shift)

        else:
            # Bayesian Conditioning: u | B(u)=w

            # Check for singular implicit operator usage
            if not self.has_explicit_equation and self._solver is None:
                raise ValueError(
                    "This subspace defines the constraint implicitly as (I-P)u = (I-P)x0. "
                    "The operator (I-P) is singular. You must provide a solver "
                    "capable of handling singular systems (e.g. MinRes) to the "
                    "AffineSubspace constructor."
                )

            # Local imports
            from .forward_problem import LinearForwardProblem
            from .linear_bayesian import LinearBayesianInversion

            solver = self._solver
            preconditioner = self._preconditioner

            constraint_problem = LinearForwardProblem(self.constraint_operator)
            constraint_inversion = LinearBayesianInversion(constraint_problem, prior)

            return constraint_inversion.model_posterior_measure(
                self.constraint_value, solver, preconditioner=preconditioner
            )

    @classmethod
    def from_linear_equation(
        cls,
        operator: LinearOperator,
        value: Vector,
        solver: Optional[LinearSolver] = None,
        preconditioner: Optional[LinearOperator] = None,
    ) -> AffineSubspace:
        """Constructs subspace from B(u)=w."""
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
            projector,
            translation,
            constraint_operator=operator,
            constraint_value=value,
            solver=solver,
            preconditioner=preconditioner,
        )

    @classmethod
    def from_tangent_basis(
        cls,
        domain: HilbertSpace,
        basis_vectors: List[Vector],
        translation: Optional[Vector] = None,
        orthonormalize: bool = True,
        solver: Optional[LinearSolver] = None,
        preconditioner: Optional[LinearOperator] = None,
    ) -> AffineSubspace:
        """
        Constructs an affine subspace from a translation and a basis for the tangent space.

        This method defines the subspace geometrically. The constraint is implicit:
        (I - P)u = (I - P)x0.

        Args:
            domain: The Hilbert space.
            basis_vectors: Basis vectors for the tangent space V.
            translation: A point x0 in the subspace.
            orthonormalize: If True, orthonormalizes the basis.
            solver: A linear solver capable of handling the singular operator (I-P).
                    Required if you intend to use this subspace for Bayesian conditioning.
            preconditioner: Optional preconditioner for the solver.
        """
        if solver is None:
            warnings.warn(
                "Constructing a subspace from a tangent basis without a solver. "
                "This defines an implicit constraint with a singular operator. "
                "Bayesian conditioning will fail; geometric projection remains available.",
                UserWarning,
                stacklevel=2,
            )

        projector = OrthogonalProjector.from_basis(
            domain, basis_vectors, orthonormalize=orthonormalize
        )

        return cls(projector, translation, solver=solver, preconditioner=preconditioner)

    @classmethod
    def from_complement_basis(
        cls,
        domain: HilbertSpace,
        basis_vectors: List[Vector],
        translation: Optional[Vector] = None,
        orthonormalize: bool = True,
    ) -> AffineSubspace:
        """
        Constructs subspace from complement basis.
        Constraint is explicit: <u, e_i> = <x0, e_i>.
        """
        if orthonormalize:
            e_vectors = domain.gram_schmidt(basis_vectors)
        else:
            e_vectors = basis_vectors

        complement_projector = OrthogonalProjector.from_basis(
            domain, e_vectors, orthonormalize=False
        )

        def mapping(x: Any) -> Any:
            return domain.subtract(x, complement_projector(x))

        projector = OrthogonalProjector(
            domain, mapping, complement_projector=complement_projector
        )

        codomain = EuclideanSpace(len(e_vectors))

        def constraint_mapping(u: Vector) -> np.ndarray:
            return np.array([domain.inner_product(e, u) for e in e_vectors])

        def constraint_adjoint(c: np.ndarray) -> Vector:
            res = domain.zero
            for i, e in enumerate(e_vectors):
                domain.axpy(c[i], e, res)
            return res

        B = LinearOperator(
            domain, codomain, constraint_mapping, adjoint_mapping=constraint_adjoint
        )

        if translation is None:
            _translation = domain.zero
            w = codomain.zero
        else:
            _translation = translation
            w = B(_translation)

        solver = CholeskySolver(galerkin=True)

        return cls(
            projector,
            _translation,
            constraint_operator=B,
            constraint_value=w,
            solver=solver,
        )


class LinearSubspace(AffineSubspace):
    """
    Represents a linear subspace (an affine subspace passing through the origin).
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
        cls,
        operator: LinearOperator,
        solver: Optional[LinearSolver] = None,
        preconditioner: Optional[LinearOperator] = None,
    ) -> LinearSubspace:
        affine = AffineSubspace.from_linear_equation(
            operator, operator.codomain.zero, solver, preconditioner
        )
        instance = cls(affine.projector)
        instance._constraint_operator = operator
        instance._constraint_value = operator.codomain.zero
        instance._solver = affine._solver
        instance._preconditioner = preconditioner
        return instance

    @classmethod
    def from_basis(
        cls,
        domain: HilbertSpace,
        basis_vectors: List[Vector],
        orthonormalize: bool = True,
        solver: Optional[LinearSolver] = None,
        preconditioner: Optional[LinearOperator] = None,
    ) -> LinearSubspace:
        projector = OrthogonalProjector.from_basis(
            domain, basis_vectors, orthonormalize=orthonormalize
        )
        instance = cls(projector)
        instance._solver = solver
        instance._preconditioner = preconditioner
        return instance

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
        instance = cls(affine.projector)
        instance._constraint_operator = affine.constraint_operator
        instance._constraint_value = affine.constraint_value
        instance._solver = affine._solver
        return instance
