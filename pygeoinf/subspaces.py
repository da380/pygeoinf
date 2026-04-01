"""
Defines classes for representing affine and linear subspaces, including
hyperplanes and half-spaces.

The primary abstraction is the `AffineSubspace`, which represents a subset of
a Hilbert space defined by a translation and a closed linear tangent space.
This module integrates with the `subset` module, allowing subspaces to be
treated as standard geometric sets.
"""

from __future__ import annotations
from typing import List, Optional, Any, Callable, TYPE_CHECKING
import warnings
import numpy as np

from .affine_operators import AffineOperator
from .linear_operators import LinearOperator
from .hilbert_space import HilbertSpace, Vector, EuclideanSpace
from .linear_solvers import LinearSolver, CholeskySolver, IterativeLinearSolver
from .subsets import Subset, EmptySet

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
        """
        Initializes the orthogonal projector.

        Args:
            domain: The Hilbert space on which the projector acts.
            mapping: The function implementing the projection P(x).
            complement_projector: An optional LinearOperator representing (I - P).
                If provided, it avoids re-computing the complement when requested.
        """
        super().__init__(domain, domain, mapping, adjoint_mapping=mapping)
        self._complement_projector = complement_projector

    @property
    def complement(self) -> LinearOperator:
        """
        Returns the projector onto the orthogonal complement (I - P).

        If a complement projector was not provided at initialization, one is
        constructed automatically as the difference between the identity and self.
        """
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
        """
        Constructs a projector P onto the span of the provided basis vectors.

        Args:
            domain: The Hilbert space.
            basis_vectors: A list of vectors spanning the subspace.
            orthonormalize: If True, performs Gram-Schmidt orthonormalization
                on the basis vectors before constructing the projector.
                If False, assumes the basis is already orthonormal.

        Returns:
            An OrthogonalProjector instance.
        """
        if not basis_vectors:
            return domain.zero_operator(domain)

        if orthonormalize:
            e_vectors = domain.gram_schmidt(basis_vectors)
        else:
            e_vectors = basis_vectors

        # P = sum (v_i x v_i)
        tensor_op = LinearOperator.self_adjoint_from_tensor_product(domain, e_vectors)
        return cls(domain, tensor_op)


class AffineSubspace(Subset):
    """
    Represents an affine subspace A = x0 + V.

    This class serves two primary roles:
    1. A geometric subset that can project points and check membership.
    2. A constraint definition for Bayesian inversion (conditioning a Gaussian
       measure on the subspace).
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

        Args:
            projector: The orthogonal projector P onto the tangent space V.
            translation: A vector x0 in the subspace. Defaults to the origin.
            constraint_operator: The operator B defining the subspace implicitly
                as {u | B(u) = w}. Used for Bayesian conditioning.
            constraint_value: The RHS vector w for the implicit definition.
            solver: A LinearSolver used to invert the constraint operator during
                conditioning. If None, defaults to a CholeskySolver if an
                explicit constraint operator is provided.
            preconditioner: An optional preconditioner for iterative solvers.
        """
        super().__init__(projector.domain)
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
    def translation(self) -> Vector:
        """Returns the translation vector x0."""
        return self._translation

    @property
    def projector(self) -> OrthogonalProjector:
        """Returns the orthogonal projector P onto the tangent space."""
        return self._projector

    @property
    def solver(self) -> Optional[LinearSolver]:
        """Returns the linear solver associated with this subspace."""
        return self._solver

    @property
    def preconditioner(self) -> Optional[LinearOperator]:
        """Returns the preconditioner associated with the solver, if any."""
        return self._preconditioner

    @property
    def tangent_space(self) -> LinearSubspace:
        """Returns the LinearSubspace V parallel to this affine subspace."""
        return LinearSubspace(self._projector)

    def get_tangent_basis(self) -> List[Vector]:
        r"""
        Returns an orthonormal basis for the tangent space of this affine subspace.

        Extracts orthonormal basis vectors that span the tangent space $V$ by
        applying the projector $P$ to each standard basis vector $e_i$, then
        performing a tolerant Gram-Schmidt orthogonalisation to discard linearly
        dependent projections.

        Returns:
            List[Vector]: Orthonormal basis vectors for the tangent space.
        """
        tolerance = 1e-10
        domain = self.domain

        candidates = []
        for i in range(domain.dim):
            e_i = domain.basis_vector(i)
            v_i = self._projector(e_i)
            if domain.norm(v_i) > tolerance:
                candidates.append(v_i)

        basis = []
        for v in candidates:
            w = domain.copy(v)
            for b in basis:
                w = domain.axpy(-domain.inner_product(w, b), b, w)
            norm_w = domain.norm(w)
            if norm_w > tolerance:
                domain.ax(1.0 / norm_w, w)
                basis.append(w)

        return basis

    @property
    def has_explicit_equation(self) -> bool:
        """True if defined by B(u)=w, False if defined only by geometry."""
        return self._constraint_operator is not None

    @property
    def constraint_operator(self) -> LinearOperator:
        """
        Returns the operator B defining the subspace as {u | B(u)=w}.

        If no explicit operator was provided (geometric construction), this
        falls back to the complement projector (I - P).
        """
        if self._constraint_operator is None:
            return self._projector.complement
        return self._constraint_operator

    @property
    def constraint_value(self) -> Vector:
        """
        Returns the value w defining the subspace as {u | B(u)=w}.

        If no explicit operator was provided, this falls back to (I - P)x0.
        """
        if self._constraint_value is None:
            complement = self._projector.complement
            return complement(self._translation)
        return self._constraint_value

    @property
    def pseudo_inverse(self) -> LinearOperator:
        """
        Returns the right pseudo-inverse operator B^dagger = B* (B B*)^{-1}.
        """
        if not self.has_explicit_equation:
            raise ValueError(
                "Cannot compute pseudo-inverse without an explicit equation."
            )

        B = self.constraint_operator
        G = B @ B.adjoint

        if isinstance(self.solver, IterativeLinearSolver):
            G_inv = self.solver(G, preconditioner=self.preconditioner)
        else:
            G_inv = self.solver(G)

        return B.adjoint @ G_inv

    @property
    def projection_operator(self) -> AffineOperator:
        """
        Returns the affine orthogonal projection operator onto the subspace.
        P_A(x) = P(x) + (I - P)x_0
        """
        linear_part = self.projector

        # The translation term is x0 - P(x0)
        translation_part = self.domain.subtract(
            self.translation, self.projector(self.translation)
        )

        return AffineOperator(linear_part, translation_part)

    @property
    def boundary(self) -> Subset:
        """
        Returns the boundary of the affine subspace.

        Geometrically, an affine subspace (like a line or plane) is a closed
        manifold without a boundary. Returns EmptySet.
        """
        return EmptySet(self.domain)

    @classmethod
    def from_linear_equation(
        cls,
        operator: LinearOperator,
        value: Vector,
        solver: Optional[LinearSolver] = None,
        preconditioner: Optional[LinearOperator] = None,
    ) -> AffineSubspace:
        """
        Constructs a subspace defined by the linear equation B(u) = w.

        Args:
            operator: The linear operator B.
            value: The RHS vector w.
            solver: Solver used to invert the Gram matrix (B B*) during
                construction and later conditioning. Defaults to CholeskySolver.
            preconditioner: Optional preconditioner for iterative solvers.
        """
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
        Constructs a subspace defined by orthogonality to a set of complement basis vectors.

        The subspace is defined as {u | <u - x0, v_i> = 0} for all v_i in basis.
        This provides an explicit constraint operator B where B(u)_i = <u, v_i>.

        Args:
            domain: The Hilbert space.
            basis_vectors: Basis vectors for the orthogonal complement.
            translation: A point x0 in the subspace.
            orthonormalize: If True, orthonormalizes the complement basis.
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

    @classmethod
    def from_hyperplanes(
        cls,
        hyperplanes: List["Subset"],
        solver: Optional[LinearSolver] = None,
        preconditioner: Optional[LinearOperator] = None,
    ) -> AffineSubspace:
        """
        Constructs an affine subspace as the intersection of hyperplanes.

        Each hyperplane is defined as {x | ⟨a_i, x⟩ = b_i}. The intersection
        of m hyperplanes defines an affine subspace of codimension m (assuming
        the normal vectors are linearly independent).

        Args:
            hyperplanes: A list of HyperPlane objects from subsets module.
                All hyperplanes must have the same domain.
            solver: Solver used to invert the Gram matrix (B B*) during
                construction. Defaults to CholeskySolver.
            preconditioner: Optional preconditioner for iterative solvers.

        Returns:
            AffineSubspace: The affine subspace defined by the intersection.

        Raises:
            ValueError: If hyperplanes list is empty or domains don't match.
            ImportError: If hyperplanes don't have the required attributes.
        """
        if not hyperplanes:
            raise ValueError("At least one hyperplane is required.")

        domain = hyperplanes[0].domain

        for hp in hyperplanes:
            if hp.domain != domain:
                raise ValueError("All hyperplanes must have the same domain.")

        try:
            normal_vectors = [hp.normal_vector for hp in hyperplanes]
            offsets = [hp.offset for hp in hyperplanes]
        except AttributeError as e:
            raise ImportError(
                f"Hyperplane objects must have 'normal_vector' and 'offset' attributes. "
                f"Error: {e}"
            )

        m = len(hyperplanes)
        codomain = EuclideanSpace(m)

        def constraint_mapping(x: Vector) -> np.ndarray:
            return np.array([
                domain.inner_product(a_i, x) for a_i in normal_vectors
            ])

        def constraint_adjoint(c: np.ndarray) -> Vector:
            result = domain.zero
            for i, a_i in enumerate(normal_vectors):
                result = domain.axpy(c[i], a_i, result)
            return result

        B = LinearOperator(
            domain, codomain, constraint_mapping, adjoint_mapping=constraint_adjoint
        )

        w = np.array(offsets)

        return cls.from_linear_equation(B, w, solver=solver, preconditioner=preconditioner)

    def to_hyperplanes(self) -> List["Subset"]:
        """
        Decomposes this affine subspace into a minimal set of hyperplanes.

        Returns a list of HyperPlane objects whose intersection equals this
        affine subspace. The number of hyperplanes equals the codimension of
        the subspace (i.e., the rank of the constraint operator).

        Returns:
            List[HyperPlane]: The minimal set of hyperplanes defining this subspace.

        Raises:
            ValueError: If the subspace does not have an explicit constraint operator.
            ImportError: If HyperPlane class is not available.
        """
        try:
            from .subsets import HyperPlane
        except ImportError as e:
            raise ImportError(
                f"Cannot import HyperPlane from subsets module. Error: {e}"
            )

        if not self.has_explicit_equation:
            raise ValueError(
                "Cannot convert to hyperplanes: this subspace was constructed "
                "geometrically without an explicit constraint operator."
            )

        B = self.constraint_operator
        w = self.constraint_value
        codomain = B.codomain

        if not isinstance(codomain, EuclideanSpace):
            raise ValueError(
                "Cannot convert to hyperplanes: constraint operator codomain "
                "must be a finite-dimensional Euclidean space."
            )

        m = codomain.dim
        hyperplanes = []

        for i in range(m):
            e_i = np.zeros(m)
            e_i[i] = 1.0
            a_i = B.adjoint(e_i)
            b_i = float(w[i]) if isinstance(w, np.ndarray) else w
            hyperplane = HyperPlane(self.domain, a_i, b_i)
            hyperplanes.append(hyperplane)

        return hyperplanes

    def project(self, x: Vector) -> Vector:
        """
        Orthogonally projects a vector x onto the affine subspace.

        Formula: P_A(x) = P(x - x0) + x0
        """
        diff = self.domain.subtract(x, self.translation)
        proj_diff = self.projector(diff)
        return self.domain.add(self.translation, proj_diff)

    def is_element(self, x: Vector, /, *, rtol: float = 1e-6) -> bool:
        """
        Returns True if the vector x lies within the subspace.

        Checks if the projection residual ||x - P_A(x)|| is small relative
        to the norm of x (or 1.0).

        Args:
            x: The vector to check.
            rtol: Relative tolerance for the residual check.
        """
        proj = self.project(x)
        diff = self.domain.subtract(x, proj)
        norm_diff = self.domain.norm(diff)

        # Scale tolerance by norm of x to handle units/scaling, consistent with Sphere/Ball
        norm_x = self.domain.norm(x)
        scale = max(1.0, norm_x)
        return norm_diff <= rtol * scale

    def with_translation(self, new_translation: Vector) -> AffineSubspace:
        """
        Returns a new AffineSubspace parallel to this one, shifted to pass
        through the new translation vector.
        """
        if not self.domain.is_element(new_translation):
            raise ValueError("Translation vector must be in the domain.")

        # If we have an explicit B operator, we must update the w value to match the new translation
        new_constraint_value = None
        if self.has_explicit_equation:
            new_constraint_value = self.constraint_operator(new_translation)

        return AffineSubspace(
            self.projector,
            translation=new_translation,
            constraint_operator=self._constraint_operator,
            constraint_value=new_constraint_value,
            solver=self.solver,
            preconditioner=self.preconditioner,
        )

    def with_constraint_value(self, new_value: Vector) -> AffineSubspace:
        """
        Returns a new AffineSubspace parallel to this one, shifted to satisfy
        the new explicit constraint equation B(u) = new_value.
        """
        if not self.has_explicit_equation:
            raise ValueError(
                "Cannot shift by constraint value without an explicit constraint operator."
            )

        # Calculate the new base translation using the pseudo-inverse
        new_translation = self.pseudo_inverse(new_value)

        return AffineSubspace(
            self.projector,
            translation=new_translation,
            constraint_operator=self._constraint_operator,
            constraint_value=new_value,
            solver=self.solver,
            preconditioner=self.preconditioner,
        )

    def condition_gaussian_measure(
        self, prior: GaussianMeasure, geometric: bool = False
    ) -> GaussianMeasure:
        """
        Conditions a Gaussian measure on this subspace.

        Args:
            prior: The prior Gaussian measure.
            geometric: If True, performs a geometric projection of the measure
                (equivalent to conditioning on "measurement = truth" with infinite
                precision, effectively squashing the distribution onto the
                subspace).
                If False (default), performs standard Bayesian conditioning
                using the constraint equation B(u) = w.

        Returns:
            The posterior (conditioned) GaussianMeasure.

        Raises:
            ValueError: If geometric=False and the subspace was constructed
                without a solver capable of handling the constraint operator.
        """

        if geometric:
            # Geometric Projection: u -> P(u - x0) + x0
            # Affine Map: u -> P(u) + (I-P)x0

            # Optimization: If it's a linear subspace, the shift is exactly zero.
            # Pass None to preserve the zero-expectation optimization in the measure.
            if isinstance(self, LinearSubspace):
                shift = None
            else:
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

            # Local imports to avoid circular dependency
            from .forward_problem import LinearForwardProblem
            from .linear_bayesian import LinearBayesianInversion

            solver = self._solver
            preconditioner = self._preconditioner

            constraint_problem = LinearForwardProblem(self.constraint_operator)
            constraint_inversion = LinearBayesianInversion(constraint_problem, prior)

            return constraint_inversion.model_posterior_measure(
                self.constraint_value, solver, preconditioner=preconditioner
            )


class LinearSubspace(AffineSubspace):
    """
    Represents a linear subspace (an affine subspace passing through the origin).
    """

    def __init__(self, projector: OrthogonalProjector) -> None:
        """
        Initializes the LinearSubspace.

        Args:
            projector: The orthogonal projector P onto the subspace.
        """
        super().__init__(projector, translation=None)

    @property
    def complement(self) -> LinearSubspace:
        """
        Returns the orthogonal complement of this subspace as a new LinearSubspace.
        """
        op_perp = self.projector.complement
        if isinstance(op_perp, OrthogonalProjector):
            return LinearSubspace(op_perp)
        # Wrap if the complement isn't strictly an OrthogonalProjector instance
        p_perp = OrthogonalProjector(self.domain, op_perp._mapping)
        return LinearSubspace(p_perp)

    @classmethod
    def from_kernel(
        cls,
        operator: LinearOperator,
        solver: Optional[LinearSolver] = None,
        preconditioner: Optional[LinearOperator] = None,
    ) -> LinearSubspace:
        """
        Constructs the subspace corresponding to the kernel (null space) of an operator.
        K = {u | A(u) = 0}.

        Args:
            operator: The operator A.
            solver: Solver used for the Gram matrix (A A*).
            preconditioner: Optional preconditioner.
        """
        affine = AffineSubspace.from_linear_equation(
            operator, operator.codomain.zero, solver, preconditioner
        )
        instance = cls(affine.projector)
        instance._constraint_operator = operator
        instance._constraint_value = operator.codomain.zero
        instance._solver = affine.solver
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
        """
        Constructs a linear subspace from a set of basis vectors.

        Args:
            domain: The Hilbert space.
            basis_vectors: List of vectors spanning the subspace.
            orthonormalize: Whether to orthonormalize the basis.
            solver: Optional solver for implicit constraints (see AffineSubspace.from_tangent_basis).
            preconditioner: Optional preconditioner.
        """
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
        """
        Constructs a linear subspace defined by orthogonality to a complement basis.
        S = {u | <u, v_i> = 0}.

        Args:
            domain: The Hilbert space.
            basis_vectors: Basis vectors for the complement.
            orthonormalize: Whether to orthonormalize the complement basis.
        """
        affine = AffineSubspace.from_complement_basis(
            domain, basis_vectors, translation=None, orthonormalize=orthonormalize
        )
        instance = cls(affine.projector)
        instance._constraint_operator = affine.constraint_operator
        instance._constraint_value = affine.constraint_value
        instance._solver = affine.solver
        return instance
