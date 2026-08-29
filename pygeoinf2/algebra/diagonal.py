"""
Operators that are diagonal in a space's own basis.

This is the v2 home for what v1 calls ``InvariantLinearAutomorphism``: an
operator stored as its eigenvalues, closed under the algebra, and carrying a
complete functional calculus evaluated pointwise on the spectrum.

It is a **class**, not a trait, because it carries data. That is the rule from
DESIGN.md section 4: mathematical properties are traits, representations are
classes. Self-adjointness is a trait this class happens to claim; being
diagonal is a fact about how it is stored.

Self-adjointness holds only when the space's metric is also diagonal. With
component matrix ``diag(d)`` the Galerkin matrix is ``G diag(d)``, symmetric
exactly when ``G`` and ``diag(d)`` commute — so, for general ``d``, exactly
when ``G`` is diagonal. That is why ``CoordinateSpace.has_diagonal_metric``
exists, and why this class checks it rather than assuming it.

The functional calculus is a separate question and takes a separate answer.
An eigendecomposition is a property of the map: the basis vectors are
eigenvectors and ``d`` is the spectrum on *any* inner product, so ``f(A)`` is
``diag(f(d))`` however the metric is shaped. The calculus therefore gates on
the spectrum — ``sqrt`` wants ``d >= 0`` — and leaves the metric to decide
only what the *result* claims. See ``_require_spectrum``.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from ..traits import Traits, close, compose_traits, scale_traits, sum_traits
from .operators import LinearOperator, Operator
from .spaces import CoordinateSpace

__all__ = ["DiagonalLinearOperator"]


class DiagonalLinearOperator[V](LinearOperator[V, V]):
    """An operator stored as its eigenvalues in a space's basis."""

    def __init__(
        self,
        domain: CoordinateSpace[V],
        eigenvalues: np.ndarray,
        /,
        *,
        traits: Traits = Traits.NONE,
    ) -> None:
        """
        Args:
            domain: the space, whose basis the operator is diagonal in.
            eigenvalues: one value per basis direction.
            traits: extra claims. Symmetry and definiteness are deduced from
                the values and the metric, so they rarely need supplying.
        """
        if not isinstance(domain, CoordinateSpace):
            raise TypeError(
                f"A diagonal operator needs a basis to be diagonal in, and "
                f"{type(domain).__name__} provides no coordinate map."
            )
        values = np.asarray(eigenvalues, dtype=float)
        if values.shape != (domain.dim,):
            raise ValueError(
                f"Got {values.shape} eigenvalues for a space of dimension "
                f"{domain.dim}."
            )
        super().__init__(
            domain, domain, traits=close(traits | self._deduce_traits(domain, values))
        )
        self._eigenvalues = values

    @staticmethod
    def _deduce_traits(domain: CoordinateSpace, values: np.ndarray) -> Traits:
        """Read off the traits the eigenvalues and metric already imply."""
        if not domain.has_diagonal_metric:
            # Diagonal in the components, but not symmetric in the metric.
            return Traits.NONE
        result = Traits.SELF_ADJOINT
        if np.all(values >= 0.0):
            result |= Traits.POSITIVE_SEMIDEFINITE
        if np.all(values > 0.0):
            result |= Traits.POSITIVE_DEFINITE
        if np.all(values != 0.0):
            result |= Traits.INVERTIBLE
        if np.all(np.abs(values) == 1.0):
            result |= Traits.UNITARY
        if np.all((values == 0.0) | (values == 1.0)):
            result |= Traits.IDEMPOTENT
        return result

    # ----------------------------------------------------------------- #
    #                              Structure                            #
    # ----------------------------------------------------------------- #

    @property
    def eigenvalues(self) -> np.ndarray:
        """The diagonal, one value per basis direction."""
        return self._eigenvalues

    @property
    def trace(self) -> float:
        """The sum of the eigenvalues, exactly."""
        return float(np.sum(self._eigenvalues))

    def _value(self, x: V) -> V:
        space: CoordinateSpace = self.domain
        return space.from_components(self._eigenvalues * space.to_components(x))

    def _adjoint_value(self, y: V) -> V:
        if Traits.SELF_ADJOINT & self.traits:
            return self._value(y)
        # Without a diagonal metric the adjoint is G^-1 diag(d) G, which is not
        # diagonal, so it is not this class's business to represent it.
        space: CoordinateSpace = self.domain
        components = space.to_components(y)
        return space.from_components(
            space.solve_gram(self._eigenvalues * space.apply_gram(components))
        )

    def __repr__(self) -> str:
        return f"DiagonalLinearOperator(dim={self.domain.dim})"

    # ----------------------------------------------------------------- #
    #                      Structure-preserving algebra                 #
    # ----------------------------------------------------------------- #
    #
    # The specialisation protocol of DESIGN.md 5.4. Without these, the sum of
    # two diagonal operators would be a generic node and would lose both the
    # eigenvalues and the functional calculus that makes them worth storing.

    def _rebuild(self, eigenvalues: np.ndarray) -> DiagonalLinearOperator[V]:
        """Build an operator of this class. Subclasses override to stay in theirs."""
        return DiagonalLinearOperator(self.domain, eigenvalues)

    # Each combination carries the traits that follow from the operands',
    # on top of what the new eigenvalues let the constructor deduce. On a
    # diagonal metric the two agree; on any other the constructor deduces
    # nothing, and without this a product of two self-adjoint diagonals --
    # ``(sigma I)(sigma I)``, the covariance of every isotropic measure --
    # came out claiming nothing at all.

    def _combine_add(self, other: Operator) -> Operator | None:
        if isinstance(other, DiagonalLinearOperator) and other.domain == self.domain:
            return self._rebuild(self._eigenvalues + other.eigenvalues).with_traits(
                sum_traits(self.traits, other.traits)
            )
        return None

    def _combine_radd(self, other: Operator) -> Operator | None:
        return self._combine_add(other)

    def _combine_compose(self, other: Operator) -> Operator | None:
        if isinstance(other, DiagonalLinearOperator) and other.domain == self.domain:
            traits = compose_traits(self.traits, other.traits, square=True)
            # Diagonal in one basis, the two commute, so a product of
            # self-adjoint factors is self-adjoint, and definiteness carries.
            shared = self.traits & other.traits
            traits |= shared & (
                Traits.SELF_ADJOINT
                | Traits.POSITIVE_SEMIDEFINITE
                | Traits.POSITIVE_DEFINITE
                | Traits.INVERTIBLE
            )
            return self._rebuild(self._eigenvalues * other.eigenvalues).with_traits(
                traits
            )
        return None

    def _combine_rcompose(self, other: Operator) -> Operator | None:
        return self._combine_compose(other)

    def _combine_scale(self, alpha: float) -> Operator | None:
        return self._rebuild(alpha * self._eigenvalues).with_traits(
            scale_traits(self.traits, alpha, square=True)
        )

    # ----------------------------------------------------------------- #
    #                         Functional calculus                       #
    # ----------------------------------------------------------------- #

    def apply_function(
        self, function: Callable[[np.ndarray], np.ndarray], /
    ) -> DiagonalLinearOperator[V]:
        """``f(A)``, evaluated pointwise on the spectrum.

        Exact, and costs one array operation. This is the fast path that
        ``numerics.functional_calculus.operator_function`` dispatches to.

        Args:
            function: applied to the whole eigenvalue array at once, so it
                must be vectorised and must not change the shape.

        Returns:
            The diagonal operator with the transformed spectrum.

        Raises:
            ValueError: if the function returns a different shape, which is
                the sign of one written for a scalar.
        """
        values = np.asarray(function(self._eigenvalues), dtype=float)
        if values.shape != self._eigenvalues.shape:
            raise ValueError(
                f"The function returned shape {values.shape}, expected "
                f"{self._eigenvalues.shape}."
            )
        return self._rebuild(values)

    @property
    def inverse(self) -> DiagonalLinearOperator[V]:
        """``A^-1``, by reciprocating the spectrum.

        Returns:
            The inverse, itself diagonal.

        Raises:
            ZeroDivisionError: if any eigenvalue is zero. A singular operator
                has no inverse, and returning infinities instead would push
                the failure somewhere further away.
        """
        if np.any(self._eigenvalues == 0.0):
            raise ZeroDivisionError(
                "The operator is singular: some eigenvalues are zero."
            )
        return self.apply_function(np.reciprocal)

    @property
    def sqrt(self) -> DiagonalLinearOperator[V]:
        """``A^(1/2)``, the operator with the square roots as its spectrum.

        Returns:
            The diagonal operator ``B`` with ``B B == A``. It is *the*
            positive semidefinite square root only where ``A`` is
            self-adjoint, which on a metric that is not diagonal it is not;
            see :meth:`_require_spectrum`. A caller wanting a covariance
            factor -- ``L L* == C`` -- needs the self-adjoint case and should
            check the returned operator's traits.

        Raises:
            ValueError: if any eigenvalue is negative, naming it.
        """
        self._require_spectrum(
            self._eigenvalues >= 0.0, "a square root", "non-negative eigenvalues"
        )
        return self.apply_function(np.sqrt)

    @property
    def inverse_sqrt(self) -> DiagonalLinearOperator[V]:
        """``A^(-1/2)``.

        Returns:
            The diagonal operator whose square is ``A^-1``; as for
            :attr:`sqrt`, self-adjoint only where ``A`` is.

        Raises:
            ValueError: if any eigenvalue is not strictly positive.
        """
        self._require_spectrum(
            self._eigenvalues > 0.0,
            "an inverse square root",
            "strictly positive eigenvalues",
        )
        return self.apply_function(lambda d: 1.0 / np.sqrt(d))

    @property
    def exp(self) -> DiagonalLinearOperator[V]:
        """``exp(A)``, defined for any spectrum."""
        return self.apply_function(np.exp)

    @property
    def log(self) -> DiagonalLinearOperator[V]:
        """``log(A)``.

        Returns:
            The diagonal operator ``B`` with ``exp(B) == A``.

        Raises:
            ValueError: if any eigenvalue is not strictly positive.
        """
        self._require_spectrum(
            self._eigenvalues > 0.0, "a logarithm", "strictly positive eigenvalues"
        )
        return self.apply_function(np.log)

    def _components_action(self) -> Callable[[np.ndarray], np.ndarray] | None:
        eigenvalues = self._eigenvalues
        return lambda c: eigenvalues * c

    def _components_adjoint_action(
        self,
    ) -> Callable[[np.ndarray], np.ndarray] | None:
        if Traits.SELF_ADJOINT & self.traits:
            return self._components_action()
        eigenvalues, space = self._eigenvalues, self.domain
        return lambda c: space.solve_gram(eigenvalues * space.apply_gram(c))

    def _known_matrix(self, form: str) -> np.ndarray | None:
        matrix = np.diag(self._eigenvalues)
        if form == "components":
            return matrix
        return self.domain.apply_gram_to_columns(matrix)

    def _known_diagonals(
        self, offsets: tuple[int, ...], form: str
    ) -> np.ndarray | None:
        """The diagonals, read off the spectrum rather than probed.

        Only the main diagonal is non-zero in the components form. In the
        Galerkin form the metric multiplies the matrix, ``G diag(d)``, which
        stays diagonal on a diagonal metric and does not otherwise: there
        the main diagonal is still a read, ``G_ii d_i``, and any other
        offset is ``G_ij d_j`` -- an entry of the Gram matrix this class does
        not hold, so for those the answer is ``None`` and the base class
        probes rather than guessing.
        """
        from .spaces import DiagonalMetricSpace, OrthonormalSpace

        space = self.domain
        if form == "galerkin":
            if isinstance(space, OrthonormalSpace):
                values = self._eigenvalues
            elif isinstance(space, DiagonalMetricSpace):
                values = space.metric_values * self._eigenvalues
            elif all(offset == 0 for offset in offsets):
                values = space.gram_diagonal() * self._eigenvalues
            else:
                return None
        else:
            values = self._eigenvalues
        result = np.zeros((len(offsets), space.dim))
        for index, offset in enumerate(offsets):
            if offset == 0:
                result[index] = values
        return result

    def apply_block(
        self, vectors: Sequence[V], /, *, n_jobs: int | None = None
    ) -> list[V]:
        """The spectrum broadcast over the vectors' components at once.

        Args:
            vectors: the inputs.
            n_jobs: accepted for the protocol and unused: one broadcast
                multiply is the whole cost.

        Returns:
            The images, in order.
        """
        space: CoordinateSpace = self.domain
        return space.vectors_from(
            self._eigenvalues[:, None] * space.components_of(vectors)
        )

    def _adjoint_apply_block(
        self, vectors: Sequence[V], /, *, n_jobs: int | None = None
    ) -> list[V]:
        if Traits.SELF_ADJOINT & self.traits:
            return self.apply_block(vectors, n_jobs=n_jobs)
        space: CoordinateSpace = self.domain
        weighted = space.apply_gram_to_columns(space.components_of(vectors))
        return space.vectors_from(
            space.solve_gram_to_columns(self._eigenvalues[:, None] * weighted)
        )

    @property
    def log_determinant(self) -> float:
        """``log det A``, exactly.

        The determinant of an endomorphism is the product of its eigenvalues
        whatever the inner product -- ``det`` is a property of the map, not of
        the metric -- so this is exact on any space.

        Returns:
            ``sum(log d)``.

        Raises:
            ValueError: if any eigenvalue is not strictly positive, so that
                the logarithm of the determinant is not real.
        """
        self._require_spectrum(
            self._eigenvalues > 0.0,
            "a log determinant",
            "strictly positive eigenvalues",
        )
        return float(np.sum(np.log(self._eigenvalues)))

    def __abs__(self) -> DiagonalLinearOperator[V]:
        """``|A|``, the pointwise absolute value of the spectrum."""
        return self.apply_function(np.abs)

    def __pow__(self, power: float) -> DiagonalLinearOperator[V]:
        """``A^p``.

        Args:
            power: the exponent. A fractional one needs a non-negative
                spectrum; an integer one is defined for any.

        Returns:
            The diagonal operator with spectrum ``d ** power``.

        Raises:
            ValueError: for a fractional power of a spectrum with a negative
                eigenvalue, naming it.
        """
        if power != int(power):
            self._require_spectrum(
                self._eigenvalues >= 0.0,
                "a fractional power",
                "non-negative eigenvalues",
            )
        return self.apply_function(lambda d: d**power)

    def _require_spectrum(self, admissible: np.ndarray, what: str, needed: str) -> None:
        """Raise unless every eigenvalue lies where the function needs it.

        **The calculus gates on the spectrum, not on the traits.** A diagonal
        operator's eigenvectors are the space's basis vectors and its
        eigenvalues are the stored values *whatever the inner product* -- an
        eigendecomposition is a statement about the map, not about the metric
        -- so ``f(A)`` is ``diag(f(d))`` on any space, and what ``f`` needs is
        that ``d`` lies in its domain.

        Gating on ``POSITIVE_SEMIDEFINITE`` instead, as this used to, refused
        :attr:`sqrt`, :attr:`log`, :attr:`log_determinant` and fractional
        powers outright on every space whose Gram matrix is not diagonal:
        there ``G diag(d)`` is symmetric only where the two commute, so
        :meth:`_deduce_traits` deduces nothing and the operator "has NONE"
        however positive its spectrum. That is a true statement about
        self-adjointness and the wrong question to ask of a square root.

        What the metric does decide is what the *result* is: on a diagonal
        metric ``sqrt`` returns the positive semidefinite square root, and on
        any other it returns a square root that is not self-adjoint. The
        returned operator's traits say which, deduced by the constructor from
        the same rule, so a caller who needs ``L L* == C`` rather than
        ``B B == A`` can see that it did not get one.

        Args:
            admissible: a boolean mask over the eigenvalues, true where the
                function is defined.
            what: the operation, for the message.
            needed: what the spectrum must satisfy, for the message.

        Raises:
            ValueError: naming the first offending eigenvalue and how many
                offend.
        """
        if bool(np.all(admissible)):
            return
        offending = np.flatnonzero(~np.asarray(admissible))
        first = int(offending[0])
        raise ValueError(
            f"{what.capitalize()} needs {needed}; eigenvalue {first} is "
            f"{self._eigenvalues[first]:g}, and {offending.size} of "
            f"{self._eigenvalues.size} are inadmissible. Note the gate is on "
            f"the spectrum, not on the traits: this operator claims "
            f"{self.traits!s}, which on a metric that is not diagonal is "
            f"NONE however positive its eigenvalues."
        )
