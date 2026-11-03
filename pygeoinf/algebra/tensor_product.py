"""
Tensor products of coordinate spaces, and the operators that respect them.

The tensor product ``X_1 (x) ... (x) X_n`` of Hilbert spaces is the space
spanned by the products ``x_1 ... x_n`` of one vector from each factor, under
the inner product that makes the inner product of two such products the
product of the factors' inner products. The example this exists for is a
field through time: a space of functions on an interval of time and a space
of fields on a sphere give the space of fields ``u(t, x)`` on the sphere
through that interval, with the norm of ``H^s(T) (x) H^r(S)``.

**Coordinates are what make it possible.** Every factor supplies a basis, and
the products of basis vectors, one from each factor, are a basis of the
product; so a vector is an array with one axis per factor, holding its
components on that basis, and the Gram matrix is the Kronecker product of the
factors'. Both act factor by factor: ``G`` on an array is each factor's Gram
map applied along its own axis, likewise ``G^-1`` and the whitening that
draws white noise, so nothing of size ``dim^2`` is formed and the product
costs what its factors cost. A space without coordinates has no such basis; a
vector of its product would be a sum of rank-one terms whose length grows
under every addition, and the product is not provided. The product's metric
is diagonal exactly when every factor's is, and the library's diagonal fast
paths then apply to it unchanged.

**The product norm is anisotropic**, and that is the point. With Sobolev
factors of orders ``s`` and ``r`` the two orders are independent, so for ``s >
1/2`` the restriction of a history to one time is a field of the field space
and evaluation at a time is bounded, where an isotropic norm of the same
total order on the same basis would lose half a derivative in space at each
time. A norm that couples the factors is a different inner product on the
same vectors: a :class:`~pygeoinf.algebra.spaces.MassWeightedSpace` over the
product when it is a reweighting, and a space of its own when it is not.

Two operator constructions come with the space.
:meth:`TensorProduct.contraction` applies a functional on one factor,
evaluation at a time say, leaving an operator onto the product of the
remaining factors; its adjoint is the functional's representer tensored with
the data, so the remaining factors' metrics cancel and the contracted
factor's is applied once. :class:`TensorProductLinearOperator` is ``A_1 (x)
... (x) A_n``, each factor acting along its own axis, with the identity on a
factor free: an operator that acts at each time alone is ``I (x) B``, and
costs one application of ``B`` per time mode whatever the number of times
observed. A Gaussian measure on the product with covariance ``C_1 (x) ...
(x) C_n`` is
:meth:`~pygeoinf.probability.gaussian.GaussianMeasure.from_tensor_product`.
"""

from __future__ import annotations

from functools import cached_property, reduce
from typing import Callable, Hashable, Sequence

import numpy as np
from numpy.random import Generator

from ..traits import close
from .diagonal import DiagonalLinearOperator
from .operators import LinearFunctional, LinearOperator, Operator
from .spaces import ArrayVectorMixin, CoordinateSpace, HilbertSpace, resolve_rng

__all__ = ["TensorProduct", "TensorProductLinearOperator"]


def _along_axis(
    array: np.ndarray,
    axis: int,
    columns_map: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """A map on ``(dim, k)`` column arrays, applied along one axis of an array.

    The axis is brought to the front, every other axis is flattened into
    columns, the map is applied, and the axes are restored. The map may
    change the length of the axis, as an operator between different factors
    does.
    """
    moved = np.moveaxis(array, axis, 0)
    shape = moved.shape
    columns = np.ascontiguousarray(moved).reshape(shape[0], -1)
    result = columns_map(columns)
    return np.moveaxis(result.reshape((result.shape[0],) + shape[1:]), 0, axis)


def _apply_columns(operator: LinearOperator, columns: np.ndarray) -> np.ndarray:
    """An operator between coordinate spaces applied to component columns.

    The identity is a copy, a diagonal operator a broadcast, an operator
    with an action on components that action column by column, and any
    other goes through the vectors in one block application, so that a
    matrix-backed factor still multiplies all the columns at once.
    """
    if operator.is_identity:
        return columns.copy()
    if isinstance(operator, DiagonalLinearOperator):
        return operator.eigenvalues[:, None] * columns
    if columns.shape[1] == 0:
        return np.zeros((operator.codomain.dim, 0))
    action = operator.components_action()
    if action is not None:
        return np.stack([action(column) for column in columns.T], axis=1)
    domain: CoordinateSpace = operator.domain
    codomain: CoordinateSpace = operator.codomain
    return codomain.components_of(operator.apply_block(domain.vectors_from(columns)))


def _outer(arrays: Sequence[np.ndarray], /) -> np.ndarray:
    """The outer product of several one-dimensional arrays."""
    return reduce(np.multiply.outer, arrays)


class TensorProduct(ArrayVectorMixin, CoordinateSpace[np.ndarray]):
    """The tensor product of coordinate spaces.

    A vector is an array with one axis per factor, its entries the components
    on the products of the factors' basis vectors, and its components are
    that array read row by row. The Gram matrix is the Kronecker product of
    the factors' and is applied factor by factor, along the axes.
    """

    def __init__(self, factors: Sequence[CoordinateSpace], /) -> None:
        """
        Args:
            factors: the spaces, two or more, each with a coordinate map.

        Raises:
            ValueError: with fewer than two factors.
            TypeError: if a factor provides no coordinate map.
        """
        factors = tuple(factors)
        if len(factors) < 2:
            raise ValueError("A tensor product needs at least two factors.")
        for index, factor in enumerate(factors):
            if not isinstance(factor, CoordinateSpace):
                raise TypeError(
                    f"Factor {index} is a {type(factor).__name__}, which provides "
                    f"no coordinate map; a tensor product needs a basis in "
                    f"every factor."
                )
        self._factors = factors
        self._shape = tuple(factor.dim for factor in factors)
        self._dim = int(np.prod(self._shape))

    # ----------------------------------------------------------------- #
    #                              Structure                            #
    # ----------------------------------------------------------------- #

    @property
    def factors(self) -> tuple[CoordinateSpace, ...]:
        """The factor spaces, in the order of the axes."""
        return self._factors

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of a vector: the factors' dimensions."""
        return self._shape

    @property
    def dim(self) -> int:
        """The product of the factors' dimensions."""
        return self._dim

    @property
    def is_function_space(self) -> bool:
        """True when any factor discretizes a space of functions."""
        return any(factor.is_function_space for factor in self._factors)

    @property
    def has_diagonal_metric(self) -> bool:
        """True exactly when every factor's metric is diagonal."""
        return all(factor.has_diagonal_metric for factor in self._factors)

    @property
    def is_orthonormal(self) -> bool:
        """True exactly when every factor's basis is orthonormal."""
        return all(factor.is_orthonormal for factor in self._factors)

    @cached_property
    def metric_values(self) -> np.ndarray:
        """The diagonal of the Gram matrix, the outer product of the
        factors', on a product whose metric is diagonal.

        Raises:
            ValueError: if some factor's metric is not diagonal.
        """
        if not self.has_diagonal_metric:
            raise ValueError("The metric is not diagonal: some factor's is not.")
        return self.gram_diagonal()

    def _key(self) -> Hashable:
        return ("tensor product", self._factors)

    def __repr__(self) -> str:
        inner = ", ".join(repr(factor) for factor in self._factors)
        return f"TensorProduct([{inner}])"

    def _resolve_axis(self, space: HilbertSpace, axis: int | None, what: str) -> int:
        """The axis of the factor equal to *space*, found or checked."""
        if axis is None:
            matches = [i for i, factor in enumerate(self._factors) if factor == space]
            if len(matches) == 1:
                return matches[0]
            if not matches:
                raise ValueError(f"No factor of the product is the {what}'s space.")
            raise ValueError(
                f"Factors {matches} are all the {what}'s space; give axis= to say "
                f"which is meant."
            )
        if not 0 <= axis < len(self._factors):
            raise ValueError(
                f"axis {axis} is outside the {len(self._factors)} factors."
            )
        if self._factors[axis] != space:
            raise ValueError(f"Factor {axis} is not the {what}'s space.")
        return axis

    def _remaining(self, axis: int) -> HilbertSpace:
        """The product of the factors other than *axis*: the factor itself
        when one is left."""
        remaining = self._factors[:axis] + self._factors[axis + 1 :]
        return remaining[0] if len(remaining) == 1 else TensorProduct(remaining)

    # ----------------------------------------------------------------- #
    #                               Vectors                             #
    # ----------------------------------------------------------------- #

    def zero(self) -> np.ndarray:
        """The zero array."""
        return np.zeros(self._shape)

    def to_components(self, x: np.ndarray) -> np.ndarray:
        """The array read row by row.

        Raises:
            ValueError: if the array is not of the product's shape.
        """
        x = np.asarray(x, dtype=float)
        if x.shape != self._shape:
            raise ValueError(
                f"Expected an array of shape {self._shape}, got {x.shape}."
            )
        return x.reshape(-1)

    def from_components(self, c: np.ndarray) -> np.ndarray:
        """The array with those entries, copied.

        Raises:
            ValueError: if the number of components is not the dimension.
        """
        c = np.asarray(c, dtype=float)
        if c.shape != (self._dim,):
            raise ValueError(f"Expected {self._dim} components, got {c.shape}.")
        return np.array(c).reshape(self._shape)

    def rank_one(self, *vectors: object) -> np.ndarray:
        """The product ``x_1 (x) ... (x) x_n`` of one vector from each factor.

        Its norm is the product of the factors' norms, and the inner product
        of two such vectors is the product of the factors' inner products.

        Args:
            *vectors: one vector of each factor, in order.

        Returns:
            The outer product of the vectors' components.

        Raises:
            ValueError: if the number of vectors is not the number of factors.
        """
        if len(vectors) != len(self._factors):
            raise ValueError(
                f"Expected {len(self._factors)} vectors, one per factor, got "
                f"{len(vectors)}."
            )
        return _outer(
            [
                factor.to_components(vector)
                for factor, vector in zip(self._factors, vectors)
            ]
        )

    # ----------------------------------------------------------------- #
    #                              The metric                           #
    # ----------------------------------------------------------------- #

    def _factorwise(self, array: np.ndarray, method: str) -> np.ndarray:
        """A column method of every factor, applied along its axis. Any
        trailing axes beyond the factors' are carried along as columns."""
        for axis, factor in enumerate(self._factors):
            array = _along_axis(array, axis, getattr(factor, method))
        return array

    def apply_gram(self, c: np.ndarray) -> np.ndarray:
        """``G c``: each factor's Gram map along its axis."""
        return self._factorwise(
            np.asarray(c, dtype=float).reshape(self._shape), "apply_gram_to_columns"
        ).reshape(-1)

    def solve_gram(self, c: np.ndarray) -> np.ndarray:
        """``G^-1 c``: each factor's inverse Gram map along its axis."""
        return self._factorwise(
            np.asarray(c, dtype=float).reshape(self._shape), "solve_gram_to_columns"
        ).reshape(-1)

    def apply_gram_to_columns(self, columns: np.ndarray, /) -> np.ndarray:
        """``G`` on every column, the columns carried as a trailing axis."""
        count = columns.shape[1]
        return self._factorwise(
            np.asarray(columns, dtype=float).reshape(self._shape + (count,)),
            "apply_gram_to_columns",
        ).reshape(self._dim, count)

    def solve_gram_to_columns(self, columns: np.ndarray, /) -> np.ndarray:
        """``G^-1`` on every column, the columns carried as a trailing axis."""
        count = columns.shape[1]
        return self._factorwise(
            np.asarray(columns, dtype=float).reshape(self._shape + (count,)),
            "solve_gram_to_columns",
        ).reshape(self._dim, count)

    def gram_diagonal(self) -> np.ndarray:
        """The outer product of the factors' Gram diagonals."""
        return _outer([factor.gram_diagonal() for factor in self._factors]).reshape(-1)

    def gram_matrix(self) -> np.ndarray:
        """The Kronecker product of the factors' Gram matrices, dense."""
        return reduce(np.kron, [factor.gram_matrix() for factor in self._factors])

    def whiten_columns(self, xi: np.ndarray, /) -> np.ndarray:
        """``L^-T`` on every column: each factor's whitening along its axis."""
        count = xi.shape[1]
        return self._factorwise(
            np.asarray(xi, dtype=float).reshape(self._shape + (count,)),
            "whiten_columns",
        ).reshape(self._dim, count)

    def white_noise_components(self, *, rng: Generator | None = None) -> np.ndarray:
        """Components drawn from ``N(0, G^-1)``, whitened factor by factor."""
        xi = resolve_rng(rng).standard_normal(self._shape)
        return self._factorwise(xi, "whiten_columns").reshape(-1)

    # ----------------------------------------------------------------- #
    #                              Operators                            #
    # ----------------------------------------------------------------- #

    def contraction(
        self, functional: LinearFunctional, /, *, axis: int | None = None
    ) -> LinearOperator[np.ndarray, object]:
        """A functional applied on one factor, as an operator onto the rest.

        For a functional ``f`` on the factor along *axis*, the map
        ``x_1 (x) ... (x) x_n -> f(x_i) x_1 (x) ... (x) x_n`` with the
        ``i``-th factor removed, extended linearly: with ``g`` the
        functional's derivative components, the array is contracted with
        ``g`` along the axis. Evaluation of a history at a time is the
        contraction with the time space's Dirac functional there, and the
        order the time space needs for that to be bounded is the Dirac
        functional's to check.

        The adjoint of a vector ``y`` of the codomain is ``v (x) y`` with
        ``v`` the functional's representer: the codomain's metric cancels,
        and the contracted factor's inverse metric is applied to ``g`` once.
        Both actions are also given on component arrays.

        Args:
            functional: a linear functional on one of the factors.
            axis: which factor, needed only when two factors are the same
                space. Found from the functional's domain otherwise.

        Returns:
            The operator from this space onto the product of the remaining
            factors, which is the remaining factor itself when one is left.

        Raises:
            ValueError: if no factor is the functional's space, or more than
                one is and *axis* is not given, or *axis* names another.
        """
        axis = self._resolve_axis(functional.domain, axis, "functional")
        factor = self._factors[axis]
        g = np.asarray(functional.derivative_components, dtype=float)
        v = factor.solve_gram(g)
        codomain = self._remaining(axis)
        shape = self._shape
        rest = shape[:axis] + shape[axis + 1 :]

        def contract(array: np.ndarray) -> np.ndarray:
            return np.tensordot(g, array, axes=([0], [axis]))

        def expand(array: np.ndarray) -> np.ndarray:
            return np.moveaxis(np.multiply.outer(v, array), 0, axis)

        return LinearOperator.from_callables(
            self,
            codomain,
            lambda x: codomain.from_components(contract(x).reshape(-1)),
            adjoint=lambda y: expand(codomain.to_components(y).reshape(rest)),
            components_action=lambda c: contract(c.reshape(shape)).reshape(-1),
            components_adjoint_action=lambda c: expand(c.reshape(rest)).reshape(-1),
        )

    def lift(
        self, operator: LinearOperator, /, *, axis: int | None = None
    ) -> TensorProductLinearOperator:
        """An operator on one factor, acting on the product: ``I (x) A (x) I``.

        The operator acts along its factor's axis and the identity along
        every other, so an operator that acts at each time alone is lifted
        from the field space to the space of histories at one application
        per time mode.

        Args:
            operator: an operator from one of the factors to a coordinate
                space.
            axis: which factor, needed only when two factors are the same
                space. Found from the operator's domain otherwise.

        Returns:
            The tensor product of identities with *operator* at *axis*, into
            the product with that factor replaced by the operator's codomain.

        Raises:
            ValueError: if no factor is the operator's domain, or more than
                one is and *axis* is not given, or *axis* names another.
        """
        axis = self._resolve_axis(operator.domain, axis, "operator")
        operators = [LinearOperator.identity(factor) for factor in self._factors]
        operators[axis] = operator
        return TensorProductLinearOperator(operators)


class TensorProductLinearOperator(LinearOperator[np.ndarray, np.ndarray]):
    """``A_1 (x) ... (x) A_n``, between tensor products of coordinate spaces.

    Each factor acts along its own axis of the array, so an application
    costs one application of each factor per column of the others, and the
    identity on a factor costs nothing. The adjoint is the tensor product of
    the adjoints, the composition of two such operators the tensor product
    of the compositions, and the traits are those every factor claims: a
    tensor product of self-adjoint operators is self-adjoint, of positive
    definite ones positive definite, and so on through the list.
    """

    def __init__(self, operators: Sequence[LinearOperator], /) -> None:
        """
        Args:
            operators: the factors, two or more, each between coordinate
                spaces.

        Raises:
            ValueError: with fewer than two factors.
            TypeError: if a factor's domain or codomain provides no
                coordinate map.
        """
        operators = tuple(operators)
        if len(operators) < 2:
            raise ValueError("A tensor product of operators needs at least two.")
        domain = TensorProduct([operator.domain for operator in operators])
        codomain = TensorProduct([operator.codomain for operator in operators])
        traits = reduce(lambda a, b: a & b, [close(op.traits) for op in operators])
        super().__init__(domain, codomain, traits=traits)
        self._operators = operators

    @property
    def operators(self) -> tuple[LinearOperator, ...]:
        """The factors, in the order of the axes."""
        return self._operators

    def __repr__(self) -> str:
        inner = ", ".join(repr(operator) for operator in self._operators)
        return f"TensorProductLinearOperator([{inner}])"

    @cached_property
    def _adjoints(self) -> tuple[LinearOperator, ...]:
        return tuple(operator.adjoint for operator in self._operators)

    @staticmethod
    def _apply(operators: Sequence[LinearOperator], array: np.ndarray) -> np.ndarray:
        for axis, operator in enumerate(operators):
            array = _along_axis(
                array, axis, lambda columns, op=operator: _apply_columns(op, columns)
            )
        return array

    def _value(self, x: np.ndarray) -> np.ndarray:
        return self._apply(self._operators, x)

    def _adjoint_value(self, y: np.ndarray) -> np.ndarray:
        return self._apply(self._adjoints, y)

    def _make_adjoint(self) -> LinearOperator:
        result = TensorProductLinearOperator(self._adjoints)
        result._link_adjoint(self)
        return result

    def apply_block(
        self, vectors: Sequence[np.ndarray], /, *, n_jobs: int | None = None
    ) -> list[np.ndarray]:
        """The factors applied to all the vectors at once, stacked along a
        trailing axis, so that each factor sees every column of the block.

        Args:
            vectors: the inputs.
            n_jobs: accepted for the protocol and unused: each factor's own
                block application is what runs.

        Returns:
            The images, in order.
        """
        return self._block(self._operators, vectors)

    def _adjoint_apply_block(
        self, vectors: Sequence[np.ndarray], /, *, n_jobs: int | None = None
    ) -> list[np.ndarray]:
        return self._block(self._adjoints, vectors)

    def _block(
        self, operators: Sequence[LinearOperator], vectors: Sequence[np.ndarray]
    ) -> list[np.ndarray]:
        vectors = list(vectors)
        if not vectors:
            return []
        images = self._apply(operators, np.stack(vectors, axis=-1))
        return [np.array(images[..., j]) for j in range(images.shape[-1])]

    def _components_action(self) -> Callable[[np.ndarray], np.ndarray] | None:
        operators, shape = self._operators, self.domain.shape
        return lambda c: self._apply(operators, c.reshape(shape)).reshape(-1)

    def _components_adjoint_action(
        self,
    ) -> Callable[[np.ndarray], np.ndarray] | None:
        operators, shape = self._adjoints, self.codomain.shape
        return lambda c: self._apply(operators, c.reshape(shape)).reshape(-1)

    def _known_scaling(self) -> np.ndarray | None:
        scalings = [operator._known_scaling() for operator in self._operators]
        if any(scaling is None for scaling in scalings):
            return None
        return _outer([np.asarray(s, dtype=float) for s in scalings]).reshape(-1)

    def _known_matrix(self, form: str) -> np.ndarray | None:
        matrices = [
            operator._known_matrix("components") for operator in self._operators
        ]
        if any(matrix is None for matrix in matrices):
            return None
        matrix = reduce(np.kron, matrices)
        if form == "components":
            return matrix
        return self.codomain.apply_gram_to_columns(matrix)

    def _combine_compose(self, other: Operator) -> Operator | None:
        if (
            isinstance(other, TensorProductLinearOperator)
            and len(other.operators) == len(self._operators)
            and self.domain == other.codomain
        ):
            return TensorProductLinearOperator(
                [a @ b for a, b in zip(self._operators, other.operators)]
            )
        return None

    def _combine_scale(self, alpha: float) -> Operator | None:
        first, *rest = self._operators
        return TensorProductLinearOperator([alpha * first, *rest])
