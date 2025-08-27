"""
Defines the foundational abstractions for working with Hilbert spaces.

This module provides the core `HilbertSpace` abstract base class (ABC), which
serves as a mathematical abstraction for real vector spaces equipped with an
inner product. The design separates abstract vector operations from their
concrete representations (e.g., as NumPy arrays), allowing for generic and
reusable implementations of linear operators and algorithms.

The inner product of a space is defined by its Riesz representation map
(`to_dual` and `from_dual` methods), which connects the space to its dual.
Concrete subclasses must implement the abstract methods to define a specific
type of space.

Key Classes
-----------
- `HilbertSpace`: The primary ABC defining the interface for all Hilbert spaces.
- `DualHilbertSpace`: A wrapper class representing the dual of a Hilbert space.
- `HilbertModule`: An ABC for Hilbert spaces that also support vector multiplication.
- `EuclideanSpace`: A concrete implementation for R^n using NumPy arrays.
- `MassWeightedHilbertSpace`: A space whose inner product is weighted by a
  mass operator relative to an underlying space.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    TypeVar,
    List,
    Optional,
    Any,
    TYPE_CHECKING,
    final,
)

import numpy as np

# This block only runs for type checkers, not at runtime
if TYPE_CHECKING:
    from .operators import LinearOperator
    from .linear_forms import LinearForm

# Define a generic type for vectors in a Hilbert space
Vector = TypeVar("Vector")


class HilbertSpace(ABC):
    """
    An abstract base class for real Hilbert spaces.

    This class provides a mathematical abstraction for a vector space equipped
    with an inner product. It defines a formal interface that separates
    abstract vector operations from their concrete representation (e.g., as
    NumPy arrays). Subclasses must implement all abstract methods to be
    instantiable.
    """

    # ------------------------------------------------------------------- #
    #               Abstract methods that must be provided                #
    # ------------------------------------------------------------------- #

    @property
    @abstractmethod
    def dim(self) -> int:
        """The finite dimension of the space."""

    @abstractmethod
    def to_dual(self, x: Vector) -> Any:
        """
        Maps a vector to its canonical dual vector (a linear functional).

        This method, along with `from_dual`, defines the Riesz representation
        map and implicitly defines the inner product of the space.

        Args:
            x: A vector in the primal space.

        Returns:
            The corresponding vector in the dual space.
        """

    @abstractmethod
    def from_dual(self, xp: Any) -> Vector:
        """
        Maps a dual vector back to its representative in the primal space.

        This is the inverse of the Riesz representation map defined by `to_dual`.

        Args:
            xp: A vector in the dual space.

        Returns:
            The corresponding vector in the primal space.
        """

    @abstractmethod
    def to_components(self, x: Vector) -> np.ndarray:
        """
        Maps a vector to its representation as a NumPy component array.

        Args:
            x: A vector in the space.

        Returns:
            The components of the vector as a NumPy array.
        """

    @abstractmethod
    def from_components(self, c: np.ndarray) -> Vector:
        """
        Maps a NumPy component array back to a vector in the space.

        Args:
            c: The components of the vector as a NumPy array.

        Returns:
            The corresponding vector in the space.
        """

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """
        Defines equality between two HilbertSpace instances.

        This is an abstract method, requiring all concrete subclasses to
        implement a meaningful comparison. This ensures that equality is
        defined by mathematical equivalence rather than object identity.
        """

    # ------------------------------------------------------------------- #
    #            Default implementations that can be overridden           #
    # ------------------------------------------------------------------- #

    @property
    def dual(self) -> HilbertSpace:
        """
        The dual of this Hilbert space.

        The dual space is the space of all continuous linear functionals
        (i.e., `LinearForm` objects) that map vectors from this space to
        real numbers. This implementation returns a `DualHilbertSpace` wrapper.
        """
        return DualHilbertSpace(self)

    @property
    def zero(self) -> Vector:
        """The zero vector (additive identity) of the space."""
        return self.from_components(np.zeros((self.dim)))

    def is_element(self, x: Any) -> bool:
        """
        Checks if an object is a valid element of the space.

        Note: The default implementation checks the object's type against the
        type of the `zero` vector. This may not be robust for all vector
        representations and can be overridden if needed.

        Args:
            x: The object to check.

        Returns:
            True if the object is an element of the space, False otherwise.
        """
        return isinstance(x, type(self.zero))

    def duality_product(self, xp: LinearForm, x: Vector) -> float:
        """
        Computes the duality product <xp, x>.

        This evaluates the linear functional `xp` (an element of the dual space)
        at the vector `x` (an element of the primal space).

        Args:
            xp: The linear functional from the dual space.
            x: The vector from the primal space.

        Returns:
            The result of the evaluation xp(x).
        """
        return xp(x)

    def add(self, x: Vector, y: Vector) -> Vector:
        """Computes the sum of two vectors. Defaults to `x + y`."""
        return x + y

    def subtract(self, x: Vector, y: Vector) -> Vector:
        """Computes the difference of two vectors. Defaults to `x - y`."""
        return x - y

    def multiply(self, a: float, x: Vector) -> Vector:
        """Computes scalar multiplication. Defaults to `a * x`."""
        return a * x

    def negative(self, x: Vector) -> Vector:
        """Computes the additive inverse of a vector. Defaults to `-1 * x`."""
        return -1 * x

    def ax(self, a: float, x: Vector) -> None:
        """Performs in-place scaling `x := a*x`. Defaults to `x *= a`."""
        x *= a

    def axpy(self, a: float, x: Vector, y: Vector) -> None:
        """Performs in-place operation `y := y + a*x`. Defaults to `y += a*x`."""
        y += a * x

    def copy(self, x: Vector) -> Vector:
        """Returns a deep copy of a vector. Defaults to `x.copy()`."""
        return x.copy()

    def random(self) -> Vector:
        """
        Generates a random vector from the space.

        The vector's components are drawn from a standard normal distribution.

        Returns:
            A new random vector.
        """
        return self.from_components(np.random.randn(self.dim))

    # ------------------------------------------------------------------- #
    #                      Final (Non-Overridable) Methods                #
    # ------------------------------------------------------------------- #

    

    @final
    @property
    def coordinate_inclusion(self) -> LinearOperator:
        """
        The linear operator mapping R^n component vectors into this space.
        """
        from .operators import LinearOperator

        domain = EuclideanSpace(self.dim)

        def dual_mapping(xp: Any) -> Any:
            cp = self.dual.to_components(xp)
            return domain.to_dual(cp)

        def adjoint_mapping(y: Vector) -> np.ndarray:
            yp = self.to_dual(y)
            return self.dual.to_components(yp)

        return LinearOperator(
            domain,
            self,
            self.from_components,
            dual_mapping=dual_mapping,
            adjoint_mapping=adjoint_mapping,
        )

    @final
    @property
    def coordinate_projection(self) -> LinearOperator:
        """
        The linear operator projecting vectors from this space to R^n.
        """
        from .operators import LinearOperator

        codomain = EuclideanSpace(self.dim)

        def dual_mapping(cp: Any) -> Any:
            c = codomain.from_dual(cp)
            return self.dual.from_components(c)

        def adjoint_mapping(c: np.ndarray) -> Vector:
            xp = self.dual.from_components(c)
            return self.from_dual(xp)

        return LinearOperator(
            self,
            codomain,
            self.to_components,
            dual_mapping=dual_mapping,
            adjoint_mapping=adjoint_mapping,
        )

    @final
    @property
    def riesz(self) -> LinearOperator:
        """The Riesz map (dual to primal) as a `LinearOperator`."""
        from .operators import LinearOperator

        return LinearOperator.self_dual(self.dual, self.from_dual)

    @final
    @property
    def inverse_riesz(self) -> LinearOperator:
        """The inverse Riesz map (primal to dual) as a `LinearOperator`."""
        from .operators import LinearOperator

        return LinearOperator.self_dual(self, self.to_dual)

    @final
    def inner_product(self, x1: Vector, x2: Vector) -> float:
        """
        Computes the inner product of two vectors, `(x1, x2)`.

        This is defined via the duality product as `<R(x1), x2>`, where `R` is
        the Riesz map (`to_dual`).

        Args:
            x1: The first vector.
            x2: The second vector.

        Returns:
            The inner product as a float.
        """
        return self.duality_product(self.to_dual(x1), x2)

    @final
    def squared_norm(self, x: Vector) -> float:
        """
        Computes the squared norm of a vector, `||x||^2`.

        Args:
            x: The vector.

        Returns:
            The squared norm of the vector.
        """
        return self.inner_product(x, x)

    @final
    def norm(self, x: Vector) -> float:
        """
        Computes the norm of a vector, `||x||`.

        Args:
            x: The vector.

        Returns:
            The norm of the vector.
        """
        return np.sqrt(self.squared_norm(x))

    @final
    def gram_schmidt(self, vectors: List[Vector]) -> List[Vector]:
        """
        Orthonormalizes a list of vectors using the Gram-Schmidt process.

        Args:
            vectors: A list of linearly independent vectors.

        Returns:
            A list of orthonormalized vectors spanning the same subspace.

        Raises:
            ValueError: If not all items in the list are elements of the space.
        """
        if not all(self.is_element(vector) for vector in vectors):
            raise ValueError("Not all vectors are elements of the space")

        orthonormalised_vectors: List[Vector] = []
        for i, vector in enumerate(vectors):
            vec_copy = self.copy(vector)
            for j in range(i):
                product = self.inner_product(vec_copy, orthonormalised_vectors[j])
                self.axpy(-product, orthonormalised_vectors[j], vec_copy)
            norm = self.norm(vec_copy)
            if norm < 1e-12:
                raise ValueError("Vectors are not linearly independent.")
            self.ax(1 / norm, vec_copy)
            orthonormalised_vectors.append(vec_copy)

        return orthonormalised_vectors

    @final
    def basis_vector(self, i: int) -> Vector:
        """
        Returns the i-th standard basis vector.

        This is the vector whose component array is all zeros except for a one
        at index `i`.

        Args:
            i: The index of the basis vector.

        Returns:
            The i-th basis vector.
        """
        c = np.zeros(self.dim)
        c[i] = 1
        return self.from_components(c)

    @final
    def sample_expectation(self, vectors: List[Vector]) -> Vector:
        """
        Computes the sample mean of a list of vectors.

        Args:
            vectors: A list of vectors from the space.

        Returns:
            The sample mean (average) vector.

        Raises:
            TypeError: If not all items in the list are elements of the space.
        """
        n = len(vectors)
        if not n > 0:
            raise ValueError("Cannot compute expectation of an empty list.")
        if not all(self.is_element(x) for x in vectors):
            raise TypeError("Not all items in list are elements of the space.")
        xbar = self.zero
        for x in vectors:
            self.axpy(1 / n, x, xbar)
        return xbar

    @final
    def identity_operator(self) -> LinearOperator:
        """Returns the identity operator `I` on the space."""
        from .operators import LinearOperator

        return LinearOperator(
            self,
            self,
            lambda x: x,
            adjoint_mapping=lambda y: y,
        )

    @final
    def zero_operator(self, codomain: Optional[HilbertSpace] = None) -> LinearOperator:
        """
        Returns the zero operator `0` from this space to a codomain.

        Args:
            codomain: The target space of the operator. If None, the operator
                maps to this space itself.

        Returns:
            The zero linear operator.
        """
        from .operators import LinearOperator

        codomain = self if codomain is None else codomain
        return LinearOperator(
            self,
            codomain,
            lambda x: codomain.zero,
            dual_mapping=lambda yp: self.dual.zero,
            adjoint_mapping=lambda y: self.zero,
        )


class DualHilbertSpace(HilbertSpace):
    """
    A wrapper class representing the dual of a `HilbertSpace`.

    An element of a dual space is a continuous linear functional, represented
    in this library by the `LinearForm` class. This wrapper provides a full
    `HilbertSpace` interface for these `LinearForm` objects, allowing them to be
    treated as vectors in their own right.
    """

    def __init__(self, space: HilbertSpace):
        """
        Args:
            space: The primal space from which to form the dual.
        """
        self._underlying_space = space

    @property
    def underlying_space(self) -> HilbertSpace:
        """The primal `HilbertSpace` of which this is the dual."""
        return self._underlying_space

    @property
    def dim(self) -> int:
        """The dimension of the dual space."""
        return self._underlying_space.dim

    @property
    def dual(self) -> HilbertSpace:
        """The dual of the dual space, which is the original primal space."""
        return self._underlying_space

    def to_dual(self, x: LinearForm) -> Any:
        """Maps a dual vector back to its representative in the primal space."""
        return self._underlying_space.from_dual(x)

    def from_dual(self, xp: Vector) -> LinearForm:
        """Maps a primal vector to its corresponding dual `LinearForm`."""
        return self._underlying_space.to_dual(xp)

    def to_components(self, x: LinearForm) -> np.ndarray:
        """Maps a `LinearForm` to its NumPy component array."""
        return x.components

    def from_components(self, c: np.ndarray) -> LinearForm:
        """Creates a `LinearForm` from a NumPy component array."""
        from .linear_forms import LinearForm

        return LinearForm(self._underlying_space, components=c)

    def __eq__(self, other: object) -> bool:
        """
        Checks for equality with another dual space.

        Two dual spaces are considered equal if and only if their underlying
        primal spaces are equal.
        """
        if not isinstance(other, DualHilbertSpace):
            return NotImplemented
        return self.underlying_space == other.underlying_space

    @final
    def duality_product(self, xp: LinearForm, x: Vector) -> float:
        """
        Computes the duality product <x, xp>.

        In this context, `x` is from the primal space and `xp` is the dual
        vector (a `LinearForm`). This is unconventional but maintains the
        method signature; it evaluates `x(xp)`.
        """
        return x(xp)


class HilbertModule(HilbertSpace, ABC):
    """
    An ABC for a `HilbertSpace` where vector multiplication is defined.

    This acts as a "mixin" interface, adding the `vector_multiply` requirement
    to the `HilbertSpace` contract.
    """

    @abstractmethod
    def vector_multiply(self, x1: Vector, x2: Vector) -> Vector:
        """
        Computes the product of two vectors.

        Args:
            x1: The first vector.
            x2: The second vector.

        Returns:
            The product of the two vectors.
        """


class EuclideanSpace(HilbertSpace):
    """
    An n-dimensional Euclidean space, R^n.

    This is a concrete `HilbertSpace` where vectors are represented directly by
    NumPy arrays, and the inner product is the standard dot product.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim: The dimension of the space.
        """
        if dim < 1:
            raise ValueError("Dimension must be a positive integer.")
        self._dim = dim

    @property
    def dim(self) -> int:
        """The dimension of the space."""
        return self._dim

    def to_components(self, x: np.ndarray) -> np.ndarray:
        """Returns the vector itself, as it is already a component array."""
        return x

    def from_components(self, c: np.ndarray) -> np.ndarray:
        """Returns the component array itself, as it is the vector."""
        return c

    def to_dual(self, x: np.ndarray) -> "LinearForm":
        """Maps a vector `x` to a `LinearForm` with the same components."""
        from .linear_forms import LinearForm

        return LinearForm(self, components=x)

    def from_dual(self, xp: "LinearForm") -> np.ndarray:
        """Maps a `LinearForm` back to a vector via its components."""
        return self.dual.to_components(xp)

    def __eq__(self, other: object):
        if not isinstance(other, EuclideanSpace):
            return NotImplemented
        return self.dim == other.dim


class MassWeightedHilbertSpace(HilbertSpace):
    """
    A Hilbert space with an inner product weighted by a mass operator.

    This class wraps an existing `HilbertSpace` (let's call it X) and defines a new
    inner product for a space (Y) as: `(u, v)_Y = (M @ u, v)_X`, where `M` is a
    self-adjoint, positive-definite mass operator defined on X.

    This is a common construction in numerical methods like the Finite Element
    Method, where the basis functions are not orthonormal.
    """

    def __init__(
        self,
        underlying_space: HilbertSpace,
        mass_operator: LinearOperator,
        inverse_mass_operator: LinearOperator,
    ):
        """
        Args:
            underlying_space: The original space (X) on which the inner
                product is defined.
            mass_operator: The self-adjoint, positive-definite mass
                operator (M).
            inverse_mass_operator: The inverse of the mass operator.
        """
        self._underlying_space = underlying_space
        self._mass_operator = mass_operator
        self._inverse_mass_operator = inverse_mass_operator

    @property
    def dim(self) -> int:
        """The dimension of the space."""
        return self._underlying_space.dim

    @property
    def underlying_space(self) -> HilbertSpace:
        """The underlying Hilbert space (X) without mass weighting."""
        return self._underlying_space

    @property
    def mass_operator(self) -> LinearOperator:
        """The mass operator (M) defining the weighted inner product."""
        return self._mass_operator

    @property
    def inverse_mass_operator(self) -> LinearOperator:
        """The inverse of the mass operator."""
        return self._inverse_mass_operator

    def to_components(self, x: Vector) -> np.ndarray:
        """Delegates component mapping to the underlying space."""
        return self.underlying_space.to_components(x)

    def from_components(self, c: np.ndarray) -> Vector:
        """Delegates vector creation to the underlying space."""
        return self.underlying_space.from_components(c)

    def to_dual(self, x: Vector) -> "LinearForm":
        """
        Computes the dual mapping `R_Y(x) = R_X(M x)`.
        """
        from .linear_forms import LinearForm

        y = self._mass_operator(x)
        yp = self.underlying_space.to_dual(y)
        return LinearForm(self, components=yp.components)

    def from_dual(self, xp: "LinearForm") -> Vector:
        """
        Computes the inverse dual mapping `R_Y^{-1}(xp) = M^{-1} R_X^{-1}(xp)`.
        """
        # Note: This implementation relies on the from_dual operator of the
        # underlying space not checking the domain of its argument. This is
        # acceptable and avoids an unnecessary copy.
        x = self.underlying_space.from_dual(xp)
        return self._inverse_mass_operator(x)

    def __eq__(self, other: object) -> bool:
        """
        Checks for equality with another MassWeightedHilbertSpace.

        Two mass-weighted spaces are considered equal if they share an equal
        underlying space and their mass operators are also equal.
        """
        if not isinstance(other, MassWeightedHilbertSpace):
            return NotImplemented

        return (
            self.underlying_space == other.underlying_space
            and (self.mass_operator == other.mass_operator)
            and (self.inverse_mass_operator == other.inverse_mass_operator)
        )


class MassWeightedHilbertModule(MassWeightedHilbertSpace, HilbertModule):
    """
    A mass-weighted Hilbert space that also supports vector multiplication.

    This class inherits the mass-weighted inner product structure and mixes in
    the `HilbertModule` interface, delegating the multiplication operation to
    the underlying space.
    """

    def __init__(
        self,
        underlying_space: HilbertModule,
        mass_operator: LinearOperator,
        inverse_mass_operator: LinearOperator,
    ):
        """
        Args:
            underlying_space: The original space (X) on which the inner
                product is defined.
            mass_operator: The self-adjoint, positive-definite mass
                operator (M).
            inverse_mass_operator: The inverse of the mass operator.
        """
        if not isinstance(underlying_space, HilbertModule):
            raise TypeError("Underlying space must be a HilbertModule.")

        MassWeightedHilbertSpace.__init__(
            self, underlying_space, mass_operator, inverse_mass_operator
        )

    def vector_multiply(self, x1: Vector, x2: Vector) -> Vector:
        """
        Computes vector multiplication by delegating to the underlying space.

        Note: This assumes the underlying space provided during initialization
        is itself an instance of `HilbertModule`.
        """
        return self.underlying_space.vector_multiply(x1, x2)
