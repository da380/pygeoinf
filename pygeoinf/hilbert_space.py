"""
Module defining the core HilbertSpace and EuclideanSpace classes.
"""

from __future__ import annotations
from typing import TypeVar, Callable, List, Optional, Any, TYPE_CHECKING, Generic

import numpy as np

# This block only runs for type checkers, not at runtime
if TYPE_CHECKING:
    from .operators import LinearOperator
    from .forms import LinearForm


# Define a generic type for vectors in a Hilbert space
T_vec = TypeVar("T_vec")


class HilbertSpace:
    """
    A class for real Hilbert spaces.

    This class provides a mathematical abstraction for vector spaces equipped
    with an inner product. It separates the abstract vector operations from
    their concrete representation (e.g., as NumPy arrays).

    To define an instance, a user must provide the space's dimension, methods
    for converting vectors to/from their component representations, and,
    critically, the Riesz representation map (`to_dual`). **This map implicitly
    defines the space's inner product.**
    """

    def __init__(
        self,
        dim: int,
        to_components: Callable[[T_vec], np.ndarray],
        from_components: Callable[[np.ndarray], T_vec],
        to_dual: Callable[[T_vec], Any],
        from_dual: Callable[[Any], T_vec],
        /,
        *,
        add: Optional[Callable[[T_vec, T_vec], T_vec]] = None,
        subtract: Optional[Callable[[T_vec, T_vec], T_vec]] = None,
        multiply: Optional[Callable[[float, T_vec], T_vec]] = None,
        ax: Optional[Callable[[float, T_vec], None]] = None,
        axpy: Optional[Callable[[float, T_vec, T_vec], None]] = None,
        copy: Optional[Callable[[T_vec], T_vec]] = None,
        vector_multiply: Optional[Callable[[T_vec, T_vec], T_vec]] = None,
        base: Optional[HilbertSpace] = None,
    ):
        """
        Initializes the HilbertSpace.

        Args:
            dim (int): The dimension of the space.
            to_components (callable): A function mapping vectors to their
                NumPy component arrays.
            from_components (callable): A function mapping NumPy component
                arrays back to vectors.
            to_dual (callable): The Riesz map, which maps a vector `x` to its
                corresponding dual vector (a `LinearForm`). This function
                **implicitly defines the inner product** for the space, as
                `inner_product(x, y)` is implemented as `to_dual(x)(y)`.
            from_dual (callable): The inverse Riesz map, which maps a dual
                 vector back to its representative vector in the primal space.
            add (callable, optional): Custom vector addition.
            subtract (callable, optional): Custom vector subtraction.
            multiply (callable, optional): Custom scalar multiplication.
            ax (callable, optional): Custom in-place scaling x := a*x.
            axpy (callable, optional): Custom in-place operation y := a*x + y.
            copy (callable, optional): Custom deep copy for vectors.
            base (HilbertSpace, optional): Used internally for creating
                dual spaces. Should not be set by the user.
        """
        self._dim: int = dim
        self.__to_components: Callable[[T_vec], np.ndarray] = to_components
        self.__from_components: Callable[[np.ndarray], T_vec] = from_components
        self.__from_dual: Callable[[Any], T_vec] = from_dual
        self.__to_dual: Callable[[T_vec], Any] = to_dual
        self._base: Optional[HilbertSpace] = base
        self._add: Callable[[T_vec, T_vec], T_vec] = self.__add if add is None else add
        self._subtract: Callable[[T_vec, T_vec], T_vec] = (
            self.__subtract if subtract is None else subtract
        )
        self._multiply: Callable[[float, T_vec], T_vec] = (
            self.__multiply if multiply is None else multiply
        )
        self._ax: Callable[[float, T_vec], None] = self.__ax if ax is None else ax
        self._axpy: Callable[[float, T_vec, T_vec], None] = (
            self.__axpy if axpy is None else axpy
        )
        self._copy: Callable[[T_vec], T_vec] = self.__copy if copy is None else copy
        self._vector_multiply: Optional[Callable[[T_vec, T_vec], T_vec]] = (
            vector_multiply
        )

    @property
    def dim(self) -> int:
        """The dimension of the space."""
        return self._dim

    @property
    def has_vector_multiply(self) -> bool:
        """True if multiplication of elements is defined."""
        return self._vector_multiply is not None

    @property
    def dual(self) -> HilbertSpace:
        """
        The dual of the Hilbert space.

        The dual space is the space of all continuous linear functionals
        that map vectors from the Hilbert space to real numbers.
        """
        if self._base is None:
            return HilbertSpace(
                self.dim,
                self._dual_to_components,
                self._dual_from_components,
                self.from_dual,
                self.to_dual,
                base=self,
            )
        else:
            return self._base

    @property
    def zero(self) -> T_vec:
        """Returns the zero vector for the space."""
        return self.from_components(np.zeros((self.dim)))

    @property
    def coordinate_inclusion(self) -> LinearOperator:
        """
        Returns the operator mapping coordinate vectors in R^n to vectors
        in this Hilbert space.
        """
        from .operators import LinearOperator

        domain = EuclideanSpace(self.dim)

        def dual_mapping(xp: Any) -> Any:
            cp = self.dual.to_components(xp)
            return domain.to_dual(cp)

        def adjoint_mapping(y: T_vec) -> np.ndarray:
            yp = self.to_dual(y)
            return self.dual.to_components(yp)

        return LinearOperator(
            domain,
            self,
            self.from_components,
            dual_mapping=dual_mapping,
            adjoint_mapping=adjoint_mapping,
        )

    @property
    def coordinate_projection(self) -> LinearOperator:
        """
        Returns the operator mapping vectors in this Hilbert space to their
        coordinate vectors in R^n.
        """
        from .operators import LinearOperator

        codomain = EuclideanSpace(self.dim)

        def dual_mapping(cp: Any) -> Any:
            c = codomain.from_dual(cp)
            return self.dual.from_components(c)

        def adjoint_mapping(c: np.ndarray) -> T_vec:
            xp = self.dual.from_components(c)
            return self.from_dual(xp)

        return LinearOperator(
            self,
            codomain,
            self.to_components,
            dual_mapping=dual_mapping,
            adjoint_mapping=adjoint_mapping,
        )

    @property
    def riesz(self) -> LinearOperator:
        """
        Returns the Riesz map (dual to primal) as a LinearOperator.
        """
        from .operators import LinearOperator

        return LinearOperator.self_dual(self.dual, self.from_dual)

    @property
    def inverse_riesz(self) -> LinearOperator:
        """
        Returns the inverse Riesz map (primal to dual) as a LinearOperator.
        """
        from .operators import LinearOperator

        return LinearOperator.self_dual(self, self.to_dual)

    def inner_product(self, x1: T_vec, x2: T_vec) -> float:
        """Computes the inner product of two vectors."""
        if self._base is None:
            return self.to_dual(x1)(x2)
        else:
            return x1(self.to_dual(x2))

    def squared_norm(self, x: T_vec) -> float:
        """Computes the squared norm of a vector."""
        return self.inner_product(x, x)

    def norm(self, x: T_vec) -> float:
        """Computes the norm of a vector."""
        return np.sqrt(self.squared_norm(x))

    def gram_schmidt(self, vectors: List[T_vec]) -> List[T_vec]:
        """
        Orthonormalizes a list of vectors using the Gram-Schmidt process.
        """
        if not all(self.is_element(vector) for vector in vectors):
            raise ValueError("Not all vectors are elements of the space")

        orthonormalised_vectors: List[T_vec] = []
        for i, vector in enumerate(vectors):
            vec_copy = self.copy(vector)
            for j in range(i):
                product = self.inner_product(vec_copy, orthonormalised_vectors[j])
                self.axpy(-product, orthonormalised_vectors[j], vec_copy)
            norm = self.norm(vec_copy)
            self.ax(1 / norm, vec_copy)
            orthonormalised_vectors.append(vec_copy)

        return orthonormalised_vectors

    def to_dual(self, x: T_vec) -> Any:
        """Maps a vector to its canonical dual vector (a linear functional)."""
        return self.__to_dual(x)

    def from_dual(self, xp: Any) -> T_vec:
        """Maps a dual vector to its representative in the primal space."""
        return self.__from_dual(xp)

    def is_element(self, x: Any) -> bool:
        """
        Checks if an object is an element of the space.

        Note: The current implementation checks type against the zero vector.
        This may not be robust for all vector representations.
        """
        return isinstance(x, type(self.zero))

    def add(self, x: T_vec, y: T_vec) -> T_vec:
        """Adds two vectors."""
        return self._add(x, y)

    def subtract(self, x: T_vec, y: T_vec) -> T_vec:
        """Subtracts two vectors."""
        return self._subtract(x, y)

    def multiply(self, a: float, x: T_vec) -> T_vec:
        """Performs scalar multiplication, returning a new vector."""
        return self._multiply(a, x)

    def negative(self, x: T_vec) -> T_vec:
        """Returns the additive inverse of a vector."""
        return self.multiply(-1.0, x)

    def ax(self, a: float, x: T_vec) -> None:
        """Performs the in-place scaling operation x := a*x."""
        self._ax(a, x)

    def axpy(self, a: float, x: T_vec, y: T_vec) -> None:
        """Performs the in-place vector operation y := y + a*x."""
        self._axpy(a, x, y)

    def copy(self, x: T_vec) -> T_vec:
        """Returns a deep copy of a vector."""
        return self._copy(x)

    def vector_multiply(self, x1: T_vec, x2: T_vec) -> T_vec:
        """
        Returns the product of two elements of the space, if defined.
        """
        if self._vector_multiply is None:
            raise NotImplementedError(
                "Vector multiplication not defined on this space."
            )
        return self._vector_multiply(x1, x2)

    def to_components(self, x: T_vec) -> np.ndarray:
        """Maps a vector to its NumPy component array."""
        return self.__to_components(x)

    def from_components(self, c: np.ndarray) -> T_vec:
        """Maps a NumPy component array to a vector."""
        return self.__from_components(c)

    def basis_vector(self, i: int) -> T_vec:
        """Returns the i-th standard basis vector."""
        c = np.zeros(self.dim)
        c[i] = 1
        return self.from_components(c)

    def random(self) -> T_vec:
        """
        Returns a random vector from the space.

        The vector's components are drawn from a standard Gaussian distribution.
        """
        return self.from_components(np.random.randn(self.dim))

    def sample_expectation(self, vectors: List[T_vec]) -> T_vec:
        """Computes the sample mean of a list of vectors."""
        n = len(vectors)
        if not all(self.is_element(x) for x in vectors):
            raise TypeError("Not all items in list are elements of the space.")
        xbar = self.zero
        for x in vectors:
            self.axpy(1 / n, x, xbar)
        return xbar

    def identity_operator(self) -> LinearOperator:
        """Returns the identity operator on the space."""
        from .operators import LinearOperator

        return LinearOperator(
            self,
            self,
            lambda x: x,
            dual_mapping=lambda yp: yp,
            adjoint_mapping=lambda y: y,
        )

    def zero_operator(self, codomain: Optional[HilbertSpace] = None) -> LinearOperator:
        """
        Returns the zero operator from this space to a codomain.

        If no codomain is provided, it maps to itself.
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

    def _dual_to_components(self, xp: LinearForm) -> np.ndarray:
        return xp.components

    def _dual_from_components(self, cp: np.ndarray) -> LinearForm:
        from .forms import LinearForm

        return LinearForm(self, components=cp)

    def __add(self, x: T_vec, y: T_vec) -> T_vec:
        return x + y

    def __subtract(self, x: T_vec, y: T_vec) -> T_vec:
        return x - y

    def __multiply(self, a: float, x: T_vec) -> T_vec:
        return a * x.copy()

    def __ax(self, a: float, x: T_vec) -> None:
        x *= a

    def __axpy(self, a: float, x: T_vec, y: T_vec) -> None:
        y += a * x

    def __copy(self, x: T_vec) -> T_vec:
        return x.copy()


class EuclideanSpace(HilbertSpace):
    """
    An n-dimensional Euclidean space, R^n.

    This is a concrete implementation of HilbertSpace where vectors are
    represented directly by NumPy arrays.
    """

    def __init__(self, dim: int) -> None:
        """
        Args:
            dim (int): Dimension of the space.
        """
        super().__init__(
            dim,
            lambda x: x,
            lambda x: x,
            self._to_dual_impl,
            self._from_dual_impl,
        )

    def _to_dual_impl(self, x: np.ndarray) -> LinearForm:
        return self.dual.from_components(x)

    def _from_dual_impl(self, xp: LinearForm) -> np.ndarray:
        cp = self.dual.to_components(xp)
        return self.from_components(cp)

    def __eq__(self, other: object) -> bool:
        """Checks for equality with another EuclideanSpace."""
        return isinstance(other, EuclideanSpace) and self.dim == other.dim


class LebesgueSpace(HilbertSpace, Generic[T_vec]):
    """A factory for Hilbert spaces with a standard inner product.

        This class simplifies the creation of a `HilbertSpace` where the inner
        product is the standard Euclidean dot product of the vectors' component
        representations.

        By providing only the `to_components` and `from_components` mappings,
        this class **automatically defines the necessary Riesz maps**
        (`to_dual` and `from_dual`) that are consistent with this standard
        inner product.

    Example:
        If you have a custom vector class `MyVector` and functions
        `to_numpy(v: MyVector)` and `from_numpy(arr: np.ndarray)`, you can
        define the space simply as:
        `space = LebesgueSpace(dim, to_numpy, from_numpy)`
    """

    def __init__(
        self,
        dim: int,
        to_components: Callable[[T_vec], np.ndarray],
        from_components: Callable[[np.ndarray], T_vec],
        /,
        *,
        add: Optional[Callable[[T_vec, T_vec], T_vec]] = None,
        subtract: Optional[Callable[[T_vec, T_vec], T_vec]] = None,
        multiply: Optional[Callable[[float, T_vec], T_vec]] = None,
        ax: Optional[Callable[[float, T_vec], None]] = None,
        axpy: Optional[Callable[[float, T_vec, T_vec], None]] = None,
        copy: Optional[Callable[[T_vec], T_vec]] = None,
        vector_multiply: Optional[Callable[[T_vec, T_vec], T_vec]] = None,
    ):
        """Initializes the LebesgueSpace.

        Args:
            dim (int): The dimension of the space.
            to_components (callable): A function mapping abstract vectors to their
                NumPy component arrays.
            from_components (callable): A function mapping NumPy component
                arrays back to abstract vectors.
            add (callable, optional): Custom vector addition.
            subtract (callable, optional): Custom vector subtraction.
            multiply (callable, optional): Custom scalar multiplication.
            ax (callable, optional): Custom in-place scaling x := a*x.
            axpy (callable, optional): Custom in-place operation y := a*x + y.
            copy (callable, optional): Custom deep copy for vectors.
            vector_multiply (callable, optional): Custom element-wise product.
        """
        super().__init__(
            dim,
            to_components,
            from_components,
            self._to_dual_from_components,
            self._from_dual_from_components,
            add=add,
            subtract=subtract,
            multiply=multiply,
            ax=ax,
            axpy=axpy,
            copy=copy,
            vector_multiply=vector_multiply,
        )

    def _to_dual_from_components(self, x: T_vec) -> "LinearForm":
        """Maps a vector to its dual form via its components."""
        c = self.to_components(x)
        return self.dual.from_components(c)

    def _from_dual_from_components(self, xp: "LinearForm") -> T_vec:
        """Maps a dual form back to a vector via its components."""
        cp = self.dual.to_components(xp)
        return self.from_components(cp)
