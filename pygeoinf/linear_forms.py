"""
Provides the `LinearForm` class for representing linear functionals.

A linear form is a linear mapping from a vector in a Hilbert space to a
scalar. This class provides a concrete, component-based representation for
elements of the dual space of a `HilbertSpace`. It inherits from `NonLinearForm`,
specializing it for the linear case.
"""

from __future__ import annotations
from typing import Callable, Optional, Any, TYPE_CHECKING

from joblib import Parallel, delayed

import numpy as np

from .nonlinear_forms import NonLinearForm

# This block only runs for type checkers, not at runtime
if TYPE_CHECKING:
    from .hilbert_space import HilbertSpace, Vector
    from .linear_operators import LinearOperator


class LinearForm(NonLinearForm):
    """
    Represents a linear form as an efficient, component-based functional.

    A `LinearForm` is an element of a dual `HilbertSpace` and is defined by its
    action on vectors from its `domain`. Internally, this action is represented
    by a component vector. This class provides optimized arithmetic operations
    and correctly defines the gradient (a constant vector) and the Hessian
    (the zero operator) for any linear functional.
    """

    def __init__(
        self,
        domain: HilbertSpace,
        /,
        *,
        components: Optional[np.ndarray] = None,
        mapping: Optional[Callable[[Vector], float]] = None,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> None:
        """
        Initializes the LinearForm from a mapping or component vector.

        A form must be defined by exactly one of two methods:
        1.  **components**: The explicit component vector representing the form.
        2.  **mapping**: A function `f(x)` that defines the form's action.
            The components will be automatically computed from this mapping.


        Args:
            domain: The Hilbert space on which the form is defined.
            components: The component representation of the form.
            mapping: The functional mapping `f(x)`. Used if `components` is None.
            parallel: Whether to use parallel computing components from the mapping.
            n_jobs: The number of jobs to use for parallel computing.

        Raises:
            AssertionError: If neither or both `mapping` and `components`
                are specified.

        Notes:
            Parallel options only relevant if the form is defined by a mapping.

            If both `components` and `mapping` are specified, `components`
            will take precedence.
        """

        super().__init__(
            domain,
            self._mapping_impl,
            gradient=self._gradient_impl,
            hessian=self._hessian_impl,
        )

        if components is None:
            if mapping is None:
                raise AssertionError("Neither mapping nor components specified.")
            self._compute_components(mapping, parallel, n_jobs)
        else:
            self._components: np.ndarray = components

    @staticmethod
    def from_linear_operator(operator: "LinearOperator") -> LinearForm:
        """
        Creates a LinearForm from an operator that maps to a 1D Euclidean space.
        """
        from .hilbert_space import EuclideanSpace

        assert operator.codomain == EuclideanSpace(1)
        return LinearForm(operator.domain, mapping=lambda x: operator(x)[0])

    @property
    def domain(self) -> HilbertSpace:
        """The Hilbert space on which the form is defined."""
        return self._domain

    @property
    def components(self) -> np.ndarray:
        """
        The component vector of the form.
        """
        return self._components

    @property
    def as_linear_operator(self) -> "LinearOperator":
        """
        Represents the linear form as a `LinearOperator`.

        The resulting operator maps from the form's original domain to a
        1-dimensional `EuclideanSpace`, where the single component of the output
        is the scalar result of the form's action.
        """
        from .hilbert_space import EuclideanSpace
        from .linear_operators import LinearOperator

        return LinearOperator(
            self.domain,
            EuclideanSpace(1),
            lambda x: np.array([self(x)]),
            dual_mapping=lambda y: y * self,
        )

    def copy(self) -> LinearForm:
        """
        Creates a deep copy of the linear form.
        """
        return LinearForm(self.domain, components=self.components.copy())

    def __neg__(self) -> LinearForm:
        """Returns the additive inverse of the form."""
        return LinearForm(self.domain, components=-self._components)

    def __mul__(self, a: float) -> LinearForm:
        """Returns the product of the form and a scalar."""
        return LinearForm(self.domain, components=a * self._components)

    def __rmul__(self, a: float) -> LinearForm:
        """Returns the product of the form and a scalar."""
        return self * a

    def __truediv__(self, a: float) -> LinearForm:
        """Returns the division of the form by a scalar."""
        return self * (1.0 / a)

    def __add__(self, other: NonLinearForm | LinearForm) -> NonLinearForm | LinearForm:
        """
        Returns the sum of this form and another.

        If `other` is also a `LinearForm`, this performs an optimized,
        component-wise addition. Otherwise, it delegates to the general
        implementation in the `NonLinearForm` base class.

        Args:
            other: The form to add to this one.

        Returns:
            A `LinearForm` if adding two `LinearForm`s, otherwise a `NonLinearForm`.
        """
        if isinstance(other, LinearForm):
            return LinearForm(
                self.domain, components=self.components + other.components
            )
        else:
            return super().__add__(other)

    def __sub__(self, other: NonLinearForm | LinearForm) -> NonLinearForm | LinearForm:
        """
        Returns the difference of this form and another.

        If `other` is also a `LinearForm`, this performs an optimized,
        component-wise subtraction. Otherwise, it delegates to the general
        implementation in the `NonLinearForm` base class.

        Args:
            other: The form to subtract from this one.

        Returns:
            A `LinearForm` if subtracting two `LinearForm`s, otherwise a `NonLinearForm`.
        """
        if isinstance(other, LinearForm):
            return LinearForm(
                self.domain, components=self.components - other.components
            )
        else:
            return super().__sub__(other)

    def __imul__(self, a: float) -> "LinearForm":
        """
        Performs in-place scalar multiplication: self *= a.
        """
        self._components *= a
        return self

    def __iadd__(self, other: "LinearForm") -> "LinearForm":
        """
        Performs in-place addition with another form: self += other.
        """
        if self.domain != other.domain:
            raise ValueError("Linear forms must share the same domain for addition.")
        self._components += other.components
        return self

    def __str__(self) -> str:
        """Returns the string representation of the form's components."""
        return self.components.__str__()

    def _compute_components(
        self,
        mapping: Callable[[Any], float],
        parallel: bool,
        n_jobs: Optional[int],
    ):
        """Computes the component vector of the form, with an optional parallel backend."""
        if not parallel:
            self._components = np.zeros(self.domain.dim)
            cx = np.zeros(self.domain.dim)
            for i in range(self.domain.dim):
                cx[i] = 1.0
                x = self.domain.from_components(cx)
                self._components[i] = mapping(x)
                cx[i] = 0.0
        else:

            def compute_one_component(i: int) -> float:
                """
                Computes a single component for a given basis vector index.
                This function is sent to each parallel worker.
                """

                # cx = np.zeros(self.domain.dim)
                # cx[i] = 1.0
                # x = self.domain.from_components(cx)
                x = self.domain.basis_vector(i)
                return mapping(x)

            # Run the helper function in parallel for each dimension
            results = Parallel(n_jobs=n_jobs)(
                delayed(compute_one_component)(i) for i in range(self.domain.dim)
            )
            self._components = np.array(results)

    def _mapping_impl(self, x: Vector) -> float:
        """
        Maps a vector to its scalar value.
        """
        return np.dot(self.components, self.domain.to_components(x))

    def _gradient_impl(self, _: Vector) -> Vector:
        """
        Computes the gradient of the form at a point.
        """
        return self.domain.from_dual(self)

    def _hessian_impl(self, _: Vector) -> LinearOperator:
        """
        Computes the Hessian of the form at a point.
        """
        return self.domain.zero_operator()
