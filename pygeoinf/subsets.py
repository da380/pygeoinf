"""
Defines classes for representing subsets of a Hilbert space.

This module provides a hierarchy of classes for sets, ranging from abstract
definitions to concrete geometric shapes. It supports Constructive Solid
Geometry (CSG) operations, with specialized handling for convex intersections
via functional combination.

Hierarchy:
- Subset (Abstract Base)
    - EmptySet / UniversalSet
    - LevelSet (f(x) = c)
        - EllipsoidSurface -> Sphere
    - SublevelSet (f(x) <= c)
        - ConvexSubset -> Ellipsoid -> Ball
        - ConvexIntersection (Max-Functional Combination)
    - Intersection (Generic)
    - Union (Generic)
    - Complement (S^c)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, List, Iterable
import numpy as np

from .nonlinear_forms import NonLinearForm

if TYPE_CHECKING:
    from .hilbert_space import HilbertSpace, Vector
    from .linear_operators import LinearOperator
    from .convex_analysis import SupportFunction, HalfSpaceSupportFunction


class Subset(ABC):
    """
    Abstract base class for a subset of a HilbertSpace.

    This class defines the minimal interface required for a mathematical set:
    knowing which space it lives in, determining if a vector belongs to it,
    accessing its boundary, and performing logical set operations.
    """

    def __init__(self, domain: Optional[HilbertSpace] = None) -> None:
        """
        Initializes the subset.

        Args:
            domain: The Hilbert space containing this subset. Can be None
                for sets that are not strictly attached to a specific domain
                (e.g., a generic EmptySet).
        """
        self._domain = domain

    @property
    def domain(self) -> HilbertSpace:
        """
        The underlying Hilbert space.

        Returns:
            The HilbertSpace instance associated with this subset.

        Raises:
            ValueError: If the domain was not set during initialization.
        """
        if self._domain is None:
            raise ValueError(
                f"{self.__class__.__name__} does not have an associated domain."
            )
        return self._domain

    @property
    def is_empty(self) -> bool:
        """
        Returns True if the set is known to be empty.

        Returns:
            bool: True if the set contains no elements, False otherwise.
            Note that returning False does not guarantee the set is non-empty,
            only that it is not trivially known to be empty.
        """
        return False

    @abstractmethod
    def is_element(self, x: Vector, /, *, rtol: float = 1e-6) -> bool:
        """
        Returns True if the vector x lies within the subset.

        Args:
            x: A vector from the domain.
            rtol: Relative tolerance for floating-point comparisons (e.g.,
                checking equality f(x) = c or inequality f(x) <= c).

        Returns:
            bool: True if x ∈ S, False otherwise.
        """

    @property
    @abstractmethod
    def boundary(self) -> Subset:
        """
        Returns the boundary of the subset.

        Returns:
            Subset: A new Subset instance representing ∂S.
        """

    # --- CSG Operations ---

    @property
    def complement(self) -> Subset:
        """
        Returns the complement of this set: S^c = {x | x not in S}.

        Returns:
            Complement: A generic Complement wrapper around this set.
        """
        return Complement(self)

    def intersect(self, other: Subset) -> Subset:
        """
        Returns the intersection of this set and another: S ∩ O.

        If both sets are instances of ConvexSubset, this returns a
        ConvexIntersection, which combines their functionals into a single
        convex constraint F(x) = max(f1(x), f2(x)).

        Args:
            other: Another Subset instance.

        Returns:
            Subset: A new set representing elements present in both sets.
        """
        # Collect all subsets if we are merging intersections
        subsets_to_merge = []

        if isinstance(self, Intersection):  # Includes ConvexIntersection
            subsets_to_merge.extend(self.subsets)
        else:
            subsets_to_merge.append(self)

        if isinstance(other, Intersection):
            subsets_to_merge.extend(other.subsets)
        else:
            subsets_to_merge.append(other)

        # Check if all parts are ConvexSubsets (defined by f(x) <= c)
        all_convex_functional = all(
            isinstance(s, ConvexSubset) for s in subsets_to_merge
        )

        if all_convex_functional:
            # We can combine them into a single ConvexSubset via max function
            return ConvexIntersection(subsets_to_merge)  # type: ignore

        # Fallback to generic set logic
        return Intersection(subsets_to_merge)

    def union(self, other: Subset) -> Union:
        """
        Returns the union of this set and another: S ∪ O.

        Args:
            other: Another Subset instance.

        Returns:
            Union: A new set representing elements present in either set.
        """
        subsets_to_merge = [self]
        if isinstance(other, Union):
            subsets_to_merge.extend(other.subsets)
        else:
            subsets_to_merge.append(other)
        return Union(subsets_to_merge)


class EmptySet(Subset):
    """
    Represents the empty set (∅).
    """

    @property
    def is_empty(self) -> bool:
        """Returns True, as this is the empty set."""
        return True

    def is_element(self, x: Vector, /, *, rtol: float = 1e-6) -> bool:
        """Returns False for any vector."""
        return False

    @property
    def boundary(self) -> Subset:
        """The boundary of an empty set is the empty set itself."""
        return self

    @property
    def complement(self) -> Subset:
        """The complement of the empty set is the whole space (Universal Set)."""
        return UniversalSet(self.domain)


class UniversalSet(Subset):
    """
    Represents the entire domain (Ω).
    """

    def is_element(self, x: Vector, /, *, rtol: float = 1e-6) -> bool:
        """Returns True for any vector in the domain."""
        return True

    @property
    def boundary(self) -> Subset:
        """The boundary of the entire topological space is empty."""
        return EmptySet(self.domain)

    @property
    def complement(self) -> Subset:
        """The complement of the universe is the empty set."""
        return EmptySet(self.domain)


class Complement(Subset):
    """
    Represents the complement of a set: S^c = {x | x ∉ S}.
    """

    def __init__(self, subset: Subset) -> None:
        """
        Args:
            subset: The set to complement.
        """
        super().__init__(subset.domain)
        self._subset = subset

    def is_element(self, x: Vector, /, *, rtol: float = 1e-6) -> bool:
        """
        Returns True if x is NOT in the underlying subset.
        """
        return not self._subset.is_element(x, rtol=rtol)

    @property
    def boundary(self) -> Subset:
        """
        Returns the boundary of the complement.

        Ideally, ∂(S^c) = ∂S.
        """
        return self._subset.boundary

    @property
    def complement(self) -> Subset:
        """
        Returns the complement of the complement, which is the original set.
        (S^c)^c = S.
        """
        return self._subset


class Intersection(Subset):
    """
    Represents the generic intersection of multiple subsets: S = S_1 ∩ S_2 ...

    Used when the subsets cannot be mathematically combined into a single functional
    (e.g., non-convex sets).
    """

    def __init__(self, subsets: Iterable[Subset]) -> None:
        """
        Args:
            subsets: An iterable of Subset objects to intersect.
                     All subsets must belong to the same domain.
        """
        subsets_list = list(subsets)
        if not subsets_list:
            raise ValueError("Intersection requires at least one subset.")
        domain = subsets_list[0].domain

        # Validate domains match
        for s in subsets_list:
            if s.domain != domain:
                raise ValueError("All subsets must belong to the same domain.")

        super().__init__(domain)
        self._subsets = subsets_list

    @property
    def subsets(self) -> List[Subset]:
        """Direct access to the component sets."""
        return self._subsets

    def is_element(self, x: Vector, /, *, rtol: float = 1e-6) -> bool:
        """Returns True if x is in ALL component subsets."""
        return all(s.is_element(x, rtol=rtol) for s in self._subsets)

    @property
    def boundary(self) -> Subset:
        """
        Returns the boundary of the intersection.

        The general topological boundary is complex: ∂(A ∩ B) ⊆ (∂A ∩ B) ∪ (A ∩ ∂B).
        Currently raises NotImplementedError.
        """
        raise NotImplementedError(
            "General boundary of intersection not yet implemented."
        )

    @property
    def complement(self) -> Subset:
        """
        Returns the complement of the intersection.

        Applies De Morgan's Law: (A ∩ B)^c = A^c ∪ B^c.
        Returns a Union of the complements.
        """
        return Union(s.complement for s in self._subsets)


class Union(Subset):
    """
    Represents the union of multiple subsets: S = S_1 ∪ S_2 ...
    """

    def __init__(self, subsets: Iterable[Subset]) -> None:
        """
        Args:
            subsets: An iterable of Subset objects to unite.
                     All subsets must belong to the same domain.
        """
        subsets_list = list(subsets)
        if not subsets_list:
            raise ValueError("Union requires at least one subset.")
        domain = subsets_list[0].domain
        for s in subsets_list:
            if s.domain != domain:
                raise ValueError("All subsets must belong to the same domain.")
        super().__init__(domain)
        self._subsets = subsets_list

    @property
    def subsets(self) -> List[Subset]:
        """Direct access to the component sets."""
        return self._subsets

    def is_element(self, x: Vector, /, *, rtol: float = 1e-6) -> bool:
        """Returns True if x is in ANY of the component subsets."""
        return any(s.is_element(x, rtol=rtol) for s in self._subsets)

    @property
    def boundary(self) -> Subset:
        """
        Returns the boundary of the union.
        Currently raises NotImplementedError.
        """
        raise NotImplementedError("General boundary of union not yet implemented.")

    @property
    def complement(self) -> Subset:
        """
        Returns the complement of the union.

        Applies De Morgan's Law: (A ∪ B)^c = A^c ∩ B^c.
        Returns an Intersection of the complements.
        """
        return Intersection(s.complement for s in self._subsets)


class SublevelSet(Subset):
    """
    Represents a sublevel set defined by a functional: S = {x | f(x) <= c}.

    This class serves as a base for sets defined by inequalities. Unlike
    ConvexSubset, it does not assume the defining functional is convex.
    """

    def __init__(
        self,
        form: NonLinearForm,
        level: float,
        open_set: bool = False,
    ) -> None:
        """
        Args:
            form: The defining functional f(x).
            level: The scalar upper bound c.
            open_set: If True, uses strict inequality (f(x) < c).
                      If False, uses non-strict inequality (f(x) <= c).
        """
        super().__init__(form.domain)
        self._form = form
        self._level = level
        self._open = open_set

    @property
    def form(self) -> NonLinearForm:
        """The defining functional f(x)."""
        return self._form

    @property
    def level(self) -> float:
        """The scalar upper bound c."""
        return self._level

    @property
    def is_open(self) -> bool:
        """True if the set is defined by strict inequality."""
        return self._open

    def is_element(self, x: Vector, /, *, rtol: float = 1e-6) -> bool:
        """
        Returns True if f(x) <= c (or < c).
        Tolerance is scaled by max(1.0, |c|).
        """
        val = self._form(x)
        scale = max(1.0, abs(self._level))
        margin = rtol * scale

        if self._open:
            return val < self._level + margin
        else:
            return val <= self._level + margin

    @property
    def boundary(self) -> Subset:
        """
        Returns the boundary of the sublevel set.
        The boundary is typically the LevelSet {x | f(x) = c}.
        """
        return LevelSet(self._form, self._level)


class LevelSet(Subset):
    """
    Represents a level set of a functional: S = {x | f(x) = c}.
    """

    def __init__(
        self,
        form: NonLinearForm,
        level: float,
    ) -> None:
        """
        Args:
            form: The defining functional f(x).
            level: The scalar value c.
        """
        super().__init__(form.domain)
        self._form = form
        self._level = level

    @property
    def form(self) -> NonLinearForm:
        """The defining functional f(x)."""
        return self._form

    @property
    def level(self) -> float:
        """The scalar value c."""
        return self._level

    def is_element(self, x: Vector, /, *, rtol: float = 1e-6) -> bool:
        """
        Returns True if f(x) is approximately equal to c.
        Tolerance is scaled by max(1.0, |c|).
        """
        val = self._form(x)
        scale = max(1.0, abs(self._level))
        return abs(val - self._level) <= rtol * scale

    @property
    def boundary(self) -> Subset:
        """
        Returns the boundary of the level set.
        Assuming regularity, a level set is a closed manifold without boundary.
        """
        return EmptySet(self.domain)


class ConvexSubset(SublevelSet):
    """
    Represents a closed convex set via dual representations.

    A closed convex set can be equivalently defined by:
    1. A sublevel set: S = {x | f(x) <= c} where f is convex
    2. Its support function: h(q) = sup{⟨q, x⟩ : x ∈ S}

    This class supports both representations. The support function is abstract
    and must be implemented by all concrete subclasses (Ball, Ellipsoid, etc.).
    """

    def __init__(
        self,
        form: NonLinearForm,
        level: float,
        open_set: bool = False,
        support_fn: Optional["SupportFunction"] = None,
    ) -> None:
        """
        Args:
            form: The defining functional f(x). Must be convex.
            level: The scalar upper bound c.
            open_set: If True, uses strict inequality (<). If False (default), (<=).
                Note: Support functions only apply to closed sets.
            support_fn: Optional SupportFunction object. If provided, can be used
                to define the set via its support function instead of a functional.
        """
        super().__init__(form, level, open_set=open_set)
        self._support_fn = support_fn

    @property
    def support_fn(self) -> Optional["SupportFunction"]:
        """Access the stored support function object, if any."""
        return self._support_fn

    @property
    def is_closed(self) -> bool:
        """
        Returns True if the set is closed (defined by <=), False if open (<).
        """
        return not self.is_open

    def closure(self) -> "ConvexSubset":
        """
        Returns the closure of this convex set.

        For a convex set S:
        - If S is already closed ({f <= c}), returns self (no copy needed).
        - If S is open ({f < c}), returns a new closed version.

        The closure is the smallest closed set containing S.
        Note: The returned object uses the same functional and level as self,
        with open_set flag set to False.

        Returns:
            ConvexSubset: A ConvexSubset instance (subclass) representing cl(S).
        """
        if self.is_closed:
            # Already closed; return self
            return self
        else:
            # Open set: create a closed version via the subclass's constructor
            # We use type(self) to call the appropriate subclass constructor
            # This works for Ball, Ellipsoid, etc. which override __init__
            return type(self)(
                self.form, self.level, open_set=False, support_fn=self._support_fn
            )

    def _warn_if_open(self, operation: str) -> "ConvexSubset":
        """
        Helper: warns if the set is open and returns the closure.

        Args:
            operation: Description of the operation requiring closedness.

        Returns:
            ConvexSubset: The closure if open; self if already closed.
        """
        if self.is_open:
            import warnings
            warnings.warn(
                f"Operation '{operation}' requires a closed convex set. "
                f"Using the closure cl(S) = {{x | f(x) <= {self.level}}} instead. "
                f"Original set was {{x | f(x) < {self.level}}}.",
                UserWarning,
                stacklevel=3,
            )
            return self.closure()
        return self

    @property
    @abstractmethod
    def support_function(self) -> Optional["SupportFunction"]:
        """
        Returns the SupportFunction instance for this set, or None if unavailable.

        This property should not raise errors if the support function cannot be
        created (e.g., missing inverse operators for ellipsoids). The SupportFunction
        itself is responsible for raising errors when required inputs are missing
        during evaluation.
        """

    @abstractmethod
    def directional_bound(self, direction: "Vector") -> tuple["Vector", float]:
        """
        Returns extreme point and support value in a given direction.

        For a convex set S and direction q, computes:
            x_max = argmax{⟨q, x⟩ : x ∈ S}
            h(q) = sup{⟨q, x⟩ : x ∈ S}

        This method must be implemented by all concrete ConvexSubset subclasses.

        Args:
            direction: A vector q specifying the direction.

        Returns:
            tuple[Vector, float]: (x_max, h(q)) where x_max achieves the supremum
                and h(q) is the support function value.

        Raises:
            NotImplementedError: If not implemented for this set type.
        """

    def check(
        self, n_samples: int = 10, /, *, rtol: float = 1e-5, atol: float = 1e-8
    ) -> None:
        """
        Performs a randomized check of the convexity inequality:
        f(tx + (1-t)y) <= t*f(x) + (1-t)*f(y)

        Args:
            n_samples: Number of random pairs to test.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        Raises:
            AssertionError: If the function is found to be non-convex.
        """
        for _ in range(n_samples):
            x = self.domain.random()
            y = self.domain.random()
            t = np.random.uniform(0, 1)

            # Convex combination
            tx = self.domain.multiply(t, x)
            ty = self.domain.multiply(1.0 - t, y)
            z = self.domain.add(tx, ty)

            # Evaluate form
            fz = self._form(z)
            fx = self._form(x)
            fy = self._form(y)

            # f(tx + (1-t)y) <= t f(x) + (1-t) f(y)
            rhs = t * fx + (1.0 - t) * fy

            if fz > rhs + atol + rtol * abs(rhs):
                raise AssertionError(
                    f"Convexity check failed. "
                    f"LHS={fz:.4e}, RHS={rhs:.4e}. "
                    "Functional does not appear convex."
                )
        print(f"[✓] Convexity check passed ({n_samples} samples).")


class ConvexIntersection(ConvexSubset):
    """
    Represents the intersection of multiple convex sets as a single convex set.

    This class combines the defining functionals of its components into a single
    max-functional: F(x) = max_i (f_i(x) - c_i).
    The intersection is then defined as {x | F(x) <= 0}.

    This allows the intersection to be treated as a standard ConvexSubset for
    optimization algorithms, providing gradients and Hessians of the active
    constraint, while preserving access to the individual constraints.
    """

    def __init__(self, subsets: Iterable[ConvexSubset]) -> None:
        """
        Args:
            subsets: An iterable of ConvexSubset objects.
        """
        self._subsets = list(subsets)
        if not self._subsets:
            raise ValueError("ConvexIntersection requires at least one subset.")

        domain = self._subsets[0].domain

        # 1. Define the combined max-mapping
        # F(x) = max (f_i(x) - c_i)
        def mapping(x: Vector) -> float:
            values = [s.form(x) - s.level for s in self._subsets]
            return float(np.max(values))

        # 2. Define the gradient via the active constraint
        # dF(x) = df_k(x) where k = argmax(...)
        # Note: At points where multiple constraints are active, this returns
        # a subgradient (from one of the active sets).
        def gradient(x: Vector) -> Vector:
            values = [s.form(x) - s.level for s in self._subsets]
            idx_max = np.argmax(values)
            active_subset = self._subsets[idx_max]
            return active_subset.form.gradient(x)

        # 3. Define the Hessian via the active constraint
        def hessian(x: Vector) -> LinearOperator:
            values = [s.form(x) - s.level for s in self._subsets]
            idx_max = np.argmax(values)
            active_subset = self._subsets[idx_max]
            return active_subset.form.hessian(x)

        combined_form = NonLinearForm(
            domain, mapping, gradient=gradient, hessian=hessian
        )

        # Determine strictness: if any subset is open, the intersection boundary
        # handling gets complex. We adopt a conservative approach: effectively closed
        # for calculation (level=0), but flagging open if any component is open.
        is_any_open = any(s.is_open for s in self._subsets)

        super().__init__(combined_form, level=0.0, open_set=is_any_open)

    @property
    def subsets(self) -> List[ConvexSubset]:
        """Direct access to the individual convex constraints."""
        return self._subsets

    @property
    def support_function(self) -> Optional["SupportFunction"]:
        """
        Support function of an intersection is not generally available.

        In general,
            σ_{∩_i C_i}(q) != min_i σ_{C_i}(q).

        The pointwise minimum of support functions is typically NOT sublinear,
        hence not a valid support function of any closed convex set.

        Returns:
            None: by default. Use `support_upper_bound(direction)` for a safe
            upper bound, or provide an optimizer to compute the true support.
        """
        return None

    def support_upper_bound(self, direction: "Vector") -> float:
        """
        Safe upper bound for the true intersection support.

        Because ∩_i C_i ⊆ C_i for each i, we always have
            σ_{∩_i C_i}(q) ≤ min_i σ_{C_i}(q).

        Returns:
            float: min_i σ_{C_i}(direction)

        Raises:
            ValueError: if any component subset lacks a support function.
        """
        values: List[float] = []
        for subset in self._subsets:
            h = subset.support_function
            if h is None:
                raise ValueError(
                    "Cannot compute support_upper_bound: at least one component "
                    "subset lacks a support function."
                )
            values.append(float(h(direction)))
        return float(np.min(values))

    def feasible_lower_bound(
        self, direction: "Vector", /, *, rtol: float = 1e-6
    ) -> tuple[Optional["Vector"], float]:
        """
        Feasible lower bound on the true intersection support.

        Strategy:
            Try candidate maximizers from each component set (their
            `directional_bound`). Keep the best candidate that is feasible
            for ALL constraints (i.e., belongs to the intersection).

        This is always safe (never overestimates), but may be loose.

        Returns:
            (x_best, value_best):
              - x_best is a feasible candidate point (or None if none found)
              - value_best = ⟨direction, x_best⟩, or -∞ if none found
        """
        best_x: Optional["Vector"] = None
        best_val: float = float(np.NINF)

        for subset in self._subsets:
            x_cand, _ = subset.directional_bound(direction)
            if all(s.is_element(x_cand, rtol=rtol) for s in self._subsets):
                val = float(self.domain.inner_product(direction, x_cand))
                if val > best_val:
                    best_val = val
                    best_x = x_cand

        return best_x, best_val

    def directional_bound(self, direction: "Vector") -> tuple["Vector", float]:
        """
        Directional bound for an intersection requires solving a constrained
        maximization problem.

        In general, picking the component set with the smallest support value
        is NOT correct, because the corresponding support point need not satisfy
        the other constraints.

        Use:
          - `support_upper_bound(direction)` for a safe upper bound
          - `feasible_lower_bound(direction)` for a safe (feasible) lower bound
          - or supply/implement an optimizer to solve the true problem.
        """
        raise NotImplementedError(
            "ConvexIntersection.directional_bound is not available by default. "
            "Computing the true directional bound requires solving a constrained "
            "maximization over the intersection. Use support_upper_bound() / "
            "feasible_lower_bound(), or provide an optimizer."
        )


# --- Geometric Implementations ---


class _EllipsoidalGeometry:
    """
    Mixin class that holds the common data and logic for ellipsoidal sets.

    This class constructs the quadratic form f(x) = <A(x-c), x-c> used by
    Ellipsoid and EllipsoidSurface.
    """

    def __init__(
        self,
        domain: HilbertSpace,
        center: Vector,
        radius: float,
        operator: LinearOperator,
    ) -> None:
        """
        Args:
            domain: The Hilbert space.
            center: The center vector c.
            radius: The radius r.
            operator: The self-adjoint, positive-definite operator A defining the metric.
        """
        if not domain.is_element(center):
            raise ValueError("Center vector must be in the domain.")
        if radius < 0:
            raise ValueError("Radius must be non-negative.")

        self._center = center
        self._radius = radius
        self._operator = operator

        # 1. f(x) = <A(x-c), x-c>
        def mapping(x: Vector) -> float:
            diff = domain.subtract(x, center)
            return domain.inner_product(operator(diff), diff)

        # 2. f'(x) = 2 A (x-c)
        def gradient(x: Vector) -> Vector:
            diff = domain.subtract(x, center)
            return domain.multiply(2.0, operator(diff))

        # 3. f''(x) = 2 A
        def hessian(x: Vector) -> LinearOperator:
            return 2.0 * operator

        self._generated_form = NonLinearForm(
            domain, mapping, gradient=gradient, hessian=hessian
        )
        self._generated_level = radius**2

    @property
    def center(self) -> Vector:
        """The center of the ellipsoid."""
        return self._center

    @property
    def radius(self) -> float:
        """The 'radius' parameter of the ellipsoid."""
        return self._radius

    @property
    def operator(self) -> LinearOperator:
        """The defining linear operator A."""
        return self._operator


class Ellipsoid(ConvexSubset, _EllipsoidalGeometry):
    """
    Represents a solid ellipsoid: E = {x | <A(x-c), x-c> <= r^2}.
    """

    def __init__(
        self,
        domain: HilbertSpace,
        center: Vector,
        radius: float,
        operator: LinearOperator,
        open_set: bool = False,
        *,
        inverse_operator: Optional[LinearOperator] = None,
        inverse_sqrt_operator: Optional[LinearOperator] = None,
    ) -> None:
        """
        Args:
            domain: The Hilbert space.
            center: The center vector c.
            radius: The radius r.
            operator: The operator A.
            open_set: If True, defines an open ellipsoid (< r^2).
        """
        _EllipsoidalGeometry.__init__(self, domain, center, radius, operator)
        ConvexSubset.__init__(
            self, self._generated_form, self._generated_level, open_set=open_set
        )
        self._inverse_operator = inverse_operator
        self._inverse_sqrt_operator = inverse_sqrt_operator

    @property
    def boundary(self) -> Subset:
        """Returns the boundary EllipsoidSurface."""
        return EllipsoidSurface(self.domain, self.center, self.radius, self.operator)

    @property
    def normalized(self) -> "NormalisedEllipsoid":
        """
        Returns a normalized version of this ellipsoid with radius 1.
        The operator is scaled by 1/r^2 to represent the same set.
        """
        if self.radius == 0:
            raise ValueError("Cannot normalize an ellipsoid with zero radius.")
        scale = 1.0 / (self.radius**2)
        scaled_operator = scale * self.operator
        return NormalisedEllipsoid(
            self.domain, self.center, scaled_operator, open_set=self.is_open
        )

    @property
    def support_function(self) -> Optional["SupportFunction"]:
        """
        Returns the support function object for this ellipsoid.

        This property does not require inverse operators at construction time.
        The returned SupportFunction will raise errors if missing operators are
        required for evaluation.
        """
        if self._support_fn is None:
            from .convex_analysis import EllipsoidSupportFunction

            self._support_fn = EllipsoidSupportFunction(
                self.domain,
                self.center,
                self.radius,
                self.operator,
                inverse_operator=self._inverse_operator,
                inverse_sqrt_operator=self._inverse_sqrt_operator,
            )
        return self._support_fn

    def directional_bound(self, direction: "Vector") -> tuple["Vector", float]:
        """
        Returns extreme point in given direction.

        For an ellipsoid E(c, r, A) and direction q:
            x_max = c + r * (A^{-1} q) / ||A^{-1/2} q||
            h(q) = support_function(q)

        Args:
            direction: The direction vector q.

        Returns:
            tuple[Vector, float]: (x_max, h(q))
        """
        self._warn_if_open("directional_bound")
        h = self.support_function
        if h is None:
            raise ValueError("Support function is not available for this ellipsoid.")

        x_max = h.support_point(direction)
        if x_max is None:
            raise NotImplementedError(
                "directional_bound requires inverse_operator for ellipsoid support point. "
                "Provide inverse_operator when constructing the ellipsoid."
            )

        return (x_max, h(direction))


class NormalisedEllipsoid(Ellipsoid):
    """
    Represents a normalised ellipsoid with radius 1: E = {x | <A(x-c), x-c> <= 1}.
    """

    def __init__(
        self,
        domain: HilbertSpace,
        center: Vector,
        operator: LinearOperator,
        open_set: bool = False,
    ) -> None:
        """
        Args:
            domain: The Hilbert space.
            center: The center vector c.
            operator: The operator A.
            open_set: If True, defines an open ellipsoid.
        """
        super().__init__(domain, center, 1.0, operator, open_set=open_set)


class EllipsoidSurface(LevelSet, _EllipsoidalGeometry):
    """
    Represents the surface of an ellipsoid: S = {x | <A(x-c), x-c> = r^2}.
    """

    def __init__(
        self,
        domain: HilbertSpace,
        center: Vector,
        radius: float,
        operator: LinearOperator,
    ) -> None:
        """
        Args:
            domain: The Hilbert space.
            center: The center vector c.
            radius: The radius r.
            operator: The operator A.
        """
        _EllipsoidalGeometry.__init__(self, domain, center, radius, operator)
        LevelSet.__init__(self, self._generated_form, self._generated_level)

    @property
    def boundary(self) -> Subset:
        """Returns EmptySet (manifold without boundary)."""
        return EmptySet(self.domain)

    @property
    def normalized(self) -> EllipsoidSurface:
        """
        Returns a normalized version of this surface with radius 1.
        """
        if self.radius == 0:
            raise ValueError("Cannot normalize a surface with zero radius.")
        scale = 1.0 / (self.radius**2)
        scaled_operator = scale * self.operator
        return EllipsoidSurface(self.domain, self.center, 1.0, scaled_operator)


class Ball(Ellipsoid):
    """
    Represents a ball in a Hilbert space: B = {x | ||x - c||^2 <= r^2}.
    This is an Ellipsoid where A is the Identity operator.
    """

    def __init__(
        self,
        domain: HilbertSpace,
        center: Vector,
        radius: float,
        open_set: bool = True,
    ) -> None:
        """
        Args:
            domain: The Hilbert space.
            center: The center vector c.
            radius: The radius r.
            open_set: If True (default), defines an open ball (< r).
        """
        identity = domain.identity_operator()
        super().__init__(domain, center, radius, identity, open_set=open_set)

    def is_element(self, x: Vector, /, *, rtol: float = 1e-6) -> bool:
        """
        Returns True if x lies within the ball.
        Optimized to use geometric distance ||x-c|| directly.
        """
        diff = self.domain.subtract(x, self.center)
        dist = self.domain.norm(diff)
        margin = rtol * max(1.0, self.radius)
        if self.is_open:
            return dist < self.radius + margin
        else:
            return dist <= self.radius + margin

    @property
    def boundary(self) -> Subset:
        """Returns the Sphere bounding this Ball."""
        return Sphere(self.domain, self.center, self.radius)

    @property
    def support_function(self) -> Optional["SupportFunction"]:
        """
        Returns the support function object for this ball.

        Always available for valid Ball instances.
        """
        if self._support_fn is None:
            from .convex_analysis import BallSupportFunction

            self._support_fn = BallSupportFunction(
                self.domain, self.center, self.radius
            )
        return self._support_fn

    def directional_bound(self, direction: "Vector") -> tuple["Vector", float]:
        """
        Returns extreme point in direction of 'direction'.

        For a ball B(c, r) and direction q:
            x_max = c + r * (q / ||q||)
            h(q) = ⟨q, c⟩ + r||q||

        Args:
            direction: The direction vector q.

        Returns:
            tuple[Vector, float]: (x_max, h(q))
        """
        self._warn_if_open("directional_bound")
        h = self.support_function
        if h is None:
            raise ValueError("Support function is not available for this ball.")
        x_max = h.support_point(direction)
        if x_max is None:
            # Fallback: shouldn't happen for Ball
            q_norm = self.domain.norm(direction)
            if q_norm < 1e-14:
                return (self.center, 0.0)
            q_normalized = self.domain.multiply(1.0 / q_norm, direction)
            radial_displacement = self.domain.multiply(self.radius, q_normalized)
            x_max = self.domain.add(self.center, radial_displacement)
        return (x_max, h(direction))


class Sphere(EllipsoidSurface):
    """
    Represents a sphere in a Hilbert space: S = {x | ||x - c||^2 = r^2}.
    This is an EllipsoidSurface where A is the Identity operator.
    """

    def __init__(self, domain: HilbertSpace, center: Vector, radius: float) -> None:
        """
        Args:
            domain: The Hilbert space.
            center: The center vector c.
            radius: The radius r.
        """
        identity = domain.identity_operator()
        super().__init__(domain, center, radius, identity)

    def is_element(self, x: Vector, /, *, rtol: float = 1e-6) -> bool:
        """
        Returns True if ||x - c|| is approximately equal to r.
        """
        diff = self.domain.subtract(x, self.center)
        dist = self.domain.norm(diff)
        return abs(dist - self.radius) <= rtol * max(1.0, self.radius)

class HyperPlane(Subset):
    """
    Represents a hyperplane in a Hilbert space: H = {x | ⟨a, x⟩ = b}.

    A hyperplane is a flat affine subspace of codimension 1, defined by a
    normal vector 'a' and an offset scalar 'b'.
    """

    def __init__(
        self,
        domain: "HilbertSpace",
        normal_vector: "Vector",
        offset: float,
    ) -> None:
        """
        Initializes a hyperplane.

        Args:
            domain: The Hilbert space containing this hyperplane.
            normal_vector: The normal vector 'a' defining the hyperplane.
                Must be non-zero.
            offset: The scalar offset 'b'. The hyperplane is {x | ⟨a, x⟩ = b}.

        Raises:
            ValueError: If normal_vector is zero or offset is not a scalar.
        """
        super().__init__(domain)

        # Validate normal vector is non-zero
        a_norm = self.domain.norm(normal_vector)
        if a_norm < 1e-14:
            raise ValueError(
                "Normal vector must be non-zero for a hyperplane."
            )

        # Store normal vector and offset
        self._normal_vector = normal_vector
        self._offset = float(offset)

    @property
    def normal_vector(self) -> "Vector":
        """The normal vector defining the hyperplane."""
        return self._normal_vector

    @property
    def offset(self) -> float:
        """The offset scalar b in the definition ⟨a, x⟩ = b."""
        return self._offset

    @property
    def normal_norm(self) -> float:
        """The Euclidean norm of the normal vector."""
        return self.domain.norm(self._normal_vector)

    def is_element(self, x: "Vector", /, *, rtol: float = 1e-6) -> bool:
        """
        Returns True if x lies on the hyperplane.

        Checks if ⟨a, x⟩ ≈ b within relative tolerance.

        Args:
            x: A vector from the domain.
            rtol: Relative tolerance for the equality check.

        Returns:
            bool: True if |⟨a, x⟩ - b| ≤ rtol * max(1, |b|).
        """
        inner_product = self.domain.inner_product(self._normal_vector, x)
        tolerance = rtol * max(1.0, abs(self._offset))
        return abs(inner_product - self._offset) <= tolerance

    def distance_to(self, x: "Vector") -> float:
        """
        Computes the perpendicular distance from x to the hyperplane.

        Distance = |⟨a, x⟩ - b| / ||a||

        Args:
            x: A vector from the domain.

        Returns:
            float: The perpendicular distance from x to this hyperplane.
        """
        inner_product = self.domain.inner_product(self._normal_vector, x)
        a_norm = self.normal_norm
        return abs(inner_product - self._offset) / a_norm

    def project(self, x: "Vector") -> "Vector":
        """
        Projects a point onto the hyperplane.

        The projection is: x_proj = x - ((⟨a, x⟩ - b) / ||a||²) * a

        Args:
            x: A vector from the domain.

        Returns:
            Vector: The orthogonal projection of x onto this hyperplane.
        """
        a_norm_sq = self.domain.inner_product(self._normal_vector, self._normal_vector)
        inner_product = self.domain.inner_product(self._normal_vector, x)

        # Compute the scalar projection coefficient
        coeff = (inner_product - self._offset) / a_norm_sq

        # Subtract coeff * a from x
        scaled_a = self.domain.multiply(coeff, self._normal_vector)
        x_proj = self.domain.subtract(x, scaled_a)

        return x_proj

    @property
    def boundary(self) -> Subset:
        """
        Returns the boundary of the hyperplane (itself).
        A hyperplane is 'thin' so its boundary is itself.
        """
        return self

    def dimension(self) -> int:
        """
        Returns the geometric dimension of the hyperplane: dim(domain) - 1.

        A hyperplane has codimension 1.

        Returns:
            int: The dimension of this hyperplane subspace.
        """
        # Note: We don't store domain dimension explicitly; assume it's available
        # This is a placeholder; actual domain dimension may vary
        # For a proper implementation, HilbertSpace should provide dim() method
        raise NotImplementedError(
            "Dimension requires domain.dimension() method to be available."
        )


class HalfSpace(Subset):
    """
    Represents a half-space in a Hilbert space: H_+ = {x | ⟨a, x⟩ ≤ b}.

    A half-space is an unbounded convex set defined by a linear inequality.
    It is the epigraph or hypograph of a linear functional, depending on the
    orientation of the normal vector.
    """

    def __init__(
        self,
        domain: "HilbertSpace",
        normal_vector: "Vector",
        offset: float,
        inequality_type: str = "<=",
    ) -> None:
        """
        Initializes a half-space.

        Args:
            domain: The Hilbert space containing this half-space.
            normal_vector: The normal vector 'a' defining the half-space.
                Must be non-zero.
            offset: The scalar offset 'b'.
            inequality_type: Either "<=" for {x | ⟨a, x⟩ ≤ b} (default)
                or ">=" for {x | ⟨a, x⟩ ≥ b}.

        Raises:
            ValueError: If normal_vector is zero, offset is not scalar,
                or inequality_type is invalid.
        """
        super().__init__(domain)

        # Validate normal vector is non-zero
        a_norm = self.domain.norm(normal_vector)
        if a_norm < 1e-14:
            raise ValueError(
                "Normal vector must be non-zero for a half-space."
            )

        # Validate inequality type
        if inequality_type not in ("<=", ">="):
            raise ValueError(
                f"inequality_type must be '<=' or '>=', got '{inequality_type}'."
            )

        # Store parameters
        self._normal_vector = normal_vector
        self._offset = float(offset)
        self._inequality_type = inequality_type

    @property
    def normal_vector(self) -> "Vector":
        """The normal vector defining the half-space."""
        return self._normal_vector

    @property
    def offset(self) -> float:
        """The offset scalar b in the definition ⟨a, x⟩ ≤ b or ⟨a, x⟩ ≥ b."""
        return self._offset

    @property
    def inequality_type(self) -> str:
        """The inequality type: '<=' or '>='."""
        return self._inequality_type

    @property
    def normal_norm(self) -> float:
        """The Euclidean norm of the normal vector."""
        return self.domain.norm(self._normal_vector)

    def is_element(self, x: "Vector", /, *, rtol: float = 1e-6) -> bool:
        """
        Returns True if x lies within the half-space.

        For '<=' type: checks if ⟨a, x⟩ ≤ b + rtol * max(1, |b|).
        For '>=' type: checks if ⟨a, x⟩ ≥ b - rtol * max(1, |b|).

        Args:
            x: A vector from the domain.
            rtol: Relative tolerance for the inequality check.

        Returns:
            bool: True if x satisfies the half-space inequality.
        """
        inner_product = self.domain.inner_product(self._normal_vector, x)
        tolerance = rtol * max(1.0, abs(self._offset))

        if self._inequality_type == "<=":
            return inner_product <= self._offset + tolerance
        else:  # ">="
            return inner_product >= self._offset - tolerance

    def distance_to(self, x: "Vector") -> float:
        """
        Computes the perpendicular distance from x to the half-space boundary.

        For a point inside the half-space, returns the distance to the boundary
        plane ⟨a, x⟩ = b. For points outside, distance is negative (by convention).

        Distance = (⟨a, x⟩ - b) / ||a|| (unsigned distance to boundary plane)

        Args:
            x: A vector from the domain.

        Returns:
            float: The signed distance to the boundary plane:
                - Positive if x is on the "outside" (violating constraint)
                - Negative if x is inside the half-space
                - Zero if x is on the boundary
        """
        inner_product = self.domain.inner_product(self._normal_vector, x)
        a_norm = self.normal_norm

        if self._inequality_type == "<=":
            # Distance = (⟨a, x⟩ - b) / ||a||
            return (inner_product - self._offset) / a_norm
        else:  # ">="
            # Distance = (b - ⟨a, x⟩) / ||a|| (flipped sign)
            return (self._offset - inner_product) / a_norm

    def project(self, x: "Vector") -> "Vector":
        """
        Projects a point onto the half-space boundary.

        This is the orthogonal projection onto the boundary hyperplane
        {z | ⟨a, z⟩ = b}, which is independent of inequality type.

        Projection: x_proj = x - ((⟨a, x⟩ - b) / ||a||²) * a

        Args:
            x: A vector from the domain.

        Returns:
            Vector: The orthogonal projection of x onto the boundary plane.
        """
        a_norm_sq = self.domain.inner_product(self._normal_vector, self._normal_vector)
        inner_product = self.domain.inner_product(self._normal_vector, x)

        # Compute the scalar projection coefficient
        coeff = (inner_product - self._offset) / a_norm_sq

        # Subtract coeff * a from x
        scaled_a = self.domain.multiply(coeff, self._normal_vector)
        x_proj = self.domain.subtract(x, scaled_a)

        return x_proj

    @property
    def boundary(self) -> Subset:
        """
        Returns the boundary of the half-space.

        The boundary of {x | ⟨a, x⟩ ≤ b} is the hyperplane {x | ⟨a, x⟩ = b}.

        Returns:
            HyperPlane: The boundary hyperplane.
        """
        return HyperPlane(self.domain, self._normal_vector, self._offset)

    def is_bounded(self) -> bool:
        """
        Returns False, as half-spaces are unbounded convex sets.

        A half-space extends to infinity in at least one direction.

        Returns:
            bool: Always False for half-spaces.
        """
        return False

    @property
    def is_empty(self) -> bool:
        """
        Returns False. A half-space is never empty.

        Mathematically, any half-space {x | ⟨a, x⟩ ≤ b} or {x | ⟨a, x⟩ ≥ b}
        is non-empty in a Hilbert space.

        Returns:
            bool: Always False.
        """
        return False

    @property
    def support_function(self) -> "HalfSpaceSupportFunction":
        """
        Returns the support function for this half-space.

        Note:
            Half-spaces are UNBOUNDED, so their support functions are infinite
            in SOME directions (depends on query q and inequality type).
            The support function may raise ValueError when unbounded.
            This property is provided for completeness and theoretical
            consistency with other convex sets.

            Half-spaces are more useful as constraints in optimization (via
            their indicator functions) rather than through support functions.

        Returns:
            HalfSpaceSupportFunction: Support function (may raise
                ValueError when called with unbounded directions).
        """
        if not hasattr(self, "_support_fn") or self._support_fn is None:
            from pygeoinf.convex_analysis import HalfSpaceSupportFunction

            self._support_fn = HalfSpaceSupportFunction(
                self.domain,
                self._normal_vector,
                self._offset,
                self._inequality_type,
            )
        return self._support_fn


class PolyhedralSet(Subset):
    """
    Represents a polyhedral set as the intersection of half-spaces.

    P = {x | ⟨a_i, x⟩ ≤ b_i for all i} ∩ {x | ⟨a_j, x⟩ ≥ b_j for all j}

    A polyhedral set is a closed, bounded or unbounded convex set defined
    as the intersection of finitely many half-spaces.
    """

    def __init__(
        self,
        domain: "HilbertSpace",
        half_spaces: list["HalfSpace"],
    ) -> None:
        """
        Initialize a polyhedral set as intersection of half-spaces.

        Args:
            domain: The Hilbert space containing this polyhedral set.
            half_spaces: A list of HalfSpace objects. The polyhedral set
                is the intersection of all these half-spaces.

        Raises:
            ValueError: If half_spaces is empty or domains don't match.
        """
        super().__init__(domain)

        if not half_spaces:
            raise ValueError("PolyhedralSet requires at least one half-space.")

        # Validate that all half-spaces have the same domain
        for hs in half_spaces:
            if hs.domain is not domain:
                raise ValueError(
                    "All half-spaces must have the same domain as the polyhedral set."
                )

        self._half_spaces = list(half_spaces)

    @property
    def half_spaces(self) -> list["HalfSpace"]:
        """Returns the list of half-spaces defining this polyhedral set."""
        return list(self._half_spaces)

    def is_element(self, x: "Vector", /, *, rtol: float = 1e-6) -> bool:
        """
        Check if x belongs to the polyhedral set.

        x ∈ P iff x satisfies all half-space constraints.

        Args:
            x: A vector from the domain.
            rtol: Relative tolerance for constraint checks.

        Returns:
            bool: True if x satisfies all half-space constraints.
        """
        for hs in self._half_spaces:
            if not hs.is_element(x, rtol=rtol):
                return False
        return True

    def boundary(self) -> "Subset":
        """
        Returns the boundary of the polyhedral set.

        The boundary consists of faces where one or more constraints are
        active (⟨a_i, x⟩ = b_i). Computing the complete boundary is complex
        in general Hilbert spaces, so this raises NotImplementedError.

        Raises:
            NotImplementedError: General polyhedral boundary computation
                is not yet implemented for arbitrary Hilbert spaces.
        """
        raise NotImplementedError(
            "Boundary computation for general polyhedral sets is not yet "
            "implemented. Consider specific cases (e.g., polytopes in ℝⁿ) "
            "for practical applications."
        )

    def is_bounded(self) -> bool:
        """
        Check if the polyhedral set is bounded.

        A polyhedral set is bounded iff it's a polytope (all variables
        have finite bounds). This check is non-trivial in general Hilbert
        spaces and is not yet implemented.

        Returns:
            bool: Raises NotImplementedError.

        Raises:
            NotImplementedError: Boundedness detection for general polyhedral
                sets is not yet implemented.
        """
        raise NotImplementedError(
            "Boundedness check for general polyhedral sets is not yet "
            "implemented. This would require LP feasibility analysis."
        )

    @property
    def is_empty(self) -> bool:
        """
        Check if the polyhedral set is empty.

        A polyhedral set is empty iff the intersection of half-spaces is
        empty. This is a feasibility problem and requires LP techniques
        to determine precisely.

        Returns:
            bool: Raises NotImplementedError.

        Raises:
            NotImplementedError: Emptiness checking for general polyhedral
                sets requires LP feasibility testing, not yet implemented.
        """
        raise NotImplementedError(
            "Emptiness check for general polyhedral sets is not yet "
            "implemented. This would require LP feasibility analysis."
        )

    @property
    def support_function(self) -> "Optional[SupportFunction]":
        """
        Returns the support function of the polyhedral set.

        For a polyhedral set P = ∩_i H_i (intersection of half-spaces),
        the support function is:
            σ_P(q) = inf_{i} σ_{H_i}(q)

        However, evaluating the infimum of support functions is complex
        and may require iterative methods (e.g., linear programming).

        Returns:
            None: Support function for general polyhedral sets is not yet
                implemented. Consider using the half-space constraints
                directly in optimization or implementing LP-based evaluation.

        Note:
            For specific cases (e.g., bounded polytopes or simplices),
            specialized implementations are available in other modules.
        """
        return None
