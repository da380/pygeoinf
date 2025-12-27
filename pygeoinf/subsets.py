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
    Represents a convex set defined as a sublevel set: S = {x | f(x) <= c}.

    This class assumes the defining form 'f' is convex. It includes tools
    to verify this property locally.
    """

    def __init__(
        self,
        form: NonLinearForm,
        level: float,
        open_set: bool = False,
    ) -> None:
        """
        Args:
            form: The defining functional f(x). Must be convex.
            level: The scalar upper bound c.
            open_set: If True, uses strict inequality (<).
        """
        super().__init__(form, level, open_set=open_set)

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

    @property
    def boundary(self) -> Subset:
        """Returns the boundary EllipsoidSurface."""
        return EllipsoidSurface(self.domain, self.center, self.radius, self.operator)

    @property
    def normalized(self) -> NormalisedEllipsoid:
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
