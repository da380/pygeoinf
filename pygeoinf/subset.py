"""
Defines classes for representing subsets of a Hilbert space.

This module provides a hierarchy of classes for sets, ranging from abstract
definitions to concrete geometric shapes. It utilizes scalar functionals
(`NonLinearForm`) to define general level sets and convex subsets.

The class structure uses Mixins to share geometric logic across topological types:

1. _EllipsoidalGeometry (Mixin)
    - Mixed into Ellipsoid (ConvexSubset) -> Ball, NormalisedEllipsoid
    - Mixed into EllipsoidSurface (LevelSet) -> Sphere

2. _LinearGeometry (Mixin)
    - Mixed into HalfSpace (ConvexSubset)
    - Mixed into Hyperplane (LevelSet)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional
import numpy as np

from .nonlinear_forms import NonLinearForm
from .linear_forms import LinearForm

if TYPE_CHECKING:
    from .hilbert_space import HilbertSpace, Vector
    from .linear_operators import LinearOperator


class Subset(ABC):
    """
    Abstract base class for a subset of a HilbertSpace.

    This class defines the minimal interface required for a mathematical set:
    knowing which space it lives in, determining if a vector belongs to it,
    accessing its boundary, and checking for emptiness.
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

        The default implementation returns False. Subclasses representing
        empty sets (like EmptySet) should override this to return True.
        """
        return False

    @abstractmethod
    def is_element(self, x: Vector, /, *, rtol: float = 1e-6) -> bool:
        """
        Returns True if the vector x lies within the subset.

        Args:
            x: A vector from the domain.
            rtol: Relative tolerance for floating-point comparisons.
        """

    @property
    @abstractmethod
    def boundary(self) -> Subset:
        """
        Returns the boundary of the subset as a new Subset instance.

        Notes:
            Implementations should ensure that the boundary of a boundary
            is the EmptySet (∂∂S = ∅).
        """


class EmptySet(Subset):
    """
    Represents the empty set (∅).

    This set contains no elements. It can be initialized without a domain.
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
        """
        Returns the boundary of the empty set.

        The boundary of an empty set is the empty set itself.
        """
        return self


class LevelSet(Subset):
    """
    Represents a level set of a functional: S = {x | f(x) = c}.

    This is generally a manifold (potentially with singularities) and
    forms the boundary of a SublevelSet.
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
        # Infer domain directly from the form
        super().__init__(form.domain)

        self._form = form
        self._level = level

    @property
    def form(self) -> NonLinearForm:
        """The defining functional."""
        return self._form

    @property
    def level(self) -> float:
        """The level value."""
        return self._level

    def is_element(self, x: Vector, /, *, rtol: float = 1e-6) -> bool:
        """
        Returns True if f(x) is approximately equal to the level.

        The tolerance is scaled by the absolute value of the level (or 1.0).
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


class ConvexSubset(Subset):
    """
    Represents a convex set defined as a sublevel set: S = {x | f(x) <= c}.

    This class assumes the defining form 'f' is convex. It includes a
    randomized check to verify this property locally.
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
        # Infer domain directly from the form
        super().__init__(form.domain)

        self._form = form
        self._level = level
        self._open = open_set

    @property
    def form(self) -> NonLinearForm:
        """The defining functional."""
        return self._form

    @property
    def level(self) -> float:
        """The level value."""
        return self._level

    @property
    def is_open(self) -> bool:
        """True if the set is defined by strict inequality."""
        return self._open

    def is_element(self, x: Vector, /, *, rtol: float = 1e-6) -> bool:
        """
        Returns True if f(x) <= c (or f(x) < c if open).
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
        Returns the boundary of the convex set.

        The boundary is typically the LevelSet {x | f(x) = c}.
        """
        return LevelSet(self._form, self._level)

    def check(
        self, n_samples: int = 10, /, *, rtol: float = 1e-5, atol: float = 1e-8
    ) -> None:
        """
        Performs a randomized check of the convexity inequality:
        f(tx + (1-t)y) <= t*f(x) + (1-t)*f(y)

        Args:
            n_samples: Number of random pairs to test.
            rtol: Relative tolerance for the inequality check.
            atol: Absolute tolerance for the inequality check.

        Raises:
            AssertionError: If the function is found to be non-convex.
        """
        for _ in range(n_samples):
            x = self.domain.random()
            y = self.domain.random()
            t = np.random.uniform(0, 1)

            # Convex combination point: z = t*x + (1-t)*y
            tx = self.domain.multiply(t, x)
            ty = self.domain.multiply(1.0 - t, y)
            z = self.domain.add(tx, ty)

            # Evaluate form
            fz = self._form(z)
            fx = self._form(x)
            fy = self._form(y)

            # Convexity condition: f(z) <= t*f(x) + (1-t)*f(y)
            rhs = t * fx + (1.0 - t) * fy

            # Fail if lhs is significantly greater than rhs
            if fz > rhs + atol + rtol * abs(rhs):
                raise AssertionError(
                    f"Convexity check failed.\n"
                    f"t={t:.2f}\n"
                    f"LHS: f(tx + (1-t)y) = {fz:.4e}\n"
                    f"RHS: t*f(x) + (1-t)*f(y) = {rhs:.4e}\n"
                    f"Diff: {fz - rhs:.4e} (Tol: {atol + rtol * abs(rhs):.4e})\n"
                    "The defining functional does not appear to be convex."
                )
        print(f"[✓] Convexity check passed ({n_samples} samples).")


class _EllipsoidalGeometry:
    """
    Mixin class that holds the common data and logic for ellipsoidal sets.

    This class is responsible for:
    1. Storing center, radius, and operator.
    2. Constructing the quadratic NonLinearForm: f(x) = <A(x-c), x-c>.

    It is intended to be mixed into classes that also inherit from Subset (or its subclasses).
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
        if operator.domain != domain or operator.codomain != domain:
            raise ValueError("Operator must map the domain to itself.")

        self._center = center
        self._radius = radius
        self._operator = operator

        # --- Construct the form ---

        # 1. Define the mapping: f(x) = <A(x-c), x-c>
        def mapping(x: Vector) -> float:
            diff = domain.subtract(x, center)
            return domain.inner_product(operator(diff), diff)

        # 2. Define the gradient: f'(x) = 2 A (x-c)
        def gradient(x: Vector) -> Vector:
            diff = domain.subtract(x, center)
            return domain.multiply(2.0, operator(diff))

        # 3. Define the Hessian: f''(x) = 2 A
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
        # Initialize the geometry (creates the form)
        _EllipsoidalGeometry.__init__(self, domain, center, radius, operator)

        # Initialize the ConvexSubset using the form from the geometry
        ConvexSubset.__init__(
            self, self._generated_form, self._generated_level, open_set=open_set
        )

    @property
    def boundary(self) -> Subset:
        """
        Returns the boundary of the ellipsoid.

        Returns an EllipsoidSurface object.
        """
        return EllipsoidSurface(self.domain, self.center, self.radius, self.operator)

    @property
    def normalized(self) -> NormalisedEllipsoid:
        """
        Returns a normalized version of this ellipsoid with radius 1.

        The operator is scaled by 1/r^2 to maintain the same set.
        """
        if self.radius == 0:
            raise ValueError("Cannot normalize an ellipsoid with zero radius.")

        # Determine the scaled operator A' = A / r^2
        scale = 1.0 / (self.radius**2)
        scaled_operator = scale * self.operator

        return NormalisedEllipsoid(
            self.domain, self.center, scaled_operator, open_set=self.is_open
        )


class NormalisedEllipsoid(Ellipsoid):
    """
    Represents a normalised ellipsoid with radius 1: E = {x | <A'(x-c), x-c> <= 1}.
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
        # Initialize directly with radius 1.0
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
        # Initialize the geometry (creates the form)
        _EllipsoidalGeometry.__init__(self, domain, center, radius, operator)

        # Initialize the LevelSet using the form from the geometry
        LevelSet.__init__(self, self._generated_form, self._generated_level)

    @property
    def boundary(self) -> Subset:
        """
        Returns the boundary of the ellipsoid surface.

        As a closed manifold without boundary, its boundary is the EmptySet.
        """
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

    Implemented as a specialization of Ellipsoid where the metric
    operator is the Identity.
    """

    def __init__(
        self,
        domain: HilbertSpace,
        center: Vector,
        radius: float,
        open_set: bool = True,
    ) -> None:
        # A Ball is an Ellipsoid with A = Identity
        identity = domain.identity_operator()
        super().__init__(domain, center, radius, identity, open_set=open_set)

    def is_element(self, x: Vector, /, *, rtol: float = 1e-6) -> bool:
        """
        Returns True if x lies within the ball.

        Overrides Ellipsoid.is_element to use geometric distance for tolerance.
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
        """
        Returns the boundary of the ball.

        Overrides Ellipsoid.boundary to return a concrete Sphere object.
        """
        return Sphere(self.domain, self.center, self.radius)


class Sphere(EllipsoidSurface):
    """
    Represents a sphere in a Hilbert space: S = {x | ||x - c||^2 = r^2}.

    Implemented as a specialization of EllipsoidSurface where the metric
    operator is the Identity.
    """

    def __init__(self, domain: HilbertSpace, center: Vector, radius: float) -> None:
        # A Sphere is an EllipsoidSurface with A = Identity
        identity = domain.identity_operator()
        super().__init__(domain, center, radius, identity)

    def is_element(self, x: Vector, /, *, rtol: float = 1e-6) -> bool:
        """
        Returns True if ||x - c|| is approximately equal to r.

        Overrides EllipsoidSurface.is_element to use the geometric distance
        for the tolerance check, rather than the squared energy norm, which
        is more intuitive for spheres.
        """
        diff = self.domain.subtract(x, self.center)
        dist = self.domain.norm(diff)
        return abs(dist - self.radius) <= rtol * max(1.0, self.radius)


class _LinearGeometry:
    """
    Mixin class that holds the common data for sets defined by a linear form.

    This class is responsible for storing and validating the linear form and level.
    It is intended to be mixed into classes that also inherit from Subset.
    """

    def __init__(self, form: LinearForm, level: float) -> None:
        """
        Args:
            form: The defining linear functional l(x).
            level: The scalar value c.
        """
        if not isinstance(form, LinearForm):
            raise TypeError("Geometry requires a LinearForm.")

        self._generated_form = form
        self._generated_level = level

    @property
    def linear_form(self) -> LinearForm:
        """Returns the defining linear form."""
        return self._generated_form


class Hyperplane(LevelSet, _LinearGeometry):
    """
    Represents a hyperplane defined by a linear functional: H = {x | l(x) = c}.

    This is a linear manifold (affine subspace of codimension 1).
    """

    def __init__(self, form: LinearForm, level: float) -> None:
        # Initialize geometry
        _LinearGeometry.__init__(self, form, level)
        # Initialize LevelSet
        LevelSet.__init__(self, self._generated_form, self._generated_level)

    @classmethod
    def from_vector(
        cls, domain: HilbertSpace, normal: Vector, level: float
    ) -> Hyperplane:
        """
        Creates a hyperplane defined by a normal vector: {x | <n, x> = c}.

        Args:
            domain: The Hilbert space.
            normal: The normal vector n.
            level: The scalar constant c.
        """
        form = domain.to_dual(normal)
        return cls(form, level)

    @property
    def boundary(self) -> Subset:
        """
        Returns the boundary of the hyperplane.

        As a closed manifold without boundary, its boundary is the EmptySet.
        """
        return EmptySet(self.domain)


class HalfSpace(ConvexSubset, _LinearGeometry):
    """
    Represents a half-space defined by a linear inequality: H = {x | l(x) <= c}.
    """

    def __init__(
        self,
        form: LinearForm,
        level: float,
        open_set: bool = False,
    ) -> None:
        # Initialize geometry
        _LinearGeometry.__init__(self, form, level)
        # Initialize ConvexSubset
        ConvexSubset.__init__(
            self, self._generated_form, self._generated_level, open_set=open_set
        )

    @classmethod
    def from_vector(
        cls,
        domain: HilbertSpace,
        normal: Vector,
        level: float,
        open_set: bool = False,
    ) -> HalfSpace:
        """
        Creates a half-space defined by a normal vector: {x | <n, x> <= c}.

        Args:
            domain: The Hilbert space.
            normal: The normal vector n.
            level: The scalar constant c.
            open_set: If True, uses strict inequality.
        """
        form = domain.to_dual(normal)
        return cls(form, level, open_set=open_set)

    @property
    def boundary(self) -> Subset:
        """
        Returns the boundary of the half-space.

        Returns a Hyperplane object.
        """
        return Hyperplane(self.linear_form, self.level)
