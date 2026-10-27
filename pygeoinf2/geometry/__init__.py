"""Subsets of a Hilbert space: predicates, convex sets, and subspaces."""

from .convex import (
    ConvexIntersection,
    Polytope,
    BallSurface,
    EllipsoidSurface,
    Ball,
    ConvexSet,
    Ellipsoid,
    HalfSpace,
    Hyperplane,
)
from .feasible import (
    Algorithm,
    ClosedForm,
    FeasiblePropertySet,
    InclusionResult,
    SupportResult,
    SweepResult,
)
from .sets import (
    Complement,
    EmptySet,
    Intersection,
    LevelSet,
    SublevelSet,
    Subset,
    Union,
    UniversalSet,
)
from .subspaces import AffineSubspace, LinearSubspace, OrthogonalProjector

__all__ = [
    "AffineSubspace",
    "Algorithm",
    "ClosedForm",
    "FeasiblePropertySet",
    "InclusionResult",
    "SupportResult",
    "SweepResult",
    "Ball",
    "BallSurface",
    "Complement",
    "ConvexSet",
    "ConvexIntersection",
    "Ellipsoid",
    "EllipsoidSurface",
    "EmptySet",
    "HalfSpace",
    "Hyperplane",
    "Intersection",
    "LevelSet",
    "LinearSubspace",
    "OrthogonalProjector",
    "Polytope",
    "SublevelSet",
    "Subset",
    "Union",
    "UniversalSet",
]
