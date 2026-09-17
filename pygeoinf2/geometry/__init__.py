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
