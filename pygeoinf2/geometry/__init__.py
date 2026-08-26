"""Subsets of a Hilbert space: predicates, convex sets, and subspaces."""

from .convex import (
    Polytope,
    BallSurface,
    EllipsoidSurface,
    Ball,
    ConvexSet,
    Ellipsoid,
    HalfSpace,
    Hyperplane,
)
from .sets import Complement, EmptySet, Intersection, Subset, Union, UniversalSet
from .subspaces import AffineSubspace, LinearSubspace, OrthogonalProjector

__all__ = [
    "AffineSubspace",
    "Ball",
    "BallSurface",
    "Complement",
    "ConvexSet",
    "Ellipsoid",
    "EllipsoidSurface",
    "EmptySet",
    "HalfSpace",
    "Hyperplane",
    "Intersection",
    "LinearSubspace",
    "OrthogonalProjector",
    "Polytope",
    "Subset",
    "Union",
    "UniversalSet",
]
