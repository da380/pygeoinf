"""Subsets of a Hilbert space: predicates, convex sets, and subspaces."""

from .convex import Ball, ConvexSet, Ellipsoid, HalfSpace, Hyperplane
from .sets import Complement, EmptySet, Intersection, Subset, Union, UniversalSet
from .subspaces import AffineSubspace, LinearSubspace, OrthogonalProjector

__all__ = [
    "AffineSubspace",
    "Ball",
    "Complement",
    "ConvexSet",
    "Ellipsoid",
    "EmptySet",
    "HalfSpace",
    "Hyperplane",
    "Intersection",
    "LinearSubspace",
    "OrthogonalProjector",
    "Subset",
    "Union",
    "UniversalSet",
]
