"""
pygeoinf 2.0 — development tree.

This is the shadow package described in ``DESIGN.md``. It is not shipped; the
released library remains ``pygeoinf``. Imports are relative throughout so that
the eventual promotion is a directory rename.

Both may be imported side by side for comparison testing:

    import pygeoinf as gi
    import pygeoinf2 as gi2
"""

from .algebra.diagonal import DiagonalLinearOperator
from .algebra.direct_sum import (
    BlockDiagonalLinearOperator,
    BlockDiagonalOperator,
    BlockLinearOperator,
    BlockOperator,
    ColumnLinearOperator,
    ColumnOperator,
    DirectSum,
    RowLinearOperator,
    RowOperator,
)
from .algebra.linearisation import Linearisation, QuadraticModel
from .algebra.operators import (
    AffineOperator,
    Functional,
    LinearFunctional,
    LinearOperator,
    Operator,
    require_coordinates,
)
from .algebra.spaces import (
    ArrayVectorMixin,
    CoordinateSpace,
    DiagonalMetricSpace,
    EuclideanSpace,
    HilbertSpace,
    OrthonormalSpace,
    Reals,
)
from .geometry import (
    AffineSubspace,
    Ball,
    ConvexSet,
    Ellipsoid,
    EmptySet,
    HalfSpace,
    Hyperplane,
    LinearSubspace,
    OrthogonalProjector,
    Subset,
    UniversalSet,
)
from .symmetric_space import Box, Interval, Lebesgue, PeriodicBox, Sobolev
from .probability.base import (
    ProbabilityMeasure,
    ProductMeasure,
    PushForwardMeasure,
    product,
)
from .probability.gaussian import GaussianMeasure
from .traits import Traits

__all__ = [
    # algebra.spaces
    "ArrayVectorMixin",
    "CoordinateSpace",
    "DiagonalMetricSpace",
    "EuclideanSpace",
    "HilbertSpace",
    "OrthonormalSpace",
    "Reals",
    # algebra.operators
    "AffineOperator",
    "Functional",
    "LinearFunctional",
    "LinearOperator",
    "Operator",
    "require_coordinates",
    # algebra.diagonal
    "DiagonalLinearOperator",
    # algebra.direct_sum
    "BlockDiagonalLinearOperator",
    "BlockDiagonalOperator",
    "BlockLinearOperator",
    "BlockOperator",
    "ColumnLinearOperator",
    "ColumnOperator",
    "DirectSum",
    "RowLinearOperator",
    "RowOperator",
    # algebra.linearisation
    "Linearisation",
    "QuadraticModel",
    # geometry
    "AffineSubspace",
    "Ball",
    "ConvexSet",
    "Ellipsoid",
    "EmptySet",
    "HalfSpace",
    "Hyperplane",
    "LinearSubspace",
    "OrthogonalProjector",
    "Subset",
    "UniversalSet",
    # spaces
    "Box",
    "Interval",
    "Lebesgue",
    "PeriodicBox",
    "Sobolev",
    # probability
    "GaussianMeasure",
    "ProbabilityMeasure",
    "ProductMeasure",
    "PushForwardMeasure",
    "product",
    # traits
    "Traits",
]
