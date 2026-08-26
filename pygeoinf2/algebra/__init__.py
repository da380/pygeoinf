"""The algebraic core: spaces and operators."""

from .diagonal import DiagonalLinearOperator
from .direct_sum import (
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
from .linearisation import Linearisation, QuadraticModel
from .operators import (
    AffineOperator,
    Functional,
    LinearFunctional,
    LinearOperator,
    Operator,
    require_coordinates,
)
from .spaces import (
    ArrayVectorMixin,
    CoordinateSpace,
    DiagonalMetricSpace,
    EuclideanSpace,
    HilbertSpace,
    OrthonormalSpace,
    Reals,
)

__all__ = [
    "AffineOperator",
    "BlockDiagonalLinearOperator",
    "DiagonalLinearOperator",
    "BlockDiagonalOperator",
    "BlockLinearOperator",
    "BlockOperator",
    "ColumnLinearOperator",
    "ColumnOperator",
    "DirectSum",
    "RowLinearOperator",
    "RowOperator",
    "ArrayVectorMixin",
    "CoordinateSpace",
    "DiagonalMetricSpace",
    "EuclideanSpace",
    "Functional",
    "HilbertSpace",
    "Linearisation",
    "LinearFunctional",
    "LinearOperator",
    "Operator",
    "OrthonormalSpace",
    "QuadraticModel",
    "Reals",
    "require_coordinates",
]
