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
    MatrixLinearOperator,
    Operator,
    require_coordinates,
)
from .spaces import (
    ArrayVectorMixin,
    CoordinateSpace,
    DiagonalMetricSpace,
    EuclideanSpace,
    HilbertModule,
    HilbertSpace,
    MassWeightedSpace,
    OrthonormalSpace,
    Reals,
    require_module,
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
    "HilbertModule",
    "HilbertSpace",
    "Linearisation",
    "LinearFunctional",
    "LinearOperator",
    "MassWeightedSpace",
    "MatrixLinearOperator",
    "Operator",
    "OrthonormalSpace",
    "QuadraticModel",
    "Reals",
    "require_coordinates",
    "require_module",
]
