"""
pygeoinf 2.0 — development tree.

This is the shadow package described in ``DESIGN.md``. It is not shipped; the
released library remains ``pygeoinf``. Imports are relative throughout so that
the eventual promotion is a directory rename.

Both may be imported side by side for comparison testing:

    import pygeoinf as gi
    import pygeoinf2 as gi2
"""

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
    # algebra.linearisation
    "Linearisation",
    "QuadraticModel",
    # traits
    "Traits",
]
