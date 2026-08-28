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
    MatrixLinearOperator,
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
    BallSurface,
    ConvexSet,
    Ellipsoid,
    EllipsoidSurface,
    EmptySet,
    HalfSpace,
    Hyperplane,
    LinearSubspace,
    OrthogonalProjector,
    Polytope,
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
from .probability.mixture import GaussianMixture
from .traits import Traits

# The subpackages, so that ``gi.inference.X`` works. inference and plotting
# were the two a user needs in order to *solve* anything, and the two that were
# not reachable at all; numerics arrived only by accident, pulled in
# transitively by symmetric_space. See DESIGN.md section 25.4.
from . import geometry, inference, numerics, plotting, probability, symmetric_space
from .inference import (
    BackusGilbert,
    BackusInference,
    ConstrainedLeastSquares,
    ConstrainedMinimumNorm,
    DiscrepancyPrinciple,
    FeasibleProperty,
    ForwardProblem,
    LeastSquares,
    LinearForwardProblem,
    LaplaceResult,
    LinearGaussianInversion,
    MaximumAPosteriori,
    LinearGaussianMixtureInversion,
    LocalisedPreconditioner,
    MinimumNorm,
    NormalDiagonalPreconditioner,
    NormalOperator,
    TikhonovFamily,
    TikhonovNormalOperator,
)
from .numerics.preconditioners import (
    BandedPreconditioner,
    IdentityPreconditioner,
    JacobiPreconditioner,
    SpectralPreconditioner,
    WoodburyPreconditioner,
)
from .numerics.solvers import (
    BiCGStabSolver,
    CGSolver,
    CholeskySolver,
    EigenSolver,
    FlexibleCGSolver,
    GMRESSolver,
    LSQRSolver,
    LUSolver,
    MinResSolver,
    ProgressCallback,
    SolveResult,
)
from .plotting import plot, plot_corner, plot_densities, plot_points, subplots

__all__ = [
    # subpackages
    "geometry",
    "inference",
    "numerics",
    "plotting",
    "probability",
    "symmetric_space",
    # inference: the workflow, flat, as v1 had it
    "BackusGilbert",
    "BackusInference",
    "ConstrainedLeastSquares",
    "ConstrainedMinimumNorm",
    "DiscrepancyPrinciple",
    "FeasibleProperty",
    "ForwardProblem",
    "LeastSquares",
    "LinearForwardProblem",
    "LaplaceResult",
    "LinearGaussianInversion",
    "MaximumAPosteriori",
    "LinearGaussianMixtureInversion",
    "MinimumNorm",
    "NormalOperator",
    "TikhonovFamily",
    "TikhonovNormalOperator",
    # solvers and preconditioners
    "BandedPreconditioner",
    "BiCGStabSolver",
    "CGSolver",
    "CholeskySolver",
    "EigenSolver",
    "FlexibleCGSolver",
    "GMRESSolver",
    "IdentityPreconditioner",
    "JacobiPreconditioner",
    "LocalisedPreconditioner",
    "LSQRSolver",
    "LUSolver",
    "MinResSolver",
    "NormalDiagonalPreconditioner",
    "ProgressCallback",
    "SolveResult",
    "SpectralPreconditioner",
    "WoodburyPreconditioner",
    # plotting
    "plot",
    "plot_corner",
    "plot_densities",
    "plot_points",
    "subplots",
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
    "MatrixLinearOperator",
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
    "BallSurface",
    "ConvexSet",
    "Ellipsoid",
    "EllipsoidSurface",
    "EmptySet",
    "HalfSpace",
    "Hyperplane",
    "LinearSubspace",
    "OrthogonalProjector",
    "Polytope",
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
    "GaussianMixture",
    "ProbabilityMeasure",
    "ProductMeasure",
    "PushForwardMeasure",
    "product",
    # traits
    "Traits",
]
