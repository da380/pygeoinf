"""
pygeoinf 2.0: inference on Hilbert spaces.

Vectors are whatever objects a backend provides, and every algorithm works
through the space's inner product, so the same estimator runs on a Euclidean
space, a spherical-harmonic field or a finite element space. The subpackages:

* ``algebra`` -- Hilbert spaces, direct sums, linear and nonlinear operators;
* ``numerics`` -- linear solvers, preconditioners, randomized methods, the
  operator calculus, optimization and convex solvers;
* ``geometry`` -- convex sets, subsets and subspaces;
* ``probability`` -- measures, Gaussian and otherwise;
* ``inference`` -- forward problems and the estimators built on them;
* ``symmetric_space`` and ``sem1d`` -- concrete function spaces;
* ``plotting`` -- figures for fields, measures and sets.

The names a typical workflow needs are also exported here, flat.
``compat``, ``backends``, ``datasets`` and ``testing`` are imported
separately.
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
from .algebra.linearization import Linearization, QuadraticModel
from .algebra.operators import (
    AffineOperator,
    EvaluationError,
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
    HilbertModule,
    HilbertSpace,
    MassWeightedSpace,
    OrthonormalSpace,
    Reals,
    require_module,
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
    SublevelSet,
    LevelSet,
    UniversalSet,
    enclosing_ellipsoid,
    sum_of_ellipsoids,
)
from .symmetric_space import Box, Interval, Lebesgue, PeriodicBox, Sobolev
from .probability.absolutely_continuous import AbsolutelyContinuousMeasure
from .probability.base import (
    ProbabilityMeasure,
    ProductMeasure,
    PushForwardMeasure,
    product,
)
from .probability.gaussian import GaussianMeasure
from .probability.mixture import GaussianMixture
from .traits import Traits

# The subpackages, so that ``gi.inference.X`` works without a separate import.
# compat, backends, datasets and testing are deliberately not pulled in.
from . import geometry, inference, numerics, plotting, probability, symmetric_space
from .inference import (
    BackusGilbert,
    BackusGilbertParker,
    BackusGilbertSet,
    FeasiblePropertySet,
    ConstrainedLeastSquares,
    ConstrainedMinimumNorm,
    DiscrepancyPrinciple,
    ForwardProblem,
    LeastSquares,
    LinearForwardProblem,
    LaplaceResult,
    GaussianPosterior,
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
    SolutionTrackingCallback,
    SolveResult,
    SolveStep,
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
    # inference: the workflow, flat
    "BackusGilbert",
    "BackusGilbertParker",
    "BackusGilbertSet",
    "FeasiblePropertySet",
    "ConstrainedLeastSquares",
    "ConstrainedMinimumNorm",
    "DiscrepancyPrinciple",
    "ForwardProblem",
    "LeastSquares",
    "LinearForwardProblem",
    "LaplaceResult",
    "GaussianPosterior",
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
    "SolutionTrackingCallback",
    "SolveResult",
    "SolveStep",
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
    "HilbertModule",
    "HilbertSpace",
    "MassWeightedSpace",
    "OrthonormalSpace",
    "Reals",
    "require_module",
    # algebra.operators
    "AffineOperator",
    "EvaluationError",
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
    # algebra.linearization
    "Linearization",
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
    "enclosing_ellipsoid",
    "sum_of_ellipsoids",
    "Subset",
    "SublevelSet",
    "LevelSet",
    "UniversalSet",
    # spaces
    "Box",
    "Interval",
    "Lebesgue",
    "PeriodicBox",
    "Sobolev",
    # probability
    "AbsolutelyContinuousMeasure",
    "GaussianMeasure",
    "GaussianMixture",
    "ProbabilityMeasure",
    "ProductMeasure",
    "PushForwardMeasure",
    "product",
    # traits
    "Traits",
]
