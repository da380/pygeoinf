"""
Unified imports for the package.
"""

from .hilbert_space import (
    HilbertSpace,
    DualHilbertSpace,
    EuclideanSpace,
    HilbertModule,
    MassWeightedHilbertSpace,
    MassWeightedHilbertModule,
)


from .nonlinear_forms import (
    NonLinearForm,
)


from .linear_forms import (
    LinearForm,
)

from .nonlinear_operators import NonLinearOperator

from .linear_operators import (
    LinearOperator,
    MatrixLinearOperator,
    DenseMatrixLinearOperator,
    SparseMatrixLinearOperator,
    DiagonalSparseMatrixLinearOperator,
)

from .affine_operators import AffineOperator

from .gaussian_measure import (
    GaussianMeasure,
)

from .direct_sum import (
    HilbertSpaceDirectSum,
    BlockStructure,
    BlockLinearOperator,
    ColumnLinearOperator,
    RowLinearOperator,
    BlockDiagonalLinearOperator,
)

from .linear_solvers import (
    LinearSolver,
    DirectLinearSolver,
    LUSolver,
    CholeskySolver,
    EigenSolver,
    IterativeLinearSolver,
    ScipyIterativeSolver,
    CGMatrixSolver,
    BICGMatrixSolver,
    BICGStabMatrixSolver,
    GMRESMatrixSolver,
    CGSolver,
    MinResSolver,
    BICGStabSolver,
    FCGSolver,
    ProgressCallback,
    SolutionTrackingCallback,
    ResidualTrackingCallback,
)

from .preconditioners import (
    JacobiPreconditioningMethod,
    SpectralPreconditioningMethod,
    IdentityPreconditioningMethod,
    IterativePreconditioningMethod,
    BandedPreconditioningMethod,
    ExactBlockPreconditioningMethod,
    ColumnThresholdedPreconditioningMethod,
)

from .forward_problem import ForwardProblem, LinearForwardProblem

from .linear_optimisation import (
    LinearLeastSquaresInversion,
    LinearMinimumNormInversion,
    ConstrainedLinearLeastSquaresInversion,
    ConstrainedLinearMinimumNormInversion,
)

from .linear_bayesian import (
    LinearBayesianInversion,
)

from .nonlinear_optimisation import (
    ScipyUnconstrainedOptimiser,
)


from .subspaces import OrthogonalProjector, AffineSubspace, LinearSubspace

from .subsets import (
    Subset,
    EmptySet,
    UniversalSet,
    Complement,
    Intersection,
    Union,
    SublevelSet,
    LevelSet,
    ConvexSubset,
    Ellipsoid,
    NormalisedEllipsoid,
    EllipsoidSurface,
    Ball,
    Sphere,
)

from .plot import (
    plot_1d_distributions,
    plot_corner_distributions,
    SubspaceSlicePlotter,
    plot_slice,
)

from .convex_optimisation import (
    SubgradientDescent,
    ProximalBundleMethod,
    PrimalKKTSolver,
    KKTResult,
)

from .convex_analysis import (
    SupportFunction,
    BallSupportFunction,
    EllipsoidSupportFunction,
    HalfSpaceSupportFunction,
    CallableSupportFunction,
    PointSupportFunction,
    LinearImageSupportFunction,
    MinkowskiSumSupportFunction,
    ScaledSupportFunction,
)

from .backus_gilbert import DualMasterCostFunction

from .utils import configure_threading

from .datasets import (
    load_gsn_stations,
    download_usgs_earthquakes,
    download_gsn_stations,
    sample_earthquakes,
)

from .low_rank import LowRankSVD, LowRankCholesky, LowRankEig, white_noise_measure

from .config import DATADIR

__all__ = [
    # hilbert_space
    "HilbertSpace",
    "DualHilbertSpace",
    "EuclideanSpace",
    "HilbertModule",
    "MassWeightedHilbertSpace",
    "MassWeightedHilbertModule",
    # nonlinear_forms
    "NonLinearForm",
    # linear_forms
    "LinearForm",
    # nonlinear_operators
    "NonLinearOperator",
    # linear_operators
    "LinearOperator",
    "MatrixLinearOperator",
    "DenseMatrixLinearOperator",
    "SparseMatrixLinearOperator",
    "DiagonalSparseMatrixLinearOperator",
    # affine_operators
    "AffineOperator",
    # gaussian_measure
    "GaussianMeasure",
    # direct_sum
    "HilbertSpaceDirectSum",
    "BlockStructure",
    "BlockLinearOperator",
    "ColumnLinearOperator",
    "RowLinearOperator",
    "BlockDiagonalLinearOperator",
    # linear_solvers
    "LinearSolver",
    "DirectLinearSolver",
    "LUSolver",
    "CholeskySolver",
    "EigenSolver",
    "IterativeLinearSolver",
    "ScipyIterativeSolver",
    "CGMatrixSolver",
    "BICGMatrixSolver",
    "BICGStabMatrixSolver",
    "GMRESMatrixSolver",
    "CGSolver",
    "MinResSolver",
    "BICGStabSolver",
    "FCGSolver",
    "ProgressCallback",
    "SolutionTrackingCallback",
    "ResidualTrackingCallback",
    # preconditioners
    "IdentityPreconditioningMethod",
    "JacobiPreconditioningMethod",
    "SpectralPreconditioningMethod",
    "IterativePreconditioningMethod",
    "BandedPreconditioningMethod",
    "ExactBlockPreconditioningMethod",
    "ColumnThresholdedPreconditioningMethod",
    # forward_problem
    "ForwardProblem",
    "LinearForwardProblem",
    # linear_optimisation
    "LinearLeastSquaresInversion",
    "LinearMinimumNormInversion",
    "ConstrainedLinearLeastSquaresInversion",
    "ConstrainedLinearMinimumNormInversion",
    # linear_bayesian
    "LinearBayesianInversion",
    # nonlinear_optimisation
    "ScipyUnconstrainedOptimiser",
    # Subspaces
    "OrthogonalProjector",
    "AffineSubspace",
    "LinearSubspace",
    # Subsets
    "Subset",
    "EmptySet",
    "UniversalSet",
    "Complement",
    "Intersection",
    "Union",
    "SublevelSet",
    "LevelSet",
    "ConvexSubset",
    "Ellipsoid",
    "NormalisedEllipsoid",
    "EllipsoidSurface",
    "Ball",
    "Sphere",
    # plot
    "plot_1d_distributions",
    "plot_corner_distributions",
    "SubspaceSlicePlotter",
    "plot_slice",
    # convex_optimisation
    "SubgradientDescent",
    "ProximalBundleMethod",
    "PrimalKKTSolver",
    "KKTResult",
    # convex_analysis
    "SupportFunction",
    "BallSupportFunction",
    "EllipsoidSupportFunction",
    "HalfSpaceSupportFunction",
    "CallableSupportFunction",
    "PointSupportFunction",
    "LinearImageSupportFunction",
    "MinkowskiSumSupportFunction",
    "ScaledSupportFunction",
    # backus_gilbert
    "DualMasterCostFunction",
    # utils
    "configure_threading",
    # datasets
    "load_gsn_stations",
    "download_gsn_stations",
    "download_usgs_earthquakes",
    "sample_earthquakes",
    # congif
    "DATADIR",
    # low_rank
    "LowRankSVD",
    "LowRankEig",
    "LowRankCholesky",
    "white_noise_measure",
]
