"""
Unified imports for the package.
"""

from .random_matrix import (
    fixed_rank_random_range,
    variable_rank_random_range,
    random_range,
    random_svd,
    random_eig,
    random_cholesky,
    random_diagonal,
)

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
    NormalSumOperator,
)


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
)

from .preconditioners import (
    JacobiPreconditioningMethod,
    SpectralPreconditioningMethod,
    IdentityPreconditioningMethod,
    IterativePreconditioningMethod,
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
    ConstrainedLinearBayesianInversion,
)

from .backus_gilbert import HyperEllipsoid

from .nonlinear_optimisation import (
    ScipyUnconstrainedOptimiser,
)


from .subspaces import OrthogonalProjector, AffineSubspace, LinearSubspace

from .plot import plot_1d_distributions, plot_corner_distributions

from .utils import set_serial_backend

__all__ = [
    # random_matrix
    "fixed_rank_random_range",
    "variable_rank_random_range",
    "random_range",
    "random_svd",
    "random_eig",
    "random_cholesky",
    "random_diagonal",
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
    "NormalSumOperator",
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
    # preconditioners
    "IdentityPreconditioningMethod",
    "JacobiPreconditioningMethod",
    "SpectralPreconditioningMethod",
    "IterativePreconditioningMethod",
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
    "ConstrainedLinearBayesianInversion",
    # backus_gilbert
    "HyperEllipsoid",
    # nonlinear_optimisation
    "ScipyUnconstrainedOptimiser",
    # Subspaces
    "OrthogonalProjector",
    "AffineSubspace",
    "LinearSubspace",
    # plot
    "plot_1d_distributions",
    "plot_corner_distributions",
    # utils
    "set_serial_backend",
]
