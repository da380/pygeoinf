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
    DiagonalLinearOperator,
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
)

from .forward_problem import ForwardProblem, LinearForwardProblem

from .linear_optimisation import (
    LinearLeastSquaresInversion,
    LinearMinimumNormInversion,
)

from .linear_bayesian import LinearBayesianInversion, LinearBayesianInference

from .backus_gilbert import HyperEllipsoid

from .nonlinear_optimisation import (
    ScipyUnconstrainedOptimiser,
)
