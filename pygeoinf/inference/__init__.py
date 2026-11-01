"""
Inference: mappings from data to answers about a model.

The domain is always the data space. The **prior** fixes what kind of answer
comes out — none gives a point, a measure gives a measure, a set gives a set —
and the **property operator** fixes what the answer is about, the model itself or a
property of it. An inverse problem is an inference problem whose property
operator is the identity, so there is one code path.

    problem = LinearForwardProblem(A, error=noise)

    point = MinimumNorm(problem)                       # D -> M
    post = LinearGaussianInversion(problem, prior)     # D -> Measure(M)
    feasible = BackusGilbertParker(problem, T, ball)   # D -> Subset(P)

    measure = post(data)
    property_posterior = post.push_forward(T)(data)    # D -> Measure(P)
"""

from ..numerics.normal import FactoredNormalOperator, NormalOperator
from ..numerics.preconditioners import (
    InvariantDistancePreconditioner,
    LocalizedPreconditioner,
    NormalDiagonalPreconditioner,
    WoodburyPreconditioner,
    gaspari_cohn,
)
from ..numerics.tikhonov import TikhonovFamily, TikhonovNormalOperator
from .backus import (
    BackusGilbert,
    BackusGilbertParker,
    LinearizedFeasiblePropertySet,
    FeasiblePropertySet,
)
from .estimators import (
    ComposedPointEstimator,
    Estimator,
    GaussianEstimator,
    LinearPointEstimator,
    MeasureEstimator,
    PointEstimator,
    SetEstimator,
)
from .gaussian import (
    GaussianPosterior,
    LinearGaussianInversion,
    LowRankGaussianPosterior,
)
from .laplace import LaplaceApproximation, LaplaceResult, MaximumAPosteriori
from .minimization import (
    ConstrainedMinimization,
    Minimization,
    NonlinearLeastSquares,
    Occam,
)
from .mixture import LinearGaussianMixtureInversion
from .point import (
    ConstrainedLeastSquares,
    ConstrainedMinimumNorm,
    LeastSquares,
    MinimumNorm,
)
from .problem import ForwardProblem, LinearForwardProblem


__all__ = [
    "LaplaceApproximation",
    "LaplaceResult",
    "MaximumAPosteriori",
    "Minimization",
    "NonlinearLeastSquares",
    "ConstrainedMinimization",
    "Occam",
    "BackusGilbert",
    "BackusGilbertParker",
    "LinearizedFeasiblePropertySet",
    "FeasiblePropertySet",
    "GaussianPosterior",
    "LowRankGaussianPosterior",
    "LinearGaussianInversion",
    "LinearGaussianMixtureInversion",
    "FactoredNormalOperator",
    "NormalOperator",
    "NormalDiagonalPreconditioner",
    "LocalizedPreconditioner",
    "InvariantDistancePreconditioner",
    "WoodburyPreconditioner",
    "gaspari_cohn",
    "ConstrainedLeastSquares",
    "ConstrainedMinimumNorm",
    "TikhonovFamily",
    "TikhonovNormalOperator",
    "ComposedPointEstimator",
    "Estimator",
    "ForwardProblem",
    "GaussianEstimator",
    "LeastSquares",
    "LinearForwardProblem",
    "LinearPointEstimator",
    "MeasureEstimator",
    "MinimumNorm",
    "PointEstimator",
    "SetEstimator",
]
