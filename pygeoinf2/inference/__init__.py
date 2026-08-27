"""
Inference: mappings from data to answers about a model.

The domain is always the data space. The **prior** fixes what kind of answer
comes out — none gives a point, a measure gives a measure, a set gives a set —
and the **target** fixes what the answer is about, the model itself or a
property of it. An inverse problem is an inference problem whose property
operator is the identity, so there is one code path.

    problem = LinearForwardProblem(A, error=noise)

    point = MinimumNorm(problem)                    # D -> M
    post  = LinearGaussianInversion(problem, prior)                # D -> Measure(M)

    measure = post(data)
    property_posterior = post.push_forward(T)(data)  # D -> Measure(P)

See DESIGN.md section 18.
"""

from .backus import (
    BackusGilbert,
    BackusInference,
    DualFeasibleProperty,
    FeasibleProperty,
)
from .gaussian import LinearGaussianInversion
from .normal import NormalOperator
from .preconditioners import (
    InvariantDistancePreconditioner,
    LocalisedPreconditioner,
    NormalDiagonalPreconditioner,
    gaspari_cohn,
)
from .estimators import (
    Estimator,
    GaussianEstimator,
    LinearPointEstimator,
    MeasureEstimator,
    PointEstimator,
    SetEstimator,
)
from .tikhonov import TikhonovFamily, TikhonovNormalOperator
from .point import (
    ConstrainedLeastSquares,
    ConstrainedMinimumNorm,
    DiscrepancyPrinciple,
    LeastSquares,
    MinimumNorm,
    choose_formalism,
)
from .problem import ForwardProblem, LinearForwardProblem

__all__ = [
    "BackusGilbert",
    "BackusInference",
    "LinearGaussianInversion",
    "NormalOperator",
    "NormalDiagonalPreconditioner",
    "LocalisedPreconditioner",
    "InvariantDistancePreconditioner",
    "gaspari_cohn",
    "ConstrainedLeastSquares",
    "ConstrainedMinimumNorm",
    "DiscrepancyPrinciple",
    "TikhonovFamily",
    "TikhonovNormalOperator",
    "DualFeasibleProperty",
    "FeasibleProperty",
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
    "choose_formalism",
]
