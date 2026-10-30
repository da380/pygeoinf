"""
25. The posterior as the prior, corrected in a few directions.

Two hundred point values of a smooth field do not carry two hundred directions'
worth of information. The posterior differs from the prior only where the data
inform it, and the directions and amounts are the eigenpairs of the
prior-preconditioned misfit Hessian

    H = L* A* R^-1 A L,        L the prior's covariance factor.

An eigenvalue ``lambda`` is the ratio of what the data say to what the prior
says in its direction, so the directions with ``lambda`` well below one are
the prior's business and can be dropped. ``LowRankGaussianPosterior`` keeps the
rest: its covariance is ``L (I - V D V*) L*`` with ``D = lambda / (1 + lambda)``,
nothing assembled, no solve, and a draw costs a prior draw.
"""

import numpy as np

from pygeoinf import (
    GaussianMeasure,
    LinearForwardProblem,
    LinearGaussianInversion,
    LinearOperator,
    LowRankGaussianPosterior,
)
from pygeoinf.symmetric_space import circle

rng = np.random.default_rng(25)

X = circle.Sobolev(256, 2.0, 0.2)
prior = X.heat_measure(0.3, pointwise_std=1.0)
sensors = np.sort(rng.uniform(0.0, 2.0 * np.pi, 200))
evaluation = X.point_evaluation_operator(sensors)

# The forward operator, wrapped to count its applications, adjoint included.
applications = {"count": 0}


def counted(action):
    def apply(x):
        applications["count"] += 1
        return action(x)

    return apply


A = LinearOperator.from_callables(
    X,
    evaluation.codomain,
    counted(evaluation),
    adjoint=counted(evaluation.adjoint),
)
noise = GaussianMeasure.from_standard_deviation(A.codomain, 0.05)
problem = LinearForwardProblem(A, error=noise)
truth, data = problem.synthetic_model_and_data(prior, rng=rng)

# ---------------------------------------------------------------------------
# The eigenvalues: chosen adaptively, as many as the randomized
# eigendecomposition finds above its tolerance.
# ---------------------------------------------------------------------------

adaptive = LowRankGaussianPosterior.from_linear_problem(problem, prior, data, rng=rng)
eigenvalues = adaptive.eigenvalues
above = int(np.sum(eigenvalues > 1.0))
print(f"{problem.data_space.dim} data; the adaptive rank is {adaptive.rank}")
print("leading eigenvalues of L* A* R^-1 A L:")
print("  ", "  ".join(f"{value:.3g}" for value in eigenvalues[: above + 4]))
print(f"{above} of them exceed one")
print()

# ---------------------------------------------------------------------------
# Truncated either side of the eigenvalue of order one, against the full
# posterior. Dropping directions the data inform leaves the prior there; the
# variance is too large and the mean misses the update in those directions.
# ---------------------------------------------------------------------------

full = LinearGaussianInversion(problem, prior)(data)
grid = X.grid_axes[0][::8]
exact = X.pointwise_variance_at(full, grid)


def pointwise_variance(measure):
    """The prior's pointwise variance, one, less the low-rank reduction."""
    reduction = measure.pointwise_variance_reduction()
    return 1.0 - X.evaluate(reduction, [np.array([point]) for point in grid])


print(f"{'rank':>5} {'largest dropped':>16} {'mean error':>11} {'variance error':>15}")
for rank in (above - 2, above + 2, adaptive.rank):
    truncated = LowRankGaussianPosterior.from_linear_problem(
        problem, prior, data, rank=rank, rng=rng
    )
    dropped = f"{eigenvalues[rank]:.3g}" if rank < eigenvalues.size else "(adaptive)"
    error = X.norm(X.subtract(truncated.expectation, full.expectation))
    variance = np.max(np.abs(pointwise_variance(truncated) - exact))
    print(
        f"{rank:5d} {dropped:>16s} {error / X.norm(full.expectation):11.1e}"
        f" {variance:15.1e}"
    )
print()
print("The pointwise variance came from pointwise_variance_reduction, the")
print("field sum_k d_k (L v_k)^2, with no solve; the full posterior's took one")
print("solve per point.")
print()

# ---------------------------------------------------------------------------
# A draw.
# ---------------------------------------------------------------------------

applications["count"] = 0
adaptive.sample(rng=rng)
low_rank_cost = applications["count"]
applications["count"] = 0
full.sample(rng=rng)
print(
    f"one posterior draw: {low_rank_cost} applications of the forward operator"
    f" from the low-rank posterior, {applications['count']} from the full one"
)
print("(a prior draw and a rank-r correction, against randomize-then-optimize)")
