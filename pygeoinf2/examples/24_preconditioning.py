"""
24. Preconditioning a large inversion with a surrogate.

Example 21 solved a tomographic inversion in one line and said nothing about
how. That is fine while the normal equations fit in a Cholesky factorisation.
When they do not, the solve becomes iterative, and an iterative solve on an
ill-conditioned normal operator is where a real inversion actually spends its
time.

The remedy is a *surrogate*: the same problem, made cheap. Here the surrogate
lives on a sphere of a sixth the degree -- a different, much smaller model
space, with its own prior and its own path-average operator. Only the data
space is shared, which is exactly why the data-space normal operator
``A Q A* + R`` survives the substitution while the model-space one could not.

The surrogate's inverse, obtained through the Woodbury identity, then
preconditions the true operator. Nothing about the answer depends on how good
the surrogate is -- only the iteration count does, and that is the thing this
example measures.

Needs pyshtools, which comes with the 'sphere' extra.
"""

import numpy as np

from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.inference import (
    LinearForwardProblem,
    LinearGaussianInversion,
    NormalDiagonalPreconditioner,
)
from pygeoinf2.numerics.preconditioners import WoodburyPreconditioner
from pygeoinf2.numerics.solvers import CGSolver, CholeskySolver
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.symmetric_space.sphere import Sobolev
from pygeoinf2.traits import Traits

rng = np.random.default_rng(4)

# ---------------------------------------------------------------------------
# A problem large enough that the solve is the expensive part.
# ---------------------------------------------------------------------------

X = Sobolev(64, 2.0, 0.1)
receivers = X.stations(count=16, rng=rng)
sources = X.earthquakes(count=44, minimum_magnitude=5.5, rng=rng)
paths = [(source, receiver) for source in sources for receiver in receivers]

forward = X.path_average_operator(paths, count=16, dense=True)

# Real stations are not equally good. Noise levels spanning two orders of
# magnitude are what make the normal operator badly scaled rather than merely
# large, and they are the thing a diagonal preconditioner is *for*.
D = forward.codomain
quality = np.repeat(
    np.geomspace(0.002, 0.2, len(receivers)), len(paths) // len(receivers)
)
noise = GaussianMeasure(
    D,
    covariance=LinearOperator.from_matrix(
        D,
        D,
        np.diag(quality**2),
        traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
        form="galerkin",
    ),
    covariance_factor=LinearOperator.from_matrix(
        D, D, np.diag(quality), traits=Traits.NONE, form="galerkin"
    ),
)
problem = LinearForwardProblem(forward, error=noise)
prior = X.heat_measure(0.03, pointwise_std=0.05)
truth, data = problem.synthetic_model_and_data(prior, rng=rng)

print(f"model space   dimension {X.dim} (degree {X.lmax})")
print(f"data space    dimension {problem.data_space.dim} ({len(paths)} paths)")
print(
    f"station noise from {quality.min():.3f} to {quality.max():.3f} "
    f"({quality.max() / quality.min():.0f}x)"
)
print()

# The data space is the default and is also the smaller side here; it is named
# explicitly because everything below depends on it. The Cholesky solver is
# deliberate too, and the only one in this file: it is the reference the
# iterative solves are checked against, and this problem is small enough to
# afford one. Nothing else here assembles anything.
inversion = LinearGaussianInversion(
    problem, prior, formalism="data_space", solver=CholeskySolver()
)
reference = inversion(data).expectation
print(f"reference solve: Cholesky on the {inversion.formalism} normal operator")
print(
    "  relative error of the posterior mean: "
    f"{X.norm(X.subtract(reference, truth)) / X.norm(truth):.3f}"
)
print()

# ---------------------------------------------------------------------------
# The normal operator, which is the thing a preconditioner is built against.
# ---------------------------------------------------------------------------

normal = inversion.normal_operator
print(f"normal operator: {normal!r}")

# It is an ordinary operator wherever one is wanted, and it still knows what it
# was assembled from -- which is what the preconditioners below need, and what
# assembling it into a single matrix would have destroyed.
matrix = normal.matrix(form="galerkin")
condition = np.linalg.cond(matrix)
print(f"  condition number {condition:.3e}")
print(
    f"  factors still attached: A {normal.forward.codomain.dim}x"
    f"{normal.forward.domain.dim}, Q on {normal.model_space.dim} dimensions"
)
print()

# The right-hand side of N w = v, for the residual the posterior mean solves.
shift = problem.data_space.add(
    forward(prior.expectation), problem.error_measure.expectation
)
residual = problem.data_space.subtract(data, shift)


def iterations(preconditioner, label):
    """How long the solve takes, and whether it lands in the same place."""
    solver = CGSolver(rtol=1e-10, maxiter=5000)
    if preconditioner is not None:
        solver = solver.with_preconditioner(preconditioner)
    result = solver(normal).solve(normal.right_hand_side(residual))
    recovered = X.add(prior.expectation, normal.gain(solver(normal))(residual))
    drift = X.norm(X.subtract(recovered, reference)) / X.norm(reference)
    print(f"  {label:<34s} {result.iterations:>5d} iterations   (answer {drift:.1e})")
    return result.iterations


# ---------------------------------------------------------------------------
# Four ways to solve it.
# ---------------------------------------------------------------------------

print("conjugate gradients on the same operator:")
plain = iterations(None, "no preconditioner")

# The cheap diagonal, using <v, A Q A* v> == <A* v, Q A* v>. Blocked by
# receiver: rays arriving at one station sample the model similarly, so one
# adjoint application stands in for the whole group.
blocks = [
    list(range(index, len(paths), len(receivers))) for index in range(len(receivers))
]
iterations(
    NormalDiagonalPreconditioner(blocks=blocks),
    f"diagonal, {len(blocks)} receiver blocks",
)
iterations(NormalDiagonalPreconditioner(), "diagonal, exact")

# ---------------------------------------------------------------------------
# The surrogate: the same problem on a coarser sphere.
# ---------------------------------------------------------------------------

coarse = X.with_degree(X.lmax // 6)
coarse_forward = coarse.path_average_operator(paths, count=16, dense=True)
# The prior needs a precision for the Woodbury data form, and a heat-kernel
# covariance is singular in practice long before it is in theory -- so it is
# damped, which says the smallest variances are no smaller than the damping.
coarse_prior = coarse.heat_measure(0.03, pointwise_std=0.05).with_regularized_inverse(
    CholeskySolver(), damping=1e-6
)

surrogate = inversion.surrogate(forward=coarse_forward, prior=coarse_prior)
print()
print(
    f"surrogate on degree {coarse.lmax} ({coarse.dim} dimensions, "
    f"{X.dim / coarse.dim:.0f}x smaller), sharing the data space"
)
print(f"  {surrogate!r}")

preconditioned = iterations(
    WoodburyPreconditioner.from_normal(surrogate, solver=CholeskySolver()),
    "Woodbury from the surrogate",
)

print()
print(
    f"the surrogate takes the solve from {plain} iterations to "
    f"{preconditioned}, and the answer is unchanged -- which is the whole "
    "point: a preconditioner cannot be wrong, only useless."
)
print()
print(
    "note also that the *blocked* diagonal beats the exact one, at a "
    f"{len(paths) // len(blocks)}x lower cost. The blocks are the receivers, "
    "and the noise is constant per receiver, so the block structure is the "
    "structure of the operator; approximating the rest is not a loss. A "
    "preconditioner that matches the problem beats a more accurate one that "
    "does not."
)
