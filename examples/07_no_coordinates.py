"""
7. A Hilbert space with no coordinates at all.

Example 2 said that a Hilbert space needs an inner product and not a basis. Here
that is put to the test: a space whose vectors are plain Python lists, which
supplies an inner product and the vector algebra and nothing else. It declares
no basis and no Gram matrix, and has none to give.

Conjugate gradients solve on it, and a Gaussian measure lives on it, samples and
pushes forward. The methods that genuinely need components -- a Cholesky
factorization, a matrix -- refuse it at once, by name, rather than three calls
deep.
"""

import math

import numpy as np

from pygeoinf import (
    CGSolver,
    CholeskySolver,
    EuclideanSpace,
    GaussianMeasure,
    HilbertSpace,
    LinearFunctional,
    LinearOperator,
    Traits,
)
from pygeoinf.testing import check_operator, check_traits


class Samples(HilbertSpace):
    """Functions on [0, 1], held as a Python list of their values at n nodes.

    The inner product is the trapezoidal rule, so it is the integral of the
    product to second order and not the dot product of the lists.
    """

    def __init__(self, n):
        self.n = n
        self.h = 1.0 / (n - 1)
        self.weights = [self.h] * n
        self.weights[0] = self.weights[-1] = self.h / 2.0

    @property
    def dim(self):
        return self.n

    # It discretizes functions, so the library treats it as it treats any
    # function space: no identity covariance, no density against the grid.
    @property
    def is_function_space(self):
        return True

    def _key(self):
        return self.n

    def zero(self):
        return [0.0] * self.n

    def copy(self, x):
        return list(x)

    def inner_product(self, x, y):
        return math.fsum(w * a * b for w, a, b in zip(self.weights, x, y))

    def axpy(self, a, x, y):
        for i, value in enumerate(x):
            y[i] += a * value
        return y

    def scale_inplace(self, a, x):
        for i in range(len(x)):
            x[i] *= a
        return x

    # Optional: the checks draw their probe vectors from it.
    def random(self, *, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        return [float(value) for value in rng.standard_normal(self.n)]

    def nodes(self):
        return [i * self.h for i in range(self.n)]


rng = np.random.default_rng(7)
X = Samples(201)
print(f"a vector is a {type(X.zero()).__name__}; the space has dimension {X.dim}")
one = [1.0] * X.dim
print(f"(1, 1) = {X.inner_product(one, one):.4f}, the length of the interval")
print()

# ---------------------------------------------------------------------------
# An operator, written on the lists: A = 1 - L^2 d^2/dx^2 with zero slope at
# the ends. Its second-difference part, divided by the trapezoid weights, is
# self-adjoint in this inner product, which is what the claimed traits say.
# ---------------------------------------------------------------------------

LENGTH = 0.02


def second_difference(x):
    """K x, the stiffness of -d^2/dx^2 with zero slope at the ends."""
    out = [0.0] * X.n
    for i in range(X.n - 1):
        flux = (x[i + 1] - x[i]) / X.h
        out[i] -= flux
        out[i + 1] += flux
    return out


def apply(x):
    stiff = second_difference(x)
    return [a + LENGTH**2 * k / w for a, k, w in zip(x, stiff, X.weights)]


A = LinearOperator.self_adjoint(X, apply, traits=Traits.POSITIVE_DEFINITE)
print(check_operator(A, rng=rng))
print(check_traits(A, rng=rng))
print()

# Conjugate gradients need the inner product and axpy, and nothing else. The
# equation u - L^2 u'' = x^2 with zero slope at the ends is solved by
# u = x^2 + 2 L^2 - 2 L cosh(x / L) / sinh(1 / L), which the discrete answer
# approaches.
nodes = X.nodes()
right_hand_side = [t * t for t in nodes]
result = CGSolver(rtol=1e-8)(A).solve(right_hand_side)
exact = [
    t * t
    + 2.0 * LENGTH**2
    - 2.0 * LENGTH * math.cosh(t / LENGTH) / math.sinh(1 / LENGTH)
    for t in nodes
]
error = X.norm(X.subtract(result.solution, exact)) / X.norm(exact)
print(f"CG on the lists: {result.iterations} iterations, converged {result.converged}")
print(f"  against the continuous solution, relative error {error:.1e}")
print()

# ---------------------------------------------------------------------------
# A Gaussian measure, through a factor from a Euclidean latent space: a draw
# is L xi with xi white noise in R^k, so nothing on X is ever factorized. The
# columns of L are cosines with decaying weights, a smooth prior.
# ---------------------------------------------------------------------------

k = 30
latent = EuclideanSpace(k)
deviations = [0.5 / (1.0 + (0.1 * math.pi * j) ** 2) for j in range(k)]
modes = [[1.0] * X.dim] + [
    [math.sqrt(2.0) * math.cos(math.pi * j * t) for t in nodes] for j in range(1, k)
]


def combine(c):
    field = X.zero()
    for coefficient, sigma, mode in zip(c, deviations, modes):
        field = X.axpy(float(coefficient) * sigma, mode, field)
    return field


def project(field):
    return np.array(
        [sigma * X.inner_product(mode, field) for sigma, mode in zip(deviations, modes)]
    )


factor = LinearOperator.from_callables(latent, X, combine, adjoint=project)
prior = GaussianMeasure(X, covariance_factor=factor)
print("a Gaussian on the lists, covariance", prior.covariance.traits)

# The weighted average (x, t) is a linear functional, held by its representer,
# the list of node positions; under the prior its variance is (C t, t).
ramp = LinearFunctional.from_representer(X, nodes)
claimed = X.inner_product(prior.covariance(nodes), nodes)
draws = [ramp(prior.sample(rng=rng)) for _ in range(4000)]
print(f"variance of (x, t): claimed {claimed:.5f}, sampled {np.var(draws):.5f}")
print(f"and pushed forward, {(ramp @ prior).covariance.matrix()[0, 0]:.5f}")
print()

# ---------------------------------------------------------------------------
# What needs coordinates says so.
# ---------------------------------------------------------------------------

for name, attempt in (
    ("a Cholesky factorization", lambda: CholeskySolver()(A)),
    ("a matrix", lambda: A.matrix()),
):
    try:
        attempt()
    except TypeError as refusal:
        print(f"{name} is refused: {str(refusal).split('. ')[0]}.")
