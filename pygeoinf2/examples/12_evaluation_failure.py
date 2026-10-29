"""
12. A forward that fails to converge, and an optimizer that survives it.

A real forward solver has models it cannot evaluate: the time step blows up,
the nonlinear solve diverges, the trial model is unphysical. ``EvaluationError``
is the protocol for saying so. Raised from the forward's value or
linearization, the *point* is treated as bad rather than the run as broken: a
line search backtracks as if the step went too far, and an optimizer that
meets a failure where it cannot retreat returns its last good state with
``failure`` set -- a run of hours is handed back, not lost. Any other
exception still propagates, because a bug should not be walked around by a
backtracking loop.
"""

import numpy as np

from pygeoinf2 import EuclideanSpace, EvaluationError, Functional
from pygeoinf2.numerics.line_search import ArmijoLineSearch, StrongWolfeLineSearch
from pygeoinf2.numerics.optimization import LBFGS

rng = np.random.default_rng(7)
X = EuclideanSpace(2)

# ``||x||^2``, except that the "solver" diverges wherever x_0 < -0.5. The
# minimum sits at the origin, safely evaluable; a full first quasi-Newton
# step from the start overshoots straight into the wall.
WALL = -0.5


def value(x):
    c = X.to_components(x)
    if c[0] < WALL:
        raise EvaluationError("the forward did not converge")
    return float(c @ c)


def gradient(x):
    c = X.to_components(x)
    if c[0] < WALL:
        raise EvaluationError("the forward did not converge")
    return X.from_components(2.0 * c)


objective = Functional.from_callables(X, value, gradient=gradient)
start = X.from_components(np.array([0.9, 0.0]))

# --- 1. the line searches back out of the bad region -----------------------
result = LBFGS(line_search=StrongWolfeLineSearch(), max_iterations=50).minimize(
    objective, start
)
print("with a wall at x_0 == -0.5, starting at x_0 == 0.9:")
print(f"  converged : {result.converged} ({result.message})")
print(f"  minimizer : {np.round(X.to_components(result.minimizer), 8)}")
print(f"  failure   : {result.failure}")
print()

# --- 2. a failure with nowhere to retreat returns the state ----------------
# Here the *derivative* fails everywhere but the start: a backtracking search
# accepts a point by its value alone, and the gradient needed to continue
# cannot be had. (A Wolfe search never gets that far -- it needs the gradient
# at every trial, so it reports an ordinary search failure instead.)


def brittle_gradient(x):
    c = X.to_components(x)
    if not np.allclose(c, X.to_components(start)):
        raise EvaluationError("the adjoint solve diverged")
    return X.from_components(2.0 * c)


brittle = Functional.from_callables(X, value, gradient=brittle_gradient)
result = LBFGS(line_search=ArmijoLineSearch(), max_iterations=50).minimize(
    brittle, start
)
print("when the derivative fails at an accepted point:")
print(f"  converged : {result.converged}")
print(f"  message   : {result.message}")
print(f"  failure   : {result.failure}")
print("  the minimizer is the last good point, so nothing is lost but a step:")
print(f"  minimizer : {np.round(X.to_components(result.minimizer), 8)}")
