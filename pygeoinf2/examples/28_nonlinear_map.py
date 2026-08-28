"""
28. When the forward map is not linear: the mode, and a Gaussian on it.

Everything up to here has assumed the forward operator is linear, and that
assumption does a great deal of work: it makes the posterior Gaussian, its mean
a linear function of the data, and its covariance independent of the data
altogether. None of that survives a nonlinear map.

What survives is the *recipe*. The posterior is still proportional to
``exp(-chi^2/2)`` times the prior, so its logarithm is still a sum of a misfit
and a prior term — and finding where that is largest is an optimisation, which
is a thing this library can already do. Expanding it to second order about the
answer gives a Gaussian, and the curvature of a misfit at a point is the normal
operator of the problem *linearised there*. So the linear machinery is not
discarded; it is evaluated at the mode instead of assumed everywhere.

That is the Laplace approximation. This example shows it working, and then
shows where it stops telling the truth — because an approximation whose failure
mode is never displayed is one people will trust past its range.

Needs matplotlib.
"""

import matplotlib.pyplot as plt
import numpy as np

import pygeoinf2 as gi
from pygeoinf2.algebra.operators import LinearOperator, Operator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.inference.problem import ForwardProblem

rng = np.random.default_rng(11)

# ------------------------------------------------------------------ #
# A nonlinear forward problem                                        #
# ------------------------------------------------------------------ #
#
# Attenuation along a set of paths: each datum is exp(-(a, m)) for a
# non-negative path weight a. Linear in the *log*, and it is the map itself
# that is measured, so the problem is nonlinear in the model.

X = EuclideanSpace(2)
D = EuclideanSpace(24)

paths = np.abs(rng.standard_normal((24, 2))) + 0.2


def value(components: np.ndarray) -> np.ndarray:
    """The predicted data: attenuation along each path."""
    return np.exp(-(paths @ components))


def derivative(components: np.ndarray) -> LinearOperator:
    """Its derivative, which is where the nonlinearity lives."""
    jacobian = -np.diag(np.exp(-(paths @ components))) @ paths
    return LinearOperator.from_matrix(X, D, jacobian, form="components")


forward = Operator.from_callables(X, D, value, derivative=derivative)

prior = gi.GaussianMeasure.from_standard_deviation(X, 0.6)
error = gi.GaussianMeasure.from_standard_deviation(D, 0.02)

truth = X.from_components(np.array([0.8, 0.35]))
data = D.add(forward(truth), error.sample(rng=rng))

problem = ForwardProblem(forward, error=error)

# ------------------------------------------------------------------ #
# The mode, and the Gaussian about it                                #
# ------------------------------------------------------------------ #

estimator = gi.MaximumAPosteriori(problem, prior)
result = estimator(data)

print(f"the search converged: {result.converged}")
print(f"  in {result.optimisation.iterations} iterations, "
      f"{result.optimisation.evaluations} evaluations")
print(f"  truth     {np.round(X.to_components(truth), 4)}")
print(f"  recovered {np.round(X.to_components(result.model), 4)}")
print()

# The mode is a stationary point of the objective, which is the thing that
# makes it the mode. Worth checking rather than taking on trust.
gradient = estimator.objective(data).gradient(result.model)
print(f"gradient at the mode: {X.norm(gradient):.2e}")

# The covariance is the inverse of the linearised normal operator -- the same
# construction the linear estimator makes, evaluated here rather than assumed.
covariance = result.measure.covariance.matrix(form="components")
deviations = np.sqrt(np.diag(covariance))
correlation = covariance[0, 1] / (deviations[0] * deviations[1])
print(f"posterior standard deviations: {np.round(deviations, 4)}")
print(f"posterior correlation: {correlation:+.3f}")
print()

# ------------------------------------------------------------------ #
# Is the Gaussian telling the truth?                                 #
# ------------------------------------------------------------------ #
#
# The approximation is exact for a linear map and good near a quadratic peak.
# Here we can check it directly, because the model space is two-dimensional:
# evaluate the true log posterior on a grid and compare its contours with the
# Gaussian's.

span = 4.0
axis_x = np.linspace(
    X.to_components(result.model)[0] - span * deviations[0],
    X.to_components(result.model)[0] + span * deviations[0],
    160,
)
axis_y = np.linspace(
    X.to_components(result.model)[1] - span * deviations[1],
    X.to_components(result.model)[1] + span * deviations[1],
    160,
)
mesh_x, mesh_y = np.meshgrid(axis_x, axis_y)

density = estimator.log_posterior(data)
exact = np.array(
    [
        [
            density(X.from_components(np.array([x, y])))
            for x in axis_x
        ]
        for y in axis_y
    ]
)
exact -= exact.max()

# The Gaussian's own log density, up to the same constant.
precision = np.linalg.inv(covariance)
offset = np.stack(
    [mesh_x - X.to_components(result.model)[0],
     mesh_y - X.to_components(result.model)[1]],
    axis=-1,
)
approximate = -0.5 * np.einsum("...i,ij,...j->...", offset, precision, offset)

levels = [-0.5 * s**2 for s in (3.0, 2.0, 1.0)]

figure, axes = plt.subplots(1, 1, figsize=(6.0, 5.0))
axes.contour(mesh_x, mesh_y, exact, levels=levels, colors="black", linewidths=1.4)
axes.contour(
    mesh_x, mesh_y, approximate, levels=levels, colors="C0", linewidths=1.4,
    linestyles="--",
)
axes.plot(*X.to_components(result.model), "o", color="C0", label="MAP model")
axes.plot(*X.to_components(truth), "*", color="black", ms=14, label="truth")
axes.set_xlabel("component 0")
axes.set_ylabel("component 1")
axes.set_title("true posterior (solid) against its Laplace approximation (dashed)")
axes.legend(loc="upper right")

# How far apart are they? Compare the exact and approximate densities at the
# one-, two- and three-sigma contours of the Gaussian.
for sigmas in (1.0, 2.0, 3.0):
    ring = np.abs(approximate - (-0.5 * sigmas**2)) < 0.02
    if ring.any():
        discrepancy = np.abs(exact[ring] - approximate[ring]).max()
        print(f"at {sigmas:.0f} sigma, the two log densities differ by up to "
              f"{discrepancy:.3f}")

print()
print("the dashed ellipses are the Gaussian's; the solid contours are the")
print("  posterior's own. Near the mode they agree -- which is the whole claim")
print("  of the method -- and they part company as the contours widen, because")
print("  an exponential forward map is not quadratic and the true posterior is")
print("  skewed. The approximation is exact only for a linear map.")
print()
print("that skew is why the mode is not the mean here: an optimiser finds the")
print("  peak, and for an asymmetric density the peak and the average are")
print("  different points. Every linear method in this library returns a mean.")
print("  This one returns a mode, and says so.")
print()
print("one figure drawn; matplotlib.pyplot.show() displays it")
