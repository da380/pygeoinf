"""
19. Connecting a space to real data.

An inverse problem needs a forward operator, and a forward operator is where a
function space meets an instrument. This is the layer that builds them: point
evaluation, path averages, cap averages, coefficients.

The idea to take away is that not one of these writes down an adjoint. Each is
built from *derivative components* -- the rows a numerical adjoint method
actually produces -- and the inverse metric is applied once, by the framework,
inside the adjoint. That is example 5 again, in the setting where it does the
most damage.

Needs pyshtools, which comes with the 'sphere' extra.
"""

import numpy as np

from pygeoinf2.symmetric_space.sphere import Lebesgue, Sobolev
from pygeoinf2.testing import check_operator

rng = np.random.default_rng(0)

# H^2 on the unit sphere: smooth enough that point evaluation is bounded, which
# is what makes a Dirac functional have a representer at all.
X = Sobolev(32, 2.0, 0.1)

# ---------------------------------------------------------------------------
# A real acquisition geometry, because uniform coverage flatters an inversion.
# ---------------------------------------------------------------------------

stations = X.stations(count=8, rng=rng)
sources = X.earthquakes(count=12, minimum_magnitude=5.5, rng=rng)
paths = [(source, station) for source in sources for station in stations]
print(f"{len(sources)} sources x {len(stations)} receivers = {len(paths)} paths")

separations = [X.geodesic_distance(a, b) for a, b in paths]
print(f"path lengths from {min(separations):.2f} to {max(separations):.2f} rad")
print()

# ---------------------------------------------------------------------------
# Three observation operators, none of which assembles a matrix.
# ---------------------------------------------------------------------------

A = X.path_average_operator(paths, count=12)  # tomography
E = X.point_evaluation_operator(stations)  # point data
T = X.geodesic_ball_average_operator(sources[:4], 0.15)  # a property operator

for name, operator in [("path averages", A), ("point values", E), ("caps", T)]:
    check_operator(operator, rng=rng)
    print(f"{name:14s} {X.dim:5d} -> {operator.codomain.dim:4d}  adjoint verified")
print()

# ---------------------------------------------------------------------------
# Calibration: an average of the constant one must be one.
# ---------------------------------------------------------------------------

L = Lebesgue(32, radius=2.0)
one = L.project_function(lambda point: 1.0)
a, b = sources[0], stations[0]
print("path average of 1: ", L.path_average_operator([(a, b)], count=12)(one)[0])
print("cap average of 1:  ", L.geodesic_ball_average_operator([a], 0.15)(one)[0])
print()

# The cap average has a closed form in the harmonic basis, so it costs nothing.
# Quadrature is the fallback for a region that is not a cap -- and the two
# agreeing is what says the closed form was derived correctly.
field = L.random(rng=rng)
exact = L.geodesic_ball_average_operator([a], 0.15)(field)[0]
quadrature = L.geodesic_ball_average_operator([a], 0.15, count=4000)(field)[0]
print(f"cap average, exact {exact:+.8f} vs quadrature {quadrature:+.8f}")
print()

# ---------------------------------------------------------------------------
# Matrix-free by default; assemble only when it is worth it.
# ---------------------------------------------------------------------------

dense = T.assembled()
x = X.random(rng=rng)
print("assembled agrees with matrix-free:", np.allclose(T(x), dense(x)))
print(f"its matrix is {T.codomain.dim} x {X.dim}; the path operator's would be")
print(f"{A.codomain.dim} x {X.dim} = {A.codomain.dim * X.dim:,} entries")
print()

# ---------------------------------------------------------------------------
# A prior nobody has to guess the amplitude of.
# ---------------------------------------------------------------------------

prior = X.sobolev_measure(2.0, 0.15, pointwise_std=0.05)
dirac = X.dirac(X.reference_point).representer
variance = X.inner_product(prior.covariance(dirac), dirac)
print(f"asked for a pointwise std of 0.05, got {np.sqrt(variance):.6f}")

draws = np.array(
    [X.evaluate(prior.sample(rng=rng), [X.reference_point])[0] for _ in range(200)]
)
print(f"200 samples at one point have std {draws.std():.4f}")
print()

# ---------------------------------------------------------------------------
# Synthetic data through the whole chain.
# ---------------------------------------------------------------------------

truth = prior.sample(rng=rng)
data = A(truth)
print(f"{data.size} synthetic travel-time anomalies, rms {data.std():.5f}")
print("the property we would actually like to know:", np.round(T(truth), 5))
