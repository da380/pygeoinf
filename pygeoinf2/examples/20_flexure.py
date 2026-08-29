"""
20. A spatially varying coefficient, and what it takes to have one.

v1's ``work/flexure.py`` on v2: a floating elastic plate on a sphere, stiffened
where there are continents, loaded at random, and solved for its deflection.

This is the first example where the operator is not invariant. The rigidity
varies in space, so the operator is no longer diagonal in the harmonic basis
and no closed-form inverse exists -- but the constant-rigidity inverse is still
an excellent preconditioner, and that is what makes the solve cheap.

Three things had to exist for it to be written at all: pointwise multiplication
of fields, which is a *capability* a space declares rather than something the
core assumes; the covariant flexure operator, built from a Bochner identity so
that no Hessian or tangent frame is ever formed; and a plotting layer that
dispatches on the space's type.

Needs pyshtools and cartopy, which come with the 'sphere' extra.
"""

import numpy as np
import matplotlib.pyplot as plt

from pygeoinf2 import plotting
from pygeoinf2.numerics.solvers import CGSolver
from pygeoinf2.symmetric_space.sphere import Lebesgue

rng = np.random.default_rng(0)

X = Lebesgue(64)

# Poisson's ratio, a normalised restoring force, and an oceanic rigidity.
POISSON, BUOYANCY, OCEANIC = 0.25, 1.0, 1.0e-4

# ---------------------------------------------------------------------------
# A rigidity field: ten times stiffer under the continents.
# ---------------------------------------------------------------------------

land = X.domain_mask()
rough = X.multiply(X.project_function(lambda point: OCEANIC), 1.0 + 9.0 * land)

# Sharp coastlines ring in a truncated harmonic basis, so smooth them with a
# heat kernel. The covariance of a heat-kernel measure *is* that smoother.
rigidity = X.heat_measure(0.14).covariance(rough)
# Fields are SHGrid objects; grid_values reaches their numbers.
stiffness = X.grid_values(rigidity)
print(f"rigidity from {stiffness.min():.3e} to {stiffness.max():.3e}")
print(f"land fraction {X.grid_values(land).mean():.3f}")
print()

# ---------------------------------------------------------------------------
# The operator, and the two inverses.
# ---------------------------------------------------------------------------

operator = X.flexural_operator(rigidity, POISSON, BUOYANCY)
print("flexural operator traits:", operator.traits)

# Uniform ocean: invariant, so the inverse is exact and diagonal.
uniform = X.inverse_flexural_operator(OCEANIC, POISSON, BUOYANCY)

# The real thing: preconditioned CG, with the uniform inverse as the
# preconditioner. Without it this is a badly conditioned fourth-order solve.
solver = CGSolver(rtol=1e-8, strict=False)
varying = X.inverse_flexural_operator(
    rigidity, POISSON, BUOYANCY, baseline_rigidity=OCEANIC, solver=solver
)

# ---------------------------------------------------------------------------
# A load, and the two deflections.
# ---------------------------------------------------------------------------

load = X.sobolev_measure(2.0, 0.1).sample(rng=rng)
deflection = varying(load)
baseline = uniform(load)

residual = X.norm(X.subtract(operator(deflection), load)) / X.norm(load)
print(f"solved to a relative residual of {residual:.2e}")

difference = X.subtract(deflection, baseline)
print(
    f"the continents change the deflection by {X.norm(difference) / X.norm(baseline):.1%}"
)
print()

# ---------------------------------------------------------------------------
# Four panels.
# ---------------------------------------------------------------------------

figure, axes = plotting.subplots(X, rows=2, columns=2)
panels = axes.ravel()

plotting.plot(
    X, rigidity, ax=panels[0], cmap="viridis", coasts=True, colorbar_label="rigidity"
)
panels[0].set_title("Flexural rigidity")

plotting.plot(
    X,
    load,
    ax=panels[1],
    cmap="RdBu_r",
    symmetric=True,
    coasts=True,
    colorbar_label="load",
)
panels[1].set_title("Load")

plotting.plot(
    X,
    deflection,
    ax=panels[2],
    cmap="RdBu_r",
    symmetric=True,
    coasts=True,
    colorbar_label="deflection",
)
panels[2].set_title("Deflection, variable rigidity")

plotting.plot(
    X,
    difference,
    ax=panels[3],
    cmap="PRGn",
    symmetric=True,
    coasts=True,
    colorbar_label="difference",
)
panels[3].set_title("Effect of the continents")

print("four panels drawn; matplotlib.pyplot.show() displays them")


plt.show()
