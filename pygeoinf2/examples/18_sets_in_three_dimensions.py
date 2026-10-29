"""
18. A convex set in three dimensions: hulls of supporting points, and voxels.

The routes are the same as in two dimensions -- support, level, membership,
chosen by what the set can do -- but the renderings they force are new, and
each comes with a stated side of the truth. The support route draws the
convex hull of points *attaining* the support over a spread of directions,
which is inscribed: every vertex is a point of the set, so the hull sits
inside it. The sampled routes raster a cube of the parameter space and draw
it as voxels, which is why their cost is the resolution cubed and the
default drops to thirty per axis.

A set living in more than three dimensions is sliced with ``subspace=``, and
the slice of an ellipsoid is itself an ellipsoid whose axes can be written
down -- so the picture can be checked against the geometry, not just admired.
(``engine="plotly"``, the ``plot3d`` extra, draws the same pictures as
figures that can be turned by hand.)
"""

import numpy as np

from pygeoinf2 import (
    AffineSubspace,
    Ball,
    Ellipsoid,
    EuclideanSpace,
    HalfSpace,
    LinearOperator,
    Traits,
    plotting,
)

rng = np.random.default_rng(0)
X = EuclideanSpace(3)

# --- the support route: an inscribed hull ---------------------------------
ball = Ball(X, radius=1.0)
ax, hull = plotting.plot_set(ball, directions=200)
ax.set_title("a ball, as the hull of its supporting points")
print("support route: every hull vertex lies on the set, so the hull is inside it")

# --- the sampled routes: voxels -------------------------------------------
# An intersection has no support function in closed form, so it is rastered,
# and rastering needs bounds to raster in.
cut = ball & HalfSpace(X, X.basis_vector(2), offset=0.25)
ax, voxels = plotting.plot_set(cut, bounds=((-1.1, 1.1),) * 3, resolution=20)
ax.set_title("a ball cut by a half-space, rastered")
print(f"level route: {len(voxels)} voxels with a visible face")

# --- a three-dimensional slice of a four-dimensional ellipsoid -------------
Y = EuclideanSpace(4)
root = rng.normal(size=(4, 4))
matrix = root @ root.T + 4.0 * np.identity(4)
precision = LinearOperator.from_matrix(
    Y, Y, matrix, form="galerkin", traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
)
covariance = LinearOperator.from_matrix(
    Y,
    Y,
    np.linalg.inv(matrix),
    form="galerkin",
    traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
)
ellipsoid = Ellipsoid(Y, precision, covariance=covariance)

shift = 0.1
plane = AffineSubspace.from_tangent_basis(
    Y,
    [Y.basis_vector(axis) for axis in range(3)],
    translation=Y.from_components(np.array([0.0, 0.0, 0.0, shift])),
)
# The slice x = x0 + V u of (x, P x) <= 1 is the ellipsoid
# (u - u_c, M (u - u_c)) <= rho^2 with M = V' P V, so the picture has an
# exact answer to be checked against.
pulled = matrix[:3, :3]
linear = matrix[:3, 3] * shift
inverse = np.linalg.inv(pulled)
center = -inverse @ linear
rho_squared = 1.0 - shift**2 * matrix[3, 3] + linear @ inverse @ linear
half_axes = np.sqrt(rho_squared * np.diag(inverse))

resolution = 24
bounds = tuple(zip(center - 1.2 * half_axes, center + 1.2 * half_axes))
ax, sliced = plotting.plot_set(
    ellipsoid, subspace=plane, bounds=bounds, resolution=resolution
)
ax.set_title("the slice at $x_4 = 0.1$")

cells = np.array(sorted(sliced))
edges = [np.linspace(low, high, resolution + 1) for low, high in bounds]
drawn = np.array(
    [
        (edges[axis][cells[:, axis].max() + 1] - edges[axis][cells[:, axis].min()])
        / 2.0
        for axis in range(3)
    ]
)
print("slice of the 4-D ellipsoid: an ellipsoid in its own right")
print("  exact half-extents of the slice:", np.round(half_axes, 3))
print("  rastered half-extents          :", np.round(drawn, 3))

plotting.show()
