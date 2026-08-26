"""
14. Fields on real domains.

Circles, tori, boxes and spheres. One implementation covers every periodic box
in any dimension, so the three-dimensional case costs nothing extra. A Sobolev
space is the same coordinate map as its Lebesgue counterpart with a different
metric -- so an invariant operator on it is still diagonal.
"""

import numpy as np

from pygeoinf2.symmetric_space import Interval, Sobolev

rng = np.random.default_rng(0)

print("one implementation, any dimension:")
for shape, name in [((32,), "circle"), ((16, 16), "torus"), ((8, 8, 8), "3D box")]:
    space = Sobolev(shape, 2.0, 0.2)
    print(f"  {name:8s} shape={str(shape):12s} dim={space.dim:5d}")
print()

X = Sobolev((64,), 2.0, 0.2)

# The Laplacian is diagonal, so its spectrum is exact and its calculus is free.
print("the Laplacian's traits:", X.laplacian.traits)
field = X.project_function(lambda t: np.cos(3.0 * t))
print(
    "it reproduces the analytic eigenvalue 9:",
    np.allclose(X.laplacian(field), 9.0 * field, atol=1e-10),
)
print()

# A prior. Smoothness set by the order, correlation length by the scale.
rough = X.sobolev_measure(1.0, 0.05).sample(rng=rng)
smooth = X.sobolev_measure(4.0, 0.05).sample(rng=rng)


def roughness(x):
    return X.norm(X.laplacian(x)) / X.norm(x)


print(f"roughness of an H1 sample : {roughness(rough):8.2f}")
print(f"roughness of an H4 sample : {roughness(smooth):8.2f}")
print()

# Point evaluation. The Dirac is built from derivative components, so its
# representer costs an inverse metric -- which is exactly right, and is why a
# Dirac is a function on H2 and not on L2.
point = np.array([1.3])
dirac = X.dirac(point)
print("dirac(f) == f(1.3) ?", np.isclose(dirac(field), np.cos(3.0 * 1.3), atol=1e-10))
print(
    "its representer differs from the raw components:",
    not np.allclose(X.to_components(dirac.representer), dirac.matrix().ravel()),
)
print()

# A bounded domain is the same space embedded in a padded periodic one.
line = Interval(64, lower=0.0, upper=1.0)
print(f"an interval [0, 1] with padding {line.padding[0]:.2f}")
print(
    f"  periodic domain length {line.volume:.2f}, domain proper {line.domain_volume:.2f}"
)
sampled = line.project_function(lambda t: np.sin(np.pi * t))
print("  a field vanishes on the padding:", np.all(sampled[~line.interior_mask] == 0.0))
