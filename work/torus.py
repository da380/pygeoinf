import matplotlib.pyplot as plt
import pygeoinf as inf
from pygeoinf.symmetric_space.torus import (
    Sobolev,
    plot,
    plot_geodesic_network,
)

X = Sobolev(128, 2, 0.1)

paths = X.random_source_receiver_paths(20, 20)
A = X.path_average_operator(paths)


B = inf.LowRankSVD.from_randomized(
    A, 20, method="fixed", measure=inf.white_noise_measure(A.domain)
)

v = A.codomain.random()

u = A.adjoint(v)

w = B.adjoint(v)

ax, im = plot(X, u)
plot_geodesic_network(paths, ax=ax)

ax, im = plot(X, w)
plot_geodesic_network(paths, ax=ax)


plt.show()
