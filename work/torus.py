import matplotlib.pyplot as plt
from pygeoinf.symmetric_space.torus import (
    Sobolev,
    plot,
    plot_geodesic_network,
)

X = Sobolev(128, 2, 0.1)

paths = X.random_source_receiver_paths(10, 10)
A = X.path_average_operator(paths)

v = A.codomain.random()

u = A.adjoint(v)

ax, im = plot(X, u)

plot_geodesic_network(paths, ax=ax)


plt.show()
