import matplotlib.pyplot as plt
from pygeoinf.homogeneous_space.sphere import Sobolev, LowPassFilter


X = Sobolev(64, 2, 0.1)

mu = X.sobolev_gaussian_measure(2, 0.5, 1)

u = mu.sample()

uc = X.to_components(u)


P = X.low_degree_projection(
    32,
    smoother=LowPassFilter(28, 32),
)

Y = P.codomain

v = P(u)

w = P.adjoint(v)

fig, ax, im = X.plot(u)
fig.colorbar(im, ax=ax, orientation="horizontal")

fig, ax, im = Y.plot(v)
fig.colorbar(im, ax=ax, orientation="horizontal")


fig, ax, im = X.plot(w)
fig.colorbar(im, ax=ax, orientation="horizontal")


plt.show()
