import matplotlib.pyplot as plt
import numpy as np
from pygeoinf.sphere import Sobolev, LowPassFilter


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


plt.figure()
plt.pcolormesh(u.lons(), u.lats(), u.data, cmap="seismic")
plt.colorbar()

plt.figure()
plt.pcolormesh(v.lons(), v.lats(), v.data, cmap="seismic")
plt.colorbar()

plt.figure()
plt.pcolormesh(w.lons(), w.lats(), w.data, cmap="seismic")
plt.colorbar()

plt.show()
