from pygeoinf.sphere import Sobolev
import matplotlib.pyplot as plt

X = Sobolev(2, 2, 0.1)

u = X.random()

c = X.to_components(u)

u2 = X.from_components(c)

plt.figure()
plt.pcolormesh(u.lons(), u.lats(), u.data, cmap="seismic")
plt.colorbar()

plt.figure()
plt.pcolormesh(u2.lons(), u2.lats(), u2.data, cmap="seismic")
plt.colorbar()

plt.show()
