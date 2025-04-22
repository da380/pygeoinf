from pygeoinf.linalg import LinearOperator
import matplotlib.pyplot as plt
from pygeoinf.sphere import Lebesgue, Sobolev


X = Sobolev(64, 2, 0.1)

mu = -X.sobolev_gaussian_measure(2, 0.1, 1)

u = mu.sample()

v = X.dirac_representation(20, 100)

A = LinearOperator.self_adjoint_from_tensor_product(X, [v])

w = A(u)

val = X.inner_product(v, u)


plt.figure()
plt.pcolormesh(u.lons(), u.lats(), u.data, cmap="seismic")
plt.colorbar()

plt.figure()
plt.pcolormesh(u.lons(), u.lats(), w.data, cmap="seismic")
plt.colorbar()

plt.show()
