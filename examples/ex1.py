import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf
from pygeoinf.symmetric_space.line import Sobolev

X = Sobolev.from_sobolev_parameters(3, 0.1, x1=10)
mu = X.heat_gaussian_measure(0.1, 1)

Y = inf.HilbertSpaceDirectSum([X, X])

nu = inf.GaussianMeasure.from_direct_sum([mu, mu])

v = nu.sample()

fig, ax = X.plot(v[0])
X.plot(v[1], fig=fig, ax=ax)
plt.show()

# u = X.project_function(lambda x: np.exp(-10 * (x - 5) ** 2))
# u = A(u)

# u = X.project_function(lambda x: x)

# fig, ax = X.plot(u, computational_domain=True)
# plt.show()
