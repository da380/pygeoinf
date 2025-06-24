import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf
from pygeoinf.homogeneous_space.circle import Sobolev

X = Sobolev.from_sobolev_parameters(2, 0.1)
mu = X.heat_gaussian_measure(0.1, 1)

A = X.invariant_automorphism(lambda k: 1)
B = X.invariant_automorphism(lambda k: -1)

Y = inf.HilbertSpaceDirectSum([X, X])
nu = inf.GaussianMeasure.from_direct_sum([mu, mu])

C = inf.ColumnLinearOperator([A, B])
D = inf.RowLinearOperator([A, A])

x = mu.sample()
y = C(x)
x0, x1 = y

z = D(y)

fig, ax = X.plot(x0)
X.plot(x1, fig=fig, ax=ax)
X.plot(z, fig=fig, ax=ax)
plt.show()
