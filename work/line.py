import numpy as np
from pygeoinf.symmetric_space import line

# from pygeoinf.symmetric_space.line import Sobolev, plot

scale = 0.1

X = line.Sobolev(256, 2, scale, a=0, b=2 * np.pi)

mu = X.heat_kernel_gaussian_measure(0.1)

X.check(measure=mu)

"""
u = X.dirac_representation(0)

line.plot(X, u, full=False)

Y = circle.Sobolev(256, 2, scale)

v = Y.dirac_representation(0)

circle.plot(Y, v)

plt.show()
"""
