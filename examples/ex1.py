import matplotlib.pyplot as plt

import numpy as np
from numpy import pi


import pygeoinf as inf
from pygeoinf.symmetric_space_new.sphere import Lebesgue, Sobolev

import pyshtools as sh


lmax = 32
radius = 10
grid = "DH2"
extend = True


order = 2
scale = 0.1
X = Sobolev(lmax, order, scale * radius, radius=radius)
# X = Lebesgue(lmax, radius=radius)

u = X.project_function(lambda p: 1)

print(X.squared_norm(u))
print(4 * pi * radius * radius)


mu = X.norm_scaled_heat_kernel_gaussian_measure(0.5 * radius)

us = mu.samples(1000)
squared_norms = [X.squared_norm(u) for u in us]

print(np.mean(squared_norms))
plt.hist(squared_norms, bins=100)


u = mu.sample()
X.plot(u)
plt.show()
