import matplotlib.pyplot as plt

import numpy as np

import pygeoinf as inf
from pygeoinf.symmetric_space_new.circle import Lebesgue, Sobolev

X = Sobolev(32, 2, 0.1)
# X = Lebesgue(16)


mu = X.norm_scaled_invariant_gaussian_measure(lambda k: (1 + (0.1 * k) ** 2) ** -2, 1)


us = mu.samples(10000)
ns = [X.squared_norm(u) for u in us]

n = np.mean(ns)
print(n)


plt.hist(ns, bins=50)
plt.show()
