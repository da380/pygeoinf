import matplotlib.pyplot as plt

import numpy as np

import pygeoinf as inf
from pygeoinf.symmetric_space_new.circle import Lebesgue, Sobolev

# X = Sobolev(16, 2, 0.1)
X = Lebesgue(4)


mu = X.norm_scaled_invariant_gaussian_measure(lambda k: (1 + k * k) ** 0, 1)


us = mu.samples(1000)
squared_norms = [X.squared_norm(u) for u in us]

mean_squared_norm = np.mean(squared_norms)
print(mean_squared_norm)
