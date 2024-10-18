import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pygeoinf.linalg as la
import pygeoinf.sphere as sph

lmax = 64
X = sph.Sobolev(lmax, 2.0, 0.4, radius=10)

mu = X.sobolev_gaussian_measure(3, 0.1, 1)

Y = mu.cameron_martin_space

u = mu.sample()

print(Y.inner_product(u, u))
