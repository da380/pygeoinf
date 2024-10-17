import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pygeoinf.linalg as la
import pygeoinf.sphere as sph

lmax = 32
X = sph.Sobolev(lmax, 2.0, 0.4, radius=10)

mu = X.sobolev_gaussian_measure(3, 0.1, 1)

u = mu.sample()


Y = mu.cameron_martin_space()

print(Y.norm(u))
