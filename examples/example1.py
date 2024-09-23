import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pygeoinf.linalg as la
import pygeoinf.sphere as sph

lmax = 64
X =  sph.Sobolev(lmax, 1.5, 0.1, radius=10)

mu = X.sobolev_gaussian_measure(3, 0.1, 1)

u = mu.sample()

v = X.dirac_representation(0, 180)

w = mu.covariance(v)


print(X.inner_product(v,u))

print(X.inner_product(mu.covariance(v),v))


#plt.pcolormesh(v.data)
#plt.colorbar()
#plt.show()