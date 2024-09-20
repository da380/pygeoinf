import numpy as np
import pyshtools as sh
import matplotlib.pyplot as plt
import pygeoinf.linalg as la
import pygeoinf.sphere as sphere




lmax = 128

X = sphere.Sobolev(lmax, 2, 0.1)

f = lambda l : (1+0.1*l*(l+1))**(-2)

mu = X.invariant_gaussian_measure(f)

u = mu.sample()

plt.pcolormesh(u.data, shading="gouraud")
plt.colorbar()
plt.show()


