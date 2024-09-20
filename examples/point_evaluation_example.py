import numpy as np
import pyshtools as sh
import matplotlib.pyplot as plt
import pygeoinf.linalg as la
import pygeoinf.sphere as sphere
from scipy.stats import uniform


lmax = 128

X = sphere.Sobolev(lmax, 2, 0.1)

f = lambda l : (1+0.1*l*(l+1))**(-2)
mu = X.invariant_gaussian_measure(f)
u = mu.sample()

n = 5
lats = uniform(loc = -90, scale = 180).rvs(size=n)
lons = uniform(loc = 0, scale = 360).rvs(size=n)

A = X.point_evaluation_operator(lats, lons)
Y = A.codomain

v = Y.random()

print(Y.inner_product(v, A(u)))
print(X.inner_product(A.adjoint(v), u))



