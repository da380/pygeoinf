import pygeoinf.linalg as la
from pygeoinf.sphere import Sobolev
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt


X = Sobolev(128, 2, 0.1)

kappa = X.sobolev_gaussian_measure(4, 0.1, 1)
Q = kappa.covariance

n = 10
lats = uniform(loc=-90, scale=180).rvs(size=n)
lons = uniform(loc=0, scale=360).rvs(size=n)
A = X.point_evaluation_operator(lats, lons)

Y = A.codomain

C = A @ Q @ A.adjoint

v = Y.random()

w = C(v)

solver = la.CGSolver()

D = solver(C)

x = D(w)

print(v)
print(x)
