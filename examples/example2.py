import pygeoinf.linalg as la
from pygeoinf.sphere import Sobolev
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt


dimX = 5
X = la.EuclideanSpace(dimX)

dimY = 3
Y = la.EuclideanSpace(dimY)

A = la.LinearOperator(X, Y, lambda x: x[:dimY])

x = X.random()
y = Y.random()

print(Y.inner_product(y, A(x)))
print(X.inner_product(A.adjoint(y), x))


mu = la.GaussianMeasure(X, lambda x: x, sample_using_matrix=True)

nu = mu.affine_mapping(operator=A, translation=y)

print(nu.covariance)


Z = Sobolev(128, 2, 0.1)

kappa = Z.sobolev_gaussian_measure(4, 0.1, 1)


n = 5
lats = uniform(loc=-90, scale=180).rvs(size=n)
lons = uniform(loc=0, scale=360).rvs(size=n)
B = Z.point_evaluation_operator(lats, lons)

Q = kappa.covariance

C = B @ Q @ B.adjoint

solver = la.MatrixSolverCG(rtol=1e-12)

solver.operator = C

D = solver.inverse_operator

print(D @ C)
