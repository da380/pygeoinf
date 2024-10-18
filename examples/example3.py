import numpy as np
from scipy.stats import norm
from pygeoinf import linalg as la

dim = 10
X = la.EuclideanSpace(dim)

a = norm().rvs(size=(dim, dim))
b = a @ a.T + 0.00 * np.identity(dim)
A = la.LinearOperator(X, X, lambda x: b @ x)


solver = la.CGSolver()

B = solver(A)

x = X.random()

B(x)
