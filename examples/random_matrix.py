import numpy as np
from pygeoinf.linalg import EuclideanSpace, LinearOperator
from pygeoinf.random_matrix import (
    RandomSVDApproximation,
    fixed_rank_basis,
    low_rank_svd,
)

from pygeoinf.sphere import Lebesgue
from scipy.stats import norm, uniform


"""
X = EuclideanSpace(4)


def mapping(x):
    y = x.copy()
    for i, val in enumerate(y):
        y[i] /= (i + 1) ** 4
    return y

a = LinearOperator.self_adjoint(X, mapping)
A = a.matrix()
"""

X = Lebesgue(10)

m = 4
lats = uniform(loc=-90, scale=180).rvs(size=m)
lons = uniform(loc=0, scale=360).rvs(size=m)
a = X.point_evaluation_operator(lats, lons)
A = a.matrix(galerkin=True)


Q = fixed_rank_basis(A, 3, 0)

U, S, Vh = low_rank_svd(A, Q)


print(np.max(a.matrix(dense=True) - U @ S @ Vh))
