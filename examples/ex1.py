import numpy as np
import pygeoinf as inf


X = inf.EuclideanSpace(3)

mat = np.random.randn(X.dim, X.dim)

A = inf.MatrixLinearOperator(X, X, mat, galerkin=False)

print(mat)
print(A.extract_diagonals([0, 1, -1]))
