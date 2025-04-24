import numpy as np

from pygeoinf.hilbert import (
    HilbertSpace,
    EuclideanSpace,
    LinearOperator,
    LinearForm,
    LUSolver,
    CholeskySolver,
    CGMatrixSolver,
)

X = EuclideanSpace(4)

a = np.random.randn(X.dim, X.dim) + np.identity(X.dim)
A = LinearOperator.from_matrix(X, X, a @ a.T)


x0 = X.random()
B = CGMatrixSolver(galerkin=True)(A, x0=x0)


x = X.random()

y = A(x)


z = B(y)

print(x)
print(z)
