import numpy as np

from pygeoinf.hilbert import (
    HilbertSpace,
    EuclideanSpace,
    LinearOperator,
    LinearForm,
    LUSolver,
    CholeskySolver,
    CGMatrixSolver,
    BICGMatrixSolver,
    GMRESMatrixSolver,
    CGSolver,
)

X = EuclideanSpace(4)

a = np.random.randn(X.dim, X.dim) + np.identity(X.dim)
A = LinearOperator.from_matrix(X, X, a @ a.T)


B = CGSolver(rtol=1.0e-10)(A)


x = X.random()

y = A(x)

print(A.dual @ B.dual)
