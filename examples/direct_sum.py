import numpy as np
from pygeoinf.hilbert import (
    EuclideanSpace,
    HilbertSpaceDirectSum,
    LinearOperator,
    BlockLinearOperator,
)


X = EuclideanSpace(2)
Y = EuclideanSpace(3)

Z = HilbertSpaceDirectSum([X, Y])


A = X.identity_operator()
B = LinearOperator(Y, X, lambda y: y[: X.dim])


M = BlockLinearOperator([[A, B]])

N = BlockLinearOperator([[A.adjoint], [B.adjoint]])

print(M)

print(M.adjoint)

print(N)
print(N.adjoint)
