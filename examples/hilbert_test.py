import numpy as np

from pygeoinf.hilbert import (
    HilbertSpace,
    EuclideanSpace,
    LinearOperator,
    LinearForm,
    LUSolver,
)

X = EuclideanSpace(4)

a = np.random.randn(X.dim, X.dim)
A = LinearOperator.from_matrix(X, X, a)

print(A)

B = LUSolver()(A)

print(A @ B)
