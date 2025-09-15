import pygeoinf as inf
import numpy as np


A = np.random.randn(10, 10)

A = A @ A.T

diag = A.diagonal()
print(diag)

approx_diags = inf.random_diagonal(A, 5, rtol=0.01)
print(approx_diags)
