import numpy as np
from scipy.stats import norm

from linear_inference.hilbert_space import LinearForm, LinearOperator, VectorSpace, HilbertSpace

n = 2
space = VectorSpace(n)

M = norm.rvs(size = (n,n))

X = VectorSpace(n)
Y = HilbertSpace(X, X, lambda x : x , lambda x : x)

A = LinearOperator(Y, Y, lambda x : M @ x)#, adjoint_mapping = lambda y : M.T @ y)

print(A)

print(A.adjoint)










































