import numpy as np
from scipy.stats import norm

from linear_inference.hilbert_space import LinearForm, LinearOperator, VectorSpace, HilbertSpace

n = 2
space = VectorSpace(n)

mat = norm.rvs(size = (n,n))


X = HilbertSpace(space, space, lambda x : x, lambda x : x)

def mapping(x):
    print("Hi")
    return mat @ x
A = LinearOperator(X, X, mapping)

A.adjoint @ X.random()

#print(A)
#print(A.adjoint)






































