import numpy as np
from scipy.stats import norm


from linear_inference.hilbert_space import LinearForm, LinearOperator, VectorSpace, HilbertSpace

n = 3
space = VectorSpace(n)

X = HilbertSpace(space, space, lambda x : x, lambda x : x)

A = LinearOperator(X, X, lambda x : 2 * x)


As = A.adjoint

x1 = X.random()
x2 = X.random()

























