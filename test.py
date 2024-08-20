import numpy as np
from scipy.stats import norm

from linear_inference.vector_space import VectorSpace, HilbertSpace
from linear_inference.linear_operator import LinearOperator

m = 3
n = 2
X = HilbertSpace(m)
Y = HilbertSpace(n)





A = LinearOperator(X, Y, lambda x : x[:n])

print(A)

print(A.dual)











    


























































