import numpy as np
from scipy.stats import norm

from pygeoinf import (standard_vector_space, LinearOperator, 
                      LinearForm, DualVectorSpace, DualOperator,
                      standard_euclidean_space, AdjointOperator)

dimX = 5
X = standard_euclidean_space(dimX)

dimY = 2
Y = standard_euclidean_space(dimY)

A = LinearOperator(X, Y, mapping = lambda x : 2 * x[:dimY])



As = AdjointOperator(A)

y = Y.random()

print(As)






















































    


























































