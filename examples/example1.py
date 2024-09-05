import numpy as np
from scipy.stats import norm

from pygeoinf.testing import VectorSpace, LinearOperator, DualVector, DualVectorSpace, DualOperator, Real


dimX = 5
dimY = 2
X = VectorSpace(dimX, lambda x : x, lambda x : x )
Y = VectorSpace(dimY, lambda x : x, lambda x : x )

A = LinearOperator(X, Y, mapping = lambda x : x[:dimY])
xp = DualVector(X, mapping =  lambda  x : x[0])
yp = DualVector(X, matrix = xp.matrix)

x = X.random()
print(xp)
print(DualOperator(DualOperator(yp)))














































    


























































