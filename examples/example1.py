import numpy as np
from scipy.stats import norm

from pygeoinf.testing import (VectorSpace, LinearOperator, DualVector, DualVectorSpace, 
                             DualOperator, HilbertSpace, DualHilbertSpace, RealN,      
                             AdjointOperator, Euclidean)





X = RealN(5)

Xp = DualVectorSpace(X)

A = LinearOperator(X, X, mapping = lambda x : 2 * x)
Ap = DualOperator(A)


x = X.random()

print(X.to_components(x))

xp = Xp.random()

print(Xp.to_components(xp))















































    


























































