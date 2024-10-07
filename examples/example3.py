import numpy as np
from pygeoinf import linalg as la


X = la.EuclideanSpace(2)
Y = la.EuclideanSpace(3, metric_tensor=2*np.identity(10))



Z = la.HilbertSpaceDirectSum([X,Y])


x = X.random()

z = Z.canonical_injection(0,x)

print(x)

print(z)

print(Z.canonical_projection(0,z))





