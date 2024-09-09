import pygeoinf.linalg as la
import numpy as np

dimX = 4

X = la.EuclideanSpace(dimX)

dimY = 2
Y = la.EuclideanSpace(dimY)




A = la.LinearOperator(X, Y, mapping=lambda x : x[:dimY])

B = A @ A.adjoint
C = A.adjoint @ A

print(B)
print(C)



  
