import pygeoinf.linalg as la
import numpy as np
from scipy.stats import norm
from abc import ABC, abstractmethod, abstractproperty

dimX = 4

X = la.EuclideanSpace(dimX)

dimY = 2
Y = la.EuclideanSpace(dimY)


A = la.LinearOperator(X, Y, mapping = lambda x : 2 * x[:dimY])

print(A)

print(A.galerkin_matrix)




  
