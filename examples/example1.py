import pygeoinf.linalg as la
import numpy as np
from scipy.stats import norm
from abc import ABC, abstractmethod, abstractproperty

dimX = 20

X = la.EuclideanSpace(dimX)

dimY = 2
Y = la.EuclideanSpace(dimY)


a = norm().rvs(size=(dimX, dimX))
A = la.LinearOperator(X, X, matrix=a)
print(A)

solver = la.DirectLUSolver()

solver.set_operator(A)

print(solver @ A)






  
