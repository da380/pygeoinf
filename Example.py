from LinearInference.Hilbert import Space
from LinearInference.Operator import Linear, SelfDual, SelfAdjoint
from LinearInference.Gaussian import Measure
from LinearInference.Euclidean import EuclideanSpace
from LinearInference.Solver import LU,Cholesky,CG


import numpy as np
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve
from scipy.linalg import norm
from scipy.stats import multivariate_normal
from scipy.sparse import csr_array, triu
from scipy.sparse.linalg import LinearOperator



# Return a random metric tensor. 
def RandomMetricTensor(dim):
    A = np.random.normal(loc = 0, scale= 1, size= (dim, dim))
    return np.matmul(A,A.T) + 0.1*np.identity(dim)


# Gaussian measure on a Euclidean space.
def EuclideanGaussianMeasure(space, C, xbar):
    mean = lambda: xbar
    covariance = lambda x : C @ x
    sample = lambda: multivariate_normal(mean = xbar, cov = C).rvs()
    return Measure(space, mean, covariance, sample)


# Set up the first vector space. 
m = 5
X = EuclideanSpace(m, metric = RandomMetricTensor(m)) 

# Set up the second vector space. 
n = 2
Y = EuclideanSpace(n)

# Define the projection from X onto Y.
P = Linear(domain = X, coDomain = Y, mapping =  lambda x : x[:Y.Dimension],  dualMapping = lambda yp : np.append(yp,np.zeros(X.Dimension-Y.Dimension)))

A = SelfAdjoint(X, lambda x : 2*x)

x = X.Random()
y = A(x)

B = CG(A)


print(B)





















































            

