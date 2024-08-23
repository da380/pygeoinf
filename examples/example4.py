from linear_inference.euclidean import EuclideanSpace
from linear_inference.euclidean import EuclideanGaussianMeasure
from linear_inference.vector_space import LinearOperator


import numpy as np
from scipy.stats import norm,multivariate_normal


# Set the domain. 
m = 3
X = EuclideanSpace(m)

# Set up the measure. 
a = norm.rvs(size = (m,m))
cov = a.T @ a + np.identity(m)
mu = EuclideanGaussianMeasure(cov)


# Set a second space and linear operator and vector. 
n = 2
Y = EuclideanSpace(n)
A = LinearOperator(X, Y, lambda x : x[:n])
a = Y.random()

nu = 2*mu - mu



print(nu.covariance - mu.covariance.adjoint)










