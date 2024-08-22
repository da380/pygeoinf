from linear_inference.euclidean import EuclideanSpace
from linear_inference.euclidean import EuclideanGaussianMeasure
from linear_inference.vector_space import LinearOperator


import numpy as np
from scipy.stats import norm,multivariate_normal

# Set the space. 
m = 3
X = EuclideanSpace(m)

# Define a measure. 
a = norm.rvs(size = (m,m))
cov = a.T @ a + np.identity(m)
mu = EuclideanGaussianMeasure(X, cov)

# Set a second space and linear operator and vector. 
n = 2
Y = EuclideanSpace(n)
A = LinearOperator(X, Y, lambda x : x[:n])
a = Y.random()

# Form the push-forward of the measure under an affine mapping. 
nu = mu.affine_transformation(operator = A, translation = a)

print(nu.sample())









