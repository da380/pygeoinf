from linear_inference.euclidean import EuclideanSpace
from linear_inference.gaussian_measure import GaussianMeasure

import numpy as np
from scipy.stats import norm,multivariate_normal

# Set the Hilbert space. 
n = 5
X = EuclideanSpace(n)


a = norm.rvs(size = (n,n))
a = a.T @ a + np.identity(n)

covariance = lambda x : a @ x
sample = lambda n : multivariate_normal(cov = a).rvs(n)

mu = GaussianMeasure(X, covariance, sample = sample)

print(mu.sample(5))