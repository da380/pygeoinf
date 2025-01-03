import numpy as np
import pygeoinf.linalg as la
from scipy.stats import chi2
from pygeoinf.forward_problem import ForwardProblem
from pygeoinf.bayesian import BayesianInversion, BayesianInference
from pygeoinf.sphere import Sobolev
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt


# Set the model space.
X = Sobolev(128, 0, 0.3)

# Set up the prior distribution.
mu = X.sobolev_gaussian_measure(2, 0.3, 1)

# Set up the forward operator.
n = 100
lats = uniform(loc=-90, scale=180).rvs(size=n)
lons = uniform(loc=0, scale=360).rvs(size=n)
A = X.point_evaluation_operator(lats, lons)

# Set up the property operator.
m = 4
lats = uniform(loc=-90, scale=180).rvs(size=m)
lons = uniform(loc=0, scale=360).rvs(size=m)
B = X.point_evaluation_operator(lats, lons)

# Set up the error distribution.
Y = A.codomain
sigma = 0.1
nu = Y.standard_gaussisan_measure(sigma)

# Set up the inference problem
problem = BayesianInference(A, B, mu, nu)

# Generate synthetic data.
u = mu.sample()
v = problem.data_measure(u).sample()


# Form the posterior distribution
pi = problem.property_posterior_measure(v)

print(mu.affine_mapping(operator=B).covariance)
print(pi.covariance)
