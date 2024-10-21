import numpy as np
import pygeoinf.linalg as la
from scipy.stats import chi2
from pygeoinf.forward_problem import ForwardProblem
from pygeoinf.occam import OccamInversion
from pygeoinf.sphere import Sobolev
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt


# Set the model space.
X = Sobolev(64, 2, 0.3)

mu = X.sobolev_gaussian_measure(2, 0.3, 1)
u = mu.sample()

# Set up the forward operator.
n = 100
lats = uniform(loc=-90, scale=180).rvs(size=n)
lons = uniform(loc=0, scale=360).rvs(size=n)
A = X.point_evaluation_operator(lats, lons)


# Set up the error distribution.
Y = A.codomain
sigma = 0.1
nu = Y.standard_gaussisan_measure(sigma)

problem = OccamInversion(A, nu, 0.95)
