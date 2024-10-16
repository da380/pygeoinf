import pygeoinf.linalg as la
from pygeoinf.least_squares import LeastSquaresProblem
from pygeoinf.sphere import Sobolev
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt


# Set the model space.
X = Sobolev(16, 2, 0.2)

# Generate a random model
mu = X.sobolev_gaussian_measure(2, 0.2, 1)
u = mu.sample()


# Set up the forward operator.
n = 10
lats = uniform(loc=-90, scale=180).rvs(size=n)
lons = uniform(loc=0, scale=360).rvs(size=n)
A = X.point_evaluation_operator(lats, lons)
Y = A.codomain

# Generate synthetic data.
v = A(u)


problem = LeastSquaresProblem(A)

damping = 1
solver = la.MatrixSolverCholesky(galerkin=True)
B = problem.normal_solver(damping, solver)

w = B(v)
