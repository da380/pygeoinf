import pygeoinf.linalg as la
from pygeoinf.least_squares import LeastSquaresProblem
from pygeoinf.sphere import Sobolev
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt


# Set the model space.
X = Sobolev(64, 2, 0.3)

# Generate a random model
mu = X.sobolev_gaussian_measure(2, 0.3, 1)
u = mu.sample()


# Set up the forward operator.
n = 1000
lats = uniform(loc=-90, scale=180).rvs(size=n)
lons = uniform(loc=0, scale=360).rvs(size=n)
A = X.point_evaluation_operator(lats, lons)
Y = A.codomain

# Generate synthetic data.
v = A(u)


problem = LeastSquaresProblem(A)

damping = 2
solver = la.CGSolver()
B = problem.normal_solver(damping, solver)

u2 = B(v)

v2 = A(u2)

print(Y.norm(v2-v) / Y.norm(v))

plt.figure()
plt.pcolormesh(u.data)

plt.figure()
plt.pcolormesh(u2.data)
plt.show()
