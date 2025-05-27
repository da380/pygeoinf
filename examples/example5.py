import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from pygeoinf.geometry.interval import Sobolev
from pygeoinf import (
    LinearOperator,
    LinearForm,
    LinearForwardProblem,
    LinearLeastSquaresInversion,
    GaussianMeasure,
    CholeskySolver,
    GMRESMatrixSolver,
    CGSolver,
    CGMatrixSolver,
    LUSolver,
)


X = Sobolev(0, pi, 0.001, 2, 0.05)
u = X.project_function(lambda x: (x - pi / 2) * np.exp(-5 * (x - pi / 2) ** 2))

x = X.random_points(20)
A = X.point_evaluation_operator(x)
Y = A.codomain

nu = GaussianMeasure.from_standard_deviation(Y, 0.001)

forward_problem = LinearForwardProblem(A, nu)

v = forward_problem.data_measure(u).sample()

inversion = LinearLeastSquaresInversion(forward_problem)

w = inversion.least_squares_operator(0.1, CGSolver(rtol=1.0e-12))(v)

plt.plot(x, v, ".")
X.plot(u)
X.plot(w)
plt.show()
