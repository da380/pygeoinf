import numpy as np

from pygeoinf.hilbert import EuclideanSpace, LinearOperator, CholeskySolver
from pygeoinf.forward_problem import ForwardProblem
from pygeoinf.optimisation import LeastSquaresInversion

dimX = 4
dimY = 2

X = EuclideanSpace(dimX)
Y = EuclideanSpace(dimY)

A = LinearOperator(X, Y, lambda x: x[:dimY])
nu = Y.standard_gaussisan_measure(0.1)


forward_problem = ForwardProblem(A, nu)
x = X.random()
y = forward_problem.data_measure(x).sample()


least_squares_inversion = LeastSquaresInversion.from_forward_problem(forward_problem)

damping = 0.1
least_squares_operator = least_squares_inversion.least_squares_operator(damping)

z = least_squares_operator(y)

print("model in:  ", x)
print("data:      ", y)
print("model out: ", z)

R = least_squares_inversion.resolution_operator(damping)

print(R)
