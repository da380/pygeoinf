import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf
from pygeoinf.symmetric_space.sphere import Sobolev, Lebesgue


model_space = Sobolev(64, 2, 0.1)


points1 = model_space.random_points(10)
points2 = model_space.random_points(5)

A1 = model_space.point_evaluation_operator(points1)
A2 = model_space.point_evaluation_operator(points2)

data_space1 = A1.codomain
data_space2 = A2.codomain

data_error_measure_1 = inf.GaussianMeasure.from_standard_deviation(data_space1, 0.1)
data_error_measure_2 = inf.GaussianMeasure.from_standard_deviation(data_space2, 0.1)

forward_problem1 = inf.LinearForwardProblem(A1, data_error_measure=data_error_measure_1)
forward_problem2 = inf.LinearForwardProblem(A2, data_error_measure=data_error_measure_2)

forward_problem = inf.LinearForwardProblem.from_direct_sum(
    [forward_problem1, forward_problem2]
)

mu = model_space.heat_kernel_gaussian_measure(0.4)


data = forward_problem.data_space.random()


model = forward_problem.forward_operator.adjoint(data)

model_space.plot(model)

plt.show()
