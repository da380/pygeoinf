import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf
from pygeoinf.symmetric_space.line import Sobolev

X = Sobolev.from_sobolev_parameters(2, 0.1)
A = inf.LinearOperator(X, X, lambda x: x)
mu = X.heat_gaussian_measure(0.1, 1)

fp = inf.LinearForwardProblem(A, data_error_measure=mu)

cfp = inf.LinearForwardProblem.from_direct_sum([fp, fp])

print(cfp.model_space == fp.model_space)
