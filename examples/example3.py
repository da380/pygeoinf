import numpy as np
from pygeoinf.hilbert import EuclideanSpace, LinearOperator
from pygeoinf.direct_sum import (
    HilbertSpaceDirectSum,
    BlockLinearOperator,
    BlockDiagonalLinearOperator,
)
from pygeoinf.gaussian_measure import GaussianMeasure

X = EuclideanSpace(2)
Y = EuclideanSpace(3)

mu = GaussianMeasure.from_standard_deviation(X, 1)
nu = GaussianMeasure.from_standard_deviation(Y, 2)

pi = GaussianMeasure.from_direct_sum([mu, nu])


print(pi.samples(2))
