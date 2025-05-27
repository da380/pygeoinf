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
)


X = Sobolev(0, pi, 0.01, 2, 0.1)

v = X.dirac_representation(pi / 4)

X.plot(v)

plt.show()
