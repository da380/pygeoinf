import numpy as np
from pygeoinf import linalg as la


X = la.EuclideanSpace(2)
Y = la.EuclideanSpace(3, metric_tensor=2*np.identity(10))
