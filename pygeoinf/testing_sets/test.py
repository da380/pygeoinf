# ...existing code...
from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.subsets import Ball
import numpy as np


M = EuclideanSpace(3)
center = np.array([0, 0, 0])
A = Ball(M, center, 1.0)

print(A.is_element(M.random()))
