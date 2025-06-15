import numpy as np
from numpy import pi, sqrt, sin, cos
import matplotlib.pyplot as plt

from pygeoinf.homogeneous_space.line import Sobolev

X = Sobolev(0, 10, 256, 2, 0.1)


u = X.project_function(lambda x: x)

x = 2
vp = X.dirac(x)
v = X.dirac_representation(x)
print(vp(u))
print(X.inner_product(v, u))
