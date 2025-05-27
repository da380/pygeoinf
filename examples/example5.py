import numpy as np
import matplotlib.pyplot as plt
from pygeoinf.geometry.interval import Sobolev
from pygeoinf import LinearForm

X = Sobolev(0, np.pi, 0.01, 2, 0.1)


u = X.project_function(np.sin)
v = X.project_function(lambda x: 1)

up = X.to_dual(u)


print(X.inner_product(u, v))
print(up(v))

w = X.from_dual(up)
print(X.norm(u - w))
