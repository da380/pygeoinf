import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf
from pygeoinf.homogeneous_space.line import Sobolev

X = Sobolev.from_sobolev_parameters(3, 0.1, x1=10)
print(X.dim)


x = 0.25


def f(x):
    return x * x


v = X.dirac_representation(x)
u = X.project_function(f)

print(X.inner_product(v, u), f(x))

mu = X.heat_gaussian_measure(0.1, 1)

A = X.invariant_automorphism(lambda k: k)

u = X.project_function(lambda x: np.exp(-10 * (x - 5) ** 2))
u = A(u)

fig, ax = X.plot(u)
plt.show()
