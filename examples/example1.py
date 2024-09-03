import numpy as np
import matplotlib.pyplot as plt
import pyshtools as sh
from scipy.stats import norm
from pygeoinf import Euclidean, GaussianMeasure, LinearForm, LinearOperator
from pygeoinf.S2 import Sobolev, Lebesgue

lmax = 256
radius = 2
order = 2
scale = 0.2
X = Sobolev( order, scale, radius=radius, power_of_two=True)
print(X.lmax)

mu = X.heat_kernel_gaussian_measure(0.5, 1)

u = X.dirac(20,20)

print(mu.variance_of_linear_form(u))






























    


























































