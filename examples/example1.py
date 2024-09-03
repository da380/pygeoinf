import numpy as np
import matplotlib.pyplot as plt
import pyshtools as sh
from scipy.stats import norm
from linear_inference import Euclidean, GaussianMeasure, LinearForm, LinearOperator
from linear_inference.S2 import Sobolev, Lebesgue

lmax = 256
radius = 2
X = Sobolev(lmax, 2, 0.1, radius=radius)

mu = X.heat_kernel_gaussian_measure(0.5, 1)

u = X.dirac(20,20)

print(mu.variance_of_linear_form(u))
































    


























































