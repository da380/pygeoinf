import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from linear_inference import Euclidean, GaussianMeasure, LinearForm
from linear_inference.S2 import Sobolev, Lebesgue



lmax = 1
radius = 1
X = Sobolev(lmax, 2, 0.1, radius=radius)

mu = X.sobolev_gaussian_measure(2,0.1,1)
C = mu.covariance

u = X.dirac_kernel(10,20)

print(X.inner_product(C(u),u))



































    


























































