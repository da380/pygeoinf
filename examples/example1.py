import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from linear_inference import Euclidean, GaussianMeasure
from linear_inference.sphere import SphereHS



lmax = 128
s = 2
X = SphereHS(lmax, s, length_scale=0.1)

mu = X.sobolev_gaussian_measure(2, length_scale=0.1)

u = mu.sample()

X.plot(u, show=True, colorbar=True)






















    


























































