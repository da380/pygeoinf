import numpy as np
import pyshtools as sh
import matplotlib.pyplot as plt
import pygeoinf.linalg as la
import pygeoinf.sphere as sphere
from scipy.stats import uniform


lmax = 2
X = sphere.Sobolev(lmax, 2, 0.1)
mu = X.sobolev_gaussian_measure(2, 0.1, 1)
u = mu.sample()

n = 5
lats = uniform(loc = -90, scale = 180).rvs(size=n)
lons = uniform(loc = 0, scale = 360).rvs(size=n)

A = X.point_evaluation_operator(lats, lons)

=

