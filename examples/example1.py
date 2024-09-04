import numpy as np
import matplotlib.pyplot as plt
import pyshtools as sh
from scipy.stats import norm

from pygeoinf import VectorSpace,HilbertSpace,LinearOperator,GaussianMeasure,Euclidean



help(GaussianMeasure)




'''
lmax = 128
radius = 2
order = 2
scale = 0.1
X = Sobolev(order, scale, lmax = lmax, radius=radius, power_of_two=True)

mu = X.sobolev_gaussian_measure(2,0.1,1)

A = X.invariant_linear_operator(X, lambda l : l*(l+1))

nu = mu.affine_transformation(operator=A)

X.plot(nu.sample())

'''

































    


























































