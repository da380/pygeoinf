import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pygeoinf.linalg as la
import pygeoinf.sphere as sph

lmax = 16
X = sph.Sobolev(lmax, 2.0, 0.4, radius=10)

m = 10
forms = []
for i in range(m):
    forms.append(X.dual.random())

A = la.LinearOperator.from_linear_forms(forms)

print(A.dual)
